/**
 * env.ts — the internal codegen environment shared by expressions.ts and the
 * follow-up statement/function/dual-mode emitters (C3/C4/C5).
 *
 * Responsibilities:
 * - Local variable table: GLSL name → flattened JS representation.
 *   FLAT locals become one JS var per scalar component, named
 *   `<glslname>__<i>` (i = flat component index: vecN → 0..N-1; matCxR →
 *   column-major flat index col*rows+row; structs recurse as
 *   `<name>__<member>` with vector/matrix leaves appending `__<i>`).
 *   ARRAY locals (non-const) become SCRATCH-backed storage
 *   (ctx.scratch / ctx.intScratch) — one contiguous block per variable.
 * - Scratch allocation: monotonic per-stage offsets with max-size tracking
 *   (StageCodegenResult reports scratchSize/intScratchSize; gl/ allocates
 *   the buffers per draw).
 * - Temp vars: `t0, t1, ...` for materializing dynamic indices and
 *   multi-use values. The statement/function emitter (C3/C4) MUST hoist
 *   `var t0, t1, ...` (env.temps) at the top of the generated body — the
 *   expression strings reference them bare. C2's selftest declares them
 *   manually where it evals emitted code.
 * - Storage access: the PathRef model + the read/write formula builders.
 *   The expressions walker (refOf / resolveStoragePath in expressions.ts)
 *   resolves identifier + member/index chains into PathRefs; the env
 *   provides the path-keyed read builders (uniformPathRead, blockPathRead,
 *   varyingPathRead) plus env.scalarRead/scalarWrite for simple identifiers.
 *
 * UNIFORM STORE INDEXING (CRITICAL — the linker must match):
 *   UniformSlot.slot is a FLOAT index into the store (NOT a vec4-slot × 4).
 *   A vec4 uniform at location L occupies floats [L .. L+3]; a matC
 *   occupies C*4 consecutive floats with column `col` at L + col*4 + row
 *   (column stride 4 — GLSL memory order). Int/uint/bool/sampler uniforms
 *   live in ctx.intUniforms with the same rule (1 int per component).
 *
 * UINT INVARIANT: every uint-typed Value is a JS number in [0, 2^32).
 * Codegen wraps every uint arithmetic result with `>>> 0` (and uses
 * Math.imul for uint multiply); uint VALUES read from Int32Array storage
 * (intUniforms / intScratch / blockIntStores) are wrapped at the read site.
 * INT INVARIANT: every int-typed Value is a JS number in int32 range
 * (results of int arithmetic are `| 0`-wrapped).
 *
 * BLOCK LAYOUT REQUIREMENT (linker contract): `blocks.get(bi)` must contain
 * an entry for every path PREFIX reachable from codegen (bare members 'm',
 * struct members 'm.n', array prefixes 'm[0]' with arrayStride, ARRAYED
 * block instance prefixes 'b[0]' with offset 0 + blockStride, and every
 * const-indexed element 'm[2]'). Missing entries throw — the linker must
 * emit them. Default-block uniformSlots likewise: every reachable prefix
 * ('u', 'u.m', 'u[0]', 'u[0].m', 'u[2].m', ...).
 */
import type { ShaderStage } from '../compiler.js';
import type { GLSLType } from '../types.js';
import type { Expr } from '../ast.js';
import {
  builtinConstants,
  builtinVariables,
  extensionConstants,
  extensionVariables,
} from '../builtins/index.js';
import type { BuiltinVariable } from '../builtins/index.js';
import type {
  BlockMemberLayout,
  CodegenLayout,
  UniformSlot,
  Value,
  VaryingLayout,
} from './index.js';

/* ------------------------------------------------------------------ */
/* Pure type helpers (flat component model)                            */
/* ------------------------------------------------------------------ */

/** Total flat scalar component count: scalar/sampler → 1, vector → size,
 *  matrix → cols*rows, struct → sum of members (recursive), array → size × element. */
export function flatComponents(type: GLSLType): number {
  switch (type.kind) {
    case 'void':
      return 0;
    case 'scalar':
    case 'sampler':
      return 1;
    case 'vector':
      return type.size;
    case 'matrix':
      return type.cols * type.rows;
    case 'struct':
      return type.members.reduce((n, m) => n + flatComponents(m.type), 0);
    case 'array':
      return (type.size ?? 0) * flatComponents(type.element);
  }
}

/** Base scalar of a scalar/vector type; matrix → 'float'; else null. */
export function scalarBaseOf(type: GLSLType): 'float' | 'int' | 'uint' | 'bool' | null {
  switch (type.kind) {
    case 'scalar':
    case 'vector':
      return type.base;
    case 'matrix':
      return 'float';
    default:
      return null;
  }
}

export function isUintType(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && t.base === 'uint';
}
export function isIntType(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && t.base === 'int';
}
export function isBoolType(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && t.base === 'bool';
}
export function isFloatType(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && t.base === 'float';
}

/** True when the type is int/uint/bool (integral family — int store / intScratch). */
export function isIntegralFamily(t: GLSLType): boolean {
  switch (t.kind) {
    case 'scalar':
    case 'vector':
      return t.base !== 'float';
    case 'array':
      return isIntegralFamily(t.element);
    case 'struct':
      return t.members.some((m) => isIntegralFamily(m.type));
    default:
      return false;
  }
}

/** True for a float LEAF shape (float scalar/vector or matrix) — the only
 *  shapes that carry duals at the leaf level (structs/arrays recurse before
 *  the leaf in reads()/writes()). */
export function isFloatLeaf(t: GLSLType): boolean {
  return (
    t.kind === 'matrix' ||
    ((t.kind === 'scalar' || t.kind === 'vector') && t.base === 'float')
  );
}

/** Per-flat-component float-ness of a type (column-major for matrices,
 *  structs/arrays recurse in flat order). Dual mode allocates the dx/dy
 *  names/planes exactly for the TRUE entries. */
export function flatFloatness(type: GLSLType): boolean[] {
  switch (type.kind) {
    case 'scalar':
      return [type.base === 'float'];
    case 'sampler':
      return [false];
    case 'vector':
      return Array.from({ length: type.size }, () => type.base === 'float');
    case 'matrix':
      return Array.from({ length: type.cols * type.rows }, () => true);
    case 'struct':
      return type.members.flatMap((m) => flatFloatness(m.type));
    case 'array':
      return Array.from({ length: type.size ?? 0 }, () => flatFloatness(type.element)).flat();
    default:
      return [];
  }
}

/** True when the type contains any float leaf — the C5a2 guard predicate
 *  (dual-mode constructs whose result carries duals but whose lowering is not
 *  yet implemented must throw rather than silently drop the duals). */
export function hasFloatLeaves(type: GLSLType): boolean {
  return flatFloatness(type).some(Boolean);
}

/** Wrap an expression string to int32 range (int-typed results). The result
 *  is SELF-PARENTHESIZED — wrapped values are embedded in larger expressions
 *  (`(a + (x) | 0)` would parse as `(a + x) | 0`; `|` binds looser than `+`). */
export function wrapInt(s: string): string {
  return `((${s}) | 0)`;
}

/** Wrap an expression string to uint32 range (uint-typed results). Same
 *  self-parenthesizing contract as wrapInt (`>>> 0` binds looser than `+`). */
export function wrapUint(s: string): string {
  return `((${s}) >>> 0)`;
}

/** Convert ONE value to another scalar base (per-component; vectors zip outside).
 *  int→float / uint→float are no-ops (JS numbers). int→uint wraps; uint→int
 *  wraps. bool conversions are never implicit (semantics rejects). */
export function convertScalar(v: string, from: string, to: string): string {
  if (from === to) return v;
  if (to === 'float') return v; // int/uint → float: JS numbers are already floats
  if (from === 'int' && to === 'uint') return wrapUint(v);
  if (from === 'uint' && to === 'int') return wrapInt(v);
  throw new Error(`codegen: no implicit conversion ${from} → ${to}`);
}

/**
 * Convert a Value[] (flat components of `from`) to `to`'s scalar base,
 * producing EXACTLY `flatComponents(to)` values (the caller's lvalue /
 * initializer width is authoritative):
 * - width match: per-component base conversion (the historical behavior);
 * - SCALAR source (`vals.length === 1`) with a wider target: BROADCAST — a
 *   scalar RHS applies component-wise to a vector/matrix lvalue (GLSL ES
 *   1.00 §5.6: `v /= 2.0`, `m *= 2.0` lower to per-component ops), so every
 *   target slot receives the converted scalar;
 * - any other width mismatch (defensive — semantics rejects e.g. vec→mat):
 *   cycle/truncate so the payload is exactly `n` entries — downstream code
 *   reads `conv[c].v` for c < n and would crash on an undefined entry.
 * Broadcast copies SHARE the source Value (its `pre` array and duals ride
 * along): multi-component results always share one pre array (emitPres
 * dedupes by identity, so the materialization runs once), and every
 * component equals the source, so the derivatives are the source's.
 * Same-base conversions (float→float — incl. shape-changing broadcasts/
 * truncations at the caller) return `vals` unchanged, duals and all.
 * DUAL MODE (detected by the values carrying dx — only dual mode sets it):
 * int/uint→float attaches the constant duals (dx=dy='0' — the source has no
 * derivative planes) and preserves `pre` (the dx/dy strings may reference
 * temps its statements set); float→int/uint/bool drops the duals (integral
 * results carry none). Non-dual mode keeps the historical behavior
 * byte-identical for width matches (conversion drops pre — callers re-attach
 * it when needed); shape-changing broadcasts keep pre (the caller re-attach
 * loops only walk the SOURCE length, so the extra copies would lose it).
 */
export function convertValue(vals: Value[], from: GLSLType, to: GLSLType): Value[] {
  const fb = scalarBaseOf(from);
  const tb = scalarBaseOf(to);
  if (fb === null || tb === null) return vals;
  const n = flatComponents(to);
  if (vals.length !== n) {
    if (vals.length === 0) return vals; // empty payload — deeper invariant break
    if (fb === tb) {
      // Same base, different width: share the source Values (pre/duals ride
      // along; cycling fills every slot, truncation keeps the first n).
      const out: Value[] = [];
      for (let c = 0; c < n; c++) out.push(vals[c % vals.length]);
      return out;
    }
    const dual = vals[0].dx !== undefined;
    const out: Value[] = [];
    for (let c = 0; c < n; c++) {
      const src = vals[c % vals.length];
      if (!dual) {
        out.push({ v: convertScalar(src.v, fb, tb), pre: src.pre });
      } else {
        const conv: Value = { v: convertScalar(src.v, fb, tb) };
        if (tb === 'float') {
          // int/uint → float: the source carries no duals — constant duals.
          conv.dx = '0';
          conv.dy = '0';
        }
        // float → int/uint/bool: integral result — no duals (left absent).
        if (src.pre && src.pre.length > 0) conv.pre = src.pre;
        out.push(conv);
      }
    }
    return out;
  }
  if (fb === tb) return vals;
  const dual = vals.length > 0 && vals[0].dx !== undefined;
  if (!dual) return vals.map((v) => ({ v: convertScalar(v.v, fb, tb) }));
  return vals.map((v) => {
    const out: Value = { v: convertScalar(v.v, fb, tb) };
    if (tb === 'float') {
      // int/uint → float: the source carries no duals — constant duals.
      out.dx = '0';
      out.dy = '0';
    }
    // float → int/uint/bool: integral result — no duals (left absent).
    if (v.pre && v.pre.length > 0) out.pre = v.pre;
    return out;
  });
}

/* ------------------------------------------------------------------ */
/* Local variables                                                     */
/* ------------------------------------------------------------------ */

/**
 * One declared local. FLAT kind: `compNames` holds the JS var name of every
 * flat component (scalar → ['x']; vec3 → ['x__0','x__1','x__2']; mat2 → 4
 * names in column-major flat order; structs flattened recursively). SCRATCH
 * kind: array variables live in ctx.scratch / ctx.intScratch at
 * `scratchBase`; `elemSlots` = flat components per array element; struct
 * elements carry member → relative flat offset in `members` (recursive via
 * the member's own type). `int` = integral family → ctx.intScratch.
 *
 * DUAL MODE (env.dual — fragment stages with layout.uses.derivatives):
 * every FLOAT component carries a (v, dx, dy) triple.
 * - FLAT locals: the v JS name stays `compNames[c]`; the derivative names are
 *   `compNames[c] + '_dx'` / `compNames[c] + '_dy'` (registered in
 *   usedJsNames_ at declaration). `dxNames[c]` = the dx name (null for
 *   non-float components; dy is always derivable as `compNames[c] + '_dy'`).
 * - SCRATCH locals (float blocks): the block occupies THREE planes of
 *   `blockSize` elements each — v at `scratchBase`, dx at
 *   `scratchBase + blockSize`, dy at `scratchBase + 2*blockSize`
 *   (`allocScratch` charges 3× in dual mode, so scratchSize reflects the
 *   planes). Int/uint/bool locals never carry duals.
 */
export interface LocalVar {
  name: string;
  type: GLSLType;
  kind: 'flat' | 'scratch';
  scratchBase?: number;
  elemSlots?: number;
  int?: boolean;
  compNames?: string[];
  members?: Map<string, number>;
  /** Dual-mode dx JS name per flat component (null = non-float); absent for scratch. */
  dxNames?: (string | null)[];
  /**
   * Dual-mode dy JS name per flat component (null = non-float). Present only
   * when the dy names deviate from the `compNames[c] + '_dy'` convention —
   * synthesized member/index-of-call-result paths (BUG 1) whose compNames are
   * allocTemp names (`t0_dy` is never declared). leafDual/leafDualWrite fall
   * back to the derived name for ordinary locals.
   */
  dyNames?: (string | null)[];
  /** Synthetic flat local backing a member/index of a call result (BUG 1):
   *  compNames are temps, never registered in locals_/paramFrames_ (so
   *  ensureDynScratch must be bypassed); dynamic indexing spills it to a
   *  scratch block (walk in expressions.ts). */
  synth?: boolean;
  /** Scratch block size in elements (v-plane length; dual planes sit after). */
  blockSize?: number;
  /**
   * Dynamic-index spill backing for FLAT vector/matrix locals (set on first
   * dynamic index; see ensureDynScratch). The flat vars stay the source of
   * truth: every dynamic access copies in before and copies out after, so
   * static accesses continue to read/write the flat vars directly.
   */
  dynScratch?: { base: number; int: boolean; copyIn: string[]; copyOut: string[] };
}

/** One inlined call's scope (C3b): the call's param LocalVars (per-call-site
 *  unique JS names) + the function's local names (pre-scanned from its body,
 *  so an inner local shadows an outer same-named param in resolveLocal) +
 *  the function's OWN locals, materialized per call site with unique JS
 *  names (frameLocal — a callee local can never alias a caller's same-named
 *  local, and nested same-named locals never share scratch). */
export interface ParamFrame {
  params: Map<string, LocalVar>;
  localNames: Set<string>;
  /** Per-call-site LocalVars for the function's own body locals, keyed by
   *  GLSL name. Created lazily by frameLocal at each declaration statement
   *  (decl-before-use ⇒ the entry always exists before any read); sibling
   *  re-declarations reuse the cached entry. */
  locals: Map<string, LocalVar>;
  /** The call site's JS-name suffix generator (the inliner's ctx.suffix).
   *  null for frames created without one (no such callers today). */
  suffix: (() => string) | null;
}

/** Struct member → relative flat offset (within the struct). */
export function structMemberOffsets(type: GLSLType): Map<string, number> {
  if (type.kind !== 'struct') throw new Error(`codegen: structMemberOffsets on a '${type.kind}'`);
  const m = new Map<string, number>();
  let off = 0;
  for (const mem of type.members) {
    m.set(mem.name, off);
    off += flatComponents(mem.type);
  }
  return m;
}

/** Flat offset of a (possibly nested) struct member path within `type`. */
export function structMemberOffset(type: GLSLType, path: string[]): number {
  let t = type;
  let off = 0;
  for (const name of path) {
    if (t.kind !== 'struct') throw new Error(`codegen: '${name}' is not a struct member path`);
    const m = structMemberOffsets(t);
    const o = m.get(name);
    if (o === undefined) throw new Error(`codegen: struct '${t.name}' has no member '${name}'`);
    off += o;
    t = t.members.find((x) => x.name === name)!.type;
  }
  return off;
}

/** Flat JS var names for a whole (non-array) type: 'v' / 'v__0..' / struct recursion. */
function flatNames(name: string, type: GLSLType): string[] {
  switch (type.kind) {
    case 'scalar':
    case 'sampler':
      return [name];
    case 'vector':
      return Array.from({ length: type.size }, (_, i) => `${name}__${i}`);
    case 'matrix':
      return Array.from({ length: type.cols * type.rows }, (_, i) => `${name}__${i}`);
    case 'struct': {
      const out: string[] = [];
      const recType = (base: string, t: GLSLType, o: string[]): void => {
        switch (t.kind) {
          case 'scalar':
          case 'sampler':
            o.push(base);
            break;
          case 'vector':
            for (let i = 0; i < t.size; i++) o.push(`${base}__${i}`);
            break;
          case 'matrix':
            for (let i = 0; i < t.cols * t.rows; i++) o.push(`${base}__${i}`);
            break;
          case 'struct':
            for (const mem of t.members) recType(`${base}__${mem.name}`, mem.type, o);
            break;
          default:
            throw new Error(`codegen: array member '${base}' inside a flat struct is unsupported`);
        }
      };
      recType(name, type, out);
      return out;
    }
    default:
      throw new Error(`codegen: cannot flatten a '${type.kind}' local`);
  }
}

/* ------------------------------------------------------------------ */
/* Storage-backed globals                                              */
/* ------------------------------------------------------------------ */

/** Resolved info for one storage-backed global identifier (no chains).
 *  The identifier's resolvedType lives in the AST — the walker supplies it.
 *  `key` is the path key prefix for chains: the identifier name itself for
 *  uniforms/varyings (chains append '.m' / '[k]'), the bare MEMBER name for
 *  instance-less blocks, or the INSTANCE name for arrayed blocks. */
export type GlobalInfo =
  | { kind: 'uniform'; key: string }
  | {
      kind: 'block';
      blockIndex: number;
      /** Path-key prefix: '' (instance-less — the identifier IS the member
       *  name) or the instance name (arrayed block — members keyed 'b[0].m'). */
      baseKey: string;
      /** The identifier name (member name for instance-less, instance name for arrays). */
      key: string;
    }
  | { kind: 'attrib'; location: number }
  | { kind: 'varying'; key: string }
  | { kind: 'output'; location: number }
  | { kind: 'builtin'; builtin: BuiltinVariable }
  | { kind: 'const'; value: number };

/* ------------------------------------------------------------------ */
/* PathRef — a resolved read/write path (identifier + member/index chains) */
/* ------------------------------------------------------------------ */

/**
 * A resolved storage/local access path — produced by the expressions walker
 * (refOf / resolveStoragePath in expressions.ts).
 * - `type` is the type OF THE VALUE AT THE PATH (array paths resolve to the
 *   element type; `m[col]` resolves to a column vector).
 * - `read(c)` / `write(c)` produce the JS expression for ONE flat scalar
 *   component c (column-major flat index for matrices).
 * - `pre` holds expressions that must run before every read AND write
 *   (dynamic-index temp assignments, flat-local spill copy-in). They are
 *   folded into the emitted expression strings by the caller.
 * - `post` holds expressions that must run AFTER writes (spill copy-out).
 */
export interface PathRef {
  type: GLSLType;
  lvalue: boolean;
  read(c: number): string;
  write(c: number): string;
  pre: string[];
  post: string[];
}

/** Build a PathRef from explicit read/write closures. */
export function mkPath(
  type: GLSLType,
  lvalue: boolean,
  read: (c: number) => string,
  write: (c: number) => string,
  pre: string[] = [],
  post: string[] = [],
): PathRef {
  return { type, lvalue, read, write, pre, post };
}

/** Fold pre-expressions into an expression: `(pre0, pre1, expr)`. */
export function foldPre(pre: string[], expr: string): string {
  if (pre.length === 0) return expr;
  return `(${pre.join(', ')}, ${expr})`;
}

/* ------------------------------------------------------------------ */
/* Storage read/write formulas                                         */
/* ------------------------------------------------------------------ */

/** Outermost dynamic-index term carried through a storage path (at most one —
 *  GLSL allows dynamic indexing on the outermost dimension only). `stride`:
 *  uniform → FLOATS per element; block → BYTES per element; varying →
 *  components per element; attrib → locations per element; output → 1. */
export interface DynTerm {
  temp: string; // temp var holding the index value
  stride: number;
}

/** Uniform store read: slot = FLOAT index; matrix columns stride 4 floats. */
export function uniformRead(
  type: GLSLType,
  slot: number,
  isIntStore: boolean,
  dyn: DynTerm | null,
  c: number,
): string {
  const base = dyn ? `${slot} + (${dyn.temp}) * ${dyn.stride}` : String(slot);
  let idx: string;
  if (type.kind === 'matrix') {
    const col = Math.floor(c / type.rows);
    const row = c % type.rows;
    idx = `${base} + ${col} * 4 + ${row}`;
  } else {
    idx = `${base} + ${c}`;
  }
  const s = `${isIntStore ? 'ctx.intUniforms' : 'ctx.uniforms'}[${idx}]`;
  return isUintType(type) ? wrapUint(s) : s;
}

/** Block store read: byte offset → float index; matrix columns stride matrixStride. */
export function blockRead(
  type: GLSLType,
  blockIndex: number,
  offset: number,
  matrixStride: number,
  isIntStore: boolean,
  dyn: DynTerm | null,
  c: number,
): string {
  const base = dyn ? `${offset} / 4 + (${dyn.temp}) * ${dyn.stride / 4}` : `${offset} / 4`;
  let idx: string;
  if (type.kind === 'matrix') {
    const col = Math.floor(c / type.rows);
    const row = c % type.rows;
    idx = `${base} + ${col} * ${matrixStride / 4} + ${row}`;
  } else {
    idx = `${base} + ${c}`;
  }
  const s = `${isIntStore ? 'ctx.blockIntStores' : 'ctx.blockStores'}[${blockIndex}][${idx}]`;
  return isUintType(type) ? wrapUint(s) : s;
}

/** Vertex varying write/read: packed contiguously at ctx.out.varyings[offset + c]. */
export function varyingVertexAccess(type: GLSLType, offset: number, dyn: DynTerm | null, c: number): string {
  const base = dyn ? `${offset} + (${dyn.temp}) * ${dyn.stride}` : String(offset);
  return `ctx.out.varyings[${base} + ${c}]`;
}

/** Fragment varying read: ctx.varyings[index].v[...] (C5 overrides via env.varyingRead). */
export function varyingFragmentRead(
  env: CodegenEnv,
  index: number,
  elemComponents: number,
  dyn: DynTerm | null,
  c: number,
): string {
  const comp = dyn ? `(${dyn.temp}) * ${dyn.stride} + ${c}` : String(c);
  return env.varyingRead(index, comp);
}

/** Declared per-location component count of an attribute type: vector → size,
 *  matrix → rows, scalar → 1. The per-vertex fetch stride MUST come from the
 *  DECLARED type — a swizzled read (a.xy on a vec4) retypes the path but the
 *  fetch stride stays the declared width (BUG 5). */
export function attribDeclComps(type: GLSLType): number {
  return type.kind === 'vector' ? type.size : type.kind === 'matrix' ? type.rows : 1;
}

/**
 * Vertex attribute read with the constant-attribute typeof guard.
 * `declComps` = DECLARED per-location component count (see attribDeclComps) —
 * it drives the per-vertex fetch stride; `type` drives the swizzle/component
 * selection and the matrix column-location math (matrices are never swizzled,
 * so whole-matrix reads keep the matrix type).
 */
export function attribRead(type: GLSLType, location: number, declComps: number, dyn: DynTerm | null, c: number): string {
  let L = String(location);
  let comp = c;
  if (dyn) L = `${L} + (${dyn.temp}) * ${dyn.stride}`;
  if (type.kind === 'matrix') {
    L = `${L} + ${Math.floor(c / type.rows)}`;
    comp = c % type.rows;
  }
  const s = `(typeof ctx.attribs[${L}] === 'number' ? (${comp} === 0 ? ctx.attribs[${L}] : ${comp} === 3 ? 1 : 0) : ctx.attribs[${L}][ctx.attribIndices[${L}] * ${declComps} + ${comp}])`;
  // Uint attributes: gl/ passes Uint32Array; wrap defensively (an Int32Array
  // view would otherwise read back signed bit patterns).
  return isUintType(type) ? wrapUint(s) : s;
}

/** Fragment output access: ctx.out.color[loc][c]. */
export function outputAccess(type: GLSLType, location: number, dyn: DynTerm | null, c: number): string {
  const L = dyn ? `${location} + (${dyn.temp}) * ${dyn.stride}` : String(location);
  return `ctx.out.color[${L}][${c}]`;
}

/* ------------------------------------------------------------------ */
/* Path-keyed read builders (used by the expressions walker)           */
/* ------------------------------------------------------------------ */

/** Read one flat component of a uniform path (entry looked up by key). */
export function uniformPathRead(
  env: CodegenEnv,
  key: string,
  type: GLSLType,
  dyn: DynTerm | null,
  c: number,
): string {
  if (type.kind === 'struct') return structPathRead(env, 'uniform', key, type, dyn, c);
  const us = env.lookupUniformSlot(key);
  if (!us) throw new Error(`codegen: missing uniformSlots entry for '${key}' (linker must emit every reachable prefix)`);
  const isIntStore = isIntegralFamily(type) || type.kind === 'sampler';
  return uniformRead(type, us.slot, isIntStore, dyn, c);
}

/** Read one flat component of a block member path (entry looked up by key). */
export function blockPathRead(
  env: CodegenEnv,
  blockIndex: number,
  key: string,
  type: GLSLType,
  dyn: DynTerm | null,
  c: number,
): string {
  if (type.kind === 'struct') return structPathRead(env, 'block', key, type, dyn, c, blockIndex);
  const entry = env.lookupBlockMember(blockIndex, key);
  if (!entry) throw new Error(`codegen: missing block layout for '${key}' (linker must emit every reachable prefix)`);
  const isIntStore = isIntegralFamily(type);
  return blockRead(type, blockIndex, entry.offset, entry.matrixStride, isIntStore, dyn, c);
}

/** Read one flat component of a varying path (entry looked up by key). */
export function varyingPathRead(
  env: CodegenEnv,
  key: string,
  type: GLSLType,
  dyn: DynTerm | null,
  c: number,
): string {
  if (type.kind === 'struct') return structPathRead(env, 'varying', key, type, dyn, c);
  const vl = env.lookupVarying(key);
  if (!vl) throw new Error(`codegen: missing varying layout for '${key}'`);
  return env.stage === 'VERTEX'
    ? varyingVertexAccess(type, vl.offset, dyn, c)
    : varyingFragmentRead(env, vl.index, vl.elemComponents, dyn, c);
}

/** Recursive struct-path read: descends members, looking up each leaf's entry. */
function structPathRead(
  env: CodegenEnv,
  kind: 'uniform' | 'block' | 'varying',
  key: string,
  type: GLSLType,
  dyn: DynTerm | null,
  c: number,
  blockIndex?: number,
): string {
  if (type.kind !== 'struct') throw new Error('codegen: structPathRead on non-struct');
  let off = 0;
  for (const m of type.members) {
    const n = flatComponents(m.type);
    if (c < off + n) {
      const subKey = `${key}.${m.name}`;
      if (m.type.kind === 'struct') {
        return structPathRead(env, kind, subKey, m.type, dyn, c - off, blockIndex);
      }
      switch (kind) {
        case 'uniform': {
          const us = env.lookupUniformSlot(subKey);
          if (!us) throw new Error(`codegen: missing uniformSlots entry for '${subKey}'`);
          const isIntStore = isIntegralFamily(m.type) || m.type.kind === 'sampler';
          return uniformRead(m.type, us.slot, isIntStore, dyn, c - off);
        }
        case 'block': {
          const entry = env.lookupBlockMember(blockIndex!, subKey);
          if (!entry) throw new Error(`codegen: missing block layout for '${subKey}'`);
          const isIntStore = isIntegralFamily(m.type);
          return blockRead(m.type, blockIndex!, entry.offset, entry.matrixStride, isIntStore, dyn, c - off);
        }
        case 'varying': {
          const vl = env.lookupVarying(subKey);
          if (!vl) throw new Error(`codegen: missing varying layout for '${subKey}'`);
          return env.stage === 'VERTEX'
            ? varyingVertexAccess(m.type, vl.offset, dyn, c - off)
            : varyingFragmentRead(env, vl.index, vl.elemComponents, dyn, c - off);
        }
      }
    }
    off += n;
  }
  throw new Error(`codegen: struct component ${c} out of range`);
}

/* ------------------------------------------------------------------ */
/* CodegenEnv                                                          */
/* ------------------------------------------------------------------ */

export class CodegenEnv {
  readonly stage: ShaderStage;
  readonly layout: CodegenLayout;
  /** Dual-number mode (C5 flips it when layout.uses.derivatives). */
  dual = false;
  /** User struct type names (C3/C4 populate from the AST; used for ctor dispatch). */
  structNames = new Set<string>();
  /**
   * User-function call hook (C3b's inliner sets it). C2 leaves it null —
   * a user call then throws "user function call outside the function inliner".
   * `rawArgs` (4th param, optional) carries the RAW argument AST exprs —
   * the inliner needs them to emitLValue out/inout arguments (the emitted
   * `args` are already-lowered Values and cannot be re-lvalued).
   */
  emitUserCall:
    | ((name: string, args: Value[][], argTypes: GLSLType[], rawArgs?: Expr[]) => Value[] | null)
    | null = null;

  /** Active inlined-call scopes (C3b): innermost last. Each frame holds the
   *  call's param LocalVars + the function's local names; resolveLocal
   *  consults the CURRENT function's frame (bodyDepth - 1) so same-named
   *  params/locals of nested calls never collide (locals_ alone cannot hold
   *  two entries per GLSL name) while caller frames stay invisible. */
  private paramFrames_: ParamFrame[] = [];

  /**
   * Inlined-callee BODY-emission depth (functions.ts increments it around
   * emitStatements of each inlined body). The CURRENT function's frame is
   * `paramFrames_[bodyDepth - 1]` (bodyDepth 0 = top-level main → no frame).
   * GLSL scoping gives a function body ONLY its own params/locals + globals
   * — NEVER the caller's locals — so frames at index < bodyDepth - 1 (caller
   * frames) must not resolve anything. Frames at index >= bodyDepth are
   * pushed solely for a nested call's ARG MATERIALIZATION (its body has not
   * started; params/locals are not registered yet), so they resolve nothing
   * either: a read there is an expression of the current function.
   */
  bodyDepth = 0;

  /** Push a frame for one inlined call (the inliner pops it after the body).
   *  `suffix` is the call site's JS-name suffix generator — frameLocal uses
   *  it to give the function's OWN locals unique per-call-site JS names. */
  pushParamFrame(suffix: (() => string) | null = null): ParamFrame {
    const frame: ParamFrame = {
      params: new Map(),
      localNames: new Set(),
      locals: new Map(),
      suffix,
    };
    this.paramFrames_.push(frame);
    return frame;
  }

  popParamFrame(): void {
    this.paramFrames_.pop();
  }

  /**
   * Create a param LocalVar for ONE inlined call site. Flat kinds get
   * per-call-site unique JS names (`<name>$c<N>` — `$` is not a GLSL
   * identifier character, so collisions with GLSL-derived names and temps
   * are impossible); array params get a fresh scratch region. Does NOT
   * register the var — the inliner stores it in the active frame.
   */
  makeParamLocal(name: string, type: GLSLType, suffix: string): LocalVar {
    if (type.kind === 'array') {
      const elem = type.element;
      const n = flatComponents(elem) * (type.size ?? 1);
      // BUG 2: the int store is only for ALL-integral blocks — a struct with
      // any float leaf allocates in the float scratch so `s[0].color = 0.5`
      // writes a float, not an int32-truncated 0 (leafRead/leafWrite select
      // the store per MEMBER type).
      const int = isIntegralFamily(elem) && !hasFloatLeaves(elem);
      const base = int ? this.allocIntScratch(n) : this.allocScratch(n);
      return {
        name,
        type,
        kind: 'scratch',
        scratchBase: base,
        blockSize: n,
        elemSlots: flatComponents(elem),
        int,
        members: elem.kind === 'struct' ? structMemberOffsets(elem) : undefined,
      };
    }
    const compNames = flatNames(name + suffix, type);
    for (const js of compNames) {
      if (this.usedJsNames_.has(js)) {
        throw new Error(`codegen: JS name '${js}' already in use (param suffix '${suffix}')`);
      }
      this.usedJsNames_.add(js);
    }
    // Dual mode: register the dx/dy names of every FLOAT component.
    const dxNames = this.dual ? this.dualNamesFor(compNames, type) : undefined;
    return {
      name,
      type,
      kind: 'flat',
      compNames,
      dxNames,
      members: type.kind === 'struct' ? structMemberOffsets(type) : undefined,
    };
  }

  /**
   * The per-call-site LocalVar for a local declared inside an INLINED
   * function body (statements.ts's emitDeclStmt consults this BEFORE the
   * locals_ reuse path — a callee local must never alias a caller's
   * same-named local, even when the types differ). Returns null when the
   * innermost frame does not declare `name` (→ the caller falls back to
   * locals_).
   *
   * The var is created lazily on the FIRST declaration (unique JS names via
   * the frame's per-call-site suffix, per-call-site scratch for arrays) and
   * cached for sibling-scope re-declarations — mirror of makeParamLocal's
   * naming/scratch machinery, so dual-mode dx/dy triples and struct member
   * offsets come for free.
   */
  frameLocal(name: string, type: GLSLType): LocalVar | null {
    const frame = this.paramFrames_[this.paramFrames_.length - 1];
    if (!frame || !frame.localNames.has(name) || frame.suffix === null) return null;
    let lv = frame.locals.get(name);
    if (lv === undefined) {
      lv = this.makeParamLocal(name, type, frame.suffix());
      frame.locals.set(name, lv);
    }
    return lv;
  }

  /**
   * Resolve an identifier for READS: the CURRENT function's scope only —
   * its param frame (params first, then per-call-site locals — an inner
   * local shadows an outer same-named param), then globals (locals_).
   * Caller frames (index < bodyDepth - 1) are NEVER consulted: per GLSL
   * scoping a function body sees only its own scope + globals, so a free
   * name matching a caller's param/local resolves to the global instead
   * (in-parameter-passed-as-inout-argument-and-global: callee G reads the
   * global `p` while caller F's param `p` is live). Frames at index >=
   * bodyDepth are pushed for a nested call's ARG MATERIALIZATION (its body
   * has not started — params/locals are not yet registered), so they
   * resolve nothing either; a read of a name the inner call DECLARES but
   * has not materialized yet refers to the current function's scope
   * (decl-before-use ⇒ `g(x)` inside f where g declares a local x: the arg
   * x is f's x, read while g's frame is already active — the current
   * function's frame is found directly, no outward walk needed).
   *
   * Declarations use lookupLocal (locals_ only — statements.ts's sibling
   * re-declaration path must not see frames).
   */
  resolveLocal(name: string): LocalVar | null {
    const cur = this.bodyDepth - 1;
    if (cur >= 0) {
      const f = this.paramFrames_[cur];
      const p = f.params.get(name);
      if (p) return p;
      const l = f.locals.get(name);
      if (l) return l;
    }
    return this.locals_.get(name) ?? null;
  }

  private locals_ = new Map<string, LocalVar>();
  private usedJsNames_ = new Set<string>();
  private scratchTop_ = 0;
  private intScratchTop_ = 0;
  private tempCount_ = 0;
  /** Temp var names allocated so far (C3/C4 hoist `var t0, t1, ...`). */
  readonly temps: string[] = [];
  /** Max float scratch used (StageCodegenResult.scratchSize). */
  scratchSize = 0;
  /** Max int scratch used (StageCodegenResult.intScratchSize). */
  intScratchSize = 0;

  constructor(stage: ShaderStage, layout: CodegenLayout) {
    this.stage = stage;
    this.layout = layout;
  }

  /* ---------------- locals ---------------- */

  /** Register a local variable. Arrays (and `opts.array`) become scratch-backed. */
  declareLocal(name: string, type: GLSLType, opts?: { array?: boolean }): void {
    if (this.locals_.has(name)) {
      throw new Error(`codegen: duplicate local '${name}'`);
    }
    const isArr = type.kind === 'array';
    if (isArr || opts?.array) {
      const elem = isArr ? type.element : type;
      const n = flatComponents(elem) * (isArr ? type.size ?? 1 : 1);
      // BUG 2: int store only when ALL leaves are integral (see makeParamLocal).
      const int = isIntegralFamily(elem) && !hasFloatLeaves(elem);
      const base = int ? this.allocIntScratch(n) : this.allocScratch(n);
      const lv: LocalVar = {
        name,
        type,
        kind: 'scratch',
        scratchBase: base,
        blockSize: n,
        elemSlots: flatComponents(elem),
        int,
        members: elem.kind === 'struct' ? structMemberOffsets(elem) : undefined,
      };
      this.locals_.set(name, lv);
      return;
    }
    const compNames = flatNames(name, type);
    for (const js of compNames) {
      if (this.usedJsNames_.has(js)) {
        throw new Error(
          `codegen: local '${name}' collides with JS name '${js}' (rename the GLSL variable)`,
        );
      }
      this.usedJsNames_.add(js);
    }
    const dxNames = this.dual ? this.dualNamesFor(compNames, type) : undefined;
    this.locals_.set(name, {
      name,
      type,
      kind: 'flat',
      compNames,
      dxNames,
      members: type.kind === 'struct' ? structMemberOffsets(type) : undefined,
    });
  }

  /** Dual-mode: derive + register the `_dx`/`_dy` JS names for every FLOAT
   *  flat component of `type` (parallel to `compNames`; null = non-float).
   *  Convention: dx = `compNames[c] + '_dx'`, dy = `compNames[c] + '_dy'` —
   *  every reader (leafRead/leafWrite dual paths, statements, the inliner)
   *  derives the same names, so nothing else needs to store them. */
  private dualNamesFor(compNames: string[], type: GLSLType): (string | null)[] {
    const floatness = flatFloatness(type);
    const dxNames: (string | null)[] = compNames.map((n, c) =>
      floatness[c] ? `${n}_dx` : null,
    );
    for (let c = 0; c < compNames.length; c++) {
      if (!floatness[c]) continue;
      const dx = dxNames[c]!;
      const dy = `${compNames[c]}_dy`;
      if (this.usedJsNames_.has(dx)) {
        throw new Error(`codegen: local '${compNames[c]}' collides with JS name '${dx}' (rename the GLSL variable)`);
      }
      if (this.usedJsNames_.has(dy)) {
        throw new Error(`codegen: local '${compNames[c]}' collides with JS name '${dy}' (rename the GLSL variable)`);
      }
      this.usedJsNames_.add(dx);
      this.usedJsNames_.add(dy);
    }
    return dxNames;
  }

  /** Look up a declared local (null = not a local; try globalInfo next). */
  lookupLocal(name: string): LocalVar | null {
    return this.locals_.get(name) ?? null;
  }

  /**
   * Dynamic-index spill for a FLAT vector/matrix local: allocates (once per
   * variable, cached) a scratch block of `flatComponents(type)` elements and
   * returns the per-access copy-in / copy-out assignments that keep the flat
   * vars in sync. `copyIn` (flat → scratch) must run before the dynamic
   * access, `copyOut` (scratch → flat) after a dynamic WRITE. Static access
   * to the same variable is unaffected (flat vars stay the source of truth).
   * Uint locals wrap reads with `>>> 0` to restore the uint invariant
   * (Int32Array reads come back signed).
   */
  ensureDynScratch(name: string): { base: number; int: boolean; copyIn: string[]; copyOut: string[] } {
    // Same scope rule as resolveLocal (current function's frame, then
    // globals): the callers always pass p.local.name, so this must find the
    // var resolveLocal produced — never a same-named global shadowed by a
    // current-frame param/local, and never a caller frame's var (a callee
    // body cannot see caller locals).
    const lv = this.resolveLocal(name);
    if (!lv) throw new Error(`codegen: unknown local '${name}'`);
    if (lv.kind === 'scratch') {
      throw new Error(`codegen: '${name}' is already scratch-backed (cannot dyn-spill)`);
    }
    if (lv.type.kind !== 'vector' && lv.type.kind !== 'matrix') {
      throw new Error(`codegen: '${name}' (${lv.type.kind}) cannot be dynamically indexed`);
    }
    if (lv.dynScratch) return lv.dynScratch;
    const n = flatComponents(lv.type);
    const int = isIntegralFamily(lv.type);
    const base = int ? this.allocIntScratch(n) : this.allocScratch(n);
    const store = int ? 'ctx.intScratch' : 'ctx.scratch';
    const wrap = lv.type.kind === 'vector' && lv.type.base === 'uint' ? ' >>> 0' : '';
    const dual = this.dual && !int;
    const copyIn: string[] = [];
    const copyOut: string[] = [];
    for (let k = 0; k < n; k++) {
      copyIn.push(`${store}[${base} + ${k}] = ${lv.compNames![k]}`);
      // Self-parenthesized read-back: the `>>> 0` wrap must sit inside the
      // parens so the RHS is an atom (`x = (e) >>> 0` mis-parses if this
      // statement is ever embedded in a larger expression). Non-uint reads
      // keep the plain `(e)` form (already an atom).
      copyOut.push(
        wrap ? `${lv.compNames![k]} = ((${store}[${base} + ${k}])${wrap})` : `${lv.compNames![k]} = (${store}[${base} + ${k}])`,
      );
      if (dual) {
        // Dual planes: dx at base+n, dy at base+2n (allocScratch charged 3n).
        const dx = `${lv.compNames![k]}_dx`;
        const dy = `${lv.compNames![k]}_dy`;
        copyIn.push(`${store}[${base} + ${n} + ${k}] = ${dx}`);
        copyIn.push(`${store}[${base} + ${2 * n} + ${k}] = ${dy}`);
        copyOut.push(`${dx} = (${store}[${base} + ${n} + ${k}])`);
        copyOut.push(`${dy} = (${store}[${base} + ${2 * n} + ${k}])`);
      }
    }
    lv.dynScratch = { base, int, copyIn, copyOut };
    return lv.dynScratch;
  }

  /* ---------------- scratch ---------------- */

  /** Allocate `n` float scratch elements; returns the base offset. In dual
   *  mode every float block is THREE planes (v, dx, dy) — the allocation
   *  charges 3×n and `scratchSize` reflects the planes (the caller's base
   *  stays the v-plane start). */
  allocScratch(n: number): number {
    const base = this.scratchTop_;
    this.scratchTop_ += n * (this.dual ? 3 : 1);
    if (this.scratchTop_ > this.scratchSize) this.scratchSize = this.scratchTop_;
    return base;
  }

  /** Allocate `n` int scratch elements; returns the base offset. */
  allocIntScratch(n: number): number {
    const base = this.intScratchTop_;
    this.intScratchTop_ += n;
    if (this.intScratchTop_ > this.intScratchSize) this.intScratchSize = this.intScratchTop_;
    return base;
  }

  /** Allocate one temp var name (`t0`, `t1`, ...) for index materialization. */
  allocTemp(): string {
    const t = `t${this.tempCount_++}`;
    this.temps.push(t);
    return t;
  }

  /* ---------------- storage access ---------------- */

  /** Storage-backed global info for a simple identifier; null = unknown.
   *  Struct/array ROOT identifiers resolve by PREFIX scan: the linker keys
   *  leaves ('u.m', 'u[0].m'), so a bare struct/array name ('u') that is not
   *  itself keyed resolves when any key starts with 'name.' or 'name['. */
  globalInfo(name: string): GlobalInfo | null {
    const layout = this.layout;
    const us = layout.uniformSlots.get(name);
    if (us) return { kind: 'uniform', key: name };
    for (const k of layout.uniformSlots.keys()) {
      if (k.startsWith(name + '.') || k.startsWith(name + '[')) {
        return { kind: 'uniform', key: name };
      }
    }
    const bi = layout.blockIndices.get(name);
    if (bi !== undefined) {
      const members = layout.blocks.get(bi);
      if (members && members.has(name)) {
        // Instance-less block: the identifier IS a bare member name.
        return { kind: 'block', blockIndex: bi, baseKey: '', key: name };
      }
      // Arrayed block: the identifier is the instance name ('b[0].m' keys).
      return { kind: 'block', blockIndex: bi, baseKey: name, key: name };
    }
    const al = layout.attribLocations.get(name);
    if (al !== undefined) return { kind: 'attrib', location: al };
    const vl = layout.varyings.get(name);
    if (vl) return { kind: 'varying', key: name };
    for (const k of layout.varyings.keys()) {
      if (k.startsWith(name + '.') || k.startsWith(name + '[')) {
        return { kind: 'varying', key: name };
      }
    }
    const ol = layout.outputLocations.get(name);
    if (ol !== undefined) return { kind: 'output', location: ol };
    const version = layout.version;
    const bt =
      builtinVariables(version).find((b) => b.name === name) ??
      extensionVariables.find((b) => b.name === name);
    if (bt) return { kind: 'builtin', builtin: bt };
    const bc =
      builtinConstants(version).find((b) => b.name === name) ??
      extensionConstants.find((b) => b.name === name);
    if (bc) return { kind: 'const', value: bc.value };
    return null;
  }

  /** Uniform slot for a path key ('u', 'u.m', 'u[0]', 'u[2].m', ...). */
  lookupUniformSlot(key: string): UniformSlot | null {
    return this.layout.uniformSlots.get(key) ?? null;
  }

  /** Block member layout for (blockIndex, path key). */
  lookupBlockMember(blockIndex: number, key: string): BlockMemberLayout | null {
    return this.layout.blocks.get(blockIndex)?.get(key) ?? null;
  }

  /** Varying layout for a path key ('v', 'v.m' for struct varyings). */
  lookupVarying(key: string): VaryingLayout | null {
    return this.layout.varyings.get(key) ?? null;
  }

  /** Fragment varying read hook (C5's dual mode overrides for .ddx/.ddy).
   *  `c` is a component EXPRESSION (may include a dynamic-index term).
   *  This is the v-plane read; the dual planes go through varyingReadDual. */
  varyingRead(index: number, c: string): string {
    return `ctx.varyings[${index}].v[${c}]`;
  }

  /**
   * Fragment varying DUAL read hook (C5): the [dx, dy] plane reads for one
   * component expression `c`. FLAT varyings are constant across the fragment
   * (dx=dy=0); non-flat float varyings read the raster-supplied derivative
   * arrays. The raster provides `ctx.varyings[i].ddx/ddy` whenever the
   * program usesDerivatives (contract §1: "derivative arrays valid only when
   * usesDerivatives") — the generated code reads them DIRECTLY, no guards
   * (gl/ must always supply the arrays in dual mode).
   */
  varyingReadDual(index: number, c: string, flat: boolean): [string, string] {
    if (flat) return ['0', '0'];
    return [`ctx.varyings[${index}].ddx[${c}]`, `ctx.varyings[${index}].ddy[${c}]`];
  }

  /**
   * Dual-mode write expression for ONE float component.
   * `target` = the v-plane lvalue string; `dual` = [dx, dy] lvalue strings
   * (null when the target has no dual planes — outputs, gl_FragDepth — or
   * the component is not float); `val` = the RHS Value (dx/dy absent = the
   * constant 0 — covers int→float conversions and literals). With `op`
   * ('+','-','*','/','%') emits a compound update. Returns a comma
   * expression ENDING WITH THE v READ:
   *   '='      : (vslot = vv, dxslot = dxv, dyslot = dyv, vslot)
   *   '+='/'-=': (vslot = vslot op vv, dxslot = dxslot op dxv, dyslot = dyslot op dyv, vslot)
   * so both `target = <result>` statement emitters and expression contexts
   * stay correct.
   *
   * NON-LINEAR COMPOUNDS ('*','/','%' — the C5a2 arithmetic dual templates):
   * the dual terms read the OLD v/dx slots, so they are ordered FIRST (comma
   * evaluates left-to-right), the v update LAST. The RHS `val.v` is captured
   * into a fresh temp (`t0 = val.v`) as the FIRST term — this runs any folded
   * RHS pres exactly once (the RHS dx/dy strings may reference temps those
   * pres set) and makes the RHS available to all three plane updates without
   * re-evaluation (a side-effectful RHS — assignment, inlined call — must
   * run once):
   *   '*=': (t0 = vv, dxslot = dxslot * t0 + target * dxv,
   *          dyslot = dyslot * t0 + target * dyv, target = target * t0, target)
   *   '/=': (t0 = vv, dxslot = (dxslot * t0 - target * dxv) / (t0 * t0),
   *          dyslot = (dyslot * t0 - target * dyv) / (t0 * t0),
   *          target = target / t0, target)
   *   '%=': (t0 = vv, dxslot = dxslot - Math.floor(target / t0) * dxv,
   *          dyslot = dyslot - Math.floor(target / t0) * dyv,
   *          target = target % t0, target)
   * The '%=' v plane mirrors the non-dual float '%' (JS remainder); the dual
   * planes use the GLSL mod view dv = da − floor(a/b)·db (floor is a.e.
   * constant, so its derivative vanishes). Int/uint compounds ('<<=' etc.)
   * never reach this hook — the assignment emitters gate on float targets
   * (int targets have no dual planes).
   *
   * Non-dual mode / no-dual-plane targets degrade to a plain
   * `(target = val.v)` — byte-identical to the pre-dual emitters.
   */
  dualWrite(target: string, dual: [string, string] | null, val: Value, op?: string): string {
    if (!this.dual) return `(${target} = ${val.v})`;
    if (!dual) {
      if (op !== undefined) {
        throw new Error(`codegen: dual-mode compound-assign '${op}=' on a target without dual planes`);
      }
      return `(${target} = ${val.v})`;
    }
    const dxv = val.dx ?? '0';
    const dyv = val.dy ?? '0';
    if (op === '+' || op === '-') {
      return `(${target} = ${target} ${op} ${val.v}, ${dual[0]} = ${dual[0]} ${op} ${dxv}, ${dual[1]} = ${dual[1]} ${op} ${dyv}, ${target})`;
    }
    if (op === '*') {
      // Matrix×matrix '*=' NEVER reaches this template — the assignment
      // emitters (statements.ts emitAssignStmt/updateString, expressions.ts
      // emitAssign) intercept it and lower via matrixCompoundMul (matrix
      // PRODUCT with LHS snapshot + RHS materialization; dual aware). This
      // per-component product rule serves scalar/vector targets and mat×scalar
      // broadcast only.
      const t = this.allocTemp();
      return `(${t} = ${val.v}, ${dual[0]} = ${dual[0]} * ${t} + ${target} * ${dxv}, ${dual[1]} = ${dual[1]} * ${t} + ${target} * ${dyv}, ${target} = ${target} * ${t}, ${target})`;
    }
    if (op === '/') {
      const t = this.allocTemp();
      return `(${t} = ${val.v}, ${dual[0]} = (${dual[0]} * ${t} - ${target} * ${dxv}) / (${t} * ${t}), ${dual[1]} = (${dual[1]} * ${t} - ${target} * ${dyv}) / (${t} * ${t}), ${target} = ${target} / ${t}, ${target})`;
    }
    if (op === '%') {
      const t = this.allocTemp();
      return `(${t} = ${val.v}, ${dual[0]} = ${dual[0]} - Math.floor(${target} / ${t}) * ${dxv}, ${dual[1]} = ${dual[1]} - Math.floor(${target} / ${t}) * ${dyv}, ${target} = ${target} % ${t}, ${target})`;
    }
    if (op !== undefined) {
      throw new Error(`codegen: dual-mode compound-assign '${op}=' requires arithmetic dual templates (C5a2)`);
    }
    return `(${target} = ${val.v}, ${dual[0]} = ${dxv}, ${dual[1]} = ${dyv}, ${target})`;
  }

  /**
   * JS expression reading ONE flat component of a simple identifier
   * (local or storage-backed global; no member/index chains). `type` is the
   * identifier's resolvedType (required for matrix storage mapping; defaults
   * to scalar when omitted).
   */
  scalarRead(name: string, component: number, type?: GLSLType): string {
    const local = this.locals_.get(name);
    if (local) {
      if (local.kind === 'scratch') {
        throw new Error(`codegen: cannot read array '${name}' as a value (index it)`);
      }
      return local.compNames![component];
    }
    const info = this.globalInfo(name);
    if (!info) throw new Error(`codegen: unknown identifier '${name}'`);
    const t = type ?? { kind: 'scalar', base: 'float' };
    const ref = globalPathRef(this, info, t);
    return foldPre(ref.pre, ref.read(component));
  }

  /** JS lvalue string for ONE flat component of a simple writable identifier. */
  scalarWrite(name: string, component: number, type?: GLSLType): string {
    const local = this.locals_.get(name);
    if (local) {
      if (local.kind === 'scratch') {
        throw new Error(`codegen: cannot write array '${name}' as a value (index it)`);
      }
      return local.compNames![component];
    }
    const info = this.globalInfo(name);
    if (!info) throw new Error(`codegen: unknown identifier '${name}'`);
    if (info.kind === 'uniform' || info.kind === 'attrib' || info.kind === 'const') {
      throw new Error(`codegen: '${name}' is read-only`);
    }
    if (info.kind === 'builtin' && !info.builtin.writable) {
      throw new Error(`codegen: cannot write read-only builtin '${name}'`);
    }
    const t = type ?? { kind: 'scalar', base: 'float' };
    const ref = globalPathRef(this, info, t);
    if (!ref.lvalue) throw new Error(`codegen: '${name}' is not writable`);
    return foldPre(ref.pre, ref.write(component));
  }

  /* ---------------- literals ---------------- */

  /** Emit a JS literal for a GLSL constant. Float constants integral-valued
   *  get a `.0` suffix so JS keeps them float; uint values > 2^31 are emitted
   *  as plain JS numbers (exact in fp64; ops wrap at their use site). */
  emitConstNumber(v: number | boolean, type: GLSLType): string {
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    switch (type.kind) {
      case 'scalar':
        switch (type.base) {
          case 'float': {
            const s = String(v);
            // Integer-valued floats get a `.0` suffix so JS keeps them float —
            // but never when the string form already carries an exponent:
            // "1e+100.0" is a JS SyntaxError inside new Function, while
            // "1e+100" alone is a valid JS numeric literal (CTS
            // float_literal.vert / overflow_leak.vert use 1E100 literals).
            if (Number.isInteger(v) && Number.isFinite(v) && !/[eE]/.test(s)) return `${s}.0`;
            if (v === Infinity) return 'Infinity';
            if (v === -Infinity) return '-Infinity';
            return s;
          }
          case 'uint':
            return String(v >>> 0);
          case 'int':
            return String(v | 0);
          case 'bool':
            return v ? 'true' : 'false';
        }
        break;
      default:
        break;
    }
    throw new Error(`codegen: emitConstNumber on non-scalar type`);
  }
}

/* ------------------------------------------------------------------ */
/* Identifier-level PathRef construction                               */
/* ------------------------------------------------------------------ */

/** Build the PathRef for a storage-backed global identifier. */
export function globalPathRef(env: CodegenEnv, info: GlobalInfo, type: GLSLType): PathRef {
  switch (info.kind) {
    case 'uniform': {
      const read = (c: number): string => uniformPathRead(env, info.key, type, null, c);
      return mkPath(type, false, read, () => {
        throw new Error('codegen: uniforms are read-only');
      });
    }
    case 'block':
      return blockIdentifierRef(env, info, type);
    case 'attrib': {
      // Identifier-level reads carry the DECLARED type — declComps derives
      // from it (swizzles happen in the expressions walker, which carries
      // declComps on the storage instead).
      const declComps = attribDeclComps(type);
      const read = (c: number): string => attribRead(type, info.location, declComps, null, c);
      return mkPath(type, false, read, () => {
        throw new Error('codegen: attributes are read-only');
      });
    }
    case 'varying': {
      // varyingPathRead dispatches by stage AND handles struct varyings via
      // per-member layout lookups ('v.m' keys).
      if (env.stage === 'VERTEX') {
        return mkPath(
          type,
          true,
          (c) => varyingPathRead(env, info.key, type, null, c),
          (c) => varyingPathRead(env, info.key, type, null, c),
        );
      }
      return mkPath(
        type,
        false,
        (c) => varyingPathRead(env, info.key, type, null, c),
        () => {
          throw new Error('codegen: fragment varyings are read-only');
        },
      );
    }
    case 'output': {
      const read = (c: number): string => outputAccess(type, info.location, null, c);
      const write = (c: number): string => outputAccess(type, info.location, null, c);
      return mkPath(type, true, read, write);
    }
    case 'builtin':
      return builtinRef(env, info, type);
    case 'const':
      throw new Error(`codegen: const global has no storage (fold via constValue)`);
  }
}

/** Identifier-level PathRef for a uniform-block path. */
function blockIdentifierRef(
  env: CodegenEnv,
  info: Extract<GlobalInfo, { kind: 'block' }>,
  type: GLSLType,
): PathRef {
  const members = env.layout.blocks.get(info.blockIndex);
  if (!members) throw new Error(`codegen: missing layout for block ${info.blockIndex}`);
  if (info.baseKey !== '') {
    // Instance array: the identifier is the instance name — requires indexing.
    return mkPath(type, false, () => {
      throw new Error(`codegen: block instance '${info.key}' must be indexed`);
    }, () => {
      throw new Error('codegen: block members are read-only');
    });
  }
  // Instance-less: the identifier is a bare member name.
  const entry = members.get(info.key);
  if (!entry) {
    throw new Error(
      `codegen: missing block member layout for '${info.key}' (linker must emit every reachable prefix)`,
    );
  }
  const read = (c: number): string => blockPathRead(env, info.blockIndex, info.key, type, null, c);
  return mkPath(type, false, read, () => {
    throw new Error('codegen: block members are read-only');
  });
}

function builtinRef(env: CodegenEnv, info: Extract<GlobalInfo, { kind: 'builtin' }>, type: GLSLType): PathRef {
  const name = info.builtin.name;
  const ro = (): never => {
    throw new Error(`codegen: '${name}' is read-only`);
  };
  switch (name) {
    case 'gl_Position':
      return mkPath(type, true, (c) => `ctx.out.position[${c}]`, (c) => `ctx.out.position[${c}]`);
    case 'gl_PointSize':
      return mkPath(type, true, () => 'ctx.out.pointSize', () => 'ctx.out.pointSize');
    case 'gl_FragCoord':
      return mkPath(type, false, (c) => `ctx.fragCoord[${c}]`, ro);
    case 'gl_FrontFacing':
      return mkPath(type, false, () => 'ctx.frontFacing', ro);
    case 'gl_PointCoord':
      return mkPath(type, false, (c) => `ctx.pointCoord[${c}]`, ro);
    case 'gl_VertexID':
      return mkPath(type, false, () => 'ctx.vertexId', ro);
    case 'gl_InstanceID':
      return mkPath(type, false, () => 'ctx.instanceId', ro);
    case 'gl_DrawID':
      return mkPath(type, false, () => 'ctx.drawId', ro);
    case 'gl_FragDepth':
    case 'gl_FragDepthEXT':
      return mkPath(type, true, () => 'ctx.out.fragDepth', () => 'ctx.out.fragDepth');
    case 'gl_FragColor': {
      const loc = env.layout.outputLocations.get('gl_FragColor') ?? 0;
      return mkPath(type, true, (c) => `ctx.out.color[${loc}][${c}]`, (c) => `ctx.out.color[${loc}][${c}]`);
    }
    case 'gl_FragData':
      // Handled by the walker (identRef) as an output-array path; the
      // identifier-level whole-array access is an error.
      return mkPath(type, true, () => {
        throw new Error('codegen: gl_FragData must be indexed');
      }, () => {
        throw new Error('codegen: gl_FragData must be indexed');
      });
    default:
      throw new Error(`codegen: unsupported builtin '${name}'`);
  }
}
