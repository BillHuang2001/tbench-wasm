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

/** Wrap an expression string to int32 range (int-typed results). */
export function wrapInt(s: string): string {
  return `(${s}) | 0`;
}

/** Wrap an expression string to uint32 range (uint-typed results). */
export function wrapUint(s: string): string {
  return `(${s}) >>> 0`;
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

/** Convert a Value[] (flat components of `from`) to `to`'s scalar base. */
export function convertValue(vals: Value[], from: GLSLType, to: GLSLType): Value[] {
  const fb = scalarBaseOf(from);
  const tb = scalarBaseOf(to);
  if (fb === null || tb === null || fb === tb) return vals;
  return vals.map((v) => ({ v: convertScalar(v.v, fb, tb) }));
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
  /**
   * Dynamic-index spill backing for FLAT vector/matrix locals (set on first
   * dynamic index; see ensureDynScratch). The flat vars stay the source of
   * truth: every dynamic access copies in before and copies out after, so
   * static accesses continue to read/write the flat vars directly.
   */
  dynScratch?: { base: number; int: boolean; copyIn: string[]; copyOut: string[] };
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

/** Vertex attribute read with the constant-attribute typeof guard. */
export function attribRead(type: GLSLType, location: number, dyn: DynTerm | null, c: number): string {
  let L = String(location);
  let comp = c;
  if (dyn) L = `${L} + (${dyn.temp}) * ${dyn.stride}`;
  if (type.kind === 'matrix') {
    L = `${L} + ${Math.floor(c / type.rows)}`;
    comp = c % type.rows;
  }
  const comps = type.kind === 'matrix' ? type.rows : type.kind === 'vector' ? type.size : 1;
  const s = `(typeof ctx.attribs[${L}] === 'number' ? (${comp} === 0 ? ctx.attribs[${L}] : ${comp} === 3 ? 1 : 0) : ctx.attribs[${L}][ctx.attribIndices[${L}] * ${comps} + ${comp}])`;
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
   * User-function call hook (C3's inliner sets it). C2 leaves it null —
   * a user call then throws "user function call outside the function inliner".
   */
  emitUserCall: ((name: string, args: Value[][], argTypes: GLSLType[]) => Value[] | null) | null = null;

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
      const int = isIntegralFamily(elem);
      const base = int ? this.allocIntScratch(n) : this.allocScratch(n);
      const lv: LocalVar = {
        name,
        type,
        kind: 'scratch',
        scratchBase: base,
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
    this.locals_.set(name, {
      name,
      type,
      kind: 'flat',
      compNames,
      members: type.kind === 'struct' ? structMemberOffsets(type) : undefined,
    });
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
    const lv = this.locals_.get(name);
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
    const copyIn: string[] = [];
    const copyOut: string[] = [];
    for (let k = 0; k < n; k++) {
      copyIn.push(`${store}[${base} + ${k}] = ${lv.compNames![k]}`);
      copyOut.push(`${lv.compNames![k]} = (${store}[${base} + ${k}])${wrap}`);
    }
    lv.dynScratch = { base, int, copyIn, copyOut };
    return lv.dynScratch;
  }

  /* ---------------- scratch ---------------- */

  /** Allocate `n` float scratch elements; returns the base offset. */
  allocScratch(n: number): number {
    const base = this.scratchTop_;
    this.scratchTop_ += n;
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
   *  `c` is a component EXPRESSION (may include a dynamic-index term). */
  varyingRead(index: number, c: string): string {
    return `ctx.varyings[${index}].v[${c}]`;
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
            if (Number.isInteger(v) && Number.isFinite(v)) return `${v}.0`;
            if (v === Infinity) return 'Infinity';
            if (v === -Infinity) return '-Infinity';
            return String(v);
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
      const read = (c: number): string => attribRead(type, info.location, null, c);
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
