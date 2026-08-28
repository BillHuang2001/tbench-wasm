/**
 * expressions.ts — non-dual GLSL→JS expression lowering (contract §1 + env.ts).
 *
 * CONVENTION (the Value[]-per-expression model): emitExpr returns ONE `Value`
 * per flat scalar component, in column-major order (matrix component =
 * col*rows + row; total = cols*rows). Scalars/samplers → length-1. Non-dual
 * mode sets only `v` (plus optional `pre`); the dual-mode task (C5) fills
 * `dx`/`dy` on the same seam.
 *
 * `pre` (Value.pre) = PURE materialization statements that must run ONCE
 * before `v` evaluates (texture-sample chains, dynamic-index temps,
 * flat-local spill copy-in). They are folded inline as `(pre, v)`; when a
 * value is USED MORE THAN ONCE (ternary arms, ctor splat args, && / || rhs
 * with side-effect-free-but-expensive pres), materialize() hoists it into a
 * temp var (`tN` — C3/C4 must declare `var t0, t1, ...` at the body top).
 * GLSL side effects (assignments, out params) NEVER go into pre — they fold
 * into `v` so they execute exactly when the surrounding JS evaluates them.
 *
 * UNIFORM STORE INDEXING (CRITICAL — linker contract, see env.ts):
 * UniformSlot.slot is a FLOAT index. vec4 at slot L → ctx.uniforms[L..L+3];
 * matC at L → C*4 consecutive floats, column col at L + col*4 + row.
 * Int/uint/bool/sampler uniforms live in ctx.intUniforms (1 int per
 * component). The linker MUST match this exact layout.
 *
 * STORAGE PATH KEYS (linker contract): uniformSlots keys include every
 * reachable prefix ('u', 'u.m', 'u[0]', 'u[0].m', 'u[2].m'); block member
 * keys likewise ('m', 'm[0]', 'm[2]', 'b[0].m' + the 'b[0]' instance prefix
 * with blockStride); varying keys EXCLUDE array indices entirely (const
 * indices fold into the flat offset via VaryingLayout.elemComponents;
 * dynamic indices use elemComponents as stride) and struct-array members are
 * keyed 'v.m' (offset = element-0 member offset).
 */
import type { Expr } from '../ast.js';
import type { GLSLType } from '../types.js';
import { typeEquals, typeName } from '../types.js';
import {
  CodegenEnv,
  LocalVar,
  flatComponents,
  flatFloatness,
  scalarBaseOf,
  isUintType,
  isIntType,
  isFloatType,
  isFloatLeaf,
  hasFloatLeaves,
  isIntegralFamily,
  wrapInt,
  wrapUint,
  convertValue,
  foldPre,
  structMemberOffset,
  structMemberOffsets,
  uniformPathRead,
  blockPathRead,
  varyingPathRead,
  attribRead,
  attribDeclComps,
  outputAccess,
  unpackVaryingCell,
  packVaryingWrite,
  packVaryingCompound,
} from './env.js';
import type { Value } from './index.js';
import { emitConstructorCall } from './expr-ctor.js';
import { emitBuiltinCall } from './expr-builtins.js';

/* ------------------------------------------------------------------ */
/* The path model                                                      */
/* ------------------------------------------------------------------ */

type StorageKind =
  | { kind: 'uniform'; key: string }
  | { kind: 'block'; blockIndex: number; key: string; baseKey: string }
  | { kind: 'varying'; key: string }
  | {
      kind: 'attrib';
      location: number;
      /** Declared per-location component count (BUG 5): the per-vertex fetch
       *  stride. Set at identifier resolution and carried through swizzles
       *  (which retype p.type but must NOT change the stride). */
      declComps: number;
    }
  | { kind: 'output'; location: number; index: number };

/** Walked access path: identifier + member/index chains. `dyn` (outermost
 *  dynamic index) is at most one — GLSL allows dynamic indexing on the
 *  outermost array dimension only. `dyn.stride` = storage stride per element
 *  (uniform: floats; block: bytes; varying: components; attrib: locations;
 *  output: 1); `dyn.elemSlots` = flat components per element for LOCAL
 *  scratch storage; `dyn.blockElements` = the dynamic index selects among the
 *  per-element STORES of an ARRAYED uniform block (store array index strides
 *  by 1, offsets element-local — see env.ts DynTerm). */
interface P {
  type: GLSLType;      // type OF THE VALUE AT THE PATH END
  lvalue: boolean;
  local: LocalVar | null;
  flatOff: number;     // flat component offset (locals: from storage base; storage: from leaf base)
  storage: StorageKind | null;
  builtin: string | null; // gl_Position / gl_FragCoord / ... (gl_FragData converts to output)
  swz: number[] | null;   // swizzle remap: leafRead(c) = baseRead(swz[c])
  dyn: { temp: string; stride: number; elemSlots: number; blockElements?: boolean } | null;
  pre: string[];
  post: string[];
}

function freshP(type: GLSLType): P {
  return {
    type,
    lvalue: false,
    local: null,
    flatOff: 0,
    storage: null,
    builtin: null,
    swz: null,
    dyn: null,
    pre: [],
    post: [],
  };
}

/* ------------------------------------------------------------------ */
/* Leaf read / write                                                   */
/* ------------------------------------------------------------------ */

function leafRead(p: P, env: CodegenEnv, c: number): string {
  const cc = p.swz ? p.swz[c] : c;
  if (p.local) {
    const lv = p.local;
    if (lv.kind === 'flat') {
      return lv.compNames![p.flatOff + cc];
    }
    // BUG 2: the store is chosen per MEMBER type. lv.int is set only when ALL
    // leaves of the block are integral, so an int member of a mixed struct
    // array (float store) reads from ctx.scratch — float32 storage is exact
    // for |v| < 2^24, and uint wraps below via isUintType.
    const store = lv.int && isIntegralFamily(p.type) ? 'ctx.intScratch' : 'ctx.scratch';
    const dyn = p.dyn ? ` + (${p.dyn.temp}) * ${p.dyn.elemSlots}` : '';
    const s = `${store}[${lv.scratchBase} + ${p.flatOff}${dyn} + ${cc}]`;
    return isUintType(p.type) ? wrapUint(s) : s;
  }
  if (p.storage) {
    const st = p.storage;
    const flatC = p.flatOff + cc;
    switch (st.kind) {
      case 'uniform':
        return uniformPathRead(env, st.key, p.type, p.dyn, flatC);
      case 'block':
        return blockPathRead(env, st.blockIndex, st.key, p.type, p.dyn, flatC);
      case 'varying': {
        const s = varyingPathRead(env, st.key, p.type, p.dyn, flatC);
        // Packed int/uint varying read-back (VERTEX stage — see
        // packVaryingWrite): the cell holds the value's bit pattern — unpack
        // to the int32/uint32 value.
        if (env.stage === 'VERTEX' && isUintType(p.type)) return unpackVaryingCell(s, false);
        if (env.stage === 'VERTEX' && isIntType(p.type)) return unpackVaryingCell(s, true);
        return s;
      }
      case 'attrib':
        // BUG 5: fetch stride = DECLARED comps (st.declComps), not p.type's
        // (swizzles retype p.type to the swizzle width).
        return attribRead(p.type, st.location, st.declComps, p.dyn, flatC);
      case 'output':
        return outputAccess(p.type, st.location, st.index, p.dyn, flatC);
    }
  }
  if (p.builtin) {
    const i = p.flatOff + cc;
    switch (p.builtin) {
      case 'gl_Position':
        return `ctx.out.position[${i}]`;
      case 'gl_PointSize':
        return 'ctx.out.pointSize';
      case 'gl_FragCoord':
        return `ctx.fragCoord[${i}]`;
      case 'gl_FrontFacing':
        return 'ctx.frontFacing';
      case 'gl_PointCoord':
        return `ctx.pointCoord[${i}]`;
      case 'gl_VertexID':
        return 'ctx.vertexId';
      case 'gl_InstanceID':
        return 'ctx.instanceId';
      case 'gl_DrawID':
        return 'ctx.drawId';
      case 'gl_FragDepth':
      case 'gl_FragDepthEXT':
        return 'ctx.out.fragDepth';
      case 'gl_FragColor': {
        const loc = env.layout.outputLocations.get('gl_FragColor') ?? 0;
        return `ctx.out.color[${loc}][${i}]`;
      }
      case 'gl_DepthRange':
        // Builtin struct uniform gl_DepthRange: members near/far/diff are
        // flat offsets 0/1/2 → ctx.depthRange = [near, far, far−near].
        return `ctx.depthRange[${i}]`;
      default:
        throw new Error(`codegen: unsupported builtin '${p.builtin}'`);
    }
  }
  throw new Error('codegen: empty path');
}

/**
 * Dual-mode READ planes for one flat component of a float leaf: [dx, dy]
 * JS expressions (null = the component carries no duals). Mirror of
 * leafRead — same swizzle/offset/dyn resolution; the caller gates on
 * isFloatLeaf(p.type) (int/bool leaves never reach here).
 * - flat locals: the registered `_dx`/`_dy` JS names (env convention);
 * - float scratch locals: the dx plane at base+blockSize, dy at
 *   base+2*blockSize (allocScratch charged 3× in dual mode);
 * - uniform/block reads, outputs: constants (dx=dy=0);
 * - fragment varyings: env.varyingReadDual (`.ddx[c]`/`.ddy[c]` reads,
 *   no guards — the raster supplies them whenever usesDerivatives; flat
 *   varyings read 0,0);
 * - gl_FragCoord: x → (1,0), y → (0,1), z/w → (0,0); gl_PointCoord and
 *   other float builtins → (0,0).
 */
function leafDual(p: P, env: CodegenEnv, c: number): [string, string] | null {
  if (!env.dual) return null;
  const cc = p.swz ? p.swz[c] : c;
  if (p.local) {
    const lv = p.local;
    const idx = p.flatOff + cc;
    if (lv.kind === 'flat') {
      const dx = lv.dxNames?.[idx];
      if (!dx) return null;
      // BUG 1: synthesized paths (member/index of a call result) carry
      // explicit dyNames (their compNames are allocTemp names — the derived
      // `compNames[idx]_dy` would never be declared); ordinary locals fall
      // back to the registered `_dy` convention.
      return [dx, lv.dyNames?.[idx] ?? `${lv.compNames![idx]}_dy`];
    }
    if (lv.int) return null;
    const n = flatComponents(lv.type);
    const dyn = p.dyn ? ` + (${p.dyn.temp}) * ${p.dyn.elemSlots}` : '';
    return [
      `ctx.scratch[${lv.scratchBase} + ${n} + ${p.flatOff}${dyn} + ${cc}]`,
      `ctx.scratch[${lv.scratchBase} + ${2 * n} + ${p.flatOff}${dyn} + ${cc}]`,
    ];
  }
  if (p.storage) {
    const st = p.storage;
    const flatC = p.flatOff + cc;
    switch (st.kind) {
      case 'uniform':
      case 'block':
        return ['0', '0'];
      case 'varying': {
        if (env.stage !== 'FRAGMENT') return null; // vertex: no duals
        const vl = env.lookupVarying(st.key);
        if (!vl) throw new Error(`codegen: missing varying layout for '${st.key}'`);
        const comp = p.dyn ? `(${p.dyn.temp}) * ${p.dyn.stride} + ${flatC}` : String(flatC);
        return env.varyingReadDual(vl.index, comp, vl.flat);
      }
      case 'attrib':
        return null;
      case 'output':
        return ['0', '0'];
    }
  }
  if (p.builtin) {
    const i = p.flatOff + cc;
    switch (p.builtin) {
      case 'gl_FragCoord':
        // Screen-space derivatives of the fragment coordinate.
        return i === 0 ? ['1', '0'] : i === 1 ? ['0', '1'] : ['0', '0'];
      case 'gl_PointCoord':
      case 'gl_FragColor':
      case 'gl_FragDepth':
      case 'gl_FragDepthEXT':
        return ['0', '0'];
      case 'gl_DepthRange':
        // Builtin uniform state — no screen-space derivative (constant duals).
        return ['0', '0'];
      default:
        // gl_Position/gl_PointSize (vertex) — never dual; gl_FrontFacing is
        // bool (caller-gated). Constant duals are a safe fallback.
        return ['0', '0'];
    }
  }
  return null;
}

function leafWrite(p: P, env: CodegenEnv, c: number): string {
  const cc = p.swz ? p.swz[c] : c;
  if (p.local) {
    const lv = p.local;
    if (lv.kind === 'flat') return lv.compNames![p.flatOff + cc];
    // BUG 2: per-MEMBER store selection (see leafRead) — the write mirror.
    const store = lv.int && isIntegralFamily(p.type) ? 'ctx.intScratch' : 'ctx.scratch';
    const dyn = p.dyn ? ` + (${p.dyn.temp}) * ${p.dyn.elemSlots}` : '';
    return `${store}[${lv.scratchBase} + ${p.flatOff}${dyn} + ${cc}]`;
  }
  if (p.storage) {
    const st = p.storage;
    const flatC = p.flatOff + cc;
    switch (st.kind) {
      case 'uniform':
        throw new Error('codegen: uniforms are read-only');
      case 'block':
        throw new Error('codegen: block members are read-only');
      case 'varying':
        if (env.stage !== 'VERTEX') throw new Error('codegen: fragment varyings are read-only');
        return varyingPathRead(env, st.key, p.type, p.dyn, flatC);
      case 'attrib':
        throw new Error('codegen: attributes are read-only');
      case 'output':
        return outputAccess(p.type, st.location, st.index, p.dyn, flatC);
    }
  }
  if (p.builtin) {
    const i = p.flatOff + cc;
    switch (p.builtin) {
      case 'gl_Position':
        return `ctx.out.position[${i}]`;
      case 'gl_PointSize':
        return 'ctx.out.pointSize';
      case 'gl_FragDepth':
      case 'gl_FragDepthEXT':
        return 'ctx.out.fragDepth';
      case 'gl_FragColor': {
        const loc = env.layout.outputLocations.get('gl_FragColor') ?? 0;
        return `ctx.out.color[${loc}][${i}]`;
      }
      case 'gl_DepthRange':
        // lvalue:false in the builtin table (semantics rejects writes); the
        // explicit case is defensive — never reachable.
        throw new Error('codegen: gl_DepthRange is read-only');
      default:
        throw new Error(`codegen: '${p.builtin}' is read-only`);
    }
  }
  throw new Error('codegen: empty path');
}

/**
 * Dual-mode WRITE slots for one flat component: [dx, dy] lvalues (null =
 * no dual planes — int/bool components, outputs, gl_FragDepth). Mirror of
 * leafWrite (same swizzle/offset/dyn resolution). The assignment emitters
 * pass the pair to env.dualWrite (v-plane target + this pair + RHS Value).
 */
function leafDualWrite(p: P, env: CodegenEnv, c: number): [string, string] | null {
  if (!env.dual) return null;
  const cc = p.swz ? p.swz[c] : c;
  if (p.local) {
    const lv = p.local;
    const idx = p.flatOff + cc;
    if (lv.kind === 'flat') {
      const dx = lv.dxNames?.[idx];
      if (!dx) return null;
      // BUG 1: synthesized paths carry explicit dyNames (see leafDual).
      return [dx, lv.dyNames?.[idx] ?? `${lv.compNames![idx]}_dy`];
    }
    if (lv.int) return null;
    const n = flatComponents(lv.type);
    const dyn = p.dyn ? ` + (${p.dyn.temp}) * ${p.dyn.elemSlots}` : '';
    return [
      `ctx.scratch[${lv.scratchBase} + ${n} + ${p.flatOff}${dyn} + ${cc}]`,
      `ctx.scratch[${lv.scratchBase} + ${2 * n} + ${p.flatOff}${dyn} + ${cc}]`,
    ];
  }
  if (p.storage) {
    switch (p.storage.kind) {
      case 'uniform':
      case 'block':
      case 'attrib':
        throw new Error('codegen: read-only storage');
      case 'varying':
        if (env.stage !== 'VERTEX') throw new Error('codegen: fragment varyings are read-only');
        return null; // vertex stage never runs dual mode
      case 'output':
        return null; // outputs have no dual planes (v-only write)
    }
  }
  if (p.builtin) {
    switch (p.builtin) {
      case 'gl_Position':
      case 'gl_PointSize':
      case 'gl_FragDepth':
      case 'gl_FragDepthEXT':
      case 'gl_FragColor':
        return null; // no dual planes
      default:
        throw new Error(`codegen: '${p.builtin}' is read-only`);
    }
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Flat component enumeration (struct/array recursion)                 */
/* ------------------------------------------------------------------ */

/** All flat component VALUES for p (recursing struct members / array
 *  elements). Dual mode: float leaves carry (v, dx, dy) — locals read the
 *  `_dx`/`_dy` names or the scratch planes, storage reads constant or
 *  varying-derived duals (see leafDual). Non-dual: {v} only, byte-identical
 *  to the pre-dual strings. */
function reads(p: P, env: CodegenEnv): Value[] {
  const t = p.type;
  if (t.kind === 'struct') {
    const out: Value[] = [];
    let off = 0;
    for (const m of t.members) {
      out.push(...reads(subP(p, m.name, m.type, off, env), env));
      off += flatComponents(m.type);
    }
    return out;
  }
  if (t.kind === 'array') {
    const out: Value[] = [];
    const n = t.size ?? 0;
    for (let k = 0; k < n; k++) {
      out.push(...reads(subPIdx(p, k, t.element, env), env));
    }
    return out;
  }
  const out: Value[] = [];
  const n = flatComponents(t);
  const dual = env.dual && isFloatLeaf(t);
  const hasPre = p.pre.length > 0;
  for (let c = 0; c < n; c++) {
    const s = leafRead(p, env, c);
    if (dual) {
      // Dual mode: pre lines (dyn-index temps, spill copy-in) ATTACH to the
      // Value instead of folding into the v string — the dx/dy plane strings
      // reference the same temps, so folding into v only would strand them
      // when a consumer reads just one plane (dFdx(x) consumes only x.dx).
      const d = leafDual(p, env, c);
      const base = hasPre ? { v: s, pre: p.pre } : { v: s };
      out.push(d ? { ...base, dx: d[0], dy: d[1] } : base);
    } else {
      const v = hasPre ? `(${p.pre.join(', ')}, ${s})` : s;
      out.push({ v });
    }
  }
  return out;
}

/** All flat write targets for p (lvalue; recursing struct/array). */
function writes(p: P, env: CodegenEnv): string[] {
  const t = p.type;
  if (t.kind === 'struct') {
    const out: string[] = [];
    let off = 0;
    for (const m of t.members) {
      out.push(...writes(subP(p, m.name, m.type, off, env), env));
      off += flatComponents(m.type);
    }
    return out;
  }
  if (t.kind === 'array') {
    const out: string[] = [];
    const n = t.size ?? 0;
    for (let k = 0; k < n; k++) out.push(...writes(subPIdx(p, k, t.element, env), env));
    return out;
  }
  const out: string[] = [];
  const n = flatComponents(t);
  for (let c = 0; c < n; c++) out.push(leafWrite(p, env, c));
  return out;
}

/** All dual write slots for p (parallel to writes(); null per component =
 *  no dual planes). Only meaningful in dual mode. */
function dualWrites(p: P, env: CodegenEnv): ([string, string] | null)[] {
  const t = p.type;
  if (t.kind === 'struct') {
    const out: ([string, string] | null)[] = [];
    let off = 0;
    for (const m of t.members) {
      out.push(...dualWrites(subP(p, m.name, m.type, off, env), env));
      off += flatComponents(m.type);
    }
    return out;
  }
  if (t.kind === 'array') {
    const out: ([string, string] | null)[] = [];
    const n = t.size ?? 0;
    for (let k = 0; k < n; k++) out.push(...dualWrites(subPIdx(p, k, t.element, env), env));
    return out;
  }
  const out: ([string, string] | null)[] = [];
  const n = flatComponents(t);
  for (let c = 0; c < n; c++) out.push(leafDualWrite(p, env, c));
  return out;
}

/** Descend into struct member `name` (flat offset `off` within the struct). */
function subP(p: P, name: string, mtype: GLSLType, off: number, env: CodegenEnv): P {
  const q = { ...p, type: mtype, swz: null, dyn: p.dyn, pre: p.pre, post: p.post };
  if (q.local) q.flatOff += off;
  else if (q.storage && (q.storage.kind === 'uniform' || q.storage.kind === 'block' || q.storage.kind === 'varying')) {
    q.storage = { ...q.storage, key: q.storage.key + '.' + name };
    q.flatOff = 0; // leaf entries are keyed per member (offset relative to the member)
  } else if (q.builtin === 'gl_DepthRange') {
    // Builtin struct uniform gl_DepthRangeParameters { near; far; diff; }
    // (GLSL ES 1.00 §7.6 / 3.00 §7.7, BOTH stages, read-only): the members
    // are flat offsets 0/1/2 of the builtin — leafRead/leafDual lower them
    // to `ctx.depthRange[0/1/2]` ([near, far, far−near]); no storage key.
    q.flatOff += off;
  } else {
    throw new Error(`codegen: struct member on ${q.storage ? q.storage.kind : 'builtin'} path`);
  }
  return q;
}

/** Descend into const array element `k`. */
function subPIdx(p: P, k: number, etype: GLSLType, env: CodegenEnv): P {
  const q = { ...p, type: etype, swz: null };
  if (q.local) {
    q.flatOff += k * flatComponents(etype);
  } else if (q.storage) {
    switch (q.storage.kind) {
      case 'uniform':
        q.storage = { ...q.storage, key: q.storage.key + `[${k}]` };
        break;
      case 'block': {
        // A const index on the INSTANCE identifier itself (key === baseKey —
        // an ARRAYED block) selects element k, which has its OWN unique block
        // index (per-element stores). Any other const index descends a MEMBER
        // array → same block, key only.
        const st = q.storage;
        q.storage =
          st.key === st.baseKey ? { ...st, key: st.key + `[${k}]`, blockIndex: st.blockIndex + k } : { ...st, key: st.key + `[${k}]` };
        break;
      }
      case 'varying': {
        const vl = env.lookupVarying(q.storage.key);
        q.flatOff += k * (vl ? vl.elemComponents : flatComponents(etype));
        break;
      }
      case 'attrib':
        q.storage = { ...q.storage, location: q.storage.location + k * attribElemStride(etype) };
        break;
      case 'output':
        q.storage = { ...q.storage, location: q.storage.location + k };
        break;
    }
  } else if (q.builtin) {
    // Whole-array builtin values (gl_FragData) — identifier-level only; error otherwise.
    throw new Error(`codegen: cannot index builtin '${q.builtin}' as a whole-array value`);
  }
  return q;
}

function attribElemStride(et: GLSLType): number {
  return et.kind === 'matrix' ? et.cols : 1;
}

/* ------------------------------------------------------------------ */
/* The walker: identifier + member + index chains                      */
/* ------------------------------------------------------------------ */

function walk(e: Expr, env: CodegenEnv): P {
  switch (e.kind) {
    case 'identifier': {
      const t = e.resolvedType!;
      const lv = env.resolveLocal(e.name);
      if (lv) {
        const p = freshP(t);
        p.local = lv;
        p.lvalue = true;
        return p;
      }
      const info = env.globalInfo(e.name);
      if (!info) throw new Error(`codegen: unknown identifier '${e.name}'`);
      const p = freshP(t);
      switch (info.kind) {
        case 'uniform':
          p.storage = { kind: 'uniform', key: info.key };
          return p;
        case 'block':
          p.storage = {
            kind: 'block',
            blockIndex: info.blockIndex,
            key: info.key,
            // '' for instance-less blocks (identifier IS the member name),
            // else the instance name — a dynamic index whose key EQUALS
            // baseKey is an INSTANCE-element index (arrayed block → stride =
            // blockStride); any other dynamic index descends a MEMBER array
            // (stride = the member's arrayStride).
            baseKey: info.baseKey,
          };
          return p;
        case 'attrib':
          // BUG 5: carry the DECLARED component count — swizzles retype
          // p.type, but the per-vertex fetch stride must stay the declared
          // width (a.xy on a vec4 fetches with stride 4).
          p.storage = { kind: 'attrib', location: info.location, declComps: attribDeclComps(t) };
          return p;
        case 'varying':
          p.storage = { kind: 'varying', key: info.key };
          p.lvalue = env.stage === 'VERTEX';
          return p;
        case 'output':
          p.storage = { kind: 'output', location: info.location, index: info.index };
          p.lvalue = true;
          return p;
        case 'builtin': {
          if (e.name === 'gl_FragData') {
            p.storage = { kind: 'output', location: env.layout.outputLocations.get('gl_FragData') ?? 0, index: 0 };
            p.lvalue = true;
            return p;
          }
          p.builtin = e.name;
          p.lvalue = info.builtin.writable;
          return p;
        }
        case 'const':
          throw new Error(`codegen: const identifier '${e.name}' must be folded (constValue)`);
      }
      break;
    }
    case 'member': {
      // BUG 1: the object may be a call/binary/... result — materialize it
      // into temps instead of crashing ("not a path expression").
      const p = walkObject(e.object, env);
      const ot = e.object.resolvedType!;
      if (ot.kind === 'struct') {
        const m = ot.members.find((x) => x.name === e.name);
        if (!m) throw new Error(`codegen: struct '${ot.name}' has no member '${e.name}'`);
        return subP(p, e.name, m.type, structMemberOffset(ot, [e.name]), env);
      }
      if (ot.kind === 'vector') {
        const sel = swizzleComponents(ot.size, e.name);
        const dup = new Set(sel).size !== sel.length;
        p.type = { kind: 'vector', base: ot.base, size: sel.length as 2 | 3 | 4 };
        p.swz = p.swz ? sel.map((i) => p.swz![i]) : sel;
        if (dup) p.lvalue = false; // v.xx = ... is illegal
        return p;
      }
      throw new Error(`codegen: member access on '${typeName(ot)}'`);
    }
    case 'index': {
      // BUG 1: the object may be a call/binary/... result — materialize it
      // into temps instead of crashing (see walkObject).
      const p = walkObject(e.object, env);
      const ot = p.type;
      const cv = e.index.constValue;
      const isConst = typeof cv === 'number' && Number.isInteger(cv);
      const idxV = isConst ? null : emitExpr(e.index, env)[0];
      if (ot.kind === 'array') {
        const et = ot.element!;
        p.type = et;
        p.swz = null;
        if (isConst) return subPIdx(p, cv, et, env);
        // dynamic element index
        const t = env.allocTemp();
        p.pre.push(`${t} = ${idxV!.pre && idxV!.pre.length ? foldPre(idxV!.pre, idxV!.v) : idxV!.v}`);
        if (p.local) {
          // BUG 1: a synthesized flat local (call-result object) is not
          // addressable by index — spill its temps into a scratch block so the
          // dyn term can stride (flat-local reads ignore p.dyn).
          if (p.local.synth) spillSynthLocal(p, env, flatComponents(et));
          p.dyn = { temp: t, stride: 0, elemSlots: flatComponents(et) };
        } else if (p.storage) {
          switch (p.storage.kind) {
            case 'uniform': {
              const key = p.storage.key + '[0]';
              const us = env.lookupUniformSlot(key);
              if (!us) {
                throw new Error(
                  `codegen: missing uniformSlots prefix '${key}' for dynamic index (linker must emit '[0]' prefixes)`,
                );
              }
              p.storage = { ...p.storage, key };
              p.dyn = { temp: t, stride: us.stride, elemSlots: 0 };
              break;
            }
            case 'block': {
              const key = p.storage.key + '[0]';
              const entry = env.lookupBlockMember(p.storage.blockIndex, key);
              if (!entry) {
                throw new Error(
                  `codegen: missing block layout prefix '${key}' for dynamic index (linker must emit '[0]' prefixes)`,
                );
              }
              // Instance-element index (pre-update key still equals the
              // instance name) selects among the per-element BLOCK STORES —
              // each element has its OWN unique block index (linker), so the
              // store array index strides by 1 and member offsets are
              // element-local (blockElements). A member-array index (key ≠
              // baseKey) strides by the member's arrayStride within one store
              // (blockStride is absent on non-arrayed-instance member entries).
              const atInstance = p.storage.key === p.storage.baseKey;
              p.storage = { ...p.storage, key };
              const stride = atInstance ? entry.blockStride ?? 0 : entry.arrayStride;
              if (stride <= 0) {
                throw new Error(
                  `codegen: block path '${key}' has no stride for dynamic indexing (linker must set arrayStride/blockStride)`,
                );
              }
              p.dyn = atInstance
                ? { temp: t, stride, blockElements: true, elemSlots: 0 }
                : { temp: t, stride, elemSlots: 0 };
              break;
            }
            case 'varying': {
              const vl = env.lookupVarying(p.storage.key);
              if (!vl) throw new Error(`codegen: missing varying layout for '${p.storage.key}'`);
              p.dyn = { temp: t, stride: vl.elemComponents, elemSlots: 0 };
              break;
            }
            case 'attrib':
              p.dyn = { temp: t, stride: attribElemStride(et), elemSlots: 0 };
              break;
            case 'output':
              p.dyn = { temp: t, stride: 1, elemSlots: 0 };
              break;
          }
        }
        return p;
      }
      if (ot.kind === 'vector') {
        p.type = { kind: 'scalar', base: ot.base };
        if (isConst) {
          // The swizzle applies to the INDEX, not to the flat offset:
          // `v.zyx[1]` addresses base component swz[1] (v.y), not flat
          // component 1 (v.x). Fold the remap into flatOff (constant), then
          // consume the swizzle.
          p.flatOff += p.swz ? p.swz[cv] : cv;
          p.swz = null;
          return p;
        }
        // dynamic component (ES 3.00 allows it): spill locals / stride-1 storage
        const t = env.allocTemp();
        p.pre.push(`${t} = ${idxV!.pre && idxV!.pre.length ? foldPre(idxV!.pre, idxV!.v) : idxV!.v}`);
        // A dynamic index into a SWIZZLED vector addresses base component
        // swz[t] (`v.zyx[i]` ↔ v[swz[i]]) — remap the runtime index through
        // the compile-time swizzle permutation, then consume the swizzle so
        // leafRead/leafWrite don't ALSO apply it as a constant offset.
        let idx = t;
        if (p.swz) {
          const j = env.allocTemp();
          p.pre.push(`${j} = ${swzLookup(p.swz, t)}`);
          idx = j;
          p.swz = null;
        }
        if (p.local) {
          if (p.local.synth) {
            // BUG 1: call-result object — scratch spill (see the array branch).
            spillSynthLocal(p, env, 1);
            p.dyn = { temp: idx, stride: 0, elemSlots: 1 };
          } else if (p.local.kind === 'scratch') {
            // Scratch-backed local (array / struct-with-array): already
            // index-addressable — no spill needed. An existing dyn (outer
            // element index, e.g. `V[func()][i]` on vec4[2]) folds into a
            // combined offset temp; otherwise the index temp strides 1.
            if (p.dyn) {
              const j = env.allocTemp();
              p.pre.push(`${j} = (${p.dyn.temp}) * ${p.dyn.elemSlots} + ${idx}`);
              p.dyn = { temp: j, stride: 0, elemSlots: 1 };
            } else {
              p.dyn = { temp: idx, stride: 0, elemSlots: 1 };
            }
          } else {
            const ds = env.ensureDynScratch(p.local.name);
            p.pre.unshift(...ds.copyIn);
            p.post.push(...ds.copyOut);
            p.local = { ...p.local, kind: 'scratch', scratchBase: ds.base, elemSlots: 1, int: ds.int };
            p.dyn = { temp: idx, stride: 0, elemSlots: 1 };
          }
        } else if (p.storage) {
          p.dyn = { temp: idx, stride: storageElemStride(p, 1), elemSlots: 0 };
        }
        return p;
      }
      if (ot.kind === 'matrix') {
        const rows = ot.rows;
        // ROW-MAJOR BLOCK MEMBER: a matrix COLUMN is not contiguous in memory —
        // component r of column c lives at float index c + r*4 (rows are
        // 16-byte strided). Materialize the column into a synthesized flat
        // local so const/dynamic component indexing and swizzles read the
        // strided bytes (the generic path folds const columns as contiguous
        // flatOff, which is only valid for column-major storage).
        if (p.storage && p.storage.kind === 'block') {
          const entry = env.lookupBlockMember(p.storage.blockIndex, p.storage.key);
          if (entry !== null && entry.rowMajor) {
            // Arrayed-block dynamic instance index (blockElements): the store
            // array index strides by 1, offsets element-local.
            const store = p.dyn && p.dyn.blockElements ? `${p.storage.blockIndex} + (${p.dyn.temp}) * 1` : String(p.storage.blockIndex);
            const base = `${entry.offset} / 4${p.dyn ? (p.dyn.blockElements ? '' : ` + (${p.dyn.temp}) * ${p.dyn.stride / 4}`) : ''}`;
            let colTerm: string;
            if (isConst) {
              colTerm = String(cv);
            } else {
              const t = env.allocTemp();
              p.pre.push(`${t} = ${idxV!.pre && idxV!.pre.length ? foldPre(idxV!.pre, idxV!.v) : idxV!.v}`);
              colTerm = t;
            }
            const compNames: string[] = [];
            const dxNames: (string | null)[] = [];
            const dyNames: (string | null)[] = [];
            for (let r = 0; r < rows; r++) {
              compNames.push(`ctx.blockStores[${store}][${base} + ${colTerm} + ${r} * 4]`);
              dxNames.push('0');
              dyNames.push('0');
            }
            const q = freshP({ kind: 'vector', base: 'float', size: rows as 2 | 3 | 4 });
            q.local = { name: '_rmcol', type: q.type, kind: 'flat', compNames, dxNames, dyNames, synth: true };
            q.pre = p.pre;
            q.lvalue = false;
            return q;
          }
        }
        p.type = { kind: 'vector', base: 'float', size: rows };
        p.swz = null;
        if (isConst) {
          if (p.local) p.flatOff += cv * rows;
          else if (p.storage) {
            switch (p.storage.kind) {
              case 'uniform':
                p.flatOff += cv * 4; // GLSL memory order: column stride 4 floats
                break;
              case 'block': {
                const entry = env.lookupBlockMember(p.storage.blockIndex, p.storage.key);
                if (!entry) throw new Error(`codegen: missing block layout for '${p.storage.key}'`);
                p.flatOff += (cv * entry.matrixStride) / 4;
                break;
              }
              case 'varying':
                p.flatOff += cv * rows;
                break;
              case 'attrib':
                p.storage = { ...p.storage, location: p.storage.location + cv };
                break;
              case 'output':
                throw new Error('codegen: cannot index a matrix output');
            }
          }
          return p;
        }
        // dynamic column (spill locals / storage stride)
        const t = env.allocTemp();
        p.pre.push(`${t} = ${idxV!.pre && idxV!.pre.length ? foldPre(idxV!.pre, idxV!.v) : idxV!.v}`);
        if (p.local) {
          if (p.local.synth) {
            // BUG 1: call-result object — scratch spill (see the array branch).
            spillSynthLocal(p, env, rows);
            p.dyn = { temp: t, stride: 0, elemSlots: rows };
          } else if (p.local.kind === 'scratch') {
            // Scratch-backed local: already index-addressable — no spill.
            // An existing dyn (outer element index, e.g. `M[func()][i]` on
            // mat4[2]) folds into a combined offset temp; otherwise the index
            // temp strides `rows` (column-major layout).
            if (p.dyn) {
              const j = env.allocTemp();
              p.pre.push(`${j} = (${p.dyn.temp}) * ${p.dyn.elemSlots} + ${t}`);
              p.dyn = { temp: j, stride: 0, elemSlots: rows };
            } else {
              p.dyn = { temp: t, stride: 0, elemSlots: rows };
            }
          } else {
            const ds = env.ensureDynScratch(p.local.name);
            p.pre.unshift(...ds.copyIn);
            p.post.push(...ds.copyOut);
            p.local = { ...p.local, kind: 'scratch', scratchBase: ds.base, elemSlots: rows, int: ds.int };
            p.dyn = { temp: t, stride: 0, elemSlots: rows };
          }
        } else if (p.storage) {
          // Block matrices: the column stride is the std140 matrixStride
          // (16 bytes — 4*rows only matches square matrices).
          const stride =
            p.storage.kind === 'block'
              ? (env.lookupBlockMember(p.storage.blockIndex, p.storage.key)?.matrixStride ?? 4 * rows)
              : storageElemStride(p, rows);
          p.dyn = { temp: t, stride, elemSlots: 0 };
        }
        return p;
      }
      throw new Error(`codegen: cannot index a '${typeName(ot)}'`);
    }
    default:
      throw new Error('codegen: not a path expression');
  }
  throw new Error('codegen: unreachable');
}

/** True when `e` is a plain lvalue path (identifier/member/index chain). */
function isPathExpr(e: Expr): boolean {
  return e.kind === 'identifier' || e.kind === 'member' || e.kind === 'index';
}

/**
 * BUG 1: resolve the OBJECT of a member/index access. Plain path expressions
 * walk normally; ANY other expression kind (call — builtin/user/ctor — binary,
 * unary, ternary, assign, comma) is not addressable, so its VALUE is
 * materialized into temps and a synthesized temp-backed flat-local path is
 * built. The member/index descent then reads the temps (struct member offsets,
 * const/dynamic indices and swizzles all work on the flat compNames). The
 * assignments are carried on p.pre UNCONDITIONALLY — a call result with no pre
 * (an IIFE string) must not become the "path" itself, or reads() would emit it
 * as a bare expression statement and discard the value. p.lvalue stays false →
 * emitLValue throws "not an lvalue" for write attempts.
 *
 * Materialization goes through materializeSharedPre: SHARED pres (an
 * array-returning call carries ONE [iife] on every component) run ONCE — the
 * naive fold-per-component temping would re-run the callee per component. The
 * resulting temps ARE the flat compNames (the materialization buffer becomes
 * p.pre).
 *
 * DUAL MODE: float components materialize as (v, dx, dy) temp triples with the
 * plane names recorded in dxNames/dyNames (allocTemp names violate the
 * `compNames[c]+'_dy'` convention — leafDual/leafDualWrite consult dyNames).
 * Int/bool components get no duals.
 */
function walkObject(e: Expr, env: CodegenEnv): P {
  if (isPathExpr(e)) return walk(e, env);
  const t = e.resolvedType!;
  const vals = materializeSharedPre(emitExpr(e, env), env);
  const n = flatComponents(t);
  const floatness = flatFloatness(t);
  const compNames: string[] = [];
  const dxNames: (string | null)[] = [];
  const dyNames: (string | null)[] = [];
  const pre: string[] = [];
  const seen = new Set<string[]>();
  for (let c = 0; c < n; c++) {
    const v = vals[c];
    if (v.pre && v.pre.length > 0 && !seen.has(v.pre)) {
      seen.add(v.pre);
      pre.push(...v.pre);
    }
    compNames.push(v.v);
    if (env.dual && floatness[c]) {
      dxNames.push(v.dx ?? '0');
      dyNames.push(v.dy ?? '0');
    } else {
      dxNames.push(null);
      dyNames.push(null);
    }
  }
  const p = freshP(t);
  p.local = {
    name: '_tmp',
    type: t,
    kind: 'flat',
    compNames,
    dxNames,
    dyNames,
    synth: true,
    members: t.kind === 'struct' ? structMemberOffsets(t) : undefined,
  };
  p.pre = pre;
  return p;
}

/**
 * BUG 1: spill a synthesized temp-backed flat local into a scratch block so a
 * DYNAMIC index can stride (flat-local reads ignore p.dyn — temp names are not
 * addressable by index). The copy-in assignments (v + dual planes, offset by
 * p.flatOff for struct-member descent) join p.pre, so they run before every
 * read. `elemSlots` = flat components per indexed element. The store follows
 * the BUG 2 rule: all-integral objects → intScratch, otherwise float scratch
 * (leafRead/leafWrite then select per member type).
 */
function spillSynthLocal(p: P, env: CodegenEnv, elemSlots: number): void {
  const lv = p.local!;
  const n = flatComponents(lv.type);
  const int = isIntegralFamily(lv.type) && !hasFloatLeaves(lv.type);
  const base = int ? env.allocIntScratch(n) : env.allocScratch(n);
  const store = int ? 'ctx.intScratch' : 'ctx.scratch';
  for (let k = 0; k < n; k++) {
    p.pre.push(`${store}[${base} + ${p.flatOff} + ${k}] = ${lv.compNames![k]}`);
    if (env.dual && lv.dxNames?.[k]) {
      p.pre.push(`${store}[${base} + ${n} + ${p.flatOff} + ${k}] = ${lv.dxNames[k]}`);
      p.pre.push(`${store}[${base} + ${2 * n} + ${p.flatOff} + ${k}] = ${lv.dyNames![k]}`);
    }
  }
  p.local = { ...lv, kind: 'scratch', scratchBase: base, blockSize: n, elemSlots, int };
}

/** Storage stride for a dynamic VECTOR-component / MATRIX-column index. */
function storageElemStride(p: P, comps: number): number {
  switch (p.storage!.kind) {
    case 'uniform':
      return comps === 1 ? 1 : 4; // vec component: 1 float; mat column: 4 floats
    case 'block': {
      // 4 bytes per float component
      return 4 * comps;
    }
    case 'varying':
      return comps;
    case 'attrib':
      return 1; // mat columns occupy consecutive locations
    case 'output':
      return 1;
  }
}

/** Swizzle name → component indices (position within its letter set). */
export function swizzleComponents(size: number, name: string): number[] {
  const sets = ['xyzw', 'rgba', 'stpq'];
  const out: number[] = [];
  for (const ch of name) {
    const set = sets.find((s) => s.includes(ch));
    if (!set) throw new Error(`codegen: bad swizzle component '${ch}'`);
    const i = set.indexOf(ch);
    if (i >= size) throw new Error(`codegen: swizzle '${name}' out of range for size ${size}`);
    out.push(i);
  }
  return out;
}

/** Compile-time swizzle permutation applied to a RUNTIME index: `v.zyx[i]`
 *  reads base component swz[i] (vectors have ≤ 4 components, so a nested
 *  ternary chain is compact and allocation-free). */
function swzLookup(swz: number[], t: string): string {
  let s = String(swz[swz.length - 1]);
  for (let i = swz.length - 2; i >= 0; i--) s = `(${t} === ${i} ? ${swz[i]} : ${s})`;
  return s;
}

/* ------------------------------------------------------------------ */
/* emitExpr / emitLValue                                               */
/* ------------------------------------------------------------------ */

/** The lvalue of an assignment target: one JS lvalue string per flat component. */
export interface LValue {
  type: GLSLType;
  targets: string[];
  /** Dual-mode write slots per component: [dx, dy] lvalues (null = no dual
   *  planes — int/bool components, outputs). Absent in non-dual mode. */
  dualTargets?: ([string, string] | null)[];
  /** Per-component bit-pack kind when the target cell is a PACKED int/uint
   *  varying (VERTEX stage — see packVaryingWrite): the cell stores the
   *  value's BIT PATTERN as float32, so writes must pack (`R.u2f`) and the
   *  cell read back unpack (`R.f2i` for 'int', `R.f2u` for 'uint'). Absent
   *  when no component is packed (false entries mark non-packed components). */
  bits?: ('uint' | 'int' | false)[];
  /** Statements that must run BEFORE the writes (dynamic-index temps, spill copy-in). */
  prelude?: string;
  /** Statements that must run AFTER the writes (spill copy-out). */
  copyBack?: string;
}

/** Per-flat-component packed-ness of an lvalue path (parallel to writes():
 *  struct/array recursion; a leaf is packed when it is a VERTEX-stage
 *  int/uint varying — the only shapes that store bit patterns in the
 *  record). The kind distinguishes the int32 vs uint32 unpack. */
function writeBitKinds(p: P, env: CodegenEnv): ('uint' | 'int' | false)[] {
  const t = p.type;
  if (t.kind === 'struct') {
    const out: ('uint' | 'int' | false)[] = [];
    let off = 0;
    for (const m of t.members) {
      out.push(...writeBitKinds(subP(p, m.name, m.type, off, env), env));
      off += flatComponents(m.type);
    }
    return out;
  }
  if (t.kind === 'array') {
    const out: ('uint' | 'int' | false)[] = [];
    const n = t.size ?? 0;
    for (let k = 0; k < n; k++) out.push(...writeBitKinds(subPIdx(p, k, t.element, env), env));
    return out;
  }
  const packed: 'uint' | 'int' | false =
    env.stage === 'VERTEX' && p.storage?.kind === 'varying'
      ? isUintType(t)
        ? 'uint'
        : isIntType(t)
          ? 'int'
          : false
      : false;
  const n = flatComponents(t);
  const out: ('uint' | 'int' | false)[] = new Array(n);
  out.fill(packed);
  return out;
}

/** Emit one JS expression per flat component (column-major; scalar → 1). */
export function emitExpr(e: Expr, env: CodegenEnv): Value[] {
  const t = e.resolvedType;
  if (!t) throw new Error('codegen: expression lacks resolvedType (semantics must run first)');
  // Scalar constants fold (const globals, folded subexpressions, literals).
  // Dual mode: float constants are constant duals (dx=dy=0).
  if (e.constValue !== undefined && t.kind === 'scalar') {
    const v = env.emitConstNumber(e.constValue, t);
    return env.dual && t.base === 'float' ? [{ v, dx: '0', dy: '0' }] : [{ v }];
  }
  switch (e.kind) {
    case 'literal': {
      const lt: GLSLType =
        e.literalType === 'int'
          ? { kind: 'scalar', base: 'int' }
          : e.literalType === 'uint'
            ? { kind: 'scalar', base: 'uint' }
            : e.literalType === 'bool'
              ? { kind: 'scalar', base: 'bool' }
              : { kind: 'scalar', base: 'float' };
      const v = env.emitConstNumber(e.value, lt);
      return env.dual && lt.base === 'float' ? [{ v, dx: '0', dy: '0' }] : [{ v }];
    }
    case 'identifier':
    case 'member':
    case 'index': {
      const p = walk(e, env);
      return reads(p, env);
    }
    case 'unary':
      return emitUnary(e, env);
    case 'binary':
      return emitBinary(e, env);
    case 'assign':
      return emitAssign(e, env);
    case 'ternary':
      return emitTernary(e, env);
    case 'call':
      return emitCall(e, env);
    case 'comma':
      return emitComma(e, env);
    default:
      throw new Error(`codegen: unsupported expression '${(e as { kind: string }).kind}'`);
  }
}

/** Resolve an lvalue (assignment / ++ / -- target). */
export function emitLValue(e: Expr, env: CodegenEnv): LValue {
  const p = walk(e, env);
  if (!p.lvalue) throw new Error('codegen: expression is not an lvalue');
  const targets = writes(p, env);
  const bits = writeBitKinds(p, env);
  return {
    type: p.type,
    targets,
    bits: bits.some((b) => b) ? bits : undefined,
    dualTargets: env.dual ? dualWrites(p, env) : undefined,
    prelude: p.pre.length ? p.pre.join('; ') + ';' : undefined,
    copyBack: p.post.length ? p.post.join('; ') + ';' : undefined,
  };
}

/**
 * Dedupe SHARED pre arrays by array identity across components. A
 * multi-component user-call result carries ONE `[iife]` pre array on EVERY
 * component (functions.ts) — consumers that fold each component's pre inline
 * (materialize, binary/unary/comma/ctor paths) would re-run the callee once
 * per component. Each DISTINCT pre array is kept on the FIRST component
 * carrying it only; later components drop it (their v/dx/dy strings reference
 * the temps the first component's pre sets — emission order is 0..n-1, so the
 * first component's fold always runs first). Values with no pre or with
 * per-component-unique pres pass through unchanged. SAFE ONLY for consumers
 * that use each component exactly once (ctor/unary/comma operands) — the
 * caller must guarantee this.
 */
export function dedupeSharedPre(vals: Value[]): Value[] {
  const seen = new Set<string[]>();
  const out: Value[] = [];
  for (const v of vals) {
    const p = v.pre;
    if (p && p.length > 0 && seen.has(p)) {
      out.push({ ...v, pre: undefined });
      continue;
    }
    if (p && p.length > 0) seen.add(p);
    out.push(v);
  }
  return out;
}

/** Split an LValue.prelude ('; '-joined statements, trailing ';') into
 *  individual statement strings (trailing ';' stripped) — emit as Value.pre
 *  so the statement/expression emitters run them BEFORE the value. */
function preludeLines(prelude: string | undefined): string[] {
  if (!prelude) return [];
  const s = prelude.endsWith(';') ? prelude.slice(0, -1) : prelude;
  return s.split('; ');
}

/** Convert an LValue.copyBack ('; '-joined statements, trailing ';') into a
 *  comma-joined expression fragment (no trailing comma) — semicolons are
 *  invalid inside parens, so the copy-back must fold as comma terms. */
function copyBackComma(copyBack: string | undefined): string | null {
  if (!copyBack) return null;
  const s = copyBack.endsWith(';') ? copyBack.slice(0, -1) : copyBack;
  return s.split('; ').join(', ');
}

/** Materialize values with pres into temps (for multi-use contexts). Dual
 *  mode: a float value materializes as a TRIPLE of temps (v, dx, dy — the
 *  pre lines run v first, then the planes, so plane reads that reference
 *  temps from the v pre are assigned). */
export function materialize(vals: Value[], env: CodegenEnv): Value[] {
  return vals.map((v) => {
    if (!v.pre || v.pre.length === 0) return v;
    const t = env.allocTemp();
    const pre = [`${t} = ${foldPre(v.pre, v.v)}`];
    if (env.dual && v.dx !== undefined) {
      const tx = env.allocTemp();
      const ty = env.allocTemp();
      pre.push(`${tx} = ${v.dx}`, `${ty} = ${v.dy}`);
      return { v: t, dx: tx, dy: ty, pre };
    }
    return { v: t, pre };
  });
}

/**
 * Materialize EVERY value into a fresh temp, running each value's pre exactly
 * ONCE — pres SHARED by array identity across components (multi-component
 * call results carry ONE `[iife]` array on every component; array assignments
 * in dual mode share one write-composite array) are deduped, so the callee /
 * side effects never re-run per component. Unlike materialize(), no-pre
 * values are temped too — their v-strings may embed GLSL side effects
 * (assignment composites `(a0 = b0)` fold into v, not pre), and temping them
 * pins those effects at their exact position. Processing is left-to-right
 * (GLSL evaluation order): per component, emit the deduped pre lines, then
 * the temp assignment (+ dual planes in dual mode).
 *
 * All returned values carry the SAME `pre` array (the materialization
 * buffer) — consumers must dedupe by identity (emitPres) or fold it once
 * (walkObject, the struct/array compare path, .length()); folding per
 * component would re-run the buffer.
 */
export function materializeSharedPre(vals: Value[], env: CodegenEnv): Value[] {
  const pre: string[] = [];
  const seen = new Set<string[]>();
  const out: Value[] = [];
  for (const v of vals) {
    const p = v.pre;
    if (p && p.length > 0 && !seen.has(p)) {
      seen.add(p);
      pre.push(...p);
    }
    const t = env.allocTemp();
    pre.push(`${t} = ${v.v}`);
    if (env.dual && v.dx !== undefined && v.dy !== undefined) {
      const tx = env.allocTemp();
      const ty = env.allocTemp();
      pre.push(`${tx} = ${v.dx}`, `${ty} = ${v.dy}`);
      out.push({ v: t, dx: tx, dy: ty, pre });
    } else {
      out.push({ v: t, pre });
    }
  }
  return out;
}

/**
 * Materialize every operand component into a FRESH temp, appending the
 * assignments to `pre` (one shared buffer — the caller attaches it to
 * component 0 only, the established comp0-hoist convention). Pres carried by
 * the values are folded per component BEFORE its temp assignment, deduped by
 * array identity (a multi-component result shares ONE pre array, so its side
 * effects run exactly once, at the first component). All returned reads
 * reference the temps, so sequential target writes (`v = m * v`,
 * `m = m1 * m2`) never observe partially-updated operands (matrixCompoundMul's
 * RHS-materialization idiom).
 */
function materializeOperands(vals: Value[], env: CodegenEnv, pre: string[]): string[] {
  const seen = new Set<string[]>();
  return vals.map((v) => {
    const t = env.allocTemp();
    if (v.pre && v.pre.length > 0 && !seen.has(v.pre)) {
      seen.add(v.pre);
      pre.push(`${t} = ${foldPre(v.pre, v.v)}`);
    } else {
      pre.push(`${t} = ${v.v}`);
    }
    return t;
  });
}

/**
 * Does the expression subtree fold GLSL side effects (assignments, ++/--)
 * into its codegen v strings? User-function calls put their whole inline in
 * Value.pre instead (detected separately there) — but when a call is nested
 * inside arithmetic/comparison combinators, its pre folds INTO the result's
 * v, so calls must be flagged here too. Builtin/constructor calls are pure
 * (only their ARGS can carry side effects) — they cannot be distinguished
 * from user calls at the AST level, so any non-constructor identifier call is
 * conservatively flagged (temping a pure builtin is redundant but correct).
 */
function astHasSideEffects(e: Expr): boolean {
  switch (e.kind) {
    case 'assign':
      return true;
    case 'unary':
      return e.op === '++' || e.op === '--' || astHasSideEffects(e.operand);
    case 'binary':
      return astHasSideEffects(e.left) || astHasSideEffects(e.right);
    case 'ternary':
      return astHasSideEffects(e.cond) || astHasSideEffects(e.whenTrue) || astHasSideEffects(e.whenFalse);
    case 'comma':
      return e.exprs.some((x) => astHasSideEffects(x));
    case 'call':
      if (e.args.some((a) => astHasSideEffects(a))) return true;
      return e.callee.kind !== 'identifier' || !TYPE_NAMES.has(e.callee.name);
    case 'index':
      return astHasSideEffects(e.object) || astHasSideEffects(e.index);
    case 'member':
      return astHasSideEffects(e.object);
    default:
      return false; // identifier / literal — pure reads
  }
}

/**
 * Broadcast-safe scalar operand (BUG: vector-scalar-arithmetic-inside-loop).
 * Broadcasting a SCALAR to a vector/matrix duplicates its v string per
 * component — if the scalar embeds GLSL side effects (assignments / ++-- in
 * v per astHasSideEffects, or a pre: user-call IIFEs / texture samples), the
 * effect would re-run per component, mutating the target between components.
 * Such scalars are materialized ONCE into a temp; the temp assignment is
 * appended to `sharedPre` (emitted once, before the component loop). Pure
 * scalars (empty pre, no side effects) pass through unchanged — the existing
 * fast path. Dual mode: the v/dx/dy planes become pure temp reads after the
 * single hoist.
 */
function broadcastScalar(v: Value, src: Expr, env: CodegenEnv, sharedPre: string[]): Value {
  const sideEffects = (v.pre !== undefined && v.pre.length > 0) || astHasSideEffects(src);
  if (!sideEffects) return v;
  const t = env.allocTemp();
  sharedPre.push(`${t} = ${foldPre(v.pre ?? [], v.v)}`);
  if (env.dual && v.dx !== undefined) {
    const tx = env.allocTemp();
    const ty = env.allocTemp();
    sharedPre.push(`${tx} = ${v.dx}`, `${ty} = ${v.dy}`);
    return { v: t, dx: tx, dy: ty };
  }
  return { v: t };
}

/* ------------------------------------------------------------------ */
/* Unary / binary / ternary / comma / assign                           */
/* ------------------------------------------------------------------ */

function emitUnary(e: Extract<Expr, { kind: 'unary' }>, env: CodegenEnv): Value[] {
  const t = e.resolvedType!;
  const base = scalarBaseOf(t);
  const n = flatComponents(t);
  const op = e.op;
  if (op === '++' || op === '--') {
    // GLSL ES: prefix `++x` yields the NEW value, postfix `x++` the OLD.
    // Both write the lvalue ±1. Per-component on the lvalue.
    const lv = emitLValue(e.operand, env);
    const delta = op === '++' ? '1' : '-1';
    const preludes = preludeLines(lv.prelude);
    const post = copyBackComma(lv.copyBack);
    const out: Value[] = [];
    for (let c = 0; c < n; c++) {
      const target = lv.targets[c];
      if (base === null || base === 'bool') throw new Error('codegen: cannot increment a bool');
      // Packed int/uint varying cell (see packVaryingWrite): the old value is
      // the UNPACKED cell read; the write re-packs the incremented value; the
      // expression's value is the unpacked post-write cell (new value for
      // prefix, old for postfix).
      const bitKind = lv.bits !== undefined ? lv.bits[c] : false;
      const read = bitKind ? unpackVaryingCell(target, bitKind === 'int') : target;
      const write = (rhs: string): string =>
        base === 'float'
          ? `${target} = ${rhs}`
          : base === 'int'
            ? bitKind
              ? `${target} = R.u2f(((${rhs}) | 0))`
              : `${target} = ((${rhs}) | 0)`
            : bitKind
              ? `${target} = R.u2f(((${rhs}) >>> 0))`
              : `${target} = ((${rhs}) >>> 0)`;
      let s: string;
      if (bitKind) {
        // Packed cell: the write re-packs; the expression's value is the
        // unpacked post-write cell read (new value for prefix, old for postfix
        // via the snapshot temp).
        if (e.postfix) {
          const t = env.allocTemp();
          s = `(${t} = ${read}, ${write(`${t} + ${delta}`)}, ${t})`;
        } else {
          s = `(${write(`${read} + ${delta}`)}, ${read})`;
        }
      } else if (e.postfix) {
        // Postfix result = OLD value: snapshot the target BEFORE the write,
        // then write old ± 1, then yield the snapshot:
        // (t = target, target = t ± 1, t).
        const t = env.allocTemp();
        s = `(${t} = ${target}, ${write(`${t} + ${delta}`)}, ${t})`;
      } else {
        s = `(${write(`${target} + ${delta}`)})`;
      }
      // Prelude (dyn-index temps / spill copy-in) runs BEFORE the write, the
      // spill copy-back AFTER it — and the expression's VALUE must be the
      // increment result (old value for postfix, new for prefix). Semicolons
      // are invalid inside parens, so fold copyBack as comma terms via a temp:
      // (t = <inc expr>, cb, t).
      let v = s;
      if (post) {
        const t = env.allocTemp();
        v = `(${t} = ${s}, ${post}, ${t})`;
      }
      const val: Value = preludes.length > 0 ? { v, pre: preludes } : { v };
      // Dual mode, float leaf: the ±1 offset leaves the dx/dy planes
      // untouched, so the CURRENT plane values are the derivatives of BOTH
      // the old and the new value (missing planes = constant 0, as usual).
      const dual = lv.dualTargets ? lv.dualTargets[c] : undefined;
      if (env.dual && dual) {
        val.dx = dual[0];
        val.dy = dual[1];
      }
      out.push(val);
    }
    return out;
  }
  // BUG (shared-pre re-run): dedupe the operand's pre arrays by identity — a
  // multi-component call result carries ONE [iife] on every component; the
  // per-component fold below would re-run the callee once per component
  // (`vec2 v = -f();` ran f twice). Only the first component keeps the shared
  // pre (folded inline into its v / attached for dual-mode consumers); later
  // components read the call's retTemps directly.
  const vals = dedupeSharedPre(emitExpr(e.operand, env));
  return vals.map((v) => {
    const x = v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v;
    switch (op) {
      case '+':
        // Unary plus is the identity — duals pass through unchanged. In dual
        // mode the operand's pre attaches to the RESULT (folding it into v
        // would strand the temps when only dx/dy are consumed — dFdx(-v)).
        if (env.dual && v.dx !== undefined) return { v: `(${v.v})`, dx: v.dx, dy: v.dy, pre: v.pre };
        return { v: `(${x})` };
      case '-':
        if (base === 'uint') return { v: `((0 - (${x})) >>> 0)` };
        if (base === 'int') return { v: `((-(${x})) | 0)` };
        // Float: negation is linear — negate the duals too.
        if (env.dual && v.dx !== undefined) {
          return { v: `(-(${v.v}))`, dx: `(-(${v.dx}))`, dy: `(-(${v.dy}))`, pre: v.pre };
        }
        return { v: `(-(${x}))` };
      case '!':
        return { v: `(!(${x}))` };
      case '~':
        if (base === 'uint') return { v: `((~(${x})) >>> 0)` };
        return { v: `((~(${x})) | 0)` };
      default:
        throw new Error(`codegen: bad unary '${op}'`);
    }
  });
}

/** Common scalar base of two operands (float > uint > int; bool stays bool). */
function commonBase(a: string | null, b: string | null): string | null {
  if (a === null || b === null || a === 'bool' || b === 'bool') return a === b ? a : null;
  if (a === 'float' || b === 'float') return 'float';
  if (a === 'uint' || b === 'uint') return 'uint';
  return 'int';
}

/** Per-leaf comparison strings for struct ==/!= (GLSL ES 1.00 §5.9 / ES 3.00
 *  §5.9: same-typed structs compare MEMBER-WISE — semantics guarantees
 *  identical type/name; const operands fold at semantics, this is the runtime
 *  path). `av`/`bv` are the flat operand Values in member order (emitExpr
 *  expands structs/arrays/matrices in declaration order, exactly matching
 *  flatComponents). Nested structs (and arrays of structs) recurse; every
 *  other member contributes one comparison string per flat leaf. Bool leaves
 *  normalize both sides with `!!` (uniform-store 0/1 vs literal true/false —
 *  mirrors the scalar/vector path); each leaf's pre folds inline (pres are
 *  pure, each leaf is used exactly once). */
function structCompareParts(
  op: '==' | '!=',
  t: GLSLType,
  av: Value[],
  bv: Value[],
): string[] {
  const parts: string[] = [];
  if (t.kind === 'array') {
    const n = flatComponents(t.element);
    let off = 0;
    for (let k = 0; k < (t.size ?? 0); k++) {
      parts.push(...structCompareParts(op, t.element, av.slice(off, off + n), bv.slice(off, off + n)));
      off += n;
    }
    return parts;
  }
  if (t.kind === 'struct') {
    let off = 0;
    for (const m of t.members) {
      parts.push(
        ...structCompareParts(op, m.type, av.slice(off, off + flatComponents(m.type)), bv.slice(off, off + flatComponents(m.type))),
      );
      off += flatComponents(m.type);
    }
    return parts;
  }
  // Leaf: scalar/vector/matrix (base null is unreachable — fall back to plain).
  const base = scalarBaseOf(t);
  for (let c = 0; c < av.length; c++) {
    const a = av[c];
    const b = bv[c];
    const avs = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
    const bvs = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
    if (base === 'bool') {
      parts.push(op === '==' ? `(!!(${avs})) === (!!(${bvs}))` : `(!!(${avs})) !== (!!(${bvs}))`);
    } else {
      parts.push(op === '==' ? `(${avs} === (${bvs}))` : `(${avs} !== (${bvs}))`);
    }
  }
  return parts;
}

function emitBinary(e: Extract<Expr, { kind: 'binary' }>, env: CodegenEnv): Value[] {
  const t = e.resolvedType!;
  const op = e.op;
  const lt = e.left.resolvedType!;
  const rt = e.right.resolvedType!;
  const lb = scalarBaseOf(lt);
  const rb = scalarBaseOf(rt);
  switch (op) {
    case '&&':
    case '||': {
      const a = materialize(emitExpr(e.left, env), env)[0];
      const b = emitExpr(e.right, env)[0];
      const bv = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
      // `a` was materialized — its pre (the temp assignment) MUST carry on the
      // result, or the left operand reads an unassigned temp (always falsy).
      const out: Value = { v: op === '&&' ? `(${a.v} && (${bv}))` : `(${a.v} || (${bv}))` };
      if (a.pre && a.pre.length > 0) out.pre = a.pre;
      return [out];
    }
    case '^^': {
      const a = emitExpr(e.left, env)[0];
      const b = emitExpr(e.right, env)[0];
      const av = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
      const bv = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
      // BUG 3: ^^ operands are always bool — the uniform store holds numbers
      // (0/1) while literals emit true/false; strict !== would compare 1 !==
      // true (false for equal operands). Normalize both sides.
      return [{ v: `((!!(${av})) !== (!!(${bv})))` }];
    }
    case '==':
    case '!=': {
      // GLSL ES 1.00 §5.9 / ES 3.00 §5.9: ==/!= on same-typed STRUCTS and
      // ARRAYS (ES 3.00 §5.9: element-wise) is legal; semantics guarantees
      // identical type (const operands fold at semantics, this is the runtime
      // path). The operand Values come back in flat member/element order;
      // compare recursively and join: == → all members, != → any member.
      // Both operand component sets are materialized FIRST (left fully before
      // right — GLSL evaluation order) via materializeSharedPre, so shared
      // pres (an array-returning call's single [iife] on every component)
      // run exactly ONCE — the naive fold-per-leaf would re-run the callee
      // per compared component. structCompareParts then compares pure temps.
      if ((lt.kind === 'struct' || lt.kind === 'array') && typeEquals(lt, rt)) {
        const avSrc = materializeSharedPre(emitExpr(e.left, env), env);
        const bvSrc = materializeSharedPre(emitExpr(e.right, env), env);
        const pre: string[] = [];
        const seen = new Set<string[]>();
        for (const v of avSrc) {
          if (v.pre && v.pre.length > 0 && !seen.has(v.pre)) {
            seen.add(v.pre);
            pre.push(...v.pre);
          }
        }
        for (const v of bvSrc) {
          if (v.pre && v.pre.length > 0 && !seen.has(v.pre)) {
            seen.add(v.pre);
            pre.push(...v.pre);
          }
        }
        const strip = (v: Value): Value => ({ v: v.v });
        const parts = structCompareParts(op, lt, avSrc.map(strip), bvSrc.map(strip));
        const out: Value = { v: `(${parts.join(op === '==' ? ' && ' : ' || ')})` };
        if (pre.length > 0) out.pre = pre;
        return [out];
      }
      // semantics: ALWAYS resolves to scalar bool (all-components compare).
      const cb = commonBase(lb, rb);
      if (!cb) throw new Error(`codegen: cannot compare ${typeName(lt)} and ${typeName(rt)}`);
      const ct = shapeOf(lt, cb);
      const avSrc = emitExpr(e.left, env);
      const bvSrc = emitExpr(e.right, env);
      const av = convertValue(avSrc, lt, ct);
      const bv = convertValue(bvSrc, rt, ct);
      // convertValue DROPS Value.pre when it converts scalar bases — re-attach
      // (mirrors statements.ts convertPreserving) so operand pres survive.
      for (let c = 0; c < flatComponents(lt); c++) {
        const as = avSrc[c];
        const bs = bvSrc[c];
        if (av[c] !== as && as.pre && as.pre.length > 0) {
          av[c] = { ...av[c], pre: as.pre };
        }
        if (bv[c] !== bs && bs.pre && bs.pre.length > 0) {
          bv[c] = { ...bv[c], pre: bs.pre };
        }
      }
      const parts: string[] = [];
      for (let c = 0; c < flatComponents(lt); c++) {
        const ap = av[c].pre;
        const bp = bv[c].pre;
        const a = ap && ap.length ? foldPre(ap, av[c].v) : av[c].v;
        const b = bp && bp.length ? foldPre(bp, bv[c].v) : bv[c].v;
        // BUG 3: bool operands may mix uniform-store numbers (0/1) and
        // literals (true/false) — strict === on 1 vs true is false. `!!`
        // normalizes both sides (vector bools: all components).
        if (cb === 'bool') {
          parts.push(op === '==' ? `(!!(${a})) === (!!(${b}))` : `(!!(${a})) !== (!!(${b}))`);
        } else {
          parts.push(op === '==' ? `(${a} === (${b}))` : `(${a} !== (${b}))`);
        }
      }
      return [{ v: `(${parts.join(op === '==' ? ' && ' : ' || ')})` }];
    }
    case '<':
    case '>':
    case '<=':
    case '>=': {
      const cb = commonBase(lb, rb);
      if (!cb) throw new Error(`codegen: cannot compare ${typeName(lt)} and ${typeName(rt)}`);
      const ct = shapeOf(lt, cb);
      const aSrc = emitExpr(e.left, env);
      const bSrc = emitExpr(e.right, env);
      const ac = convertValue(aSrc, lt, ct);
      const bc = convertValue(bSrc, rt, ct);
      // convertValue DROPS Value.pre when it converts scalar bases — re-attach.
      if (ac[0] !== aSrc[0] && aSrc[0].pre && aSrc[0].pre.length > 0) {
        ac[0] = { ...ac[0], pre: aSrc[0].pre };
      }
      if (bc[0] !== bSrc[0] && bSrc[0].pre && bSrc[0].pre.length > 0) {
        bc[0] = { ...bc[0], pre: bSrc[0].pre };
      }
      const a = ac[0];
      const b = bc[0];
      const av = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
      const bv = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
      return [{ v: `(${av} ${op} (${bv}))` }];
    }
    case '<<':
    case '>>': {
      // result = LEFT operand's type; right is an int/uint scalar.
      const b = emitExpr(e.right, env)[0];
      const bv = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
      const shift = `((${bv}) >>> 0)`;
      return emitExpr(e.left, env).map((a) => {
        const x = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
        // Wrap INSIDE the parens — `| 0` / `>>> 0` bind looser than `+`, so a
        // bare `(X) | 0` embedded in a larger expression would mis-parse.
        if (op === '<<') {
          return { v: lb === 'uint' ? `(((${x}) << ${shift}) >>> 0)` : `(((${x}) << ${shift}) | 0)` };
        }
        return { v: lb === 'uint' ? `(((${x}) >>> ${shift}) >>> 0)` : `(((${x}) >> ${shift}) | 0)` };
      });
    }
    case '&':
    case '|':
    case '^': {
      const cb = commonBase(lb, rb);
      if (!cb) throw new Error(`codegen: incompatible bitwise operands`);
      const ct = shapeOf(lt, cb);
      const avSrc = emitExpr(e.left, env);
      const bvSrc = emitExpr(e.right, env);
      const av = convertValue(avSrc, lt, ct);
      const bv = convertValue(bvSrc, rt, ct);
      // convertValue DROPS Value.pre when it converts scalar bases — re-attach
      // (mirrors statements.ts convertPreserving) so operand pres survive.
      for (let c = 0; c < flatComponents(lt); c++) {
        const as = avSrc[c];
        const bs = bvSrc[c];
        if (av[c] !== as && as.pre && as.pre.length > 0) {
          av[c] = { ...av[c], pre: as.pre };
        }
        if (bv[c] !== bs && bs.pre && bs.pre.length > 0) {
          bv[c] = { ...bv[c], pre: bs.pre };
        }
      }
      const isU = cb === 'uint';
      return av.map((a, c) => {
        const x = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
        const y = bv[c].pre && bv[c].pre.length ? foldPre(bv[c].pre, bv[c].v) : bv[c].v;
        // Self-parenthesized wrap (see the shift case above).
        return { v: isU ? `(((${x}) ${op} (${y})) >>> 0)` : `(((${x}) ${op} (${y})) | 0)` };
      });
    }
    default:
      return emitArith(e, env, lt, rt, t);
  }
}

/** Same-shape type with base `b` (scalar stays scalar; vector/matrix keep shape). */
function shapeOf(t: GLSLType, base: string): GLSLType {
  if (t.kind === 'vector') return { kind: 'vector', base: base as never, size: t.size };
  if (t.kind === 'matrix') return t; // matrices are always float
  return { kind: 'scalar', base: base as never };
}

/**
 * Dual-mode arithmetic on ONE float component: (v, dx, dy) per the op.
 * Operands are materialized — v/dx/dy each reference the operand strings,
 * so pre-carrying values (inlined-call IIFEs, assignment composites) must
 * run exactly once via their hoisted pres; pure reads (varyings, uniforms,
 * temps) are untouched by materialize and repeat safely. Missing duals
 * (int/uint operands, literals) are the constant 0 — mixed float/int
 * arithmetic's derivatives come only from the float side.
 *   + / - : linear — the op applied to each plane.
 *   *     : product rule  d(a·b) = da·b + a·db.
 *   /     : quotient rule d(a/b) = (da·b − a·db) / b².
 *   %     : v mirrors the non-dual float '%' (JS remainder — the compiler
 *           rejects float '%' so this is the permissive path); the dual
 *           planes use the GLSL-mod view dv = da − floor(a/b)·db (floor is
 *           a.e. constant, so its derivative vanishes).
 */
function arithDual(op: string, a: Value, b: Value, env: CodegenEnv): Value {
  const am = materialize([a], env)[0];
  const bm = materialize([b], env)[0];
  const av = am.v;
  const bv = bm.v;
  const adx = am.dx ?? '0';
  const ady = am.dy ?? '0';
  const bdx = bm.dx ?? '0';
  const bdy = bm.dy ?? '0';
  const pre: string[] = [];
  if (am.pre && am.pre.length > 0) pre.push(...am.pre);
  if (bm.pre && bm.pre.length > 0 && bm.pre !== am.pre) pre.push(...bm.pre);
  let v: string;
  let dx: string;
  let dy: string;
  switch (op) {
    case '+':
      v = `(${av} + ${bv})`;
      dx = `(${adx} + ${bdx})`;
      dy = `(${ady} + ${bdy})`;
      break;
    case '-':
      v = `(${av} - ${bv})`;
      dx = `(${adx} - ${bdx})`;
      dy = `(${ady} - ${bdy})`;
      break;
    case '*':
      v = `(${av} * ${bv})`;
      dx = `(${adx} * ${bv} + ${av} * ${bdx})`;
      dy = `(${ady} * ${bv} + ${av} * ${bdy})`;
      break;
    case '/':
      v = `(${av} / ${bv})`;
      dx = `((${adx} * ${bv} - ${av} * ${bdx}) / (${bv} * ${bv}))`;
      dy = `((${ady} * ${bv} - ${av} * ${bdy}) / (${bv} * ${bv}))`;
      break;
    case '%': {
      v = `(${av} % ${bv})`;
      const q = `Math.floor(${av} / ${bv})`;
      dx = `(${adx} - ${q} * ${bdx})`;
      dy = `(${ady} - ${q} * ${bdy})`;
      break;
    }
    default:
      throw new Error(`codegen: bad arithmetic op '${op}'`);
  }
  const out: Value = { v, dx, dy };
  if (pre.length > 0) out.pre = pre;
  return out;
}

/** Dual-aware left fold of '+' (dot-product sums of matrix multiplies). */
function foldAdd(terms: Value[], env: CodegenEnv): Value {
  let acc = terms[0];
  for (let i = 1; i < terms.length; i++) acc = arithDual('+', acc, terms[i], env);
  return acc;
}

function emitArith(
  e: Extract<Expr, { kind: 'binary' }>,
  env: CodegenEnv,
  lt: GLSLType,
  rt: GLSLType,
  t: GLSLType,
): Value[] {
  const op = e.op;
  const lb = scalarBaseOf(lt);
  const rb = scalarBaseOf(rt);
  const base = commonBase(lb, rb)!;
  const isU = base === 'uint';
  const isI = base === 'int';
  // Dual mode: float-typed results carry (v, dx, dy) per the arithmetic
  // dual templates (arithDual); int/uint arithmetic has no duals.
  const dual = env.dual && hasFloatLeaves(t);
  // matrix * matrix / matrix * vector / vector * matrix
  if (op === '*') {
    if (lt.kind === 'matrix' && rt.kind === 'matrix') {
      // A (aCols × aRows) * B (bCols × bRows), aCols == bRows → bCols × aRows.
      const aRows = lt.rows;
      const aCols = lt.cols;
      const bRows = rt.rows;
      const av = emitExpr(e.left, env);
      const bv = emitExpr(e.right, env);
      const out: Value[] = [];
      // ALIASING FIX (sequential-assignment aliasing): the non-dual result
      // strings below reference the OPERAND components directly — a sequential
      // target write (`v = m * v`, `m = m1 * m2`) would observe partially-
      // updated operands (GLSL requires every RHS read to see pre-assignment
      // values). Materialize BOTH operands into fresh temps (ONE shared pre on
      // component 0 — the comp0-hoist convention, matrixCompoundMul idiom) so
      // all reads are captured before any target write. The dual branch
      // materializes per term inside arithDual.
      const matPre: string[] = [];
      const aT = !dual ? materializeOperands(av, env, matPre) : null;
      const bT = !dual ? materializeOperands(bv, env, matPre) : null;
      for (let c = 0; c < rt.cols; c++) {
        for (let r = 0; r < aRows; r++) {
          if (dual) {
            const terms: Value[] = [];
            for (let s = 0; s < aCols; s++) {
              terms.push(arithDual('*', av[s * aRows + r], bv[c * bRows + s], env));
            }
            out.push(terms.length === 1 ? terms[0] : foldAdd(terms, env));
          } else {
            const parts: string[] = [];
            for (let s = 0; s < aCols; s++) {
              parts.push(`(${aT![s * aRows + r]} * (${bT![c * bRows + s]}))`);
            }
            const res: Value = { v: `(${parts.join(' + ')})` };
            if (out.length === 0) res.pre = matPre;
            out.push(res);
          }
        }
      }
      return out;
    }
    if (lt.kind === 'matrix' && rt.kind === 'vector') {
      // M (C × R) * v (C) → v (R): result[r] = Σ_c M[c*R + r] * v[c]
      const R = lt.rows;
      const C = lt.cols;
      const av = emitExpr(e.left, env);
      const bv = emitExpr(e.right, env);
      const out: Value[] = [];
      // ALIASING FIX — same as the matrix×matrix branch: materialize both
      // operands into temps (shared pre on component 0) so in-place targets
      // (`v = m * v` — result[r] reads ALL of v) never observe partially-
      // updated operands during sequential writes.
      const matPre: string[] = [];
      const aT = !dual ? materializeOperands(av, env, matPre) : null;
      const bT = !dual ? materializeOperands(bv, env, matPre) : null;
      for (let r = 0; r < R; r++) {
        if (dual) {
          const terms: Value[] = [];
          for (let c = 0; c < C; c++) {
            terms.push(arithDual('*', av[c * R + r], bv[c], env));
          }
          out.push(terms.length === 1 ? terms[0] : foldAdd(terms, env));
        } else {
          const parts: string[] = [];
          for (let c = 0; c < C; c++) {
            parts.push(`(${aT![c * R + r]} * (${bT![c]}))`);
          }
          const res: Value = { v: `(${parts.join(' + ')})` };
          if (out.length === 0) res.pre = matPre;
          out.push(res);
        }
      }
      return out;
    }
    if (lt.kind === 'vector' && rt.kind === 'matrix') {
      // v (R) * M (C × R) → v (C): result[c] = Σ_r v[r] * M[c*R + r]
      const R = rt.rows;
      const C = rt.cols;
      const av = emitExpr(e.left, env);
      const bv = emitExpr(e.right, env);
      const out: Value[] = [];
      // ALIASING FIX — same as the matrix×matrix branch: materialize both
      // operands into temps (shared pre on component 0) so in-place targets
      // (`v = v * M` — result[c] reads ALL of v) never observe partially-
      // updated operands during sequential writes.
      const matPre: string[] = [];
      const aT = !dual ? materializeOperands(av, env, matPre) : null;
      const bT = !dual ? materializeOperands(bv, env, matPre) : null;
      for (let c = 0; c < C; c++) {
        if (dual) {
          const terms: Value[] = [];
          for (let r = 0; r < R; r++) {
            terms.push(arithDual('*', av[r], bv[c * R + r], env));
          }
          out.push(terms.length === 1 ? terms[0] : foldAdd(terms, env));
        } else {
          const parts: string[] = [];
          for (let r = 0; r < R; r++) {
            parts.push(`(${aT![r]} * (${bT![c * R + r]}))`);
          }
          const res: Value = { v: `(${parts.join(' + ')})` };
          if (out.length === 0) res.pre = matPre;
          out.push(res);
        }
      }
      return out;
    }
  }
  // Component-wise with scalar expansion.
  const av = emitExpr(e.left, env);
  const bv = emitExpr(e.right, env);
  const n = flatComponents(t);
  const out: Value[] = [];
  // BUG (vector-scalar-arithmetic-inside-loop): a SCALAR operand's string is
  // duplicated per component (n > 1) — a side-effectful scalar (compound
  // assign, ++/--, call) is materialized ONCE into a temp (assignment in
  // sharedPre, attached to every result component below so it runs exactly
  // once). Scalar×scalar (n === 1) uses each operand once — no hoist.
  const sharedPre: string[] = [];
  const aScalar = lt.kind === 'scalar' && n > 1 ? broadcastScalar(av[0], e.left, env, sharedPre) : null;
  const bScalar = rt.kind === 'scalar' && n > 1 ? broadcastScalar(bv[0], e.right, env, sharedPre) : null;
  for (let c = 0; c < n; c++) {
    const a = aScalar ?? av[c];
    const b = bScalar ?? bv[c];
    if (dual) {
      out.push(arithDual(op, a, b, env));
      continue;
    }
    const x = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
    const y = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
    let s: string;
    // Int/uint results wrap INSIDE the parens (`| 0` / `>>> 0` bind looser
    // than `+` — a bare `(X) | 0` would mis-parse when embedded in a larger
    // expression, e.g. float(int sum) + float(uint sum)).
    switch (op) {
      case '+':
        s = isU ? `(((${x}) + (${y})) >>> 0)` : isI ? `(((${x}) + (${y})) | 0)` : `(${x} + ${y})`;
        break;
      case '-':
        s = isU ? `(((${x}) - (${y})) >>> 0)` : isI ? `(((${x}) - (${y})) | 0)` : `(${x} - ${y})`;
        break;
      case '*':
        s = isU ? `((Math.imul(${x}, ${y})) >>> 0)` : isI ? `(((${x}) * (${y})) | 0)` : `(${x} * ${y})`;
        break;
      case '/':
        s = isU ? `(((${x}) / (${y})) >>> 0)` : isI ? `(((${x}) / (${y})) | 0)` : `(${x} / ${y})`;
        break;
      case '%':
        s = isU ? `(((${x}) % (${y})) >>> 0)` : isI ? `(((${x}) % (${y})) | 0)` : `(${x} % ${y})`;
        break;
      default:
        throw new Error(`codegen: bad arithmetic op '${op}'`);
    }
    out.push({ v: s });
  }
  if (sharedPre.length > 0) {
    // Attach the hoist ONLY to component 0 (the FIRST emitted/consumed
    // component): statement emitters dedupe pres by array identity and
    // expression-context consumers (walkObject, ternary, comparisons,
    // bitwise, comma) fold each component's pre INLINE in 0..n-1 order —
    // comp0's pre runs first and sets the hoist temp, later components just
    // read it. A shared array on EVERY component would re-run the side
    // effect per component in expression contexts (each folds the same
    // hoist inline). comp0 keeps any pre it already has (dual arithDual
    // results) with the hoist terms prepended.
    out[0].pre = out[0].pre ? [...sharedPre, ...out[0].pre] : sharedPre;
  }
  return out;
}

/** One component of a compound-assignment op (lhs already converted).
 *  The wrap sits inside the outer parens so the whole assignment is an atom
 *  (`| 0` / `>>> 0` bind looser than `+`). */
function compoundOp(op: string, target: string, rhs: string, base: string): string {
  const isU = base === 'uint';
  const isI = base === 'int';
  switch (op) {
    case '+':
      return isU ? `(${target} = (((${target}) + (${rhs})) >>> 0))` : isI ? `(${target} = (((${target}) + (${rhs})) | 0))` : `(${target} = ${target} + ${rhs})`;
    case '-':
      return isU ? `(${target} = (((${target}) - (${rhs})) >>> 0))` : isI ? `(${target} = (((${target}) - (${rhs})) | 0))` : `(${target} = ${target} - ${rhs})`;
    case '*':
      return isU ? `(${target} = ((Math.imul(${target}, ${rhs})) >>> 0))` : isI ? `(${target} = (((${target}) * (${rhs})) | 0))` : `(${target} = ${target} * ${rhs})`;
    case '/':
      return isU ? `(${target} = (((${target}) / (${rhs})) >>> 0))` : isI ? `(${target} = (((${target}) / (${rhs})) | 0))` : `(${target} = ${target} / ${rhs})`;
    case '%':
      return isU ? `(${target} = (((${target}) % (${rhs})) >>> 0))` : isI ? `(${target} = (((${target}) % (${rhs})) | 0))` : `(${target} = ${target} % ${rhs})`;
    case '<<':
      return isU ? `(${target} = (((${target}) << ((${rhs}) >>> 0)) >>> 0))` : `(${target} = (((${target}) << ((${rhs}) >>> 0)) | 0))`;
    case '>>':
      return isU ? `(${target} = (((${target}) >>> ((${rhs}) >>> 0)) >>> 0))` : `(${target} = (((${target}) >> ((${rhs}) >>> 0)) | 0))`;
    case '&':
      return isU ? `(${target} = (((${target}) & (${rhs})) >>> 0))` : `(${target} = (((${target}) & (${rhs})) | 0))`;
    case '^':
      return isU ? `(${target} = (((${target}) ^ (${rhs})) >>> 0))` : `(${target} = (((${target}) ^ (${rhs})) | 0))`;
    case '|':
      return isU ? `(${target} = (((${target}) | (${rhs})) >>> 0))` : `(${target} = (((${target}) | (${rhs})) | 0))`;
    default:
      throw new Error(`codegen: bad compound op '${op}'`);
  }
}

/**
 * Matrix × matrix compound '*=' codegen (`m *= n` ≡ `m = m * n` — a MATRIX
 * PRODUCT, NOT the component-wise Hadamard product the generic compound
 * emitters produce for scalar/vector targets; CTS
 * conformance/glsl/matrices/matrix-compound-multiply.html). Mirrors the
 * emitArith mat×mat branch: output flat component k = c*aRows + r (column
 * major) = Σ_s lhs[s*aRows + r] * rhs[c*bRows + s], with aRows/aCols from
 * lhsType and bRows/bCols from rhsType. EVERY output column reads ALL LHS
 * columns, so a sequential per-component write would corrupt later reads —
 * the whole LHS is SNAPSHOTTED into temps before any write. The RHS is also
 * materialized into temps: (a) so its side-effect pres run exactly once
 * (deduped by pre-array identity — multi-component results share one chain;
 * without materialization the pres would re-run per output term) and (b)
 * because the RHS may ALIAS the LHS (`a *= a`) — raw strings would read
 * already-clobbered slots inside the write expressions. Dual mode: the
 * snapshot captures the v/dx/dy planes; each output term folds via
 * arithDual's product rule + foldAdd; the write updates all three planes
 * through env.dualWrite.
 * `rhs` must be the RAW emitExpr result at rhsType's own width (NOT
 * convertValue'd to the lvalue width — the RHS is read at its native stride
 * c*bRows+s, which differs from the lvalue layout for non-square shapes).
 * Returns `pre` (pure statements to run once, in order, AFTER the lvalue
 * prelude) and `writes` (one self-contained assignment expression per flat
 * component, in lvalue order — valid as statements or comma terms).
 */
export function matrixCompoundMul(
  env: CodegenEnv,
  targets: string[],
  dualTargets: ([string, string] | null)[] | undefined,
  lhsType: GLSLType,
  rhs: Value[],
  rhsType: GLSLType,
): { pre: string[]; writes: string[] } {
  const aCols = lhsType.kind === 'matrix' ? lhsType.cols : 0;
  const aRows = lhsType.kind === 'matrix' ? lhsType.rows : 0;
  const bCols = rhsType.kind === 'matrix' ? rhsType.cols : 0;
  const bRows = rhsType.kind === 'matrix' ? rhsType.rows : 0;
  const n = targets.length;
  const dual = env.dual && dualTargets !== undefined;
  const pre: string[] = [];
  // LHS snapshot: every flat component (v + dual planes) into temps BEFORE
  // any write — a naive sequential write corrupts later reads.
  const sV: string[] = [];
  const sDx: (string | null)[] = [];
  const sDy: (string | null)[] = [];
  for (let j = 0; j < n; j++) {
    const t = env.allocTemp();
    sV.push(t);
    pre.push(`${t} = ${targets[j]}`);
    const dj = dualTargets && dualTargets[j];
    if (dual && dj) {
      const tx = env.allocTemp();
      const ty = env.allocTemp();
      sDx.push(tx);
      sDy.push(ty);
      pre.push(`${tx} = ${dj[0]}`, `${ty} = ${dj[1]}`);
    } else {
      sDx.push(null);
      sDy.push(null);
    }
  }
  // RHS materialization: fold pres into the v temp (a shared pre chain runs
  // once, with the first component that carries it); dx/dy captured too —
  // they may reference the LHS, which the writes clobber.
  const rV: string[] = [];
  const rDx: (string | null)[] = [];
  const rDy: (string | null)[] = [];
  const preSeen = new Set<string[]>();
  for (let j = 0; j < rhs.length; j++) {
    const r = rhs[j];
    const t = env.allocTemp();
    if (r.pre && r.pre.length > 0 && !preSeen.has(r.pre)) {
      preSeen.add(r.pre);
      pre.push(`${t} = ${foldPre(r.pre, r.v)}`);
    } else {
      pre.push(`${t} = ${r.v}`);
    }
    rV.push(t);
    if (dual && r.dx !== undefined) {
      const tx = env.allocTemp();
      const ty = env.allocTemp();
      rDx.push(tx);
      rDy.push(ty);
      pre.push(`${tx} = ${r.dx}`, `${ty} = ${r.dy ?? '0'}`);
    } else {
      rDx.push(null);
      rDy.push(null);
    }
  }
  const writes: string[] = [];
  for (let c = 0; c < bCols; c++) {
    for (let r = 0; r < aRows; r++) {
      const k = c * aRows + r;
      if (dual) {
        const terms: Value[] = [];
        for (let s = 0; s < aCols; s++) {
          const l: Value = { v: sV[s * aRows + r] };
          if (sDx[s * aRows + r] !== null) {
            l.dx = sDx[s * aRows + r]!;
            l.dy = sDy[s * aRows + r]!;
          }
          const rr: Value = { v: rV[c * bRows + s] };
          if (rDx[c * bRows + s] !== null) {
            rr.dx = rDx[c * bRows + s]!;
            rr.dy = rDy[c * bRows + s]!;
          }
          terms.push(arithDual('*', l, rr, env));
        }
        const res = terms.length === 1 ? terms[0] : foldAdd(terms, env);
        writes.push(env.dualWrite(targets[k], dualTargets[k], res));
      } else {
        const parts: string[] = [];
        for (let s = 0; s < aCols; s++) {
          parts.push(`(${sV[s * aRows + r]} * (${rV[c * bRows + s]}))`);
        }
        writes.push(`(${targets[k]} = ${parts.join(' + ')})`);
      }
    }
  }
  return { pre, writes };
}

function emitAssign(e: Extract<Expr, { kind: 'assign' }>, env: CodegenEnv): Value[] {
  const lv = emitLValue(e.target, env);
  const rhs = emitExpr(e.value, env);
  const t = lv.type;
  const preludes = preludeLines(lv.prelude);
  const post = copyBackComma(lv.copyBack);
  const n = lv.targets.length;
  const out: Value[] = [];
  // BUG (vector-scalar-arithmetic-inside-loop): a SCALAR RHS broadcast to a
  // vector/matrix lvalue duplicates the RHS string per component — a
  // side-effectful RHS (compound assign, ++/--, call) is materialized ONCE
  // into a temp first (assignment in broadcastPre, emitted after the lvalue
  // preludes and before the writes — GLSL evaluates the lvalue before the
  // RHS).
  const broadcastPre: string[] = [];
  let rhsSrc = rhs;
  if (rhs.length === 1 && n > 1) {
    rhsSrc = [broadcastScalar(rhs[0], e.value, env, broadcastPre)];
  }
  if (e.op === '=') {
    let conv = convertValue(rhsSrc, e.value.resolvedType!, t);
    // convertValue DROPS Value.pre when it converts scalar bases — re-attach
    // (mirrors statements.ts convertPreserving) so RHS pres survive. Iterate
    // the SOURCE length: a broadcast RHS (scalar → vector/matrix) is shorter
    // than the lvalue — reading past it would hit `undefined.pre`.
    const srcN = Math.min(n, rhs.length);
    for (let c = 0; c < srcN; c++) {
      const src = rhs[c];
      if (conv[c] !== src && src.pre && src.pre.length > 0) {
        conv = conv.map((v, i) => (i === c ? { ...v, pre: src.pre } : v));
      }
    }
    if (env.dual && lv.dualTargets) {
      // Dual mode: the write itself is the shared `pre` — one comma
      // expression per component `(vslot = rv, dxslot = dxv, dyslot = dyv,
      // vslot)` (the RHS pres fold into rv, so they run exactly when the
      // composite runs — even when only the dx/dy planes are consumed, e.g.
      // dFdx(t = v)). Prelude lines (dyn-index temps / spill copy-in) run
      // first, copyBack (spill copy-out) after all composites. The VALUE of
      // the assignment is the target read back; its duals are the RHS duals
      // (pure — they reference temps the composites' folded pres set).
      const pre: string[] = [];
      if (preludes.length > 0) pre.push(...preludes);
      if (broadcastPre.length > 0) pre.push(...broadcastPre);
      for (let c = 0; c < n; c++) {
        const cp = conv[c].pre;
        const rv = cp && cp.length ? foldPre(cp, conv[c].v) : conv[c].v;
        const bitKind = lv.bits !== undefined ? lv.bits[c] : false;
        if (bitKind) {
          // Packed int/uint varying cell (VERTEX): store the value's BIT
          // PATTERN; the write composite ends with the unpacked read (the
          // assignment's value is the assigned int/uint).
          pre.push(packVaryingWrite(lv.targets[c], rv, bitKind === 'int'));
        } else {
          pre.push(env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...conv[c], v: rv }));
        }
      }
      if (post) pre.push(...post.split(', '));
      for (let c = 0; c < n; c++) {
        const bitKind = lv.bits !== undefined ? lv.bits[c] : false;
        if (bitKind) {
          out.push({ v: unpackVaryingCell(lv.targets[c], bitKind === 'int'), dx: '0', dy: '0', pre });
        } else {
          out.push({ v: lv.targets[c], dx: conv[c].dx ?? '0', dy: conv[c].dy ?? '0', pre });
        }
      }
      return out;
    }
    // Per-component pre for the non-dual '=' path: comp0 carries the lvalue
    // preludes PLUS the scalar-broadcast hoist; comps1+ carry only the
    // (idempotent) preludes. NOT a single shared array — expression-context
    // consumers (walkObject, ternary, comparisons) fold each component's pre
    // inline in 0..n-1 order, so comp0's hoist must run first and the later
    // components must NOT re-run it. Statement emitters dedupe by identity:
    // comp0's array runs the hoist exactly once, comps1+'s shared preludes
    // array re-emits only the idempotent preludes.
    const comp0Pre = broadcastPre.length > 0 ? [...preludes, ...broadcastPre] : preludes;
    for (let c = 0; c < n; c++) {
      const cp = conv[c].pre;
      const rv = cp && cp.length ? foldPre(cp, conv[c].v) : conv[c].v;
      const bitKind = lv.bits !== undefined ? lv.bits[c] : false;
      // Dual mode: write the whole triple; the comma expression ends with
      // the v read so the value of the assignment is the assigned v.
      // Prelude (dyn-index temps / spill copy-in) must run BEFORE the write,
      // the spill copy-back AFTER it — and the expression's VALUE must be the
      // assigned value. Semicolons are invalid inside parens, so fold copyBack
      // as comma terms via a temp: (t = (target = rv), cb, t).
      let v =
        bitKind
          ? packVaryingWrite(lv.targets[c], rv, bitKind === 'int')
          : env.dual && lv.dualTargets
            ? env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...conv[c], v: rv })
            : `(${lv.targets[c]} = ${rv})`;
      if (post) {
        const t = env.allocTemp();
        v = `(${t} = ${v}, ${post}, ${t})`;
      }
      out.push(c === 0 && comp0Pre.length > 0 ? { v, pre: comp0Pre } : { v });
    }
    return out;
  }
  // compound: target op= rhs  (read target once — targets are pure paths)
  const base = scalarBaseOf(t);
  if (!base) throw new Error('codegen: cannot compound-assign a non-scalar-shaped value');
  const cop = e.op.slice(0, -1); // parser emits '*=' — compoundOp switches on '*'
  // mat×mat '*=' is a MATRIX PRODUCT, not the component-wise lowering below
  // (regression: matrix-compound-multiply CTS page rendered black). Emit via
  // matrixCompoundMul (LHS snapshot + RHS materialization, dual aware) with
  // the RAW RHS at its own width — never convertValue'd/broadcast.
  if (cop === '*' && t.kind === 'matrix' && e.value.resolvedType!.kind === 'matrix') {
    const mm = matrixCompoundMul(env, lv.targets, lv.dualTargets, t, rhs, e.value.resolvedType!);
    if (env.dual && lv.dualTargets) {
      // Dual mode mirrors the generic dual compound path below: the snapshot,
      // materialization and write composites share ONE pre array (consumers
      // may read only the dx/dy planes, so every component must trigger the
      // writes); the assignment's value reads the slots back.
      const pre: string[] = [];
      if (preludes.length > 0) pre.push(...preludes);
      if (broadcastPre.length > 0) pre.push(...broadcastPre);
      pre.push(...mm.pre, ...mm.writes);
      if (post) pre.push(...post.split(', '));
      for (let c = 0; c < n; c++) {
        const d = lv.dualTargets[c];
        out.push({ v: lv.targets[c], dx: d ? d[0] : '0', dy: d ? d[1] : '0', pre });
      }
      return out;
    }
    // Non-dual: comp0 carries preludes + broadcast hoist + snapshot/
    // materialization; the writes fold into v per component (same shape as
    // the generic compound path below — comp0's pre runs first, the writes
    // read the snapshot temps).
    const pre: string[] = [...preludes, ...broadcastPre, ...mm.pre];
    for (let c = 0; c < n; c++) {
      let v = mm.writes[c];
      if (post) {
        const t2 = env.allocTemp();
        v = `(${t2} = ${v}, ${post}, ${t2})`;
      }
      out.push(c === 0 && pre.length > 0 ? { v, pre } : { v });
    }
    return out;
  }
  let conv = convertValue(rhsSrc, e.value.resolvedType!, t);
  // convertValue DROPS Value.pre when it converts scalar bases — re-attach
  // (mirrors statements.ts convertPreserving) so RHS pres survive. Iterate
  // the SOURCE length: a broadcast RHS (scalar → vector/matrix — `v += s`,
  // `m *= s`) is shorter than the lvalue — reading past it would hit
  // `undefined.pre`.
  const srcN = Math.min(n, rhs.length);
  for (let c = 0; c < srcN; c++) {
    const src = rhs[c];
    if (conv[c] !== src && src.pre && src.pre.length > 0) {
      conv = conv.map((v, i) => (i === c ? { ...v, pre: src.pre } : v));
    }
  }
  // (cop computed above — parser emits '*='; compoundOp switches on '*')
  if (env.dual && lv.dualTargets && base === 'float') {
    // Dual mode, float target: the compound composite (updates all three
    // planes — dualWrite) is the shared `pre`; the expression's value reads
    // the target back and its duals are the post-write slot reads (valid
    // after the composite ran). Prelude/copyBack order as in the '=' path.
    const pre: string[] = [];
    if (preludes.length > 0) pre.push(...preludes);
    if (broadcastPre.length > 0) pre.push(...broadcastPre);
    for (let c = 0; c < n; c++) {
      const cp = conv[c].pre;
      const rv = cp && cp.length ? foldPre(cp, conv[c].v) : conv[c].v;
      if (lv.dualTargets[c]) {
        pre.push(env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...conv[c], v: rv }, cop));
      } else {
        pre.push(`(${lv.targets[c]} = ${rv})`);
      }
    }
    if (post) pre.push(...post.split(', '));
    for (let c = 0; c < n; c++) {
      const d = lv.dualTargets[c];
      out.push({ v: lv.targets[c], dx: d ? d[0] : conv[c].dx ?? '0', dy: d ? d[1] : conv[c].dy ?? '0', pre });
    }
    return out;
  }
  // Per-component pre for the non-dual compound path: comp0 carries the
  // lvalue preludes PLUS the scalar-broadcast hoist; comps1+ carry only the
  // (idempotent) preludes. NOT a single shared array — expression-context
  // consumers fold each component's pre inline in 0..n-1 order, so comp0's
  // hoist must run first and later components must NOT re-run it (see the
  // '=' path above).
  const comp0Pre = broadcastPre.length > 0 ? [...preludes, ...broadcastPre] : preludes;
  for (let c = 0; c < n; c++) {
    const cp = conv[c].pre;
    const rv = cp && cp.length ? foldPre(cp, conv[c].v) : conv[c].v;
    const bitKind = lv.bits !== undefined ? lv.bits[c] : false;
    // Dual mode, float target: linear ops (+=, -=) update all three planes
    // via dualWrite; non-linear compounds throw (C5a2 templates).
    let v: string;
    if (bitKind) {
      // Packed int/uint varying cell (VERTEX): unpack the old value, apply
      // the op with the int32/uint32 wrap, repack (packVaryingCompound
      // mirrors compoundOp — int vs uint diverge on / % >>).
      v = packVaryingCompound(cop, lv.targets[c], rv, bitKind === 'int');
    } else if (env.dual && lv.dualTargets && base === 'float' && lv.dualTargets[c]) {
      v = env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...conv[c], v: rv }, cop);
    } else {
      v = compoundOp(cop, lv.targets[c], rv, base);
    }
    if (post) {
      const t = env.allocTemp();
      v = `(${t} = ${v}, ${post}, ${t})`;
    }
    out.push(c === 0 && comp0Pre.length > 0 ? { v, pre: comp0Pre } : { v });
  }
  return out;
}

function emitTernary(e: Extract<Expr, { kind: 'ternary' }>, env: CodegenEnv): Value[] {
  // The condition is a scalar bool embedded in EVERY result component's
  // selects (and in the guarded arm hoists below) — hoist it ONCE when it
  // carries side effects (Value.pre chains OR side effects folded into its v
  // string — user calls nested in comparators), so per-component duplication
  // cannot re-run them (BUG: multi-component ternary condition side effects
  // ran n times). Pure conditions pass through unchanged (broadcastScalar).
  const sharedPre: string[] = [];
  const cond = broadcastScalar(emitExpr(e.cond, env)[0], e.cond, env, sharedPre);
  const a = emitExpr(e.whenTrue, env);
  const b = emitExpr(e.whenFalse, env);
  const n = flatComponents(e.resolvedType!);
  const out: Value[] = [];
  // Dual mode, float-typed ternary: BOTH arms carry triples; the cond is bool
  // (v-only); its pres join the result's pre so materialized cond temps are
  // set even when only the dx/dy planes are consumed (dFdx(cond ? a : b)).
  const dual = env.dual && hasFloatLeaves(e.resolvedType!);
  // Distinct arm pre arrays by IDENTITY: a multi-component user call shares
  // ONE pre array (the inliner's IIFE). Only the FIRST component with a given
  // pre embeds it in its guarded hoist; later components just read the temps
  // it sets (their guarded lines run after comp0's in 0..n-1 order).
  // Embedding the pre per component would run the arm's side effects once per
  // component (BUG: multi-component ternary arm side effects ran n times).
  const seenAPre = new Set<readonly string[]>();
  const seenBPre = new Set<readonly string[]>();
  for (let c = 0; c < n; c++) {
    const ac = a[c];
    const bc = b[c];
    const ap = ac.pre && ac.pre.length > 0 ? ac.pre : null;
    const bp = bc.pre && bc.pre.length > 0 ? bc.pre : null;
    const pre: string[] = [];
    if (ap === null && bp === null) {
      // FAST PATH (no arm materializes — pres are the only reason
      // materialize would hoist): the select folds the raw arm strings, which
      // JS evaluates lazily (side effects in v, e.g. `cond ? (x = 1.0) :
      // (y = 2.0)`, only run for the taken arm).
      const val: Value =
        dual
          ? {
              v: `(${cond.v} ? (${ac.v}) : (${bc.v}))`,
              dx: `(${cond.v} ? (${ac.dx ?? '0'}) : (${bc.dx ?? '0'}))`,
              dy: `(${cond.v} ? (${ac.dy ?? '0'}) : (${bc.dy ?? '0'}))`,
            }
          : { v: `(${cond.v} ? (${ac.v}) : (${bc.v}))` };
      if (pre.length > 0) val.pre = pre;
      out.push(val);
      continue;
    }
    // LAZY PATH (BUG: sequence-operator-evaluation-order): an arm whose pre
    // carries side effects (user-call IIFEs, texture samples) must NOT run
    // when the condition takes the OTHER arm — GLSL sequence semantics:
    // only the selected operand evaluates. Guard each arm's hoist with the
    // condition — `tA = (cond ? (preA, vA) : tA)` — a JS ternary evaluates
    // only the taken branch, so the untaken arm's pre never runs. The final
    // select reads the temps; an untaken arm's temp stays undefined and is
    // never read. Dual mode: the arm pres also feed the dx/dy planes — they
    // run once via the guarded v hoist, and the plane selects are lazy too,
    // reading the taken arm's dual strings (which reference temps its pre
    // set).
    const ta = ap !== null ? env.allocTemp() : null;
    const tb = bp !== null ? env.allocTemp() : null;
    const aFirst = ap !== null && !seenAPre.has(ap);
    if (aFirst) seenAPre.add(ap);
    const bFirst = bp !== null && !seenBPre.has(bp);
    if (bFirst) seenBPre.add(bp);
    if (ta) {
      pre.push(`${ta} = (${cond.v} ? ${aFirst ? `(${foldPre(ap!, ac.v)})` : `(${ac.v})`} : ${ta})`);
    }
    if (tb) {
      pre.push(`${tb} = (${cond.v} ? ${tb} : ${bFirst ? `(${foldPre(bp!, bc.v)})` : `(${bc.v})`})`);
    }
    const av = ta ?? ac.v;
    const bv = tb ?? bc.v;
    const val: Value =
      dual
        ? {
            v: `(${cond.v} ? (${av}) : (${bv}))`,
            dx: `(${cond.v} ? (${ac.dx ?? '0'}) : (${bc.dx ?? '0'}))`,
            dy: `(${cond.v} ? (${ac.dy ?? '0'}) : (${bc.dy ?? '0'}))`,
          }
        : { v: `(${cond.v} ? (${av}) : (${bv}))` };
    if (pre.length > 0) val.pre = pre;
    out.push(val);
  }
  if (sharedPre.length > 0) {
    // Attach the cond hoist ONLY to component 0 (the FIRST emitted/consumed
    // component): statement emitters dedupe pres by array identity and
    // expression-context consumers fold each component's pre INLINE in
    // 0..n-1 order — comp0's pre runs first and sets the hoist temp, later
    // components just read it (mirrors emitArith's sharedPre handling). A
    // shared array on EVERY component would re-run the hoisted side effects
    // per component in expression contexts.
    out[0].pre = out[0].pre ? [...sharedPre, ...out[0].pre] : sharedPre;
  }
  return out;
}

function emitComma(e: Extract<Expr, { kind: 'comma' }>, env: CodegenEnv): Value[] {
  const n = e.exprs.length;
  if (n === 0) throw new Error('codegen: empty comma expression');
  // BUG (shared-pre re-run): dedupe the LAST operand's pre arrays by identity
  // — its components BECOME the comma result, and folding a shared [iife] per
  // component (below) would re-run the callee once per component.
  const last = dedupeSharedPre(emitExpr(e.exprs[n - 1], env));
  const pre: string[] = [];
  for (let i = 0; i < n - 1; i++) {
    // BUG (expression-list-in-declarator-initializer): emit EVERY flat
    // component of an intermediate expression — a vector/matrix operand's
    // side effects span all components (`b = vec2(0.0, 1.0)` writes b.x AND
    // b.y; taking only [0] left b.y unassigned → the final value read NaN).
    // The intermediate result is discarded, so each component's pres fold
    // inline and the whole sequence lands in one temp as a comma term.
    // BUG (shared-pre re-run): the dedupe keeps a shared pre array (a
    // multi-component call result's ONE [iife]) on the FIRST component only —
    // folding it per component ran the callee n times.
    const vals = dedupeSharedPre(emitExpr(e.exprs[i], env));
    const t = env.allocTemp();
    const terms: string[] = [];
    for (const v of vals) {
      terms.push(v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v);
    }
    pre.push(`${t} = (${terms.join(', ')})`);
  }
  if (pre.length === 0) return last;
  if (env.dual) {
    // ONE shared pre array (the intermediate terms + the last expr's deduped
    // pre) attached to ONLY component 0 — statement emitters dedupe by array
    // identity (run once), and expression-context consumers fold each
    // component's pre INLINE in 0..n-1 order, so comp0's fold runs the
    // intermediates + the last expr's IIFE first and later components read
    // the temps it sets. A shared array on EVERY component would re-run the
    // intermediate side effects per component in expression contexts (same
    // comp0-only convention as emitArith's sharedPre / emitTernary's hoist).
    const shared: string[] = [...pre];
    const seen = new Set<string[]>();
    for (const v of last) {
      if (v.pre && v.pre.length > 0 && !seen.has(v.pre)) {
        seen.add(v.pre);
        shared.push(...v.pre);
      }
    }
    const arr = shared.length > 0 ? shared : undefined;
    return last.map((v, c) => {
      const out: Value = { v: `(${v.v})`, dx: v.dx, dy: v.dy };
      if (c === 0 && arr) out.pre = arr;
      return out;
    });
  }
  // Non-dual: the prelude is embedded in the v string of component 0 (its
  // evaluation runs the intermediate effects and sets the last expr's
  // retTemps; later components read the temps — 0..n-1 emission order
  // guarantees the first component's v ran first). Any DISTINCT pre still on
  // later components (dedupeSharedPre above kept only shared arrays on comp0)
  // folds inline per component.
  const prelude = `(${pre.join(', ')}, `;
  return last.map((v, c) => {
    const x = v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v;
    const s = c === 0 ? `${prelude}${x})` : `(${x})`;
    return { v: s };
  });
}

/* ------------------------------------------------------------------ */
/* Calls: user / constructor / builtin                                 */
/* ------------------------------------------------------------------ */

function emitCall(e: Extract<Expr, { kind: 'call' }>, env: CodegenEnv): Value[] {
  const callee = e.callee;
  if (callee.kind === 'identifier') {
    const name = callee.name;
    const argTypes = e.args.map((a) => a.resolvedType!);
    if (env.emitUserCall) {
      const r = env.emitUserCall(name, e.args.map((a) => emitExpr(a, env)), argTypes, e.args);
      if (r) return r;
    }
    if (isConstructorName(name, env)) return emitConstructorCall(name, e.args, e.resolvedType!, env);
    return emitBuiltinCall(e, env);
  }
  if (callee.kind === 'index') {
    // Array constructor: float[3](...)
    const t = e.resolvedType!;
    const elem = t.kind === 'array' ? t.element : undefined;
    if (!elem) throw new Error('codegen: indexed callee is not an array constructor');
    return emitArrayCtor(e.args, elem, t, env);
  }
  if (callee.kind === 'member') {
    // `.length()` on an array-typed expression (GLSL ES 3.00 §5.9): the
    // result is the STATIC element count, but the OBJECT's side effects
    // still evaluate exactly once (`(a = b).length()` runs the assignment;
    // `(f()).length()` runs the call). Materialize the object's components
    // (shared pres — the call's single [iife] — run once) and keep only the
    // materialization lines; the object's VALUE strings are discarded.
    if (callee.name === 'length') {
      const objType = callee.object.resolvedType;
      if (!objType || objType.kind !== 'array') {
        throw new Error(`codegen: '.length()' on a non-array type`);
      }
      const objVals = materializeSharedPre(emitExpr(callee.object, env), env);
      const pre: string[] = [];
      const seen = new Set<string[]>();
      for (const v of objVals) {
        if (v.pre && v.pre.length > 0 && !seen.has(v.pre)) {
          seen.add(v.pre);
          pre.push(...v.pre);
        }
      }
      const out: Value = { v: String(objType.size ?? 0) };
      if (pre.length > 0) out.pre = pre;
      return [out];
    }
    throw new Error(`codegen: unsupported member call '${callee.name}'`);
  }
  throw new Error('codegen: unsupported call callee');
}

/** Emit float[N](...) / vec3[N](...) array constructors (element-wise). Dual
 *  mode: convertValue preserves float→float duals and attaches constant duals
 *  for int→float elements, so element derivatives flow through. */
function emitArrayCtor(args: Expr[], elem: GLSLType, t: GLSLType, env: CodegenEnv): Value[] {
  const n = t.kind === 'array' ? (t.size ?? 0) : 0;
  const out: Value[] = [];
  for (let k = 0; k < n; k++) {
    const a = args[k];
    if (!a) throw new Error('codegen: array constructor arity mismatch');
    const srcVals = emitExpr(a, env);
    const vals = convertValue(srcVals, a.resolvedType!, elem);
    // convertValue DROPS Value.pre when it converts scalar bases — re-attach
    // so element pres survive.
    for (let c = 0; c < srcVals.length; c++) {
      const src = srcVals[c];
      if (vals[c] !== src && src.pre && src.pre.length > 0) {
        vals[c] = { ...vals[c], pre: src.pre };
      }
    }
    out.push(...vals);
  }
  return out;
}

const TYPE_NAMES = new Set([
  'float', 'int', 'uint', 'bool',
  'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4',
  'uvec2', 'uvec3', 'uvec4', 'bvec2', 'bvec3', 'bvec4',
  'mat2', 'mat3', 'mat4', 'mat2x2', 'mat2x3', 'mat2x4',
  'mat3x2', 'mat3x3', 'mat3x4', 'mat4x2', 'mat4x3', 'mat4x4',
]);

function isConstructorName(name: string, env: CodegenEnv): boolean {
  return TYPE_NAMES.has(name) || env.structNames.has(name);
}
