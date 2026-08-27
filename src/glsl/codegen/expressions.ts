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
import { typeName } from '../types.js';
import {
  CodegenEnv,
  LocalVar,
  flatComponents,
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
  uniformPathRead,
  blockPathRead,
  varyingPathRead,
  attribRead,
  outputAccess,
} from './env.js';
import type { Value } from './index.js';
import { emitConstructorCall } from './expr-ctor.js';
import { emitBuiltinCall } from './expr-builtins.js';

/* ------------------------------------------------------------------ */
/* The path model                                                      */
/* ------------------------------------------------------------------ */

type StorageKind =
  | { kind: 'uniform'; key: string }
  | { kind: 'block'; blockIndex: number; key: string; instance: boolean }
  | { kind: 'varying'; key: string }
  | { kind: 'attrib'; location: number }
  | { kind: 'output'; location: number };

/** Walked access path: identifier + member/index chains. `dyn` (outermost
 *  dynamic index) is at most one — GLSL allows dynamic indexing on the
 *  outermost array dimension only. `dyn.stride` = storage stride per element
 *  (uniform: floats; block: bytes; varying: components; attrib: locations;
 *  output: 1); `dyn.elemSlots` = flat components per element for LOCAL
 *  scratch storage. */
interface P {
  type: GLSLType;      // type OF THE VALUE AT THE PATH END
  lvalue: boolean;
  local: LocalVar | null;
  flatOff: number;     // flat component offset (locals: from storage base; storage: from leaf base)
  storage: StorageKind | null;
  builtin: string | null; // gl_Position / gl_FragCoord / ... (gl_FragData converts to output)
  swz: number[] | null;   // swizzle remap: leafRead(c) = baseRead(swz[c])
  dyn: { temp: string; stride: number; elemSlots: number } | null;
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
    const store = lv.int ? 'ctx.intScratch' : 'ctx.scratch';
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
      case 'varying':
        return varyingPathRead(env, st.key, p.type, p.dyn, flatC);
      case 'attrib':
        return attribRead(p.type, st.location, p.dyn, flatC);
      case 'output':
        return outputAccess(p.type, st.location, p.dyn, flatC);
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
      case 'gl_FragDepth':
      case 'gl_FragDepthEXT':
        return 'ctx.out.fragDepth';
      case 'gl_FragColor': {
        const loc = env.layout.outputLocations.get('gl_FragColor') ?? 0;
        return `ctx.out.color[${loc}][${i}]`;
      }
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
      return [dx, `${lv.compNames![idx]}_dy`];
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
    const store = lv.int ? 'ctx.intScratch' : 'ctx.scratch';
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
        return outputAccess(p.type, st.location, p.dyn, flatC);
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
      return [dx, `${lv.compNames![idx]}_dy`];
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
      case 'block':
        q.storage = { ...q.storage, key: q.storage.key + `[${k}]` };
        break;
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
            instance: info.baseKey !== '',
          };
          return p;
        case 'attrib':
          p.storage = { kind: 'attrib', location: info.location };
          return p;
        case 'varying':
          p.storage = { kind: 'varying', key: info.key };
          p.lvalue = env.stage === 'VERTEX';
          return p;
        case 'output':
          p.storage = { kind: 'output', location: info.location };
          p.lvalue = true;
          return p;
        case 'builtin': {
          if (e.name === 'gl_FragData') {
            p.storage = { kind: 'output', location: env.layout.outputLocations.get('gl_FragData') ?? 0 };
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
      const p = walk(e.object, env);
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
      const p = walk(e.object, env);
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
              p.storage = { ...p.storage, key };
              const stride = p.storage.instance ? entry.blockStride ?? 0 : entry.arrayStride;
              if (stride <= 0) {
                throw new Error(
                  `codegen: block path '${key}' has no stride for dynamic indexing (linker must set arrayStride/blockStride)`,
                );
              }
              p.dyn = { temp: t, stride, elemSlots: 0 };
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
        p.swz = null;
        if (isConst) {
          p.flatOff += cv;
          return p;
        }
        // dynamic component (ES 3.00 allows it): spill locals / stride-1 storage
        const t = env.allocTemp();
        p.pre.push(`${t} = ${idxV!.pre && idxV!.pre.length ? foldPre(idxV!.pre, idxV!.v) : idxV!.v}`);
        if (p.local) {
          const ds = env.ensureDynScratch(p.local.name);
          p.pre.unshift(...ds.copyIn);
          p.post.push(...ds.copyOut);
          p.local = { ...p.local, kind: 'scratch', scratchBase: ds.base, elemSlots: 1, int: ds.int };
          p.dyn = { temp: t, stride: 0, elemSlots: 1 };
        } else if (p.storage) {
          p.dyn = { temp: t, stride: storageElemStride(p, 1), elemSlots: 0 };
        }
        return p;
      }
      if (ot.kind === 'matrix') {
        const rows = ot.rows;
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
          const ds = env.ensureDynScratch(p.local.name);
          p.pre.unshift(...ds.copyIn);
          p.post.push(...ds.copyOut);
          p.local = { ...p.local, kind: 'scratch', scratchBase: ds.base, elemSlots: rows, int: ds.int };
          p.dyn = { temp: t, stride: 0, elemSlots: rows };
        } else if (p.storage) {
          p.dyn = { temp: t, stride: storageElemStride(p, rows), elemSlots: 0 };
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
  /** Statements that must run BEFORE the writes (dynamic-index temps, spill copy-in). */
  prelude?: string;
  /** Statements that must run AFTER the writes (spill copy-out). */
  copyBack?: string;
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
  return {
    type: p.type,
    targets,
    dualTargets: env.dual ? dualWrites(p, env) : undefined,
    prelude: p.pre.length ? p.pre.join('; ') + ';' : undefined,
    copyBack: p.post.length ? p.post.join('; ') + ';' : undefined,
  };
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

/* ------------------------------------------------------------------ */
/* Unary / binary / ternary / comma / assign                           */
/* ------------------------------------------------------------------ */

function emitUnary(e: Extract<Expr, { kind: 'unary' }>, env: CodegenEnv): Value[] {
  const t = e.resolvedType!;
  const base = scalarBaseOf(t);
  const n = flatComponents(t);
  const op = e.op;
  if (op === '++' || op === '--') {
    // GLSL ES: ++ / -- are PREFIX-only. Per-component on the lvalue.
    const lv = emitLValue(e.operand, env);
    const delta = op === '++' ? '1' : '-1';
    const preludes = preludeLines(lv.prelude);
    const post = copyBackComma(lv.copyBack);
    const out: Value[] = [];
    for (let c = 0; c < n; c++) {
      const target = lv.targets[c];
      let s: string;
      if (base === 'float') s = `(${target} = ${target} + ${delta})`;
      else if (base === 'int') s = `(${target} = (${target} + ${delta}) | 0)`;
      else if (base === 'uint') s = `(${target} = (${target} + ${delta}) >>> 0)`;
      else throw new Error('codegen: cannot increment a bool');
      // Prelude (dyn-index temps / spill copy-in) runs BEFORE the write, the
      // spill copy-back AFTER it — and the expression's VALUE must be the
      // incremented target. Semicolons are invalid inside parens, so fold
      // copyBack as comma terms via a temp: (t = (target = target + 1), cb, t).
      let v = s;
      if (post) {
        const t = env.allocTemp();
        v = `(${t} = ${s}, ${post}, ${t})`;
      }
      out.push(preludes.length > 0 ? { v, pre: preludes } : { v });
    }
    return out;
  }
  const vals = emitExpr(e.operand, env);
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
      return [{ v: `(${av} !== (${bv}))` }];
    }
    case '==':
    case '!=': {
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
        if (av[c] !== avSrc[c] && avSrc[c].pre && avSrc[c].pre.length > 0) {
          av[c] = { ...av[c], pre: avSrc[c].pre };
        }
        if (bv[c] !== bvSrc[c] && bvSrc[c].pre && bvSrc[c].pre.length > 0) {
          bv[c] = { ...bv[c], pre: bvSrc[c].pre };
        }
      }
      const parts: string[] = [];
      for (let c = 0; c < flatComponents(lt); c++) {
        const ap = av[c].pre;
        const bp = bv[c].pre;
        const a = ap && ap.length ? foldPre(ap, av[c].v) : av[c].v;
        const b = bp && bp.length ? foldPre(bp, bv[c].v) : bv[c].v;
        parts.push(op === '==' ? `(${a} === (${b}))` : `(${a} !== (${b}))`);
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
        if (op === '<<') {
          return { v: lb === 'uint' ? `((${x}) << ${shift}) >>> 0` : `((${x}) << ${shift}) | 0` };
        }
        return { v: lb === 'uint' ? `((${x}) >>> ${shift}) >>> 0` : `((${x}) >> ${shift}) | 0` };
      });
    }
    case '&':
    case '|':
    case '^': {
      const cb = commonBase(lb, rb);
      if (!cb) throw new Error(`codegen: incompatible bitwise operands`);
      const ct = shapeOf(lt, cb);
      const av = convertValue(emitExpr(e.left, env), lt, ct);
      const bv = convertValue(emitExpr(e.right, env), rt, ct);
      const isU = cb === 'uint';
      return av.map((a, c) => {
        const x = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
        const y = bv[c].pre && bv[c].pre.length ? foldPre(bv[c].pre, bv[c].v) : bv[c].v;
        return { v: isU ? `((${x}) ${op} (${y})) >>> 0` : `((${x}) ${op} (${y})) | 0` };
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
              const a = av[s * aRows + r];
              const b = bv[c * bRows + s];
              const ax = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
              const bx = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
              parts.push(`(${ax} * (${bx}))`);
            }
            out.push({ v: `(${parts.join(' + ')})` });
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
            const a = av[c * R + r];
            const b = bv[c];
            const ax = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
            const bx = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
            parts.push(`(${ax} * (${bx}))`);
          }
          out.push({ v: `(${parts.join(' + ')})` });
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
            const a = av[r];
            const b = bv[c * R + r];
            const ax = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
            const bx = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
            parts.push(`(${ax} * (${bx}))`);
          }
          out.push({ v: `(${parts.join(' + ')})` });
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
  for (let c = 0; c < n; c++) {
    const a = av[lt.kind === 'scalar' ? 0 : c];
    const b = bv[rt.kind === 'scalar' ? 0 : c];
    if (dual) {
      out.push(arithDual(op, a, b, env));
      continue;
    }
    const x = a.pre && a.pre.length ? foldPre(a.pre, a.v) : a.v;
    const y = b.pre && b.pre.length ? foldPre(b.pre, b.v) : b.v;
    let s: string;
    switch (op) {
      case '+':
        s = isU ? `((${x}) + (${y})) >>> 0` : isI ? `((${x}) + (${y})) | 0` : `(${x} + ${y})`;
        break;
      case '-':
        s = isU ? `((${x}) - (${y})) >>> 0` : isI ? `((${x}) - (${y})) | 0` : `(${x} - ${y})`;
        break;
      case '*':
        s = isU ? `(Math.imul(${x}, ${y})) >>> 0` : isI ? `((${x}) * (${y})) | 0` : `(${x} * ${y})`;
        break;
      case '/':
        s = isU ? `((${x}) / (${y})) >>> 0` : isI ? `((${x}) / (${y})) | 0` : `(${x} / ${y})`;
        break;
      case '%':
        s = isU ? `((${x}) % (${y})) >>> 0` : isI ? `((${x}) % (${y})) | 0` : `(${x} % ${y})`;
        break;
      default:
        throw new Error(`codegen: bad arithmetic op '${op}'`);
    }
    out.push({ v: s });
  }
  return out;
}

/** One component of a compound-assignment op (lhs already converted). */
function compoundOp(op: string, target: string, rhs: string, base: string): string {
  const isU = base === 'uint';
  const isI = base === 'int';
  switch (op) {
    case '+':
      return isU ? `(${target} = ((${target}) + (${rhs})) >>> 0)` : isI ? `(${target} = ((${target}) + (${rhs})) | 0)` : `(${target} = ${target} + ${rhs})`;
    case '-':
      return isU ? `(${target} = ((${target}) - (${rhs})) >>> 0)` : isI ? `(${target} = ((${target}) - (${rhs})) | 0)` : `(${target} = ${target} - ${rhs})`;
    case '*':
      return isU ? `(${target} = (Math.imul(${target}, ${rhs})) >>> 0)` : isI ? `(${target} = ((${target}) * (${rhs})) | 0)` : `(${target} = ${target} * ${rhs})`;
    case '/':
      return isU ? `(${target} = ((${target}) / (${rhs})) >>> 0)` : isI ? `(${target} = ((${target}) / (${rhs})) | 0)` : `(${target} = ${target} / ${rhs})`;
    case '%':
      return isU ? `(${target} = ((${target}) % (${rhs})) >>> 0)` : isI ? `(${target} = ((${target}) % (${rhs})) | 0)` : `(${target} = ${target} % ${rhs})`;
    case '<<':
      return isU ? `(${target} = ((${target}) << ((${rhs}) >>> 0)) >>> 0)` : `(${target} = ((${target}) << ((${rhs}) >>> 0)) | 0)`;
    case '>>':
      return isU ? `(${target} = ((${target}) >>> ((${rhs}) >>> 0)) >>> 0)` : `(${target} = ((${target}) >> ((${rhs}) >>> 0)) | 0)`;
    case '&':
      return isU ? `(${target} = ((${target}) & (${rhs})) >>> 0)` : `(${target} = ((${target}) & (${rhs})) | 0)`;
    case '^':
      return isU ? `(${target} = ((${target}) ^ (${rhs})) >>> 0)` : `(${target} = ((${target}) ^ (${rhs})) | 0)`;
    case '|':
      return isU ? `(${target} = ((${target}) | (${rhs})) >>> 0)` : `(${target} = ((${target}) | (${rhs})) | 0)`;
    default:
      throw new Error(`codegen: bad compound op '${op}'`);
  }
}

function emitAssign(e: Extract<Expr, { kind: 'assign' }>, env: CodegenEnv): Value[] {
  const lv = emitLValue(e.target, env);
  const rhs = emitExpr(e.value, env);
  const t = lv.type;
  const preludes = preludeLines(lv.prelude);
  const post = copyBackComma(lv.copyBack);
  const n = lv.targets.length;
  const out: Value[] = [];
  if (e.op === '=') {
    let conv = convertValue(rhs, e.value.resolvedType!, t);
    // convertValue DROPS Value.pre when it converts scalar bases — re-attach
    // (mirrors statements.ts convertPreserving) so RHS pres survive.
    for (let c = 0; c < n; c++) {
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
      for (let c = 0; c < n; c++) {
        const cp = conv[c].pre;
        const rv = cp && cp.length ? foldPre(cp, conv[c].v) : conv[c].v;
        pre.push(env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...conv[c], v: rv }));
      }
      if (post) pre.push(...post.split(', '));
      for (let c = 0; c < n; c++) {
        out.push({ v: lv.targets[c], dx: conv[c].dx ?? '0', dy: conv[c].dy ?? '0', pre });
      }
      return out;
    }
    for (let c = 0; c < n; c++) {
      const cp = conv[c].pre;
      const rv = cp && cp.length ? foldPre(cp, conv[c].v) : conv[c].v;
      // Dual mode: write the whole triple; the comma expression ends with
      // the v read so the value of the assignment is the assigned v.
      // Prelude (dyn-index temps / spill copy-in) must run BEFORE the write,
      // the spill copy-back AFTER it — and the expression's VALUE must be the
      // assigned value. Semicolons are invalid inside parens, so fold copyBack
      // as comma terms via a temp: (t = (target = rv), cb, t).
      let v =
        env.dual && lv.dualTargets
          ? env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...conv[c], v: rv })
          : `(${lv.targets[c]} = ${rv})`;
      if (post) {
        const t = env.allocTemp();
        v = `(${t} = ${v}, ${post}, ${t})`;
      }
      out.push(preludes.length > 0 ? { v, pre: preludes } : { v });
    }
    return out;
  }
  // compound: target op= rhs  (read target once — targets are pure paths)
  const base = scalarBaseOf(t);
  if (!base) throw new Error('codegen: cannot compound-assign a non-scalar-shaped value');
  const conv = convertValue(rhs, e.value.resolvedType!, t);
  const cop = e.op.slice(0, -1); // parser emits '+=' — compoundOp switches on '+'
  if (env.dual && lv.dualTargets && base === 'float') {
    // Dual mode, float target: the compound composite (updates all three
    // planes — dualWrite) is the shared `pre`; the expression's value reads
    // the target back and its duals are the post-write slot reads (valid
    // after the composite ran). Prelude/copyBack order as in the '=' path.
    const pre: string[] = [];
    if (preludes.length > 0) pre.push(...preludes);
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
  for (let c = 0; c < n; c++) {
    const cp = conv[c].pre;
    const rv = cp && cp.length ? foldPre(cp, conv[c].v) : conv[c].v;
    // Dual mode, float target: linear ops (+=, -=) update all three planes
    // via dualWrite; non-linear compounds throw (C5a2 templates).
    let v: string;
    if (env.dual && lv.dualTargets && base === 'float' && lv.dualTargets[c]) {
      v = env.dualWrite(lv.targets[c], lv.dualTargets[c], { ...conv[c], v: rv }, cop);
    } else {
      v = compoundOp(cop, lv.targets[c], rv, base);
    }
    if (post) {
      const t = env.allocTemp();
      v = `(${t} = ${v}, ${post}, ${t})`;
    }
    out.push(preludes.length > 0 ? { v, pre: preludes } : { v });
  }
  return out;
}

function emitTernary(e: Extract<Expr, { kind: 'ternary' }>, env: CodegenEnv): Value[] {
  const cond = materialize(emitExpr(e.cond, env), env)[0];
  const a = materialize(emitExpr(e.whenTrue, env), env);
  const b = materialize(emitExpr(e.whenFalse, env), env);
  const n = flatComponents(e.resolvedType!);
  const out: Value[] = [];
  // Dual mode, float-typed ternary: BOTH arms carry triples — materialize
  // (which temps all three planes of pre-carrying values) hoists their pres,
  // then each plane is a plain `cond ? a : b` select. The cond is bool
  // (v-only); its pres join the result's pre so materialized cond temps are
  // set even when only the dx/dy planes are consumed (dFdx(cond ? a : b)).
  const dual = env.dual && hasFloatLeaves(e.resolvedType!);
  for (let c = 0; c < n; c++) {
    if (dual) {
      const pre: string[] = [];
      const cp = cond.pre;
      if (cp) pre.push(...cp);
      const ap = a[c].pre;
      if (ap) pre.push(...ap);
      const bp = b[c].pre;
      if (bp && bp !== ap) pre.push(...bp);
      const val: Value = {
        v: `(${cond.v} ? (${a[c].v}) : (${b[c].v}))`,
        dx: `(${cond.v} ? (${a[c].dx ?? '0'}) : (${b[c].dx ?? '0'}))`,
        dy: `(${cond.v} ? (${a[c].dy ?? '0'}) : (${b[c].dy ?? '0'}))`,
      };
      if (pre.length > 0) val.pre = pre;
      out.push(val);
    } else {
      out.push({ v: `(${cond.v} ? (${a[c].v}) : (${b[c].v}))` });
    }
  }
  return out;
}

function emitComma(e: Extract<Expr, { kind: 'comma' }>, env: CodegenEnv): Value[] {
  const n = e.exprs.length;
  if (n === 0) throw new Error('codegen: empty comma expression');
  const last = emitExpr(e.exprs[n - 1], env);
  const pre: string[] = [];
  for (let i = 0; i < n - 1; i++) {
    const v = emitExpr(e.exprs[i], env)[0];
    const t = env.allocTemp();
    pre.push(`${t} = ${v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v}`);
  }
  if (pre.length === 0) return last;
  const prelude = `(${pre.join(', ')}, `;
  // Dual mode: the comma's value is the LAST expr — propagate its duals.
  // The prelude terms + the last expr's pre attach to the Value (embedding
  // the prelude in the v string would strand side effects when only dx/dy
  // are consumed).
  return last.map((v) => {
    if (env.dual) return { v: `(${v.v})`, dx: v.dx, dy: v.dy, pre: [...pre, ...(v.pre ?? [])] };
    const s = `${prelude}${v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v})`;
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
    const vals = convertValue(emitExpr(a, env), a.resolvedType!, elem);
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
