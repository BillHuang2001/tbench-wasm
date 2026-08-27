/**
 * expr-builtins.ts — GLSL builtin-function call lowering (non-dual): emitBuiltinCall.
 *
 * Resolution: the call's argument types are matched (exactly, or via the ES
 * implicit numeric widenings int→float / uint→float / int→uint) against
 * `builtinSignatures(version)` + `extensionFunctions`. The resolved signature's
 * `stage` restriction is enforced. Each argument is emitted EXACTLY once (its
 * `pre` materializations fold inline or hoist into temps when used more than
 * once).
 *
 * Lowering coverage:
 * - §8.3 common functions (radians..roundEven, mod, min/max/clamp, mix incl.
 *   the bvec selector, step/smoothstep, isnan/isinf, float↔int bit
 *   reinterpretation, modf with out-param).
 * - §8.4 pack/unpack (via the R.pack / R.unpack helpers; unpack results land in scratch).
 * - §8.5 geometry (length/distance/dot/cross/normalize/faceforward/reflect/
 *   refract — dot-heavy forms materialize the dot into a temp).
 * - §8.6 matrix (matrixCompMult/outerProduct/transpose/determinant/inverse;
 *   determinant = inline cofactor expansions; inverse = R.invN into scratch).
 * - §8.10 bitfield ops (R.bitfield*; findMSB dispatches int→R.findMSBI,
 *   uint→R.findMSB) and the out-param extended ops (uaddCarry/usubBorrow/
 *   umulExtended/imulExtended — R writes [sum,carry]/[hi,lo] into
 *   ctx.intScratch, the out-arg lvalues are written from there).
 * - §8.7/§8.8–8.9 texture lookups — see emitTextureCall below.
 * - Relational (lessThan..notEqual, any/all/not).
 *
 * OUT-PARAM CONTRACT: GLSL out-param writes are side effects, so they fold
 * into the emitted code rather than being dropped. uaddCarry/usubBorrow/modf
 * (which RETURN a value) carry their out-arg write in the shared `pre` array
 * (the pre runs exactly once before any component read — see the Value
 * contract in index.ts); umulExtended/imulExtended (void return) fold the
 * entire call into a single comma-expression Value so the side effects execute
 * even though no component values are produced.
 *
 * DERIVATIVES (dFdx/dFdy/fwidth): these are handled by the dual-number mode
 * (C5) — the non-dual path throws a clear error instead of emitting garbage.
 */
import type { CallExpr, Expr } from '../ast.js';
import type { GLSLType } from '../types.js';
import { typeEquals } from '../types.js';
import { matches, builtinSignatures, extensionFunctions } from '../builtins/index.js';
import type { BuiltinSignature } from '../builtins/index.js';
import type { CodegenEnv } from './env.js';
import { scalarBaseOf, flatComponents, isBoolType, wrapInt, wrapUint, convertScalar, foldPre, hasFloatLeaves } from './env.js';
import type { Value } from './index.js';
import { emitExpr, materialize, emitLValue } from './expressions.js';
import type { LValue } from './expressions.js';

/** One-use value folded inline; multi-use values must be materialized first. */
function use(v: Value): string {
  return v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v;
}

/** Same-shape scalar/vector check (for implicit numeric widening). */
function sameShape(a: GLSLType, b: GLSLType): boolean {
  if (a.kind === 'scalar' && b.kind === 'scalar') return true;
  if (a.kind === 'vector' && b.kind === 'vector') return a.size === b.size;
  return false;
}

/** Does `param` accept `arg`? Exact match or an ES-legal implicit widening. */
function paramAccepts(param: GLSLType, arg: GLSLType, version: 100 | 300): boolean {
  if (typeEquals(param, arg)) return true;
  const pb = scalarBaseOf(param);
  const ab = scalarBaseOf(arg);
  if (pb === null || ab === null || !sameShape(param, arg)) return false;
  if (pb === 'float') return ab === 'int' || (version === 300 && ab === 'uint');
  if (pb === 'uint' && version === 300) return ab === 'int';
  return false;
}

/** Resolve the builtin signature for a call (first table match wins). */
function resolveSignature(name: string, argTypes: GLSLType[], env: CodegenEnv): BuiltinSignature | null {
  const cands = [...matches(name, builtinSignatures(env.layout.version)), ...matches(name, extensionFunctions)];
  for (const s of cands) {
    if (s.params.length !== argTypes.length) continue;
    let ok = true;
    for (let i = 0; i < s.params.length; i++) {
      if (!paramAccepts(s.params[i], argTypes[i], env.layout.version)) {
        ok = false;
        break;
      }
    }
    if (ok) return s;
  }
  return null;
}

/** Convert emitted arg components to the signature param's base (no-op for equal bases). */
function convertToParam(vals: Value[], argT: GLSLType, paramT: GLSLType): Value[] {
  const fb = scalarBaseOf(argT);
  const tb = scalarBaseOf(paramT);
  if (fb === null || tb === null || fb === tb) return vals;
  return vals.map((v) => ({ v: convertScalar(v.v, fb, tb), pre: v.pre }));
}

/**
 * Lower a builtin call. `argVals[i]` = the flat components of argument i
 * (already converted to the matched signature's parameter bases).
 */
export function emitBuiltinCall(e: CallExpr, env: CodegenEnv): Value[] {
  const callee = e.callee;
  if (callee.kind !== 'identifier') throw new Error('codegen: builtin call with a non-identifier callee');
  const name = callee.name;
  const argTypes = e.args.map((a) => a.resolvedType!);
  const sig = resolveSignature(name, argTypes, env);
  if (!sig) throw new Error(`codegen: no builtin signature for '${name}'`);
  if (sig.stage && sig.stage !== env.stage) {
    throw new Error(`codegen: '${name}' is restricted to the ${sig.stage} stage (current stage: ${env.stage})`);
  }
  const argVals = e.args.map((a, i) => convertToParam(emitExpr(a, env), argTypes[i], sig.params[i]));
  return lowerBuiltin(name, sig, e.args, argVals, argTypes, env);
}

const DERIV_NAMES = new Set(['dFdx', 'dFdy', 'fwidth']);

/**
 * dFdx / dFdy / fwidth (dual mode only — the non-dual path throws a clear
 * error instead of emitting garbage).
 *   dFdx(x)  → { v: x.dx, dx: '0', dy: '0' }
 *   dFdy(x)  → { v: x.dy, dx: '0', dy: '0' }
 *   fwidth(x)→ { v: Math.abs(x.dx) + Math.abs(x.dy), dx: '0', dy: '0' }
 * The RESULT is itself a float value whose own duals are 0 (second
 * derivatives are not tracked). The operand's `pre` is attached (shared
 * array — statement emitters dedupe by identity) so materialized temps the
 * dx/dy strings reference are set before `v` evaluates.
 */
function emitDerivatives(name: string, sig: BuiltinSignature, argVals: Value[][], env: CodegenEnv): Value[] {
  if (!env.dual) {
    throw new Error(`codegen: '${name}' requires dual-number mode (C5); non-dual lowering is unsupported`);
  }
  const x = argVals[0];
  const out: Value[] = [];
  for (let c = 0; c < x.length; c++) {
    const xv = x[c];
    if (name === 'dFdx') out.push({ v: `(${xv.dx ?? '0'})`, dx: '0', dy: '0', pre: xv.pre });
    else if (name === 'dFdy') out.push({ v: `(${xv.dy ?? '0'})`, dx: '0', dy: '0', pre: xv.pre });
    else out.push({ v: `(Math.abs(${xv.dx ?? '0'}) + Math.abs(${xv.dy ?? '0'}))`, dx: '0', dy: '0', pre: xv.pre });
  }
  return out;
}

function lowerBuiltin(
  name: string,
  sig: BuiltinSignature,
  args: Expr[],
  argVals: Value[][],
  argTypes: GLSLType[],
  env: CodegenEnv,
): Value[] {
  if (DERIV_NAMES.has(name)) {
    return emitDerivatives(name, sig, argVals, env);
  }
  // Dual mode: float-result builtins need dual lowering (per-builtin
  // derivative templates — C5a2). Throw rather than silently dropping the
  // duals. Int/uint/bool-result builtins (textureSize, bitfield ops,
  // relationals, floatBitsToInt, ...) carry no duals and stay legal.
  if (env.dual && hasFloatLeaves(sig.ret)) {
    throw new Error(`codegen: dual-mode lowering of '${name}' requires the C5a2 builtin dual templates`);
  }
  const ret = sig.ret;
  const n = flatComponents(ret);
  const argN = argTypes.map((t) => flatComponents(t));
  if (name.startsWith('texture') || name.startsWith('texelFetch')) {
    return emitTextureCall(name, sig, args, argVals, argTypes, argN, ret, env);
  }
  switch (name) {
    /* ---------------- §8.3 common functions ---------------- */
    case 'radians':
      return perComp(n, argVals, argTypes, ([x]) => `(${x} * 0.017453292519943295)`);
    case 'degrees':
      return perComp(n, argVals, argTypes, ([x]) => `(${x} * 57.29577951308232)`);
    case 'sin':
    case 'cos':
    case 'tan':
    case 'asin':
    case 'acos':
    case 'sinh':
    case 'cosh':
    case 'tanh':
    case 'asinh':
    case 'acosh':
    case 'atanh':
    case 'exp':
    case 'log':
    case 'log2':
    case 'sqrt':
      return perComp(n, argVals, argTypes, ([x]) => `Math.${name}(${x})`);
    case 'exp2':
      return perComp(n, argVals, argTypes, ([x]) => `Math.pow(2, ${x})`);
    case 'inversesqrt':
      return perComp(n, argVals, argTypes, ([x]) => `(1 / Math.sqrt(${x}))`);
    case 'atan': {
      if (argVals.length === 1) return perComp(n, argVals, argTypes, ([x]) => `Math.atan(${x})`);
      return perComp(n, argVals, argTypes, ([y, x]) => `Math.atan2(${y}, ${x})`);
    }
    case 'pow':
      return perComp(n, argVals, argTypes, ([x, y]) => `Math.pow(${x}, ${y})`);
    case 'abs':
      return perComp(n, argVals, argTypes, ([x]) => `Math.abs(${x})`);
    case 'sign':
      return perComp(n, argVals, argTypes, ([x]) => `((${x}) > 0 ? 1 : ((${x}) < 0 ? -1 : 0))`);
    case 'floor':
      return perComp(n, argVals, argTypes, ([x]) => `Math.floor(${x})`);
    case 'ceil':
      return perComp(n, argVals, argTypes, ([x]) => `Math.ceil(${x})`);
    case 'trunc':
      return perComp(n, argVals, argTypes, ([x]) => `Math.trunc(${x})`);
    case 'fract':
      return perComp(n, argVals, argTypes, ([x]) => `((${x}) - Math.floor(${x}))`);
    case 'round':
      return perComp(n, argVals, argTypes, ([x]) => `Math.round(${x})`);
    case 'roundEven': {
      const x = materialize(argVals[0], env);
      const t = env.allocTemp();
      return perComp(n, [x], argTypes, ([xv]) => {
        const r = `${t} = Math.round(${xv})`;
        const tie = `(Math.abs(${xv} - Math.trunc(${xv})) === 0.5 && (${t} & 1))`;
        return `(${r}, ${tie} ? ${t} - 1 : ${t})`;
      });
    }
    case 'mod': {
      const x = materialize(argVals[0], env);
      const y = materialize(argVals[1], env);
      return perComp(n, [x, y], argTypes, ([xv, yv]) => `((${xv}) - (${yv}) * Math.floor((${xv}) / (${yv})))`);
    }
    case 'min':
      return perComp(n, argVals, argTypes, ([x, y]) => `Math.min(${x}, ${y})`);
    case 'max':
      return perComp(n, argVals, argTypes, ([x, y]) => `Math.max(${x}, ${y})`);
    case 'clamp': {
      const x = materialize(argVals[0], env);
      return perComp(n, [x, argVals[1], argVals[2]], argTypes, ([xv, lo, hi]) => `Math.min(Math.max(${xv}, ${lo}), ${hi})`);
    }
    case 'mix': {
      const a = materialize(argVals[2], env);
      if (isBoolType(argTypes[2])) {
        return perComp(n, [argVals[0], argVals[1], a], argTypes, ([x, y, s]) => `(${s} ? ${y} : ${x})`);
      }
      return perComp(n, [argVals[0], argVals[1], a], argTypes, ([x, y, s]) => `((${x}) * (1 - (${s})) + (${y}) * (${s}))`);
    }
    case 'step':
      // Table params are (edge, x): result is 0 when x < edge, else 1.
      return perComp(n, argVals, argTypes, ([edge, x]) => `((${x}) < (${edge}) ? 0.0 : 1.0)`);
    case 'smoothstep': {
      const e0 = materialize(argVals[0], env);
      const e1 = materialize(argVals[1], env);
      const x = materialize(argVals[2], env);
      const t = env.allocTemp();
      return perComp(n, [e0, e1, x], argTypes, ([a, b, c]) => {
        const clamp = `${t} = Math.min(Math.max(((${c}) - (${a})) / ((${b}) - (${a})), 0), 1)`;
        return `(${clamp}, ${t} * ${t} * (3 - 2 * ${t}))`;
      });
    }
    case 'isnan':
      return perComp(n, argVals, argTypes, ([x]) => `((${x}) !== (${x}))`);
    case 'isinf':
      return perComp(n, argVals, argTypes, ([x]) => `((${x}) === Infinity || (${x}) === -Infinity)`);
    case 'floatBitsToInt':
      return perComp(n, argVals, argTypes, ([x]) => `R.f2i(${x})`);
    case 'floatBitsToUint':
      return perComp(n, argVals, argTypes, ([x]) => wrapUint(`R.f2u(${x})`));
    case 'intBitsToFloat':
      return perComp(n, argVals, argTypes, ([x]) => `R.i2f(${x})`);
    case 'uintBitsToFloat':
      return perComp(n, argVals, argTypes, ([x]) => `R.u2f(${x})`);

    /* ---------------- §8.5 geometry ---------------- */
    case 'length': {
      const v = argVals[0];
      const parts: string[] = [];
      for (let c = 0; c < argN[0]; c++) {
        const x = use(v[c]);
        parts.push(`(${x} * ${x})`);
      }
      return [{ v: `Math.sqrt(${parts.join(' + ')})` }];
    }
    case 'distance': {
      const a = materialize(argVals[0], env);
      const b = materialize(argVals[1], env);
      const parts: string[] = [];
      for (let c = 0; c < argN[0]; c++) {
        const d = `((${a[c].v}) - (${b[c].v}))`;
        parts.push(`(${d} * ${d})`);
      }
      return [{ v: `Math.sqrt(${parts.join(' + ')})` }];
    }
    case 'dot': {
      const parts: string[] = [];
      for (let c = 0; c < argN[0]; c++) {
        parts.push(`(${use(argVals[0][c])} * ${use(argVals[1][c])})`);
      }
      return [{ v: `(${parts.join(' + ')})` }];
    }
    case 'cross': {
      const a = argVals[0];
      const b = argVals[1];
      const x0 = use(a[0]);
      const x1 = use(a[1]);
      const x2 = use(a[2]);
      const y0 = use(b[0]);
      const y1 = use(b[1]);
      const y2 = use(b[2]);
      return [
        { v: `((${x1}) * (${y2}) - (${x2}) * (${y1}))` },
        { v: `((${x2}) * (${y0}) - (${x0}) * (${y2}))` },
        { v: `((${x0}) * (${y1}) - (${x1}) * (${y0}))` },
      ];
    }
    case 'normalize': {
      const v = materialize(argVals[0], env);
      const t = env.allocTemp();
      const parts: string[] = [];
      for (let c = 0; c < argN[0]; c++) parts.push(`(${v[c].v} * ${v[c].v})`);
      const pre = [`${t} = Math.sqrt(${parts.join(' + ')})`];
      const out: Value[] = [];
      for (let c = 0; c < argN[0]; c++) out.push({ v: `((${v[c].v}) / ${t})`, pre });
      return out;
    }
    case 'faceforward': {
      const N = materialize(argVals[0], env);
      const t = env.allocTemp();
      const parts: string[] = [];
      for (let c = 0; c < argN[0]; c++) {
        parts.push(`(${use(argVals[1][c])} * ${use(argVals[2][c])})`);
      }
      const pre = [`${t} = (${parts.join(' + ')})`];
      const out: Value[] = [];
      for (let c = 0; c < argN[0]; c++) out.push({ v: `((${t}) < 0 ? (${N[c].v}) : (-(${N[c].v})))`, pre });
      return out;
    }
    case 'reflect': {
      const t = env.allocTemp();
      const parts: string[] = [];
      for (let c = 0; c < argN[0]; c++) {
        parts.push(`(${use(argVals[1][c])} * ${use(argVals[0][c])})`); // dot(N, I)
      }
      const pre = [`${t} = (${parts.join(' + ')})`];
      const out: Value[] = [];
      for (let c = 0; c < argN[0]; c++) {
        out.push({ v: `((${use(argVals[0][c])}) - 2 * (${t}) * (${use(argVals[1][c])}))`, pre });
      }
      return out;
    }
    case 'refract': {
      const eta = materialize(argVals[2], env)[0];
      const td = env.allocTemp();
      const tk = env.allocTemp();
      const parts: string[] = [];
      for (let c = 0; c < argN[0]; c++) {
        parts.push(`(${use(argVals[1][c])} * ${use(argVals[0][c])})`); // dot(N, I)
      }
      const pre = [
        `${td} = (${parts.join(' + ')})`,
        `${tk} = (1 - (${eta.v}) * (${eta.v}) * (1 - (${td}) * (${td})))`,
      ];
      const out: Value[] = [];
      for (let c = 0; c < argN[0]; c++) {
        out.push({
          v: `(((${tk}) < 0) ? 0 : (${eta.v}) * (${use(argVals[0][c])}) - (((${eta.v}) * (${td}) + Math.sqrt(${tk})) * (${use(argVals[1][c])})))`,
          pre,
        });
      }
      return out;
    }

    /* ---------------- §8.6 matrix ---------------- */
    case 'matrixCompMult':
      return perComp(n, argVals, argTypes, ([a, b]) => `(${a} * ${b})`);
    case 'outerProduct': {
      if (ret.kind !== 'matrix') throw new Error('codegen: outerProduct must return a matrix');
      const cv = materialize(argVals[0], env); // vec(cols)
      const rv = materialize(argVals[1], env); // vec(rows)
      const out: Value[] = [];
      for (let c = 0; c < ret.cols; c++) {
        for (let r = 0; r < ret.rows; r++) out.push({ v: `(${cv[c].v} * ${rv[r].v})` });
      }
      return out;
    }
    case 'transpose': {
      if (argTypes[0].kind !== 'matrix') throw new Error('codegen: transpose requires a matrix');
      const srcCols = argTypes[0].cols;
      const srcRows = argTypes[0].rows;
      const m = argVals[0];
      const out: Value[] = [];
      for (let c = 0; c < srcRows; c++) {
        for (let r = 0; r < srcCols; r++) out.push({ v: use(m[r * srcCols + c]) });
      }
      return out;
    }
    case 'determinant': {
      if (argTypes[0].kind !== 'matrix') throw new Error('codegen: determinant requires a matrix');
      const m = materialize(argVals[0], env);
      const cols = argTypes[0].cols;
      if (cols === 2) {
        return [{ v: `((${m[0].v}) * (${m[3].v}) - (${m[1].v}) * (${m[2].v}))` }];
      }
      if (cols === 3) {
        return [{ v: det3s(m[0].v, m[1].v, m[2].v, m[3].v, m[4].v, m[5].v, m[6].v, m[7].v, m[8].v) }];
      }
      return [{ v: det4s(m) }];
    }
    case 'inverse': {
      if (argTypes[0].kind !== 'matrix') throw new Error('codegen: inverse requires a matrix');
      const cols = argTypes[0].cols;
      const m = argVals[0];
      const b = env.allocScratch(cols * cols);
      const flat: string[] = [];
      for (let i = 0; i < cols * cols; i++) flat.push(use(m[i]));
      const pre = [`R.inv${cols}(ctx.scratch, ${b}, ${flat.join(', ')})`];
      const out: Value[] = [];
      for (let i = 0; i < cols * cols; i++) out.push({ v: `ctx.scratch[${b} + ${i}]`, pre });
      return out;
    }

    /* ---------------- §8.4 pack / unpack ---------------- */
    case 'packSnorm2x16':
    case 'packUnorm2x16':
    case 'packHalf2x16':
      // Table param is a single vec2 (both components packed).
      return [{ v: wrapUint(`R.${name}(${use(argVals[0][0])}, ${use(argVals[0][1])})`) }];
    case 'packUnorm4x8':
    case 'packSnorm4x8':
      // Table param is a single vec4 (all four components packed).
      return [
        {
          v: wrapUint(
            `R.${name}(${use(argVals[0][0])}, ${use(argVals[0][1])}, ${use(argVals[0][2])}, ${use(argVals[0][3])})`,
          ),
        },
      ];
    case 'unpackSnorm2x16':
    case 'unpackUnorm2x16':
    case 'unpackHalf2x16': {
      const b = env.allocScratch(2);
      const pre = [`R.${name}(${use(argVals[0][0])}, ctx.scratch, ${b})`];
      return [
        { v: `ctx.scratch[${b} + 0]`, pre },
        { v: `ctx.scratch[${b} + 1]`, pre },
      ];
    }
    case 'unpackUnorm4x8':
    case 'unpackSnorm4x8': {
      const b = env.allocScratch(4);
      const pre = [`R.${name}(${use(argVals[0][0])}, ctx.scratch, ${b})`];
      const out: Value[] = [];
      for (let i = 0; i < 4; i++) out.push({ v: `ctx.scratch[${b} + ${i}]`, pre });
      return out;
    }

    /* ---------------- §8.10 bitfield ops ---------------- */
    case 'bitfieldExtract': {
      const isU = scalarBaseOf(argTypes[0]) === 'uint';
      const fn = isU ? 'R.bitfieldExtractU' : 'R.bitfieldExtractI';
      const o = use(argVals[1][0]);
      const b = use(argVals[2][0]);
      return perComp(n, argVals, argTypes, ([v]) => {
        const s = `${fn}(${v}, ${o}, ${b})`;
        return isU ? wrapUint(s) : s;
      });
    }
    case 'bitfieldInsert': {
      const isU = scalarBaseOf(ret) === 'uint';
      const o = use(argVals[2][0]);
      const b = use(argVals[3][0]);
      return perComp(n, argVals, argTypes, ([base, ins]) => {
        const s = `R.bitfieldInsert(${base}, ${ins}, ${o}, ${b})`;
        return isU ? wrapUint(s) : wrapInt(s);
      });
    }
    case 'bitfieldReverse': {
      const isU = scalarBaseOf(ret) === 'uint';
      return perComp(n, argVals, argTypes, ([v]) => {
        const s = `R.bitfieldReverse(${v})`;
        return isU ? wrapUint(s) : wrapInt(s);
      });
    }
    case 'bitCount':
      return perComp(n, argVals, argTypes, ([v]) => `R.bitCount(${v})`);
    case 'findLSB':
      return perComp(n, argVals, argTypes, ([v]) => `R.findLSB(${v})`);
    case 'findMSB': {
      const isU = scalarBaseOf(argTypes[0]) === 'uint';
      return perComp(n, argVals, argTypes, ([v]) => (isU ? `R.findMSB(${v})` : `R.findMSBI(${v})`));
    }

    /* ---------------- §8.11 out-param extended ops ---------------- */
    case 'uaddCarry':
    case 'usubBorrow': {
      const nc = argN[0];
      const x = materialize(argVals[0], env);
      const y = materialize(argVals[1], env);
      const outLv = emitLValue(args[2], env);
      const b = env.allocIntScratch(2 * nc);
      const pre: string[] = [];
      for (let c = 0; c < nc; c++) {
        pre.push(`R.${name}(${x[c].v}, ${y[c].v}, ctx.intScratch, ${b + 2 * c})`);
        pre.push(lvWrite(outLv, c, `ctx.intScratch[${b + 2 * c + 1}]`));
      }
      const out: Value[] = [];
      for (let c = 0; c < nc; c++) out.push({ v: wrapUint(`ctx.intScratch[${b + 2 * c}]`), pre });
      return out;
    }
    case 'umulExtended':
    case 'imulExtended': {
      const nc = argN[0];
      const isU = name === 'umulExtended';
      const x = materialize(argVals[0], env);
      const y = materialize(argVals[1], env);
      const o1 = emitLValue(args[2], env);
      const o2 = emitLValue(args[3], env);
      const b = env.allocIntScratch(2 * nc);
      const parts: string[] = [];
      for (let c = 0; c < nc; c++) {
        parts.push(`R.${name}(${x[c].v}, ${y[c].v}, ctx.intScratch, ${b + 2 * c})`);
        parts.push(lvWrite(o1, c, isU ? wrapUint(`ctx.intScratch[${b + 2 * c}]`) : `ctx.intScratch[${b + 2 * c}]`));
        parts.push(lvWrite(o2, c, wrapUint(`ctx.intScratch[${b + 2 * c + 1}]`)));
      }
      return [{ v: `(${parts.join(', ')})` }];
    }
    case 'modf': {
      const nc = argN[0];
      const x = materialize(argVals[0], env);
      const outLv = emitLValue(args[1], env);
      const pre: string[] = [];
      const ts: string[] = [];
      for (let c = 0; c < nc; c++) {
        const t = env.allocTemp();
        ts.push(t);
        pre.push(`${t} = Math.trunc(${x[c].v})`);
        pre.push(lvWrite(outLv, c, t));
      }
      const out: Value[] = [];
      for (let c = 0; c < nc; c++) out.push({ v: `((${x[c].v}) - ${ts[c]})`, pre });
      return out;
    }

    /* ---------------- relational ---------------- */
    case 'lessThan':
    case 'lessThanEqual':
    case 'greaterThan':
    case 'greaterThanEqual':
    case 'equal':
    case 'notEqual': {
      const op: Record<string, string> = {
        lessThan: '<',
        lessThanEqual: '<=',
        greaterThan: '>',
        greaterThanEqual: '>=',
        equal: '===',
        notEqual: '!==',
      };
      return perComp(n, argVals, argTypes, ([a, b]) => `(${a} ${op[name]} (${b}))`);
    }
    case 'any': {
      const parts = argVals[0].map((v) => use(v));
      return [{ v: `(${parts.join(' || ')})` }];
    }
    case 'all': {
      const parts = argVals[0].map((v) => use(v));
      return [{ v: `(${parts.join(' && ')})` }];
    }
    case 'not':
      return perComp(n, argVals, argTypes, ([a]) => `(!(${a}))`);

    default:
      throw new Error(`codegen: builtin '${name}' not lowered`);
  }
}

/* ------------------------------------------------------------------ */
/* Per-component helpers                                               */
/* ------------------------------------------------------------------ */

/** Value of arg i's component c (scalars broadcast). */
function comp(vals: Value[], t: GLSLType, c: number): Value {
  return vals[t.kind === 'scalar' ? 0 : c];
}

/** Component-wise lowering: f receives one string per arg for result component c. */
function perComp(
  n: number,
  argVals: Value[][],
  argTypes: GLSLType[],
  f: (xs: string[], c: number) => string,
): Value[] {
  const out: Value[] = [];
  for (let c = 0; c < n; c++) {
    const xs = argVals.map((vals, i) => use(comp(vals, argTypes[i], c)));
    out.push({ v: f(xs, c) });
  }
  return out;
}

/** 3×3 determinant over column-major flat components (row-0 cofactor expansion). */
function det3s(
  x0: string, x1: string, x2: string,
  x3: string, x4: string, x5: string,
  x6: string, x7: string, x8: string,
): string {
  return (
    `((${x0}) * ((${x4}) * (${x8}) - (${x5}) * (${x7}))` +
    ` - (${x1}) * ((${x3}) * (${x8}) - (${x5}) * (${x6}))` +
    ` + (${x2}) * ((${x3}) * (${x7}) - (${x4}) * (${x6})))`
  );
}

/** 4×4 determinant over column-major flat components (row-0 cofactor expansion). */
function det4s(m: Value[]): string {
  const M00 = det3s(m[5].v, m[6].v, m[7].v, m[9].v, m[10].v, m[11].v, m[13].v, m[14].v, m[15].v);
  const M01 = det3s(m[1].v, m[2].v, m[3].v, m[9].v, m[10].v, m[11].v, m[13].v, m[14].v, m[15].v);
  const M02 = det3s(m[1].v, m[2].v, m[3].v, m[5].v, m[6].v, m[7].v, m[13].v, m[14].v, m[15].v);
  const M03 = det3s(m[1].v, m[2].v, m[3].v, m[5].v, m[6].v, m[7].v, m[9].v, m[10].v, m[11].v);
  return `((${m[0].v}) * ${M00} - (${m[1].v}) * ${M01} + (${m[2].v}) * ${M02} - (${m[3].v}) * ${M03})`;
}

/* ------------------------------------------------------------------ */
/* Out-param lvalue writes                                            */
/* ------------------------------------------------------------------ */

/**
 * Build a comma-expression string performing ONE component write to an lvalue
 * (its prelude/copy-back statement lists are converted to comma expressions).
 */
function lvWrite(lv: LValue, c: number, value: string): string {
  const parts: string[] = [];
  if (lv.prelude) parts.push(stmtToExpr(lv.prelude));
  parts.push(`${lv.targets[c]} = ${value}`);
  if (lv.copyBack) parts.push(stmtToExpr(lv.copyBack));
  return `(${parts.join(', ')})`;
}

/** 'a = 1; b = 2;' → 'a = 1, b = 2' (statement list → comma expression). */
function stmtToExpr(s: string): string {
  return s.replace(/;\s*$/g, '').replace(/;\s+/g, ', ');
}

/* ------------------------------------------------------------------ */
/* Texture lookups (1.00 §8.7, EXT_shader_texture_lod, 3.00 §8.8–8.9) */
/* ------------------------------------------------------------------ */

const IMPLICIT_LOD_COMMENT = '/* implicit-LOD approximated as lod 0 (dual mode pending) */';
const OFFSET_COMMENT = '/* offset approximated (no exact helper) */';

function emitTextureCall(
  name: string,
  sig: BuiltinSignature,
  args: Expr[],
  argVals: Value[][],
  argTypes: GLSLType[],
  argN: number[],
  ret: GLSLType,
  env: CodegenEnv,
): Value[] {
  const samplerT = argTypes[0];
  if (samplerT.kind !== 'sampler') throw new Error(`codegen: '${name}': first argument must be a sampler`);
  const kind = samplerT.sampler;
  const isInt = kind.startsWith('isampler') || kind.startsWith('usampler');
  /** Sampler family without the i/u integer prefix ('sampler2D', 'sampler3D', ...). */
  const fam = isInt ? kind.slice(1) : kind;
  const isShadow = fam.endsWith('Shadow');
  const unit = use(argVals[0][0]);
  const P = argVals[1];
  const pc = (i: number): string => use(P[i]);
  const coordCount = (f: string): number => (f === 'sampler2D' ? 2 : 3);

  const is2D = fam === 'sampler2D';
  const is3D = fam === 'sampler3D';
  const is2DArray = fam === 'sampler2DArray';

  /** Result read from the fragment ctx.tex scratch (float/int/uint by kind). */
  const ctxRead = (i: number): string => {
    if (kind.startsWith('isampler')) return `ctx.tex.outInt[${i}]`;
    if (kind.startsWith('usampler')) return wrapUint(`ctx.tex.outUint[${i}]`);
    return `ctx.tex.out[${i}]`;
  };

  /** Result Values from a fragment ctx.tex call (the call string is the shared pre). */
  const fromCtxTex = (call: string, comps: number): Value[] => {
    const pre = [call];
    const out: Value[] = [];
    for (let i = 0; i < comps; i++) out.push({ v: ctxRead(i), pre });
    return out;
  };

  /**
   * Result Values from an R helper: `R.<helper>(ctx, <inner>, <store>, <off>)`.
   * Integer samplers write into ctx.intScratch (bit-exact); float samplers use
   * ctx.scratch. Shadow helpers produce a single compare component.
   */
  const fromR = (helper: string, inner: string, comps = 4, intStore?: boolean): Value[] => {
    const store = intStore ?? isInt ? 'ctx.intScratch' : 'ctx.scratch';
    const b = intStore ?? isInt ? env.allocIntScratch(4) : env.allocScratch(4);
    const pre = [`R.${helper}(ctx, ${inner}, ${store}, ${b})`];
    const out: Value[] = [];
    for (let i = 0; i < comps; i++) {
      const s = `${store}[${b} + ${i}]`;
      out.push({ v: kind.startsWith('usampler') ? wrapUint(s) : s, pre });
    }
    return out;
  };

  /** Implicit-LOD sample (texture/texture2D family): fragment via ctx.tex, vertex via R (LOD 0). */
  const implicit = (coords: string[], bias: string | null, ref?: string): Value[] => {
    const b = bias ?? '0';
    const c = IMPLICIT_LOD_COMMENT + ' ';
    if (env.stage === 'FRAGMENT') {
      switch (fam) {
        case 'sampler2D':
          return fromCtxTex(`${c}ctx.tex.sample2D(${unit}, ${coords[0]}, ${coords[1]}, 0, 0, 0, 0, ${b})`, 4);
        case 'sampler3D':
          return fromCtxTex(
            `${c}ctx.tex.sample3D(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, 0, 0, 0, 0, 0, 0, ${b})`,
            4,
          );
        case 'samplerCube':
          return fromCtxTex(
            `${c}ctx.tex.sampleCube(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, 0, 0, 0, 0, 0, 0, ${b})`,
            4,
          );
        case 'sampler2DArray':
          return fromCtxTex(`${c}ctx.tex.sample2DArray(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, 0, 0, 0, 0, ${b})`, 4);
        case 'sampler2DShadow':
          return fromCtxTex(`${c}ctx.tex.sample2DShadow(${unit}, ${coords[0]}, ${coords[1]}, ${ref ?? '0'}, 0, 0, 0, 0, ${b})`, 1);
        case 'samplerCubeShadow':
          return fromCtxTex(
            `${c}ctx.tex.sampleCubeShadow(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${ref ?? '0'}, 0, 0, 0, 0, 0, 0, ${b})`,
            1,
          );
        case 'sampler2DArrayShadow':
          return fromCtxTex(
            `${c}ctx.tex.sample2DArrayShadow(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${ref ?? '0'}, 0, 0, 0, 0, ${b})`,
            1,
          );
        default:
          throw new Error(`codegen: cannot implicitly sample '${kind}'`);
      }
    }
    // VERTEX: explicit LOD 0 (GLSL ES: vertex texture fetches use LOD 0).
    switch (fam) {
      case 'sampler2D':
        return fromR('tex2DLod', `${unit}, ${coords[0]}, ${coords[1]}, ${b}`);
      case 'sampler3D':
        return fromR('tex3DLod', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${b}`);
      case 'samplerCube':
        return fromR('texCubeLod', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${b}`);
      case 'sampler2DArray':
        return fromR('tex2DArrayLod', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${b}`);
      case 'sampler2DShadow':
        return fromR('sampleShadowLod', `${unit}, ${coords[0]}, ${coords[1]}, ${ref ?? '0'}, ${b}`, 1);
      case 'samplerCubeShadow':
        // No cube-shadow-LOD helper: zero-gradient shadow grad sample (LOD 0).
        return fromR('texCubeShadowGrad', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${ref ?? '0'}, 0, 0, 0, 0, 0, 0`, 1);
      case 'sampler2DArrayShadow':
        return fromR('tex2DArrayShadowGrad', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${ref ?? '0'}, 0, 0, 0, 0`, 1);
      default:
        throw new Error(`codegen: cannot implicitly sample '${kind}'`);
    }
  };

  /** Explicit-LOD sample: fragment (non-shadow) via ctx.tex, otherwise R. */
  const lodSample = (coords: string[], lod: string, ref?: string): Value[] => {
    if (env.stage === 'FRAGMENT' && !isShadow) {
      switch (fam) {
        case 'sampler2D':
          return fromCtxTex(`ctx.tex.sample2DLod(${unit}, ${coords[0]}, ${coords[1]}, ${lod})`, 4);
        case 'sampler3D':
          return fromCtxTex(`ctx.tex.sample3DLod(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${lod})`, 4);
        case 'samplerCube':
          return fromCtxTex(`ctx.tex.sampleCubeLod(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${lod})`, 4);
        case 'sampler2DArray':
          return fromCtxTex(`ctx.tex.sample2DArrayLod(${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${lod})`, 4);
        default:
          throw new Error(`codegen: cannot LOD-sample '${kind}'`);
      }
    }
    switch (fam) {
      case 'sampler2D':
        return fromR('tex2DLod', `${unit}, ${coords[0]}, ${coords[1]}, ${lod}`);
      case 'sampler3D':
        return fromR('tex3DLod', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${lod}`);
      case 'samplerCube':
        return fromR('texCubeLod', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${lod}`);
      case 'sampler2DArray':
        return fromR('tex2DArrayLod', `${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${lod}`);
      case 'sampler2DShadow':
        return fromR('sampleShadowLod', `${unit}, ${coords[0]}, ${coords[1]}, ${ref ?? '0'}, ${lod}`, 1);
      default:
        throw new Error(`codegen: cannot LOD-sample '${kind}'`);
    }
  };

  /** textureGrad — gradients flat: [dudx, dvdx(, dwdx), dudy, dvdy(, dwdy)]. */
  const gradSample = (coords: string[], g: string[], ref?: string, comment?: string): Value[] => {
    const c = comment ? comment + ' ' : '';
    switch (fam) {
      case 'sampler2D':
        return fromR('tex2DGrad', `${c}${unit}, ${coords[0]}, ${coords[1]}, ${g[0]}, ${g[1]}, ${g[2]}, ${g[3]}`);
      case 'sampler2DArray':
        return fromR('tex2DArrayGrad', `${c}${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${g[0]}, ${g[1]}, ${g[2]}, ${g[3]}`);
      case 'sampler3D':
        return fromR('tex3DGrad', `${c}${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${g[0]}, ${g[1]}, ${g[2]}, ${g[3]}, ${g[4]}, ${g[5]}`);
      case 'samplerCube':
        return fromR('texCubeGrad', `${c}${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${g[0]}, ${g[1]}, ${g[2]}, ${g[3]}, ${g[4]}, ${g[5]}`);
      case 'sampler2DShadow':
        return fromR('tex2DShadowGrad', `${c}${unit}, ${coords[0]}, ${coords[1]}, ${ref ?? '0'}, ${g[0]}, ${g[1]}, ${g[2]}, ${g[3]}`, 1);
      case 'samplerCubeShadow':
        return fromR('texCubeShadowGrad', `${c}${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${ref ?? '0'}, ${g[0]}, ${g[1]}, ${g[2]}, ${g[3]}, ${g[4]}, ${g[5]}`, 1);
      case 'sampler2DArrayShadow':
        return fromR('tex2DArrayShadowGrad', `${c}${unit}, ${coords[0]}, ${coords[1]}, ${coords[2]}, ${ref ?? '0'}, ${g[0]}, ${g[1]}, ${g[2]}, ${g[3]}`, 1);
      default:
        throw new Error(`codegen: cannot gradient-sample '${kind}'`);
    }
  };

  /** Flat gradient components of the dPdx/dPdy args (argIdx = index of dPdx). */
  const gradComps = (argIdx: number): string[] => {
    const dx = argVals[argIdx];
    const dy = argVals[argIdx + 1];
    const out: string[] = [];
    for (let i = 0; i < dx.length; i++) out.push(use(dx[i]));
    for (let i = 0; i < dy.length; i++) out.push(use(dy[i]));
    return out;
  };

  /** P component after perspective divide by q (q used twice — pure, folded inline). */
  const divP = (i: number, q: string): string => `((${pc(i)}) / (${q}))`;

  switch (name) {
    /* ---------------- 1.00 core ---------------- */
    case 'texture2D': {
      const bias = args.length > 2 ? use(argVals[2][0]) : null;
      return implicit([pc(0), pc(1)], bias);
    }
    case 'texture2DProj': {
      const bias = args.length > 2 ? use(argVals[2][0]) : null;
      const q = argN[1] === 3 ? pc(2) : pc(3);
      if (env.stage === 'FRAGMENT') return implicit([divP(0, q), divP(1, q)], bias);
      return fromR('tex2DProjLod', `${unit}, ${pc(0)}, ${pc(1)}, ${q}, ${bias ?? '0'}`);
    }
    case 'texture2DLod':
      return fromR('tex2DLod', `${unit}, ${pc(0)}, ${pc(1)}, ${use(argVals[2][0])}`);
    case 'texture2DProjLod': {
      const q = argN[1] === 3 ? pc(2) : pc(3);
      return fromR('tex2DProjLod', `${unit}, ${pc(0)}, ${pc(1)}, ${q}, ${use(argVals[2][0])}`);
    }
    case 'textureCube': {
      const bias = args.length > 2 ? use(argVals[2][0]) : null;
      return implicit([pc(0), pc(1), pc(2)], bias);
    }
    case 'textureCubeLod':
      return fromR('texCubeLod', `${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${use(argVals[2][0])}`);

    /* ---------------- GL_EXT_shader_texture_lod (fragment) ---------------- */
    case 'texture2DLodEXT':
      return fromCtxTex(`ctx.tex.sample2DLod(${unit}, ${pc(0)}, ${pc(1)}, ${use(argVals[2][0])})`, 4);
    case 'texture2DProjLodEXT': {
      const q = argN[1] === 3 ? pc(2) : pc(3);
      return fromCtxTex(`ctx.tex.sample2DLod(${unit}, ${divP(0, q)}, ${divP(1, q)}, ${use(argVals[2][0])})`, 4);
    }
    case 'textureCubeLodEXT':
      return fromCtxTex(`ctx.tex.sampleCubeLod(${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${use(argVals[2][0])})`, 4);
    case 'texture2DGradEXT':
      return gradSample([pc(0), pc(1)], gradComps(2));
    case 'texture2DProjGradEXT': {
      const q = argN[1] === 3 ? pc(2) : pc(3);
      return gradSample([divP(0, q), divP(1, q)], gradComps(2));
    }
    case 'textureCubeGradEXT':
      return gradSample([pc(0), pc(1), pc(2)], gradComps(2));
    case 'texture2DShadowLodEXT': {
      // EXT declares a vec4 result (compare in .x per the shadow2D convention);
      // the compare is single-channel — pad the remaining components.
      const r = fromR('sampleShadowLod', `${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${use(argVals[2][0])}`, 1);
      return [r[0], { v: '0.0', pre: r[0].pre }, { v: '0.0', pre: r[0].pre }, { v: '1.0', pre: r[0].pre }];
    }

    /* ---------------- 3.00 core ---------------- */
    case 'texture': {
      const bias = args.length > 2 ? use(argVals[2][0]) : null;
      if (fam === 'sampler2DShadow') return implicit([pc(0), pc(1)], bias, pc(2));
      if (fam === 'samplerCubeShadow') return implicit([pc(0), pc(1), pc(2)], bias, pc(3));
      if (fam === 'sampler2DArrayShadow') return implicit([pc(0), pc(1), pc(2)], bias, pc(3));
      const coords = is2D ? [pc(0), pc(1)] : [pc(0), pc(1), pc(2)];
      return implicit(coords, bias);
    }
    case 'textureProj': {
      const bias = args.length > 2 ? use(argVals[2][0]) : null;
      const q = argN[1] === 3 ? pc(2) : pc(3);
      if (is2D) {
        if (env.stage === 'FRAGMENT') return implicit([divP(0, q), divP(1, q)], bias);
        return fromR('tex2DProjLod', `${unit}, ${pc(0)}, ${pc(1)}, ${q}, ${bias ?? '0'}`);
      }
      if (is3D) return implicit([divP(0, q), divP(1, q), divP(2, q)], bias);
      if (fam === 'sampler2DShadow') {
        const ref = argN[1] === 4 ? divP(2, q) : pc(2);
        if (env.stage === 'FRAGMENT') return implicit([divP(0, q), divP(1, q)], bias, ref);
        return fromR('sampleShadowLod', `${unit}, ${divP(0, q)}, ${divP(1, q)}, ${ref}, ${bias ?? '0'}`, 1);
      }
      throw new Error(`codegen: textureProj does not support '${kind}'`);
    }
    case 'textureLod': {
      const lod = use(argVals[2][0]);
      if (fam === 'sampler2DShadow') return fromR('sampleShadowLod', `${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${lod}`, 1);
      const coords = is2D ? [pc(0), pc(1)] : [pc(0), pc(1), pc(2)];
      return lodSample(coords, lod);
    }
    case 'textureOffset': {
      const bias = args.length > 3 ? use(argVals[3][0]) : null;
      const ox = use(argVals[2][0]);
      const oy = use(argVals[2][1]);
      if (is2D) return fromR('tex2DOffsetApprox', `${unit}, ${pc(0)}, ${pc(1)}, ${ox}, ${oy}`);
      if (is2DArray) return fromR('tex2DArrayOffsetApprox', `${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${ox}, ${oy}`);
      if (is3D) return fromR('tex3DOffsetApprox', `${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${ox}, ${oy}, ${use(argVals[2][2])}`);
      if (fam === 'sampler2DShadow') {
        return fromR('sampleShadowLod', `${OFFSET_COMMENT} ${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${bias ?? '0'}`, 1);
      }
      throw new Error(`codegen: textureOffset does not support '${kind}'`);
    }
    case 'textureLodOffset': {
      const lod = use(argVals[2][0]);
      if (is2D) return fromR('tex2DLod', `${OFFSET_COMMENT} ${unit}, ${pc(0)}, ${pc(1)}, ${lod}`);
      if (is3D) return fromR('tex3DLod', `${OFFSET_COMMENT} ${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${lod}`);
      if (is2DArray) return fromR('tex2DArrayLod', `${OFFSET_COMMENT} ${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${lod}`);
      if (fam === 'sampler2DShadow') {
        return fromR('sampleShadowLod', `${OFFSET_COMMENT} ${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${lod}`, 1);
      }
      throw new Error(`codegen: textureLodOffset does not support '${kind}'`);
    }
    case 'textureProjLod': {
      const lod = use(argVals[2][0]);
      const q = argN[1] === 3 ? pc(2) : pc(3);
      if (is2D) return fromR('tex2DProjLod', `${unit}, ${pc(0)}, ${pc(1)}, ${q}, ${lod}`);
      if (is3D) return lodSample([divP(0, q), divP(1, q), divP(2, q)], lod);
      if (fam === 'sampler2DShadow') {
        return fromR('sampleShadowLod', `${unit}, ${divP(0, q)}, ${divP(1, q)}, ${divP(2, q)}, ${lod}`, 1);
      }
      throw new Error(`codegen: textureProjLod does not support '${kind}'`);
    }
    case 'textureProjOffset': {
      const bias = args.length > 3 ? use(argVals[3][0]) : null;
      const q = argN[1] === 3 ? pc(2) : pc(3);
      if (is2D) {
        if (env.stage === 'FRAGMENT') return implicit([divP(0, q), divP(1, q)], bias);
        return fromR('tex2DProjLod', `${OFFSET_COMMENT} ${unit}, ${pc(0)}, ${pc(1)}, ${q}, ${bias ?? '0'}`);
      }
      if (is3D) return implicit([divP(0, q), divP(1, q), divP(2, q)], bias);
      if (fam === 'sampler2DShadow') {
        return fromR('sampleShadowLod', `${OFFSET_COMMENT} ${unit}, ${divP(0, q)}, ${divP(1, q)}, ${divP(2, q)}, ${bias ?? '0'}`, 1);
      }
      throw new Error(`codegen: textureProjOffset does not support '${kind}'`);
    }
    case 'textureProjLodOffset': {
      const lod = use(argVals[2][0]);
      const q = argN[1] === 3 ? pc(2) : pc(3);
      if (is2D) return fromR('tex2DProjLod', `${OFFSET_COMMENT} ${unit}, ${pc(0)}, ${pc(1)}, ${q}, ${lod}`);
      if (is3D) return lodSample([divP(0, q), divP(1, q), divP(2, q)], lod);
      if (fam === 'sampler2DShadow') {
        return fromR('sampleShadowLod', `${OFFSET_COMMENT} ${unit}, ${divP(0, q)}, ${divP(1, q)}, ${divP(2, q)}, ${lod}`, 1);
      }
      throw new Error(`codegen: textureProjLodOffset does not support '${kind}'`);
    }
    case 'textureGrad': {
      const g = gradComps(2);
      if (fam === 'sampler2DShadow') return gradSample([pc(0), pc(1)], g, pc(2));
      if (fam === 'samplerCubeShadow') return gradSample([pc(0), pc(1), pc(2)], g, pc(3));
      if (fam === 'sampler2DArrayShadow') return gradSample([pc(0), pc(1), pc(2)], g, pc(3));
      const coords = is2D ? [pc(0), pc(1)] : [pc(0), pc(1), pc(2)];
      return gradSample(coords, g);
    }
    case 'textureGradOffset': {
      const g = gradComps(2);
      if (fam === 'sampler2DShadow') return gradSample([pc(0), pc(1)], g, pc(2), OFFSET_COMMENT);
      if (fam === 'sampler2DArrayShadow') return gradSample([pc(0), pc(1), pc(2)], g, pc(3), OFFSET_COMMENT);
      const coords = is2D ? [pc(0), pc(1)] : [pc(0), pc(1), pc(2)];
      return gradSample(coords, g, undefined, OFFSET_COMMENT);
    }
    case 'textureProjGrad': {
      const g = gradComps(2);
      const q = argN[1] === 3 ? pc(2) : pc(3);
      if (is2D) return gradSample([divP(0, q), divP(1, q)], g);
      if (is3D) return gradSample([divP(0, q), divP(1, q), divP(2, q)], g);
      if (fam === 'sampler2DShadow') return gradSample([divP(0, q), divP(1, q)], g, divP(2, q));
      throw new Error(`codegen: textureProjGrad does not support '${kind}'`);
    }
    case 'textureProjGradOffset': {
      const g = gradComps(2);
      const q = argN[1] === 3 ? pc(2) : pc(3);
      if (is2D) return gradSample([divP(0, q), divP(1, q)], g, undefined, OFFSET_COMMENT);
      if (is3D) return gradSample([divP(0, q), divP(1, q), divP(2, q)], g, undefined, OFFSET_COMMENT);
      if (fam === 'sampler2DShadow') return gradSample([divP(0, q), divP(1, q)], g, divP(2, q), OFFSET_COMMENT);
      throw new Error(`codegen: textureProjGradOffset does not support '${kind}'`);
    }
    case 'textureSize': {
      const lod = use(argVals[1][0]);
      const b = env.allocIntScratch(3);
      const pre = [`R.textureSize(ctx, ${unit}, ${lod}, ctx.intScratch, ${b})`];
      const comps = ret.kind === 'vector' && ret.size === 3 ? 3 : 2;
      const out: Value[] = [];
      for (let i = 0; i < comps; i++) out.push({ v: `ctx.intScratch[${b} + ${i}]`, pre });
      return out;
    }
    case 'texelFetch': {
      const level = use(argVals[2][0]);
      if (is2D) return fromCtxTex(`ctx.tex.texelFetch2D(${unit}, ${pc(0)}, ${pc(1)}, ${level})`, 4);
      if (is3D) return fromCtxTex(`ctx.tex.texelFetch3D(${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${level})`, 4);
      return fromCtxTex(`ctx.tex.texelFetch2DArray(${unit}, ${pc(0)}, ${pc(1)}, ${pc(2)}, ${level})`, 4);
    }
    case 'texelFetchOffset': {
      // NOTE: this table declares texelFetchOffset as (s, P, lod, offset) —
      // the GLSL ES spec order is (s, P, offset, lod); codegen follows the
      // table (semantics resolves against it). TODO: reconcile the table.
      const level = use(argVals[2][0]);
      const ox = use(argVals[3][0]);
      const oy = use(argVals[3][1]);
      if (is2D) {
        return fromCtxTex(`ctx.tex.texelFetch2D(${unit}, (${pc(0)}) + (${ox}), (${pc(1)}) + (${oy}), ${level})`, 4);
      }
      if (is3D) {
        return fromCtxTex(
          `ctx.tex.texelFetch3D(${unit}, (${pc(0)}) + (${ox}), (${pc(1)}) + (${oy}), (${pc(2)}) + (${use(argVals[3][2])}), ${level})`,
          4,
        );
      }
      return fromCtxTex(`ctx.tex.texelFetch2DArray(${unit}, (${pc(0)}) + (${ox}), (${pc(1)}) + (${oy}), ${pc(2)}, ${level})`, 4);
    }
    default:
      throw new Error(`codegen: texture function '${name}' not lowered`);
  }
}