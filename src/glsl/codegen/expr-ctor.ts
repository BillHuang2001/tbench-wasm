/**
 * expr-ctor.ts — GLSL constructor-call lowering: emitConstructorCall.
 *
 * Handles the scalar / vector / matrix / struct constructor families (GLSL ES
 * §5.4.2). The general model:
 * - SCALAR ctors convert one argument's scalar value to the target base
 *   (bool→float/int/uint via ternaries; float/int/uint→bool via `!== 0`).
 * - VECTOR ctors concatenate the flat components of their arguments (each
 *   converted to the target base), SPLAT a lone scalar argument across all
 *   N components, and pad a short component list with 0s and a final 1
 *   (vec4(vec2) = (v0, v1, 0, 1)). A MATRIX argument is flattened
 *   COLUMN-MAJOR (GLSL ES 1.00 §5.4.2: vec4(mat2) = [m00, m10, m01, m11])
 *   and is legal as the SOLE argument or as the LAST argument of a multi-arg
 *   ctor (vec4(scalar, mat2) valid; vec4(mat2, scalar) invalid).
 * - MATRIX ctors fill column-major from the concatenated argument components
 *   and pad with (col===row ? 1 : 0). Special cases: a lone scalar argument
 *   builds a diagonal; a lone matrix argument copies the overlapping
 *   sub-matrix and fills the rest with identity (mat3(mat2), mat2(mat3), ...).
 * - STRUCT ctors evaluate each argument and concatenate its flat components
 *   (member types match by construction — semantics verified arity/types).
 *
 * DUAL MODE (env.dual — C5a2): every FLOAT result component carries the
 * source component's (v, dx, dy) — each ctorComp preserves the source duals
 * for float→float and attaches the constant 0 for int/uint/bool sources
 * (those values carry no derivative planes). Int/uint/bool RESULTS carry no
 * duals. Pads (0.0 / 1.0) are constant duals. `pre` attaches to the result
 * components (shared array) so materialized temps the dual strings reference
 * are set even when only the dx/dy planes are consumed.
 *
 * Multi-use argument values (splat, diagonal, padded matrix) are materialized
 * once into a temp var via materialize() (only values carrying `pre` actually
 * allocate a temp); single-use values fold their `pre` inline via foldPre.
 *
 * The circular import with expressions.ts is intentional and safe: both
 * modules only call each other inside function bodies (deferred to call time).
 */
import type { Expr } from '../ast.js';
import type { GLSLType } from '../types.js';
import type { CodegenEnv } from './env.js';
import { scalarBaseOf, foldPre, convertValue } from './env.js';
import type { Value } from './index.js';
import { emitExpr, materialize, dedupeSharedPre } from './expressions.js';

/** One-use value folded inline; multi-use values must be materialized first. */
function use(v: Value): string {
  return v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v;
}

/**
 * Scalar-base conversion for constructor arguments (explicit conversions —
 * bool is convertible in ctors, unlike implicit coercion).
 */
function ctorConvert(v: string, from: string, to: string): string {
  if (from === to) return v;
  switch (to) {
    case 'float':
      return from === 'bool' ? `(${v} ? 1.0 : 0.0)` : v; // int/uint → float: JS numbers
    case 'int':
      return from === 'bool' ? `(${v} ? 1 : 0)` : `((${v}) | 0)`;
    case 'uint':
      return from === 'bool' ? `(${v} ? 1 : 0)` : `((${v}) >>> 0)`;
    case 'bool':
      return from === 'float' ? `(${v} !== 0.0)` : `((${v}) !== 0)`;
  }
  throw new Error(`codegen: no conversion ${from} → ${to}`);
}

/**
 * One constructor component. Non-dual: `{ v: ctorConvert(v.v), pre: v.pre }`
 * (byte-identical to the pre-dual emitters). Dual mode: float RESULTS attach
 * the source duals — preserved for float→float (the source carries the
 * derivative planes), constant 0 for int/uint/bool sources; integral results
 * carry no duals. `pre` attaches so source temps stay alive for dx-only
 * consumers (dFdx(float(f()))).
 */
function ctorComp(v: Value, from: string | null, to: string, env: CodegenEnv): Value {
  const out: Value = { v: ctorConvert(v.v, from ?? to, to), pre: v.pre };
  if (env.dual && to === 'float') {
    out.dx = from === 'float' ? (v.dx ?? '0') : '0';
    out.dy = from === 'float' ? (v.dy ?? '0') : '0';
  }
  return out;
}

/** Padding constant per vector base (0s then a final 1). */
const PAD_ZERO: Record<string, string> = { float: '0.0', int: '0', uint: '0', bool: 'false' };
const PAD_ONE: Record<string, string> = { float: '1.0', int: '1', uint: '1', bool: 'true' };

/** A constant pad component — constant duals in dual mode. */
function padComp(base: string, v: string, env: CodegenEnv): Value {
  const out: Value = { v };
  if (env.dual && base === 'float') {
    out.dx = '0';
    out.dy = '0';
  }
  return out;
}

/** Pin a SIDE-EFFECTING scalar value to a temp so a multi-use broadcast
 *  (vector splat, matrix diagonal) evaluates it EXACTLY ONCE. Postfix/prefix
 *  `++`/`--` and assignments emit self-contained comma expressions with NO
 *  `pre` (materialize() only hoists pre-carrying values), so without the pin
 *  the broadcast would duplicate the string and re-run the effect per
 *  component: vec4(i++) would yield vec4(1,2,3,4) — CTS
 *  glsl-construct-vec-mat-index requires mat2(i++, vec4(i++)) ==
 *  mat2(0,1,1,1), i.e. each i++ runs once and the splat broadcasts the single
 *  evaluated value. The FIRST use carries the inline `(t = expr, t)`; all
 *  later uses read the pure temp `t` (so a downstream materialize() — the
 *  matrix-ctor general path — cannot duplicate the effect either). Pure
 *  values (no assignment `=`) and pre-carrying values (already pinned by
 *  materialize) return null and broadcast verbatim. */
function pinSideEffect(v: Value, env: CodegenEnv): { first: Value; rest: Value } | null {
  if (v.pre && v.pre.length > 0) return null;
  if (!/=(?!=)/.test(v.v)) return null;
  const t = env.allocTemp();
  const first: Value = { v: `(${t} = ${v.v}, ${t})` };
  const rest: Value = { v: t };
  if (env.dual && v.dx !== undefined) {
    first.dx = v.dx;
    first.dy = v.dy;
    rest.dx = v.dx;
    rest.dy = v.dy;
  }
  return { first, rest };
}

/** Lower a constructor call (name is a builtin type name or a user struct). */
export function emitConstructorCall(name: string, args: Expr[], retType: GLSLType, env: CodegenEnv): Value[] {
  switch (retType.kind) {
    case 'scalar':
      return emitScalarCtor(name, args, retType, env);
    case 'vector':
      return emitVectorCtor(name, args, retType, env);
    case 'matrix':
      return emitMatrixCtor(name, args, retType, env);
    case 'struct':
      return emitStructCtor(name, args, retType, env);
    default:
      throw new Error(`codegen: cannot construct a '${retType.kind}'`);
  }
}

/* ------------------------------------------------------------------ */
/* Scalar constructors                                                 */
/* ------------------------------------------------------------------ */

function emitScalarCtor(
  name: string,
  args: Expr[],
  retType: Extract<GLSLType, { kind: 'scalar' }>,
  env: CodegenEnv,
): Value[] {
  if (args.length !== 1) throw new Error(`codegen: '${name}' takes exactly one argument`);
  const base = retType.base;
  const raw = emitExpr(args[0], env)[0];
  const from = scalarBaseOf(args[0].resolvedType!);
  if (!env.dual) return [{ v: ctorConvert(use(raw), from ?? base, base) }];
  // Dual mode: materialize so the result's duals reference stable temps and
  // pre-carrying sources (inlined calls) run exactly once even when only the
  // duals are consumed (dFdx(float(f()))).
  const v = materialize([raw], env)[0];
  return [ctorComp(v, from ?? base, base, env)];
}

/* ------------------------------------------------------------------ */
/* Vector constructors                                                 */
/* ------------------------------------------------------------------ */

function emitVectorCtor(
  name: string,
  args: Expr[],
  retType: Extract<GLSLType, { kind: 'vector' }>,
  env: CodegenEnv,
): Value[] {
  const n = retType.size;
  const base = retType.base;
  const flat: Value[] = [];
  for (let i = 0; i < args.length; i++) {
    const at = args[i].resolvedType!;
    const from = scalarBaseOf(at);
    // BUG (shared-pre re-run): dedupe BEFORE materialize — a multi-component
    // call result carries ONE `[iife]` pre array on every component, and
    // materialize() folds it per component, re-running the callee
    // (vec4(f(), 0.0, 1.0) with a vec2-returning f ran f twice). Only the
    // first component keeps the iife; later components read the retTemps.
    const av = materialize(dedupeSharedPre(emitExpr(args[i], env)), env);
    if (at.kind === 'matrix') {
      // GLSL ES 1.00 §5.4.2: a matrix argument is flattened COLUMN-MAJOR
      // (vec4(mat2) = [m00,m10,m01,m11]). Legal as the SOLE argument or as
      // the LAST argument of a multi-arg ctor (vec4(scalar, mat2) valid;
      // vec4(mat2, scalar) invalid — semantics rejects non-last matrices via
      // its component-count rule, so this throw is defensive).
      if (i !== args.length - 1) {
        throw new Error('codegen: matrix argument must be the last argument of a vector constructor');
      }
      for (const vv of av) flat.push(ctorComp(vv, from ?? base, base, env));
    } else {
      for (const vv of av) flat.push(ctorComp(vv, from ?? base, base, env));
    }
  }
  if (flat.length === 0) throw new Error(`codegen: empty '${name}' constructor`);
  if (flat.length === 1 && n > 1) {
    // Splat: one scalar argument fills every component (same Value object —
    // its duals broadcast to all components). Side-effecting arguments
    // (i++ / assignments — comma expressions with no `pre`) are pinned so
    // the effect runs once, not once per component (vec4(i++) must be
    // vec4(1,1,1,1); CTS glsl-construct-vec-mat-index).
    const s = flat[0];
    const pin = pinSideEffect(s, env);
    const out: Value[] = [];
    for (let i = 0; i < n; i++) out.push(pin ? (i === 0 ? pin.first : pin.rest) : s);
    // BUG (shared-pre re-run): the splat pushes the SAME value object n times
    // — a call-result scalar (vec4(f()) with float f) carries one [iife] pre
    // array; dedupe by identity so it folds exactly once (comp0), the rest
    // read the temp (per-component folding would re-run the callee n times).
    return dedupeSharedPre(out);
  }
  if (flat.length > n) flat.length = n; // permissive truncation (semantics rejects)
  const out: Value[] = [...flat];
  while (out.length < n) out.push(padComp(base, out.length === n - 1 ? PAD_ONE[base] : PAD_ZERO[base], env));
  return out;
}

/* ------------------------------------------------------------------ */
/* Matrix constructors                                                 */
/* ------------------------------------------------------------------ */

function emitMatrixCtor(
  name: string,
  args: Expr[],
  retType: Extract<GLSLType, { kind: 'matrix' }>,
  env: CodegenEnv,
): Value[] {
  const cols = retType.cols;
  const rows = retType.rows;
  const total = cols * rows;
  // Lone scalar → diagonal. A side-effecting scalar (i++) is pinned so the
  // diagonal broadcast evaluates it once (see pinSideEffect).
  if (args.length === 1 && args[0].resolvedType!.kind === 'scalar') {
    const s0 = materialize(emitExpr(args[0], env), env)[0];
    const pin = pinSideEffect(s0, env);
    const s = pin ? pin.first : s0;
    const rest = pin ? pin.rest : s0;
    const raw: Value[] = [];
    let first = true;
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        if (c === r) {
          raw.push(first ? s : rest);
          first = false;
        } else {
          raw.push(padComp('float', '0.0', env));
        }
      }
    }
    // BUG (shared-pre re-run): the diagonal broadcast reuses the SAME value
    // object once per diagonal slot — a call-result scalar (mat2(f()) with
    // float f) carries one [iife] pre array, and per-slot folding re-ran the
    // callee once per slot. Dedupe by identity BEFORE the use()/ctorComp wrap
    // so the [iife] folds exactly once (first slot); the rest read the temp.
    return dedupeSharedPre(raw).map((v) =>
      env.dual ? ctorComp(v, scalarBaseOf(args[0].resolvedType!) ?? 'float', 'float', env) : { v: use(v) },
    );
  }
  // Lone matrix → overlapping copy, identity elsewhere.
  if (args.length === 1 && args[0].resolvedType!.kind === 'matrix') {
    const src = args[0].resolvedType!;
    // BUG (shared-pre re-run): dedupe the source components by identity — a
    // matrix-returning call result carries ONE [iife] pre array on every
    // component, and the per-component use()/ctorComp below re-ran the callee
    // once per component (mat2(f()) with mat2 f ran f 4×).
    const sv = dedupeSharedPre(emitExpr(args[0], env));
    const out: Value[] = [];
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        if (c < src.cols && r < src.rows) {
          out.push(env.dual ? ctorComp(sv[c * src.rows + r], 'float', 'float', env) : { v: use(sv[c * src.rows + r]) });
        } else {
          out.push(padComp('float', c === r ? '1.0' : '0.0', env));
        }
      }
    }
    return out;
  }
  // General: concatenate in column-major order, pad with (col===row ? 1 : 0).
  const raw: Value[] = [];
  for (const a of args) {
    // BUG (shared-pre re-run): dedupe BEFORE materialize — a multi-component
    // call result carries ONE [iife] on every component; materialize() folds
    // it per component, re-running the callee. Only the first component keeps
    // the iife; the rest read the retTemps.
    const av = materialize(dedupeSharedPre(emitExpr(a, env)), env);
    for (const vv of av) {
      raw.push(ctorComp(vv, scalarBaseOf(a.resolvedType!) ?? 'float', 'float', env));
    }
  }
  // GLSL ES 1.00 §5.4.2: constructors SHORTEN — extra components are DROPPED,
  // the first `total` (cols×rows) in column-major order are kept
  // (mat2(float, vec4) = the scalar + the vec4's first 3 components). All
  // arguments are evaluated in the loop BEFORE truncation, so side effects
  // (mat2(i++, vec4(i++)) — CTS glsl-construct-vec-mat-index) always execute;
  // semantics guarantees the dropped tail belongs to the LAST argument only.
  const flat = dedupeSharedPre(raw);
  if (flat.length > total) flat.length = total;
  const out: Value[] = [];
  for (let k = 0; k < total; k++) {
    if (k < flat.length) {
      out.push(env.dual ? flat[k] : { v: use(flat[k]) });
    } else {
      const c = Math.floor(k / rows);
      const r = k % rows;
      out.push(padComp('float', c === r ? '1.0' : '0.0', env));
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Struct constructors                                                 */
/* ------------------------------------------------------------------ */

function emitStructCtor(
  name: string,
  args: Expr[],
  retType: Extract<GLSLType, { kind: 'struct' }>,
  env: CodegenEnv,
): Value[] {
  if (args.length !== retType.members.length) {
    throw new Error(`codegen: struct '${name}' constructor arity mismatch`);
  }
  const out: Value[] = [];
  for (let i = 0; i < args.length; i++) {
    const m = retType.members[i];
    // convertValue preserves float→float duals; int→float attaches constant
    // duals; int members carry no duals. convertValue DROPS Value.pre when it
    // converts scalar bases — re-attach so member pres survive.
    const srcVals = emitExpr(args[i], env);
    const conv = convertValue(srcVals, args[i].resolvedType!, m.type);
    for (let c = 0; c < srcVals.length; c++) {
      const src = srcVals[c];
      if (conv[c] !== src && src.pre && src.pre.length > 0) {
        conv[c] = { ...conv[c], pre: src.pre };
      }
    }
    out.push(...conv);
  }
  // BUG (shared-pre re-run): a multi-component call argument (vec4-returning
  // f for a vec4 member) carries ONE [iife] pre array on every component —
  // dedupe by identity so it folds exactly once (first component); the rest
  // read the retTemps. Each member value is used exactly once downstream.
  return dedupeSharedPre(out);
}
