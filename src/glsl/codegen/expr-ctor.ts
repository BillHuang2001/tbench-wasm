/**
 * expr-ctor.ts — GLSL constructor-call lowering (non-dual): emitConstructorCall.
 *
 * Handles the scalar / vector / matrix / struct constructor families (GLSL ES
 * §5.4.2). The general model:
 * - SCALAR ctors convert one argument's scalar value to the target base
 *   (bool→float/int/uint via ternaries; float/int/uint→bool via `!== 0`).
 * - VECTOR ctors concatenate the flat components of their arguments (each
 *   converted to the target base), SPLAT a lone scalar argument across all
 *   N components, and pad a short component list with 0s and a final 1
 *   (vec4(vec2) = (v0, v1, 0, 1)). A lone MATRIX argument contributes its
 *   DIAGONAL (not legal GLSL — kept permissive for robustness; semantics
 *   rejects such programs).
 * - MATRIX ctors fill column-major from the concatenated argument components
 *   and pad with (col===row ? 1 : 0). Special cases: a lone scalar argument
 *   builds a diagonal; a lone matrix argument copies the overlapping
 *   sub-matrix and fills the rest with identity (mat3(mat2), mat2(mat3), ...).
 * - STRUCT ctors evaluate each argument and concatenate its flat components
 *   (member types match by construction — semantics verified arity/types).
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
import { scalarBaseOf, foldPre, convertValue, hasFloatLeaves } from './env.js';
import type { Value } from './index.js';
import { emitExpr, materialize } from './expressions.js';

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

/** Padding constants per vector base (0s then a final 1). */
const PAD_ZERO: Record<string, string> = { float: '0.0', int: '0', uint: '0', bool: 'false' };
const PAD_ONE: Record<string, string> = { float: '1.0', int: '1', uint: '1', bool: 'true' };

/** Lower a constructor call (name is a builtin type name or a user struct). */
export function emitConstructorCall(name: string, args: Expr[], retType: GLSLType, env: CodegenEnv): Value[] {
  // Dual mode: float-typed constructors need the dual-aware argument
  // concatenation/splat/pad lowering (C5a2) — a v-only flattening would
  // silently zero the result derivatives. Int/uint/bool ctors stay legal.
  if (env.dual && hasFloatLeaves(retType)) {
    throw new Error(`codegen: dual-mode constructor '${name}' requires C5a2 (constructors)`);
  }
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
  const v = emitExpr(args[0], env)[0];
  const from = scalarBaseOf(args[0].resolvedType!);
  return [{ v: ctorConvert(use(v), from ?? base, base) }];
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
    const av = materialize(emitExpr(args[i], env), env);
    if (at.kind === 'matrix') {
      // Not legal GLSL; permissive: the matrix's diagonal feeds the vector.
      if (args.length !== 1) {
        throw new Error(`codegen: matrix argument in a multi-argument vector constructor`);
      }
      const d = Math.min(at.cols, at.rows);
      for (let k = 0; k < d; k++) {
        const vv = av[k * at.rows + k];
        flat.push({ v: ctorConvert(vv.v, from ?? base, base), pre: vv.pre });
      }
    } else {
      for (const vv of av) flat.push({ v: ctorConvert(vv.v, from ?? base, base), pre: vv.pre });
    }
  }
  if (flat.length === 0) throw new Error(`codegen: empty '${name}' constructor`);
  if (flat.length === 1 && n > 1) {
    // Splat: one scalar argument fills every component.
    const s = flat[0];
    const out: Value[] = [];
    for (let i = 0; i < n; i++) out.push(s);
    return out;
  }
  if (flat.length > n) flat.length = n; // permissive truncation (semantics rejects)
  const out: Value[] = [...flat];
  while (out.length < n) out.push({ v: out.length === n - 1 ? PAD_ONE[base] : PAD_ZERO[base] });
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
  // Lone scalar → diagonal.
  if (args.length === 1 && args[0].resolvedType!.kind === 'scalar') {
    const s = materialize(emitExpr(args[0], env), env)[0];
    const sv = use(s);
    const out: Value[] = [];
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) out.push({ v: c === r ? sv : '0.0' });
    }
    return out;
  }
  // Lone matrix → overlapping copy, identity elsewhere.
  if (args.length === 1 && args[0].resolvedType!.kind === 'matrix') {
    const src = args[0].resolvedType!;
    const sv = emitExpr(args[0], env);
    const out: Value[] = [];
    for (let c = 0; c < cols; c++) {
      for (let r = 0; r < rows; r++) {
        if (c < src.cols && r < src.rows) {
          out.push({ v: use(sv[c * src.rows + r]) });
        } else {
          out.push({ v: c === r ? '1.0' : '0.0' });
        }
      }
    }
    return out;
  }
  // General: concatenate in column-major order, pad with (col===row ? 1 : 0).
  const flat: Value[] = [];
  for (const a of args) {
    for (const vv of materialize(emitExpr(a, env), env)) flat.push({ v: vv.v, pre: vv.pre });
  }
  if (flat.length > total) throw new Error(`codegen: '${name}' has too many components`);
  const out: Value[] = [];
  for (let k = 0; k < total; k++) {
    if (k < flat.length) {
      out.push({ v: use(flat[k]) });
    } else {
      const c = Math.floor(k / rows);
      const r = k % rows;
      out.push({ v: c === r ? '1.0' : '0.0' });
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
    out.push(...convertValue(emitExpr(args[i], env), args[i].resolvedType!, m.type));
  }
  return out;
}
