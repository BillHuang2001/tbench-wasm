/**
 * semantics-const.ts — const-expression evaluation (BUGS 2+5: const folding).
 *
 * `evalConstExpr(e, scope, ctx)` evaluates a fully ANALYZED expression tree to
 * its flat component array `(number | boolean)[]` (column-major matrices,
 * struct members flattened in declaration order, arrays element-major,
 * scalar = [v]), or `undefined` when ANY leaf is non-constant (no partial
 * folding — a single non-const leaf makes the whole expression non-const).
 *
 * It is the engine behind:
 * - `declareVariables` (semantics.ts): const initializers are accepted iff
 *   this returns a value — the CTS `const vec4 green = vec4(0.0, 1.0, 0.0, 1.0)`
 *   boilerplate (BUG 2), ogles comma/struct/matrix const-expression forms
 *   (BUG 5) and the tightened array/struct acceptance ("constant expression"
 *   instead of "any constructor call").
 * - use-site folding (semantics-expr.ts analyzeMember/analyzeIndex): scalar
 *   results become `constValue`, aggregates are mutated into annotated ctor
 *   calls (codegen is untouched — see the mutation contract there).
 *
 * The evaluator mirrors the SEMANTICS rules exactly (strict operand typing is
 * already enforced by the analysis pass; this pass only EVALUATES):
 * - scalar binary ops delegate to `foldBinary` (semantics-expr.ts) — uint wrap
 *   and int truncation rules identical;
 * - unary `-`/`~` use the same uint-wrap rules as analyzeUnary;
 * - constructor conversion uses `convertConst` (the same ctor conversion
 *   semantics as the scalar-ctor fold);
 * - matrix products use the exact codegen indexing (expressions.ts emitArith);
 * - builtin calls: scalar forms fold via the analysis-time scalar fold
 *   (`e.constValue`); vector/matrix forms fold here via `evalBuiltinConstFold`
 *   (GLSL ES §5.10: "a built-in function call whose arguments are all
 *   constant expressions, with the exception of the texture lookup functions"
 *   — CTS const-variable-initialization.html). Texture lookups, derivative
 *   functions and other non-math builtins stay non-constant; user function
 *   calls never fold.
 *
 * Also implements `validateGlobalInit` — ANGLE ValidateGlobalInitializer
 * parity for GLOBAL variable initializers (WebGL CTS
 * global-variable-init.html): uniforms/other globals are allowed in WebGL1
 * (legacy compatibility), but attributes, varyings, builtin non-constants,
 * user function calls, non-math builtin calls (texture lookups) and lvalue
 * operations are compile errors; ESSL 3.00 allows only const symbols.
 *
 * This module intentionally has no other dependencies than the shared type
 * modules; the import cycle with semantics-expr.ts is function-declaration
 * only (both sides call each other lazily at analysis time — never at module
 * init), the same benign pattern as semantics ↔ semantics-expr.
 */
import type { CallExpr, Expr, IdentifierExpr } from './ast.js';
import type { GLSLType } from './types.js';
import type { Scope, SemContext } from './semantics.js';
import { builtinType } from './semantics.js';
import { convertConst, foldBinary, SWIZZLE_SETS } from './semantics-expr.js';

/** Flat component count of a type: scalar 1, vector size, matrix cols*rows,
 *  struct = sum of members (recursive), array = size × element. */
export function flatSize(t: GLSLType): number {
  switch (t.kind) {
    case 'scalar':
      return 1;
    case 'vector':
      return t.size;
    case 'matrix':
      return t.cols * t.rows;
    case 'struct': {
      let n = 0;
      for (const m of t.members) n += flatSize(m.type);
      return n;
    }
    case 'array':
      return (t.size ?? 0) * flatSize(t.element);
    default:
      return 0; // void / sampler: no components
  }
}

/** Scalar base of a scalar/vector; 'float' for matrices (never bool here). */
function baseOf(t: GLSLType): 'float' | 'int' | 'uint' | 'bool' {
  return t.kind === 'scalar' || t.kind === 'vector' ? t.base : 'float';
}

/**
 * Evaluate a fully analyzed expression to its flat constant components.
 * Returns undefined when the expression is not a constant expression.
 */
export function evalConstExpr(e: Expr, scope: Scope, ctx: SemContext): (number | boolean)[] | undefined {
  switch (e.kind) {
    case 'literal':
      return [e.value];
    case 'identifier':
      return evalConstIdentifier(e, scope);
    case 'member': {
      const ot = e.object.resolvedType;
      if (ot === undefined) return undefined;
      const obj = evalConstExpr(e.object, scope, ctx);
      if (obj === undefined) return undefined;
      if (ot.kind === 'struct') {
        let off = 0;
        for (const m of ot.members) {
          const sz = flatSize(m.type);
          if (m.name === e.name) return obj.slice(off, off + sz);
          off += sz;
        }
        return undefined;
      }
      if (ot.kind === 'vector') {
        const idx = swizzleIndices(ot.size, e.name);
        if (idx === null) return undefined;
        return idx.map((i) => obj[i]);
      }
      return undefined;
    }
    case 'index': {
      const ot = e.object.resolvedType;
      if (ot === undefined) return undefined;
      const obj = evalConstExpr(e.object, scope, ctx);
      if (obj === undefined) return undefined;
      const iv = evalConstExpr(e.index, scope, ctx);
      if (iv === undefined || iv.length !== 1 || typeof iv[0] !== 'number' || !Number.isInteger(iv[0])) {
        return undefined;
      }
      const i = iv[0];
      switch (ot.kind) {
        case 'vector':
          if (i < 0 || i >= ot.size) return undefined;
          return [obj[i]];
        case 'matrix': {
          if (i < 0 || i >= ot.cols) return undefined;
          return obj.slice(i * ot.rows, i * ot.rows + ot.rows);
        }
        case 'array': {
          const esz = flatSize(ot.element);
          if (i < 0 || i >= (ot.size ?? 0)) return undefined;
          return obj.slice(i * esz, (i + 1) * esz);
        }
        default:
          return undefined;
      }
    }
    case 'unary': {
      const ot = e.operand.resolvedType;
      if (ot === undefined) return undefined;
      const op = evalConstExpr(e.operand, scope, ctx);
      if (op === undefined) return undefined;
      switch (e.op) {
        case '+':
          return op;
        case '-': {
          const base = baseOf(ot);
          const out: (number | boolean)[] = [];
          for (const v of op) {
            if (typeof v !== 'number') return undefined;
            out.push(base === 'uint' ? (-v) >>> 0 : base === 'int' ? (-v) | 0 : -v);
          }
          return out;
        }
        case '~': {
          const base = baseOf(ot);
          const out: (number | boolean)[] = [];
          for (const v of op) {
            if (typeof v !== 'number') return undefined;
            out.push(base === 'uint' ? (~v) >>> 0 : (~v) | 0);
          }
          return out;
        }
        case '!': {
          if (op.length !== 1 || typeof op[0] !== 'boolean') return undefined;
          return [!op[0]];
        }
        default:
          return undefined; // ++ / -- are never constant
      }
    }
    case 'binary':
      return evalConstBinary(e, scope, ctx);
    case 'ternary': {
      const c = evalConstExpr(e.cond, scope, ctx);
      if (c === undefined || c.length !== 1 || typeof c[0] !== 'boolean') return undefined;
      // Fold to the CHOSEN arm (the other arm was already analyzed/validated).
      return evalConstExpr(c[0] ? e.whenTrue : e.whenFalse, scope, ctx);
    }
    case 'comma': {
      // ES 3.00: the sequence operator NEVER yields a constant expression
      // (ESSL 3.00 §5.9 — CTS sequence-operator-returns-non-constant.html:
      // `const float a = (0.0, 1.0);` must fail to compile). ES 1.00 folds
      // ONLY when every operand is a constant expression (the ogles
      // CorrectComma_frag build test requires `const vec4 v = (vec4(1,2,3,4),
      // vec4(5,6,7,8));` to compile); a non-constant operand (user call,
      // assignment) keeps the sequence non-constant.
      if (ctx.version !== 100) return undefined;
      const last = e.exprs[e.exprs.length - 1];
      if (last === undefined) return undefined;
      for (const x of e.exprs) {
        if (evalConstExpr(x, scope, ctx) === undefined) return undefined;
      }
      return evalConstExpr(last, scope, ctx);
    }
    case 'call':
      return evalConstCall(e, scope, ctx);
    default:
      return undefined; // assignments are never constant expressions
  }
}

/** Identifier: a const variable with folded data, or a builtin gl_Max* constant. */
function evalConstIdentifier(e: IdentifierExpr, scope: Scope): (number | boolean)[] | undefined {
  const sym = scope.lookup(e.name);
  if (sym === undefined) return undefined;
  switch (sym.kind) {
    case 'var':
      // Const-qualified PARAMS never carry constData (declareVariables only);
      // non-const variables and unresolved consts (failed initializers) are
      // not constant expressions.
      if (sym.storage !== 'const') return undefined;
      if (sym.constData !== undefined) return sym.constData;
      if (sym.constValue !== undefined) return [sym.constValue];
      return undefined;
    case 'builtin-const':
      return [sym.value];
    default:
      return undefined;
  }
}

/** Swizzle name → component indices within the vector (null when invalid). */
function swizzleIndices(size: number, name: string): number[] | null {
  if (name.length === 0 || name.length > 4) return null;
  let set: string | null = null;
  const out: number[] = [];
  for (const ch of name) {
    let found = false;
    for (const s of SWIZZLE_SETS) {
      const i = s.indexOf(ch);
      if (i >= 0) {
        if (set !== null && set !== s) return null; // mixes component sets
        set = s;
        if (i >= size) return null; // out of range (analysis already rejected)
        out.push(i);
        found = true;
        break;
      }
    }
    if (!found) return null;
  }
  return out;
}

function evalConstBinary(
  e: Extract<Expr, { kind: 'binary' }>,
  scope: Scope,
  ctx: SemContext,
): (number | boolean)[] | undefined {
  const lt = e.left.resolvedType;
  const rt = e.right.resolvedType;
  if (lt === undefined || rt === undefined) return undefined;
  const lv = evalConstExpr(e.left, scope, ctx);
  const rv = evalConstExpr(e.right, scope, ctx);
  if (lv === undefined || rv === undefined) return undefined;
  const op = e.op;

  // True linear-algebra products (codegen emitArith indexing).
  if (op === '*') {
    if (lt.kind === 'matrix' && rt.kind === 'matrix') {
      // A (lt.cols × lt.rows) * B (rt.cols × rt.rows), legal iff lt.cols === rt.rows.
      // result[c][r] = Σ_{s<lt.cols} A[s][r] * B[c][s]
      const aRows = lt.rows;
      const aCols = lt.cols;
      const bRows = rt.rows;
      const out: number[] = [];
      for (let c = 0; c < rt.cols; c++) {
        for (let r = 0; r < aRows; r++) {
          let s = 0;
          for (let k = 0; k < aCols; k++) {
            s += (lv[k * aRows + r] as number) * (rv[c * bRows + k] as number);
          }
          out.push(s);
        }
      }
      return out;
    }
    if (lt.kind === 'matrix' && rt.kind === 'vector') {
      // M (lt.cols × lt.rows) * v (lt.rows) → v (lt.cols):
      // result[r] = Σ_{c<lt.cols} M[c*lt.rows + r] * v[c]
      const R = lt.rows;
      const C = lt.cols;
      const out: number[] = [];
      for (let r = 0; r < R; r++) {
        let s = 0;
        for (let c = 0; c < C; c++) s += (lv[c * R + r] as number) * (rv[c] as number);
        out.push(s);
      }
      return out;
    }
    if (lt.kind === 'vector' && rt.kind === 'matrix') {
      // v (rt.rows) * M (rt.cols × rt.rows) → v (rt.cols):
      // result[c] = Σ_{r<rt.rows} v[r] * M[c*rt.rows + r]
      const R = rt.rows;
      const C = rt.cols;
      const out: number[] = [];
      for (let c = 0; c < C; c++) {
        let s = 0;
        for (let r = 0; r < R; r++) s += (lv[r] as number) * (rv[c * R + r] as number);
        out.push(s);
      }
      return out;
    }
  }

  // Logical + relational operators: scalar-only (analysis enforces).
  if (op === '&&' || op === '||' || op === '^^' || op === '<' || op === '>' || op === '<=' || op === '>=') {
    if (lv.length !== 1 || rv.length !== 1) return undefined;
    const r = foldBinary(op, lt, rt, lv[0], rv[0], ctx.version);
    return r === undefined ? undefined : [r];
  }

  // Equality: scalars delegate; aggregates compare component-wise → bool.
  // STRUCT operands land here too (semantics accepts struct == / != per
  // GLSL ES 1.00 §5.9 / ES 3.00 §5.9): their flattened component lists have
  // equal length, and foldBinary per component handles both numeric (exact
  // for int/uint — float64 represents int32/uint32 exactly) and bool members.
  if (op === '==' || op === '!=') {
    if (lv.length === 1 && rv.length === 1) {
      const r = foldBinary(op, lt, rt, lv[0], rv[0], ctx.version);
      return r === undefined ? undefined : [r];
    }
    if (lv.length !== rv.length) return undefined;
    let eq = true;
    for (let i = 0; i < lv.length; i++) {
      const p = foldBinary('==', lt, rt, lv[i], rv[i], ctx.version);
      if (typeof p !== 'boolean') return undefined;
      if (!p) eq = false;
    }
    return [op === '==' ? eq : !eq];
  }

  // Arithmetic / modulo / bitwise: component-wise with scalar broadcast
  // (foldBinary applies the same uint-wrap / int-truncation rules per pair).
  const n = Math.max(lv.length, rv.length);
  const out: (number | boolean)[] = [];
  for (let i = 0; i < n; i++) {
    if (lv.length !== 1 && i >= lv.length) return undefined; // shape mismatch (analysis already errored)
    if (rv.length !== 1 && i >= rv.length) return undefined;
    const a = lv.length === 1 ? lv[0] : lv[i];
    const b = rv.length === 1 ? rv[0] : rv[i];
    const r = foldBinary(op, lt, rt, a, b, ctx.version);
    if (r === undefined) return undefined;
    out.push(r);
  }
  return out;
}

function evalConstCall(e: CallExpr, scope: Scope, ctx: SemContext): (number | boolean)[] | undefined {
  const t = e.resolvedType;
  if (t === undefined) return undefined;
  const callee = e.callee;
  if (callee.kind === 'index') {
    // T[N](...) array constructor (ES 3.00).
    if (t.kind !== 'array') return undefined;
    return evalArrayCtor(e, t, scope, ctx);
  }
  if (callee.kind !== 'identifier') return undefined;
  const name = callee.name;
  if (builtinType(name, ctx.version) !== undefined || isStructName(name, scope)) {
    return evalCtor(e, t, scope, ctx);
  }
  // Builtin / user function call. Scalar forms fold at analysis time
  // (`e.constValue`); vector/matrix forms fold here (evalBuiltinConstFold).
  // User function calls never fold (no constValue, not in the math family).
  if (e.constValue !== undefined) return [e.constValue];
  return evalBuiltinConstFold(e, name, scope, ctx);
}

function isStructName(name: string, scope: Scope): boolean {
  const s = scope.lookup(name);
  return s !== undefined && s.kind === 'struct';
}

function evalCtor(e: CallExpr, t: GLSLType, scope: Scope, ctx: SemContext): (number | boolean)[] | undefined {
  switch (t.kind) {
    case 'scalar': {
      const a = e.args[0];
      const at = a !== undefined ? a.resolvedType : undefined;
      if (at === undefined) return undefined;
      const av = evalConstExpr(a, scope, ctx);
      if (av === undefined || av.length === 0) return undefined;
      return [convertConst(av[0], baseOf(at), t.base)];
    }
    case 'vector':
      return evalVectorCtor(e, t, scope, ctx);
    case 'matrix':
      return evalMatrixCtor(e, t, scope, ctx);
    case 'struct':
      return evalStructCtor(e, t, scope, ctx);
    case 'array':
      return evalArrayCtor(e, t, scope, ctx);
    default:
      return undefined; // samplers / void cannot be constructed
  }
}

function evalVectorCtor(
  e: CallExpr,
  t: Extract<GLSLType, { kind: 'vector' }>,
  scope: Scope,
  ctx: SemContext,
): (number | boolean)[] | undefined {
  const n = t.size;
  const args = e.args;
  // Splat: a lone scalar argument fills every component (GLSL ES §5.4.2).
  if (args.length === 1 && args[0].resolvedType?.kind === 'scalar') {
    const at = args[0].resolvedType;
    const av = evalConstExpr(args[0], scope, ctx);
    if (av === undefined) return undefined;
    const c = convertConst(av[0], at.base, t.base);
    const out: (number | boolean)[] = [];
    for (let i = 0; i < n; i++) out.push(c);
    return out;
  }
  // General form: concatenate argument components (column-major for matrix
  // args), converting each to the target base; truncate trailing extras.
  const flat: (number | boolean)[] = [];
  for (const a of args) {
    const at = a.resolvedType;
    if (at === undefined) return undefined;
    const av = evalConstExpr(a, scope, ctx);
    if (av === undefined) return undefined;
    const from = at.kind === 'scalar' || at.kind === 'vector' ? at.base : 'float';
    for (const v of av) flat.push(convertConst(v, from, t.base));
  }
  if (flat.length < n) return undefined; // analysis already rejected the program
  return flat.slice(0, n);
}

function evalMatrixCtor(
  e: CallExpr,
  t: Extract<GLSLType, { kind: 'matrix' }>,
  scope: Scope,
  ctx: SemContext,
): (number | boolean)[] | undefined {
  const cols = t.cols;
  const rows = t.rows;
  const total = cols * rows;
  const args = e.args;
  if (args.length === 1) {
    const at = args[0].resolvedType;
    if (at === undefined) return undefined;
    const av = evalConstExpr(args[0], scope, ctx);
    if (av === undefined) return undefined;
    if (at.kind === 'scalar') {
      // Lone scalar → diagonal matrix.
      const c = convertConst(av[0], at.base, 'float') as number;
      const out: number[] = [];
      for (let c2 = 0; c2 < cols; c2++) {
        for (let r = 0; r < rows; r++) out.push(c2 === r ? c : 0);
      }
      return out;
    }
    if (at.kind === 'matrix') {
      // matN(matM): overlapping sub-matrix copied, rest filled from identity.
      const out: number[] = [];
      for (let c2 = 0; c2 < cols; c2++) {
        for (let r = 0; r < rows; r++) {
          if (c2 < at.cols && r < at.rows) out.push(av[c2 * at.rows + r] as number);
          else out.push(c2 === r ? 1 : 0);
        }
      }
      return out;
    }
    // Lone vector: fall through to the general concatenation form.
  }
  const flat: number[] = [];
  for (const a of args) {
    const at = a.resolvedType;
    if (at === undefined) return undefined;
    const av = evalConstExpr(a, scope, ctx);
    if (av === undefined) return undefined;
    const from = at.kind === 'scalar' || at.kind === 'vector' ? at.base : 'float';
    for (const v of av) flat.push(convertConst(v, from, 'float') as number);
  }
  if (flat.length < total) return undefined; // analysis already rejected
  const out: number[] = [];
  for (let k = 0; k < total; k++) {
    if (k < flat.length) out.push(flat[k]);
    else out.push(Math.floor(k / rows) === k % rows ? 1 : 0); // identity pad (defensive)
  }
  return out;
}

function evalStructCtor(
  e: CallExpr,
  t: Extract<GLSLType, { kind: 'struct' }>,
  scope: Scope,
  ctx: SemContext,
): (number | boolean)[] | undefined {
  if (e.args.length !== t.members.length) return undefined;
  const out: (number | boolean)[] = [];
  for (let i = 0; i < t.members.length; i++) {
    const at = e.args[i].resolvedType;
    if (at === undefined) return undefined;
    const av = evalConstExpr(e.args[i], scope, ctx);
    if (av === undefined) return undefined;
    // Members match by exact type (convertible is strict); flatten as-is.
    if (av.length !== flatSize(t.members[i].type)) return undefined;
    out.push(...av);
  }
  return out;
}

function evalArrayCtor(
  e: CallExpr,
  t: Extract<GLSLType, { kind: 'array' }>,
  scope: Scope,
  ctx: SemContext,
): (number | boolean)[] | undefined {
  const n = t.size ?? 0;
  if (e.args.length !== n) return undefined;
  const esz = flatSize(t.element);
  const out: (number | boolean)[] = [];
  for (const a of e.args) {
    const av = evalConstExpr(a, scope, ctx);
    if (av === undefined || av.length !== esz) return undefined;
    out.push(...av);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Builtin const folding (vector/matrix args & results)                */
/* ------------------------------------------------------------------ */

/**
 * The MATH builtin family (ANGLE "Group Math" — radians..imulExtended).
 * These are the only builtins allowed in constant expressions (GLSL ES
 * §5.10: "a built-in function call whose arguments are all constant
 * expressions, with the exception of the texture lookup functions") and in
 * global variable initializers (ANGLE ValidateGlobalInitializer). Texture
 * lookups, derivative functions, pack/bitfield helpers etc. are excluded.
 */
const MATH_BUILTIN_NAMES: ReadonlySet<string> = new Set([
  // MathTrigonometric
  'radians', 'degrees', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
  'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
  // MathExponential
  'pow', 'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
  // MathCommon
  'abs', 'sign', 'floor', 'trunc', 'round', 'roundEven', 'ceil', 'fract',
  'mod', 'min', 'max', 'clamp', 'mix', 'step', 'smoothstep', 'modf',
  'isnan', 'isinf', 'fma', 'frexp', 'ldexp',
  'floatBitsToInt', 'floatBitsToUint', 'intBitsToFloat', 'uintBitsToFloat',
  'packSnorm2x16', 'packHalf2x16', 'unpackSnorm2x16', 'unpackHalf2x16',
  'packUnorm2x16', 'unpackUnorm2x16', 'packUnorm4x8', 'packSnorm4x8',
  'unpackUnorm4x8', 'unpackSnorm4x8',
  // MathGeometric
  'length', 'distance', 'dot', 'cross', 'normalize', 'faceforward', 'reflect', 'refract',
  // MathMatrix
  'matrixCompMult', 'outerProduct', 'transpose', 'determinant', 'inverse',
  // MathVector
  'lessThan', 'lessThanEqual', 'greaterThan', 'greaterThanEqual', 'equal', 'notEqual',
  'any', 'all', 'not',
  // MathInteger (ESSL 3.00)
  'bitfieldExtract', 'bitfieldInsert', 'bitfieldReverse', 'bitCount', 'findLSB', 'findMSB',
  'uaddCarry', 'usubBorrow', 'umulExtended', 'imulExtended',
]);

/** One const-evaluated call argument: flat components + scalar base. */
interface ConstArg {
  data: (number | boolean)[];
  base: 'float' | 'int' | 'uint' | 'bool';
}

/** Dot product of two equal-length numeric vectors. */
function dotN(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Numeric view of one const arg (null when a component is a boolean). */
function numArg(a: ConstArg): number[] | null {
  const out: number[] = [];
  for (const v of a.data) {
    if (typeof v !== 'number') return null;
    out.push(v);
  }
  return out;
}

/** Component value of arg `a` at output index `i` (scalar args broadcast). */
function compAt(a: ConstArg, i: number): number | boolean {
  return a.data.length === 1 ? a.data[0] : a.data[i];
}

/**
 * Component-wise application of `f` over const args (scalar args broadcast to
 * the output length), converting each result to the result base
 * (int |0 / uint >>>0). Returns undefined when a component is not numeric.
 */
function foldElem(
  args: ConstArg[],
  n: number,
  f: (xs: number[]) => number,
  outBase: 'float' | 'int' | 'uint' | 'bool',
): (number | boolean)[] | undefined {
  const out: (number | boolean)[] = [];
  for (let i = 0; i < n; i++) {
    const xs: number[] = [];
    for (const a of args) {
      const v = compAt(a, i);
      if (typeof v !== 'number') return undefined;
      xs.push(v);
    }
    const r = f(xs);
    out.push(outBase === 'int' ? r | 0 : outBase === 'uint' ? r >>> 0 : r);
  }
  return out;
}

/** Per-component comparison predicate → bool vector (undefined on non-numeric). */
function foldPred(
  args: ConstArg[],
  n: number,
  f: (a: number, b: number) => boolean,
): (number | boolean)[] | undefined {
  const out: (number | boolean)[] = [];
  for (let i = 0; i < n; i++) {
    const av = compAt(args[0], i);
    const bv = compAt(args[1], i);
    if (typeof av !== 'number' || typeof bv !== 'number') return undefined;
    out.push(f(av, bv));
  }
  return out;
}

/**
 * Fold a MATH builtin call whose arguments are ALL constant expressions over
 * their flat component arrays: component-wise for the elementwise builtins
 * (with scalar broadcast), proper vector math for length/distance/dot/cross/
 * normalize/faceforward/reflect/refract, element-wise for matrixCompMult,
 * per-component predicates for the lessThan/equal/notEqual family. Returns
 * the flat result (column-major for matrices), or undefined when the builtin
 * has no fold implementation here or an argument is non-constant. User
 * function calls (incl. builtin names with user overloads) never fold.
 */
function evalBuiltinConstFold(e: CallExpr, name: string, scope: Scope, ctx: SemContext): (number | boolean)[] | undefined {
  const sym = scope.lookup(name);
  if (sym !== undefined && sym.kind === 'fn') {
    // A user function is never a constant expression. For a builtin name with
    // user overloads the analysis resolution is not recorded on the node, so
    // bail out conservatively (old behavior: not foldable).
    if (!sym.builtin) return undefined;
    if (sym.siblings.some((s) => !s.builtin)) return undefined;
  }
  if (!MATH_BUILTIN_NAMES.has(name)) return undefined;
  const args: ConstArg[] = [];
  for (const a of e.args) {
    const at = a.resolvedType;
    if (at === undefined || at.kind === 'sampler' || at.kind === 'void') return undefined;
    const av = evalConstExpr(a, scope, ctx);
    if (av === undefined || av.length === 0) return undefined;
    args.push({ data: av, base: baseOf(at) });
  }
  const ret = e.resolvedType;
  if (ret === undefined || ret.kind === 'sampler' || ret.kind === 'void') return undefined;
  const outBase: 'float' | 'int' | 'uint' | 'bool' =
    ret.kind === 'scalar' || ret.kind === 'vector' ? ret.base : 'float';
  const n = ret.kind === 'scalar' ? 1 : ret.kind === 'vector' ? ret.size : ret.kind === 'matrix' ? ret.cols * ret.rows : -1;
  if (n < 0) return undefined;
  switch (name) {
    // ---- scalar-result vector math ----
    case 'length': {
      const xs = numArg(args[0]);
      if (xs === null) return undefined;
      return [Math.sqrt(dotN(xs, xs))];
    }
    case 'distance': {
      const xs = numArg(args[0]);
      const ys = numArg(args[1]);
      if (xs === null || ys === null || xs.length !== ys.length) return undefined;
      let s = 0;
      for (let i = 0; i < xs.length; i++) s += (xs[i] - ys[i]) * (xs[i] - ys[i]);
      return [Math.sqrt(s)];
    }
    case 'dot': {
      const xs = numArg(args[0]);
      const ys = numArg(args[1]);
      if (xs === null || ys === null || xs.length !== ys.length) return undefined;
      return [dotN(xs, ys)];
    }
    case 'any': {
      const d = args[0].data;
      return [d.some((v) => v === true)];
    }
    case 'all': {
      const d = args[0].data;
      return [d.every((v) => v === true)];
    }
    // ---- vector-result vector math ----
    case 'cross': {
      const xs = numArg(args[0]);
      const ys = numArg(args[1]);
      if (xs === null || ys === null || xs.length !== 3 || ys.length !== 3) return undefined;
      return [
        xs[1] * ys[2] - xs[2] * ys[1],
        xs[2] * ys[0] - xs[0] * ys[2],
        xs[0] * ys[1] - xs[1] * ys[0],
      ];
    }
    case 'normalize': {
      const xs = numArg(args[0]);
      if (xs === null) return undefined;
      const l = Math.sqrt(dotN(xs, xs));
      return l === 0 ? xs.map(() => 0) : xs.map((v) => v / l);
    }
    case 'faceforward': {
      const nv = numArg(args[0]);
      const iv = numArg(args[1]);
      const nr = numArg(args[2]);
      if (nv === null || iv === null || nr === null || iv.length !== nr.length) return undefined;
      return dotN(nr, iv) < 0 ? nv : nv.map((v) => -v);
    }
    case 'reflect': {
      const iv = numArg(args[0]);
      const nv = numArg(args[1]);
      if (iv === null || nv === null || iv.length !== nv.length) return undefined;
      const f = 2 * dotN(nv, iv);
      return nv.map((v, i) => iv[i] - f * v);
    }
    case 'refract': {
      const iv = numArg(args[0]);
      const nv = numArg(args[1]);
      const eta = numArg(args[2]);
      if (iv === null || nv === null || eta === null || iv.length !== nv.length || eta.length !== 1) return undefined;
      const e = eta[0];
      const d = dotN(nv, iv);
      const k = 1 - e * e * (1 - d * d);
      if (k < 0) return nv.map(() => 0);
      const c = e * d + Math.sqrt(k);
      return nv.map((v, i) => e * iv[i] - c * v);
    }
    // ---- element-wise / component-wise family ----
    case 'radians': return foldElem(args, n, ([x]) => (x * Math.PI) / 180, outBase);
    case 'degrees': return foldElem(args, n, ([x]) => (x * 180) / Math.PI, outBase);
    case 'sin': return foldElem(args, n, ([x]) => Math.sin(x), outBase);
    case 'cos': return foldElem(args, n, ([x]) => Math.cos(x), outBase);
    case 'tan': return foldElem(args, n, ([x]) => Math.tan(x), outBase);
    case 'asin': return foldElem(args, n, ([x]) => Math.asin(x), outBase);
    case 'acos': return foldElem(args, n, ([x]) => Math.acos(x), outBase);
    case 'atan':
      return args.length === 2
        ? foldElem(args, n, ([y, x]) => Math.atan2(y, x), outBase)
        : foldElem(args, n, ([x]) => Math.atan(x), outBase);
    case 'exp': return foldElem(args, n, ([x]) => Math.exp(x), outBase);
    case 'log': return foldElem(args, n, ([x]) => Math.log(x), outBase);
    case 'exp2': return foldElem(args, n, ([x]) => Math.pow(2, x), outBase);
    case 'log2': return foldElem(args, n, ([x]) => Math.log2(x), outBase);
    case 'sqrt': return foldElem(args, n, ([x]) => Math.sqrt(x), outBase);
    case 'inversesqrt': return foldElem(args, n, ([x]) => 1 / Math.sqrt(x), outBase);
    case 'abs': return foldElem(args, n, ([x]) => Math.abs(x), outBase);
    case 'sign': return foldElem(args, n, ([x]) => Math.sign(x), outBase);
    case 'floor': return foldElem(args, n, ([x]) => Math.floor(x), outBase);
    case 'ceil': return foldElem(args, n, ([x]) => Math.ceil(x), outBase);
    case 'fract': return foldElem(args, n, ([x]) => x - Math.floor(x), outBase);
    case 'trunc': return foldElem(args, n, ([x]) => Math.trunc(x), outBase);
    case 'round': return foldElem(args, n, ([x]) => Math.sign(x) * Math.floor(Math.abs(x) + 0.5), outBase);
    case 'roundEven': return foldElem(args, n, ([x]) => roundEvenN(x), outBase);
    case 'pow': return foldElem(args, n, ([x, y]) => Math.pow(x, y), outBase);
    case 'mod': return foldElem(args, n, ([x, y]) => x - y * Math.floor(x / y), outBase);
    case 'min': return foldElem(args, n, ([x, y]) => Math.min(x, y), outBase);
    case 'max': return foldElem(args, n, ([x, y]) => Math.max(x, y), outBase);
    case 'step': return foldElem(args, n, ([edge, x]) => (x < edge ? 0 : 1), outBase);
    case 'clamp': return foldElem(args, n, ([x, lo, hi]) => Math.min(Math.max(x, lo), hi), outBase);
    case 'mix': {
      if (args.length === 3 && args[2].base === 'bool') {
        // ESSL 3.00 selector form mix(x, y, bvec): b ? y : x.
        const sel = args[2].data;
        const out: (number | boolean)[] = [];
        for (let i = 0; i < n; i++) {
          const b = sel.length === 1 ? sel[0] : sel[i];
          const x = compAt(args[0], i);
          const y = compAt(args[1], i);
          if (typeof x !== 'number' || typeof y !== 'number' || typeof b !== 'boolean') return undefined;
          out.push(b ? y : x);
        }
        return out;
      }
      return foldElem(args, n, ([x, y, a]) => x * (1 - a) + y * a, outBase);
    }
    case 'smoothstep': return foldElem(args, n, ([e0, e1, x]) => smoothstepN(e0, e1, x), outBase);
    case 'matrixCompMult': return foldElem(args, n, ([a, b]) => a * b, outBase);
    case 'not': {
      const d = args[0].data;
      const out: (number | boolean)[] = [];
      for (let i = 0; i < n; i++) {
        const v = d.length === 1 ? d[0] : d[i];
        if (typeof v !== 'boolean') return undefined;
        out.push(!v);
      }
      return out;
    }
    case 'lessThan': return foldPred(args, n, (a, b) => a < b);
    case 'lessThanEqual': return foldPred(args, n, (a, b) => a <= b);
    case 'greaterThan': return foldPred(args, n, (a, b) => a > b);
    case 'greaterThanEqual': return foldPred(args, n, (a, b) => a >= b);
    case 'equal': return foldPred(args, n, (a, b) => a === b);
    case 'notEqual': return foldPred(args, n, (a, b) => a !== b);
    default:
      return undefined; // math builtins without a fold here (pack/bit ops etc.)
  }
}

/** Round half to even (GLSL roundEven). */
function roundEvenN(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/** GLSL smoothstep. */
function smoothstepN(e0: number, e1: number, x: number): number {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

/* ------------------------------------------------------------------ */
/* Global variable initializer validation (ANGLE parity)               */
/* ------------------------------------------------------------------ */

/**
 * ANGLE ValidateGlobalInitializer parity (WebGL CTS global-variable-init.html).
 * Global variable initializers need NOT be constant expressions — uniforms
 * and other globals are allowed for legacy WebGL1 compatibility — but must
 * not reference per-invocation state or non-math calls:
 * - symbol references: const / plain-global / uniform allowed in WebGL1;
 *   ESSL 3.00 allows ONLY const symbols (plus builtin gl_Max* constants);
 *   attributes, varyings and builtin non-constants (gl_FragCoord, ...) error;
 * - calls: constructors and MATH builtins are allowed; user function calls
 *   and all other builtins (texture lookups, derivatives, pack/bit ops) error;
 * - assignments (incl. pre/post ++/--) error; every other operator is fine
 *   (its children are walked).
 * Reports one error at the first offending node; the initializer must already
 * be ANALYZED (resolvedType annotated). Called from declareVariables for
 * non-const GLOBAL declarations (scope.parent === null).
 */
export function validateGlobalInit(e: Expr, scope: Scope, ctx: SemContext): void {
  if (e.resolvedType === undefined) return; // analysis already failed
  switch (e.kind) {
    case 'literal':
      return;
    case 'identifier': {
      const sym = scope.lookup(e.name);
      if (sym === undefined) return; // analysis already failed
      switch (sym.kind) {
        case 'var':
          if (sym.storage === 'const') return;
          // Plain globals and uniforms: WebGL1 legacy compatibility; ESSL
          // 3.00 has no legacy content to deal with → strict.
          if ((sym.storage === 'uniform' || sym.storage === undefined) && ctx.version === 100) return;
          break;
        case 'builtin-const':
          return; // gl_Max* constants
        case 'builtin-var':
          break; // gl_FragCoord, gl_Position, ... → error
        default:
          return; // fn/struct names cannot appear as values (analysis failed)
      }
      break;
    }
    case 'member':
      validateGlobalInit(e.object, scope, ctx);
      return;
    case 'index':
      validateGlobalInit(e.object, scope, ctx);
      validateGlobalInit(e.index, scope, ctx);
      return;
    case 'unary':
      if (e.op === '++' || e.op === '--') break; // lvalue modification → error
      validateGlobalInit(e.operand, scope, ctx);
      return;
    case 'binary':
      validateGlobalInit(e.left, scope, ctx);
      validateGlobalInit(e.right, scope, ctx);
      return;
    case 'assign':
      break; // lvalue modification → error
    case 'ternary':
      validateGlobalInit(e.cond, scope, ctx);
      validateGlobalInit(e.whenTrue, scope, ctx);
      validateGlobalInit(e.whenFalse, scope, ctx);
      return;
    case 'comma':
      for (const x of e.exprs) validateGlobalInit(x, scope, ctx);
      return;
    case 'call':
      validateGlobalInitCall(e, scope, ctx);
      return;
    default:
      return;
  }
  ctx.error(e.loc.line, "'=' : global variable initializers must be constant expressions");
}

/** A call inside a global initializer: constructors and math builtins are
 *  allowed; user function calls and non-math builtins are rejected. */
function validateGlobalInitCall(e: CallExpr, scope: Scope, ctx: SemContext): void {
  const callee = e.callee;
  if (callee.kind === 'index') {
    // T[N](...) array constructor — constant-constructible, walk args.
    for (const a of e.args) validateGlobalInit(a, scope, ctx);
    return;
  }
  if (callee.kind !== 'identifier') {
    for (const a of e.args) validateGlobalInit(a, scope, ctx);
    return;
  }
  const name = callee.name;
  if (builtinType(name, ctx.version) !== undefined || isStructName(name, scope)) {
    // Constructor call — walk args.
    for (const a of e.args) validateGlobalInit(a, scope, ctx);
    return;
  }
  // Function call: user functions (incl. builtin names with user overloads —
  // the resolution is not recorded on the node, bail out conservatively) and
  // non-math builtins (texture lookups, derivatives, ...) are rejected.
  const sym = scope.lookup(name);
  const userOverloaded = sym !== undefined && sym.kind === 'fn' && sym.siblings.some((s) => !s.builtin);
  if (userOverloaded || !MATH_BUILTIN_NAMES.has(name)) {
    ctx.error(e.loc.line, "'=' : global variable initializers must be constant expressions");
    return;
  }
  for (const a of e.args) validateGlobalInit(a, scope, ctx);
}
