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
 * - builtin calls fold only via the analysis-time scalar fold (`e.constValue`);
 *   builtin calls with vector results are NOT constant expressions (GLSL ES
 *   const-expression definition) and evaluate to undefined.
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
      // The comma operator's value is its LAST operand.
      const last = e.exprs[e.exprs.length - 1];
      return last === undefined ? undefined : evalConstExpr(last, scope, ctx);
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
      // A (lt.cols × lt.rows) * B (rt.cols × rt.rows), legal iff lt.rows === rt.cols.
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
  // Builtin / user function call: fold only when the analysis-time scalar
  // fold already produced a value (builtin calls with vector results are not
  // constant expressions per GLSL ES; user calls never are).
  if (e.constValue !== undefined) return [e.constValue];
  return undefined;
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
