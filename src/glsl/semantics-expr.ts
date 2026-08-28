/**
 * semantics-expr.ts — expression semantic analysis for GLSL ES 1.00 / 3.00.
 *
 * `analyzeExpr` annotates every expression node in place:
 * - `resolvedType` — the GLSLType of the expression (undefined on error).
 * - `constValue` — folded scalar constant (number for numeric, boolean for
 *   bool; undefined when not constant). Only SCALAR constants are folded
 *   (vectors/matrices/arrays/structs have no constValue representation).
 * - `lvalue` — true when the expression can be an assignment/++/-- target.
 *
 * Covers: literals, identifier resolution (locals → globals → builtin
 * variables → gl_Max* constants, with stage filtering and writability),
 * unary/binary operators (STRICT operand typing — no implicit conversions:
 * binary ops, assignments, initializers, function calls, ternary and struct
 * constructors require exact type matches in both ES 1.00 and ES 3.00;
 * explicit conversions happen only in type constructors), assignment (incl.
 * compound), ternary, indexing,
 * struct member access + vector swizzles, and calls (constructors, builtin
 * overload resolution with per-argument scoring, user functions with
 * recursion edges recorded on `ctx.currentFunction.calls`).
 *
 * The runtime dependency on `builtinType` from semantics.js is a benign
 * import cycle (function declarations only — never called at module init).
 */
import type {
  AssignExpr, BinaryExpr, BinaryOp, CallExpr, CommaExpr, Expr, IdentifierExpr,
  IndexExpr, LiteralExpr, Loc, MemberExpr, TernaryExpr, UnaryExpr,
} from './ast.js';
import type { BaseScalar, GLSLType } from './types.js';
import { typeEquals, typeName } from './types.js';
import type { BuiltinSignature } from './builtins/index.js';
import { builtinSignatures, extensionFunctions, matches } from './builtins/index.js';
import type { FnSymbol, Scope, SemContext, StructSymbol, VarSymbol } from './semantics.js';
import { builtinType } from './semantics.js';
import { evalConstExpr, flatSize } from './semantics-const.js';

/* ------------------------------------------------------------------ */
/* Implicit conversions (shared with declaration/statement analysis)   */
/* ------------------------------------------------------------------ */

/**
 * The common base of two scalar bases after implicit conversion, or null.
 * ES 1.00: int→float only. ES 3.00: int→uint, int→float, uint→float.
 * bool NEVER converts implicitly.
 *
 * Used ONLY by foldBinary (constant folding of already-validated same-base
 * operand pairs) — binary operators themselves never promote (see sameBase).
 */
function commonBase(a: BaseScalar, b: BaseScalar, version: 100 | 300): BaseScalar | null {
  if (a === b) return a;
  if (version === 100) {
    if ((a === 'int' && b === 'float') || (a === 'float' && b === 'int')) return 'float';
    return null;
  }
  if (a === 'float' || b === 'float') return 'float';
  if ((a === 'int' && b === 'uint') || (a === 'uint' && b === 'int')) return 'uint';
  return null;
}

/**
 * Strict same-base check for binary operators: both operands must have the
 * SAME base — NO implicit promotion. GLSL ES 1.00 §5.9 operator rules require
 * operands of identical type (e.g. `1.0 + 1` is an error), and ES 3.00/3.20
 * §4 state "There are no implicit conversions between types" (so `1u + 2` is
 * also an error). Version-independent.
 */
function sameBase(a: BaseScalar, b: BaseScalar, version: 100 | 300): BaseScalar | null {
  void version;
  return a === b ? a : null;
}

/**
 * True when `from` can be implicitly converted to `to` in `version`.
 * STRICT in both versions: only an exact type match. GLSL ES 1.00 §5.8
 * (assignment/initializers), §6.1 ("No promotion or demotion of the input
 * argument types is done") and ES 3.00 §4 ("There are no implicit conversions
 * between types") forbid implicit scalar/vector promotions — the 65 CTS
 * `conformance/glsl/implicit/*` pages all expect compile FAIL for int→float
 * in binary ops, assignments, initializers, function calls, ternary and
 * struct constructors. Explicit conversions happen only in constructors
 * (ctorBaseConvertible).
 */
export function convertible(from: GLSLType, to: GLSLType, version: 100 | 300): boolean {
  void version;
  return typeEquals(from, to);
}

/** Convert a folded scalar constant from one base to another (constructor + implicit conversions).
 *  Exported for the const-expression evaluator (semantics-const.ts). */
export function convertConst(v: number | boolean, from: BaseScalar, to: BaseScalar): number | boolean {
  if (typeof v === 'boolean') {
    return to === 'bool' ? v : v ? 1 : 0;
  }
  if (from === to) return v;
  switch (from) {
    case 'int':
      switch (to) {
        case 'float': return v;
        case 'uint': return v >>> 0;
        case 'bool': return v !== 0;
      }
      break;
    case 'uint':
      switch (to) {
        case 'float': return v;
        case 'int': return v | 0;
        case 'bool': return v !== 0;
      }
      break;
    case 'float':
      switch (to) {
        case 'int': return Math.trunc(v) | 0;
        case 'uint': return Math.trunc(v) >>> 0;
        case 'bool': return v !== 0;
      }
      break;
    case 'bool':
      break; // handled above
  }
  return v;
}

/* ------------------------------------------------------------------ */
/* Small type predicates                                               */
/* ------------------------------------------------------------------ */

function isNumericScalarOrVector(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && t.base !== 'bool';
}

function isIntScalarOrVector(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && (t.base === 'int' || t.base === 'uint');
}

/** The base of a scalar/vector (fold paths only ever see scalars); float otherwise. */
function scalarBase(t: GLSLType): BaseScalar {
  return t.kind === 'scalar' || t.kind === 'vector' ? t.base : 'float';
}

/** Common type of two numeric scalars/vectors (shape must match), or null.
 * Bases must be IDENTICAL (no implicit promotion — ES 1.00/3.00 §5.9);
 * float-scalar × float-vector and int-scalar × int-vector stay legal. */
function sameShapeType(lt: GLSLType, rt: GLSLType, version: 100 | 300): GLSLType | null {
  if (!isNumericScalarOrVector(lt) || !isNumericScalarOrVector(rt)) return null;
  if (lt.kind === 'scalar' && rt.kind === 'scalar') {
    const b = sameBase(lt.base, rt.base, version);
    return b === null ? null : { kind: 'scalar', base: b };
  }
  if (lt.kind === 'vector' && rt.kind === 'vector') {
    if (lt.size !== rt.size) return null;
    const b = sameBase(lt.base, rt.base, version);
    return b === null ? null : { kind: 'vector', base: b, size: lt.size };
  }
  // scalar + vector (component-wise application)
  if (lt.kind === 'vector' && rt.kind === 'scalar') {
    const b = sameBase(lt.base, rt.base, version);
    return b === null ? null : { kind: 'vector', base: b, size: lt.size };
  }
  if (lt.kind === 'scalar' && rt.kind === 'vector') {
    const b = sameBase(rt.base, lt.base, version);
    return b === null ? null : { kind: 'vector', base: b, size: rt.size };
  }
  return null;
}

/**
 * Result type of the arithmetic binary operators (+ - * /). Returns null when
 * the operand combination is not legal; the caller reports the error.
 * Matrix rules: scalar applied component-wise; matrix±matrix same dims;
 * matrix*matrix: (matC2xR2)*(matC1xR1) legal iff R1==C2 → matC2xR1
 * (ES 1.00 has only square matrices, so this reduces to same-dims);
 * matrix*vector: matCxR * vecR → vecC; vector*matrix: vecC * matCxR → vecR.
 */
function arithmeticType(op: BinaryOp, lt: GLSLType, rt: GLSLType, version: 100 | 300): GLSLType | null {
  const ss = sameShapeType(lt, rt, version);
  if (ss !== null) return ss;
  if (lt.kind === 'matrix' || rt.kind === 'matrix') {
    if (op === '*') {
      if (lt.kind === 'scalar' && rt.kind === 'matrix') return lt.base === 'float' ? rt : null;
      if (lt.kind === 'matrix' && rt.kind === 'scalar') return rt.base === 'float' ? lt : null;
      if (lt.kind === 'matrix' && rt.kind === 'matrix') {
        if (lt.rows !== rt.cols) return null;
        return { kind: 'matrix', cols: rt.cols, rows: lt.rows };
      }
      if (lt.kind === 'matrix' && rt.kind === 'vector') {
        if (rt.base !== 'float' || rt.size !== lt.rows) return null;
        return { kind: 'vector', base: 'float', size: lt.cols };
      }
      if (lt.kind === 'vector' && rt.kind === 'matrix') {
        if (lt.base !== 'float' || lt.size !== rt.cols) return null;
        return { kind: 'vector', base: 'float', size: rt.rows };
      }
      return null;
    }
    // + - / with matrices: same dims, or scalar applied component-wise
    if (lt.kind === 'matrix' && rt.kind === 'matrix') {
      if (lt.cols !== rt.cols || lt.rows !== rt.rows) return null;
      return lt;
    }
    if (lt.kind === 'matrix' && rt.kind === 'scalar') return rt.base === 'float' ? lt : null;
    if (lt.kind === 'scalar' && rt.kind === 'matrix') return lt.base === 'float' ? rt : null;
    return null;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Constant folding                                                    */
/* ------------------------------------------------------------------ */

/** Round half to even (GLSL roundEven). */
function roundEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0), 1);
  return t * t * (3 - 2 * t);
}

/** Fold a binary operator with constant scalar operands; undefined when not foldable.
 *  Exported for the const-expression evaluator (semantics-const.ts). */
export function foldBinary(
  op: BinaryOp,
  lt: GLSLType,
  rt: GLSLType,
  lv: number | boolean,
  rv: number | boolean,
  version: 100 | 300,
): number | boolean | undefined {
  switch (op) {
    case '&&': return typeof lv === 'boolean' && typeof rv === 'boolean' ? lv && rv : undefined;
    case '||': return typeof lv === 'boolean' && typeof rv === 'boolean' ? lv || rv : undefined;
    case '^^': return typeof lv === 'boolean' && typeof rv === 'boolean' ? lv !== rv : undefined;
    case '<':
    case '>':
    case '<=':
    case '>=':
    case '==':
    case '!=': {
      if (typeof lv !== 'number' || typeof rv !== 'number') {
        if ((op === '==' || op === '!=') && typeof lv === 'boolean' && typeof rv === 'boolean') {
          return op === '==' ? lv === rv : lv !== rv;
        }
        return undefined;
      }
      const b = commonBase(scalarBase(lt), scalarBase(rt), version);
      if (b === null) return undefined;
      const a = convertConst(lv, scalarBase(lt), b) as number;
      const c = convertConst(rv, scalarBase(rt), b) as number;
      switch (op) {
        case '<': return a < c;
        case '>': return a > c;
        case '<=': return a <= c;
        case '>=': return a >= c;
        case '==': return a === c;
        case '!=': return a !== c;
      }
    }
    default: {
      if (typeof lv !== 'number' || typeof rv !== 'number') return undefined;
      let base: BaseScalar;
      if (op === '<<' || op === '>>') base = scalarBase(lt);
      else {
        const b = commonBase(scalarBase(lt), scalarBase(rt), version);
        if (b === null) return undefined;
        base = b;
      }
      const a = convertConst(lv, scalarBase(lt), base) as number;
      const c = convertConst(rv, scalarBase(rt), base) as number;
      switch (op) {
        case '+': return base === 'float' ? a + c : base === 'uint' ? (a + c) >>> 0 : (a + c) | 0;
        case '-': return base === 'float' ? a - c : base === 'uint' ? (a - c) >>> 0 : (a - c) | 0;
        case '*': return base === 'float' ? a * c : base === 'uint' ? (a * c) >>> 0 : (a * c) | 0;
        case '/': return base === 'float' ? a / c : base === 'uint' ? Math.trunc(a / c) >>> 0 : Math.trunc(a / c) | 0;
        case '%': return base === 'uint' ? (a % c) >>> 0 : (a % c) | 0;
        case '<<': return base === 'uint' ? (a << c) >>> 0 : (a << c) | 0;
        case '>>': return base === 'uint' ? a >>> c : a >> c;
        case '&': return base === 'uint' ? (a & c) >>> 0 : (a & c) | 0;
        case '|': return base === 'uint' ? (a | c) >>> 0 : (a | c) | 0;
        case '^': return base === 'uint' ? (a ^ c) >>> 0 : (a ^ c) | 0;
      }
    }
  }
}

/**
 * Fold a builtin function call with all-const arguments (SCALAR path only —
 * the practical const-eval subset). Returns undefined when the call is not in
 * the subset or an argument is not constant. Skip: pack/unpack, floatBitsTo*,
 * vector-typed folds.
 */
function foldBuiltin(name: string, ret: GLSLType, args: Expr[]): number | boolean | undefined {
  if (ret.kind !== 'scalar') return undefined;
  const vals = args.map((a) => a.constValue);
  if (vals.some((v) => v === undefined)) return undefined;
  const x = vals[0] as number;
  const y = vals[1] as number;
  const z = vals[2] as number;
  const w = vals[3] as number;
  switch (name) {
    case 'abs': return ret.base === 'int' ? Math.abs(x) | 0 : Math.abs(x);
    case 'sign': return Math.sign(x);
    case 'floor': return Math.floor(x);
    case 'ceil': return Math.ceil(x);
    case 'trunc': return Math.trunc(x);
    case 'round': return Math.sign(x) * Math.floor(Math.abs(x) + 0.5);
    case 'roundEven': return roundEven(x);
    case 'fract': return x - Math.floor(x);
    case 'mod': return x - y * Math.floor(x / y);
    case 'min': return Math.min(x, y);
    case 'max': return Math.max(x, y);
    case 'clamp': return Math.min(Math.max(x, y), z);
    case 'mix': return x * (1 - z) + y * z;
    case 'step': return x < y ? 1 : 0; // step(edge, x)
    case 'smoothstep': return smoothstep(x, y, z);
    case 'sin': return Math.sin(x);
    case 'cos': return Math.cos(x);
    case 'tan': return Math.tan(x);
    case 'asin': return Math.asin(x);
    case 'acos': return Math.acos(x);
    case 'atan': return vals.length === 2 ? Math.atan2(x, y) : Math.atan(x);
    case 'pow': return Math.pow(x, y);
    case 'exp': return Math.exp(x);
    case 'log': return Math.log(x);
    case 'exp2': return Math.pow(2, x);
    case 'log2': return Math.log2(x);
    case 'sqrt': return Math.sqrt(x);
    case 'inversesqrt': return 1 / Math.sqrt(x);
    case 'radians': return (x * Math.PI) / 180;
    case 'degrees': return (x * 180) / Math.PI;
    case 'isnan': return Number.isNaN(x);
    case 'isinf': return !Number.isFinite(x);
    // scalar-path vector functions (a scalar IS a 1-component vector)
    case 'length': return Math.abs(x);
    case 'distance': return Math.abs(x - y);
    case 'dot': return x * y;
    case 'normalize': return x / Math.abs(x);
    // bitfield helpers (ES 3.00) — signedness follows the RESULT type
    case 'bitCount': {
      const v = x >>> 0;
      let c = 0;
      for (let i = 0; i < 32; i++) c += (v >>> i) & 1;
      return c;
    }
    case 'findLSB': {
      const v = x >>> 0;
      if (v === 0) return -1;
      for (let i = 0; i < 32; i++) if ((v >>> i) & 1) return i;
      return -1;
    }
    case 'findMSB': {
      const at = args[0].resolvedType;
      if (at !== undefined && at.kind === 'scalar' && at.base === 'int') {
        const v = x | 0;
        if (v === 0) return -1;
        if (v < 0) return 31; // sign bit
        return 31 - Math.clz32(v);
      }
      const v = x >>> 0;
      if (v === 0) return -1;
      return 31 - Math.clz32(v);
    }
    case 'bitfieldExtract': {
      const offset = y | 0;
      const bits = z | 0;
      if (bits <= 0) return 0;
      const isInt = ret.base === 'int';
      const v = isInt ? x | 0 : x >>> 0;
      const mask = bits >= 32 ? 0xffffffff : ((1 << bits) - 1) >>> 0;
      let r = (v >>> offset) & mask;
      if (isInt && bits < 32 && (r & (1 << (bits - 1))) !== 0) r = (r | ~mask) >>> 0;
      return isInt ? r | 0 : r >>> 0;
    }
    case 'bitfieldInsert': {
      const offset = z | 0;
      const bits = w | 0;
      const isInt = ret.base === 'int';
      const b = isInt ? x | 0 : x >>> 0;
      const ins = isInt ? y | 0 : y >>> 0;
      const mask = bits >= 32 ? 0xffffffff : ((1 << bits) - 1) >>> 0;
      const r = ((b & ~((mask << offset) >>> 0)) | ((ins & mask) << offset)) >>> 0;
      return isInt ? r | 0 : r >>> 0;
    }
  }
  return undefined;
}

/* ------------------------------------------------------------------ */
/* Expression analysis                                                 */
/* ------------------------------------------------------------------ */

/**
 * Analyze one expression in place. Annotates `resolvedType` on every node,
 * `constValue` on foldable scalar constants and `lvalue` on assignables.
 * Errors are pushed onto ctx (capped); analysis continues with best effort.
 */
export function analyzeExpr(e: Expr, scope: Scope, ctx: SemContext): void {
  if (e.resolvedType !== undefined) return; // already analyzed
  switch (e.kind) {
    case 'literal': {
      e.resolvedType = { kind: 'scalar', base: e.literalType };
      e.constValue = e.value;
      e.lvalue = false;
      return;
    }
    case 'identifier':
      analyzeIdentifier(e, scope, ctx);
      return;
    case 'unary':
      analyzeUnary(e, scope, ctx);
      return;
    case 'binary':
      analyzeBinary(e, scope, ctx);
      return;
    case 'assign':
      analyzeAssign(e, scope, ctx);
      return;
    case 'ternary':
      analyzeTernary(e, scope, ctx);
      return;
    case 'index':
      analyzeIndex(e, scope, ctx);
      return;
    case 'member':
      analyzeMember(e, scope, ctx);
      return;
    case 'call':
      analyzeCall(e, scope, ctx);
      return;
    case 'comma': {
      for (const x of e.exprs) analyzeExpr(x, scope, ctx);
      const last = e.exprs[e.exprs.length - 1];
      if (last !== undefined && last.resolvedType !== undefined) {
        e.resolvedType = last.resolvedType;
        e.constValue = last.constValue;
      }
      e.lvalue = false; // comma result is never an lvalue
      return;
    }
  }
}

function analyzeIdentifier(e: IdentifierExpr, scope: Scope, ctx: SemContext): void {
  const sym = scope.lookup(e.name);
  if (sym === undefined) {
    if (builtinType(e.name, ctx.version) !== undefined) {
      ctx.error(e.loc.line, `'${e.name}' : type name used as a value`);
    } else {
      ctx.error(e.loc.line, `'${e.name}' : undeclared identifier`);
    }
    return;
  }
  switch (sym.kind) {
    case 'var':
      e.resolvedType = sym.type;
      e.lvalue = varIsWritable(sym, ctx, scope);
      if (sym.constValue !== undefined) e.constValue = sym.constValue;
      // GLOBAL const AGGREGATES have no codegen storage — every read must
      // fold at semantics. Mutate the identifier node in place into an
      // annotated constructor-call node (callee = type name; args = constant
      // literal/ctor nodes) so codegen emits the constructor computation.
      // Scalar consts fold via constValue (above); LOCAL consts are ordinary
      // JS locals in codegen and are never rewritten.
      if (
        sym.constData !== undefined &&
        sym.type.kind !== 'scalar' &&
        isGlobalSymbol(sym, scope)
      ) {
        mutateConstUse(e, sym.type, sym.constData);
      }
      return;
    case 'builtin-var':
      if (sym.stage !== 'BOTH' && sym.stage !== ctx.stage) {
        ctx.error(e.loc.line, `'${e.name}' : not available in ${ctx.stage === 'VERTEX' ? 'fragment' : 'vertex'} shaders`);
        return;
      }
      e.resolvedType = sym.type;
      e.lvalue = sym.writable;
      return;
    case 'builtin-const':
      e.resolvedType = sym.type;
      e.constValue = sym.value;
      e.lvalue = false;
      return;
    case 'fn':
      ctx.error(e.loc.line, `'${e.name}' : function name used as a value`);
      return;
    case 'struct':
      ctx.error(e.loc.line, `'${e.name}' : type name used as a value`);
      return;
  }
}

/**
 * Writability of a variable symbol as an lvalue (assignment / ++ / -- / out
 * or inout argument). GLSL ES 1.00 §4.3.3/§4.3.5/§4.3.6: uniforms, attributes
 * and fragment-stage varyings are read-only; `const` variables are read-only.
 * ES 3.00 `in` GLOBAL inputs (vertex attributes / fragment varyings) are
 * read-only, but `in` PARAMETERS are writable local copies (GLSL ES §6.1.1 —
 * the ogles CorrectFull_vert writes to plain `in` params). Vertex-stage
 * varyings (1.00) and ES 3.00 `out` variables (vertex varying / fragment
 * output) ARE writable. Storage `undefined` = plain variable. Scope
 * distinguishes params (non-global) from global `in` inputs.
 */
function varIsWritable(sym: VarSymbol, ctx: SemContext, scope: Scope): boolean {
  switch (sym.storage) {
    case undefined:
      return true;
    case 'const':
    case 'uniform':
    case 'attribute':
      return false;
    case 'in':
      return scope.parent !== null; // param copy (writable) vs global input (read-only)
    case 'varying':
      return ctx.stage === 'VERTEX';
    case 'out':
      return true;
    default:
      return true; // 'inout' never appears on variables
  }
}

/**
 * True when a (possibly nested/arrayed) struct type contains a sampler
 * member — sampler-typed struct members make the struct non-assignable and
 * non-comparable (GLSL ES: samplers may only be uniform variables/params).
 */
function containsSampler(t: GLSLType): boolean {
  switch (t.kind) {
    case 'sampler':
      return true;
    case 'struct':
      return t.members.some((m) => containsSampler(m.type));
    case 'array':
      return containsSampler(t.element);
    default:
      return false;
  }
}

/**
 * True when a (possibly nested) type contains an array member — GLSL ES 1.00
 * §5.7/§5.8: "The assignment and equality operators are not defined for
 * structures that contain arrays or sampler types"; "structures containing
 * arrays ... may not be used as the target of an assignment". The CTS
 * struct-assign/struct-equals pages require struct-with-array ASSIGNMENT to
 * fail to compile at BOTH versions; struct-with-array ==/!= is rejected at
 * version 100 only — ES 3.00 compares element-wise, recursively (CTS
 * compare-structs-containing-arrays.html) — so callers gate on ctx.version.
 */
function containsArray(t: GLSLType): boolean {
  switch (t.kind) {
    case 'array':
      return true;
    case 'struct':
      return t.members.some((m) => containsArray(m.type));
    default:
      return false;
  }
}

function analyzeUnary(e: UnaryExpr, scope: Scope, ctx: SemContext): void {
  analyzeExpr(e.operand, scope, ctx);
  const t = e.operand.resolvedType;
  if (t === undefined) return;
  if (t.kind === 'void') {
    ctx.error(e.operand.loc.line, `'${e.op}' : cannot apply a unary operator to a void expression`);
    return;
  }
  switch (e.op) {
    case '+':
    case '-': {
      if (!isNumericScalarOrVector(t) && t.kind !== 'matrix') {
        ctx.error(e.loc.line, `'${e.op}' : operand must be a numeric scalar, vector or matrix`);
        return;
      }
      e.resolvedType = t;
      if (typeof e.operand.constValue === 'number') {
        const v = e.operand.constValue;
        e.constValue = e.op === '-' ? (t.kind === 'scalar' && t.base === 'uint' ? (-v) >>> 0 : -v) : v;
      }
      return;
    }
    case '!': {
      if (!(t.kind === 'scalar' && t.base === 'bool')) {
        ctx.error(e.loc.line, `'!' : operand must be a boolean scalar`);
        return;
      }
      e.resolvedType = t;
      if (typeof e.operand.constValue === 'boolean') e.constValue = !e.operand.constValue;
      return;
    }
    case '~': {
      if (!isIntScalarOrVector(t)) {
        ctx.error(e.loc.line, `'~' : operand must be an integer scalar or vector`);
        return;
      }
      e.resolvedType = t;
      if (typeof e.operand.constValue === 'number') {
        e.constValue = t.kind === 'scalar' && t.base === 'uint' ? (~e.operand.constValue) >>> 0 : (~e.operand.constValue) | 0;
      }
      return;
    }
    case '++':
    case '--': {
      if (e.operand.lvalue !== true) {
        ctx.error(e.loc.line, `'${e.op}' : operand must be an lvalue`);
        return;
      }
      // GLSL ES 1.00 §5.9: ++/-- apply to numeric scalars, VECTORS and
      // MATRICES (the ogles CorrectFull_vert shader increments vec2/mat2 —
      // the ES 1.00 spec added vector/matrix increments over desktop GLSL).
      if (!isNumericScalarOrVector(t) && t.kind !== 'matrix') {
        ctx.error(e.loc.line, `'${e.op}' : operand must be a numeric scalar, vector or matrix`);
        return;
      }
      e.resolvedType = t;
      e.lvalue = false; // ++/-- results are rvalues
      return;
    }
  }
}

function analyzeBinary(e: BinaryExpr, scope: Scope, ctx: SemContext): void {
  analyzeExpr(e.left, scope, ctx);
  analyzeExpr(e.right, scope, ctx);
  const lt = e.left.resolvedType;
  const rt = e.right.resolvedType;
  if (lt === undefined || rt === undefined) return;
  if (lt.kind === 'void' || rt.kind === 'void') {
    ctx.error(e.loc.line, `'${e.op}' : cannot apply operator to a void expression`);
    return;
  }
  switch (e.op) {
    case '+':
    case '-':
    case '*':
    case '/': {
      const t = arithmeticType(e.op, lt, rt, ctx.version);
      if (t === null) {
        ctx.error(e.loc.line, `'${e.op}' : operands of type '${typeName(lt)}' and '${typeName(rt)}' are incompatible`);
        return;
      }
      e.resolvedType = t;
      if (e.left.constValue !== undefined && e.right.constValue !== undefined) {
        e.constValue = foldBinary(e.op, lt, rt, e.left.constValue, e.right.constValue, ctx.version);
      }
      return;
    }
    case '%': {
      if (!isIntScalarOrVector(lt) || !isIntScalarOrVector(rt)) {
        ctx.error(e.loc.line, `'%' : operands must be integers`);
        return;
      }
      const t = sameShapeType(lt, rt, ctx.version);
      if (t === null) {
        ctx.error(e.loc.line, `'%' : operands of type '${typeName(lt)}' and '${typeName(rt)}' are incompatible`);
        return;
      }
      e.resolvedType = t;
      if (e.left.constValue !== undefined && e.right.constValue !== undefined) {
        e.constValue = foldBinary(e.op, lt, rt, e.left.constValue, e.right.constValue, ctx.version);
      }
      return;
    }
    case '<':
    case '>':
    case '<=':
    case '>=': {
      if (!(lt.kind === 'scalar' && rt.kind === 'scalar') || lt.base === 'bool' || rt.base === 'bool') {
        ctx.error(e.loc.line, `'${e.op}' : relational operators require scalar numeric operands`);
        return;
      }
      if (lt.base !== rt.base) {
        ctx.error(e.loc.line, `'${e.op}' : operands of type '${typeName(lt)}' and '${typeName(rt)}' are incompatible`);
        return;
      }
      e.resolvedType = { kind: 'scalar', base: 'bool' };
      if (e.left.constValue !== undefined && e.right.constValue !== undefined) {
        e.constValue = foldBinary(e.op, lt, rt, e.left.constValue, e.right.constValue, ctx.version);
      }
      return;
    }
    case '==':
    case '!=': {
      let ok = false;
      // Same base required (bool==bool legal; int==float and int==uint are not).
      if (lt.kind === 'scalar' && rt.kind === 'scalar') ok = sameBase(lt.base, rt.base, ctx.version) !== null;
      else if (lt.kind === 'vector' && rt.kind === 'vector' && lt.size === rt.size) {
        ok = sameBase(lt.base, rt.base, ctx.version) !== null;
      } else if (lt.kind === 'matrix' && rt.kind === 'matrix' && lt.cols === rt.cols && lt.rows === rt.rows) ok = true;
      // GLSL ES 1.00 §5.9 / ES 3.00 §5.9: equality is defined for STRUCTS of
      // the same type (result: one bool). ES 3.00 §5.9 ALSO defines equality
      // for ARRAYS (element-wise, same element type + same size); ES 1.00
      // arrays are never comparable. Structs containing a sampler are NOT
      // comparable at ANY version (samplers may not be struct members per
      // §4.1.7 — the CTS struct-equals page requires the comparison to fail
      // to compile). Structs containing ARRAYS: rejected at version 100
      // (GLSL ES 1.00 §5.9; CTS struct-equals.html), ALLOWED at version 300
      // (element-wise, recursively — the CTS compare-structs-containing-
      // arrays.html page requires `b == c` on structs with array members to
      // compile and render correctly).
      else if (lt.kind === 'struct' && rt.kind === 'struct') {
        if (containsSampler(lt)) {
          ctx.error(e.loc.line, `'${e.op}' : cannot compare structs containing a sampler`);
          return;
        }
        if (ctx.version === 100 && containsArray(lt)) {
          ctx.error(e.loc.line, `'${e.op}' : cannot compare structs containing an array`);
          return;
        }
        ok = typeEquals(lt, rt);
      } else if (lt.kind === 'array' && rt.kind === 'array') {
        if (ctx.version === 300 && typeEquals(lt, rt)) {
          // ES 3.00: array equality compares ELEMENT-wise. Struct elements
          // are comparable at 300 (element-wise, recursively — CTS compare-
          // structs-containing-arrays.html); version 100 never reaches this
          // block (outer guard) and falls through to the generic
          // 'cannot be compared' error. Sampler elements stay rejected.
          // Scalar/vector/matrix elements need no extra check. NOT
          // const-folded — array-typed nodes carry no scalar constValue, and
          // the comparison is a runtime element-wise op (codegen's concern).
          const el = lt.element;
          if (el.kind === 'struct' && containsSampler(el)) {
            ctx.error(e.loc.line, `'${e.op}' : cannot compare structs containing a sampler`);
            return;
          }
          ok = true;
        }
        // version 100 (or a 300 size/element mismatch) → ok stays false and
        // the 'cannot be compared' error below fires.
      }
      if (!ok) {
        ctx.error(e.loc.line, `'${e.op}' : operands of type '${typeName(lt)}' and '${typeName(rt)}' cannot be compared`);
        return;
      }
      e.resolvedType = { kind: 'scalar', base: 'bool' };
      if (e.left.constValue !== undefined && e.right.constValue !== undefined) {
        e.constValue = foldBinary(e.op, lt, rt, e.left.constValue, e.right.constValue, ctx.version);
      } else if (lt.kind === 'struct') {
        // Struct operands are aggregates (no scalar constValue): fold the
        // whole comparison via the const-expression evaluator, which compares
        // the two flattened component lists component-wise. The single bool
        // result annotates the node — codegen folds ANY scalar-constValue node
        // (expressions.ts emitExpr), so const-folded struct equality needs no
        // codegen support. Non-const struct comparisons stay unfolded here
        // (runtime emit is codegen's concern; see the codegen contract).
        const data = evalConstExpr(e, scope, ctx);
        if (data !== undefined && data.length === 1 && typeof data[0] === 'boolean') {
          e.constValue = data[0];
        }
      }
      return;
    }
    case '&&':
    case '||':
    case '^^': {
      if (!(lt.kind === 'scalar' && lt.base === 'bool' && rt.kind === 'scalar' && rt.base === 'bool')) {
        ctx.error(e.loc.line, `'${e.op}' : logical operators require boolean scalar operands`);
        return;
      }
      e.resolvedType = { kind: 'scalar', base: 'bool' };
      if (e.left.constValue !== undefined && e.right.constValue !== undefined) {
        e.constValue = foldBinary(e.op, lt, rt, e.left.constValue, e.right.constValue, ctx.version);
      }
      return;
    }
    case '&':
    case '|':
    case '^': {
      if (!isIntScalarOrVector(lt) || !isIntScalarOrVector(rt)) {
        ctx.error(e.loc.line, `'${e.op}' : bitwise operators require integer operands`);
        return;
      }
      const t = sameShapeType(lt, rt, ctx.version);
      if (t === null) {
        ctx.error(e.loc.line, `'${e.op}' : operands of type '${typeName(lt)}' and '${typeName(rt)}' are incompatible`);
        return;
      }
      e.resolvedType = t;
      if (e.left.constValue !== undefined && e.right.constValue !== undefined) {
        e.constValue = foldBinary(e.op, lt, rt, e.left.constValue, e.right.constValue, ctx.version);
      }
      return;
    }
    case '<<':
    case '>>': {
      if (!isIntScalarOrVector(lt) || !(rt.kind === 'scalar' && (rt.base === 'int' || rt.base === 'uint'))) {
        ctx.error(e.loc.line, `'${e.op}' : left operand must be an integer scalar/vector and right operand an integer scalar`);
        return;
      }
      e.resolvedType = lt; // result is the LEFT operand's type
      if (e.left.constValue !== undefined && e.right.constValue !== undefined) {
        e.constValue = foldBinary(e.op, lt, rt, e.left.constValue, e.right.constValue, ctx.version);
      }
      return;
    }
  }
}

function analyzeAssign(e: AssignExpr, scope: Scope, ctx: SemContext): void {
  analyzeExpr(e.target, scope, ctx);
  analyzeExpr(e.value, scope, ctx);
  const tt = e.target.resolvedType;
  const vt = e.value.resolvedType;
  if (tt === undefined || vt === undefined) return;
  if (e.target.lvalue !== true) {
    ctx.error(e.loc.line, `'${e.op}' : lvalue required`);
    return;
  }
  if (e.op === '=') {
    if (!convertible(vt, tt, ctx.version)) {
      ctx.error(e.value.loc.line, `cannot convert from '${typeName(vt)}' to '${typeName(tt)}'`);
      return;
    }
    // GLSL ES 3.00 §5.8: whole-array assignment is legal (element + size
    // match already enforced by convertible/typeEquals above). ES 1.00 has
    // no array assignment — arrays may not be assignment targets there.
    if (tt.kind === 'array') {
      if (ctx.version === 100) {
        ctx.error(e.loc.line, `'=' : cannot assign to an array in GLSL ES 1.00`);
        return;
      }
    } else {
      // GLSL ES: structs containing a sampler or an ARRAY cannot be assigned
      // (samplers may not be struct members; structures containing arrays may
      // not be assignment targets — spec §5.7/§5.8; the CTS struct-assign
      // page requires both to fail to compile, in 100 AND 300).
      if (containsSampler(tt)) {
        ctx.error(e.loc.line, `'=' : cannot assign a struct containing a sampler`);
        return;
      }
      if (containsArray(tt)) {
        ctx.error(e.loc.line, `'=' : cannot assign a struct containing an array`);
        return;
      }
    }
    e.resolvedType = tt;
    if (e.target.constValue !== undefined && e.value.constValue !== undefined) {
      e.constValue = convertConst(e.value.constValue, scalarBase(vt), scalarBase(tt));
    }
  } else {
    const core = e.op.slice(0, -1) as BinaryOp;
    const t = arithmeticType(core, tt, vt, ctx.version);
    if (t === null) {
      ctx.error(e.loc.line, `'${e.op}' : operands of type '${typeName(tt)}' and '${typeName(vt)}' are incompatible`);
      return;
    }
    if (!typeEquals(t, tt)) {
      ctx.error(e.loc.line, `cannot convert from '${typeName(t)}' to '${typeName(tt)}'`);
      return;
    }
    e.resolvedType = tt;
    if (e.target.constValue !== undefined && e.value.constValue !== undefined) {
      e.constValue = foldBinary(core, tt, vt, e.target.constValue, e.value.constValue, ctx.version);
    }
  }
  e.lvalue = false;
}

function analyzeTernary(e: TernaryExpr, scope: Scope, ctx: SemContext): void {
  analyzeExpr(e.cond, scope, ctx);
  analyzeExpr(e.whenTrue, scope, ctx);
  analyzeExpr(e.whenFalse, scope, ctx);
  const ct = e.cond.resolvedType;
  const tt = e.whenTrue.resolvedType;
  const ft = e.whenFalse.resolvedType;
  if (ct === undefined || tt === undefined || ft === undefined) return;
  if (!(ct.kind === 'scalar' && ct.base === 'bool')) {
    ctx.error(e.cond.loc.line, `'?:' : condition must be a boolean expression`);
    return;
  }
  if (tt.kind === 'void' || ft.kind === 'void') {
    ctx.error(e.loc.line, `'?:' : cannot use a void expression as a ternary operand`);
    return;
  }
  // GLSL ES: ternary arms must be of the SAME type (ES 1.00 §5.8 / ES 3.00
  // §5.9 — no implicit conversion between the arms; the 65 CTS implicit
  // pages include ternary_int_float/ternary_ivec*_vec* expecting FAIL).
  let t: GLSLType | null = null;
  if (tt.kind === 'scalar' && ft.kind === 'scalar') {
    if (typeEquals(tt, ft)) t = tt;
  } else if (tt.kind === 'vector' && ft.kind === 'vector' && tt.size === ft.size) {
    if (typeEquals(tt, ft)) t = tt;
  }
  if (t === null) {
    ctx.error(e.loc.line, `'?:' : operands of type '${typeName(tt)}' and '${typeName(ft)}' are incompatible`);
    return;
  }
  e.resolvedType = t;
  if (typeof e.cond.constValue === 'boolean') {
    const arm = e.cond.constValue ? e.whenTrue : e.whenFalse;
    if (arm.constValue !== undefined) e.constValue = arm.constValue;
  }
  e.lvalue = false;
}

function analyzeIndex(e: IndexExpr, scope: Scope, ctx: SemContext): void {
  analyzeExpr(e.object, scope, ctx);
  analyzeExpr(e.index, scope, ctx);
  const ot = e.object.resolvedType;
  const it = e.index.resolvedType;
  if (ot === undefined || it === undefined) return;
  if (!(it.kind === 'scalar' && (it.base === 'int' || it.base === 'uint'))) {
    ctx.error(e.index.loc.line, `'[' : index must be an integer`);
    return;
  }
  const cv = e.index.constValue;
  const constIdx = typeof cv === 'number' && Number.isInteger(cv) ? cv : null;
  switch (ot.kind) {
    case 'array': {
      if (constIdx !== null && ot.size !== null && (constIdx < 0 || constIdx >= ot.size)) {
        ctx.error(e.index.loc.line, `'[' : array index ${constIdx} out of range [0, ${ot.size - 1}]`);
        return;
      }
      e.resolvedType = ot.element;
      e.lvalue = e.object.lvalue === true;
      break;
    }
    case 'vector': {
      if (constIdx !== null && (constIdx < 0 || constIdx >= ot.size)) {
        ctx.error(e.index.loc.line, `'[' : component index ${constIdx} out of range [0, ${ot.size - 1}]`);
        return;
      }
      e.resolvedType = { kind: 'scalar', base: ot.base };
      e.lvalue = e.object.lvalue === true;
      break;
    }
    case 'matrix': {
      if (constIdx !== null && (constIdx < 0 || constIdx >= ot.cols)) {
        ctx.error(e.index.loc.line, `'[' : column index ${constIdx} out of range [0, ${ot.cols - 1}]`);
        return;
      }
      e.resolvedType = { kind: 'vector', base: 'float', size: ot.rows };
      e.lvalue = e.object.lvalue === true;
      break;
    }
    default:
      ctx.error(e.loc.line, `'[' : cannot index a value of type '${typeName(ot)}'`);
      return;
  }
  // Post-analysis const folding: a constant-index read of a const object
  // (global or local — never an lvalue) folds to a scalar constValue or
  // mutates into a ctor call.
  foldConstRead(e, scope, ctx);
}

/** Swizzle component sets (exported for the const-expression evaluator). */
export const SWIZZLE_SETS: readonly string[] = ['xyzw', 'rgba', 'stpq'];

/** Swizzle result for a vector: { type, noDupes } or { error }. */
function swizzleInfo(base: BaseScalar, size: number, name: string):
  | { type: GLSLType; noDupes: boolean }
  | { error: string } {
  if (name.length > 4) return { error: `swizzle '${name}' has more than 4 components` };
  let set: string | null = null;
  const indices: number[] = [];
  for (const ch of name) {
    let found = false;
    for (const s of SWIZZLE_SETS) {
      const i = s.indexOf(ch);
      if (i >= 0) {
        if (set !== null && set !== s) return { error: `swizzle '${name}' mixes component sets` };
        set = s;
        indices.push(i);
        found = true;
        break;
      }
    }
    if (!found) return { error: `'${ch}' : invalid swizzle component` };
  }
  for (const i of indices) {
    if (i >= size) return { error: `swizzle component out of range for a ${size}-component vector` };
  }
  const noDupes = new Set(indices).size === indices.length;
  if (indices.length === 1) return { type: { kind: 'scalar', base }, noDupes: true };
  return { type: { kind: 'vector', base, size: indices.length as 2 | 3 | 4 }, noDupes };
}

function analyzeMember(e: MemberExpr, scope: Scope, ctx: SemContext): void {
  analyzeExpr(e.object, scope, ctx);
  const ot = e.object.resolvedType;
  if (ot === undefined) return;
  if (ot.kind === 'void') {
    ctx.error(e.loc.line, `'.' : cannot access a member of a void expression`);
    return;
  }
  if (ot.kind === 'struct') {
    const m = ot.members.find((mm) => mm.name === e.name);
    if (m === undefined) {
      ctx.error(e.loc.line, `'${e.name}' : no member named '${e.name}' in struct '${ot.name}'`);
      return;
    }
    e.resolvedType = m.type;
    e.lvalue = e.object.lvalue === true;
  } else if (ot.kind === 'vector') {
    const r = swizzleInfo(ot.base, ot.size, e.name);
    if ('error' in r) {
      ctx.error(e.loc.line, r.error);
      return;
    }
    e.resolvedType = r.type;
    e.lvalue = r.noDupes && e.object.lvalue === true;
  } else {
    ctx.error(e.loc.line, `'.' : cannot access a member of type '${typeName(ot)}'`);
    return;
  }
  // Post-analysis const folding: a member read of a const (global or local —
  // never an lvalue) folds to a scalar constValue or mutates into a ctor call.
  foldConstRead(e, scope, ctx);
}

/* ------------------------------------------------------------------ */
/* Const-expression use-site folding (BUGS 2+5)                        */
/* ------------------------------------------------------------------ */

/** True when `sym` is declared in the GLOBAL scope (locals are ordinary JS
 *  codegen locals and are never rewritten; global consts have no storage). */
function isGlobalSymbol(sym: VarSymbol, scope: Scope): boolean {
  let s: Scope | null = scope;
  while (s !== null && s.parent !== null) s = s.parent;
  return s !== null && s.lookupLocal(sym.name) === sym;
}

/**
 * Post-analysis const folding for member/index reads. Runs AFTER the normal
 * analysis of the node: evaluates the full expression chain; a SCALAR result
 * sets `constValue` on the outer node (codegen folds ANY node kind with a
 * scalar constValue — expressions.ts emitExpr), an aggregate result mutates
 * the outer node into an annotated constructor-call node built from the
 * sliced components. Chained reads (s11.ss.i) fold at the outer node; inner
 * mutated nodes are never emitted (the outer node folds before descent).
 * Non-const chains (uniform/attribute reads, dynamic indices, lvalues) are
 * untouched — evalConstExpr returns undefined on any non-const leaf.
 */
function foldConstRead(e: MemberExpr | IndexExpr, scope: Scope, ctx: SemContext): void {
  if (e.resolvedType === undefined) return; // analysis failed
  const data = evalConstExpr(e, scope, ctx);
  if (data === undefined) return;
  const t = e.resolvedType;
  if (t.kind === 'scalar') {
    e.constValue = data[0];
  } else if (t.kind === 'vector' || t.kind === 'matrix' || t.kind === 'struct' || t.kind === 'array') {
    mutateConstUse(e, t, data);
  }
}

/**
 * Mutate a fully-analyzed const expression node (identifier/member/index —
 * consts are never lvalues, so emitLValue is unreachable on them) IN PLACE
 * into an annotated constructor-call node: callee = the type name (array
 * types use the `T[N](...)` IndexExpr callee form), args = constant
 * literal/ctor nodes, all with resolvedType/constValue set. Codegen lowers
 * the call via emitConstructorCall/emitArrayCtor (codegen/expr-ctor.ts) —
 * constant args emit as literals with constant duals, so dual mode is
 * correct with zero codegen changes.
 */
function mutateConstUse(e: Expr, type: GLSLType, data: (number | boolean)[]): void {
  const loc = e.loc;
  const callee: Expr =
    type.kind === 'array'
      ? {
          kind: 'index',
          loc,
          object: { kind: 'identifier', name: typeName(type.element), loc },
          index: { kind: 'literal', value: type.size ?? 0, literalType: 'int', loc },
        }
      : { kind: 'identifier', name: typeName(type), loc };
  const call: CallExpr = {
    kind: 'call',
    loc,
    callee,
    args: buildConstCtorArgs(type, data, loc),
    resolvedType: type,
    lvalue: false,
  };
  Object.assign(e, call);
}

/** Constructor arguments for `type` from flat components: one literal per
 *  scalar/vector/matrix component, one node per struct member / array
 *  element (recursively built). */
function buildConstCtorArgs(type: GLSLType, data: (number | boolean)[], loc: Loc): Expr[] {
  switch (type.kind) {
    case 'scalar':
      return [constLiteral(data[0], type.base, loc)];
    case 'vector': {
      const args: Expr[] = [];
      for (let i = 0; i < type.size; i++) args.push(constLiteral(data[i], type.base, loc));
      return args;
    }
    case 'matrix': {
      const args: Expr[] = [];
      for (let i = 0; i < type.cols * type.rows; i++) args.push(constLiteral(data[i], 'float', loc));
      return args;
    }
    case 'struct': {
      const args: Expr[] = [];
      let off = 0;
      for (const m of type.members) {
        const sz = flatSize(m.type);
        args.push(buildConstValueNode(m.type, data.slice(off, off + sz), loc));
        off += sz;
      }
      return args;
    }
    case 'array': {
      const args: Expr[] = [];
      const sz = flatSize(type.element);
      for (let i = 0; i < (type.size ?? 0); i++) {
        args.push(buildConstValueNode(type.element, data.slice(i * sz, (i + 1) * sz), loc));
      }
      return args;
    }
    default:
      return []; // void / sampler: unreachable (no const value)
  }
}

/** A constant expression node for `type` from flat components: a literal for
 *  scalars, a fully annotated ctor-call node for aggregates. */
function buildConstValueNode(type: GLSLType, data: (number | boolean)[], loc: Loc): Expr {
  switch (type.kind) {
    case 'scalar':
      return constLiteral(data[0], type.base, loc);
    case 'vector':
    case 'matrix':
    case 'struct':
    case 'array': {
      const callee: Expr =
        type.kind === 'array'
          ? {
              kind: 'index',
              loc,
              object: { kind: 'identifier', name: typeName(type.element), loc },
              index: { kind: 'literal', value: type.size ?? 0, literalType: 'int', loc },
            }
          : { kind: 'identifier', name: typeName(type), loc };
      const call: CallExpr = {
        kind: 'call',
        loc,
        callee,
        args: buildConstCtorArgs(type, data, loc),
        resolvedType: type,
        lvalue: false,
      };
      return call;
    }
    default:
      return constLiteral(0, 'int', loc); // unreachable
  }
}

/** A fully annotated constant literal node. */
function constLiteral(v: number | boolean, base: BaseScalar, loc: Loc): LiteralExpr {
  return {
    kind: 'literal',
    value: v,
    literalType: base,
    loc,
    resolvedType: { kind: 'scalar', base },
    constValue: v,
    lvalue: false,
  };
}

/* ------------------------------------------------------------------ */
/* Calls: constructors, builtins, user functions                       */
/* ------------------------------------------------------------------ */

/** Is `name` a type name here (builtin type or user struct)? */
function isTypeName(name: string, scope: Scope, ctx: SemContext): boolean {
  if (builtinType(name, ctx.version) !== undefined) return true;
  const s = scope.lookup(name);
  return s !== undefined && s.kind === 'struct';
}

function structTypeOf(name: string, scope: Scope): GLSLType | null {
  const s = scope.lookup(name);
  return s !== undefined && s.kind === 'struct' ? (s as StructSymbol).type : null;
}

/** Base convertibility in CONSTRUCTORS (explicit conversion — bool participates). */
function ctorBaseConvertible(a: BaseScalar, b: BaseScalar, version: 100 | 300): boolean {
  if (a === b) return true;
  if (version === 100) {
    // ES 1.00: int(float), int(bool), float(int), float(bool), bool(int), bool(float)
    if (a === 'uint' || b === 'uint') return false;
    return true;
  }
  // ES 3.00: all 12 conversions among int/uint/float/bool
  return true;
}

function analyzeCall(e: CallExpr, scope: Scope, ctx: SemContext): void {
  for (const a of e.args) {
    analyzeExpr(a, scope, ctx);
    if (a.resolvedType !== undefined && a.resolvedType.kind === 'void') {
      ctx.error(a.loc.line, `'void' : cannot use a void expression as a function argument`);
    }
  }
  const callee = e.callee;
  if (callee.kind === 'identifier') {
    const name = callee.name;
    if (isTypeName(name, scope, ctx)) {
      analyzeConstructor(e, name, scope, ctx);
      return;
    }
    const sym = scope.lookup(name);
    if (sym !== undefined && sym.kind === 'fn' && !sym.builtin) {
      analyzeUserCall(e, sym, ctx);
      return;
    }
    if (sym !== undefined && sym.kind === 'fn' && sym.builtin && sym.siblings.some((s) => !s.builtin)) {
      // Builtin function name with USER overloads (GLSL ES 1.00 §6.1: user
      // functions may overload builtins with different signatures): resolve
      // across BOTH the user signatures and the builtin signature tables.
      analyzeHybridCall(e, sym, ctx);
      return;
    }
    if (matches(name, builtinSignatures(ctx.version)).length > 0 || extensionFunctions.some((s) => s.name === name)) {
      analyzeBuiltinCall(e, name, ctx);
      return;
    }
    ctx.error(e.loc.line, `'${name}' : no matching function`);
    return;
  }
  if (callee.kind === 'index' && callee.object.kind === 'identifier' && isTypeName(callee.object.name, scope, ctx)) {
    analyzeArrayConstructor(e, callee, scope, ctx);
    return;
  }
  // GLSL ES 3.00 §5.9: `.length()` is defined on ARRAYS — and, per the
  // ESSL 3.20 clarification the CTS array-length-side-effects page relies on,
  // on ANY expression of array type (`a.length()`, `(f()).length()`,
  // `(int[1](0)).length()`, `(a = b).length()`). The callee object is
  // analyzed for its side effects even though only its type is used; the
  // result is a plain int and is deliberately NOT const-folded (the call may
  // have side effects — `(f()).length()` must still evaluate f()).
  if (callee.kind === 'member' && callee.name === 'length') {
    analyzeExpr(callee.object, scope, ctx);
    const ot = callee.object.resolvedType;
    if (ot === undefined) {
      ctx.error(e.loc.line, `'(' : invalid function or constructor call`);
      return;
    }
    if (ot.kind !== 'array') {
      ctx.error(e.loc.line, `'.length()' : only defined for arrays`);
      return;
    }
    if (e.args.length !== 0) {
      ctx.error(e.loc.line, `'.length()' : function takes no arguments`);
      return;
    }
    e.resolvedType = { kind: 'scalar', base: 'int' };
    e.lvalue = false;
    return;
  }
  ctx.error(e.loc.line, `'(' : invalid function or constructor call`);
}

/** Score one signature against the analyzed arguments: Σ(0 exact | 1 implicit conversion), or null if no match. */
function scoreSignature(params: GLSLType[], args: Expr[], ctx: SemContext): number | null {
  if (params.length !== args.length) return null;
  let score = 0;
  for (let i = 0; i < params.length; i++) {
    const at = args[i].resolvedType;
    if (at === undefined) return null;
    if (typeEquals(at, params[i])) continue;
    if (convertible(at, params[i], ctx.version)) {
      score += 1;
      continue;
    }
    return null;
  }
  return score;
}

/** Select the best-scoring signature among `candidates`; reports ambiguity. */
function pickBest<T extends { params: GLSLType[]; ret: GLSLType }>(
  candidates: T[],
  args: Expr[],
  ctx: SemContext,
  name: string,
  line: number,
): T | null {
  let best: T | null = null;
  let bestScore = Infinity;
  for (const c of candidates) {
    const sc = scoreSignature(c.params, args, ctx);
    if (sc === null) continue;
    if (sc < bestScore) {
      bestScore = sc;
      best = c;
    }
  }
  if (best === null) return null;
  for (const c of candidates) {
    if (c === best) continue;
    if (scoreSignature(c.params, args, ctx) === bestScore) {
      ctx.error(line, `'${name}' : ambiguous call (multiple matching signatures)`);
      return null;
    }
  }
  return best;
}

function analyzeUserCall(e: CallExpr, sym: FnSymbol, ctx: SemContext): void {
  const sigs = sym.siblings.map((s) => ({
    params: s.params.map((p) => p.type),
    storage: s.params.map((p) => p.storage),
    ret: s.retType,
  }));
  const best = pickBest(sigs, e.args, ctx, sym.name, e.loc.line);
  if (best === null) {
    if (ctx.errors.length === 0 || !ctx.errors[ctx.errors.length - 1].message.includes('ambiguous')) {
      ctx.error(e.loc.line, `'${sym.name}' : no matching function`);
    }
    return;
  }
  // GLSL ES §6.1.1: `out`/`inout` arguments are written by the callee — they
  // must be writable lvalues. Uniforms/attributes/fragment varyings/consts
  // are read-only (their identifiers are not lvalues), so passing one here is
  // an error (ogles build function4: `function(uniformInt)` with `out int i`).
  for (let i = 0; i < e.args.length; i++) {
    const st = best.storage[i];
    if ((st === 'out' || st === 'inout') && e.args[i].lvalue !== true) {
      ctx.error(e.args[i].loc.line, `'${sym.name}' : out/inout argument must be a writable lvalue`);
      return;
    }
  }
  e.resolvedType = best.ret;
  // Recursion-detection edge keyed by the RESOLVED callee signature: calling a
  // DIFFERENT overload of the same name is legal (`process(S1)` calling
  // `process(S2)` — ogles CorrectFuncOverload); only a same-signature self-call
  // (or a call cycle through distinct signatures) is recursion.
  ctx.currentFunction?.calls.add(sym.siblings[sigs.indexOf(best)]);
}

function analyzeBuiltinCall(e: CallExpr, name: string, ctx: SemContext): void {
  const all: BuiltinSignature[] = [...matches(name, builtinSignatures(ctx.version))];
  for (const s of extensionFunctions) {
    if (s.name !== name) continue;
    // Skip extension entries duplicating a core signature exactly (e.g. dFdx
    // is core in 3.00 AND listed for GL_OES_standard_derivatives).
    const dup = all.some(
      (c) =>
        c.params.length === s.params.length &&
        c.params.every((p, i) => typeEquals(p, s.params[i])) &&
        typeEquals(c.ret, s.ret) &&
        c.stage === s.stage,
    );
    if (!dup) all.push(s);
  }
  if (all.length === 0) {
    ctx.error(e.loc.line, `'${name}' : no matching function`);
    return;
  }
  // GL_OES_standard_derivatives functions (dFdx/dFdy/fwidth) are CORE in
  // GLSL ES 3.00 — their extension gate applies to 1.00 shaders only.
  const coreIn300 = (s: BuiltinSignature): boolean =>
    ctx.version === 300 && s.extension === 'GL_OES_standard_derivatives';
  const enabled = all.filter((s) => s.extension === undefined || coreIn300(s) || ctx.enabledExtensions.has(s.extension));
  if (enabled.length === 0) {
    ctx.error(e.loc.line, `'${name}' : requires extension '${all[0].extension}' which is not enabled`);
    return;
  }
  const staged = enabled.filter((s) => s.stage === undefined || s.stage === ctx.stage);
  if (staged.length === 0) {
    ctx.error(e.loc.line, `'${name}' : not available in ${ctx.stage === 'VERTEX' ? 'fragment' : 'vertex'} shaders`);
    return;
  }
  const best = pickBest(staged, e.args, ctx, name, e.loc.line);
  if (best === null) {
    if (ctx.errors.length === 0 || !ctx.errors[ctx.errors.length - 1].message.includes('ambiguous')) {
      ctx.error(e.loc.line, `'${name}' : no matching function`);
    }
    return;
  }
  e.resolvedType = best.ret;
  e.constValue = foldBuiltin(name, best.ret, e.args);
}

/**
 * The builtin signatures of `name` usable in this shader: core table plus
 * extension entries (deduped against core), filtered to enabled extensions
 * and the current stage — the same visibility rules analyzeBuiltinCall
 * applies, as a plain candidate list for hybrid resolution.
 */
function stagedBuiltinSigs(name: string, ctx: SemContext): BuiltinSignature[] {
  const all: BuiltinSignature[] = [...matches(name, builtinSignatures(ctx.version))];
  for (const s of extensionFunctions) {
    if (s.name !== name) continue;
    // Skip extension entries duplicating a core signature exactly (e.g. dFdx
    // is core in 3.00 AND listed for GL_OES_standard_derivatives).
    const dup = all.some(
      (c) =>
        c.params.length === s.params.length &&
        c.params.every((p, i) => typeEquals(p, s.params[i])) &&
        typeEquals(c.ret, s.ret) &&
        c.stage === s.stage,
    );
    if (!dup) all.push(s);
  }
  // GL_OES_standard_derivatives functions (dFdx/dFdy/fwidth) are CORE in
  // GLSL ES 3.00 — their extension gate applies to 1.00 shaders only.
  const coreIn300 = (s: BuiltinSignature): boolean =>
    ctx.version === 300 && s.extension === 'GL_OES_standard_derivatives';
  const enabled = all.filter((s) => s.extension === undefined || coreIn300(s) || ctx.enabledExtensions.has(s.extension));
  return enabled.filter((s) => s.stage === undefined || s.stage === ctx.stage);
}

/**
 * Call to a builtin function name that user code has overloaded: GLSL ES
 * 1.00 allows user-defined functions to overload builtins (any signature
 * differing from every visible builtin signature). Resolution picks the best
 * match across the user signatures AND the builtin tables — the user
 * overload wins on an exact/int-better match, the builtin still serves
 * calls that only it can take (e.g. radians(float) for a float argument).
 */
function analyzeHybridCall(e: CallExpr, sym: FnSymbol, ctx: SemContext): void {
  const name = sym.name;
  const userCands = sym.siblings
    .filter((s) => !s.builtin)
    .map((s) => ({ params: s.params.map((p) => p.type), ret: s.retType, user: true }));
  const builtinCands = stagedBuiltinSigs(name, ctx).map((s) => ({ params: s.params, ret: s.ret, user: false }));
  const best = pickBest([...userCands, ...builtinCands], e.args, ctx, name, e.loc.line);
  if (best === null) {
    if (ctx.errors.length === 0 || !ctx.errors[ctx.errors.length - 1].message.includes('ambiguous')) {
      ctx.error(e.loc.line, `'${name}' : no matching function`);
    }
    return;
  }
  e.resolvedType = best.ret;
  if (best.user) {
    // Recursion-detection edge keyed by the RESOLVED user overload (a call to
    // a different overload of the same name is not a self-edge).
    const userSigs = sym.siblings.filter((s) => !s.builtin);
    ctx.currentFunction?.calls.add(userSigs[userCands.indexOf(best)]);
  } else {
    e.constValue = foldBuiltin(name, best.ret, e.args);
  }
}

/* ------------------------------------------------------------------ */
/* Constructors                                                        */
/* ------------------------------------------------------------------ */

function analyzeConstructor(e: CallExpr, name: string, scope: Scope, ctx: SemContext): void {
  const bt = builtinType(name, ctx.version);
  const t = bt !== undefined ? bt : structTypeOf(name, scope);
  if (t === null) {
    ctx.error(e.loc.line, `'${name}' : not a type`);
    return;
  }
  switch (t.kind) {
    case 'void':
      ctx.error(e.loc.line, `'${name}' : cannot construct void`);
      return;
    case 'scalar': {
      if (e.args.length !== 1) {
        ctx.error(e.loc.line, `'${name}' : constructor requires exactly one argument`);
        return;
      }
      const at = e.args[0].resolvedType;
      if (at === undefined) return;
      // Scalar constructors take the FIRST element of a non-scalar
      // (GLSL ES 1.00 §5.4.1: float(vec3) selects the first component).
      if (at.kind !== 'scalar' && at.kind !== 'vector' && at.kind !== 'matrix') {
        ctx.error(e.args[0].loc.line, `'${name}' : cannot construct from '${typeName(at)}'`);
        return;
      }
      e.resolvedType = t;
      if (at.kind === 'scalar' && e.args[0].constValue !== undefined) {
        e.constValue = convertConst(e.args[0].constValue, at.base, t.base);
      }
      return;
    }
    case 'vector':
      analyzeVectorConstructor(e, t, ctx);
      return;
    case 'matrix':
      analyzeMatrixConstructor(e, t, ctx);
      return;
    case 'struct': {
      if (e.args.length !== t.members.length) {
        ctx.error(e.loc.line, `'${t.name}' : constructor requires ${t.members.length} argument(s)`);
        return;
      }
      for (let i = 0; i < e.args.length; i++) {
        const at = e.args[i].resolvedType;
        if (at === undefined) return;
        if (!convertible(at, t.members[i].type, ctx.version)) {
          ctx.error(e.args[i].loc.line, `cannot convert from '${typeName(at)}' to '${typeName(t.members[i].type)}'`);
          return;
        }
      }
      e.resolvedType = t;
      return;
    }
    case 'sampler':
      ctx.error(e.loc.line, `'${name}' : samplers cannot be constructed`);
      return;
    case 'array':
      return; // unreachable via a bare type name
  }
}

/** Component count an argument contributes to a constructor: scalar → 1,
 * vector → size, matrix → cols*rows (components read in column-major order). */
function ctorArgComponents(at: GLSLType): number {
  switch (at.kind) {
    case 'scalar':
      return 1;
    case 'vector':
      return at.size;
    case 'matrix':
      return at.cols * at.rows;
    default:
      return 0;
  }
}

function analyzeVectorConstructor(
  e: CallExpr,
  t: Extract<GLSLType, { kind: 'vector' }>,
  ctx: SemContext,
): void {
  const args = e.args;
  const targetName = typeName(t);
  const n = t.size;
  if (args.length === 1) {
    const at = args[0].resolvedType;
    if (at === undefined) return;
    if (at.kind === 'scalar') {
      if (!ctorBaseConvertible(at.base, t.base, ctx.version)) {
        ctx.error(args[0].loc.line, `cannot convert from '${typeName(at)}' to '${targetName}'`);
        return;
      }
      e.resolvedType = t; // splat
      return;
    }
  }
  // General form: scalars, vectors and matrices (matrix components are read
  // in column-major order — GLSL ES 1.00 §5.4.2: vecN(matM) = the first N
  // components of the matrix). Constructors can SHORTEN: extra trailing
  // components are dropped, but an argument that contributes no component at
  // all is an error (matches ANGLE and the CTS constructor generator:
  // vec3(5.0, 4.0, ivec2(...)) OK, vec4(v, v, v) rejected).
  let total = 0;
  const comps: number[] = [];
  for (const a of args) {
    const at = a.resolvedType;
    if (at === undefined) return;
    if (at.kind !== 'scalar' && at.kind !== 'vector' && at.kind !== 'matrix') {
      ctx.error(a.loc.line, `'${targetName}' : invalid constructor argument of type '${typeName(at)}'`);
      return;
    }
    const c = ctorArgComponents(at);
    comps.push(c);
    total += c;
  }
  if (total < n) {
    ctx.error(e.loc.line, `'${targetName}' : constructor requires ${n} components`);
    return;
  }
  if (total - comps[comps.length - 1] >= n) {
    ctx.error(e.loc.line, `'${targetName}' : too many arguments for constructor`);
    return;
  }
  for (const a of args) {
    const at = a.resolvedType;
    if (at === undefined) return;
    // Matrix components are float — convertibility is checked against float.
    let abase: BaseScalar;
    if (at.kind === 'scalar' || at.kind === 'vector') abase = at.base;
    else if (at.kind === 'matrix') abase = 'float';
    else return;
    if (!ctorBaseConvertible(abase, t.base, ctx.version)) {
      ctx.error(a.loc.line, `cannot convert from '${typeName(at)}' to '${targetName}'`);
      return;
    }
  }
  e.resolvedType = t;
}

function analyzeMatrixConstructor(
  e: CallExpr,
  t: Extract<GLSLType, { kind: 'matrix' }>,
  ctx: SemContext,
): void {
  const args = e.args;
  const targetName = typeName(t);
  const size = t.cols * t.rows;
  if (args.length === 1) {
    const at = args[0].resolvedType;
    if (at === undefined) return;
    if (at.kind === 'scalar') {
      if (!ctorBaseConvertible(at.base, 'float', ctx.version)) {
        ctx.error(args[0].loc.line, `cannot convert from '${typeName(at)}' to '${targetName}'`);
        return;
      }
      e.resolvedType = t; // diagonal
      return;
    }
    if (at.kind === 'matrix') {
      // matN(matM) of any dimensions: components at corresponding col/row
      // indices are copied, the rest is filled from the identity matrix.
      e.resolvedType = t;
      return;
    }
  }
  // Mixed scalars/vectors. A matrix argument inside a MULTI-argument matrix
  // constructor is an error (GLSL/ANGLE: "constructing matrix from matrix can
  // only take one argument" — CTS generator rejects any such argument list).
  let total = 0;
  const comps: number[] = [];
  for (const a of args) {
    const at = a.resolvedType;
    if (at === undefined) return;
    if (at.kind === 'matrix') {
      ctx.error(a.loc.line, `'${targetName}' : constructing a matrix from a matrix can only take one argument`);
      return;
    }
    if (at.kind !== 'scalar' && at.kind !== 'vector') {
      ctx.error(a.loc.line, `'${targetName}' : invalid constructor argument of type '${typeName(at)}'`);
      return;
    }
    const c = ctorArgComponents(at);
    comps.push(c);
    total += c;
  }
  // Same component-count rules as vectors: enough components AND the last
  // argument must contribute at least one (extra trailing components drop).
  if (total < size || total - comps[comps.length - 1] >= size) {
    ctx.error(e.loc.line, `'${targetName}' : wrong number of arguments for matrix constructor`);
    return;
  }
  for (const a of args) {
    const at = a.resolvedType;
    if (at === undefined) return;
    if (at.kind !== 'scalar' && at.kind !== 'vector') return;
    if (!ctorBaseConvertible(at.base, 'float', ctx.version)) {
      ctx.error(a.loc.line, `cannot convert from '${typeName(at)}' to '${targetName}'`);
      return;
    }
  }
  e.resolvedType = t;
}

/** `T[size](...)` array constructor (ES 3.00 only). `callee` is the IndexExpr. */
function analyzeArrayConstructor(e: CallExpr, callee: IndexExpr, scope: Scope, ctx: SemContext): void {
  if (ctx.version === 100) {
    ctx.error(e.loc.line, `'[' : array constructors require GLSL ES 3.00`);
    return;
  }
  const inner = callee.object;
  if (inner.kind !== 'identifier') return;
  const bt = builtinType(inner.name, ctx.version);
  const elem = bt !== undefined ? bt : structTypeOf(inner.name, scope);
  if (elem === null || elem.kind === 'void' || elem.kind === 'sampler') {
    ctx.error(e.loc.line, `'${inner.name}' : cannot construct an array of this type`);
    return;
  }
  analyzeExpr(callee.index, scope, ctx);
  const sz = callee.index.constValue;
  if (typeof sz !== 'number' || !Number.isInteger(sz) || sz <= 0) {
    ctx.error(callee.index.loc.line, `'[' : array constructor size must be a constant positive integer`);
    return;
  }
  if (e.args.length !== sz) {
    ctx.error(e.loc.line, `'${inner.name}[${sz}]' : constructor requires ${sz} argument(s)`);
    return;
  }
  for (const a of e.args) {
    const at = a.resolvedType;
    if (at === undefined) return;
    if (!convertible(at, elem, ctx.version)) {
      ctx.error(a.loc.line, `cannot convert from '${typeName(at)}' to '${typeName(elem)}'`);
      return;
    }
  }
  e.resolvedType = { kind: 'array', element: elem, size: sz };
}
