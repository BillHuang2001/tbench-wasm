/**
 * parser-expr.ts — expression parsing for GLSL ES 1.00 / 3.00.
 *
 * Pipeline position: expressions are parsed by parser.ts / parser-stmt.ts
 * (declarator initializers, array dimensions, statement expressions,
 * if/while/for conditions, switch case values, layout(...) values). This
 * module is intentionally dependency-free of the other parser modules (it
 * only type-imports the Parser class), so parser.ts can import it without
 * creating a module cycle. It also owns the shared keyword sets and token
 * helpers the other parser files use.
 *
 * Precedence ladder (low → high), per GLSL ES 1.00 §5 / ES 3.00 §5:
 *   comma (left) → assignment (right) → ternary → || → ^^ (1.00 only) → &&
 *   → | → ^ → & → ==/!= → </>/<=/>= → <</>> → +/- → (* / %) → unary
 *   (+ - ! ~ ++ -- prefix) → postfix (++ --, calls, indexing, member
 *   access) → primary (literals, identifiers, type-name constructors, parens).
 *
 * Version rules enforced here:
 * - Bitwise binary/assignment operators and `~` in a 1.00 shader are
 *   rejected with "'X' : bitwise operators require GLSL ES 3.00" (the lexer
 *   emits the tokens in both versions; the parser is the gate).
 * - `^^` is 1.00-only; in 3.00 the lexer splits it into two `^` tokens, so
 *   the grammar rejects it naturally (no special case needed here).
 * - 1.00 reserved words used in expression position produce a
 *   "'X' is reserved in GLSL ES 1.00" error.
 * - Type-name keywords are accepted as primary expressions (constructors:
 *   `vec4(...)`, and the ES 3.00 array-constructor form `float[3](...)`
 *   which parses naturally as CallExpr(IndexExpr(IdentifierExpr('float'),
 *   LiteralExpr(3)), ...) — semantics resolves the callee).
 *
 * Recovery: on an unexpected token the parser emits an error, consumes the
 * token and returns a placeholder identifier expression (name ''), letting
 * the enclosing statement/declaration machinery skip to the next `;`/`}`.
 */
import type { Token } from './lexer.js';
import type { Parser } from './parser.js';
import type { AssignOp, BinaryOp, Expr, Loc, UnaryOp } from './ast.js';

/* ------------------------------------------------------------------ */
/* Shared keyword sets / helpers (also used by parser.ts, parser-stmt.ts) */
/* ------------------------------------------------------------------ */

/** EOF sentinel token text (never a real operator). */
export const EOF = '\u0000';

/** Sampler type names (valid precision-declaration bases). */
export const SAMPLER_KEYWORDS: ReadonlySet<string> = new Set([
  'sampler2D', 'samplerCube', 'sampler3D', 'sampler2DArray',
  'sampler2DShadow', 'samplerCubeShadow', 'sampler2DArrayShadow',
  'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray',
  'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray',
]);

/** All type-name keywords of both versions (sampler3D etc. are reserved in 100, checked separately). */
export const TYPE_KEYWORDS: ReadonlySet<string> = new Set([
  'void', 'float', 'int', 'uint', 'bool',
  'vec2', 'vec3', 'vec4',
  'ivec2', 'ivec3', 'ivec4',
  'uvec2', 'uvec3', 'uvec4',
  'bvec2', 'bvec3', 'bvec4',
  'mat2', 'mat3', 'mat4',
  'mat2x2', 'mat2x3', 'mat2x4',
  'mat3x2', 'mat3x3', 'mat3x4',
  'mat4x2', 'mat4x3', 'mat4x4',
  ...SAMPLER_KEYWORDS,
]);

/**
 * Words that are keyword tokens in a 1.00 shader but reserved (not usable):
 * GLSL ES 1.00 §3.6 reserved words that the lexer classifies as keywords.
 * Using them is a compile error with a dedicated message.
 */
export const RESERVED_100: ReadonlySet<string> = new Set([
  'switch', 'case', 'default', 'flat', 'sampler3D', 'sampler2DShadow',
  'in', 'out', 'inout',
]);

/** Keywords that can begin a declaration (statement or global scope). */
export const DECL_KEYWORDS: ReadonlySet<string> = new Set([
  'const', 'uniform', 'attribute', 'varying', 'in', 'out', 'inout',
  'invariant', 'layout', 'centroid', 'smooth', 'noperspective', 'precise', 'flat',
  'struct', 'highp', 'mediump', 'lowp',
  ...TYPE_KEYWORDS,
]);

const ASSIGN_OPS: ReadonlySet<string> = new Set([
  '=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '^=', '|=',
]);

const BITWISE_BINARY: ReadonlySet<string> = new Set(['&', '|', '^', '<<', '>>']);
const BITWISE_ASSIGN: ReadonlySet<string> = new Set(['<<=', '>>=', '&=', '^=', '|=']);
const UNARY_OPS: ReadonlySet<string> = new Set(['+', '-', '!', '~', '++', '--']);

/** Binary operator precedences (higher binds tighter). */
const BINARY_PRECEDENCE: Readonly<Record<string, number>> = {
  '||': 1,
  '^^': 2,
  '&&': 3,
  '|': 4,
  '^': 5,
  '&': 6,
  '==': 7, '!=': 7,
  '<': 8, '>': 8, '<=': 8, '>=': 8,
  '<<': 9, '>>': 9,
  '+': 10, '-': 10,
  '*': 11, '/': 11, '%': 11,
};

/** Human-readable token description for error messages. */
export function describeToken(t: Token): string {
  switch (t.kind) {
    case 'identifier':
    case 'keyword':
      return `'${t.name}'`;
    case 'int':
    case 'uint':
    case 'float':
      return `'${t.value}'`;
    case 'op':
      return t.text === EOF ? 'end of file' : `'${t.text}'`;
  }
}

/** Source position of a token. */
export function locOf(t: Token): Loc {
  return { line: t.line, column: t.column };
}

/** Placeholder expression used for error recovery (always accompanies an error). */
function dummyExpr(t: Token): Expr {
  return { kind: 'identifier', name: '', loc: locOf(t) };
}

/* ------------------------------------------------------------------ */
/* Expression grammar                                                  */
/* ------------------------------------------------------------------ */

/** Full expression: comma-separated assignment expressions. */
export function parseExpression(p: Parser): Expr {
  const first = parseAssignmentExpr(p);
  if (!p.atOp(',')) return first;
  const exprs: Expr[] = first.kind === 'comma' ? first.exprs.slice() : [first];
  while (p.atOp(',')) {
    p.next();
    exprs.push(parseAssignmentExpr(p));
  }
  return { kind: 'comma', exprs, loc: exprs[0].loc };
}

/** Assignment expression (right-associative). */
export function parseAssignmentExpr(p: Parser): Expr {
  const target = parseConditionalExpr(p);
  const t = p.peek();
  if (t.kind === 'op' && ASSIGN_OPS.has(t.text)) {
    p.next();
    if (p.version === 100 && BITWISE_ASSIGN.has(t.text)) {
      p.error(t.line, `'${t.text}' : bitwise operators require GLSL ES 3.00`);
    }
    const value = parseAssignmentExpr(p);
    return { kind: 'assign', op: t.text as AssignOp, target, value, loc: target.loc };
  }
  return target;
}

/** Ternary `cond ? expr : assignment-expr` (middle allows commas). */
function parseConditionalExpr(p: Parser): Expr {
  const cond = parseBinaryExpr(p, 1);
  if (!p.atOp('?')) return cond;
  p.next();
  const whenTrue = parseExpression(p);
  p.expectOp(':');
  const whenFalse = parseAssignmentExpr(p);
  return { kind: 'ternary', cond, whenTrue, whenFalse, loc: cond.loc };
}

/** Binary expression with precedence climbing (left-associative). */
function parseBinaryExpr(p: Parser, minPrec: number): Expr {
  let left = parseUnaryExpr(p);
  for (;;) {
    const t = p.peek();
    if (t.kind !== 'op') break;
    const prec = BINARY_PRECEDENCE[t.text];
    if (prec === undefined || prec < minPrec) break;
    p.next();
    if (p.version === 100 && BITWISE_BINARY.has(t.text)) {
      p.error(t.line, `'${t.text}' : bitwise operators require GLSL ES 3.00`);
    }
    const right = parseBinaryExpr(p, prec + 1);
    left = { kind: 'binary', op: t.text as BinaryOp, left, right, loc: left.loc };
  }
  return left;
}

/** Prefix unary expression. */
function parseUnaryExpr(p: Parser): Expr {
  const t = p.peek();
  if (t.kind === 'op' && UNARY_OPS.has(t.text)) {
    p.next();
    if (t.text === '~' && p.version === 100) {
      p.error(t.line, `'~' : bitwise operators require GLSL ES 3.00`);
    }
    const operand = parseUnaryExpr(p);
    return { kind: 'unary', op: t.text as UnaryOp, operand, loc: locOf(t) };
  }
  return parsePostfixExpr(p);
}

/** Postfix chain: indexing, member access, calls, postfix ++/--. */
function parsePostfixExpr(p: Parser): Expr {
  let e = parsePrimaryExpr(p);
  for (;;) {
    const t = p.peek();
    if (t.kind !== 'op') break;
    if (t.text === '[') {
      p.next();
      const index = parseExpression(p);
      p.expectOp(']');
      e = { kind: 'index', object: e, index, loc: e.loc };
    } else if (t.text === '.') {
      p.next();
      const nameT = p.expectIdentifier();
      e = { kind: 'member', object: e, name: nameT ? nameT.name : '', loc: e.loc };
    } else if (t.text === '(') {
      p.next();
      const args: Expr[] = [];
      if (!p.atOp(')')) {
        for (;;) {
          args.push(parseAssignmentExpr(p));
          if (p.atOp(',')) {
            p.next();
            continue;
          }
          break;
        }
      }
      p.expectOp(')');
      e = { kind: 'call', callee: e, args, loc: e.loc };
    } else if (t.text === '++' || t.text === '--') {
      p.next();
      e = { kind: 'unary', op: t.text, operand: e, loc: e.loc };
    } else {
      break;
    }
  }
  return e;
}

/** Primary expressions: literals, identifiers, type constructors, parens. */
function parsePrimaryExpr(p: Parser): Expr {
  const t = p.peek();
  switch (t.kind) {
    case 'int':
      p.next();
      return { kind: 'literal', value: t.value, literalType: 'int', loc: locOf(t) };
    case 'uint':
      p.next();
      return { kind: 'literal', value: t.value, literalType: 'uint', loc: locOf(t) };
    case 'float':
      p.next();
      return { kind: 'literal', value: t.value, literalType: 'float', loc: locOf(t) };
    case 'identifier':
      p.next();
      return { kind: 'identifier', name: t.name, loc: locOf(t) };
    case 'keyword': {
      if (t.name === 'true' || t.name === 'false') {
        p.next();
        return { kind: 'literal', value: t.name === 'true', literalType: 'bool', loc: locOf(t) };
      }
      if (p.version === 100 && RESERVED_100.has(t.name)) {
        p.error(t.line, `'${t.name}' is reserved in GLSL ES 1.00`);
        p.next();
        return dummyExpr(t);
      }
      if (TYPE_KEYWORDS.has(t.name)) {
        // Type name used as a constructor: `vec4(...)`, `float[3](...)`, ...
        p.next();
        return { kind: 'identifier', name: t.name, loc: locOf(t) };
      }
      p.error(t.line, `syntax error, unexpected '${t.name}'`);
      p.next();
      return dummyExpr(t);
    }
    case 'op': {
      if (t.text === '(') {
        p.next();
        const inner = parseExpression(p);
        p.expectOp(')');
        return inner;
      }
      p.error(t.line, `syntax error, unexpected ${describeToken(t)}`);
      p.next();
      return dummyExpr(t);
    }
  }
}
