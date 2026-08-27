/**
 * parser-stmt.ts — statement parsing for GLSL ES 1.00 / 3.00.
 *
 * Statements: compound, declaration (local), expression (incl. empty `;`),
 * if/else, for (declaration or expression init), while, do-while, switch
 * (ES 3.00 — case/default labels inside the compound body), break/continue/
 * return/discard. Case labels are parsed wherever they appear (semantics
 * validates they are inside a switch).
 *
 * Version rules:
 * - `switch`/`case`/`default`/`flat`/`sampler3D`/`sampler2DShadow`/
 *   `in`/`out`/`inout` are reserved words in 1.00 → "'X' is reserved in
 *   GLSL ES 1.00" + recovery (a 1.00 `switch` is skipped as a whole,
 *   balanced braces, to avoid cascading errors).
 * - Declaration-vs-expression disambiguation: a statement starting with a
 *   type/qualifier keyword (or `identifier identifier`) is a declaration;
 *   `vec4(...)` / `float[3](...)` constructor calls are expressions.
 *
 * Recovery: after a hard error, tokens are skipped to the next `;` (which is
 * consumed) or `}` (left for the enclosing compound), and an EmptyStmt is
 * returned so the enclosing block keeps parsing.
 */
import type { Parser } from './parser.js';
import type {
  CaseLabelStmt, CompoundStmt, DeclStmt, DoWhileStmt, Expr, ExprStmt,
  ForStmt, IfStmt, ReturnStmt, Stmt, SwitchStmt, WhileStmt,
} from './ast.js';
import {
  parseExpression, parseAssignmentExpr,
  DECL_KEYWORDS, RESERVED_100, TYPE_KEYWORDS, locOf,
} from './parser-expr.js';
import { parseTypeSpec, parseDeclarators, skipBalanced } from './parser.js';

/** Parse one statement (always returns a node; errors are collected). */
export function parseStatement(p: Parser): Stmt {
  const t = p.peek();
  if (t.kind === 'op') {
    if (t.text === '{') return parseCompound(p);
    if (t.text === ';') {
      p.next();
      return { kind: 'empty', loc: locOf(t) };
    }
  }
  if (t.kind === 'keyword') {
    switch (t.name) {
      case 'if':
        return parseIfStmt(p);
      case 'for':
        return parseForStmt(p);
      case 'while':
        return parseWhileStmt(p);
      case 'do':
        return parseDoWhileStmt(p);
      case 'switch': {
        if (p.version === 100) {
          p.error(t.line, "'switch' is reserved in GLSL ES 1.00");
          p.next();
          skipBalanced(p, '(', ')');
          skipBalanced(p, '{', '}');
          return { kind: 'empty', loc: locOf(t) };
        }
        return parseSwitchStmt(p);
      }
      case 'break':
        p.next();
        p.expectOp(';');
        return { kind: 'break', loc: locOf(t) };
      case 'continue':
        p.next();
        p.expectOp(';');
        return { kind: 'continue', loc: locOf(t) };
      case 'discard':
        p.next();
        p.expectOp(';');
        return { kind: 'discard', loc: locOf(t) };
      case 'return':
        return parseReturnStmt(p);
      case 'case':
        if (p.version === 100) {
          p.error(t.line, "'case' is reserved in GLSL ES 1.00");
          recoverStatement(p);
          return { kind: 'empty', loc: locOf(t) };
        }
        return parseCaseLabelStmt(p);
      case 'default':
        if (p.version === 100) {
          p.error(t.line, "'default' is reserved in GLSL ES 1.00");
          recoverStatement(p);
          return { kind: 'empty', loc: locOf(t) };
        }
        return parseCaseLabelStmt(p);
      default:
        if (p.version === 100 && RESERVED_100.has(t.name)) {
          p.error(t.line, `'${t.name}' is reserved in GLSL ES 1.00`);
          recoverStatement(p);
          return { kind: 'empty', loc: locOf(t) };
        }
        break;
    }
  }
  if (startsDeclaration(p)) return parseDeclStmt(p);
  return parseExprStmt(p);
}

/** `{ stmt* }` */
export function parseCompound(p: Parser): CompoundStmt {
  const open = p.next(); // '{'
  const body: Stmt[] = [];
  while (!p.atOp('}') && !p.atEof()) {
    body.push(parseStatement(p));
  }
  if (p.atEof()) {
    p.error(p.peek().line, "unterminated block, expected '}'");
  } else {
    p.next(); // '}'
  }
  return { kind: 'compound', body, loc: locOf(open) };
}

/** `if (expr) stmt [else stmt]` — dangling else binds to the nearest if. */
function parseIfStmt(p: Parser): IfStmt {
  const start = p.next(); // 'if'
  p.expectOp('(');
  const cond = parseExpression(p);
  p.expectOp(')');
  const then = parseStatement(p);
  let elseStmt: Stmt | null = null;
  if (p.atKeyword('else')) {
    p.next();
    elseStmt = parseStatement(p);
  }
  return { kind: 'if', cond, then, else: elseStmt, loc: locOf(start) };
}

/** `for (init; cond; update) stmt` — init may be a declaration or expression. */
function parseForStmt(p: Parser): ForStmt {
  const start = p.next(); // 'for'
  p.expectOp('(');
  let init: Stmt | null = null;
  if (p.atOp(';')) {
    p.next();
  } else if (startsDeclaration(p)) {
    init = parseDeclStmt(p);
  } else {
    const e = parseExpression(p);
    p.expectOp(';', "expected ';' after for-loop initializer");
    const es: ExprStmt = { kind: 'expr-stmt', expr: e, loc: e.loc };
    init = es;
  }
  let cond: Expr | null = null;
  if (p.atOp(';')) {
    p.next();
  } else {
    cond = parseExpression(p);
    p.expectOp(';', "expected ';' after for-loop condition");
  }
  let update: Expr | null = null;
  if (!p.atOp(')')) {
    update = parseExpression(p);
  }
  p.expectOp(')');
  const body = parseStatement(p);
  return { kind: 'for', init, cond, update, body, loc: locOf(start) };
}

/** `while (expr) stmt` */
function parseWhileStmt(p: Parser): WhileStmt {
  const start = p.next(); // 'while'
  p.expectOp('(');
  const cond = parseExpression(p);
  p.expectOp(')');
  const body = parseStatement(p);
  return { kind: 'while', cond, body, loc: locOf(start) };
}

/** `do stmt while (expr) ;` */
function parseDoWhileStmt(p: Parser): DoWhileStmt {
  const start = p.next(); // 'do'
  const body = parseStatement(p);
  p.expectKeyword('while');
  p.expectOp('(');
  const cond = parseExpression(p);
  p.expectOp(')');
  p.expectOp(';');
  return { kind: 'do-while', body, cond, loc: locOf(start) };
}

/** `switch (expr) { case ...: ... default: ... }` — ES 3.00 only. */
function parseSwitchStmt(p: Parser): SwitchStmt {
  const start = p.next(); // 'switch'
  p.expectOp('(');
  const expr = parseExpression(p);
  p.expectOp(')');
  let body: Stmt;
  if (p.atOp('{')) {
    body = parseCompound(p);
  } else {
    p.error(p.peek().line, 'switch body must be a compound statement');
    body = parseStatement(p);
  }
  return { kind: 'switch', expr, body, loc: locOf(start) };
}

/** `case <const-expr> :` / `default :` (value null = default). */
function parseCaseLabelStmt(p: Parser): CaseLabelStmt {
  const start = p.next(); // 'case' | 'default' (keyword)
  let value: Expr | null = null;
  if (start.kind === 'keyword' && start.name === 'case') {
    value = parseAssignmentExpr(p);
  }
  p.expectOp(':', "expected ':' after case label");
  return { kind: 'case', value, loc: locOf(start) };
}

/** `return [expr] ;` */
function parseReturnStmt(p: Parser): ReturnStmt {
  const start = p.next(); // 'return'
  let value: Expr | null = null;
  if (!p.atOp(';')) value = parseExpression(p);
  p.expectOp(';', "expected ';' after return");
  return { kind: 'return', value, loc: locOf(start) };
}

/** Local declaration statement (possibly `struct S {...};` with no declarators). */
function parseDeclStmt(p: Parser): DeclStmt {
  const start = p.peek();
  const type = parseTypeSpec(p, { param: false, member: false });
  if (type.base.kind === 'struct-definition' && p.atOp(';')) {
    p.next();
    return { kind: 'decl-stmt', type, declarators: [], loc: locOf(start) };
  }
  const declarators = parseDeclarators(p, false);
  p.expectOp(';', "expected ';' after declaration");
  return { kind: 'decl-stmt', type, declarators, loc: locOf(start) };
}

/** Expression statement: `expr ;` */
function parseExprStmt(p: Parser): ExprStmt {
  const start = p.peek();
  const expr = parseExpression(p);
  p.expectOp(';', "expected ';' after expression");
  return { kind: 'expr-stmt', expr, loc: locOf(start) };
}

/**
 * Lookahead: does the current statement start a declaration? Type/qualifier
 * keywords do, except type keywords followed by `(` or `[` (constructor
 * calls: `vec4(1.0)`, `float[3](...)`). `identifier identifier` (struct
 * type + name) is also a declaration.
 */
export function startsDeclaration(p: Parser): boolean {
  const t = p.peek();
  if (t.kind === 'keyword') {
    if (!DECL_KEYWORDS.has(t.name)) return false;
    const t2 = p.peek(1);
    if (TYPE_KEYWORDS.has(t.name) && t2.kind === 'op' && (t2.text === '(' || t2.text === '[')) {
      return false;
    }
    return true;
  }
  if (t.kind === 'identifier') {
    const t2 = p.peek(1);
    return t2.kind === 'identifier' || (t2.kind === 'keyword' && DECL_KEYWORDS.has(t2.name));
  }
  return false;
}

/** Skip to the next `;` (consumed) or `}` (left in place), then EOF. */
function recoverStatement(p: Parser): void {
  while (!p.atEof() && !p.atOp(';') && !p.atOp('}')) p.next();
  if (p.atOp(';')) p.next();
}
