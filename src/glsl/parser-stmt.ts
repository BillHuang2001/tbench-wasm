/**
 * parser-stmt.ts — statement parsing for GLSL ES 1.00 / 3.00.
 *
 * Statements: compound, declaration (local), expression (incl. empty `;`),
 * if/else, for (declaration or expression init), while, do-while, switch
 * (ES 3.00 — case/default labels inside the compound body), break/continue/
 * return/discard, and precision statements (legal declaration statements per
 * the ESSL Appendix A grammar — ANGLE accepts them inside function bodies;
 * the parsed PrecisionDecl is hoisted to global scope before the enclosing
 * function definition so it takes effect, see Parser.fnPrecisionDecls).
 * Case labels are parsed wherever they appear (semantics validates they are
 * inside a switch).
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
  DECL_KEYWORDS, RESERVED_100, TYPE_KEYWORDS, locOf, EOF,
} from './parser-expr.js';
import {
  parseTypeSpec, parseDeclarators, parsePrecisionDecl, nameAnonymousStruct, skipBalanced,
  parseArrayDims,
} from './parser.js';

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
      case 'precision': {
        // Precision statements are legal declaration statements in ESSL
        // (Appendix A `declaration`: ... | PRECISION precision_qualifier
        // type_specifier_no_prec SEMICOLON) — ANGLE accepts them inside
        // function bodies (conformance/glsl/misc/
        // ternary-operators-in-initializers.html). The statement itself is
        // parsed as a PrecisionDecl that is HOISTED to just before the
        // enclosing function definition (see Parser.fnPrecisionDecls); the
        // statement slot is an `empty` node (semantics + codegen no-op).
        const decl = parsePrecisionDecl(p);
        if (p.currentFn !== null) p.fnPrecisionDecls.push({ fn: p.currentFn, decl });
        return { kind: 'empty', loc: decl.loc };
      }
      case 'case':
        if (p.version === 100) {
          p.error(t.line, "'case' is reserved in GLSL ES 1.00");
          recoverStatement(p);
          return { kind: 'empty', loc: locOf(t) };
        }
        // GLSL ES 3.10 §6.2: case labels may only appear immediately within
        // the switch body — a label nested inside a block (`case 1: { case
        // 0: ... }`) is a compile error (CTS switch-case.html).
        if (p.inSwitchBody && p.switchBlockNesting > 0) {
          p.error(t.line, "'case' : case label nested inside a block within a switch statement");
        }
        return parseCaseLabelStmt(p);
      case 'default':
        if (p.version === 100) {
          p.error(t.line, "'default' is reserved in GLSL ES 1.00");
          recoverStatement(p);
          return { kind: 'empty', loc: locOf(t) };
        }
        if (p.inSwitchBody && p.switchBlockNesting > 0) {
          p.error(t.line, "'default' : default label nested inside a block within a switch statement");
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

/**
 * `{ stmt* }`. `isSwitchBody` marks the compound that IS a switch body:
 * its direct statements may carry case labels; any nested compound inside
 * a switch body increments `switchBlockNesting` so nested labels error
 * (parseStatement 'case'/'default').
 */
export function parseCompound(p: Parser, isSwitchBody = false): CompoundStmt {
  const open = p.next(); // '{'
  const savedIn = p.inSwitchBody;
  const savedNesting = p.switchBlockNesting;
  if (isSwitchBody) {
    p.inSwitchBody = true;
    p.switchBlockNesting = 0;
  } else if (p.inSwitchBody) {
    p.switchBlockNesting++;
  }
  const body: Stmt[] = [];
  while (!p.atOp('}') && !p.atEof()) {
    body.push(parseStatement(p));
  }
  if (p.atEof()) {
    p.error(p.peek().line, "unterminated block, expected '}'");
  } else {
    p.next(); // '}'
  }
  p.inSwitchBody = savedIn;
  p.switchBlockNesting = savedNesting;
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
    // The body compound's DIRECT statements may carry case labels; nested
    // blocks inside it may not (parseCompound tracks the nesting).
    body = parseCompound(p, true);
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

/** Local declaration statement (possibly `struct S {...};` with no declarators).
 *  Returns `{kind:'empty'}` for the standalone-layout-decl form (§4.4
 *  `type_qualifier SEMICOLON`), which declares nothing. */
function parseDeclStmt(p: Parser): Stmt {
  const start = p.peek();
  const type = parseTypeSpec(p, { param: false, member: false });
  if (type.base.kind === 'struct-definition') {
    if (type.base.name === null) {
      // Anonymous struct: legal only WITH a declarator (`struct { ... } s;` —
      // GLSL ES grammar; ANGLE accepts). A bare `struct { ... };` is rejected.
      if (p.atOp(';')) {
        p.error(start.line, 'anonymous structs are not allowed in GLSL ES');
      } else {
        type.base.name = nameAnonymousStruct(p);
      }
    }
    if (p.atOp(';')) {
      p.next();
      return { kind: 'decl-stmt', type, declarators: [], loc: locOf(start) };
    }
  }
  // Standalone layout declaration inside a function body (`layout(std140)
  // uniform;` — the `type_qualifier SEMICOLON` grammar form is a legal
  // declaration_statement). The marker type spec (empty type-name) can only
  // come from parseTypeSpec's §4.4 path; accept and drop the statement —
  // layout defaults only matter at global scope (semantics processes them in
  // the global pre-pass) and ANGLE treats in-body instances as no-ops.
  if (type.base.kind === 'type-name' && type.base.name === '' && p.atOp(';')) {
    p.next();
    return { kind: 'empty', loc: locOf(start) };
  }
  // GLSL ES 3.00: array dims may precede the declarator name (`vec4[2] V;` —
  // ANGLE/CTS accept; tricky-loop-conditions.html). Invalid in 1.00 —
  // report and skip. Pre-name dims are TYPE-level: they prefix EVERY
  // declarator's own dims (`vec4[2] a, b;` declares two vec4[2]).
  let preNameDims: Expr[] = [];
  if (p.atOp('[')) {
    if (p.version === 100) {
      p.error(p.peek().line, 'array dimensions before the name require GLSL ES 3.00');
      skipBalanced(p, '[', ']');
    } else {
      preNameDims = parseArrayDims(p, false);
    }
  }
  const declarators = parseDeclarators(p, false);
  if (preNameDims.length > 0) {
    for (const d of declarators) d.arrayDims = preNameDims.concat(d.arrayDims);
  }
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
 * type + name) is also a declaration. A `type [` is a declaration UNLESS
 * the brackets belong to an array-constructor call (`float[3](...)` — the
 * balanced group is followed by `(`): `vec4[2] V;` is an array-typed
 * declaration (ES 3.00 dims-before-name, tricky-loop-conditions.html).
 */
export function startsDeclaration(p: Parser): boolean {
  const t = p.peek();
  if (t.kind === 'keyword') {
    if (!DECL_KEYWORDS.has(t.name)) return false;
    const t2 = p.peek(1);
    if (TYPE_KEYWORDS.has(t.name)) {
      if (t2.kind === 'op' && t2.text === '(') return false;
      if (t2.kind === 'op' && t2.text === '[') return !isArrayCtorCall(p);
    }
    return true;
  }
  if (t.kind === 'identifier') {
    const t2 = p.peek(1);
    return t2.kind === 'identifier' || (t2.kind === 'keyword' && DECL_KEYWORDS.has(t2.name));
  }
  return false;
}

/** At a `[` directly after a type keyword: is the balanced bracket group
 *  followed by `(` (an array-constructor call → expression)? Any other
 *  following token means the dims are a declaration type (`vec4[2] V;`). */
function isArrayCtorCall(p: Parser): boolean {
  let depth = 0;
  for (let i = 0; ; i++) {
    const t = p.peek(i);
    if (t.kind === 'op' && t.text === '[') {
      depth++;
    } else if (t.kind === 'op' && t.text === ']') {
      depth--;
      if (depth === 0) {
        const n = p.peek(i + 1);
        return n.kind === 'op' && n.text === '(';
      }
    } else if (t.kind === 'op' && t.text === EOF) {
      return false;
    }
  }
}

/** Skip to the next `;` (consumed) or `}` (left in place), then EOF. */
function recoverStatement(p: Parser): void {
  while (!p.atEof() && !p.atOp(';') && !p.atOp('}')) p.next();
  if (p.atOp(';')) p.next();
}
