/**
 * parser.ts — the GLSL ES 1.00 / 3.00 parser: external declarations, type
 * specs (qualifiers + base types), declarators, function prototypes and
 * definitions, and ES 3.00 interface blocks. Statements live in
 * parser-stmt.ts, expressions in parser-expr.ts (both import the Parser
 * class and helpers from here / from parser-expr.ts).
 *
 * Pipeline position: `compileShader` runs preprocessor → lexer → parser →
 * semantics. This module consumes the lexer's `Token[]` stream and produces
 * the `TranslationUnit` AST defined in ast.ts (semantics annotates it in
 * place). `#extension` directives arrive as `extensionDirectives` (from the
 * preprocessor) and become ExtensionDecl nodes PREPENDED to the declaration
 * list, in order.
 *
 * Grammar coverage (GLSL ES 1.00 Appendix A + ES 3.00 additions):
 * - external declarations: precision decls (base restricted to
 *   float/int/sampler kinds), invariant decls (`invariant gl_Position;` and
 *   the `invariant <qualifiers> <type> <name>;` qualifier form), struct
 *   definitions (bare `struct S {...};` or with declarators), global
 *   variable declarations, function prototypes/definitions, ES 3.00
 *   interface blocks (`uniform [layout(binding=N)] B { ... } [inst][dims];`).
 * - type specs: storage (attribute/varying/const/uniform for 1.00;
 *   in/out/uniform/const for 3.00), precision, interpolation + centroid
 *   (3.00), invariant, precise (3.00; accepted, no effect), layout(...)
 *   (3.00; `location`/`binding` captured, unknown ids such as std140 and
 *   column_major accepted and ignored — the CTS uses them).
 * - declarators: name, array dims (constant-size exprs; `[]` unsized is an
 *   error except for function parameters), `= initializer`.
 *
 * Version rules enforced here (the lexer already classified keywords):
 * - 1.00 reserved words (`switch`, `case`, `default`, `flat`, `sampler3D`,
 *   `sampler2DShadow`, `in`, `out`, `inout`) are rejected with a
 *   "'X' is reserved in GLSL ES 1.00" error.
 * - 3.00-only syntax in a 1.00 shader (interface blocks, `in`/`out`
 *   storage, `inout` params) errors; `attribute`/`varying` are plain
 *   identifiers in 3.00 and a declaration like `attribute vec4 x;` fails
 *   naturally ("expected identifier, found 'vec4'").
 * - Anonymous structs and nested struct definitions are rejected.
 *
 * Errors: Khronos-style 1-based lines; up to 20 errors are collected, with
 * recovery (skip to `;`/`}`) so one bad construct doesn't hide the rest.
 */
import type { Token } from './lexer.js';
import type { ExtensionDirective } from './preprocessor.js';
import type { CompileError } from './compiler.js';
import type { Precision, SamplerKind, StorageClass, TypeQualifiers, LayoutQualifiers } from './types.js';
import type {
  ExternalDecl, Expr, FunctionPrototype, GlobalVarDecl, InterfaceBlockDecl,
  InvariantDecl, ParamDecl, PrecisionDecl, StructDecl, StructDefinition,
  StructMemberDecl, TranslationUnit, TypeName, TypeSpec, VarDeclarator,
} from './ast.js';
import {
  parseExpression, parseAssignmentExpr,
  TYPE_KEYWORDS, SAMPLER_KEYWORDS, RESERVED_100, DECL_KEYWORDS,
  describeToken, locOf, EOF,
} from './parser-expr.js';
import { parseCompound } from './parser-stmt.js';

/** Maximum number of errors collected before error storage is capped. */
const MAX_ERRORS = 20;

/* ------------------------------------------------------------------ */
/* Parser state (shared with parser-expr.ts / parser-stmt.ts)          */
/* ------------------------------------------------------------------ */

/**
 * Token-stream cursor + error collector. parser-expr.ts and parser-stmt.ts
 * operate on this through its public methods (they import the type only).
 */
export class Parser {
  readonly version: 100 | 300;
  readonly errors: CompileError[] = [];
  private readonly tokens: Token[];
  private pos = 0;
  private readonly eofLine: number;

  constructor(tokens: Token[], version: 100 | 300) {
    this.tokens = tokens;
    this.version = version;
    const last = tokens[tokens.length - 1];
    this.eofLine = last !== undefined ? last.line : 1;
  }

  /** Current token (a synthetic EOF token past the end; line = last line). */
  peek(offset = 0): Token {
    const t = this.tokens[this.pos + offset];
    return t ?? { kind: 'op', text: EOF, line: this.eofLine, column: 0 };
  }

  /** Consume and return the current token (EOF is idempotent). */
  next(): Token {
    const t = this.tokens[this.pos];
    if (t !== undefined) {
      this.pos++;
      return t;
    }
    return this.peek();
  }

  atEof(): boolean {
    return this.pos >= this.tokens.length;
  }

  atOp(text: string): boolean {
    const t = this.peek();
    return t.kind === 'op' && t.text === text;
  }

  atKeyword(name: string): boolean {
    const t = this.peek();
    return t.kind === 'keyword' && t.name === name;
  }

  atIdentifier(): boolean {
    return this.peek().kind === 'identifier';
  }

  /** Consume an operator token, or record `expected '<text>'` (custom msg wins). */
  expectOp(text: string, msg?: string): Token | null {
    if (this.atOp(text)) return this.next();
    this.error(this.peek().line, msg ?? `expected '${text}'`);
    return null;
  }

  /** Consume a keyword token, or record an error. */
  expectKeyword(name: string): Token | null {
    if (this.atKeyword(name)) return this.next();
    this.error(this.peek().line, `expected '${name}'`);
    return null;
  }

  /** Consume an identifier token, or record an error and return null. */
  expectIdentifier(): Extract<Token, { kind: 'identifier' }> | null {
    const t = this.peek();
    if (t.kind === 'identifier') return this.next() as Extract<Token, { kind: 'identifier' }>;
    this.error(t.line, `expected identifier, found ${describeToken(t)}`);
    return null;
  }

  /** Record a compile error (1-based line); the store is capped. */
  error(line: number, message: string): void {
    if (this.errors.length < MAX_ERRORS) this.errors.push({ line, message });
  }
}

/* ------------------------------------------------------------------ */
/* Entry point                                                         */
/* ------------------------------------------------------------------ */

export type ParseResult = { ok: true; ast: TranslationUnit } | { ok: false; errors: CompileError[] };

/**
 * Parse a token stream into a TranslationUnit. `opts.version` is the shader
 * language version that gates the grammar (the declared `#version` as
 * reported by the preprocessor); extension directives are converted to
 * ExtensionDecl nodes and prepended in order.
 */
export function parse(
  tokens: Token[],
  opts: { version: 100 | 300; extensionDirectives: ExtensionDirective[] },
): ParseResult {
  const p = new Parser(tokens, opts.version);
  const declarations: ExternalDecl[] = [];
  for (const d of opts.extensionDirectives) {
    declarations.push({
      kind: 'extension-decl',
      name: d.name,
      behavior: d.behavior,
      loc: { line: d.line, column: 0 },
    });
  }
  while (!p.atEof()) {
    // Stray top-level `;` is tolerated (recovery-friendly; harmless).
    if (p.atOp(';')) {
      p.next();
      continue;
    }
    const decl = parseExternalDecl(p);
    if (decl !== null) declarations.push(decl);
  }
  if (p.errors.length > 0) return { ok: false, errors: p.errors };
  const ast: TranslationUnit = {
    kind: 'translation-unit',
    version: opts.version,
    declarations,
    loc: { line: 1, column: 0 },
  };
  return { ok: true, ast };
}

/* ------------------------------------------------------------------ */
/* External declarations                                               */
/* ------------------------------------------------------------------ */

function parseExternalDecl(p: Parser): ExternalDecl | null {
  const t = p.peek();
  if (t.kind === 'keyword') {
    switch (t.name) {
      case 'precision':
        return parsePrecisionDecl(p);
      case 'invariant':
        return parseInvariantDecl(p);
      case 'struct':
        return parseStructDecl(p);
      default:
        break;
    }
  }
  return parseDeclarationOrFunction(p);
}

/** `precision <highp|mediump|lowp> <float|int|sampler-kind>;` */
function parsePrecisionDecl(p: Parser): PrecisionDecl {
  const start = p.next(); // 'precision'
  let precision: Precision = 'highp';
  const pt = p.peek();
  if (pt.kind === 'keyword' && (pt.name === 'highp' || pt.name === 'mediump' || pt.name === 'lowp')) {
    precision = pt.name;
    p.next();
  } else {
    p.error(pt.line, `expected precision qualifier (highp, mediump or lowp), found ${describeToken(pt)}`);
  }
  let base: PrecisionDecl['base'] = 'float';
  const bt = p.peek();
  if (bt.kind === 'keyword' && (bt.name === 'float' || bt.name === 'int')) {
    base = bt.name;
    p.next();
  } else if (
    bt.kind === 'keyword' &&
    SAMPLER_KEYWORDS.has(bt.name) &&
    (p.version === 300 || bt.name === 'sampler2D' || bt.name === 'samplerCube')
  ) {
    base = bt.name as SamplerKind;
    p.next();
  } else {
    p.error(bt.line, `${describeToken(bt)} : invalid base type in precision declaration (expected float, int or a sampler type)`);
    // Consume a stray keyword/identifier so the `;` can be found.
    if (bt.kind === 'keyword' || bt.kind === 'identifier') p.next();
  }
  p.expectOp(';', "expected ';' after precision declaration");
  return { kind: 'precision-decl', precision, base, loc: locOf(start) };
}

/**
 * `invariant gl_Position;` / `invariant <identifier>;` or the qualifier form
 * `invariant <type-spec> <declarators>;` (e.g. `invariant varying vec4 v;`).
 */
function parseInvariantDecl(p: Parser): ExternalDecl {
  const start = p.next(); // 'invariant'
  const t = p.peek();
  // Short form: `invariant <identifier>;` (gl_Position or a varying name).
  // Qualifier form: `invariant <keyword-type/qualifier> ...` (e.g.
  // `invariant varying vec4 v;`). Anything else is reported and recovered.
  const declStart = t.kind === 'keyword' && t.name !== 'invariant' && DECL_KEYWORDS.has(t.name);
  if (!declStart) {
    const nameT = p.expectIdentifier();
    p.expectOp(';');
    return { kind: 'invariant-decl', name: nameT ? nameT.name : '', loc: locOf(start) };
  }
  const type = parseTypeSpec(p, { param: false, member: false });
  // The leading 'invariant' was consumed above — re-attach it to the type
  // spec (the qualifier form `invariant varying vec4 v;` must record it).
  type.qualifiers.invariant = true;
  // `invariant <type> <name>(...)` — invariant on a function is invalid;
  // parse it anyway (semantics rejects) so recovery stays clean.
  const nameT = p.peek();
  const t3 = p.peek(1);
  if (nameT.kind === 'identifier' && t3.kind === 'op' && t3.text === '(') {
    p.next();
    p.next();
    const params = parseParams(p);
    p.expectOp(')');
    const prototype: FunctionPrototype = {
      kind: 'function-prototype',
      name: nameT.name,
      returnType: type,
      params,
      loc: locOf(start),
    };
    if (p.atOp('{')) {
      const body = parseCompound(p);
      return { kind: 'function-definition', prototype, body, loc: locOf(start) };
    }
    p.expectOp(';');
    return prototype;
  }
  const declarators = parseDeclarators(p, false);
  p.expectOp(';', "expected ';' after declaration");
  const decl: GlobalVarDecl = { kind: 'global-var-decl', type, declarators, loc: locOf(start) };
  return decl;
}

/**
 * `struct S { ... };` (bare type declaration) or `struct S { ... } v;`
 * (declaration with declarators).
 */
function parseStructDecl(p: Parser): ExternalDecl {
  const start = p.peek();
  const def = parseStructDefinition(p);
  if (p.atOp(';')) {
    p.next();
    const decl: StructDecl = { kind: 'struct-decl', name: def.name ?? '', members: def.members, loc: locOf(start) };
    return decl;
  }
  const type: TypeSpec = { kind: 'type-spec', qualifiers: {}, base: def, loc: locOf(start) };
  const declarators = parseDeclarators(p, false);
  p.expectOp(';', "expected ';' after declaration");
  const decl: GlobalVarDecl = { kind: 'global-var-decl', type, declarators, loc: locOf(start) };
  return decl;
}

/**
 * Global variable declaration or function prototype/definition. The type
 * spec is parsed first; a following `{` means an ES 3.00 interface block, a
 * `(` after the name means a function.
 */
function parseDeclarationOrFunction(p: Parser): ExternalDecl | null {
  const start = p.peek();
  const type = parseTypeSpec(p, { param: false, member: false });
  if (type.base.kind === 'struct-definition') {
    // `uniform struct S {...} u;` etc.
    const declarators = parseDeclarators(p, false);
    p.expectOp(';', "expected ';' after declaration");
    const decl: GlobalVarDecl = { kind: 'global-var-decl', type, declarators, loc: locOf(start) };
    return decl;
  }
  if (p.atOp('{')) {
    return parseInterfaceBlock(p, type);
  }
  const nameT = p.peek();
  if (nameT.kind !== 'identifier') {
    p.error(nameT.line, `expected identifier, found ${describeToken(nameT)}`);
    recoverTopLevel(p);
    return null;
  }
  const t2 = p.peek(1);
  if (t2.kind === 'op' && t2.text === '(') {
    p.next(); // name
    p.next(); // '('
    const params = parseParams(p);
    p.expectOp(')');
    const prototype: FunctionPrototype = {
      kind: 'function-prototype',
      name: nameT.name,
      returnType: type,
      params,
      loc: locOf(start),
    };
    if (p.atOp('{')) {
      const body = parseCompound(p);
      return { kind: 'function-definition', prototype, body, loc: locOf(start) };
    }
    p.expectOp(';', "expected ';' after function prototype");
    return prototype;
  }
  const declarators = parseDeclarators(p, false);
  p.expectOp(';', "expected ';' after declaration");
  const decl: GlobalVarDecl = { kind: 'global-var-decl', type, declarators, loc: locOf(start) };
  return decl;
}

/* ------------------------------------------------------------------ */
/* Type specs & qualifiers                                             */
/* ------------------------------------------------------------------ */

interface TypeSpecCtx {
  /** Function-parameter context: in/out/inout/const + precision only. */
  param: boolean;
  /** Struct/interface-block member context: precision only. */
  member: boolean;
}

const STORAGE_MAP: Readonly<Record<string, StorageClass>> = {
  const: 'const',
  uniform: 'uniform',
  attribute: 'attribute',
  varying: 'varying',
  in: 'in',
  out: 'out',
  inout: 'inout',
};

/** Parse qualifiers + base type into a TypeSpec. */
export function parseTypeSpec(p: Parser, ctx: TypeSpecCtx): TypeSpec {
  const start = p.peek();
  const qualifiers: TypeQualifiers = {};
  for (;;) {
    const t = p.peek();
    if (t.kind !== 'keyword') break;
    const name = t.name;
    switch (name) {
      case 'const':
      case 'uniform':
      case 'attribute':
      case 'varying':
      case 'in':
      case 'out':
      case 'inout': {
        if (ctx.member) {
          p.error(t.line, `'${name}' : storage qualifiers are not allowed on struct or block members`);
          p.next();
          continue;
        }
        if (p.version === 100 && (name === 'in' || name === 'out' || name === 'inout')) {
          p.error(t.line, `'${name}' is reserved in GLSL ES 1.00`);
          p.next();
          continue;
        }
        if (name === 'inout' && !ctx.param) {
          p.error(t.line, "'inout' is only allowed on function parameters");
          p.next();
          continue;
        }
        if (name === 'uniform' && ctx.param) {
          p.error(t.line, "'uniform' : invalid qualifier on function parameter");
          p.next();
          continue;
        }
        if (ctx.param && (name === 'attribute' || name === 'varying')) {
          p.error(t.line, `'${name}' : invalid qualifier on function parameter`);
          p.next();
          continue;
        }
        if (name === 'inout') {
          // 'inout' is a real storage class in types.ts (ES 3.00 params only —
          // enforced above); it flows through setStorage like 'in'/'out'.
          setStorage(p, t, qualifiers, 'inout');
          p.next();
          continue;
        }
        setStorage(p, t, qualifiers, STORAGE_MAP[name]);
        p.next();
        continue;
      }
      case 'highp':
      case 'mediump':
      case 'lowp': {
        if (qualifiers.precision !== undefined) {
          p.error(t.line, `duplicate precision qualifier '${name}'`);
        }
        qualifiers.precision = name;
        p.next();
        continue;
      }
      case 'flat':
      case 'smooth':
      case 'noperspective': {
        if (p.version === 100) {
          p.error(t.line, `'${name}' is reserved in GLSL ES 1.00`);
          p.next();
          continue;
        }
        if (ctx.member || ctx.param) {
          p.error(t.line, `'${name}' : interpolation qualifiers are not allowed on ${ctx.member ? 'struct or block members' : 'function parameters'}`);
          p.next();
          continue;
        }
        if (qualifiers.interpolation !== undefined) {
          p.error(t.line, `duplicate interpolation qualifier '${name}'`);
        }
        qualifiers.interpolation = name;
        p.next();
        continue;
      }
      case 'centroid': {
        if (ctx.member || ctx.param) {
          p.error(t.line, `'centroid' : invalid qualifier on ${ctx.member ? 'struct or block member' : 'function parameter'}`);
          p.next();
          continue;
        }
        qualifiers.centroid = true;
        p.next();
        continue;
      }
      case 'invariant': {
        if (ctx.member || ctx.param) {
          p.error(t.line, `'invariant' : invalid qualifier on ${ctx.member ? 'struct or block member' : 'function parameter'}`);
          p.next();
          continue;
        }
        qualifiers.invariant = true;
        p.next();
        continue;
      }
      case 'precise': {
        if (ctx.member || ctx.param) {
          p.error(t.line, `'precise' : invalid qualifier on ${ctx.member ? 'struct or block member' : 'function parameter'}`);
          p.next();
          continue;
        }
        qualifiers.precise = true;
        p.next();
        continue;
      }
      case 'layout': {
        // `layout` is an identifier in 1.00 — unreachable there.
        if (ctx.member || ctx.param) {
          p.error(t.line, `'layout' : layout qualifiers are not allowed on ${ctx.member ? 'struct or block members' : 'function parameters'}`);
          p.next();
          continue;
        }
        if (qualifiers.layout !== undefined) {
          p.error(t.line, "duplicate 'layout' qualifier");
        }
        qualifiers.layout = parseLayoutQualifiers(p);
        continue;
      }
      default:
        break;
    }
    break;
  }
  // Base type: type name or inline struct definition.
  let base: TypeName | StructDefinition;
  const bt = p.peek();
  if (bt.kind === 'keyword' && bt.name === 'struct') {
    if (ctx.member) p.error(bt.line, 'struct definitions are not allowed inside structs');
    else if (ctx.param) p.error(bt.line, 'struct definitions are not allowed in function parameters');
    base = parseStructDefinition(p);
  } else {
    base = parseTypeName(p);
  }
  return { kind: 'type-spec', qualifiers, base, loc: locOf(start) };
}

/** Set a storage qualifier with duplicate/conflict checks (`const` combines with in/out). */
function setStorage(p: Parser, t: Token, q: TypeQualifiers, storage: StorageClass): void {
  const prev = q.storage;
  if (prev === undefined) {
    q.storage = storage;
    return;
  }
  if (prev === storage) {
    p.error(t.line, `duplicate storage qualifier '${storage}'`);
    return;
  }
  if (prev !== 'const' && storage !== 'const') {
    p.error(t.line, `conflicting storage qualifiers '${prev}' and '${storage}'`);
  }
  q.storage = storage;
}

/**
 * `layout ( id [= const-expr] [, ...] )` — `location`/`binding` are captured
 * (literal values only; semantics resolves non-literal constants); unknown
 * ids (`std140`, `column_major`, ...) are accepted and ignored (the CTS uses
 * them in valid WebGL2 shaders).
 */
function parseLayoutQualifiers(p: Parser): LayoutQualifiers {
  const layout: LayoutQualifiers = {};
  p.next(); // 'layout'
  if (!p.expectOp('(')) return layout;
  for (;;) {
    const idT = p.peek();
    if (idT.kind === 'identifier') {
      p.next();
      if (p.atOp('=')) {
        p.next();
        const v = parseAssignmentExpr(p);
        if (idT.name === 'location') layout.location = intValueOf(v);
        else if (idT.name === 'binding') layout.binding = intValueOf(v);
      }
    } else {
      p.error(idT.line, `expected layout qualifier identifier, found ${describeToken(idT)}`);
    }
    if (p.atOp(',')) {
      p.next();
      continue;
    }
    break;
  }
  p.expectOp(')');
  return layout;
}

/** Literal integer value of an expression, or undefined (semantics resolves the rest). */
function intValueOf(e: Expr): number | undefined {
  if (e.kind === 'literal' && (e.literalType === 'int' || e.literalType === 'uint')) {
    return e.value as number;
  }
  return undefined;
}

/** Base type name: identifier (struct type) or a type keyword. */
function parseTypeName(p: Parser): TypeName {
  const t = p.peek();
  if (t.kind === 'identifier') {
    p.next();
    return { kind: 'type-name', name: t.name, loc: locOf(t) };
  }
  if (t.kind === 'keyword') {
    if (p.version === 100 && RESERVED_100.has(t.name)) {
      p.error(t.line, `'${t.name}' is reserved in GLSL ES 1.00`);
      p.next();
      return { kind: 'type-name', name: '', loc: locOf(t) };
    }
    if (TYPE_KEYWORDS.has(t.name)) {
      p.next();
      return { kind: 'type-name', name: t.name, loc: locOf(t) };
    }
    p.error(t.line, `syntax error, unexpected '${t.name}'`);
    p.next();
    return { kind: 'type-name', name: '', loc: locOf(t) };
  }
  p.error(t.line, `syntax error, unexpected ${describeToken(t)}`);
  p.next();
  return { kind: 'type-name', name: '', loc: locOf(t) };
}

/* ------------------------------------------------------------------ */
/* Structs                                                             */
/* ------------------------------------------------------------------ */

/** `struct [Name] { members }` — anonymous structs are rejected. */
function parseStructDefinition(p: Parser): StructDefinition {
  const start = p.next(); // 'struct'
  let name: string | null = null;
  const t = p.peek();
  if (t.kind === 'identifier') {
    p.next();
    name = t.name;
  } else if (t.kind === 'op' && t.text === '{') {
    p.error(t.line, 'anonymous structs are not allowed in GLSL ES');
  } else {
    p.error(t.line, `expected struct name, found ${describeToken(t)}`);
  }
  p.expectOp('{');
  const members = parseStructMembers(p);
  p.expectOp('}');
  return { kind: 'struct-definition', name, members, loc: locOf(start) };
}

/** Struct / interface-block member list: `type name [dims] ;` */
function parseStructMembers(p: Parser): StructMemberDecl[] {
  const members: StructMemberDecl[] = [];
  while (!p.atOp('}') && !p.atEof()) {
    const start = p.peek();
    const type = parseTypeSpec(p, { param: false, member: true });
    const nameT = p.expectIdentifier();
    const arrayDims = parseArrayDims(p, false);
    p.expectOp(';', "expected ';' after struct member");
    members.push({ kind: 'struct-member', name: nameT ? nameT.name : '', type, arrayDims, loc: locOf(start) });
  }
  return members;
}

/* ------------------------------------------------------------------ */
/* Declarators                                                         */
/* ------------------------------------------------------------------ */

/**
 * `name [dims] [= init] [, ...]`. `allowUnsized` permits `[]` (function
 * parameters only); elsewhere an unsized array declaration is an error.
 */
export function parseDeclarators(p: Parser, allowUnsized: boolean): VarDeclarator[] {
  const out: VarDeclarator[] = [];
  for (;;) {
    const nameT = p.expectIdentifier();
    const arrayDims = parseArrayDims(p, allowUnsized);
    let init: Expr | null = null;
    if (p.atOp('=')) {
      p.next();
      init = parseAssignmentExpr(p);
    }
    out.push({
      kind: 'var-declarator',
      name: nameT ? nameT.name : '',
      arrayDims,
      init,
      loc: locOf(nameT ?? p.peek()),
    });
    if (!p.atOp(',')) break;
    p.next();
  }
  return out;
}

/**
 * Array dimension brackets. An unsized `[]` pushes `null` (per the AST
 * contract: `[null]` = unsized) and errors unless `allowUnsized`.
 */
function parseArrayDims(p: Parser, allowUnsized: boolean): Expr[] {
  const dims: (Expr | null)[] = [];
  while (p.atOp('[')) {
    const open = p.next();
    if (p.atOp(']')) {
      p.next();
      if (!allowUnsized) {
        p.error(open.line, 'unsized array declarations are only allowed for function parameters');
      }
      dims.push(null);
    } else {
      const e = parseExpression(p);
      dims.push(e);
      p.expectOp(']');
    }
  }
  return dims as Expr[];
}

/* ------------------------------------------------------------------ */
/* Function parameters                                                 */
/* ------------------------------------------------------------------ */

/** Parameter list; `f(void)` and `f()` both yield an empty list. */
function parseParams(p: Parser): ParamDecl[] {
  const params: ParamDecl[] = [];
  if (p.atOp(')')) return params;
  const v2 = p.peek(1);
  if (p.atKeyword('void') && v2.kind === 'op' && v2.text === ')') {
    p.next(); // 'void'
    return params;
  }
  for (;;) {
    params.push(parseParamDecl(p));
    if (p.atOp(',')) {
      p.next();
      continue;
    }
    break;
  }
  return params;
}

/**
 * One parameter: qualifiers (3.00: in/out/inout/const; 1.00: const only —
 * in/out/inout are reserved-word errors), precision, type, optional name,
 * optional array dims (unsized `[]` allowed in both versions — the ES 1.00
 * grammar permits it and the spec requires names only loosely).
 */
function parseParamDecl(p: Parser): ParamDecl {
  const start = p.peek();
  const type = parseTypeSpec(p, { param: true, member: false });
  let name = '';
  const nt = p.peek();
  if (nt.kind === 'identifier') {
    p.next();
    name = nt.name;
  }
  const arrayDims = parseArrayDims(p, true);
  return { kind: 'param-decl', name, type, arrayDims, loc: locOf(start) };
}

/* ------------------------------------------------------------------ */
/* Interface blocks (ES 3.00)                                          */
/* ------------------------------------------------------------------ */

/**
 * `uniform [layout(binding=N)] BlockName { members } [instance][dims];`
 * Invoked with the type spec already parsed and the cursor at `{`.
 */
function parseInterfaceBlock(p: Parser, type: TypeSpec): InterfaceBlockDecl {
  const blockName = type.base.kind === 'type-name' ? type.base.name : '';
  if (p.version === 100) {
    p.error(p.peek().line, 'interface blocks require GLSL ES 3.00');
    skipBalanced(p, '{', '}');
    // Skip a trailing instance name / array dims up to the `;` so the next
    // declaration parses cleanly (no cascading 'expected identifier' error).
    recoverTopLevel(p);
    return {
      kind: 'interface-block',
      qualifiers: type.qualifiers,
      blockName,
      instanceName: null,
      members: [],
      arrayDims: [],
      loc: type.loc,
    };
  }
  p.expectOp('{');
  const members = parseStructMembers(p);
  p.expectOp('}');
  let instanceName: string | null = null;
  const instT = p.peek();
  if (instT.kind === 'identifier') {
    p.next();
    instanceName = instT.name;
  }
  const arrayDims = parseArrayDims(p, false);
  p.expectOp(';', "expected ';' after interface block");
  return {
    kind: 'interface-block',
    qualifiers: type.qualifiers,
    blockName,
    instanceName,
    members,
    arrayDims,
    loc: type.loc,
  };
}

/* ------------------------------------------------------------------ */
/* Recovery helpers                                                    */
/* ------------------------------------------------------------------ */

/** Skip tokens up to (and consuming) the next `;`, or to EOF. */
function recoverTopLevel(p: Parser): void {
  while (!p.atEof() && !p.atOp(';')) p.next();
  if (p.atOp(';')) p.next();
}

/** Skip a balanced `open ... close` group (used to skip invalid constructs). */
export function skipBalanced(p: Parser, open: string, close: string): void {
  if (!p.atOp(open)) return;
  let depth = 0;
  while (!p.atEof()) {
    const t = p.next();
    if (t.kind === 'op' && t.text === open) depth++;
    else if (t.kind === 'op' && t.text === close) {
      depth--;
      if (depth === 0) return;
    }
  }
}
