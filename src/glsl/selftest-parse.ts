/**
 * selftest-parse.ts — sanity checks for the GLSL ES 1.00 / 3.00 parser stage.
 *
 * Run: npx tsx src/glsl/selftest-parse.ts
 *
 * Drives the full pipeline (preprocess → tokenize → parse) and asserts EXACT
 * AST shapes:
 * - ES 1.00: attribute/varying/uniform/const globals (multi-declarator +
 *   initializers), precision decls, struct decl + struct-typed variable,
 *   constant array dims, comments/whitespace tolerance, function definitions.
 * - ES 3.00: `#version 300 es` with in/out + interpolation (flat/smooth/
 *   centroid), layout(location=)/layout(binding=), interface block with
 *   instance + array dims, switch/case/default, uint literals, bitwise ops,
 *   ternary, every assignment operator, `float[3](...)` array constructor.
 * - Expression precedence: binary nesting, right-assoc assignment, unary vs
 *   binary minus, postfix chains, calls, comma (incl. ternary middle).
 * - #extension directive → ExtensionDecl nodes prepended in order.
 * - Error cases with EXACT 1-based line numbers (missing `;`, unterminated
 *   block, reserved words in 1.00, bitwise in 1.00, anonymous/nested structs,
 *   invalid precision base, bad case label, interface block in 1.00).
 * - in/out/inout function parameter qualifiers: legal in 1.00 (§6.1) and 3.00
 *   (recorded as type.qualifiers.storage); still reserved words elsewhere in
 *   1.00 (storage-class position, declarator/function names, expressions).
 *
 * Prints "parser selftest: N checks, M failure(s)" then "OK" and exits 0 on
 * success, or exits 1 when any check fails.
 */
import { preprocess } from './preprocessor.js';
import { tokenize } from './lexer.js';
import { parse } from './parser.js';
import type { CompileError } from './compiler.js';
import type {
  TranslationUnit, ExternalDecl, GlobalVarDecl, FunctionDefinition,
  FunctionPrototype, PrecisionDecl, StructDecl, InterfaceBlockDecl,
  InvariantDecl, ExtensionDecl, DeclStmt, ExprStmt, ForStmt, IfStmt,
  DoWhileStmt, SwitchStmt, CaseLabelStmt, ReturnStmt, CompoundStmt, Expr, BinaryExpr,
  AssignExpr, UnaryExpr, TernaryExpr, CallExpr, IndexExpr, MemberExpr,
  CommaExpr,
} from './ast.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/** preprocess → tokenize → parse; returns the AST or the collected errors. */
function parseSrc(
  src: string,
  version: 100 | 300,
  opts?: { extensions?: Set<string> },
): { ast: TranslationUnit | null; errors: CompileError[] } {
  const pp = preprocess(src, { version, extensions: opts?.extensions });
  if (!pp.ok) return { ast: null, errors: pp.errors };
  const lex = tokenize(pp.tokens, pp.version);
  if (!lex.ok) return { ast: null, errors: lex.errors };
  const res = parse(lex.tokens, { version: pp.version, extensionDirectives: pp.extensionDirectives });
  if (res.ok) return { ast: res.ast, errors: [] };
  return { ast: null, errors: res.errors };
}

function parseOk(src: string, version: 100 | 300, opts?: { extensions?: Set<string> }): TranslationUnit {
  const r = parseSrc(src, version, opts);
  check(
    r.ast !== null,
    `parse should succeed (v${version}): ${JSON.stringify(src.slice(0, 70))}` +
      (r.ast === null ? ` — errors: ${JSON.stringify(r.errors)}` : ''),
  );
  return r.ast as TranslationUnit;
}

function parseFail(src: string, version: 100 | 300, opts?: { extensions?: Set<string> }): CompileError[] {
  const r = parseSrc(src, version, opts);
  check(r.ast === null, `parse should FAIL (v${version}): ${JSON.stringify(src.slice(0, 70))}`);
  return r.errors;
}

/* ------------------------------------------------------------------ */
/* External-declaration type guards                                     */
/* ------------------------------------------------------------------ */

function gvar(d: ExternalDecl): GlobalVarDecl {
  check(d.kind === 'global-var-decl', `expected global-var-decl, got ${d.kind}`);
  return d as GlobalVarDecl;
}
function fdef(d: ExternalDecl): FunctionDefinition {
  check(d.kind === 'function-definition', `expected function-definition, got ${d.kind}`);
  return d as FunctionDefinition;
}
function fproto(d: ExternalDecl): FunctionPrototype {
  check(d.kind === 'function-prototype', `expected function-prototype, got ${d.kind}`);
  return d as FunctionPrototype;
}
function pdecl(d: ExternalDecl): PrecisionDecl {
  check(d.kind === 'precision-decl', `expected precision-decl, got ${d.kind}`);
  return d as PrecisionDecl;
}
function sdecl(d: ExternalDecl): StructDecl {
  check(d.kind === 'struct-decl', `expected struct-decl, got ${d.kind}`);
  return d as StructDecl;
}
function iblock(d: ExternalDecl): InterfaceBlockDecl {
  check(d.kind === 'interface-block', `expected interface-block, got ${d.kind}`);
  return d as InterfaceBlockDecl;
}
function idecl(d: ExternalDecl): InvariantDecl {
  check(d.kind === 'invariant-decl', `expected invariant-decl, got ${d.kind}`);
  return d as InvariantDecl;
}
function edecl(d: ExternalDecl): ExtensionDecl {
  check(d.kind === 'extension-decl', `expected extension-decl, got ${d.kind}`);
  return d as ExtensionDecl;
}

/* ------------------------------------------------------------------ */
/* Expression assertion helpers (null-tolerant: a failed check never   */
/* crashes the run — the helper returns null and callers short-circuit)*/
/* ------------------------------------------------------------------ */

function id(e: Expr | null, name: string): void {
  check(
    e !== null && e.kind === 'identifier' && e.name === name,
    `expected identifier '${name}', got ${e === null ? 'null' : e.kind}`,
  );
}

function lit(e: Expr | null, value: number | boolean, literalType: 'int' | 'uint' | 'float' | 'bool'): void {
  const ok = e !== null && e.kind === 'literal' && e.value === value && e.literalType === literalType;
  check(
    ok,
    `expected literal ${String(value)}:${literalType}, got ${e === null ? 'null' : e.kind === 'literal' ? `${String(e.value)}:${e.literalType}` : e.kind}`,
  );
}

function asBin(e: Expr | null, op: string): BinaryExpr | null {
  const ok = e !== null && e.kind === 'binary' && e.op === op;
  check(
    ok,
    `expected binary '${op}', got ${e === null ? 'null' : e.kind === 'binary' ? `'${e.op}'` : e.kind}`,
  );
  return ok ? (e as BinaryExpr) : null;
}

function asAssign(e: Expr | null, op: string): AssignExpr | null {
  const ok = e !== null && e.kind === 'assign' && e.op === op;
  check(ok, `expected assign '${op}', got ${e === null ? 'null' : e.kind === 'assign' ? `'${e.op}'` : e.kind}`);
  return ok ? (e as AssignExpr) : null;
}

function asUnary(e: Expr | null, op: string): UnaryExpr | null {
  const ok = e !== null && e.kind === 'unary' && e.op === op;
  check(ok, `expected unary '${op}', got ${e === null ? 'null' : e.kind === 'unary' ? `'${e.op}'` : e.kind}`);
  return ok ? (e as UnaryExpr) : null;
}

function asTernary(e: Expr | null): TernaryExpr | null {
  check(e !== null && e.kind === 'ternary', `expected ternary, got ${e === null ? 'null' : e.kind}`);
  return e !== null && e.kind === 'ternary' ? e : null;
}

function asCall(e: Expr | null): CallExpr | null {
  check(e !== null && e.kind === 'call', `expected call, got ${e === null ? 'null' : e.kind}`);
  return e !== null && e.kind === 'call' ? e : null;
}

function asIndex(e: Expr | null): IndexExpr | null {
  check(e !== null && e.kind === 'index', `expected index, got ${e === null ? 'null' : e.kind}`);
  return e !== null && e.kind === 'index' ? e : null;
}

function asMember(e: Expr | null, name: string): MemberExpr | null {
  const ok = e !== null && e.kind === 'member' && e.name === name;
  check(ok, `expected member '.${name}', got ${e === null ? 'null' : e.kind}`);
  return ok ? (e as MemberExpr) : null;
}

function asComma(e: Expr | null, n: number): CommaExpr | null {
  const ok = e !== null && e.kind === 'comma' && e.exprs.length === n;
  check(ok, `expected comma(${n}), got ${e === null ? 'null' : e.kind}`);
  return ok ? (e as CommaExpr) : null;
}

/* ------------------------------------------------------------------ */
/* ES 1.00: globals, precision, structs, arrays, function definition   */
/* ------------------------------------------------------------------ */

{
  const ast = parseOk(
    `// leading comment
attribute vec4 aPos;
varying vec2 vUV;
uniform mat4 uMVP;
const float PI = 3.14159;
uniform vec3 uColor = vec3(1.0), uTint = vec3(0.5);
precision highp float;
precision lowp sampler2D;
struct Light {
  vec3 position;
  vec4 color;
};
Light light;
uniform float weights[4];
void main() {
  gl_Position = vec4(aPos.xy, 0.0, 1.0);
}
`,
    100,
  );
  check(ast.version === 100, '100 ast version');
  const d = ast.declarations;
  check(d.length === 11, `100 decl count: ${d.length}`);

  const g0 = gvar(d[0]);
  check(g0.type.qualifiers.storage === 'attribute', 'attribute storage');
  check(g0.type.base.kind === 'type-name' && g0.type.base.name === 'vec4', 'attribute base vec4');
  check(g0.declarators.length === 1 && g0.declarators[0].name === 'aPos', 'attribute declarator aPos');
  check(g0.declarators[0].arrayDims.length === 0 && g0.declarators[0].init === null, 'aPos no dims/init');
  check(g0.loc.line === 2, `attribute decl line: ${g0.loc.line}`);

  const g1 = gvar(d[1]);
  check(g1.type.qualifiers.storage === 'varying', 'varying storage');
  const g2 = gvar(d[2]);
  check(g2.type.qualifiers.storage === 'uniform' && g2.type.base.kind === 'type-name' && g2.type.base.name === 'mat4', 'uniform mat4');

  const g3 = gvar(d[3]);
  check(g3.type.qualifiers.storage === 'const' && g3.type.base.kind === 'type-name' && g3.type.base.name === 'float', 'const float');
  lit(g3.declarators[0].init, 3.14159, 'float');

  const g4 = gvar(d[4]);
  check(g4.declarators.length === 2, `two declarators: ${g4.declarators.length}`);
  check(g4.declarators[0].name === 'uColor' && g4.declarators[1].name === 'uTint', 'uColor, uTint names');
  check(g4.declarators[0].init !== null && g4.declarators[1].init !== null, 'both declarators initialized');
  const c0 = asCall(g4.declarators[0].init);
  id(c0 === null ? null : c0.callee, 'vec3');
  check(c0 !== null && c0.args.length === 1, 'vec3(1.0) one arg');
  lit(c0 === null ? null : c0.args[0], 1, 'float');

  const p1 = pdecl(d[5]);
  check(p1.precision === 'highp' && p1.base === 'float', 'precision highp float');
  const p2 = pdecl(d[6]);
  check(p2.precision === 'lowp' && p2.base === 'sampler2D', 'precision lowp sampler2D');

  const s = sdecl(d[7]);
  check(s.name === 'Light', 'struct Light');
  check(s.members.length === 2, `Light members: ${s.members.length}`);
  check(s.members[0].name === 'position' && s.members[0].type.base.kind === 'type-name' && s.members[0].type.base.name === 'vec3', 'member position vec3');
  check(Object.keys(s.members[0].type.qualifiers).length === 0, 'member has no qualifiers');
  check(s.members[1].name === 'color', 'member color');

  const g5 = gvar(d[8]);
  check(g5.type.base.kind === 'type-name' && g5.type.base.name === 'Light', 'struct-typed variable');
  check(g5.declarators[0].name === 'light', 'light declarator');

  const g6 = gvar(d[9]);
  check(g6.type.base.kind === 'type-name' && g6.type.base.name === 'float', 'weights base float');
  const dims = g6.declarators[0].arrayDims;
  check(dims.length === 1, 'weights has one dim');
  lit(dims[0], 4, 'int');

  const fn = fdef(d[10]);
  check(fn.prototype.name === 'main' && fn.prototype.params.length === 0, 'main prototype');
  check(fn.prototype.returnType.base.kind === 'type-name' && fn.prototype.returnType.base.name === 'void', 'main returns void');
  check(fn.loc.line === 15, `main line: ${fn.loc.line}`);
  check(fn.body.kind === 'compound' && fn.body.body.length === 1, 'main body compound');
  const es = fn.body.body[0] as ExprStmt;
  check(es.kind === 'expr-stmt', 'body[0] expr stmt');
  const asg = asAssign(es.expr, '=');
  id(asg === null ? null : asg.target, 'gl_Position');
  const call = asCall(asg === null ? null : asg.value);
  id(call === null ? null : call.callee, 'vec4');
  check(call !== null && call.args.length === 3, `vec4 args: ${call === null ? 'null' : call.args.length}`);
  const m0 = asMember(call === null ? null : call.args[0], 'xy');
  id(m0 === null ? null : m0.object, 'aPos');
  lit(call === null ? null : call.args[1], 0, 'float');
  lit(call === null ? null : call.args[2], 1, 'float');
}

{
  // Struct member precision qualifier is accepted (and ignored).
  const ast = parseOk('struct S {\n  highp float x;\n};\n', 100);
  const s = sdecl(ast.declarations[0]);
  check(s.members[0].type.qualifiers.precision === 'highp', 'member precision accepted');
}

{
  // precision mediump int; — int base.
  const ast = parseOk('precision mediump int;\n', 100);
  const p = pdecl(ast.declarations[0]);
  check(p.precision === 'mediump' && p.base === 'int', 'precision mediump int');
}

/* ------------------------------------------------------------------ */
/* ES 3.00: qualifiers, layout, interface blocks, switch, operators    */
/* ------------------------------------------------------------------ */

{
  const ast = parseOk(
    `#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) flat in vec3 aColor;
smooth out vec2 vUV;
centroid out vec4 vPos;
layout(binding = 0) uniform sampler2D uTex;
const uint MAXN = 5u;
uniform UB {
  vec4 color;
  mat4 mvp;
} ub[2];
void main() {
  int x = (a & b) | (c ^ d);
  uint n = MAXN;
  float f = x > 2 ? 1.0 : 2.0;
  x = 1, x += 2, x -= 3, x *= 4, x /= 5, x %= 6, x <<= 1, x >>= 2, x &= 3, x ^= 4, x |= 5;
  switch (x) {
    case 1:
      break;
    default:
      x = 0;
  }
}
`,
    300,
  );
  check(ast.version === 300, '300 ast version');
  const d = ast.declarations;
  check(d.length === 8, `300 decl count: ${d.length}`);

  const g0 = gvar(d[0]);
  check(g0.type.qualifiers.storage === 'in', 'in storage');
  check(g0.type.qualifiers.layout !== undefined && g0.type.qualifiers.layout.location === 0, 'layout(location=0)');
  check(g0.type.base.kind === 'type-name' && g0.type.base.name === 'vec2', 'aPos vec2');
  check(g0.declarators[0].name === 'aPos', 'aPos declarator');

  const g1 = gvar(d[1]);
  check(g1.type.qualifiers.storage === 'in' && g1.type.qualifiers.interpolation === 'flat', 'flat in');
  check(g1.type.qualifiers.layout !== undefined && g1.type.qualifiers.layout.location === 1, 'layout(location=1)');

  const g2 = gvar(d[2]);
  check(g2.type.qualifiers.storage === 'out' && g2.type.qualifiers.interpolation === 'smooth', 'smooth out');

  const g3 = gvar(d[3]);
  check(g3.type.qualifiers.storage === 'out' && g3.type.qualifiers.centroid === true, 'centroid out');

  const g4 = gvar(d[4]);
  check(
    g4.type.qualifiers.storage === 'uniform' && g4.type.qualifiers.layout !== undefined && g4.type.qualifiers.layout.binding === 0,
    'layout(binding=0) uniform',
  );
  check(g4.type.base.kind === 'type-name' && g4.type.base.name === 'sampler2D', 'sampler2D base');

  const g5 = gvar(d[5]);
  check(g5.type.base.kind === 'type-name' && g5.type.base.name === 'uint', 'uint base');
  lit(g5.declarators[0].init, 5, 'uint');

  const blk = iblock(d[6]);
  check(blk.blockName === 'UB' && blk.instanceName === 'ub', 'interface block UB ub');
  check(blk.qualifiers.storage === 'uniform', 'block uniform');
  check(blk.members.length === 2 && blk.members[0].name === 'color' && blk.members[1].name === 'mvp', 'block members');
  check(blk.members[1].type.base.kind === 'type-name' && blk.members[1].type.base.name === 'mat4', 'block mvp mat4');
  check(blk.arrayDims.length === 1, 'ub has one dim');
  lit(blk.arrayDims[0], 2, 'int');

  const fn = fdef(d[7]);
  const body = fn.body.body;
  check(body.length === 5, `main body stmts: ${body.length}`);

  // body[0]: int x = (a & b) | (c ^ d) — bitwise nesting
  const st0 = body[0] as DeclStmt;
  check(st0.kind === 'decl-stmt' && st0.type.base.kind === 'type-name' && st0.type.base.name === 'int', 'decl int x');
  const orE = asBin(st0.declarators[0].init, '|');
  const andE = orE === null ? null : asBin(orE.left, '&');
  id(andE === null ? null : andE.left, 'a');
  id(andE === null ? null : andE.right, 'b');
  const xorE = orE === null ? null : asBin(orE.right, '^');
  id(xorE === null ? null : xorE.left, 'c');
  id(xorE === null ? null : xorE.right, 'd');

  // body[1]: uint n = MAXN
  const st1 = body[1] as DeclStmt;
  check(st1.type.base.kind === 'type-name' && st1.type.base.name === 'uint', 'decl uint n');
  id(st1.declarators[0].init, 'MAXN');

  // body[2]: float f = x > 2 ? 1.0 : 2.0 — ternary
  const st2 = body[2] as DeclStmt;
  const tern = asTernary(st2.declarators[0].init);
  const tcond = tern === null ? null : asBin(tern.cond, '>');
  id(tcond === null ? null : tcond.left, 'x');
  lit(tcond === null ? null : tcond.right, 2, 'int');
  lit(tern === null ? null : tern.whenTrue, 1, 'float');
  lit(tern === null ? null : tern.whenFalse, 2, 'float');

  // body[3]: comma expression covering every assignment operator
  const st3 = body[3] as ExprStmt;
  const comma = asComma(st3.expr, 11);
  const ops = comma === null ? [] : comma.exprs.map((e) => (e.kind === 'assign' ? e.op : '?'));
  check(
    JSON.stringify(ops) === JSON.stringify(['=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '^=', '|=']),
    `assignment ops: ${JSON.stringify(ops)}`,
  );

  // body[4]: switch / case / default
  const sw = body[4] as SwitchStmt;
  check(sw.kind === 'switch', 'switch stmt');
  id(sw.expr, 'x');
  check(sw.body.kind === 'compound', 'switch body compound');
  const swb = (sw.body as CompoundStmt).body;
  check(swb.length === 4, `switch body stmts: ${swb.length}`);
  const c1 = swb[0] as CaseLabelStmt;
  check(c1.kind === 'case' && c1.value !== null && c1.value.kind === 'literal' && c1.value.value === 1, 'case 1:');
  check(swb[1].kind === 'break', 'break');
  const def = swb[2] as CaseLabelStmt;
  check(def.kind === 'case' && def.value === null, 'default:');
  const swa = swb[3] as ExprStmt;
  const swasg = asAssign(swa.expr, '=');
  id(swasg === null ? null : swasg.target, 'x');
  lit(swasg === null ? null : swasg.value, 0, 'int');
}

{
  // Unknown layout ids (std140, column_major) are accepted and ignored.
  const ast = parseOk('#version 300 es\nlayout(std140, column_major) uniform UB {\n  vec4 color;\n} ub;\n', 300);
  const blk = iblock(ast.declarations[0]);
  check(blk.blockName === 'UB' && blk.qualifiers.storage === 'uniform', 'std140 block parsed');
}

/* ------------------------------------------------------------------ */
/* Expression precedence                                               */
/* ------------------------------------------------------------------ */

{
  // 1 + 2 * 3 → +(1, *(2, 3))
  const init = gvar(parseOk('float r = 1 + 2 * 3;', 100).declarations[0]).declarators[0].init;
  const plus = asBin(init, '+');
  lit(plus === null ? null : plus.left, 1, 'int');
  const times = plus === null ? null : asBin(plus.right, '*');
  lit(times === null ? null : times.left, 2, 'int');
  lit(times === null ? null : times.right, 3, 'int');

  // a = b = c → right-associative
  const init2 = gvar(parseOk('float r = a = b = c;', 100).declarations[0]).declarators[0].init;
  const a1 = asAssign(init2, '=');
  id(a1 === null ? null : a1.target, 'a');
  const a2 = a1 === null ? null : asAssign(a1.value, '=');
  id(a2 === null ? null : a2.target, 'b');
  id(a2 === null ? null : a2.value, 'c');

  // a || b && c → ||(a, &&(b, c))
  const init3 = gvar(parseOk('float r = a || b && c;', 100).declarations[0]).declarators[0].init;
  const orE = asBin(init3, '||');
  id(orE === null ? null : orE.left, 'a');
  const andE = orE === null ? null : asBin(orE.right, '&&');
  id(andE === null ? null : andE.left, 'b');
  id(andE === null ? null : andE.right, 'c');

  // unary minus binds tighter than binary '*': -a * b
  const init4 = gvar(parseOk('float r = -a * b;', 100).declarations[0]).declarators[0].init;
  const mul = asBin(init4, '*');
  const neg = mul === null ? null : asUnary(mul.left, '-');
  id(neg === null ? null : neg.operand, 'a');
  id(mul === null ? null : mul.right, 'b');
  // ...while 'a - b' is a plain binary '-'
  const init5 = gvar(parseOk('float r = a - b;', 100).declarations[0]).declarators[0].init;
  const sub = asBin(init5, '-');
  id(sub === null ? null : sub.left, 'a');
  id(sub === null ? null : sub.right, 'b');

  // postfix chain: a[0].x → member(index(a, 0), x)
  const init6 = gvar(parseOk('float r = a[0].x;', 100).declarations[0]).declarators[0].init;
  const mem = asMember(init6, 'x');
  const idx = mem === null ? null : asIndex(mem.object);
  id(idx === null ? null : idx.object, 'a');
  lit(idx === null ? null : idx.index, 0, 'int');

  // call arguments
  const init7 = gvar(parseOk('float r = f(1, 2);', 100).declarations[0]).declarators[0].init;
  const call = asCall(init7);
  id(call === null ? null : call.callee, 'f');
  check(call !== null && call.args.length === 2, `call args: ${call === null ? 'null' : call.args.length}`);
  lit(call === null ? null : call.args[0], 1, 'int');
  lit(call === null ? null : call.args[1], 2, 'int');

  // comma operator at statement level
  const ast8 = parseOk('void main() { a = 1, b = 2; }', 100);
  const st8 = fdef(ast8.declarations[0]).body.body[0] as ExprStmt;
  const comma = asComma(st8.expr, 2);
  const e0 = comma === null ? null : comma.exprs[0];
  const a0 = asAssign(e0, '=');
  id(a0 === null ? null : a0.target, 'a');
  lit(a0 === null ? null : a0.value, 1, 'int');
  const e1 = comma === null ? null : comma.exprs[1];
  const a1b = asAssign(e1, '=');
  id(a1b === null ? null : a1b.target, 'b');
  lit(a1b === null ? null : a1b.value, 2, 'int');

  // ternary middle allows comma: a ? 1.0, 2.0 : 3.0
  const init9 = gvar(parseOk('float r = a ? 1.0, 2.0 : 3.0;', 100).declarations[0]).declarators[0].init;
  const tern = asTernary(init9);
  id(tern === null ? null : tern.cond, 'a');
  const mid = tern === null ? null : asComma(tern.whenTrue, 2);
  lit(mid === null ? null : mid.exprs[0], 1, 'float');
  lit(mid === null ? null : mid.exprs[1], 2, 'float');
  lit(tern === null ? null : tern.whenFalse, 3, 'float');
}

/* ------------------------------------------------------------------ */
/* #extension directives → ExtensionDecl (prepended, in order)         */
/* ------------------------------------------------------------------ */

{
  const exts = new Set(['GL_OES_standard_derivatives', 'GL_EXT_shader_texture_lod']);
  const ast = parseOk(
    `#extension GL_OES_standard_derivatives : enable
#extension GL_EXT_shader_texture_lod : require
precision highp float;
void main() { gl_Position = vec4(0.0); }
`,
    100,
    { extensions: exts },
  );
  const d = ast.declarations;
  check(d.length === 4, `extension decl count: ${d.length}`);
  const e0 = edecl(d[0]);
  check(e0.name === 'GL_OES_standard_derivatives' && e0.behavior === 'enable', 'ext1 enable');
  check(e0.loc.line === 1 && e0.loc.column === 0, `ext1 loc: ${e0.loc.line},${e0.loc.column}`);
  const e1 = edecl(d[1]);
  check(e1.name === 'GL_EXT_shader_texture_lod' && e1.behavior === 'require', 'ext2 require');
  check(e1.loc.line === 2 && e1.loc.column === 0, `ext2 loc: ${e1.loc.line},${e1.loc.column}`);
  check(d[2].kind === 'precision-decl' && d[3].kind === 'function-definition', 'parsed decls follow extensions');
}

/* ------------------------------------------------------------------ */
/* Empty declarators (conformance/glsl/misc/empty-declaration.html)     */
/* ------------------------------------------------------------------ */

{
  // OLD BEHAVIOR (bug): `float;` was rejected with "expected identifier,
  // found ';'". Per the ESSL Appendix A grammar `single_declaration` =
  // `fully_specified_type`, an empty declarator is legal (a side effect of
  // how the grammar for structs is defined). It declares NOTHING: no
  // VarDeclarator node is produced.
  const ast = parseOk('float;\n', 100);
  check(ast.declarations.length === 1, `empty-decl count: ${ast.declarations.length}`);
  const g = gvar(ast.declarations[0]);
  check(g.declarators.length === 0, 'float; → zero declarators');
}

{
  // OLD BEHAVIOR (bug): `float, a = 0.0;` was rejected — the empty FIRST
  // declarator is legal (`init_declarator_list COMMA init_declarator` with
  // `single_declaration` = just the type). Only the first slot may be empty.
  const ast = parseOk('float, a = 0.0;\n', 100);
  const g = gvar(ast.declarations[0]);
  check(g.declarators.length === 1, `float, a = 0.0 → one declarator (got ${g.declarators.length})`);
  check(g.declarators[0].name === 'a', 'empty-first declarator name a');
  lit(g.declarators[0].init, 0.0, 'float');
}

{
  // OLD BEHAVIOR (bug): `struct S {...}, a;` was rejected at the `,`.
  const ast = parseOk('struct S { float member; }, a;\n', 100);
  const g = gvar(ast.declarations[0]);
  check(g.type.base.kind === 'struct-definition', 'struct empty-first is a global-var-decl');
  check(g.declarators.length === 1 && g.declarators[0].name === 'a', 'struct empty-first declarator a');
}

{
  // The same empty-first form is legal INSIDE function bodies (declaration
  // statement → decl-stmt with zero declarators).
  const ast = parseOk('void main() { float; gl_Position = vec4(0.0); }\n', 100);
  const fn = fdef(ast.declarations[0]);
  const st = fn.body.body[0] as DeclStmt;
  check(st.kind === 'decl-stmt', 'in-body float; is a decl-stmt');
  check(st.declarators.length === 0, 'in-body float; → zero declarators');
}

{
  // In-struct declarations use struct_declarator_list (a NAME is required) —
  // empty declarators there must STILL fail (the 2 in-struct subtests of
  // empty-declaration.html expect failure; do not loosen).
  const errs1 = parseFail('struct S { float; float a; };\n', 100);
  check(errs1.length >= 1, `in-struct float; fails: ${errs1.length} error(s)`);
  check(errs1[0].message.includes("expected identifier, found ';'"), `in-struct float; msg: ${errs1[0].message}`);
  const errs2 = parseFail('struct S { float, a; float b; };\n', 100);
  check(errs2.length >= 1, `in-struct float, a fails: ${errs2.length} error(s)`);
}

{
  // The empty slot is only legal in FIRST position: `float a, ;` still fails
  // (grammar: init_declarator_list COMMA init_declarator — after a comma a
  // real declarator is required).
  const errs = parseFail('void main() { float a, ; }\n', 100);
  check(errs.length >= 1, `trailing comma fails: ${errs.length} error(s)`);
  check(errs[0].message.includes("expected identifier, found ';'"), `trailing comma msg: ${errs[0].message}`);
}

/* ------------------------------------------------------------------ */
/* Precision statements inside function bodies                         */
/* (conformance/glsl/misc/ternary-operators-in-initializers.html)       */
/* ------------------------------------------------------------------ */

{
  // OLD BEHAVIOR (bug): `precision mediump float;` as the first statement in
  // main() was rejected with "syntax error, unexpected 'precision'". It is a
  // legal declaration statement (ESSL Appendix A) — ANGLE accepts it. The
  // statement slot parses as an `empty` node and the PrecisionDecl is
  // HOISTED to just before the enclosing function definition in the global
  // declaration list (so the semantics pre-pass processes it before the
  // body; fragment float declarations then see the default precision).
  const ast = parseOk('void main() { precision mediump float; float i = 2.0; gl_FragColor = vec4(0.0, i, 0.0, 1.0); }\n', 100);
  check(ast.declarations.length === 2, `hoisted precision-decl + fn (got ${ast.declarations.length})`);
  check(ast.declarations[0].kind === 'precision-decl', 'hoisted decl precedes the function definition');
  const p = pdecl(ast.declarations[0]);
  check(p.precision === 'mediump' && p.base === 'float', 'hoisted precision mediump float');
  const fn = fdef(ast.declarations[1]);
  const st = fn.body.body[0];
  check(st.kind === 'empty', `in-body precision statement slot is empty (got ${st.kind})`);
}

{
  // Multiple in-body precision statements keep source order, both hoisted
  // before the function definition.
  const ast = parseOk('void main() { precision mediump float; precision lowp int; }\n', 100);
  check(ast.declarations.length === 3, `two hoisted + fn (got ${ast.declarations.length})`);
  const p1 = pdecl(ast.declarations[0]);
  const p2 = pdecl(ast.declarations[1]);
  check(p1.base === 'float' && p1.precision === 'mediump', 'first hoisted = mediump float');
  check(p2.base === 'int' && p2.precision === 'lowp', 'second hoisted = lowp int');
}

{
  // The same form works in ES 3.00 and in vertex shaders.
  const ast300 = parseOk('#version 300 es\nvoid main() { precision highp float; vec2 i = vec2(1.0); }\n', 300);
  check(ast300.declarations[0].kind === 'precision-decl', '300 in-body precision hoisted');
  const astV = parseOk('void main() { precision lowp float; }\n', 100);
  check(astV.declarations[0].kind === 'precision-decl', 'vertex in-body precision hoisted');
}

{
  // Invalid precision statements inside bodies still error (bad base type).
  const errs = parseFail('void main() { precision highp vec3; }\n', 100);
  check(errs.length >= 1, `in-body precision vec3 fails: ${errs.length} error(s)`);
  check(errs[0].message.includes('invalid base type in precision declaration'), `in-body precision vec3 msg: ${errs[0].message}`);
}

/* ------------------------------------------------------------------ */
/* Error cases with exact line numbers (ES 1.00)                       */
/* ------------------------------------------------------------------ */

{
  // Missing semicolon: error reported on the `}` line (the token at fault).
  const errs = parseFail('void main() {\n  float x = 1.0\n}\n', 100);
  check(errs.length === 1, `missing-semicolon count: ${errs.length}`);
  check(errs[0].line === 3, `missing-semicolon line: ${errs[0].line}`);
  check(errs[0].message.includes("expected ';' after declaration"), `missing-semicolon msg: ${errs[0].message}`);
}

{
  // Unterminated function body: error on the EOF line (last source line).
  const errs = parseFail('void main() {\n  gl_Position = vec4(0.0);\n', 100);
  check(errs.length === 1, `unterminated count: ${errs.length}`);
  check(errs[0].line === 2, `unterminated line: ${errs[0].line}`);
  check(errs[0].message.includes("unterminated block, expected '}'"), `unterminated msg: ${errs[0].message}`);
}

{
  // `switch` is reserved in GLSL ES 1.00.
  const errs = parseFail('void main() {\n  switch (x) {\n    case 1: break;\n  }\n}\n', 100);
  check(errs.length === 1, `switch-100 count: ${errs.length}`);
  check(errs[0].line === 2, `switch-100 line: ${errs[0].line}`);
  check(errs[0].message === "'switch' is reserved in GLSL ES 1.00", `switch-100 msg: ${errs[0].message}`);
}

{
  // `in` is reserved in GLSL ES 1.00.
  const errs = parseFail('in vec4 pos;\n', 100);
  check(errs.length === 1, `in-100 count: ${errs.length}`);
  check(errs[0].line === 1, `in-100 line: ${errs[0].line}`);
  check(errs[0].message === "'in' is reserved in GLSL ES 1.00", `in-100 msg: ${errs[0].message}`);
}

{
  // `in`/`out`/`inout` ARE legal in GLSL ES 1.00 function parameter lists
  // (§6.1) — the `inout` rejection below was the OLD buggy behavior.
  const ast = parseOk('float f(inout float x);\n', 100);
  const f = fproto(ast.declarations[0]);
  check(f.params.length === 1 && f.params[0].type.qualifiers.storage === 'inout', '1.00 inout param accepted, storage inout');
  check(f.params[0].name === 'x', '1.00 inout param name');
}

{
  // `layout` is an identifier in 1.00 → natural syntax error (message free).
  const errs = parseFail('layout(location = 0) in vec4 pos;\n', 100);
  check(errs.length === 1, `layout-100 count: ${errs.length}`);
  check(errs[0].line === 1, `layout-100 line: ${errs[0].line}`);
}

{
  // `attribute` is an identifier in 3.00 → `attribute vec4 pos;` is
  // "expected identifier, found 'vec4'" (two type names in a row).
  const errs = parseFail('#version 300 es\nattribute vec4 pos;\n', 300);
  check(errs.length === 1, `attribute-300 count: ${errs.length}`);
  check(errs[0].line === 2, `attribute-300 line: ${errs[0].line}`);
  check(errs[0].message.includes("expected identifier, found 'vec4'"), `attribute-300 msg: ${errs[0].message}`);
}

{
  // Binary bitwise operators require GLSL ES 3.00.
  const errs = parseFail('void main() {\n  int x = a & b;\n}\n', 100);
  check(errs.length === 1, `bitwise-100 count: ${errs.length}`);
  check(errs[0].line === 2, `bitwise-100 line: ${errs[0].line}`);
  check(errs[0].message === "'&' : bitwise operators require GLSL ES 3.00", `bitwise-100 msg: ${errs[0].message}`);
}

{
  // Bitwise ASSIGNMENT operators require GLSL ES 3.00 too.
  const errs = parseFail('void main() {\n  x <<= 1;\n}\n', 100);
  check(errs.length === 1, `bitwise-assign-100 count: ${errs.length}`);
  check(errs[0].line === 2, `bitwise-assign-100 line: ${errs[0].line}`);
  check(errs[0].message === "'<<=' : bitwise operators require GLSL ES 3.00", `bitwise-assign-100 msg: ${errs[0].message}`);
}

{
  // Anonymous structs are not allowed in GLSL ES.
  const errs = parseFail('struct {\n  float x;\n} s;\n', 100);
  check(errs.length === 1, `anon-struct count: ${errs.length}`);
  check(errs[0].line === 1, `anon-struct line: ${errs[0].line}`);
  check(errs[0].message.includes('anonymous structs'), `anon-struct msg: ${errs[0].message}`);
}

{
  // Struct definitions inside a struct member are not allowed.
  const errs = parseFail('struct Outer {\n  struct Inner { float x; } i;\n};\n', 100);
  check(errs.length === 1, `nested-struct count: ${errs.length}`);
  check(errs[0].line === 2, `nested-struct line: ${errs[0].line}`);
  check(errs[0].message === 'struct definitions are not allowed inside structs', `nested-struct msg: ${errs[0].message}`);
}

{
  // `precision highp vec3;` — base must be float/int/sampler.
  const errs = parseFail('precision highp vec3;\n', 100);
  check(errs.length === 1, `precision-vec3 count: ${errs.length}`);
  check(errs[0].line === 1, `precision-vec3 line: ${errs[0].line}`);
  check(errs[0].message.includes('invalid base type in precision declaration'), `precision-vec3 msg: ${errs[0].message}`);
}

{
  // Interface blocks require GLSL ES 3.00.
  const errs = parseFail('uniform UB {\n  vec4 color;\n} ub;\n', 100);
  check(errs.length === 1, `block-100 count: ${errs.length}`);
  check(errs[0].line === 1, `block-100 line: ${errs[0].line}`);
  check(errs[0].message === 'interface blocks require GLSL ES 3.00', `block-100 msg: ${errs[0].message}`);
}

/* ------------------------------------------------------------------ */
/* Error cases with exact line numbers (ES 3.00)                       */
/* ------------------------------------------------------------------ */

{
  // Bad case label: `case 1 2:` → missing ':' after the value expression.
  const errs = parseFail('#version 300 es\nvoid main() {\n  switch (x) {\n    case 1 2:\n      break;\n  }\n}\n', 300);
  check(errs.length >= 1, 'bad-case errors present');
  check(errs[0].line === 4, `bad-case first line: ${errs[0].line}`);
  check(errs[0].message === "expected ':' after case label", `bad-case msg: ${errs[0].message}`);
}

{
  // `case 1.5:` is a PARSE-TIME SUCCESS — integrality of case labels is a
  // semantics (type-checking) concern, not syntax. Documents permissiveness.
  const ast = parseOk('#version 300 es\nvoid main() {\n  switch (x) {\n    case 1.5:\n      break;\n  }\n}\n', 300);
  const sw = fdef(ast.declarations[0]).body.body[0] as SwitchStmt;
  const c = (sw.body as CompoundStmt).body[0] as CaseLabelStmt;
  lit(c.value, 1.5, 'float');
}

{
  // A switch body must be a compound statement.
  const errs = parseFail('#version 300 es\nvoid main() {\n  switch (x) x = 1;\n}\n', 300);
  check(errs.length === 1, `switch-noncompound count: ${errs.length}`);
  check(errs[0].line === 3, `switch-noncompound line: ${errs[0].line}`);
  check(errs[0].message === 'switch body must be a compound statement', `switch-noncompound msg: ${errs[0].message}`);
}

/* ------------------------------------------------------------------ */
/* Function parameters and prototypes                                  */
/* ------------------------------------------------------------------ */

{
  // Unnamed params in prototypes; `f(void)` = empty list; named def params.
  const ast = parseOk('float f(float);\nfloat g(void);\nfloat h(float x) { return x; }\n', 100);
  const d = ast.declarations;
  check(d.length === 3, `param decl count: ${d.length}`);
  const f = fproto(d[0]);
  check(f.name === 'f' && f.params.length === 1, 'f prototype one param');
  check(
    f.params[0].name === '' && f.params[0].type.base.kind === 'type-name' && f.params[0].type.base.name === 'float' && f.params[0].arrayDims.length === 0,
    'unnamed float param',
  );
  const g = fproto(d[1]);
  check(g.name === 'g' && g.params.length === 0, 'f(void) empty params');
  const h = fdef(d[2]);
  check(h.prototype.params.length === 1 && h.prototype.params[0].name === 'x', 'h param named x');
  const rt = h.body.body[0] as ReturnStmt;
  check(rt.kind === 'return', 'h body return stmt');
  id(rt.value, 'x');
}

{
  // Unnamed params in DEFINITIONS are allowed (permissive).
  const ast = parseOk('float f(float) { return 0.0; }\n', 100);
  const fn = fdef(ast.declarations[0]);
  check(fn.prototype.params.length === 1 && fn.prototype.params[0].name === '', 'unnamed definition param');
}

{
  // Unsized `[]` array params: allowed in BOTH versions (permissive).
  const ast = parseOk('float f(float a[]);\n', 100);
  const f = fproto(ast.declarations[0]);
  check(f.params[0].arrayDims.length === 1 && f.params[0].arrayDims[0] === null, '100 unsized param [] → null dim');
  const ast2 = parseOk('#version 300 es\nfloat f(float a[]);\n', 300);
  const f2 = fproto(ast2.declarations[0]);
  check(f2.params[0].arrayDims.length === 1 && f2.params[0].arrayDims[0] === null, '300 unsized param [] → null dim');
}

{
  // inout params are accepted in 3.00 and recorded with storage 'inout'
  // (semantics uses the qualifier for by-reference marshaling).
  const ast = parseOk('#version 300 es\nfloat f(inout float x);\n', 300);
  const f = fproto(ast.declarations[0]);
  check(f.params[0].type.qualifiers.storage === 'inout', 'inout recorded as inout');
  check(f.params[0].name === 'x', 'inout param name');
}

{
  // GLSL ES 1.00 §6.1: in/out/inout are legal function parameter qualifiers
  // (reserved words everywhere ELSE in 1.00). Stored exactly like 3.00.
  const ast = parseOk('float f(in float a, out float b, inout float c);\n', 100);
  const f = fproto(ast.declarations[0]);
  check(f.params.length === 3, '1.00 three qualified params');
  check(f.params[0].type.qualifiers.storage === 'in', '1.00 in param storage');
  check(f.params[1].type.qualifiers.storage === 'out', '1.00 out param storage');
  check(f.params[2].type.qualifiers.storage === 'inout', '1.00 inout param storage');
  // `const in` combination also parses in 1.00.
  const ast2 = parseOk('float g(const in float x);\n', 100);
  const g = fproto(ast2.declarations[0]);
  check(g.params[0].type.qualifiers.storage === 'in', '1.00 const in param storage');
  // Same qualifiers work in 1.00 function DEFINITIONS, not just prototypes.
  const ast3 = parseOk('float h(out float x) { x = 1.0; return x; }\n', 100);
  const h = fdef(ast3.declarations[0]);
  check(h.prototype.params[0].type.qualifiers.storage === 'out', '1.00 out param in definition');
  // Reserved-word use OUTSIDE parameter lists still fails in 1.00: as a
  // function name / declarator identifier, and in storage-class position.
  const errs1 = parseFail('float in(float x);\n', 100);
  check(errs1.length === 1, `fn-name-in-100 count: ${errs1.length}`);
  check(errs1[0].message.includes("expected identifier, found 'in'"), `fn-name-in-100 msg: ${errs1[0].message}`);
  const errs2 = parseFail('void main() { out vec4 pos; }\n', 100);
  check(errs2.length === 1, `stmt-storage-out-100 count: ${errs2.length}`);
  check(errs2[0].message === "'out' is reserved in GLSL ES 1.00", `stmt-storage-out-100 msg: ${errs2[0].message}`);
  // 3.00 regression: in/out params still parse there.
  const ast4 = parseOk('#version 300 es\nfloat i(out float x);\n', 300);
  check(fproto(ast4.declarations[0]).params[0].type.qualifiers.storage === 'out', '3.00 out param storage');
}

/* ------------------------------------------------------------------ */
/* ES 3.00 array constructor: float[3](...)                            */
/* ------------------------------------------------------------------ */

{
  const ast = parseOk('#version 300 es\nvoid main() {\n  vec3 v = float[3](1.0, 2.0, 3.0);\n}\n', 300);
  const st = fdef(ast.declarations[0]).body.body[0] as DeclStmt;
  check(st.kind === 'decl-stmt', 'array-ctor decl stmt');
  const call = asCall(st.declarators[0].init);
  check(call !== null && call.args.length === 3, 'array ctor 3 args');
  const callee = call === null ? null : asIndex(call.callee);
  id(callee === null ? null : callee.object, 'float');
  lit(callee === null ? null : callee.index, 3, 'int');
}

/* ------------------------------------------------------------------ */
/* Statement forms: for/do-while/nested if-else/discard                */
/* ------------------------------------------------------------------ */

{
  const ast = parseOk(
    `void main() {
  for (int i = 0; i < 10; i++) {
    if (i > 5) {
      x = 1;
    } else if (i == 5) {
      x = 2;
    } else {
      x = 3;
    }
  }
  do {
    x++;
  } while (x < 10);
  if (x > 100) discard;
}
`,
    100,
  );
  const fn = fdef(ast.declarations[0]);
  const b = fn.body.body;
  check(b.length === 3, `stmt count: ${b.length}`);

  // for with declaration init
  const for1 = b[0] as ForStmt;
  check(for1.kind === 'for', 'for stmt');
  const initSt = for1.init as DeclStmt;
  check(initSt.kind === 'decl-stmt' && initSt.type.base.kind === 'type-name' && initSt.type.base.name === 'int', 'for decl init');
  check(initSt.declarators.length === 1 && initSt.declarators[0].name === 'i', 'for init i');
  lit(initSt.declarators[0].init, 0, 'int');
  const cond = asBin(for1.cond, '<');
  id(cond === null ? null : cond.left, 'i');
  lit(cond === null ? null : cond.right, 10, 'int');
  const upd = asUnary(for1.update, '++');
  id(upd === null ? null : upd.operand, 'i');
  check(upd !== null && upd.postfix === true, 'for update is POSTFIX ++');

  // nested if / else-if / else
  const if1 = (for1.body as CompoundStmt).body[0] as IfStmt;
  check(if1.kind === 'if', 'if stmt');
  const ifcond = asBin(if1.cond, '>');
  id(ifcond === null ? null : ifcond.left, 'i');
  lit(ifcond === null ? null : ifcond.right, 5, 'int');
  check(if1.then.kind === 'compound' && (if1.then as CompoundStmt).body.length === 1, 'if then compound');
  const els = if1.else as IfStmt;
  check(els.kind === 'if', 'else if');
  const elscond = asBin(els.cond, '==');
  id(elscond === null ? null : elscond.left, 'i');
  lit(elscond === null ? null : elscond.right, 5, 'int');
  check(els.else !== null && els.else.kind === 'compound', 'final else compound');

  // do-while with postfix ++ in the body
  const dow = b[1] as DoWhileStmt;
  check(dow.kind === 'do-while', 'do-while');
  const inc = (dow.body as CompoundStmt).body[0] as ExprStmt;
  const incU = asUnary(inc.expr, '++');
  id(incU === null ? null : incU.operand, 'x');
  check(incU !== null && incU.postfix === true, 'do-while body ++ is POSTFIX');
  const dcond = asBin(dow.cond, '<');
  id(dcond === null ? null : dcond.left, 'x');
  lit(dcond === null ? null : dcond.right, 10, 'int');

  // discard
  const if2 = b[2] as IfStmt;
  const if2c = asBin(if2.cond, '>');
  id(if2c === null ? null : if2c.left, 'x');
  lit(if2c === null ? null : if2c.right, 100, 'int');
  check(if2.then.kind === 'discard', 'discard stmt');
  check(if2.else === null, 'no else');
}

/* ------------------------------------------------------------------ */
/* Invariant declarations                                              */
/* ------------------------------------------------------------------ */

{
  const ast = parseOk('invariant gl_Position;\ninvariant varying vec4 v;\n', 100);
  const d = ast.declarations;
  check(d.length === 2, `invariant count: ${d.length}`);
  const iv = idecl(d[0]);
  check(iv.name === 'gl_Position', 'invariant gl_Position');
  const gv = gvar(d[1]);
  check(gv.type.qualifiers.invariant === true && gv.type.qualifiers.storage === 'varying', 'invariant varying vec4 v');
  check(gv.declarators[0].name === 'v', 'invariant declarator v');
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`parser selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
