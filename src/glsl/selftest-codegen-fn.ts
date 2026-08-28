/**
 * selftest-codegen-fn.ts — direct unit checks for codegen/functions.ts
 * (the USER-FUNCTION INLINER: installUserFunctions + per-call-site inlining).
 *
 * Run: npx tsx src/glsl/selftest-codegen-fn.ts
 *
 * Modeled on selftest-codegen-stmt.ts: real shaders are compiled via
 * compileShader (preprocess → lexer → parse → semantics — all implemented),
 * installUserFunctions wires the inliner into the env, `main` is lowered
 * with emitStatements into a JS function body (`new Function('ctx','R', body)`),
 * and the resulting ctx state is asserted. All shaders are VERTEX mains
 * writing gl_Position (env maps it straight to ctx.out.position — no layout
 * entries needed).
 *
 * Prints "OK" and exits 0 on success.
 */
import type {
  DeclStmt, Expr, ExprStmt, FunctionDefinition, GlobalVarDecl, LiteralExpr, ParamDecl,
  ReturnStmt, Stmt, TranslationUnit, TypeSpec,
} from './ast.js';
import type { GLSLType } from './types.js';
import { compileShader } from './compiler.js';
import { CodegenEnv } from './codegen/env.js';
import { installUserFunctions, installUserGlobals } from './codegen/functions.js';
import { emitStatements } from './codegen/statements.js';
import type { CodegenLayout } from './codegen/index.js';
import { R } from './codegen/runtime.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/* ------------------------------------------------------------------ */
/* Driver: compile a shader, install user functions, emit main, run it */
/* ------------------------------------------------------------------ */

function findFn(tu: TranslationUnit, name: string): FunctionDefinition {
  for (const d of tu.declarations) {
    if (d.kind === 'function-definition' && d.prototype.name === name) return d;
  }
  throw new Error(`no function '${name}' in shader`);
}

function baseLayout(version: 100 | 300): CodegenLayout {
  return {
    version,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map(),
    outputLocations: new Map([['gl_FragColor', 0]]),
    uses: {
      pointSize: false,
      fragCoord: false,
      frontFacing: false,
      pointCoord: false,
      fragDepth: false,
      vertexId: false,
      instanceId: false,
      drawId: false,
      derivatives: false,
      depthRange: false,
    },
  };
}

interface RunResult {
  body: string;
  ctx: Record<string, any>;
}

/** Compile + install + emit + run a VERTEX main. */
function runVertex(src: string, opts?: { version?: 100 | 300 }): RunResult {
  const version = opts?.version ?? 100;
  // out/inout params are ES 3.00-only syntax — prefix the shader accordingly.
  const fullSrc = version === 300 ? `#version 300 es\n${src}` : src;
  const r = compileShader(fullSrc, { type: 'VERTEX', version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('VERTEX', baseLayout(version));
  installUserFunctions(r.shader.ast, env);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    out: { position: [0, 0, 0, 0], pointSize: 0 },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
  };
  fn(ctx, R);
  return { body, ctx };
}

/* ------------------------------------------------------------------ */
/* 1. scalar function with return                                      */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(`float twice(float x) { return 2.0 * x; } void main() { gl_Position.x = twice(3.0); }`);
  check(r.ctx.out.position[0] === 6, `scalar return: twice(3.0) === 6 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 2. vector return                                                    */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `vec4 makev(float a) { return vec4(a, a + 1.0, a + 2.0, a + 3.0); } void main() { gl_Position = makev(1.0); }`,
  );
  const p = r.ctx.out.position;
  check(
    p[0] === 1 && p[1] === 2 && p[2] === 3 && p[3] === 4,
    `vector return: makev(1.0) === [1,2,3,4] (got [${p.join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* 3. void function with out param                                     */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void setv(out float x) { x = 42.0; } void main() { float a = 0.0; setv(a); gl_Position.x = a; }`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 42, `out param write-back: a === 42 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 4. inout param (statement-position += is fine)                      */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void bump(inout float x) { x += 1.0; } void main() { float a = 0.0; bump(a); bump(a); gl_Position.x = a; }`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 2, `inout twice: a === 2 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 5. multiple params, mixed in/out/inout                              */
/* ------------------------------------------------------------------ */

{
  // calc(1.0, b, c): a=1 (in), b = a + c = 4 (out), c = c*2 = 6 (inout), return a = 1
  const r = runVertex(
    `float calc(float a, out float b, inout float c) { b = a + c; c = c * 2.0; return a; }
     void main() { float b = 0.0; float c = 3.0; float rr = calc(1.0, b, c); gl_Position.x = rr * 100.0 + b * 10.0 + c; }`,
    { version: 300 },
  );
  check(
    r.ctx.out.position[0] === 146,
    `mixed params: r=1, b=4, c=6 → 146 (got ${r.ctx.out.position[0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 6. early return + out write-back                                    */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void f(out float x) { x = 1.0; return; x = 2.0; } void main() { float a = 0.0; f(a); gl_Position.x = a; }`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 1, `early return still write-backs: a === 1 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 7. nested calls (f inside g)                                        */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float f(float x) { return x + 1.0; } float g(float x) { return f(x) * 2.0; }
     void main() { gl_Position.x = g(5.0); }`,
  );
  check(r.ctx.out.position[0] === 12, `nested call: g(5.0) === 12 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 8. internal loop with break/continue                                */
/* ------------------------------------------------------------------ */

{
  // i: 0 → +0; 1 → continue; 2 → +2; 3 → +3; 4 → break. sum = 5
  const r = runVertex(
    `float ssum() {
       float s = 0.0;
       for (int i = 0; i < 5; i++) {
         if (i == 1) { continue; }
         if (i == 4) { break; }
         s = s + float(i);
       }
       return s;
     }
     void main() { gl_Position.x = ssum(); }`,
  );
  check(r.ctx.out.position[0] === 5, `loop break/continue: sum === 5 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 9. return mid-loop                                                  */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float sum() {
       float s = 0.0;
       for (int i = 0; i < 10; i++) {
         if (i == 3) { return s; }
         s = s + 1.0;
       }
       return s;
     }
     void main() { gl_Position.x = sum(); }`,
  );
  check(r.ctx.out.position[0] === 3, `return mid-loop: sum() === 3 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 10. depth-3 call chain                                              */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float a(float x) { return x + 1.0; }
     float b(float x) { return a(x) * 2.0; }
     float c(float x) { return b(x) + 3.0; }
     void main() { gl_Position.x = c(1.0); }`,
  );
  check(r.ctx.out.position[0] === 7, `depth-3 chain: c(1.0) === 7 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 11. builtins inside a function body                                 */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float f(float x) { return sin(x) + mix(1.0, 3.0, 0.5); } void main() { gl_Position.x = f(0.0); }`,
  );
  check(r.ctx.out.position[0] === 2, `builtins in body: sin(0)+mix(1,3,.5) === 2 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 12. void function with no return statement                          */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void nop() { float x = 1.0; } void main() { nop(); gl_Position.x = 7.0; }`,
  );
  check(r.ctx.out.position[0] === 7, `void no-return call + code after: x === 7 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 13. integer-returning function                                      */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `int seven() { return 7; } void main() { gl_Position.x = float(seven()); }`,
  );
  check(r.ctx.out.position[0] === 7, `int return: float(seven()) === 7 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 14. same-named params in sibling functions (frame isolation)        */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float sq(float x) { return x * x; } float dbl(float x) { return x + x; }
     void main() { gl_Position.x = sq(3.0) + dbl(4.0); }`,
  );
  check(r.ctx.out.position[0] === 17, `sibling same-named params: 9 + 8 === 17 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 15. nested calls with SAME-named params (frame top-down)            */
/* ------------------------------------------------------------------ */

{
  // g's x stays 5 while f's x (also named x) is live inside f.
  const r = runVertex(
    `float f(float x) { return x + 1.0; } float g(float x) { return f(x) + x; }
     void main() { gl_Position.x = g(5.0); }`,
  );
  check(r.ctx.out.position[0] === 11, `nested same-named params: f(5)+5 === 11 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 16. out param through a dynamic-index lvalue                        */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void setarr(out float x) { x = 9.0; }
     void main() {
       float a[4]; a[0] = 0.0; a[1] = 0.0; a[2] = 0.0; a[3] = 0.0;
       int i = 2; setarr(a[i]); gl_Position.x = a[2];
     }`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 9, `out via a[i]: a[2] === 9 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 17. function with an internal loop calling another function         */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float inner(float x) { return x * 2.0; }
     float acc() { float s = 0.0; for (int i = 0; i < 3; i++) { s = s + inner(float(i)); } return s; }
     void main() { gl_Position.x = acc(); }`,
  );
  check(r.ctx.out.position[0] === 6, `loop + inner call: 0+2+4 === 6 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 18. void call INSIDE another function (IIFE-in-IIFE)                */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void helper(inout float x) { x = x + 1.0; }
     void run(inout float v) { helper(v); }
     void main() { float a = 0.0; run(a); gl_Position.x = a; }`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 1, `void-in-void chain: a === 1 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 19. call result inside a CONDITION (foldPre comma-term path)        */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float f(float x) { return x; }
     void main() { if (f(2.0) > 1.0) { gl_Position.x = 5.0; } else { gl_Position.x = 7.0; } }`,
  );
  check(r.ctx.out.position[0] === 5, `call in if-condition: x === 5 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 20. call result inside a TERNARY                                    */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float f(float x) { return x; }
     void main() { gl_Position.x = f(1.0) > 0.5 ? 3.0 : 4.0; }`,
  );
  check(r.ctx.out.position[0] === 3, `call in ternary cond: x === 3 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 21. same-named local in main and in a function (IIFE var scoping)   */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float f() { float x = 100.0; return x; }
     void main() { float x = 1.0; gl_Position.x = f() + x; }`,
  );
  check(r.ctx.out.position[0] === 101, `main-local x survives inline's x: 100+1 === 101 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 22. ogles functions collision: CALLEE local named like a CALLER     */
/*     local that is passed as an ARG (functions.ts per-call-site      */
/*     locals — regresses the 9 black-rendering ogles functions pages: */
/*     functions_057_to_126.html). Before the fix the callee's `var`   */
/*     reused the caller's JS name, and IIFE hoisting shadowed the arg */
/*     materialization read with `undefined` → is_all(ret,true) got a  */
/*     garbage arg → gray stayed 0 (black pixels).                     */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `bvec4 function(in bvec4 par);
     bool is_all(const in bvec4 par, const in bool value);
     void set_all(out bvec4 par, const in bool value);
     void main(void) {
       bvec4 par = bvec4(true, true, true, true);
       bvec4 ret = bvec4(false, false, false, false);
       ret = function(par);
       if (is_all(par, true) && is_all(ret, true)) { gl_Position = vec4(1.0); }
       else { gl_Position = vec4(0.0); }
     }
     bvec4 function(in bvec4 par) {
       if (is_all(par, true)) { set_all(par, false); return bvec4(true, true, true, true); }
       else { return bvec4(false, false, false, false); }
     }
     bool is_all(const in bvec4 par, const in bool value) {
       bool ret = true;
       if (par[0] != value) ret = false;
       if (par[1] != value) ret = false;
       if (par[2] != value) ret = false;
       if (par[3] != value) ret = false;
       return ret;
     }
     void set_all(out bvec4 par, const in bool value) {
       par[0] = value; par[1] = value; par[2] = value; par[3] = value;
     }`,
    { version: 300 },
  );
  check(
    r.ctx.out.position[0] === 1,
    `ogles collision (callee local 'ret' vs caller arg 'ret'): white path (got ${r.ctx.out.position[0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 23. nested call: f's local x passed as ARG to g, which declares its */
/*     own local x (resolveLocal's enclosing-frame fallback). Before   */
/*     the fix the innermost frame's `localNames` short-circuited to   */
/*     locals_, missing f's per-call-site x → `unknown identifier` or  */
/*     a hoisted-undefined arg.                                        */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float g(float b) { float x = b * 2.0; return x; }
     float f(float a) { float x = a * 3.0; return g(x) + x; }
     void main() { gl_Position.x = f(1.0); }`,
  );
  check(r.ctx.out.position[0] === 9, `nested same-named locals: f(1.0) === 9 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 24. same-named ARRAY local in caller and callee (per-call-site      */
/*     scratch). Before the fix inlined-body array locals shared one   */
/*     scratch region with the caller (silent aliasing).               */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `float sum2(float a[2]) { float x[2]; x[0] = a[0]; x[1] = a[1]; return x[0] + x[1]; }
     void main() {
       float x[2]; x[0] = 1.0; x[1] = 2.0;
       float y[2]; y[0] = 3.0; y[1] = 4.0;
       if (sum2(x) == 3.0 && sum2(y) == 7.0 && x[0] == 1.0 && y[1] == 4.0) { gl_Position.x = 1.0; }
       else { gl_Position.x = 0.0; }
     }`,
  );
  check(
    r.ctx.out.position[0] === 1,
    `same-named array locals (caller+callee, 2 call sites) don't alias (got ${r.ctx.out.position[0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 25. STRUCT param with an ARRAY MEMBER, passed by value (flatNames   */
/*     struct recursion must flatten per element). Pinned regression   */
/*     for CTS conformance/glsl/bugs/sampler-array-struct-function-    */
/*     arg.html (`struct S { sampler2D sam[2]; }; vec4 useSampler(S    */
/*     arg) { return texture2D(arg.sam[0], ...); }`): before the       */
/*     `array` case in flatNames's struct recursion (env.ts),          */
/*     makeParamLocal hit `default:` and threw "codegen: array member  */
/*     'arg$c0__f' inside a flat struct is unsupported" → the link     */
/*     failed (page: 1 PASS / 1 FAIL). The array case flattens per     */
/*     element (`arg__f__0`, `arg__f__1`, ...), matching flatComponents */
/*     and the path model's const-index folding, so the inliner's      */
/*     per-component binding copies `arg.f[0]` into the right local.   */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `struct S { float f[2]; float g; };
     float useS(S arg) { return arg.f[0] + arg.f[1] + arg.g; }
     void main() {
       S s;
       s.f[0] = 1.0;
       s.f[1] = 2.0;
       s.g = 3.0;
       gl_Position = vec4(useS(s), 0.0, 0.0, 1.0);
     }`,
  );
  check(
    r.ctx.out.position[0] === 6,
    `struct param with array member by value: useS(s) === 6 (got ${r.ctx.out.position[0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* ES 3.00 arrays (hand-built ASTs)                                    */
/*                                                                     */
/* The GLSL ES 3.00 array features below (array-returning functions,   */
/* array ==/!=, whole-array assignment, .length()) are typed by the    */
/* semantics stage in a PARALLEL wave — at this HEAD compileShader     */
/* rejects them, so the ASTs are built by hand (every node carries     */
/* resolvedType), exactly like selftest-codegen-expr.ts. The driver    */
/* installs user functions + globals, emits main, runs it, and asserts */
/* on the resulting ctx state.                                         */
/* ------------------------------------------------------------------ */

const LOC = { line: 1, column: 0 };
const iT = (): GLSLType => ({ kind: 'scalar', base: 'int' });
const fT = (): GLSLType => ({ kind: 'scalar', base: 'float' });
const bT = (): GLSLType => ({ kind: 'scalar', base: 'bool' });
const vec4T = (): GLSLType => ({ kind: 'vector', base: 'float', size: 4 });
const arrT = (elem: GLSLType, size: number): GLSLType => ({ kind: 'array', element: elem, size });

function lit(v: number | boolean, t: GLSLType): LiteralExpr {
  return {
    kind: 'literal', loc: LOC, value: v,
    literalType: (t.kind === 'scalar' ? t.base : 'float') as 'int' | 'uint' | 'float' | 'bool',
    resolvedType: t, constValue: v,
  };
}
function ident(name: string, t: GLSLType): Expr {
  return { kind: 'identifier', loc: LOC, name, resolvedType: t };
}
function call(name: string, args: Expr[], t: GLSLType): Expr {
  return { kind: 'call', loc: LOC, callee: { kind: 'identifier', loc: LOC, name, resolvedType: { kind: 'void' } }, args, resolvedType: t };
}
function arrCtor(name: string, size: number, args: Expr[], t: GLSLType): Expr {
  return {
    kind: 'call', loc: LOC,
    callee: {
      kind: 'index', loc: LOC,
      object: { kind: 'identifier', loc: LOC, name, resolvedType: { kind: 'void' } },
      index: lit(size, iT()), resolvedType: { kind: 'void' },
    },
    args, resolvedType: t,
  };
}
function memCall(obj: Expr, t: GLSLType): Expr {
  return { kind: 'call', loc: LOC, callee: { kind: 'member', loc: LOC, object: obj, name: 'length', resolvedType: { kind: 'void' } }, args: [], resolvedType: t };
}
function idx(obj: Expr, index: Expr, t: GLSLType): Expr {
  return { kind: 'index', loc: LOC, object: obj, index, resolvedType: t };
}
function assignE(target: Expr, value: Expr, t: GLSLType): Expr {
  return { kind: 'assign', loc: LOC, op: '=', target, value, resolvedType: t };
}
function bin(op: string, l: Expr, r: Expr, t: GLSLType): Expr {
  return { kind: 'binary', loc: LOC, op: op as never, left: l, right: r, resolvedType: t };
}
function unary(op: string, operand: Expr, t: GLSLType): Expr {
  return { kind: 'unary', loc: LOC, op: op as never, operand, resolvedType: t };
}
function tern(cond: Expr, a: Expr, b: Expr, t: GLSLType): Expr {
  return { kind: 'ternary', loc: LOC, cond, whenTrue: a, whenFalse: b, resolvedType: t };
}
function ret(value: Expr | null): ReturnStmt {
  return { kind: 'return', loc: LOC, value };
}
function exprStmt(e: Expr): ExprStmt {
  return { kind: 'expr-stmt', loc: LOC, expr: e };
}
function typeSpec(t: GLSLType, storage?: string): TypeSpec {
  return { kind: 'type-spec', loc: LOC, qualifiers: storage ? { storage: storage as never } : {}, base: { kind: 'type-name', loc: LOC, name: 'int' }, resolved: t };
}
function declStmt(name: string, baseT: GLSLType, init: Expr | null, arrayDims: Expr[] = []): DeclStmt {
  return {
    kind: 'decl-stmt', loc: LOC,
    type: typeSpec(baseT),
    declarators: [{ kind: 'var-declarator', loc: LOC, name, arrayDims, init }],
  };
}
function globalVar(name: string, baseT: GLSLType, init: Expr | null): GlobalVarDecl {
  return {
    kind: 'global-var-decl', loc: LOC,
    type: typeSpec(baseT),
    declarators: [{ kind: 'var-declarator', loc: LOC, name, arrayDims: [], init }],
  };
}
function param(name: string, baseT: GLSLType, arrayDims: Expr[], storage?: string): ParamDecl {
  return { kind: 'param-decl', loc: LOC, name, type: typeSpec(baseT, storage), arrayDims };
}
function fnDef(name: string, params: ParamDecl[], retT: GLSLType, retDims: Expr[], body: Stmt[]): FunctionDefinition {
  return {
    kind: 'function-definition', loc: LOC,
    prototype: {
      kind: 'function-prototype', loc: LOC, name,
      returnType: typeSpec(retT),
      returnDims: retDims,
      params,
    },
    body: { kind: 'compound', loc: LOC, body },
  };
}
function mainFn(body: Stmt[]): FunctionDefinition {
  return fnDef('main', [], { kind: 'void' }, [], body);
}
function tu300(decls: TranslationUnit['declarations']): TranslationUnit {
  return { kind: 'translation-unit', loc: LOC, version: 300, declarations: decls };
}

function runTu300(u: TranslationUnit): { body: string; ctx: Record<string, any> } {
  const main = findFn(u, 'main');
  const env = new CodegenEnv('VERTEX', baseLayout(300));
  installUserFunctions(u, env);
  const ginit = installUserGlobals(u, env);
  const stmts = emitStatements(main.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...ginit, ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    out: { position: [0, 0, 0, 0], pointSize: 0 },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
  };
  fn(ctx, R);
  return { body, ctx };
}

/* 26. array-returning function: result assigned to a local, indexed       */
/*     (`(f())[0]` directly) — values must be exact.                       */
{
  const u = tu300([
    fnDef('f', [], arrT(iT(), 2), [lit(2, iT())], [
      ret(arrCtor('int', 2, [lit(1, iT()), lit(2, iT())], arrT(iT(), 2))),
    ]),
    mainFn([
      declStmt('a', iT(), null, [lit(2, iT())]),
      declStmt('x', iT(), null, []),
      exprStmt(assignE(ident('a', arrT(iT(), 2)), call('f', [], arrT(iT(), 2)), arrT(iT(), 2))),
      exprStmt(assignE(ident('x', iT()), idx(call('f', [], arrT(iT(), 2)), lit(0, iT()), iT()), iT())),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [bin('+', bin('+', idx(ident('a', arrT(iT(), 2)), lit(0, iT()), iT()), idx(ident('a', arrT(iT(), 2)), lit(1, iT()), iT()), fT()), ident('x', iT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 4, `array return: a=f(); x=(f())[0]; a[0]+a[1]+x === 4 (got ${r.ctx.out.position[0]})`);
}

/* 27. side effects exactly once: `plus();` and `createArray() ==          */
/*     int[2](1,1)` — the callee's IIFE must run ONCE, not once per        */
/*     returned component (shared-pre dedup).                              */
{
  const u = tu300([
    globalVar('g', iT(), lit(0, iT())),
    fnDef('plus', [], arrT(iT(), 2), [lit(2, iT())], [
      exprStmt(unary('++', ident('g', iT()), iT())),
      ret(arrCtor('int', 2, [ident('g', iT()), ident('g', iT())], arrT(iT(), 2))),
    ]),
    fnDef('createArray', [], arrT(iT(), 2), [lit(2, iT())], [
      exprStmt(unary('++', ident('g', iT()), iT())),
      ret(arrCtor('int', 2, [lit(1, iT()), lit(1, iT())], arrT(iT(), 2))),
    ]),
    mainFn([
      declStmt('eq', bT(), null, []),
      exprStmt(call('plus', [], arrT(iT(), 2))),
      exprStmt(assignE(ident('eq', bT()), bin('==', call('createArray', [], arrT(iT(), 2)), arrCtor('int', 2, [lit(1, iT()), lit(1, iT())], arrT(iT(), 2)), bT()), bT())),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [tern(bin('&&', bin('==', ident('g', iT()), lit(2, iT()), bT()), ident('eq', bT()), bT()), lit(1, fT()), lit(0, fT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 1, `array call side effects once: plus(); createArray()==int[2](1,1); g===2 (got ${r.ctx.out.position[0]})`);
}

/* 28. .length(): local, function call result, constructor result.         */
{
  const u = tu300([
    fnDef('f', [], arrT(iT(), 2), [lit(2, iT())], [
      ret(arrCtor('int', 2, [lit(7, iT()), lit(7, iT())], arrT(iT(), 2))),
    ]),
    mainFn([
      declStmt('a', iT(), null, [lit(3, iT())]),
      declStmt('n', iT(), null, []),
      declStmt('ok', bT(), null, []),
      exprStmt(assignE(ident('ok', bT()), bin('==', memCall(ident('a', arrT(iT(), 3)), iT()), lit(3, iT()), bT()), bT())),
      exprStmt(assignE(ident('n', iT()), memCall(call('f', [], arrT(iT(), 2)), iT()), iT())),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [tern(bin('&&', bin('&&', ident('ok', bT()), bin('==', ident('n', iT()), lit(2, iT()), bT()), bT()), bin('==', memCall(arrCtor('int', 1, [lit(0, iT())], arrT(iT(), 1)), iT()), lit(1, iT()), bT()), bT()), lit(1, fT()), lit(0, fT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 1, `array .length(): a.length()==3 && (f()).length()==2 && (int[1](0)).length()==1 (got ${r.ctx.out.position[0]})`);
}

/* 29. array ==/!= on locals: (a==b) && (a!=c) for a==b, c differing.      */
{
  const u = tu300([
    mainFn([
      declStmt('a', iT(), null, [lit(2, iT())]),
      declStmt('b', iT(), null, [lit(2, iT())]),
      declStmt('c', iT(), null, [lit(2, iT())]),
      exprStmt(assignE(idx(ident('a', arrT(iT(), 2)), lit(0, iT()), iT()), lit(1, iT()), iT())),
      exprStmt(assignE(idx(ident('a', arrT(iT(), 2)), lit(1, iT()), iT()), lit(2, iT()), iT())),
      exprStmt(assignE(idx(ident('b', arrT(iT(), 2)), lit(0, iT()), iT()), lit(1, iT()), iT())),
      exprStmt(assignE(idx(ident('b', arrT(iT(), 2)), lit(1, iT()), iT()), lit(2, iT()), iT())),
      exprStmt(assignE(idx(ident('c', arrT(iT(), 2)), lit(0, iT()), iT()), lit(1, iT()), iT())),
      exprStmt(assignE(idx(ident('c', arrT(iT(), 2)), lit(1, iT()), iT()), lit(3, iT()), iT())),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [tern(bin('&&', bin('==', ident('a', arrT(iT(), 2)), ident('b', arrT(iT(), 2)), bT()), bin('!=', ident('a', arrT(iT(), 2)), ident('c', arrT(iT(), 2)), bT()), bT()), lit(1, fT()), lit(0, fT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 1, `array ==/!=: (a==b) && (a!=c) for a==b, c differs (got ${r.ctx.out.position[0]})`);
}

/* 30. short-circuit: array == must NOT run when the && left side is       */
/*     false, nor in the un-taken ternary arm; must run exactly once in    */
/*     the taken arm (calls counter).                                      */
{
  const u = tu300([
    globalVar('calls', iT(), lit(0, iT())),
    fnDef('minus', [], bT(), [], [ret(lit(false, bT()))]),
    fnDef('plus', [], arrT(iT(), 2), [lit(2, iT())], [
      exprStmt(unary('++', ident('calls', iT()), iT())),
      ret(arrCtor('int', 2, [lit(1, iT()), lit(1, iT())], arrT(iT(), 2))),
    ]),
    mainFn([
      declStmt('a', iT(), null, [lit(2, iT())]),
      declStmt('r1', bT(), null, []),
      declStmt('r2', bT(), null, []),
      declStmt('r3', bT(), null, []),
      exprStmt(assignE(idx(ident('a', arrT(iT(), 2)), lit(0, iT()), iT()), lit(1, iT()), iT())),
      exprStmt(assignE(idx(ident('a', arrT(iT(), 2)), lit(1, iT()), iT()), lit(1, iT()), iT())),
      exprStmt(assignE(ident('r1', bT()), bin('&&', call('minus', [], bT()), bin('==', ident('a', arrT(iT(), 2)), call('plus', [], arrT(iT(), 2)), bT()), bT()), bT())),
      exprStmt(assignE(ident('r2', bT()), tern(lit(true, bT()), lit(true, bT()), bin('==', ident('a', arrT(iT(), 2)), call('plus', [], arrT(iT(), 2)), bT()), bT()), bT())),
      exprStmt(assignE(ident('r3', bT()), tern(lit(false, bT()), lit(true, bT()), bin('==', ident('a', arrT(iT(), 2)), call('plus', [], arrT(iT(), 2)), bT()), bT()), bT())),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [tern(bin('&&', bin('&&', bin('&&', bin('==', ident('calls', iT()), lit(1, iT()), bT()), ident('r3', bT()), bT()), unary('!', ident('r1', bT()), bT()), bT()), ident('r2', bT()), bT()), lit(1, fT()), lit(0, fT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 1, `array == short-circuit: minus()&&(a==plus()) skips plus; ternary arms lazy; calls===1 (got ${r.ctx.out.position[0]})`);
}

/* 31. decl-init from call `int a[2] = bar(1);`, array ARG passing         */
/*     isSuccess(dup(9)), and `return foo();` (wrap returns bar(7)).       */
{
  const u = tu300([
    fnDef('bar', [param('x', iT(), [])], arrT(iT(), 2), [lit(2, iT())], [
      ret(arrCtor('int', 2, [ident('x', iT()), bin('+', ident('x', iT()), lit(1, iT()), iT())], arrT(iT(), 2))),
    ]),
    fnDef('dup', [param('x', iT(), [])], arrT(iT(), 2), [lit(2, iT())], [
      ret(arrCtor('int', 2, [ident('x', iT()), ident('x', iT())], arrT(iT(), 2))),
    ]),
    fnDef('isSuccess', [param('v', iT(), [lit(2, iT())])], bT(), [], [
      ret(bin('==', idx(ident('v', arrT(iT(), 2)), lit(0, iT()), iT()), idx(ident('v', arrT(iT(), 2)), lit(1, iT()), iT()), bT())),
    ]),
    fnDef('wrap', [], arrT(iT(), 2), [lit(2, iT())], [
      ret(call('bar', [lit(7, iT())], arrT(iT(), 2))),
    ]),
    mainFn([
      declStmt('a', iT(), call('bar', [lit(1, iT())], arrT(iT(), 2)), [lit(2, iT())]),
      declStmt('w', iT(), null, [lit(2, iT())]),
      declStmt('ok', bT(), null, []),
      exprStmt(assignE(ident('ok', bT()), call('isSuccess', [call('dup', [lit(9, iT())], arrT(iT(), 2))], bT()), bT())),
      exprStmt(assignE(ident('w', iT()), call('wrap', [], arrT(iT(), 2)), arrT(iT(), 2))),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [tern(bin('&&', bin('&&', bin('&&', bin('==', idx(ident('a', arrT(iT(), 2)), lit(0, iT()), iT()), lit(1, iT()), bT()), bin('==', idx(ident('a', arrT(iT(), 2)), lit(1, iT()), iT()), lit(2, iT()), bT()), bT()), bin('==', idx(ident('w', arrT(iT(), 2)), lit(0, iT()), iT()), lit(7, iT()), bT()), bT()), ident('ok', bT()), bT()), lit(1, fT()), lit(0, fT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 1, `array decl-init/args/return-of-call: a=bar(1); w=wrap()=bar(7); isSuccess(dup(9)) (got ${r.ctx.out.position[0]})`);
}

/* 32. (a = b)[0] and (a = b).length(): whole-array assignment is an       */
/*     expression whose value is the array (indexed / .length() on it).    */
{
  const u = tu300([
    mainFn([
      declStmt('a', iT(), null, [lit(2, iT())]),
      declStmt('b', iT(), null, [lit(2, iT())]),
      declStmt('x', iT(), null, []),
      declStmt('n', iT(), null, []),
      exprStmt(assignE(idx(ident('b', arrT(iT(), 2)), lit(0, iT()), iT()), lit(5, iT()), iT())),
      exprStmt(assignE(idx(ident('b', arrT(iT(), 2)), lit(1, iT()), iT()), lit(6, iT()), iT())),
      exprStmt(assignE(ident('x', iT()), idx(assignE(ident('a', arrT(iT(), 2)), ident('b', arrT(iT(), 2)), arrT(iT(), 2)), lit(0, iT()), iT()), iT())),
      exprStmt(assignE(ident('n', iT()), memCall(assignE(ident('a', arrT(iT(), 2)), ident('b', arrT(iT(), 2)), arrT(iT(), 2)), iT()), iT())),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [tern(bin('&&', bin('&&', bin('&&', bin('==', ident('x', iT()), lit(5, iT()), bT()), bin('==', idx(ident('a', arrT(iT(), 2)), lit(1, iT()), iT()), lit(6, iT()), bT()), bT()), bin('==', ident('n', iT()), lit(2, iT()), bT()), bT()), bin('==', idx(ident('a', arrT(iT(), 2)), lit(0, iT()), iT()), lit(5, iT()), bT()), bT()), lit(1, fT()), lit(0, fT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 1, `(a=b)[0]===5 && a[1]===6 && (a=b).length()===2 && a[0]===5 (got ${r.ctx.out.position[0]})`);
}

/* 33. sequence ((++j), (a == func(j))): the compare must see j AFTER      */
/*     the increment (left-to-right evaluation).                           */
{
  const u = tu300([
    fnDef('func', [param('v', iT(), [])], arrT(iT(), 2), [lit(2, iT())], [
      ret(arrCtor('int', 2, [ident('v', iT()), ident('v', iT())], arrT(iT(), 2))),
    ]),
    mainFn([
      declStmt('a', iT(), null, [lit(2, iT())]),
      declStmt('j', iT(), lit(4, iT()), []),
      declStmt('ok', bT(), null, []),
      exprStmt(assignE(idx(ident('a', arrT(iT(), 2)), lit(0, iT()), iT()), lit(5, iT()), iT())),
      exprStmt(assignE(idx(ident('a', arrT(iT(), 2)), lit(1, iT()), iT()), lit(5, iT()), iT())),
      exprStmt(assignE(ident('ok', bT()), {
        kind: 'comma', loc: LOC,
        exprs: [unary('++', ident('j', iT()), iT()), bin('==', ident('a', arrT(iT(), 2)), call('func', [ident('j', iT())], arrT(iT(), 2)), bT())],
        resolvedType: bT(),
      }, bT())),
      exprStmt(assignE(ident('gl_Position', vec4T()), call('vec4', [tern(ident('ok', bT()), lit(1, fT()), lit(0, fT()), fT())], vec4T()), vec4T())),
    ]),
  ]);
  const r = runTu300(u);
  check(r.ctx.out.position[0] === 1, `sequence ((++j), (a==func(j))): func sees j===5 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-fn selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
