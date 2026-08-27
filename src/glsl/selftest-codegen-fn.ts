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
import type { FunctionDefinition, TranslationUnit } from './ast.js';
import { compileShader } from './compiler.js';
import { CodegenEnv } from './codegen/env.js';
import { installUserFunctions } from './codegen/functions.js';
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
      derivatives: false,
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
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-fn selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
