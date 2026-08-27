/**
 * selftest-codegen-stmt.ts — direct unit checks for codegen/statements.ts
 * (the GLSL→JS STATEMENT lowering layer).
 *
 * Run: npx tsx src/glsl/selftest-codegen-stmt.ts
 *
 * Modeled on selftest-codegen-expr.ts: real shaders are compiled via
 * compileShader (preprocess → lexer → parse → semantics — all implemented),
 * the annotated `main` body is lowered with emitStatements into a JS function
 * body (`new Function('ctx','R', body)`), and the resulting ctx state is
 * asserted. VERTEX mains write gl_Position (env maps it straight to
 * ctx.out.position — no layout entries needed); the FRAGMENT discard test
 * uses outputLocations gl_FragColor → 0 and a ctx with `discarded`.
 *
 * Prints "OK" and exits 0 on success.
 */
import type { ExternalDecl, FunctionDefinition, TranslationUnit } from './ast.js';
import { compileShader } from './compiler.js';
import { CodegenEnv } from './codegen/env.js';
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
/* Driver: compile a shader, emit its `main` body, run it              */
/* ------------------------------------------------------------------ */

function findFn(tu: TranslationUnit, name: string): FunctionDefinition {
  for (const d of tu.declarations) {
    if (d.kind === 'function-definition' && d.prototype.name === name) return d;
  }
  throw new Error(`no function '${name}' in shader`);
}

/** Register a function's parameters as locals (the inliner in C3b maps them
 *  to caller-side temps; here the emitted body reads the bare JS names). */
function declareParams(fn: FunctionDefinition, env: CodegenEnv): void {
  for (const p of fn.prototype.params) {
    let t = p.type.resolved!;
    for (let i = p.arrayDims.length - 1; i >= 0; i--) {
      const cv = p.arrayDims[i]?.constValue;
      const size = typeof cv === 'number' && Number.isInteger(cv) && cv > 0 ? cv : 1;
      t = { kind: 'array', element: t, size };
    }
    env.declareLocal(p.name, t);
  }
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

/** Compile + emit + run a VERTEX main. */
function runVertex(src: string, opts?: { version?: 100 | 300; ctxExtra?: Record<string, unknown> }): RunResult {
  const version = opts?.version ?? 100;
  const r = compileShader(src, { type: 'VERTEX', version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('VERTEX', baseLayout(version));
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    out: { position: [0, 0, 0, 0], pointSize: 0 },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
    ...(opts?.ctxExtra ?? {}),
  };
  fn(ctx, R);
  return { body, ctx };
}

/** Compile + emit + run a FRAGMENT main (version 100, gl_FragColor output). */
function runFragment(src: string): RunResult {
  const r = compileShader(src, { type: 'FRAGMENT', version: 100 });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('FRAGMENT', baseLayout(100));
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: { color: [[0, 0, 0, 0]] },
    fragCoord: [0, 0, 0, 0],
    frontFacing: true,
    pointCoord: [0, 0],
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
  };
  fn(ctx, R);
  return { body, ctx };
}

/* ------------------------------------------------------------------ */
/* 1. if / else / else-if chains                                       */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void main() { float x = 2.0; if (x > 1.0) { gl_Position.x = 5.0; } else { gl_Position.x = 7.0; } }`,
  );
  check(r.ctx.out.position[0] === 5, `if/else then-branch: x === 5 (got ${r.ctx.out.position[0]})`);
}
{
  const r = runVertex(
    `void main() { float x = 0.5; if (x > 1.0) { gl_Position.x = 5.0; } else { gl_Position.x = 7.0; } }`,
  );
  check(r.ctx.out.position[0] === 7, `if/else else-branch: x === 7 (got ${r.ctx.out.position[0]})`);
}
{
  const r = runVertex(
    `void main() { int x = 2; if (x == 1) { gl_Position.x = 1.0; } else if (x == 2) { gl_Position.x = 2.0; } else { gl_Position.x = 3.0; } }`,
  );
  check(r.ctx.out.position[0] === 2, `if/else-if chain picks the middle branch (got ${r.ctx.out.position[0]})`);
}
{
  // Conditional on a COMPUTED local, no braces on the then-branch (single stmt).
  const r = runVertex(
    `void main() { float a = 3.0; float b = 4.0; if (a * b > 10.0) gl_Position.x = a * b; else gl_Position.x = 0.0; }`,
  );
  check(r.ctx.out.position[0] === 12, `unbraced if/else with computed cond: x === 12 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 2. for loop: declaration init, accumulation, break, continue        */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void main() { float s = 0.0; for (int i = 0; i < 4; i++) { s = s +float(i); } gl_Position.x = s; }`,
  );
  check(r.ctx.out.position[0] === 6, `for accumulation 0+1+2+3 === 6 (got ${r.ctx.out.position[0]})`);
}
{
  const r = runVertex(
    `void main() { float s = 0.0; for (int i = 0; i < 4; i++) { if (i == 2) { break; } s = s +float(i); } gl_Position.x = s; }`,
  );
  check(r.ctx.out.position[0] === 1, `for break at i==2: s === 1 (got ${r.ctx.out.position[0]})`);
}
{
  const r = runVertex(
    `void main() { float s = 0.0; for (int i = 0; i < 4; i++) { if (i % 2 == 1) { continue; } s = s +float(i); } gl_Position.x = s; }`,
  );
  check(r.ctx.out.position[0] === 2, `for continue skips odds: s === 2 (got ${r.ctx.out.position[0]})`);
}
{
  // Two SIBLING for-loops reusing the same `i` name (legal GLSL — each
  // for-init lives in its own scope; codegen reuses the same JS var and
  // re-initializes it, which is correct since the scopes don't overlap).
  const r = runVertex(
    `void main() {
      float s = 0.0;
      for (int i = 0; i < 3; i++) { s = s + 1.0; }
      for (int i = 0; i < 2; i++) { s = s + 10.0; }
      gl_Position.x = s;
    }`,
  );
  check(r.ctx.out.position[0] === 23, `sibling same-name for loops: s === 23 (got ${r.ctx.out.position[0]})`);
}
{
  // Infinite-style `for (;;)` with an internal break.
  const r = runVertex(
    `void main() { float s = 0.0; for (;;) { s = s +1.0; if (s >= 3.0) { break; } } gl_Position.x = s; }`,
  );
  check(r.ctx.out.position[0] === 3, `for(;;) with break: s === 3 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 3. while + do-while                                                 */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void main() { float s = 0.0; int i = 0; while (i < 4) { s = s +float(i); i++; } gl_Position.x = s; }`,
  );
  check(r.ctx.out.position[0] === 6, `while loop: s === 6 (got ${r.ctx.out.position[0]})`);
}
{
  const r = runVertex(
    `void main() { float s = 0.0; int i = 0; do { s = s +float(i); i++; } while (i < 4); gl_Position.x = s; }`,
  );
  check(r.ctx.out.position[0] === 6, `do-while loop: s === 6 (got ${r.ctx.out.position[0]})`);
}
{
  // do-while runs its body at least once even when the cond is false.
  const r = runVertex(
    `void main() { float s = 0.0; int i = 0; do { s = s +1.0; } while (i > 5); gl_Position.x = s; }`,
  );
  check(r.ctx.out.position[0] === 1, `do-while runs once on false cond: s === 1 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 4. nested loops                                                     */
/* ------------------------------------------------------------------ */

{
  // sum_{i=0..2} sum_{j=0..2} (i+j) = 3 + 6 + 9 = 18
  const r = runVertex(
    `void main() {
      float s = 0.0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) { s = s +float(i + j); }
      }
      gl_Position.x = s;
    }`,
  );
  check(r.ctx.out.position[0] === 18, `nested 2D accumulation: s === 18 (got ${r.ctx.out.position[0]})`);
}
{
  // continue in the INNER loop must not skip the outer iterations.
  const r = runVertex(
    `void main() {
      float s = 0.0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          if (j == 1) { continue; }
          s = s +1.0;
        }
      }
      gl_Position.x = s;
    }`,
  );
  check(r.ctx.out.position[0] === 6, `inner continue: 3*2 === 6 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 5. switch: cases, default, fallthrough (ES 3.00)                    */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `#version 300 es
void main() {
  int x = 1;
  float s = 0.0;
  switch (x) {
    case 0: s = 10.0; break;
    case 1: s = 1.0;
    case 2: s = s + 2.0; break;
    default: s = 99.0;
  }
  gl_Position.x = s;
}`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 3, `switch case 1 falls into case 2: s === 3 (got ${r.ctx.out.position[0]})`);
}
{
  const r = runVertex(
    `#version 300 es
void main() {
  int x = 5;
  float s = 0.0;
  switch (x) {
    case 0: s = 10.0; break;
    case 1: s = 1.0; break;
    case 2: s = 2.0; break;
    default: s = 42.0;
  }
  gl_Position.x = s;
}`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 42, `switch default: s === 42 (got ${r.ctx.out.position[0]})`);
}
{
  // break exits the switch, NOT an enclosing loop.
  const r = runVertex(
    `#version 300 es
void main() {
  float s = 0.0;
  for (int i = 0; i < 2; i++) {
    switch (i) {
      case 0: s = s + 1.0; break;
      case 1: s = s + 100.0; break;
    }
  }
  gl_Position.x = s;
}`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 101, `switch break inside loop: s === 101 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 6. local declarations + constructors                                */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void main() {
      float a = 1.5;
      vec2 b = vec2(2.0, 3.0);
      vec4 c = vec4(a, b, 4.0);
      gl_Position = c;
    }`,
  );
  const p = r.ctx.out.position;
  check(
    p[0] === 1.5 && p[1] === 2 && p[2] === 3 && p[3] === 4,
    `vec4(a, vec2, 4.0) === [1.5, 2, 3, 4] (got [${p.join(', ')}])`,
  );
}
{
  // Multiple declarators in one statement + explicit int→float ctor init.
  const r = runVertex(
    `void main() { int i = 2, j = 3; float x = float(i); gl_Position.x = x + float(j); }`,
  );
  check(r.ctx.out.position[0] === 5, `multi-declarator decl + int→float init: x === 5 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 7. local arrays: static + dynamic indexing (scratch + dyn temps)    */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void main() {
      float a[4];
      a[0] = 1.0;
      a[1] = 2.0;
      a[2] = 3.0;
      a[3] = 4.0;
      float s = 0.0;
      for (int i = 0; i < 4; i++) { s = s +a[i]; }
      gl_Position.x = s;
    }`,
  );
  check(r.ctx.out.position[0] === 10, `dynamic-index array read: s === 10 (got ${r.ctx.out.position[0]})`);
}
{
  // Dynamic-index WRITES through the loop, then a static read-back.
  const r = runVertex(
    `void main() {
      float a[4];
      for (int i = 0; i < 4; i++) { a[i] = float(i) * 2.0; }
      gl_Position.x = a[3];
    }`,
  );
  check(r.ctx.out.position[0] === 6, `dynamic-index array write + static read: a[3] === 6 (got ${r.ctx.out.position[0]})`);
}
{
  // Array initializer constructor (float[4](...)) — ES 3.00 only.
  const r = runVertex(
    `#version 300 es
void main() {
  float a[4] = float[4](1.0, 2.0, 3.0, 4.0);
  float s = 0.0;
  for (int i = 0; i < 4; i++) { s = s +a[i]; }
  gl_Position.x = s;
}`,
    { version: 300 },
  );
  check(r.ctx.out.position[0] === 10, `array-ctor initializer: s === 10 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 8. discard (fragment) — terminates the whole shader                  */
/* ------------------------------------------------------------------ */

{
  const r = runFragment(
    `precision mediump float;
void main() {
  gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
  for (int i = 0; i < 2; i++) {
    if (i == 1) { discard; }
  }
  gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
}`,
  );
  check(r.ctx.discarded === true, `discard sets ctx.discarded (got ${r.ctx.discarded})`);
  check(
    r.ctx.out.color[0][0] === 1 && r.ctx.out.color[0][1] === 0,
    `statements after the discard did NOT run (color stays [1,0,0,1], got [${r.ctx.out.color[0].join(', ')}])`,
  );
  check(
    r.body.includes('ctx.discarded = true; return;'),
    `discard compiles to the exact 'ctx.discarded = true; return;' line`,
  );
}
{
  // Control path sanity: no discard → the final write lands.
  const r = runFragment(
    `precision mediump float;
void main() { gl_FragColor = vec4(0.25, 0.5, 0.75, 1.0); }`,
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 0.25 && c[1] === 0.5 && c[2] === 0.75 && c[3] === 1, `fragment color write lands (got [${c.join(', ')}])`);
}

/* ------------------------------------------------------------------ */
/* 9. empty statements + expression statements                         */
/* ------------------------------------------------------------------ */

{
  const r = runVertex(
    `void main() {
      float x = 0.0;
      ;
      x = 1.0;
      x += 1.0;
      x = x + 2.0;
      ;
      gl_Position.x = x;
    }`,
  );
  check(r.ctx.out.position[0] === 4, `empty stmts + assign/compound-assign exprs: x === 4 (got ${r.ctx.out.position[0]})`);
}
{
  // Expression-only for-init with a COMPOUND-assign update.
  const r = runVertex(
    `void main() {
      float x = 9.0;
      float s = 0.0;
      for (x = 0.0; x < 3.0; x += 1.0) { s = s + x; }
      gl_Position.x = s;
    }`,
  );
  check(r.ctx.out.position[0] === 3, `expr-init for with compound update: s === 3 (got ${r.ctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 10. return lowering with an inliner-style FnEmitContext             */
/* ------------------------------------------------------------------ */

{
  const r = compileShader(
    `float twice(float x) { return x * 2.0; } void main() { gl_Position.x = 1.0; }`,
    { type: 'VERTEX', version: 100 },
  );
  if (r.ok) {
    const fn = findFn(r.shader.ast, 'twice');
    const env = new CodegenEnv('VERTEX', baseLayout(100));
    declareParams(fn, env);
    const lines = emitStatements(fn.body.body, env, { retTemps: ['r0'], epilogueLabel: 'EP_9', retType: { kind: 'scalar', base: 'float' } });
    const body = lines.join('\n');
    check(
      body.includes('r0 = ') && body.includes('break EP_9;') && !body.includes('return;'),
      `inlined return → 'r0 = ...; break EP_9;' (got:\n${body})`,
    );
  } else {
    check(false, `twice shader compiles (${JSON.stringify(r.errors)})`);
  }
}
{
  // Void return: only the epilogue break.
  const r = compileShader(
    `void nop(float x) { if (x > 0.0) { return; } gl_Position.x = 1.0; }
     void main() { nop(1.0); gl_Position.x = 2.0; }`,
    { type: 'VERTEX', version: 100 },
  );
  if (r.ok) {
    const fn = findFn(r.shader.ast, 'nop');
    const env = new CodegenEnv('VERTEX', baseLayout(100));
    declareParams(fn, env);
    const lines = emitStatements(fn.body.body, env, { retTemps: [], epilogueLabel: 'EP_2' });
    const body = lines.join('\n');
    check(
      body.includes('break EP_2;') && !body.includes('r0') && !body.includes('return;'),
      `void inlined return → bare 'break EP_2;' (got:\n${body})`,
    );
  } else {
    check(false, `nop shader compiles (${JSON.stringify(r.errors)})`);
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-stmt selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');