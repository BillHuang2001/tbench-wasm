/**
 * selftest-predrop.ts — Value.pre-drop regression checks.
 *
 * `Value.pre` = PURE materialization expressions (texture-sample chains, temp
 * assignments from materialize()) that must run once BEFORE `v`. Any consumer
 * of `.v` that drops `.pre` reads an UNASSIGNED temp (undefined/NaN/falsy).
 * This file pins the fixed consumers: ternary (non-dual), convertValue-based
 * re-attach paths (bitwise binary, compound assign, array ctor, struct ctor,
 * decl-init), the non-dual builtins distance/normalize/faceforward/
 * refract/outerProduct/determinant/uaddCarry/usubBorrow/umulExtended/
 * imulExtended/modf (+ dual-mode modf), and walkObject member/index-of-call-
 * result materialization (materializeSharedPre identity dedup). Every shader
 * calls a USER FUNCTION so the operand carries an inliner IIFE pre, then
 * materializes.
 *
 * Each case compiles a real shader (compileShader → annotated AST), emits
 * `main` (installUserFunctions + emitStatements), runs the generated JS via
 * `new Function('ctx','R', body)` against a hand-built ctx, and asserts the
 * runtime value.
 *
 * Run: npx tsx src/glsl/codegen/selftest-predrop.ts
 *
 * Prints "selftest-predrop: N checks" and exits 0 only when all pass.
 */
import { compileShader } from '../compiler.js';
import { CodegenEnv } from './env.js';
import { emitStatements } from './statements.js';
import { installUserFunctions, installUserGlobals } from './functions.js';
import { R } from './runtime.js';
import type { CodegenLayout } from './index.js';
import type { TranslationUnit, FunctionDefinition } from '../ast.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function findFn(tu: TranslationUnit, name: string): FunctionDefinition {
  for (const d of tu.declarations) {
    if (d.kind === 'function-definition' && d.prototype.name === name) return d;
  }
  throw new Error(`no function '${name}'`);
}

function baseLayout(version: 100 | 300, derivatives = false): CodegenLayout {
  return {
    version,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map([['a_pos', 0]]),
    outputLocations: new Map([['gl_FragColor', 0], ['color', 0]]),
    uses: {
      pointSize: false, fragCoord: false, frontFacing: false, pointCoord: false,
      fragDepth: false, vertexId: false, instanceId: false, drawId: false, derivatives,
      depthRange: false,
    },
  };
}

/** Compile + emit `main` + run; returns the body and the executed ctx. */
function runMain(
  src: string,
  stage: 'VERTEX' | 'FRAGMENT',
  version: 100 | 300,
  opts?: { derivatives?: boolean; structNames?: string[] },
): { body: string; ctx: Record<string, any> } {
  const r = compileShader(src, { type: stage, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv(stage, baseLayout(version, opts?.derivatives));
  env.dual = !!opts?.derivatives; // mirror fragment.ts/vertex.ts stage assembly
  // File-scope globals (scratch-backed) first — their init lines run before
  // main's statements (mirrors vertex.ts/fragment.ts stage assembly).
  const globalInit = installUserGlobals(r.shader.ast, env);
  installUserFunctions(r.shader.ast, env);
  if (opts?.structNames) for (const s of opts.structNames) env.structNames.add(s);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...globalInit, ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: stage === 'VERTEX' ? { position: [0, 0, 0, 0], pointSize: 0 } : { color: [[0, 0, 0, 0]] },
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
  };
  fn(ctx, R);
  return { body, ctx };
}

/* ------------------------------------------------------------------ */
/* Non-dual ternary (cond + arms all pre-carrying)                     */
/* ------------------------------------------------------------------ */

// (a) ternary with pre-carrying condition (user fn bool, IIFE pre).
{
  const { ctx } = runMain(
    `precision mediump float;
bool pick() { return true; }
void main() {
  gl_FragColor = pick() ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][1] === 1 && ctx.out.color[0][0] === 0, `ternary cond userfn (got [${ctx.out.color[0].join(',')}])`);
}

// (a2) ternary with pre-carrying ARM (user fn in the taken arm).
{
  const { ctx } = runMain(
    `precision mediump float;
float f() { return 2.0; }
void main() {
  bool c = true;
  float r = c ? f() : 9.0;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === 2, `ternary arm userfn (got [${ctx.out.color[0].join(',')}])`);
}

// (a3) ternary with pre-carrying cond AND arms, vector type.
{
  const { ctx } = runMain(
    `precision mediump float;
bool pick() { return true; }
float f() { return 2.0; }
void main() {
  gl_FragColor = pick() ? vec4(f(), 1.0, 0.0, 1.0) : vec4(9.0, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(
    ctx.out.color[0][0] === 2 && ctx.out.color[0][1] === 1,
    `ternary cond+arm userfns (got [${ctx.out.color[0].join(',')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* && / || with pre-carrying left operand (feeds a ternary cond)       */
/* ------------------------------------------------------------------ */

// (b) && — left operand is a user fn call (IIFE pre).
{
  const { ctx } = runMain(
    `precision mediump float;
bool pick() { return true; }
void main() {
  float r = (pick() && true) ? 1.0 : 0.0;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === 1, `&& left userfn (got [${ctx.out.color[0].join(',')}])`);
}

// (b2) || — left operand pre-carrying (true fn; rhs false → must yield 1).
{
  const { ctx } = runMain(
    `precision mediump float;
bool pick() { return true; }
void main() {
  float r = (pick() || false) ? 1.0 : 0.0;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === 1, `|| left userfn (got [${ctx.out.color[0].join(',')}])`);
}

/* ------------------------------------------------------------------ */
/* Relational / equality with pre-carrying LEFT operand                */
/* ------------------------------------------------------------------ */

// (c) user fn (explicit float ctor) in a relational feeding a ternary.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float r = (float(f()) < 3.5) ? 1.0 : 0.0;
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(ctx.out.color[0][0] === 1, `relational userfn vs float (got [${ctx.out.color[0].join(',')}])`);
}

// (d) user fn (explicit float ctor) in an equality.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float r = (float(f()) == 3.0) ? 1.0 : 0.0;
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(ctx.out.color[0][0] === 1, `equality userfn vs float (got [${ctx.out.color[0].join(',')}])`);
}

/* ------------------------------------------------------------------ */
/* Decl-init / compound assign with a user-fn conversion               */
/* ------------------------------------------------------------------ */

// (e) float x = float(f());  (user fn in a decl-init ctor; emitDeclStmt path).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float x = float(f());
  color = vec4(x, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(ctx.out.color[0][0] === 3, `decl-init userfn ctor to float (got [${ctx.out.color[0].join(',')}])`);
}

// (f) compound assign with user-fn RHS in EXPRESSION position
//     (emitAssign compound path; float += float(f())).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float x = 1.0;
  float y = (x += float(f())) + 0.0;
  color = vec4(y, x, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(
    ctx.out.color[0][0] === 4 && ctx.out.color[0][1] === 4,
    `compound assign userfn rhs (got [${ctx.out.color[0].join(',')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* Bitwise binary with a pre-carrying left operand                     */
/* ------------------------------------------------------------------ */

// (g) user fn (explicit uint ctor) & uint literal.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 6; }
out vec4 color;
void main() {
  uint r = uint(f()) & 3u;
  color = vec4(float(r), 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(ctx.out.color[0][0] === 2, `bitwise userfn uint & 3u (got [${ctx.out.color[0].join(',')}])`);
}

/* ------------------------------------------------------------------ */
/* Struct / array constructors with pre-carrying member args           */
/* ------------------------------------------------------------------ */

// (h) struct ctor S(f()) — user fn (float) → float member.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
struct S { float x; };
float f() { return 3.0; }
out vec4 color;
void main() {
  S s = S(f());
  color = vec4(s.x, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { structNames: ['S'] },
  );
  check(ctx.out.color[0][0] === 3, `struct ctor userfn member (got [${ctx.out.color[0].join(',')}])`);
}

// (i) array ctor float[1](f()) — user fn (float) → float element.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
float f() { return 3.0; }
out vec4 color;
void main() {
  float arr[1] = float[1](f());
  color = vec4(arr[0], 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(ctx.out.color[0][0] === 3, `array ctor userfn element (got [${ctx.out.color[0].join(',')}])`);
}

/* ------------------------------------------------------------------ */
/* Non-dual builtins with pre-carrying args                            */
/* ------------------------------------------------------------------ */

// (j) modf(f(), ip).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
float f() { return 3.7; }
out vec4 color;
void main() {
  float ip = 0.0;
  float fr = modf(f(), ip);
  color = vec4(fr, ip, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(
    Math.abs(ctx.out.color[0][0] - 0.7) < 1e-6 && ctx.out.color[0][1] === 3,
    `modf pre arg (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (j2) modf(f(), ip) in DUAL mode (derivatives on — modf dual twin).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
float f() { return 3.7; }
out vec4 color;
void main() {
  float ip = 0.0;
  float fr = modf(f(), ip);
  color = vec4(fr, ip, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { derivatives: true },
  );
  check(
    Math.abs(ctx.out.color[0][0] - 0.7) < 1e-6 && ctx.out.color[0][1] === 3,
    `modf pre arg dual (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (k) normalize(vec2(f(), 4.0)).
{
  const { ctx } = runMain(
    `precision mediump float;
float f() { return 3.0; }
void main() {
  vec2 nv = normalize(vec2(f(), 4.0));
  float n = nv.x;
  gl_FragColor = vec4(n, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(Math.abs(ctx.out.color[0][0] - 0.6) < 1e-6, `normalize pre arg (got [${ctx.out.color[0].join(',')}])`);
}

// (l) distance(vec2(f(), 0.0), vec2(0.0, 4.0)).
{
  const { ctx } = runMain(
    `precision mediump float;
float f() { return 3.0; }
void main() {
  float d = distance(vec2(f(), 0.0), vec2(0.0, 4.0));
  gl_FragColor = vec4(d, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(Math.abs(ctx.out.color[0][0] - 5.0) < 1e-6, `distance pre arg (got [${ctx.out.color[0].join(',')}])`);
}

// (m) uaddCarry(f(), 1u, c) — 0xFFFFFFFF + 1 → s = 0, c = 1.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
uint f() { return 4294967295u; }
out vec4 color;
void main() {
  uint c = 0u;
  uint s = uaddCarry(f(), 1u, c);
  color = vec4(float(s), float(c), 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(ctx.out.color[0][0] === 0 && ctx.out.color[0][1] === 1, `uaddCarry pre arg (got [${ctx.out.color[0].join(',')}])`);
}

// (n) umulExtended(f(), 0xFFFFFFFF, hi, lo) — 0xFFFFFFFF² = 0xFFFFFFFE00000001:
//     msb (hi) = 0xFFFFFFFE, lsb (lo) = 1 (GLSL ES 3.00 §8.10; matches
//     selftest-codegen-expr.ts umulExtended ordering).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
uint f() { return 4294967295u; }
out vec4 color;
void main() {
  uint hi = 0u, lo = 0u;
  umulExtended(f(), 4294967295u, hi, lo);
  color = vec4(float(hi), float(lo), 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(
    ctx.out.color[0][0] === 0xfffffffe && ctx.out.color[0][1] === 1,
    `umulExtended pre arg (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (o) faceforward(f(), I, Ng) — dot = 1 ≥ 0 → result = -N → x = -1.
{
  const { ctx } = runMain(
    `precision mediump float;
vec3 f() { return vec3(1.0, 0.0, 0.0); }
void main() {
  vec3 v = faceforward(f(), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0));
  float r = v.x;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === -1, `faceforward pre N (got [${ctx.out.color[0].join(',')}])`);
}

// (p) refract(I, N, f()) — I = N = (1,0,0), d = 1, k = 1 → x = eta - (eta + 1) = -1.
{
  const { ctx } = runMain(
    `precision mediump float;
float f() { return 1.5; }
void main() {
  vec3 v = refract(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), f());
  float r = v.x;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === -1, `refract pre eta (got [${ctx.out.color[0].join(',')}])`);
}

// (q) determinant(f()) — mat2(3.0) = diag(3,3) → det = 9.
{
  const { ctx } = runMain(
    `precision mediump float;
mat2 f() { return mat2(3.0); }
void main() {
  float r = determinant(f());
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === 9, `determinant pre arg (got [${ctx.out.color[0].join(',')}])`);
}

/* ------------------------------------------------------------------ */
/* Ternary arm/cond side effects run EXACTLY ONCE (multi-component)     */
/* ------------------------------------------------------------------ */

// (r) MULTI-COMPONENT statement ternary with side-effect call arms: a vec2
// user call shares ONE pre array (the inliner IIFE) across its components;
// the guarded arm hoist must embed it once (comp0) — later components read
// the temps it sets. Regression: the pre was embedded per component and the
// arm's side effects ran once PER COMPONENT (counter hit 2 instead of 1).
// Both cond polarities and both statement and decl-init positions.
{
  const { ctx } = runMain(
    `precision mediump float;
float counter = 0.0;
vec2 f() { counter = counter + 1.0; return vec2(1.0, 2.0); }
vec2 g() { counter = counter + 10.0; return vec2(3.0, 4.0); }
void main() {
  float c = 0.0;
  (c < 1.0) ? f() : g();
  vec2 v = (c < 1.0) ? f() : g();
  c = 1.0;
  (c < 1.0) ? f() : g();
  vec2 w = (c < 1.0) ? f() : g();
  gl_FragColor = vec4(counter, v.x, v.y, w.x);
}`,
    'FRAGMENT',
    100,
  );
  check(
    ctx.out.color[0][0] === 22,
    `multi-comp ternary arms run exactly once each (got counter ${ctx.out.color[0][0]})`,
  );
  check(
    ctx.out.color[0][1] === 1 && ctx.out.color[0][2] === 2 && ctx.out.color[0][3] === 3,
    `multi-comp ternary taken-arm values (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (s) SCALAR statement-position ternary with side-effect call arms: the
// statement discards the result; the taken arm's hoisted pre (IIFE) must
// execute exactly once, the untaken arm's never.
{
  const { ctx } = runMain(
    `precision mediump float;
float counter = 0.0;
float f() { counter = counter + 1.0; return 1.0; }
float g() { counter = counter + 10.0; return 2.0; }
void main() {
  float c = 0.0;
  (c < 1.0) ? f() : g();
  float r = (c < 1.0) ? f() : g();
  gl_FragColor = vec4(counter, r, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(
    ctx.out.color[0][0] === 2,
    `statement ternary taken arm runs exactly once (got counter ${ctx.out.color[0][0]})`,
  );
  check(ctx.out.color[0][1] === 1, `statement ternary value (got ${ctx.out.color[0][1]})`);
}

// (t) MULTI-COMPONENT ternary whose CONDITION carries side effects (user call
// folded into a comparison): the condition is embedded in every component's
// select (v/dx/dy + guarded arm hoists) — it must be hoisted ONCE, not
// re-run per component. Regression: the cond side effect ran n times.
{
  const { ctx } = runMain(
    `precision mediump float;
float counter = 0.0;
float f() { counter = counter + 1.0; return 0.0; }
void main() {
  vec2 v = (f() < 1.0) ? vec2(1.0, 2.0) : vec2(3.0, 4.0);
  gl_FragColor = vec4(counter, v.x, v.y, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(
    ctx.out.color[0][0] === 1,
    `multi-comp ternary cond side effect runs exactly once (got counter ${ctx.out.color[0][0]})`,
  );
  check(
    ctx.out.color[0][1] === 1 && ctx.out.color[0][2] === 2,
    `multi-comp ternary cond-taken values (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (u) DUAL mode: multi-component ternary with a side-effect call arm — the
// guarded v-plane hoist embeds the shared IIFE once (comp0); the dx/dy
// selects read the dual ret temps the IIFE set. Regression: per-component
// embedding ran the arm twice and re-set the dual temps between component
// reads.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
out vec4 color;
float counter = 0.0;
vec2 f() { counter = counter + 1.0; return vec2(1.0, 2.0); }
void main() {
  float c = 0.0;
  vec2 v = (c < 1.0) ? f() : vec2(0.0);
  float dx = dFdx(v.x);
  color = vec4(counter, v.x, v.y, dx);
}`,
    'FRAGMENT',
    300,
    { derivatives: true },
  );
  check(
    ctx.out.color[0][0] === 1,
    `dual multi-comp ternary arm runs exactly once (got counter ${ctx.out.color[0][0]})`,
  );
  check(
    ctx.out.color[0][1] === 1 && ctx.out.color[0][2] === 2 && ctx.out.color[0][3] === 0,
    `dual multi-comp ternary values (got [${ctx.out.color[0].join(',')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* walkObject: member/index of a multi-component CALL RESULT runs the  */
/* callee side effects EXACTLY ONCE (materializeSharedPre dedup)       */
/* ------------------------------------------------------------------ */

// (r1) STRUCT member of a call result, f().x — the struct-returning call
// carries ONE shared IIFE pre array on every component; walkObject must
// materialize it once (2-component struct; pre-fix fold-per-component ran
// the callee twice → counter 2).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
struct S { float x; float y; };
int g_counter = 0;
S f() { g_counter = g_counter + 1; return S(1.0, 2.0); }
out vec4 color;
void main() {
  float x = f().x;
  color = vec4(float(g_counter), x, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { structNames: ['S'] },
  );
  check(
    ctx.out.color[0][0] === 1,
    `struct member of call result runs callee once (got counter ${ctx.out.color[0][0]})`,
  );
  check(ctx.out.color[0][1] === 1, `struct member of call result value (got ${ctx.out.color[0][1]})`);
}

// (r2) SWIZZLE of a vec4 call result, f().z — 4 components share one IIFE
// pre; pre-fix ran the callee 4 times (counter 4).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int g_counter = 0;
vec4 f() { g_counter = g_counter + 1; return vec4(1.0, 2.0, 3.0, 4.0); }
out vec4 color;
void main() {
  float x = f().z;
  color = vec4(float(g_counter), x, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(
    ctx.out.color[0][0] === 1,
    `swizzle of call result runs callee once (got counter ${ctx.out.color[0][0]})`,
  );
  check(ctx.out.color[0][1] === 3, `swizzle of call result value (got ${ctx.out.color[0][1]})`);
}

// (r3) DYNAMIC index of a vec4 call result, f()[i] — walkObject materializes
// the result into a synth flat local, then the dynamic component index spills
// it into scratch via spillSynthLocal; the shared pre must still run once.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int g_counter = 0;
vec4 f() { g_counter = g_counter + 1; return vec4(1.0, 2.0, 3.0, 4.0); }
out vec4 color;
void main() {
  int i = 1;
  float x = f()[i];
  color = vec4(float(g_counter), x, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  check(
    ctx.out.color[0][0] === 1,
    `dynamic index of call result runs callee once (got counter ${ctx.out.color[0][0]})`,
  );
  check(ctx.out.color[0][1] === 2, `dynamic index of call result value (got ${ctx.out.color[0][1]})`);
}

// (r4) DUAL mode: struct member of a call result — the materialized (v, dx,
// dy) temp triples share one pre buffer (identity-deduped); the callee must
// run once and the member's dual planes must be readable via dFdx.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
struct S { float x; float y; };
int g_counter = 0;
S f() { g_counter = g_counter + 1; return S(1.0, 2.0); }
out vec4 color;
void main() {
  float x = f().x;
  float dx = dFdx(x);
  color = vec4(float(g_counter), x, dx, 1.0);
}`,
    'FRAGMENT',
    300,
    { derivatives: true, structNames: ['S'] },
  );
  check(
    ctx.out.color[0][0] === 1,
    `dual struct member of call result runs callee once (got counter ${ctx.out.color[0][0]})`,
  );
  check(
    ctx.out.color[0][1] === 1 && ctx.out.color[0][2] === 0,
    `dual struct member of call result values (got [${ctx.out.color[0].join(',')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-predrop: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
