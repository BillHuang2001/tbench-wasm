/**
 * PROBE (temporary, deleted before commit): verify pre-drop bugs.
 * Compiles real shaders, emits main, runs generated JS against a hand-built ctx.
 */
import { compileShader } from './compiler.js';
import { CodegenEnv } from './codegen/env.js';
import { emitStatements } from './codegen/statements.js';
import { installUserFunctions } from './codegen/functions.js';
import { R } from './codegen/runtime.js';
import type { CodegenLayout } from './codegen/index.js';
import type { TranslationUnit, FunctionDefinition } from './ast.js';

function findFn(tu: TranslationUnit, name: string): FunctionDefinition {
  for (const d of tu.declarations) {
    if (d.kind === 'function-definition' && d.prototype.name === name) return d;
  }
  throw new Error(`no function '${name}'`);
}

function baseLayout(version: 100 | 300): CodegenLayout {
  const uniformSlots = new Map<string, { store: 'float' | 'int'; slot: number; stride: number }>([
    ['u_color', { store: 'float', slot: 0, stride: 0 }],
    ['u_i', { store: 'int', slot: 4, stride: 0 }],
    ['uArr', { store: 'int', slot: 8, stride: 1 }],
    ['uArr[0]', { store: 'int', slot: 8, stride: 1 }],
  ]);
  return {
    version,
    uniformSlots,
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map([['a_pos', 0]]),
    outputLocations: new Map([['gl_FragColor', 0], ['color', 0]]),
    uses: {
      pointSize: false, fragCoord: false, frontFacing: false, pointCoord: false,
      fragDepth: false, vertexId: false, instanceId: false, derivatives: false,
    },
  };
}

function runMain(src: string, stage: 'VERTEX' | 'FRAGMENT', version: 100 | 300): { body: string; ctx: Record<string, any> } {
  const r = compileShader(src, { type: stage, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv(stage, baseLayout(version));
  installUserFunctions(r.shader.ast, env);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: stage === 'VERTEX' ? { position: [0, 0, 0, 0], pointSize: 0 } : { color: [[0, 0, 0, 0]] },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
    intUniforms: new Int32Array(16),
  };
  fn(ctx, R);
  return { body, ctx };
}

function t(name: string, cond: boolean, detail: string): void {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}: ${detail}`);
}

// 1. Non-dual ternary, condition = user function call (IIFE pre).
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
  t('ternary-cond-userfn', ctx.out.color[0][1] === 1 && ctx.out.color[0][0] === 0, `got [${ctx.out.color[0].join(',')}]`);
}

// 2. && with pre-carrying left operand.
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
  t('and-left-userfn', ctx.out.color[0][0] === 1, `got [${ctx.out.color[0].join(',')}]`);
}

// 3. || with pre-carrying left operand (true fn; rhs false — must yield 1).
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
  t('or-left-userfn', ctx.out.color[0][0] === 1, `got [${ctx.out.color[0].join(',')}]`);
}

// 4b. Mixed-base relational comparison with a pre-carrying LEFT operand
//    (user function call returning int, IIFE pre).
{
  const r = compileShader(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float r = (f() < 3.5) ? 1.0 : 0.0;
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    { type: 'FRAGMENT', version: 300 },
  );
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('FRAGMENT', baseLayout(300));
  installUserFunctions(r.shader.ast, env);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: { color: [[0, 0, 0, 0]] },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
    intUniforms: new Int32Array(16),
  };
  fn(ctx, R);
  t('rel-int-userfn-vs-float', ctx.out.color[0][0] === 1, `got [${ctx.out.color[0].join(',')}] body:\n${body}`);
}

// 5. == with pre-carrying left operand (user fn returning int vs float).
{
  const r = compileShader(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float r = (f() == 3.0) ? 1.0 : 0.0;
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    { type: 'FRAGMENT', version: 300 },
  );
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('FRAGMENT', baseLayout(300));
  installUserFunctions(r.shader.ast, env);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: { color: [[0, 0, 0, 0]] },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
    intUniforms: new Int32Array(16),
  };
  fn(ctx, R);
  t('eq-int-userfn-vs-float', ctx.out.color[0][0] === 1, `got [${ctx.out.color[0].join(',')}] body:\n${body}`);
}

// 6. modf with pre-carrying arg (user function call) — ES 3.00 only.
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
  t('modf-pre-arg', Math.abs(ctx.out.color[0][0] - 0.7) < 1e-6 && ctx.out.color[0][1] === 3, `got [${ctx.out.color[0].join(',')}]`);
}

// 7. normalize with pre-carrying arg.
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
  t('normalize-pre-arg', Math.abs(ctx.out.color[0][0] - 0.6) < 1e-6, `got [${ctx.out.color[0].join(',')}]`);
}

// 8. distance with pre-carrying arg.
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
  t('distance-pre-arg', Math.abs(ctx.out.color[0][0] - 5.0) < 1e-6, `got [${ctx.out.color[0].join(',')}]`);
}

// 9. Ternary with pre-carrying arms (user function call in an arm).
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
  t('ternary-arm-userfn', ctx.out.color[0][0] === 2, `got [${ctx.out.color[0].join(',')}]`);
}

// 10. Ternary with pre-carrying cond + arms + vector type.
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
  t('ternary-cond-userfn-arm-userfn', ctx.out.color[0][0] === 2 && ctx.out.color[0][1] === 1, `got [${ctx.out.color[0].join(',')}]`);
}

// 11. Decl init with mixed-base conversion (int user fn → float).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float x = f();
  color = vec4(x, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  t('decl-init-int-userfn-to-float', ctx.out.color[0][0] === 3, `got [${ctx.out.color[0].join(',')}]`);
}

// 12. Compound assign with mixed-base RHS (float += int user fn) in
//     EXPRESSION position (emitAssign compound path).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float x = 1.0;
  float y = (x += f()) + 0.0;
  color = vec4(y, x, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  t('compound-mixed-base-rhs', ctx.out.color[0][0] === 4 && ctx.out.color[0][1] === 4, `got [${ctx.out.color[0].join(',')}]`);
}

// 13. Bitwise mixed base: int user fn & uint literal (no wrapper ctor).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
int f() { return 6; }
out vec4 color;
void main() {
  uint r = f() & 3u;
  color = vec4(float(r), 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
  );
  t('bitwise-int-userfn', ctx.out.color[0][0] === 2, `got [${ctx.out.color[0].join(',')}]`);
}

// 14. Struct ctor with mixed-base member (int user fn → float member).
{
  const r = compileShader(
    `#version 300 es
precision mediump float;
struct S { float x; };
int f() { return 3; }
out vec4 color;
void main() {
  S s = S(f());
  color = vec4(s.x, 0.0, 0.0, 1.0);
}`,
    { type: 'FRAGMENT', version: 300 },
  );
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('FRAGMENT', baseLayout(300));
  installUserFunctions(r.shader.ast, env);
  env.structNames.add('S');
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: { color: [[0, 0, 0, 0]] },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
    intUniforms: new Int32Array(16),
  };
  fn(ctx, R);
  t('struct-ctor-mixed-base', ctx.out.color[0][0] === 3, `got [${ctx.out.color[0].join(',')}] body:\n${body}`);
}

// 15. Array ctor with mixed-base element (int user fn → float element).
{
  const r = compileShader(
    `#version 300 es
precision mediump float;
int f() { return 3; }
out vec4 color;
void main() {
  float arr[1] = float[1](f());
  color = vec4(arr[0], 0.0, 0.0, 1.0);
}`,
    { type: 'FRAGMENT', version: 300 },
  );
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('FRAGMENT', baseLayout(300));
  installUserFunctions(r.shader.ast, env);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: { color: [[0, 0, 0, 0]] },
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
    intUniforms: new Int32Array(16),
  };
  fn(ctx, R);
  t('array-ctor-mixed-base', ctx.out.color[0][0] === 3, `got [${ctx.out.color[0].join(',')}] body:\n${body}`);
}

// 16. uaddCarry with pre-carrying args.
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
  t('uaddCarry-pre-arg', ctx.out.color[0][0] === 0 && ctx.out.color[0][1] === 1, `got [${ctx.out.color[0].join(',')}]`);
}

// 17. umulExtended with pre-carrying args.
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
  t('umulExtended-pre-arg', ctx.out.color[0][0] === 1 && ctx.out.color[0][1] === 0xfffffffe, `got [${ctx.out.color[0].join(',')}]`);
}

// 18. faceforward with a pre-carrying N (user fn returning vec3, IIFE pre).
//     dot(I, Ng) = 1 >= 0 → result = -N → x = -1.
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
  t('faceforward-pre-arg', ctx.out.color[0][0] === -1, `got [${ctx.out.color[0].join(',')}]`);
}

// 19. refract with a pre-carrying eta (user fn returning float, IIFE pre).
//     I = N = (1,0,0), d = 1, k = 1, result.x = eta - (eta + 1) = -1.
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
  t('refract-pre-arg', ctx.out.color[0][0] === -1, `got [${ctx.out.color[0].join(',')}]`);
}

// 20. determinant with a pre-carrying matrix (user fn returning mat2, IIFE pre).
//     mat2(3.0) = diag(3,3) → det = 9.
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
  t('determinant-pre-arg', ctx.out.color[0][0] === 9, `got [${ctx.out.color[0].join(',')}]`);
}
