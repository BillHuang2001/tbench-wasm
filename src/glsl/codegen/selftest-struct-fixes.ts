/**
 * selftest-struct-fixes.ts — regression checks for two codegen bugs:
 *
 * (1) gl_DepthRange builtin struct member lowering (expressions.ts subP /
 *     leafRead / leafDual / leafWrite). The builtin uniform struct
 *     gl_DepthRangeParameters { float near; float far; float diff; } (GLSL ES
 *     1.00 §7.6 / 3.00 §7.7, BOTH stages, read-only) resolves to a builtin
 *     path with no storage; walking `.near`/`.far`/`.diff` used to throw
 *     "codegen: struct member on builtin path". The fix lowers members to
 *     `ctx.depthRange[0/1/2]` (contract: [near, far, far−near]) and gives
 *     constant duals (uniform state has no screen-space derivative).
 *     CTS usage (read-only): /testsuites/WebGL/sdk/tests/conformance/ogles/
 *     GL/biuDepthRange/DepthRange_{vert,frag}.{vert,frag} —
 *     `vec4(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff, 1.0)`.
 *
 * (2) Runtime (non-const) struct == / != (expressions.ts emitBinary). GLSL ES
 *     1.00 §5.9 / 3.00 §5.9 make ==/!= legal on same-typed structs (member-
 *     wise comparison); semantics only folds CONST operands, so non-const
 *     struct comparisons reached codegen and threw
 *     "codegen: cannot compare struct and struct". The fix emits a recursive
 *     per-member comparison over the flat operand Values (bool leaves
 *     normalized with `!!` — uniform-store 0/1 vs literal true/false).
 *
 * Each case compiles a real shader, emits `main`, runs the generated JS
 * against a hand-built ctx, and asserts the runtime value. Every check FAILS
 * on the pre-fix code (crashes inside runMain are caught and counted).
 *
 * Run: npx tsx src/glsl/codegen/selftest-struct-fixes.ts
 *
 * Prints "selftest-struct-fixes: N checks" and exits 0 only when all pass.
 */
import { compileShader } from '../compiler.js';
import { CodegenEnv } from './env.js';
import { emitStatements } from './statements.js';
import { installUserFunctions } from './functions.js';
import { R } from './runtime.js';
import type { CodegenLayout, UniformSlot } from './index.js';
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
    attribLocations: new Map(),
    outputLocations: new Map([['gl_FragColor', 0], ['color', 0]]),
    uses: {
      pointSize: false, fragCoord: false, frontFacing: false, pointCoord: false,
      fragDepth: false, vertexId: false, instanceId: false, derivatives,
    },
  };
}

interface RunOpts {
  derivatives?: boolean;
  structNames?: string[];
  uniformSlots?: Record<string, UniformSlot>;
  attribLocations?: Record<string, number>;
  intUniforms?: Record<number, number>;
  /** Float-store uniform values (ctx.uniforms[slot]). */
  floatUniforms?: Record<number, number>;
  /** ctx.depthRange = [near, far, far−near] (gl_DepthRange builtin state). */
  depthRange?: number[];
  /** Provide a stub ctx.tex (texture2D family writes [1,2,3,4] into out). */
  tex?: boolean;
  attribs?: ArrayLike<ArrayLike<number> | number>[];
  attribIndices?: number[];
  fragCoord?: number[];
}

/** Compile + emit `main` + run; returns the body and the executed ctx. */
function runMain(
  src: string,
  stage: 'VERTEX' | 'FRAGMENT',
  version: 100 | 300,
  opts: RunOpts = {},
): { body: string; ctx: Record<string, any> } {
  const r = compileShader(src, { type: stage, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const layout = baseLayout(version, opts.derivatives);
  if (opts.uniformSlots) {
    for (const [k, v] of Object.entries(opts.uniformSlots)) layout.uniformSlots.set(k, v);
  }
  if (opts.attribLocations) {
    for (const [k, v] of Object.entries(opts.attribLocations)) layout.attribLocations.set(k, v);
  }
  const env = new CodegenEnv(stage, layout);
  // fragment.ts flips dual mode from layout.uses.derivatives — the selftest
  // drives the env directly, so mirror that here.
  if (opts.derivatives) env.dual = true;
  installUserFunctions(r.shader.ast, env);
  if (opts.structNames) for (const s of opts.structNames) env.structNames.add(s);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: stage === 'VERTEX' ? { position: [0, 0, 0, 0], pointSize: 0 } : { color: [[0, 0, 0, 0]] },
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
    fragCoord: opts.fragCoord ?? [0, 0, 0, 1],
    attribs: opts.attribs ?? [],
    attribIndices: opts.attribIndices ?? [],
  };
  if (opts.intUniforms) {
    for (const [k, v] of Object.entries(opts.intUniforms)) ctx.intUniforms[+k] = v;
  }
  if (opts.floatUniforms) {
    for (const [k, v] of Object.entries(opts.floatUniforms)) ctx.uniforms[+k] = v;
  }
  if (opts.depthRange) ctx.depthRange = new Float32Array(opts.depthRange);
  if (opts.tex) {
    // Stub sampler: sample2D(sampler2D family) writes [1,2,3,4] into out.
    ctx.tex = {
      out: [0, 0, 0, 0],
      outInt: new Int32Array(4),
      outUint: new Uint32Array(4),
      sample2D: () => {
        ctx.tex.out[0] = 1;
        ctx.tex.out[1] = 2;
        ctx.tex.out[2] = 3;
        ctx.tex.out[3] = 4;
      },
    };
  }
  fn(ctx, R);
  return { body, ctx };
}

/** runMain with crash capture: a codegen/runtime throw becomes a FAIL (so the
 *  pre-fix run reports every broken section instead of aborting at the first). */
function runMainCatch(
  src: string,
  stage: 'VERTEX' | 'FRAGMENT',
  version: 100 | 300,
  opts: RunOpts = {},
): { body: string; ctx: Record<string, any> } | null {
  try {
    return runMain(src, stage, version, opts);
  } catch (err) {
    check(false, `crash: ${(err as Error).message}`);
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* Item 3 — gl_DepthRange builtin struct member lowering                */
/* ------------------------------------------------------------------ */

// (3a) fragment 1.00: gl_DepthRange.near/far/diff → ctx.depthRange[0/1/2]
// with ctx.depthRange = [0.1, 1.0, 0.9] (near, far, far−near).
{
  const opts = { depthRange: [0.1, 1.0, 0.9] };
  const r = runMainCatch(
    `precision mediump float;
void main() {
  gl_FragColor = vec4(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff, 1.0);
}`,
    'FRAGMENT',
    100,
    opts,
  );
  if (r) {
    const dr = new Float32Array(opts.depthRange);
    check(r.body.includes('ctx.depthRange[0]'), `gl_DepthRange lowers to ctx.depthRange (body: ${r.body})`);
    check(
      r.ctx.out.color[0][0] === dr[0] && r.ctx.out.color[0][1] === dr[1] &&
        r.ctx.out.color[0][2] === dr[2] && r.ctx.out.color[0][3] === 1,
      `fragment gl_DepthRange.near/far/diff (got [${r.ctx.out.color[0].join(',')}], want [${dr[0]},${dr[1]},${dr[2]},1])`,
    );
  }
}

// (3b) vertex 1.00: same three members feed gl_Position.
{
  const opts = { depthRange: [0.1, 1.0, 0.9] };
  const r = runMainCatch(
    `void main() {
  gl_Position = vec4(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff, 1.0);
}`,
    'VERTEX',
    100,
    opts,
  );
  if (r) {
    const dr = new Float32Array(opts.depthRange);
    check(
      r.ctx.out.position[0] === dr[0] && r.ctx.out.position[1] === dr[1] &&
        r.ctx.out.position[2] === dr[2] && r.ctx.out.position[3] === 1,
      `vertex gl_DepthRange.near/far/diff (got [${r.ctx.out.position.join(',')}], want [${dr[0]},${dr[1]},${dr[2]},1])`,
    );
  }
}

// (3c) dual mode: dFdx(gl_DepthRange.near) — uniform state has no screen-space
// derivative, so the derivative must be exactly 0.
{
  const r = runMainCatch(
    `#version 300 es
precision mediump float;
out vec4 color;
void main() {
  float r = dFdx(gl_DepthRange.near);
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { derivatives: true, depthRange: [0.1, 1.0, 0.9] },
  );
  if (r) {
    check(
      r.ctx.out.color[0][0] === 0,
      `dFdx(gl_DepthRange.near) is 0 (got [${r.ctx.out.color[0].join(',')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Item 4 — runtime struct == / != (member-wise)                       */
/* ------------------------------------------------------------------ */

// (4a) equal structs from runtime (uniform-fed) values → == true, != false.
{
  const r = runMainCatch(
    `precision mediump float;
uniform float ux;
uniform float uy;
struct S { float x; float y; };
void main() {
  S a = S(ux, uy);
  S b = S(ux, uy);
  bool eq = (a == b);
  bool ne = (a != b);
  gl_FragColor = vec4(eq ? 1.0 : 0.0, ne ? 1.0 : 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    {
      uniformSlots: {
        ux: { store: 'float', slot: 0, stride: 0 },
        uy: { store: 'float', slot: 1, stride: 0 },
      },
      floatUniforms: { 0: 2.5, 1: -1.0 },
      structNames: ['S'],
    },
  );
  if (r) {
    check(r.ctx.out.color[0][0] === 1, `equal structs == true (got [${r.ctx.out.color[0].join(',')}])`);
    check(r.ctx.out.color[0][1] === 0, `equal structs != false (got [${r.ctx.out.color[0].join(',')}])`);
  }
}

// (4b) one differing member → == false, != true.
{
  const r = runMainCatch(
    `precision mediump float;
uniform float ux;
uniform float uy;
struct S { float x; float y; };
void main() {
  S a = S(ux, uy);
  S b = S(ux, 2.0);
  bool eq = (a == b);
  bool ne = (a != b);
  gl_FragColor = vec4(eq ? 1.0 : 0.0, ne ? 1.0 : 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    {
      uniformSlots: {
        ux: { store: 'float', slot: 0, stride: 0 },
        uy: { store: 'float', slot: 1, stride: 0 },
      },
      floatUniforms: { 0: 2.5, 1: -1.0 },
      structNames: ['S'],
    },
  );
  if (r) {
    check(r.ctx.out.color[0][0] === 0, `differing struct == false (got [${r.ctx.out.color[0].join(',')}])`);
    check(r.ctx.out.color[0][1] === 1, `differing struct != true (got [${r.ctx.out.color[0].join(',')}])`);
  }
}

// (4c) bool member normalization: uniform store holds 0/1 numbers, literals
// emit true/false — `!!` must make S(true, ux) == S(ub, ux) hold for ub=1.
{
  const r = runMainCatch(
    `precision mediump float;
uniform bool ub;
uniform float ux;
struct S { bool b; float x; };
void main() {
  S a = S(true, ux);
  S b = S(ub, ux);
  bool eq = (a == b);
  bool ne = (a != b);
  gl_FragColor = vec4(eq ? 1.0 : 0.0, ne ? 1.0 : 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    {
      uniformSlots: {
        ub: { store: 'int', slot: 0, stride: 0 },
        ux: { store: 'float', slot: 1, stride: 0 },
      },
      intUniforms: { 0: 1 },
      floatUniforms: { 1: 3.5 },
      structNames: ['S'],
    },
  );
  if (r) {
    check(
      r.ctx.out.color[0][0] === 1,
      `bool member normalized (uniform 1 vs literal true) == true (got [${r.ctx.out.color[0].join(',')}])`,
    );
    check(r.ctx.out.color[0][1] === 0, `bool member != false (got [${r.ctx.out.color[0].join(',')}])`);
  }
}

// (4d) nested struct member (Inner inside Outer) — recursion.
{
  const r = runMainCatch(
    `precision mediump float;
uniform float ux;
uniform float uy;
struct Inner { float x; float y; };
struct Outer { Inner i; float z; };
void main() {
  Outer a = Outer(Inner(ux, uy), 1.0);
  Outer b = Outer(Inner(ux, uy), 1.0);
  bool eq = (a == b);
  Outer c = Outer(Inner(ux, 0.25), 1.0);
  bool ne = (c == b);
  gl_FragColor = vec4(eq ? 1.0 : 0.0, ne ? 1.0 : 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    {
      uniformSlots: {
        ux: { store: 'float', slot: 0, stride: 0 },
        uy: { store: 'float', slot: 1, stride: 0 },
      },
      floatUniforms: { 0: 2.5, 1: -1.0 },
      structNames: ['Inner', 'Outer'],
    },
  );
  if (r) {
    check(
      r.ctx.out.color[0][0] === 1,
      `nested struct equal == true (got [${r.ctx.out.color[0].join(',')}])`,
    );
    check(
      r.ctx.out.color[0][1] === 0,
      `nested struct differing inner member != false (got [${r.ctx.out.color[0].join(',')}])`,
    );
  }
}

// (4e) struct with an array member — flat expansion over the array leaves.
// Struct locals containing arrays are unsupported by codegen (flat-local
// limitation), so the operands are STRUCT UNIFORMS with per-leaf slots.
{
  const r = runMainCatch(
    `precision mediump float;
struct S { float f[2]; float g; };
uniform S u;
uniform S v;
uniform S w;
void main() {
  bool eq = (u == v);
  bool ne = (u != w);
  gl_FragColor = vec4(eq ? 1.0 : 0.0, ne ? 1.0 : 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    {
      uniformSlots: {
        'u.f[0]': { store: 'float', slot: 0, stride: 0 },
        'u.f[1]': { store: 'float', slot: 1, stride: 0 },
        'u.g': { store: 'float', slot: 2, stride: 0 },
        'v.f[0]': { store: 'float', slot: 3, stride: 0 },
        'v.f[1]': { store: 'float', slot: 4, stride: 0 },
        'v.g': { store: 'float', slot: 5, stride: 0 },
        'w.f[0]': { store: 'float', slot: 6, stride: 0 },
        'w.f[1]': { store: 'float', slot: 7, stride: 0 },
        'w.g': { store: 'float', slot: 8, stride: 0 },
      },
      floatUniforms: {
        0: 2.5, 1: -1.0, 2: 1.0,
        3: 2.5, 4: -1.0, 5: 1.0,
        6: 2.5, 7: 7.0, 8: 1.0,
      },
    },
  );
  if (r) {
    check(
      r.ctx.out.color[0][0] === 1,
      `struct with array member equal == true (got [${r.ctx.out.color[0].join(',')}])`,
    );
    check(
      r.ctx.out.color[0][1] === 0,
      `struct with array member differing leaf != false (got [${r.ctx.out.color[0].join(',')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-struct-fixes: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
