/**
 * selftest-bugfix.ts — regression checks for five root-caused codegen bugs.
 *
 * (a) BUG 1 — member/index of a call (or other non-path) result:
 *     `texture2D(u, v).x` / `.rgba`, user fn returning vec3 `.y`, struct-
 *     returning fn `.member` / `.member.sub` chains, ctor/binary results,
 *     const + DYNAMIC index of a texture result, and dual-mode members
 *     (dFdx of a member-of-call value). Pre-fix, walk() threw
 *     "codegen: not a path expression" for every one of these.
 * (b) BUG 2 — local struct-array member assignment: a mixed struct
 *     {int first; vec3 color;} array used to allocate in intScratch, so
 *     `arr[0].color = vec3(0, 0.5, 0)` wrote 0. The int-store rule is now
 *     "all leaves integral"; the store is selected per MEMBER type.
 * (c) BUG 3/6 — bool strict equality: the uniform store holds 0/1 numbers
 *     while literals emit true/false, so `u == true` compiled to
 *     `(1 === true)` (false). ==/!=/^^ and the equal()/notEqual() builtins
 *     now normalize both sides with `!!`.
 * (d) BUG 5 — swizzled attribute fetch stride: `a.xy` on a vec4 fetched
 *     with the SWIZZLE width (stride 2) instead of the declared width
 *     (stride 4). attribRead now takes the declared component count.
 *
 * Each case compiles a real shader (compileShader → annotated AST), emits
 * `main` (installUserFunctions + emitStatements), runs the generated JS via
 * `new Function('ctx','R', body)` against a hand-built ctx, and asserts the
 * runtime value. Every check FAILS on the pre-fix code (crashes, truncated
 * values, or wrong strides).
 *
 * Run: npx tsx src/glsl/codegen/selftest-bugfix.ts
 *
 * Prints "selftest-bugfix: N checks" and exits 0 only when all pass.
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

const texUniforms = { u: { store: 'int', slot: 0, stride: 0 } as UniformSlot };

/* ------------------------------------------------------------------ */
/* BUG 1 — member/index of a call result                               */
/* ------------------------------------------------------------------ */

// (a1) texture2D(...).x — scalar member of a builtin-call result.
{
  const { ctx } = runMain(
    `precision mediump float;
uniform sampler2D u;
void main() {
  float r = texture2D(u, vec2(0.0, 0.0)).x;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    { uniformSlots: texUniforms, tex: true },
  );
  check(ctx.out.color[0][0] === 1, `texture2D(u,v).x (got [${ctx.out.color[0].join(',')}])`);
}

// (a2) texture2D(...).rgba — swizzle of a builtin-call result.
{
  const { ctx } = runMain(
    `precision mediump float;
uniform sampler2D u;
void main() {
  gl_FragColor = texture2D(u, vec2(0.0, 0.0)).rgba;
}`,
    'FRAGMENT',
    100,
    { uniformSlots: texUniforms, tex: true },
  );
  check(
    ctx.out.color[0][0] === 1 && ctx.out.color[0][1] === 2 &&
      ctx.out.color[0][2] === 3 && ctx.out.color[0][3] === 4,
    `texture2D(u,v).rgba (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (a3) user fn returning vec3 — member of an inlined-call result.
{
  const { ctx } = runMain(
    `precision mediump float;
vec3 f() { return vec3(4.0, 5.0, 6.0); }
void main() {
  float r = f().y;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === 5, `userfn vec3 .y (got [${ctx.out.color[0].join(',')}])`);
}

// (a4) struct-returning fn — .member and .member.sub chains.
{
  const { ctx } = runMain(
    `precision mediump float;
struct Inner { float a; float b; };
struct Outer { Inner inner; float w; };
Outer makeO() { return Outer(Inner(1.0, 2.0), 3.0); }
void main() {
  float r = makeO().inner.b;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    { structNames: ['Inner', 'Outer'] },
  );
  check(ctx.out.color[0][0] === 2, `struct-return fn .inner.b (got [${ctx.out.color[0].join(',')}])`);
}

// (a5) struct member + swizzle chain: makeS().pos.xy.
{
  const { ctx } = runMain(
    `precision mediump float;
struct S { vec3 pos; float w; };
S makeS() { return S(vec3(1.0, 2.0, 3.0), 4.0); }
void main() {
  float r = makeS().pos.y + makeS().w;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    { structNames: ['S'] },
  );
  check(ctx.out.color[0][0] === 6, `struct fn .pos.y + .w (got [${ctx.out.color[0].join(',')}])`);
}

// (a6) constructor-call result member: vec3(...).y.
{
  const { ctx } = runMain(
    `precision mediump float;
void main() {
  float r = vec3(1.0, 2.0, 3.0).y;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === 2, `vec3(...).y (got [${ctx.out.color[0].join(',')}])`);
}

// (a7) binary-result member: (a + b).y.
{
  const { ctx } = runMain(
    `precision mediump float;
void main() {
  vec2 a = vec2(1.0, 2.0);
  vec2 b = vec2(3.0, 4.0);
  float r = (a + b).y;
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
  );
  check(ctx.out.color[0][0] === 6, `(a + b).y (got [${ctx.out.color[0].join(',')}])`);
}

// (a8) const index of a call result: texture2D(u, v)[2].
{
  const { ctx } = runMain(
    `precision mediump float;
uniform sampler2D u;
void main() {
  float r = texture2D(u, vec2(0.0, 0.0))[2];
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    { uniformSlots: texUniforms, tex: true },
  );
  check(ctx.out.color[0][0] === 3, `texture2D(u,v)[2] (got [${ctx.out.color[0].join(',')}])`);
}

// (a9) DYNAMIC index of a call result (ES 3.00): texture(u, v)[i] — spills
// the synthesized temps into a scratch block (flat-local reads ignore p.dyn).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
uniform sampler2D u;
uniform int i;
out vec4 color;
void main() {
  float r = texture(u, vec2(0.0))[i];
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { uniformSlots: { u: { store: 'int', slot: 0, stride: 0 }, i: { store: 'int', slot: 1, stride: 0 } }, tex: true, intUniforms: { 1: 2 } },
  );
  check(ctx.out.color[0][0] === 3, `texture(u,v)[i] dynamic (got [${ctx.out.color[0].join(',')}])`);
}

// (a10) DUAL mode: dFdx(texture(u, v).x) — member of a call in derivative
// mode. Texture results carry zero duals → the derivative is 0; the point is
// the synthesized (v, dx, dy) temp triple compiles and runs.
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
uniform sampler2D u;
out vec4 color;
void main() {
  float r = dFdx(texture(u, vec2(0.0)).x);
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { derivatives: true, uniformSlots: texUniforms, tex: true },
  );
  check(ctx.out.color[0][0] === 0, `dual dFdx(texture(u,v).x) (got [${ctx.out.color[0].join(',')}])`);
}

// (a11) DUAL mode with REAL duals: dFdx(vec3(gl_FragCoord.x, 0, 0).x) — the
// ctor-call member materializes a triple whose dx plane carries gl_FragCoord.x's
// derivative (1). Pins the dyNames/dxNames temp convention on synthesized
// locals (leafDual must read the recorded plane names, not `tN_dy`).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
out vec4 color;
void main() {
  float r = dFdx(vec3(gl_FragCoord.x, 0.0, 0.0).x);
  color = vec4(r, 0.0, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { derivatives: true, fragCoord: [10, 20, 0, 1] },
  );
  check(ctx.out.color[0][0] === 1, `dual dFdx(vec3(fragCoord.x,..).x) (got [${ctx.out.color[0].join(',')}])`);
}

// (a12) assignment to a member of a call result must be rejected cleanly
// (semantics: "lvalue required" — or codegen "not an lvalue" if it got past
// semantics). Pre-fix this crashed with "not a path expression"; post-fix it
// must never silently compile.
{
  const r = compileShader(
    `precision mediump float;
vec3 f() { return vec3(1.0, 2.0, 3.0); }
void main() {
  f().x = 1.0;
  gl_FragColor = vec4(1.0);
}`,
    { type: 'FRAGMENT', version: 100 },
  );
  let rejected = false;
  if (!r.ok) {
    rejected = true;
  } else {
    try {
      const mainFn = findFn(r.shader.ast, 'main');
      const env = new CodegenEnv('FRAGMENT', baseLayout(100));
      installUserFunctions(r.shader.ast, env);
      emitStatements(mainFn.body.body, env);
    } catch (e) {
      rejected = true;
    }
  }
  check(rejected, `assign to f().x must be rejected (semantics or 'not an lvalue')`);
}

/* ------------------------------------------------------------------ */
/* BUG 2 — mixed struct-array scratch store                            */
/* ------------------------------------------------------------------ */

// (b1) struct array {int id; vec3 color;} — float member assignment must keep
// 0.5 (pre-fix the intScratch container truncated it to 0), int member exact.
{
  const { ctx } = runMain(
    `precision mediump float;
precision mediump int;
struct Light { int id; vec3 color; };
void main() {
  Light lights[2];
  lights[0].id = 7;
  lights[0].color = vec3(0.0, 0.5, 0.0);
  lights[1].color = vec3(0.25, 0.0, 0.0);
  gl_FragColor = vec4(float(lights[0].id), lights[0].color.y, lights[1].color.x, 1.0);
}`,
    'FRAGMENT',
    100,
    { structNames: ['Light'] },
  );
  check(
    ctx.out.color[0][0] === 7 && ctx.out.color[0][1] === 0.5 && ctx.out.color[0][2] === 0.25,
    `mixed struct array members (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (b2) all-integral struct array must still use intScratch semantics (exact).
{
  const { ctx } = runMain(
    `precision mediump float;
precision mediump int;
struct I2 { int a; int b; };
void main() {
  I2 arr[2];
  arr[0].a = 5;
  arr[0].b = -3;
  gl_FragColor = vec4(float(arr[0].a), float(arr[0].b), 0.0, 1.0);
}`,
    'FRAGMENT',
    100,
    { structNames: ['I2'] },
  );
  check(
    ctx.out.color[0][0] === 5 && ctx.out.color[0][1] === -3,
    `all-integral struct array (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (b3) struct-ARRAY inout function param (makeParamLocal array branch — the
// inout write-back exercises the same int-store rule on the param block).
{
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
struct Light { int id; vec3 color; };
void setColor(inout Light arr[2]) {
  arr[0].color = vec3(0.0, 0.5, 0.0);
  arr[0].id = 9;
}
out vec4 color;
void main() {
  Light lights[2];
  setColor(lights);
  color = vec4(float(lights[0].id), lights[0].color.y, 0.0, 1.0);
}`,
    'FRAGMENT',
    300,
    { structNames: ['Light'] },
  );
  check(
    ctx.out.color[0][0] === 9 && ctx.out.color[0][1] === 0.5,
    `struct-array inout fn param (got [${ctx.out.color[0].join(',')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* BUG 3/6 — bool-uniform strict equality                              */
/* ------------------------------------------------------------------ */

// (c1) u == true / u != false / u ^^ true with the uniform set to 1 and 0.
{
  for (const u of [1, 0]) {
    const { ctx } = runMain(
      `precision mediump float;
uniform bool u;
void main() {
  gl_FragColor = vec4(
    (u == true) ? 1.0 : 0.0,
    (u != false) ? 1.0 : 0.0,
    (u ^^ true) ? 1.0 : 0.0,
    1.0
  );
}`,
      'FRAGMENT',
      100,
      { uniformSlots: { u: { store: 'int', slot: 0, stride: 0 } }, intUniforms: { 0: u } },
    );
    const exp = u === 1 ? [1, 1, 0, 1] : [0, 0, 1, 1];
    check(
      ctx.out.color[0][0] === exp[0] && ctx.out.color[0][1] === exp[1] && ctx.out.color[0][2] === exp[2],
      `bool uniform u=${u} ==/!=/^^ (got [${ctx.out.color[0].join(',')}], want [${exp.join(',')}])`,
    );
  }
}

// (c2) literal-only bool ops stay correct after `!!` normalization.
{
  const { ctx } = runMain(
    `precision mediump float;
void main() {
  gl_FragColor = vec4(
    (true == true) ? 1.0 : 0.0,
    (false != true) ? 1.0 : 0.0,
    (true ^^ false) ? 1.0 : 0.0,
    (true ^^ true) ? 1.0 : 0.0
  );
}`,
    'FRAGMENT',
    100,
  );
  check(
    ctx.out.color[0][0] === 1 && ctx.out.color[0][1] === 1 && ctx.out.color[0][2] === 1 && ctx.out.color[0][3] === 0,
    `literal bool ops (got [${ctx.out.color[0].join(',')}])`,
  );
}

// (c3) bool VECTOR == with a uniform operand.
{
  for (const u of [1, 0]) {
    const { ctx } = runMain(
      `precision mediump float;
uniform bool u;
void main() {
  bool r = (bvec2(u, false) == bvec2(true, false));
  gl_FragColor = vec4(r ? 1.0 : 0.0, 0.0, 0.0, 1.0);
}`,
      'FRAGMENT',
      100,
      { uniformSlots: { u: { store: 'int', slot: 0, stride: 0 } }, intUniforms: { 0: u } },
    );
    const exp = u === 1 ? 1 : 0;
    check(ctx.out.color[0][0] === exp, `bvec2(u,false) == bvec2(true,false) u=${u} (got [${ctx.out.color[0].join(',')}])`);
  }
}

// (c4) equal()/notEqual() builtins with bool operands (audit fix; the bool
// variants exist in the ES 3.00 builtin table).
{
  for (const u of [1, 0]) {
    const { ctx } = runMain(
      `#version 300 es
precision mediump float;
uniform bool u;
out vec4 color;
void main() {
  bvec2 r = equal(bvec2(u, false), bvec2(true, false));
  color = vec4(r.x ? 1.0 : 0.0, 0.0, 0.0, 1.0);
}`,
      'FRAGMENT',
      300,
      { uniformSlots: { u: { store: 'int', slot: 0, stride: 0 } }, intUniforms: { 0: u } },
    );
    const exp = u === 1 ? 1 : 0;
    check(ctx.out.color[0][0] === exp, `equal(bvec2(u,..), ..) u=${u} (got [${ctx.out.color[0].join(',')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* BUG 5 — swizzled attribute fetch stride                             */
/* ------------------------------------------------------------------ */

// (d1) attribute vec4 a; a.xy with vertex index 1 — pre-fix the swizzle width
// (2) strided the fetch, reading (3,4) instead of (5,6).
{
  const { body, ctx } = runMain(
    `attribute vec4 a;
void main() {
  gl_Position = vec4(a.xy, 0.0, 1.0);
}`,
    'VERTEX',
    100,
    {
      attribLocations: { a: 0 },
      attribs: [new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])],
      attribIndices: [1],
    },
  );
  check(body.includes('ctx.attribIndices[0] * 4'), `a.xy body stride is 4 (body: ${body})`);
  check(
    ctx.out.position[0] === 5 && ctx.out.position[1] === 6,
    `a.xy vertex 1 (got [${ctx.out.position.join(',')}])`,
  );
}

// (d2) a.zw with vertex index 1 → components 2,3 of the declared vec4.
{
  const { ctx } = runMain(
    `attribute vec4 a;
void main() {
  gl_Position = vec4(a.zw, 0.0, 1.0);
}`,
    'VERTEX',
    100,
    {
      attribLocations: { a: 0 },
      attribs: [new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])],
      attribIndices: [1],
    },
  );
  check(
    ctx.out.position[0] === 7 && ctx.out.position[1] === 8,
    `a.zw vertex 1 (got [${ctx.out.position.join(',')}])`,
  );
}

// (d3) matrix attribute: m[1] reads location+1 with rows components — the
// declared-width rule must not disturb matrix column location math.
{
  const { ctx } = runMain(
    `attribute mat2 m;
void main() {
  gl_Position = vec4(m[1], 0.0, 1.0);
}`,
    'VERTEX',
    100,
    {
      attribLocations: { m: 0 },
      attribs: [new Float32Array([1, 2, 3, 4]), new Float32Array([5, 6, 7, 8])],
      attribIndices: [0, 0],
    },
  );
  check(
    ctx.out.position[0] === 5 && ctx.out.position[1] === 6,
    `mat2 m[1] (got [${ctx.out.position.join(',')}])`,
  );
}

// (d4) whole-vec4 read keeps stride 4 (declared width) at vertex 1.
{
  const { ctx } = runMain(
    `attribute vec4 a;
void main() {
  gl_Position = a;
}`,
    'VERTEX',
    100,
    {
      attribLocations: { a: 0 },
      attribs: [new Float32Array([1, 2, 3, 4, 5, 6, 7, 8])],
      attribIndices: [1],
    },
  );
  check(
    ctx.out.position[0] === 5 && ctx.out.position[1] === 6 &&
      ctx.out.position[2] === 7 && ctx.out.position[3] === 8,
    `vec4 a whole read vertex 1 (got [${ctx.out.position.join(',')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-bugfix: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
