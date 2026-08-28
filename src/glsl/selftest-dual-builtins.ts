/**
 * selftest-dual-builtins.ts — dual-mode BUILTIN derivative templates + the
 * implicit-LOD texture gradient routing (task C5b).
 *
 * In dual mode (layout.uses.derivatives → env.dual) every FLOAT value carries
 * (v, dx, dy). This selftest pins the per-builtin ANALYTIC derivative
 * templates (dualLowerBuiltin in codegen/expr-builtins.ts) and the texture
 * gradient routing: implicit-LOD fragment samples pass the coordinate
 * screen-space derivatives into the ctx.tex sample* gradient params, and the
 * sampled RESULT carries zero duals (dFdx(texture(...)) → 0 — the standard
 * approximation).
 *
 * CONSTRAINT (C5a2 runs in parallel — its arithmetic/ctor dual machinery may
 * not exist on this branch): the shaders below use ONLY varying/uniform/
 * gl_FragCoord reads + builtin calls + dFdx/dFdy/fwidth. No float arithmetic,
 * no float constructors (vec2/vec3 values come from typed VARYING reads, e.g.
 * `varying vec2 vv;` — no ctor needed).
 *
 * Run: npx tsx src/glsl/selftest-dual-builtins.ts
 *
 * Prints "OK" and exits 0 on success.
 */
import type { TranslationUnit } from './ast.js';
import { compileShader } from './compiler.js';
import type { ShaderUses } from './compiler.js';
import { generateFragmentStage } from './codegen/index.js';
import type { CodegenLayout, UniformSlot, VaryingLayout } from './codegen/index.js';
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

/** Float32Array-stored colors are frounded — compare with a small tolerance. */
function near(got: number, expected: number, msg: string): void {
  check(Math.abs(got - expected) < 1e-6, `${msg} (got ${got}, expected ${expected})`);
}

/* ------------------------------------------------------------------ */
/* Layout helpers (derivatives ALWAYS on — dual mode)                  */
/* ------------------------------------------------------------------ */

function dualUses(): ShaderUses {
  return {
    pointSize: false,
    fragCoord: false,
    frontFacing: false,
    pointCoord: false,
    fragDepth: false,
    vertexId: false,
    instanceId: false,
    drawId: false,
    derivatives: true,
    depthRange: false,
  };
}

function baseLayout(version: 100 | 300, extra?: Partial<CodegenLayout>): CodegenLayout {
  return {
    version,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map(),
    outputLocations: new Map([['gl_FragColor', 0]]),
    uses: dualUses(),
    ...extra,
  };
}

const slot = (store: 'float' | 'int', s: number, stride = 0): UniformSlot => ({ store, slot: s, stride });
const vg = (index: number, offset: number, components: number, elemComponents: number, flat = false): VaryingLayout =>
  ({ index, offset, components, elemComponents, flat });

/* ------------------------------------------------------------------ */
/* Driver: compileShader → generateFragmentStage → new Function → run  */
/* ------------------------------------------------------------------ */

/** A fragment ctx with one varying (v + ddx + ddy arrays) and float/int stores. */
function fragCtx(res: { scratchSize: number; intScratchSize: number }): Record<string, any> {
  return {
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(res.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(res.intScratchSize, 16)),
    out: { color: [new Float32Array(4)], fragDepth: 0 },
    discarded: false,
    fragCoord: new Float32Array([0, 0, 0, 0]),
    frontFacing: true,
    pointCoord: new Float32Array([0, 0]),
    varyings: [],
  };
}

interface DualRunResult {
  body: string;
  ctx: Record<string, any>;
}

function runDual(
  src: string,
  layout: CodegenLayout,
  ctxExtra: Record<string, unknown> = {},
  extensions: string[] = [],
): DualRunResult {
  const r = compileShader(src, {
    type: 'FRAGMENT',
    version: layout.version,
    extensions: new Set(['GL_OES_standard_derivatives', ...extensions]),
  });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const res = generateFragmentStage(r.shader.ast as TranslationUnit, layout);
  const fn = new Function('ctx', 'R', res.body);
  const ctx = { ...fragCtx(res), ...ctxExtra };
  fn(ctx, R);
  return { body: res.body, ctx };
}

/** Compile-only: codegen must succeed (no throw) — the body is returned. */
function compileOnly(src: string, layout: CodegenLayout, extensions: string[] = []): string {
  const r = compileShader(src, {
    type: 'FRAGMENT',
    version: layout.version,
    extensions: new Set(['GL_OES_standard_derivatives', ...extensions]),
  });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  return generateFragmentStage(r.shader.ast as TranslationUnit, layout).body;
}

/** Codegen must THROW the C5b no-derivative-template error for `name`. */
function expectDualThrow(src: string, layout: CodegenLayout, name: string): void {
  const r = compileShader(src, {
    type: 'FRAGMENT',
    version: layout.version,
    extensions: new Set(['GL_OES_standard_derivatives']),
  });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  let threw = '';
  try {
    generateFragmentStage(r.shader.ast as TranslationUnit, layout);
  } catch (e) {
    threw = String(e);
  }
  check(threw.includes('no derivative template'), `${name} in dual mode throws the C5b template error (got: ${threw || 'no throw'})`);
}

/** A ctx.tex stub shaped like raster's TextureEnv: methods write `out`. The
 *  gradient params are folded into out[2]/out[3] with distinct weights so the
 *  routing is observable. */
function makeTexStub(): Record<string, any> {
  const out = new Float32Array(4);
  const outInt = new Int32Array(4);
  const outUint = new Uint32Array(4);
  return {
    units: [],
    out,
    outInt,
    outUint,
    sample2D(_u: number, u: number, v: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      out[0] = u; out[1] = v; out[2] = 10 * dux + 20 * dvx + 30 * duy + 40 * dvy; out[3] = bias;
    },
    sample2DLod(_u: number, u: number, v: number, lod: number): void {
      out[0] = u; out[1] = v; out[2] = lod; out[3] = 1;
    },
    sample3D(_u: number, u: number, v: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      out[0] = u; out[1] = v; out[2] = 10 * dux + 20 * dvx + 30 * dwx + 40 * duy + 50 * dvy + 60 * dwy; out[3] = bias;
    },
    sample3DLod(_u: number, u: number, v: number, w: number, lod: number): void {
      out[0] = u; out[1] = v; out[2] = w; out[3] = lod;
    },
    sampleCube(_u: number, u: number, v: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      out[0] = u; out[1] = v; out[2] = w; out[3] = 10 * dux + 20 * dvx + 30 * dwx + 40 * duy + 50 * dvy + 60 * dwy;
    },
    sampleCubeLod(_u: number, u: number, v: number, w: number, lod: number): void {
      out[0] = u; out[1] = v; out[2] = w; out[3] = lod;
    },
    sample2DArray(_u: number, u: number, v: number, layer: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      out[0] = u; out[1] = v; out[2] = layer; out[3] = 10 * dux + 20 * dvx + 30 * duy + 40 * dvy;
    },
    sample2DArrayLod(_u: number, u: number, v: number, layer: number, lod: number): void {
      out[0] = u; out[1] = v; out[2] = layer; out[3] = lod;
    },
    sample2DShadow(_u: number, u: number, v: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      out[0] = ref + 10 * dux + 20 * dvx + 30 * duy + 40 * dvy;
    },
    sampleCubeShadow(_u: number, u: number, v: number, w: number, ref: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      out[0] = ref;
    },
    sample2DArrayShadow(_u: number, u: number, v: number, layer: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      out[0] = ref;
    },
    texelFetch2D(_u: number, x: number, y: number, level: number): void {
      out[0] = x; out[1] = y; out[2] = level; out[3] = 1;
    },
    texelFetch3D(_u: number, x: number, y: number, z: number, level: number): void {
      out[0] = x; out[1] = y; out[2] = z; out[3] = level;
    },
    texelFetch2DArray(_u: number, x: number, y: number, layer: number, level: number): void {
      out[0] = x; out[1] = y; out[2] = layer; out[3] = level;
    },
  };
}

/* ================================================================== */
/* 1. Scalar §8.3 templates: sin/cos/exp/log/sqrt/pow                  */
/* ================================================================== */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() {
       gl_FragColor.x = dFdx(sin(v));
       gl_FragColor.y = dFdx(cos(v));
       gl_FragColor.z = dFdx(exp(v));
       gl_FragColor.w = dFdx(log(v));
     }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  const c = r.ctx.out.color[0];
  near(c[0], Math.cos(0.5) * 2, `dFdx(sin(v)) == cos(v)*ddx`);
  near(c[1], -Math.sin(0.5) * 2, `dFdx(cos(v)) == -sin(v)*ddx`);
  near(c[2], Math.exp(0.5) * 2, `dFdx(exp(v)) == exp(v)*ddx`);
  near(c[3], (1 / 0.5) * 2, `dFdx(log(v)) == ddx/v`);
  check(
    r.body.includes('Math.cos(ctx.varyings[0].v[0]) * (ctx.varyings[0].ddx[0])'),
    `sin template emits the chain-rule dx string`,
  );
}

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() {
       gl_FragColor.x = dFdx(sqrt(v));
       gl_FragColor.y = dFdx(pow(v, 2.0));
     }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  const c = r.ctx.out.color[0];
  near(c[0], 2 / (2 * Math.sqrt(0.5)), `dFdx(sqrt(v)) == ddx/(2*sqrt(v))`);
  near(c[1], Math.pow(0.5, 2) * ((2 * 2) / 0.5), `dFdx(pow(v, 2.0)) == v*(y*ddx/x)`);
}

/* ================================================================== */
/* 2. mix: uniform endpoints, varying factor                          */
/* ================================================================== */

{
  const layout = baseLayout(100, {
    uniformSlots: new Map([
      ['a', slot('float', 0)],
      ['b', slot('float', 1)],
    ]),
    varyings: new Map([['t', vg(0, 0, 1, 1)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform float a;
     uniform float b;
     varying float t;
     void main() { gl_FragColor.x = dFdx(mix(a, b, t)); }`,
    layout,
    { uniforms: new Float32Array([1.0, 3.0]), varyings: [{ v: new Float32Array([0.25]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0.0]) }] },
  );
  near(r.ctx.out.color[0][0], (3 - 1) * 2, `dFdx(mix(a,b,t)) == (b-a)*tdx + mix of endpoint duals`);
}

/* ================================================================== */
/* 3. clamp: in-range → v.ddx; out-of-range → 0                       */
/* ================================================================== */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const src = `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = dFdx(clamp(v, 0.0, 1.0)); }`;
  for (const [val, expected] of [
    [0.5, 2.0],
    [1.5, 0.0],
    [-0.5, 0.0],
  ] as [number, number][]) {
    const r = runDual(src, layout, {
      varyings: [{ v: new Float32Array([val]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0.0]) }],
    });
    near(r.ctx.out.color[0][0], expected, `dFdx(clamp(v,0,1)) with v=${val}`);
  }
}

/* ================================================================== */
/* 4. step (0 a.e.), fract (x'), sign/floor (0)                       */
/* ================================================================== */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() {
       gl_FragColor.x = dFdx(step(0.5, v));
       gl_FragColor.y = dFdx(fract(v));
       gl_FragColor.z = dFdx(sign(v));
       gl_FragColor.w = dFdx(floor(v));
     }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 0, `dFdx(step(0.5, v)) == 0 (got ${c[0]})`);
  near(c[1], 2.0, `dFdx(fract(v)) == v.ddx`);
  check(c[2] === 0, `dFdx(sign(v)) == 0 (got ${c[2]})`);
  check(c[3] === 0, `dFdx(floor(v)) == 0 (got ${c[3]})`);
}

/* ================================================================== */
/* 5. smoothstep: 6t(1-t)*x'/(e1-e0)                                  */
/* ================================================================== */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = dFdx(smoothstep(0.0, 1.0, v)); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0.0]) }] },
  );
  near(r.ctx.out.color[0][0], 6 * 0.5 * (1 - 0.5) * 2, `dFdx(smoothstep(0,1,v)) == 6t(1-t)*ddx`);
}

/* ================================================================== */
/* 6. atan(y, x): (x*ydx - y*xdx)/(x²+y²)                             */
/* ================================================================== */

{
  const layout = baseLayout(100, {
    varyings: new Map([
      ['y', vg(0, 0, 1, 1)],
      ['x', vg(1, 0, 1, 1)],
    ]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float y;
     varying float x;
     void main() { gl_FragColor.x = dFdx(atan(y, x)); }`,
    layout,
    {
      varyings: [
        { v: new Float32Array([1.0]), ddx: new Float32Array([2.0]), ddy: new Float32Array([0.0]) },
        { v: new Float32Array([1.0]), ddx: new Float32Array([0.0]), ddy: new Float32Array([0.0]) },
      ],
    },
  );
  near(r.ctx.out.color[0][0], (1 * 2 - 1 * 0) / (1 + 1), `dFdx(atan(y,x)) == (x*ydx - y*xdx)/(x²+y²)`);
}

/* ================================================================== */
/* 7. fwidth of a builtin: |cos(v)*ddx| + |cos(v)*ddy|                */
/* ================================================================== */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = fwidth(sin(v)); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([-3.0]) }] },
  );
  near(r.ctx.out.color[0][0], Math.abs(Math.cos(0.5) * 2) + Math.abs(Math.cos(0.5) * -3), `fwidth(sin(v))`);
}

/* ================================================================== */
/* 8. Geometry via vec2 VARYINGS (no ctors): length/dot/distance      */
/* ================================================================== */

{
  const layout = baseLayout(100, {
    varyings: new Map([
      ['vv', vg(0, 0, 2, 2)],
      ['ww', vg(1, 0, 2, 2)],
    ]),
  });
  const src = `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying vec2 vv;
     varying vec2 ww;
     void main() {
       gl_FragColor.x = dFdx(length(vv));
       gl_FragColor.y = dFdx(dot(vv, ww));
       gl_FragColor.z = dFdx(distance(vv, ww));
     }`;
  const ctxExtra = {
    varyings: [
      { v: new Float32Array([3.0, 4.0]), ddx: new Float32Array([1.0, 2.0]), ddy: new Float32Array([0.0, 0.0]) },
      { v: new Float32Array([1.0, 0.0]), ddx: new Float32Array([0.0, 0.0]), ddy: new Float32Array([0.0, 0.0]) },
    ],
  };
  const r = runDual(src, layout, ctxExtra);
  const c = r.ctx.out.color[0];
  near(c[0], (3 * 1 + 4 * 2) / 5, `dFdx(length(vv)) == (Σ vi·dvi)/len`);
  near(c[1], 1 * 1 + 3 * 0 + 2 * 0 + 4 * 0, `dFdx(dot(vv,ww)) == Σ (dvi·wi + vi·dwi)`);
  near(c[2], ((3 - 1) * 1 + (4 - 0) * 2) / Math.sqrt(20), `dFdx(distance(vv,ww)) == (a−b)·(da−db)/len`);
}

/* ================================================================== */
/* 9. normalize: (vi'·len − vi·len')/len²                             */
/* ================================================================== */

{
  const layout = baseLayout(100, { varyings: new Map([['vv', vg(0, 0, 2, 2)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying vec2 vv;
     void main() {
       vec2 n = normalize(vv);
       gl_FragColor.x = dFdx(n.x);
       gl_FragColor.y = dFdx(n.y);
     }`,
    layout,
    { varyings: [{ v: new Float32Array([3.0, 4.0]), ddx: new Float32Array([1.0, 2.0]), ddy: new Float32Array([0.0, 0.0]) }] },
  );
  const c = r.ctx.out.color[0];
  near(c[0], (1 * 5 - 3 * (11 / 5)) / 25, `dFdx(normalize(vv).x) == (vx'·len − vx·len')/len²`);
  near(c[1], (2 * 5 - 4 * (11 / 5)) / 25, `dFdx(normalize(vv).y)`);
}

/* ================================================================== */
/* 10. reflect: I' − 2·(dot'·N + dot·N')                              */
/* ================================================================== */

{
  const layout = baseLayout(100, {
    varyings: new Map([
      ['ii', vg(0, 0, 2, 2)],
      ['nn', vg(1, 0, 2, 2)],
    ]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying vec2 ii;
     varying vec2 nn;
     void main() {
       vec2 rv = reflect(ii, nn);
       gl_FragColor.x = dFdx(rv.x);
       gl_FragColor.y = dFdx(rv.y);
     }`,
    layout,
    {
      varyings: [
        { v: new Float32Array([1.0, 1.0]), ddx: new Float32Array([2.0, 3.0]), ddy: new Float32Array([0.0, 0.0]) },
        { v: new Float32Array([0.0, 1.0]), ddx: new Float32Array([0.0, 0.0]), ddy: new Float32Array([0.0, 0.0]) },
      ],
    },
  );
  const c = r.ctx.out.color[0];
  near(c[0], 2.0, `dFdx(reflect(ii,nn).x) — dot=1, dot'=3 → 2 − 2·(3·0 + 1·0)`);
  near(c[1], -3.0, `dFdx(reflect(ii,nn).y) — 3 − 2·(3·1 + 1·0)`);
}

/* ================================================================== */
/* 11. Component-wise pass-through on a vec2 varying component read    */
/* ================================================================== */

{
  const layout = baseLayout(100, { varyings: new Map([['vv', vg(0, 0, 2, 2)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying vec2 vv;
     void main() { gl_FragColor.x = dFdx(abs(vv.x)); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5, 0.5]), ddx: new Float32Array([2.0, 3.0]), ddy: new Float32Array([0.0, 0.0]) }] },
  );
  near(r.ctx.out.color[0][0], 2.0, `dFdx(abs(vv.x)) applies the scalar template per component`);
}

/* ================================================================== */
/* 12. Texture gradient routing — 1.00 texture2D (implicit LOD)       */
/* ================================================================== */

{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['u', slot('int', 0)]]),
    varyings: new Map([['uv', vg(0, 0, 2, 2)]]),
  });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform sampler2D u;
     varying vec2 uv;
     void main() {
       vec4 tc = texture2D(u, uv);
       gl_FragColor = tc;
       gl_FragColor.y = dFdx(tc.x);
     }`,
    layout,
    {
      intUniforms: Int32Array.from([0]),
      varyings: [{ v: new Float32Array([0.25, 0.5]), ddx: new Float32Array([2.0, 3.0]), ddy: new Float32Array([4.0, 5.0]) }],
      tex: makeTexStub(),
    },
  );
  const c = r.ctx.out.color[0];
  near(c[0], 0.25, `texture2D implicit-LOD: coord u passes through`);
  check(c[1] === 0, `texture result duals are zeroed: dFdx(tc.x) == 0 (got ${c[1]})`);
  near(c[2], 10 * 2 + 20 * 3 + 30 * 4 + 40 * 5, `texture2D routes coord duals into the gradient slots`);
  near(c[3], 0, `texture2D bias slot stays 0`);
  check(
    r.body.includes(
      'ctx.tex.sample2D(ctx.intUniforms[0 + 0], ctx.varyings[0].v[0], ctx.varyings[0].v[1], ' +
        'ctx.varyings[0].ddx[0], ctx.varyings[0].ddx[1], ctx.varyings[0].ddy[0], ctx.varyings[0].ddy[1], 0)',
    ),
    `body pins the exact dual-routed sample2D call (got: ${r.body.slice(0, 200)}...)`,
  );
}

/* ================================================================== */
/* 13. ES 3.00 texture (sampler2D) — same routing                     */
/* ================================================================== */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([['u', slot('int', 0)]]),
    varyings: new Map([['uv', vg(0, 0, 2, 2)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     uniform sampler2D u;
     in vec2 uv;
     out vec4 oColor;
     void main() { vec4 tc = texture(u, uv); oColor = tc; }`,
    layout,
    {
      intUniforms: Int32Array.from([0]),
      varyings: [{ v: new Float32Array([0.25, 0.5]), ddx: new Float32Array([2.0, 3.0]), ddy: new Float32Array([4.0, 5.0]) }],
      tex: makeTexStub(),
    },
  );
  const c = r.ctx.out.color[0];
  near(c[0], 0.25, `ES3 texture: coord u`);
  near(c[1], 0.5, `ES3 texture: coord v`);
  near(c[2], 10 * 2 + 20 * 3 + 30 * 4 + 40 * 5, `ES3 texture routes coord duals into the gradient slots`);
  check(
    r.body.includes(
      'ctx.tex.sample2D(ctx.intUniforms[0 + 0], ctx.varyings[0].v[0], ctx.varyings[0].v[1], ' +
        'ctx.varyings[0].ddx[0], ctx.varyings[0].ddx[1], ctx.varyings[0].ddy[0], ctx.varyings[0].ddy[1], 0)',
    ),
    `ES3 texture body pins the dual-routed sample2D call`,
  );
}

/* ================================================================== */
/* 14. ES 3.00 texture (sampler3D) — 6 gradient slots                 */
/* ================================================================== */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([['u', slot('int', 0)]]),
    varyings: new Map([['P', vg(0, 0, 3, 3)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     uniform sampler3D u;
     in vec3 P;
     out vec4 oColor;
     void main() { vec4 tc = texture(u, P); oColor = tc; }`,
    layout,
    {
      intUniforms: Int32Array.from([0]),
      varyings: [{ v: new Float32Array([0.25, 0.5, 0.75]), ddx: new Float32Array([2.0, 3.0, 4.0]), ddy: new Float32Array([5.0, 6.0, 7.0]) }],
      tex: makeTexStub(),
    },
  );
  const c = r.ctx.out.color[0];
  near(c[0], 0.25, `ES3 texture3D: coord u`);
  near(c[1], 0.5, `ES3 texture3D: coord v`);
  near(c[2], 10 * 2 + 20 * 3 + 30 * 4 + 40 * 5 + 50 * 6 + 60 * 7, `ES3 texture3D routes all 6 coord duals`);
  check(
    r.body.includes(
      'ctx.tex.sample3D(ctx.intUniforms[0 + 0], ctx.varyings[0].v[0], ctx.varyings[0].v[1], ctx.varyings[0].v[2], ' +
        'ctx.varyings[0].ddx[0], ctx.varyings[0].ddx[1], ctx.varyings[0].ddx[2], ' +
        'ctx.varyings[0].ddy[0], ctx.varyings[0].ddy[1], ctx.varyings[0].ddy[2], 0)',
    ),
    `ES3 texture3D body pins the dual-routed sample3D call`,
  );
}

/* ================================================================== */
/* 15. ES 3.00 texture (sampler2DArray) — layer is value-only         */
/* ================================================================== */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([['u', slot('int', 0)]]),
    varyings: new Map([['P', vg(0, 0, 3, 3)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     uniform sampler2DArray u;
     in vec3 P;
     out vec4 oColor;
     void main() { vec4 tc = texture(u, P); oColor = tc; }`,
    layout,
    {
      intUniforms: Int32Array.from([0]),
      varyings: [{ v: new Float32Array([0.25, 0.5, 0.75]), ddx: new Float32Array([2.0, 3.0, 4.0]), ddy: new Float32Array([5.0, 6.0, 7.0]) }],
      tex: makeTexStub(),
    },
  );
  const c = r.ctx.out.color[0];
  near(c[0], 0.25, `ES3 texture2DArray: coord u`);
  near(c[2], 0.75, `ES3 texture2DArray: layer passes as value`);
  near(c[3], 10 * 2 + 20 * 3 + 30 * 5 + 40 * 6, `ES3 texture2DArray routes the 2D coord duals (layer unfiltered)`);
  check(
    r.body.includes(
      'ctx.tex.sample2DArray(ctx.intUniforms[0 + 0], ctx.varyings[0].v[0], ctx.varyings[0].v[1], ctx.varyings[0].v[2], ' +
        'ctx.varyings[0].ddx[0], ctx.varyings[0].ddx[1], ctx.varyings[0].ddy[0], ctx.varyings[0].ddy[1], 0)',
    ),
    `ES3 texture2DArray body pins the dual-routed sample2DArray call`,
  );
}

/* ================================================================== */
/* 16. ES 3.00 textureProj — quotient-rule duals on the divided coords */
/* ================================================================== */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([['u', slot('int', 0)]]),
    varyings: new Map([['P', vg(0, 0, 3, 3)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     uniform sampler2D u;
     in vec3 P;
     out vec4 oColor;
     void main() { vec4 tc = textureProj(u, P); oColor = tc; }`,
    layout,
    {
      intUniforms: Int32Array.from([0]),
      varyings: [{ v: new Float32Array([0.25, 0.5, 2.0]), ddx: new Float32Array([2.0, 3.0, 4.0]), ddy: new Float32Array([5.0, 6.0, 7.0]) }],
      tex: makeTexStub(),
    },
  );
  const c = r.ctx.out.color[0];
  const dux = (2 * 2 - 0.25 * 4) / 4;
  const dvx = (3 * 2 - 0.5 * 4) / 4;
  const duy = (5 * 2 - 0.25 * 7) / 4;
  const dvy = (6 * 2 - 0.5 * 7) / 4;
  near(c[0], 0.125, `textureProj: divided u`);
  near(c[1], 0.25, `textureProj: divided v`);
  near(c[2], 10 * dux + 20 * dvx + 30 * duy + 40 * dvy, `textureProj routes quotient-rule duals (dux=${dux}, dvx=${dvx}, duy=${duy}, dvy=${dvy})`);
  check(
    r.body.includes('ctx.tex.sample2D(') && r.body.includes('ctx.varyings[0].ddx[0]'),
    `textureProj body goes through the dual-routed sample2D`,
  );
}

/* ================================================================== */
/* 17. ES 3.00 sampler2DShadow texture — ref is value-only            */
/* ================================================================== */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([['s', slot('int', 0)]]),
    varyings: new Map([['P', vg(0, 0, 3, 3)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     uniform sampler2DShadow s;
     in vec3 P;
     out vec4 oColor;
     void main() { float rr = texture(s, P); oColor.x = rr; }`,
    layout,
    {
      intUniforms: Int32Array.from([0]),
      varyings: [{ v: new Float32Array([0.25, 0.5, 0.75]), ddx: new Float32Array([2.0, 3.0, 4.0]), ddy: new Float32Array([5.0, 6.0, 7.0]) }],
      tex: makeTexStub(),
    },
  );
  near(
    r.ctx.out.color[0][0],
    0.75 + 10 * 2 + 20 * 3 + 30 * 5 + 40 * 6,
    `texture(sampler2DShadow): ref + routed coord duals`,
  );
  check(
    r.body.includes(
      'ctx.tex.sample2DShadow(ctx.intUniforms[0 + 0], ctx.varyings[0].v[0], ctx.varyings[0].v[1], ctx.varyings[0].v[2], ' +
        'ctx.varyings[0].ddx[0], ctx.varyings[0].ddx[1], ctx.varyings[0].ddy[0], ctx.varyings[0].ddy[1], 0)',
    ),
    `sampler2DShadow body pins the dual-routed sample2DShadow call`,
  );
}

/* ================================================================== */
/* 18. Non-derivative builtins in dual mode must NOT throw             */
/*     (pack/unpack are v-only; textureSize/texelFetch carry no duals) */
/* ================================================================== */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([
      ['u', slot('int', 0)],
      ['is', slot('int', 1)],
    ]),
    varyings: new Map([['vv', vg(0, 0, 2, 2)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  let body = '';
  let threw = '';
  try {
    body = compileOnly(
      `#version 300 es
       precision mediump float;
       in vec2 vv;
       uniform sampler2D u;
       uniform isampler2D is;
       out vec4 oColor;
       void main() {
         uint p = packUnorm2x16(vv);
         vec2 up = unpackHalf2x16(p);
         ivec2 sz = textureSize(u, 0);
         int s = sz.x;
         ivec2 P = ivec2(gl_FragCoord.xy);
         ivec4 tv = texelFetch(is, P, 0);
         oColor.x = float(s);
         oColor.y = float(tv.x);
       }`,
      layout,
    );
  } catch (e) {
    threw = String(e);
  }
  check(threw === '', `pack/unpack/textureSize/texelFetch compile in dual mode (got: ${threw || 'no throw'})`);
  check(body.includes('R.packUnorm2x16('), `pack lowers through R.packUnorm2x16`);
  check(body.includes('R.unpackHalf2x16('), `unpack lowers through R.unpackHalf2x16 (v-only dual case)`);
  check(body.includes('R.textureSize('), `textureSize lowers through R.textureSize`);
  check(body.includes('ctx.tex.texelFetch2D('), `texelFetch lowers through ctx.tex.texelFetch2D`);
}

/* ================================================================== */
/* 19. textureGrad + EXT_shader_texture_lod functions in dual mode     */
/*     (explicit gradients/LOD win — coord duals ignored, no throw)   */
/* ================================================================== */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([['u', slot('int', 0)]]),
    varyings: new Map([
      ['uv', vg(0, 0, 2, 2)],
      ['dpx', vg(1, 0, 2, 2)],
      ['dpy', vg(2, 0, 2, 2)],
    ]),
    outputLocations: new Map([['oColor', 0]]),
  });
  let body = '';
  let threw = '';
  try {
    body = compileOnly(
      `#version 300 es
       precision mediump float;
       uniform sampler2D u;
       in vec2 uv;
       in vec2 dpx;
       in vec2 dpy;
       out vec4 oColor;
       void main() { vec4 tc = textureGrad(u, uv, dpx, dpy); oColor = tc; }`,
      layout,
    );
  } catch (e) {
    threw = String(e);
  }
  check(threw === '', `textureGrad compiles in dual mode (coord duals ignored, explicit gradients win) (got: ${threw || 'no throw'})`);
  check(body.includes('tex2DGrad'), `textureGrad lowers through R.tex2DGrad`);
}

{
  const layout = baseLayout(100, {
    uniformSlots: new Map([
      ['s', slot('int', 0)],
      ['sc', slot('int', 1)],
    ]),
    varyings: new Map([
      ['uv', vg(0, 0, 2, 2)],
      ['dpx', vg(1, 0, 2, 2)],
      ['dpy', vg(2, 0, 2, 2)],
      ['P', vg(3, 0, 3, 3)],
      ['dq', vg(4, 0, 3, 3)],
    ]),
  });
  let body = '';
  let threw = '';
  try {
    body = compileOnly(
      `#extension GL_OES_standard_derivatives : enable
       #extension GL_EXT_shader_texture_lod : enable
       precision mediump float;
       uniform sampler2D s;
       uniform samplerCube sc;
       varying vec2 uv;
       varying vec2 dpx;
       varying vec2 dpy;
       varying vec3 P;
       varying vec3 dq;
       void main() {
         vec4 a = texture2DGradEXT(s, uv, dpx, dpy);
         vec4 b = texture2DProjGradEXT(s, P, dpx, dpy);
         vec4 c = textureCubeGradEXT(sc, P, dq, dq);
         gl_FragColor = a;
       }`,
      layout,
      ['GL_EXT_shader_texture_lod'],
    );
  } catch (e) {
    threw = String(e);
  }
  check(threw === '', `EXT_shader_texture_lod Grad functions compile in dual mode (got: ${threw || 'no throw'})`);
  check(body.includes('tex2DGrad') && body.includes('texCubeGrad'), `GradEXT lowers through R.tex2DGrad/R.texCubeGrad`);
}

/* ================================================================== */
/* 20. Matrix builtins THROW in dual mode (no derivative template)    */
/* ================================================================== */

{
  const layout = baseLayout(100, { uniformSlots: new Map([['ma', slot('float', 0)]]) });
  expectDualThrow(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform mat2 ma;
     void main() { mat2 d = inverse(ma); gl_FragColor.x = dFdx(d[0][0]); }`,
    layout,
    'inverse',
  );
  expectDualThrow(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform mat2 ma;
     void main() { float dd = determinant(ma); gl_FragColor.x = dFdx(dd); }`,
    layout,
    'determinant',
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-dual-builtins selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
