/**
 * selftest-dual.ts — dual-number FRAGMENT codegen checks (task C5a1).
 *
 * In dual mode (layout.uses.derivatives → env.dual) every FLOAT value
 * carries (v, dx, dy) — value + screen-space derivatives as JS expression
 * strings. This selftest pins the STORAGE + READS + WRITES + dFdx/dFdy/
 * fwidth layer ONLY: value flow restricted to what works WITHOUT the C5a2
 * arithmetic dual templates (assignments from varying/uniform/gl_FragCoord
 * reads, dFdx/dFdy/fwidth of such values, compound +=/-=, ++/--, inout/out
 * params, local arrays). Arithmetic/constructors/ternaries in dual mode
 * throw "C5a2" errors (not exercised here).
 *
 * Each shader compiles (compileShader → annotated AST), lowers through
 * generateFragmentStage (env.dual = layout.uses.derivatives), runs via
 * `new Function('ctx','R', body)`, and the resulting ctx.out.color is
 * asserted. ctx.varyings entries carry v + ddx + ddy (the raster supplies
 * the derivative arrays whenever usesDerivatives — the generated code reads
 * them directly, no guards).
 *
 * Run: npx tsx src/glsl/selftest-dual.ts
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
    derivatives: true,
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
  scratchSize: number;
  intScratchSize: number;
}

function runDual(
  src: string,
  layout: CodegenLayout,
  ctxExtra: Record<string, unknown> = {},
): DualRunResult {
  const r = compileShader(src, {
    type: 'FRAGMENT',
    version: layout.version,
    extensions: new Set(['GL_OES_standard_derivatives']),
  });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const res = generateFragmentStage(r.shader.ast as TranslationUnit, layout);
  const fn = new Function('ctx', 'R', res.body);
  const ctx = { ...fragCtx(res), ...ctxExtra };
  fn(ctx, R);
  return { body: res.body, ctx, scratchSize: res.scratchSize, intScratchSize: res.intScratchSize };
}

/* ------------------------------------------------------------------ */
/* 1. dFdx/dFdy of a (non-flat) float varying == the raster ddx/ddy    */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = dFdx(v); gl_FragColor.y = dFdy(v); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 2.0 && c[1] === 3.0, `dFdx(v)==ddx, dFdy(v)==ddy (got [${Array.from(c).join(', ')}])`);
  check(
    r.body.includes('ctx.varyings[0].ddx[0]') && r.body.includes('ctx.varyings[0].ddy[0]'),
    `dual varying reads use direct .ddx/.ddy access (body: ${r.body.slice(0, 120)}...)`,
  );
}

/* ------------------------------------------------------------------ */
/* 2. FLAT varying: dFdx/dFdy must ignore the supplied ddx/ddy → 0     */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1, true)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = dFdx(v); gl_FragColor.y = dFdy(v); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([9.0]), ddy: new Float32Array([9.0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 0 && c[1] === 0, `flat varying: dFdx/dFdy == 0 (got [${Array.from(c).join(', ')}])`);
}

/* ------------------------------------------------------------------ */
/* 3. dFdx(uniform) == 0 (constant duals)                              */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { uniformSlots: new Map([['u', slot('float', 0)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform float u;
     void main() { gl_FragColor.x = dFdx(u); }`,
    layout,
    { uniforms: new Float32Array([7.0]) },
  );
  check(r.ctx.out.color[0][0] === 0, `dFdx(uniform) == 0 (got ${r.ctx.out.color[0][0]})`);
}

/* ------------------------------------------------------------------ */
/* 4. gl_FragCoord constant duals: dFdx(x)==1, dFdy(y)==1              */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100);
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     void main() { gl_FragColor.x = dFdx(gl_FragCoord.x); gl_FragColor.y = dFdy(gl_FragCoord.y); }`,
    layout,
    { fragCoord: new Float32Array([10, 20, 0.5, 1]) },
  );
  const c = r.ctx.out.color[0];
  check(c[0] === 1 && c[1] === 1, `dFdx(gl_FragCoord.x)==1, dFdy(gl_FragCoord.y)==1 (got [${Array.from(c).join(', ')}])`);
}

/* ------------------------------------------------------------------ */
/* 5. Local chain: assignment writes the triple                        */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t = v; gl_FragColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `'float t = v; dFdx(t)' == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
  check(
    r.body.includes('var t = ctx.varyings[0].v[0], t_dx = ctx.varyings[0].ddx[0], t_dy = ctx.varyings[0].ddy[0];'),
    `decl init emits the triple var line (body: ${r.body.slice(0, 150)}...)`,
  );
}

/* ------------------------------------------------------------------ */
/* 6. Compound += updates all three planes                             */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t = v; t += v; gl_FragColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 4.0,
    `'t += v; dFdx(t)' == 2*v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 7. ++ on a float local: constant delta — duals unchanged            */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { float t = v; ++t; gl_FragColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `'++t; dFdx(t)' == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 8. inout param (ES 3.00): write-back carries the param's triple     */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(300, {
    varyings: new Map([['v', vg(0, 0, 1, 1)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     in float v;
     out vec4 oColor;
     void bump(inout float x) { x += v; }
     void main() { float t = v; bump(t); oColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 4.0,
    `inout param write-back: 'bump(t); dFdx(t)' == 2*v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 9. out param (ES 3.00): write-back carries the param's triple       */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(300, {
    varyings: new Map([['v', vg(0, 0, 1, 1)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runDual(
    `#version 300 es
     precision mediump float;
     in float v;
     out vec4 oColor;
     void setv(out float x) { x = v; }
     void main() { float t = 0.0; setv(t); oColor.x = dFdx(t); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `out param write-back: 'setv(t); dFdx(t)' == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 10. Local ARRAY: 3-plane scratch, dynamic index                     */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['u_i', slot('int', 0)]]),
    varyings: new Map([['v', vg(0, 0, 1, 1)]]),
  });
  const src = `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform int u_i;
     varying float v;
     void main() { float a[2]; a[0] = v; a[1] = v; gl_FragColor.x = dFdx(a[u_i]); }`;
  for (const i of [0, 1]) {
    const r = runDual(src, layout, {
      intUniforms: Int32Array.from([i]),
      varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }],
    });
    check(
      r.ctx.out.color[0][0] === 2.0,
      `local array a[${i}] with dynamic index: dFdx(a[u_i]) == v.ddx (got ${r.ctx.out.color[0][0]})`,
    );
    if (i === 0) {
      check(
        r.scratchSize === 6,
        `dual float array scratch is 3 planes × 2 elements (got scratchSize ${r.scratchSize})`,
      );
    }
  }
}

/* ------------------------------------------------------------------ */
/* 11. fwidth == |ddx| + |ddy|                                         */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { gl_FragColor.x = fwidth(v); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([-3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 5.0,
    `fwidth(v) == |ddx| + |ddy| (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 12. vec2 local via member writes (no ctor — C5a2 scope)             */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['v', vg(0, 0, 1, 1)]]) });
  const r = runDual(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     varying float v;
     void main() { vec2 p; p.x = v; p.y = v; gl_FragColor.x = dFdx(p.x); }`,
    layout,
    { varyings: [{ v: new Float32Array([0.5]), ddx: new Float32Array([2.0]), ddy: new Float32Array([3.0]) }] },
  );
  check(
    r.ctx.out.color[0][0] === 2.0,
    `vec2 local member writes: dFdx(p.x) == v.ddx (got ${r.ctx.out.color[0][0]})`,
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-dual selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
