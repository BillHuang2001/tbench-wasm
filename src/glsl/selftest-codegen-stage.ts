/**
 * selftest-codegen-stage.ts — stage-assembly checks for codegen/vertex.ts +
 * codegen/fragment.ts (task C4): generateVertexStage / generateFragmentStage
 * lower a FULLY COMPILED shader (compileShader → annotated AST) into a JS
 * body (`new Function('ctx','R', body)`), and the resulting ctx state is
 * asserted. Covers the complete storage surface:
 *
 * VERTEX:   attributes (vec4 + scalar CONSTANT attrib via the typeof guard,
 *           mat4 occupying consecutive locations), default-block uniforms
 *           (float/int/uint incl. the >>> 0 wrap), varying writes (incl.
 *           arrays + dynamic index), gl_PointSize / gl_VertexID /
 *           gl_InstanceID, a user function (inliner integration), scratch
 *           size reporting.
 * FRAGMENT: varying reads (incl. array + const indices), gl_FragCoord /
 *           gl_FrontFacing / gl_PointCoord, ES 3.00 `out` + gl_FragDepth,
 *           texture2D implicit-LOD (→ ctx.tex.sample2D + ctx.tex.out).
 * ROUND-TRIP: vertex packs vColor at VaryingLayout.offset, fragment reads it
 *           back via VaryingLayout.index — the packed-varying convention is
 *           validated end-to-end.
 * EXTRA:    ES 3.00 varying INTERFACE BLOCKS (`out VS_OUT { vec4 c; } vs_out;`
 *           / `in VS_OUT { vec4 c; } vs_in;`) — member paths under an
 *           instance name resolve via layout.varyings keys 'vs_out.c' /
 *           'vs_in.c' (env.globalInfo's prefix scan + per-member varying
 *           layouts; semantics registers the instance as a struct-typed var).
 *
 * Run: npx tsx src/glsl/selftest-codegen-stage.ts
 *
 * Prints "OK" and exits 0 on success.
 */
import type { TranslationUnit } from './ast.js';
import { compileShader } from './compiler.js';
import type { ShaderUses } from './compiler.js';
import { generateVertexStage, generateFragmentStage } from './codegen/index.js';
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
/* Layout helpers                                                      */
/* ------------------------------------------------------------------ */

function uses(): ShaderUses {
  return {
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
    uses: uses(),
    ...extra,
  };
}

const slot = (store: 'float' | 'int', s: number, stride = 0): UniformSlot => ({ store, slot: s, stride });
const vg = (index: number, offset: number, components: number, elemComponents: number, flat = false): VaryingLayout =>
  ({ index, offset, components, elemComponents, flat });

/* ------------------------------------------------------------------ */
/* Driver: compileShader → generateStage → new Function → run          */
/* ------------------------------------------------------------------ */

function compile(src: string, stage: 'VERTEX' | 'FRAGMENT', version: 100 | 300, exts?: string[]): TranslationUnit {
  const r = compileShader(src, { type: stage, version, extensions: exts !== undefined ? new Set(exts) : undefined });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  return r.shader.ast;
}

interface RunResult {
  body: string;
  ctx: Record<string, any>;
  scratchSize: number;
  intScratchSize: number;
}

/** Compile + generate + run one stage; ctxExtra overrides the default ctx. */
function runStage(
  src: string,
  stage: 'VERTEX' | 'FRAGMENT',
  layout: CodegenLayout,
  ctxExtra: Record<string, unknown> = {},
  exts?: string[],
): RunResult {
  const ast = compile(src, stage, layout.version, exts);
  const res =
    stage === 'VERTEX' ? generateVertexStage(ast, layout) : generateFragmentStage(ast, layout);
  const fn = new Function('ctx', 'R', res.body);
  const ctx: Record<string, any> = {
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(res.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(res.intScratchSize, 16)),
    out:
      stage === 'VERTEX'
        ? { position: new Float32Array(4), pointSize: 0, varyings: new Float32Array(32) }
        : { color: [new Float32Array(4)], fragDepth: 0 },
    ...(stage === 'VERTEX'
      ? { attribs: [], attribIndices: new Int32Array(16), vertexId: 0, instanceId: 0 }
      : { discarded: false, fragCoord: new Float32Array([0, 0, 0, 0]), frontFacing: true, pointCoord: new Float32Array([0, 0]), varyings: [] }),
    ...ctxExtra,
  };
  fn(ctx, R);
  return { body: res.body, ctx, scratchSize: res.scratchSize, intScratchSize: res.intScratchSize };
}

/** A stub fragment texture unit: sample2D writes ctx.tex.out (float sampler). */
function makeTex(): any {
  const tex: any = { out: new Float32Array(4) };
  tex.sample2D = (unit: number, u: number, v: number): boolean => {
    tex.out[0] = 1;
    tex.out[1] = 0.5;
    tex.out[2] = 0.25;
    tex.out[3] = 1;
    return true;
  };
  return tex;
}

/* ------------------------------------------------------------------ */
/* 1. VERTEX: attributes — vec4 + CONSTANT scalar (typeof guard)       */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { attribLocations: new Map([['aPos', 0], ['aScale', 1]]) });
  const r = runStage(
    `attribute vec4 aPos;
     attribute float aScale;
     void main() { gl_Position = vec4(aPos.xyz * aScale, 1.0); }`,
    'VERTEX',
    layout,
    { attribs: [new Float32Array([1, 2, 3, 4]), 2.0], attribIndices: new Int32Array([0, 0]) },
  );
  const p = r.ctx.out.position;
  check(
    p[0] === 2 && p[1] === 4 && p[2] === 6 && p[3] === 1,
    `vertex attribs: aPos * 2.0 → [2,4,6,1] (got [${Array.from(p).join(', ')}])`,
  );
  check(
    r.body.includes("typeof ctx.attribs[1] === 'number'"),
    `constant attrib emits the typeof guard (body: ${r.body.slice(0, 80)}...)`,
  );
}

/* ------------------------------------------------------------------ */
/* 2. VERTEX: mat4 attribute (4 consecutive locations, column-major)   */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { attribLocations: new Map([['aM', 2]]) });
  // M = translate(4,5,6): column c at attribs[2+c], 4 floats each.
  const r = runStage(
    `attribute mat4 aM;
     void main() { gl_Position = aM * vec4(1.0, 2.0, 3.0, 1.0); }`,
    'VERTEX',
    layout,
    {
      attribs: [
        undefined,
        undefined,
        new Float32Array([1, 0, 0, 0]),
        new Float32Array([0, 1, 0, 0]),
        new Float32Array([0, 0, 1, 0]),
        new Float32Array([4, 5, 6, 1]),
      ],
      attribIndices: new Int32Array([0, 0, 0, 0, 0, 0]),
    },
  );
  const p = r.ctx.out.position;
  check(
    p[0] === 5 && p[1] === 7 && p[2] === 9 && p[3] === 1,
    `mat4 attrib: M*(1,2,3,1) → [5,7,9,1] (got [${Array.from(p).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* 3. VERTEX: default-block uniforms (float / int / uint stores)       */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(300, {
    uniformSlots: new Map([
      ['uColor', slot('float', 0)],
      ['uI', slot('int', 4)],
      ['uU', slot('int', 6)],
    ]),
  });
  const r = runStage(
    `#version 300 es
     uniform vec4 uColor;
     uniform ivec2 uI;
     uniform uint uU;
     void main() { gl_Position = vec4(uColor.rgb + vec3(float(uI.x), float(uI.y), float(uU)), uColor.a); }`,
    'VERTEX',
    layout,
    {
      uniforms: new Float32Array([0.25, 0.5, 0.75, 1.0]),
      intUniforms: Int32Array.from([0, 0, 0, 0, -2, 3, -1]), // uI at 4,5; uU at 6 (0xFFFFFFFF bits)
    },
  );
  const p = r.ctx.out.position;
  check(
    p[0] === -1.75 && p[1] === 3.5 && p[3] === 1,
    `float/int/uint uniform reads: x=-1.75, y=3.5, w=1 (got [${Array.from(p).join(', ')}])`,
  );
  check(p[2] === 4294967296, `uint uniform wraps to 4294967295 (float32-stored as 2^32; got ${p[2]})`);
  check(
    r.body.includes('ctx.intUniforms[4 + 0]'),
    `ivec2 uniform reads ctx.intUniforms[4 + 0] (body: ${r.body.slice(0, 80)}...)`,
  );
  check(r.body.includes('>>> 0'), `uint uniform read wraps with '>>> 0'`);
}

/* ------------------------------------------------------------------ */
/* 4. VERTEX: varying writes (array + dynamic index)                   */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, {
    varyings: new Map([
      ['vColor', vg(0, 0, 4, 4)],
      ['vF', vg(1, 4, 3, 1)],
    ]),
  });
  const r = runStage(
    `varying vec4 vColor;
     varying float vF[3];
     void main() {
       vColor = vec4(1.0, 2.0, 3.0, 4.0);
       vF[0] = 10.0; vF[1] = 11.0; vF[2] = 12.0;
       gl_Position = vec4(0.0);
     }`,
    'VERTEX',
    layout,
  );
  const v = r.ctx.out.varyings;
  check(
    v[0] === 1 && v[1] === 2 && v[2] === 3 && v[3] === 4,
    `varying vec4 write → out.varyings[0..3] (got [${Array.from(v.slice(0, 4)).join(', ')}])`,
  );
  check(
    v[4] === 10 && v[5] === 11 && v[6] === 12,
    `array varying vF[3] → out.varyings[4..6] (got [${Array.from(v.slice(4, 7)).join(', ')}])`,
  );
}
{
  const layout = baseLayout(100, { varyings: new Map([['vF', vg(0, 2, 3, 1)]]) });
  const r = runStage(
    `varying float vF[3];
     void main() { int i = 2; vF[i] = 7.0; gl_Position = vec4(0.0); }`,
    'VERTEX',
    layout,
  );
  check(
    r.ctx.out.varyings[2 + 2] === 7,
    `dynamic varying index write → out.varyings[4] === 7 (got ${r.ctx.out.varyings[4]})`,
  );
}

/* ------------------------------------------------------------------ */
/* 5. VERTEX: gl_PointSize + gl_VertexID + gl_InstanceID               */
/* ------------------------------------------------------------------ */

{
  // gl_VertexID / gl_InstanceID are ES 3.00 builtins; gl_DrawID is
  // extension-gated in ES 3.00 too (GL_ANGLE_multi_draw).
  const r = runStage(
    `#version 300 es
     #extension GL_ANGLE_multi_draw : require
     void main() { gl_PointSize = 3.0; gl_Position = vec4(float(gl_VertexID), float(gl_InstanceID), float(gl_DrawID), 1.0); }`,
    'VERTEX',
    baseLayout(300),
    { vertexId: 5, instanceId: 7, drawId: 9 },
    ['GL_ANGLE_multi_draw'],
  );
  const p = r.ctx.out.position;
  check(
    p[0] === 5 && p[1] === 7 && p[2] === 9 && p[3] === 1,
    `gl_VertexID/gl_InstanceID/gl_DrawID read: [5,7,9,1] (got [${Array.from(p).join(', ')}])`,
  );
  check(r.ctx.out.pointSize === 3, `gl_PointSize write (got ${r.ctx.out.pointSize})`);
}
{
  // ES 1.00 gl_DrawID (the CTS webgl-multi-draw.html shaders are ES 1.00
  // with `#extension GL_ANGLE_multi_draw : require`).
  const r = runStage(
    `#extension GL_ANGLE_multi_draw : require
     void main() { gl_Position = vec4(float(gl_DrawID), 0.0, 0.0, 1.0); }`,
    'VERTEX',
    baseLayout(100),
    { drawId: 4 },
    ['GL_ANGLE_multi_draw'],
  );
  const p = r.ctx.out.position;
  check(
    p[0] === 4 && p[1] === 0 && p[2] === 0 && p[3] === 1,
    `1.00 gl_DrawID read: [4,0,0,1] (got [${Array.from(p).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* 6. VERTEX: user function through the stage entry (inliner)          */
/* ------------------------------------------------------------------ */

{
  const r = runStage(
    `vec4 transform(vec4 p, float s) { return vec4(p.xyz * s, p.w); }
     void main() { gl_Position = transform(vec4(1.0, 2.0, 3.0, 1.0), 2.0); }`,
    'VERTEX',
    baseLayout(100),
  );
  const p = r.ctx.out.position;
  check(
    p[0] === 2 && p[1] === 4 && p[2] === 6 && p[3] === 1,
    `user function inlined via the stage entry (got [${Array.from(p).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* 7. VERTEX: scratch size reporting (local array → ctx.scratch)       */
/* ------------------------------------------------------------------ */

{
  const r = runStage(
    `void main() { float a[4]; a[0] = 1.0; a[3] = 2.0; gl_Position.x = a[0] + a[3]; }`,
    'VERTEX',
    baseLayout(100),
  );
  check(r.ctx.out.position[0] === 3, `scratch-backed local array: x === 3 (got ${r.ctx.out.position[0]})`);
  check(r.scratchSize >= 4, `scratchSize reported >= 4 (got ${r.scratchSize})`);
}

/* ------------------------------------------------------------------ */
/* 8. FRAGMENT: varying reads (vec4 + array, const indices)            */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, { varyings: new Map([['vColor', vg(0, 0, 4, 4)]]) });
  const r = runStage(
    `precision mediump float;
     varying vec4 vColor;
     void main() { gl_FragColor = vColor; }`,
    'FRAGMENT',
    layout,
    { varyings: [{ v: new Float32Array([0.125, 0.25, 0.5, 1.0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 0.125 && c[1] === 0.25 && c[2] === 0.5 && c[3] === 1,
    `fragment varying read → color (got [${Array.from(c).join(', ')}])`,
  );
}
{
  const layout = baseLayout(100, { varyings: new Map([['vF', vg(1, 4, 3, 1)]]) });
  const r = runStage(
    `precision mediump float;
     varying float vF[3];
     void main() { gl_FragColor = vec4(vF[0], vF[1], vF[2], 1.0); }`,
    'FRAGMENT',
    layout,
    { varyings: [{ v: new Float32Array(4) }, { v: new Float32Array([10, 11, 12, 0]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 10 && c[1] === 11 && c[2] === 12 && c[3] === 1,
    `fragment array varying reads → [10,11,12,1] (got [${Array.from(c).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* 9. FRAGMENT: gl_FragCoord / gl_FrontFacing / gl_PointCoord          */
/* ------------------------------------------------------------------ */

{
  const r = runStage(
    `precision mediump float;
     void main() {
       if (gl_FrontFacing) {
         gl_FragColor = vec4(gl_FragCoord.x, gl_FragCoord.y, gl_FragCoord.z, 1.0);
       } else {
         gl_FragColor = vec4(gl_PointCoord, 0.0, 1.0);
       }
     }`,
    'FRAGMENT',
    baseLayout(100),
    { fragCoord: new Float32Array([10, 20, 0.5, 1]), frontFacing: true },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 10 && c[1] === 20 && c[2] === 0.5 && c[3] === 1,
    `gl_FragCoord + frontFacing=true → [10,20,0.5,1] (got [${Array.from(c).join(', ')}])`,
  );
}
{
  const r = runStage(
    `precision mediump float;
     void main() { gl_FragColor = vec4(gl_PointCoord, 0.0, 1.0); }`,
    'FRAGMENT',
    baseLayout(100),
    { frontFacing: false, pointCoord: new Float32Array([0.25, 0.75]) },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 0.25 && c[1] === 0.75 && c[2] === 0 && c[3] === 1,
    `gl_PointCoord read → [0.25,0.75,0,1] (got [${Array.from(c).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* 10. FRAGMENT: ES 3.00 `out` variable + gl_FragDepth                 */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(300, { outputLocations: new Map([['oColor', 0]]) });
  const r = runStage(
    `#version 300 es
     precision mediump float;
     out vec4 oColor;
     void main() { oColor = vec4(1.0, 0.0, 0.0, 1.0); gl_FragDepth = 0.25; }`,
    'FRAGMENT',
    layout,
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 1 && c[1] === 0 && c[2] === 0 && c[3] === 1,
    `ES 3.00 out vec4 write (got [${Array.from(c).join(', ')}])`,
  );
  check(r.ctx.out.fragDepth === 0.25, `gl_FragDepth write (got ${r.ctx.out.fragDepth})`);
}

/* ------------------------------------------------------------------ */
/* 11. ROUND-TRIP: vertex packs varyings → fragment reads them back    */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, {
    attribLocations: new Map([['aPos', 0]]),
    varyings: new Map([['vColor', vg(0, 0, 4, 4)]]),
  });
  const v = runStage(
    `attribute vec4 aPos;
     varying vec4 vColor;
     void main() { vColor = vec4(aPos.xy * 0.5, aPos.zw); gl_Position = aPos; }`,
    'VERTEX',
    layout,
    { attribs: [new Float32Array([1, 2, 3, 4])], attribIndices: new Int32Array([0]) },
  );
  const packed = v.ctx.out.varyings;
  check(
    packed[0] === 0.5 && packed[1] === 1 && packed[2] === 3 && packed[3] === 4,
    `vertex packs vColor at offset 0 (got [${Array.from(packed.slice(0, 4)).join(', ')}])`,
  );
  const f = runStage(
    `precision mediump float;
     varying vec4 vColor;
     void main() { gl_FragColor = vColor + vec4(0.0, 0.0, 0.5, 0.0); }`,
    'FRAGMENT',
    layout,
    { varyings: [{ v: packed.slice(0, 4) }] },
  );
  const c = f.ctx.out.color[0];
  check(
    c[0] === 0.5 && c[1] === 1 && c[2] === 3.5 && c[3] === 4,
    `round-trip: vertex varying → fragment color (got [${Array.from(c).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* 12. FRAGMENT: texture2D implicit LOD (ctx.tex.sample2D + ctx.tex.out) */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(100, {
    uniformSlots: new Map([['uTex', slot('int', 0)]]),
    varyings: new Map([['vUV', vg(0, 0, 2, 2)]]),
  });
  const r = runStage(
    `precision mediump float;
     uniform sampler2D uTex;
     varying vec2 vUV;
     void main() { gl_FragColor = texture2D(uTex, vUV); }`,
    'FRAGMENT',
    layout,
    { intUniforms: Int32Array.from([3]), varyings: [{ v: new Float32Array([0.5, 0.25]) }], tex: makeTex() },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 1 && c[1] === 0.5 && c[2] === 0.25 && c[3] === 1,
    `texture2D implicit-LOD → ctx.tex.out (got [${Array.from(c).join(', ')}])`,
  );
  check(
    r.body.includes('ctx.tex.sample2D('),
    `fragment texture2D lowers to 'ctx.tex.sample2D(' (body: ${r.body.slice(0, 100)}...)`,
  );
}

/* ------------------------------------------------------------------ */
/* 13. EXTRA: ES 3.00 varying INTERFACE BLOCKS (member paths under an  */
/*     instance name: layout.varyings keys 'vs_out.c' / 'vs_in.c')     */
/* ------------------------------------------------------------------ */

{
  const layout = baseLayout(300, { varyings: new Map([['vs_out.c', vg(0, 0, 4, 4)]]) });
  const r = runStage(
    `#version 300 es
     out VS_OUT { vec4 c; } vs_out;
     void main() { vs_out.c = vec4(1.0, 2.0, 3.0, 4.0); gl_Position = vec4(0.0); }`,
    'VERTEX',
    layout,
  );
  const v = r.ctx.out.varyings;
  check(
    v[0] === 1 && v[1] === 2 && v[2] === 3 && v[3] === 4,
    `varying interface block write (key 'vs_out.c') (got [${Array.from(v.slice(0, 4)).join(', ')}])`,
  );
}
{
  const layout = baseLayout(300, {
    varyings: new Map([['vs_in.c', vg(0, 0, 4, 4)]]),
    outputLocations: new Map([['oColor', 0]]),
  });
  const r = runStage(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; } vs_in;
     out vec4 oColor;
     void main() { oColor = vs_in.c; }`,
    'FRAGMENT',
    layout,
    { varyings: [{ v: new Float32Array([4, 3, 2, 1]) }] },
  );
  const c = r.ctx.out.color[0];
  check(
    c[0] === 4 && c[1] === 3 && c[2] === 2 && c[3] === 1,
    `varying interface block read (key 'vs_in.c') (got [${Array.from(c).join(', ')}])`,
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`codegen-stage selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
