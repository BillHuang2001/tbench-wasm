/**
 * selftest-integration.ts — FINAL end-to-end pipeline selftest (integration
 * gate). Unlike the per-module selftests (which drive the front-end and
 * codegen pieces individually with hand-built layouts), this file exercises
 * the FULL public pipeline — `compileShader` → `linkProgram` → run the linked
 * stages against hand-built exec ctxs — with KITCHEN-SINK shaders that combine
 * features the other selftests only cover piecewise:
 *
 *   1. kitchen-sink ES 3.00 pair (no derivatives): layout(location=) attrib +
 *      mat4 attrib, struct-array uniform {vec2 a; float b[2];}, user function
 *      with inout param, local array with dynamic index, gl_PointSize,
 *      varying interface block (different instance names per stage), plain
 *      varying, gl_FragDepth, gl_FrontFacing → link + run + flag assertions.
 *   2. derivatives THROUGH linkProgram: dFdx/dFdy + implicit-LOD texture2D +
 *      sin/mix/clamp; ctx.varyings[i].ddx/ddy supplied; ctx.tex stub verifies
 *      the gradient routing ran; analytic derivative lands in the color.
 *   3. UBO + dual-mode + inlining composed: block members read inside a user
 *      function called from both stages; dFdx of a block-derived value
 *      (block members are constant → derivative == v.ddx * k).
 *   4. GL_EXT_draw_buffers gl_FragData[0]/[1] → 2-output program, both colors
 *      written.
 *   5. error-path integration: uniform type conflict (link), recursion
 *      (compile), gl_FragColor XOR gl_FragData (compile), unknown extension
 *      (compile, with line number).
 *   6. minimal `void main() {}` pair links and runs (no throw).
 *   7. discard through the pipeline (ctx.discarded set / cleared).
 *   8. Program contract sanity on the check-1 program: uniformMap keys,
 *      no block members in uniformMap (check-3 program), varying order,
 *      store sizing vs. location high-water marks, scratchSize > 0.
 *   9. scalar-RHS broadcast on vector/matrix lvalues: +=/-=/*=//= (and ES3
 *      %= on ivec4) with a scalar RHS — statement, expression and for-update
 *      contexts, matrix lvalues, ES 1.00 + ES 3.00, and dual mode.
 *
 * Run: npx tsx src/glsl/selftest-integration.ts
 * Prints "OK" and exits 0 on success; non-zero exit on failure.
 */
import { compileShader, linkProgram } from './compiler.js';
import type { CompileResult } from './compiler.js';
import type { Program, UniformInfo } from './program.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/** Float32Array-stored colors are rounded — compare with a small tolerance. */
function near(got: number, expected: number, msg: string): void {
  check(Math.abs(got - expected) < 1e-6, `${msg} (got ${got}, expected ${expected})`);
}

/* ------------------------------------------------------------------ */
/* Helpers (ctx shapes per program.ts — same as selftest-link.ts)      */
/* ------------------------------------------------------------------ */

function compile(src: string, type: 'VERTEX' | 'FRAGMENT', version: 100 | 300, extensions?: string[]) {
  const r = compileShader(src, { type, version, extensions: extensions ? new Set(extensions) : undefined });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  return r.shader;
}

function compileErr(
  src: string,
  type: 'VERTEX' | 'FRAGMENT',
  version: 100 | 300,
  extensions?: string[],
): CompileResult {
  return compileShader(src, { type, version, extensions: extensions ? new Set(extensions) : undefined });
}

function vertexCtx(prog: Program, extra: Record<string, unknown> = {}): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { position: new Float32Array(4), pointSize: 0, varyings: new Float32Array(32) },
    attribs: [],
    attribIndices: new Int32Array(16),
    vertexId: 0,
    instanceId: 0,
    ...extra,
  };
}

/** `varyings` entries may carry {v, ddx?, ddy?} (dual mode reads them directly). */
function fragmentCtx(prog: Program, varyings: any[] = [], extra: Record<string, unknown> = {}): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { color: [new Float32Array(4)], fragDepth: 0 },
    discarded: false,
    fragCoord: new Float32Array([0, 0, 0, 1]),
    frontFacing: true,
    pointCoord: new Float32Array([0, 0]),
    varyings,
    ...extra,
  };
}

/** A ctx.tex stub shaped like raster's TextureEnv: methods write `out`. The
 *  gradient params fold into out[2]/out[3] with distinct weights so the
 *  dual-number gradient routing is observable (selftest-dual-builtins.ts). */
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
      out[0] = u; out[1] = v; out[2] = w; out[3] = 10 * dux + 20 * dvx + 30 * dwx + 40 * duy + 50 * dvy + 60 * dwy;
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

const FLOAT_VEC2 = 0x8b50;
const FLOAT_VEC4 = 0x8b52;
const FLOAT_MAT4 = 0x8b5c;
const FLOAT = 0x1406;

/** Check-1 program is re-inspected by check 8 — keep a handle. */
let kitchenSink: Program | null = null;
/** Check-3 program is used for the "block members NOT in uniformMap" check. */
let uboDual: Program | null = null;

/* ================================================================== */
/* 1. Kitchen-sink ES 3.00 pair (no derivatives)                       */
/* ================================================================== */

{
  const vs = compile(
    `#version 300 es
     struct S { vec2 a; float b[2]; };
     layout(location = 0) in vec4 aPos;
     in mat4 aM;
     uniform S u[2];
     uniform int uIdx;
     out VS_OUT { vec4 c; float d; } vs;
     out float vPlain;
     float bump(inout float x, float k) { x += k; return x * 2.0; }
     void main() {
       float arr[2];
       arr[0] = aM[0][0];
       arr[1] = aM[1][1];
       float t = arr[uIdx];
       float r = bump(t, u[0].a.x);
       vs.c = aPos + aM * vec4(u[0].a, u[0].b[0], u[0].b[1]);
       vs.d = r;
       vPlain = u[1].a.y;
       gl_Position = aPos;
       gl_PointSize = 3.0;
     }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     struct S { vec2 a; float b[2]; };
     in VS_OUT { vec4 c; float d; } fs;
     in float vPlain;
     uniform S u[2];
     out vec4 oColor;
     void main() {
       oColor = vec4(fs.c.xy, fs.d + u[0].b[1], gl_FrontFacing ? 1.0 : 0.5);
       gl_FragDepth = 0.75;
     }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `kitchen-sink 300 pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    kitchenSink = p;

    /* Capability flags */
    check(p.usesPointSize === true, `usesPointSize true (gl_PointSize written)`);
    check(p.usesGLPointCoord === false, `usesGLPointCoord false (gl_PointCoord unused)`);
    check(p.usesFragCoord === false, `usesFragCoord false (gl_FragCoord unused)`);
    check(p.usesFrontFacing === true, `usesFrontFacing true (gl_FrontFacing read)`);
    check(p.fragment.usesFragDepth === true, `fragment.usesFragDepth true (gl_FragDepth written)`);
    check(p.fragment.usesDerivatives === false, `fragment.usesDerivatives false (no dFdx/texture)`);
    check(
      p.fragment.outputs.length === 1 && p.fragment.outputs[0].location === 0 && p.fragment.outputs[0].type === FLOAT_VEC4,
      `fragment.outputs [{0, FLOAT_VEC4}] (got ${JSON.stringify(p.fragment.outputs)})`,
    );

    /* Attributes: layout(location=0) aPos + mat4 auto-located at 1..4 */
    check(
      p.attributes.length === 2 && p.attributes[0].name === 'aPos' && p.attributes[0].location === 0 &&
        p.attributes[0].type === FLOAT_VEC4 && p.attributes[0].components === 4,
      `attrib aPos @0 FLOAT_VEC4 (got ${JSON.stringify(p.attributes)})`,
    );
    check(
      p.attributes[1].name === 'aM' && p.attributes[1].location === 1 && p.attributes[1].type === FLOAT_MAT4 &&
        p.attributes[1].components === 4,
      `attrib aM @1 FLOAT_MAT4 (got ${JSON.stringify(p.attributes)})`,
    );

    /* Varyings: block leaves in vertex declaration order + plain varying */
    check(
      p.varyings.length === 3 && p.varyings[0].name === 'vs.c' && p.varyings[0].components === 4 &&
        p.varyings[1].name === 'vs.d' && p.varyings[1].components === 1 &&
        p.varyings[2].name === 'vPlain' && p.varyings[2].components === 1,
      `varyings packed order [vs.c(4), vs.d(1), vPlain(1)] (got ${JSON.stringify(p.varyings)})`,
    );

    /* Uniforms: write via uniformMap (the contract gl/ uses) */
    const um = p.uniformMap;
    const setF = (key: string, v: number[]) => {
      const loc = um.get(key)!.location;
      for (let i = 0; i < v.length; i++) p.floatStore[loc + i] = v[i];
    };
    setF('u[0].a', [10, 20]);
    setF('u[0].b[0]', [1]);
    setF('u[0].b[1]', [2]);
    setF('u[1].a', [0, 30]);
    setF('u[1].b[0]', [5]);
    setF('u[1].b[1]', [6]);
    p.intStore[um.get('uIdx')!.location] = 1;

    /* Vertex run: aPos=[1,2,3,4]; aM cols (loc 1..4) = diag(2,3,4,5) */
    const vctx = vertexCtx(p, {
      attribs: [
        new Float32Array([1, 2, 3, 4]), // aPos
        new Float32Array([2, 0, 0, 0]), // aM col 0
        new Float32Array([0, 3, 0, 0]), // aM col 1
        new Float32Array([0, 0, 4, 0]), // aM col 2
        new Float32Array([0, 0, 0, 5]), // aM col 3
      ],
      attribIndices: new Int32Array([0, 0, 0, 0, 0]),
    });
    p.vertex.run(vctx);
    const pos = vctx.out.position;
    check(
      pos[0] === 1 && pos[1] === 2 && pos[2] === 3 && pos[3] === 4,
      `vertex position [1,2,3,4] (got [${Array.from(pos).join(', ')}])`,
    );
    check(vctx.out.pointSize === 3, `gl_PointSize written 3 (got ${vctx.out.pointSize})`);
    // vs.c = aPos + aM*[10,20,1,2] = [1,2,3,4] + [20,60,4,10] = [21,62,7,14]
    // vs.d = bump(arr[uIdx=1]=3, u[0].a.x=10) → (3+10)*2 = 26; vPlain = u[1].a.y = 30
    const vg = vctx.out.varyings;
    check(
      vg[0] === 21 && vg[1] === 62 && vg[2] === 7 && vg[3] === 14 && vg[4] === 26 && vg[5] === 30,
      `vertex varyings [21,62,7,14,26,30] (got [${Array.from(vg.slice(0, 6)).join(', ')}])`,
    );

    /* Fragment run (frontFacing true): color = [c.xy, d + u[0].b[1], 1] */
    const fctx = fragmentCtx(p, [{ v: vg.slice(0, 4) }, { v: vg.slice(4, 5) }, { v: vg.slice(5, 6) }]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(
      c[0] === 21 && c[1] === 62 && c[2] === 28 && c[3] === 1,
      `fragment color [21,62,28,1] (block members + struct uniform + frontFacing) (got [${Array.from(c).join(', ')}])`,
    );
    check(fctx.out.fragDepth === 0.75, `gl_FragDepth write 0.75 (got ${fctx.out.fragDepth})`);

    /* Re-run with frontFacing false → the ternary flips the alpha */
    const fctx2 = fragmentCtx(
      p,
      [{ v: vg.slice(0, 4) }, { v: vg.slice(4, 5) }, { v: vg.slice(5, 6) }],
      { frontFacing: false },
    );
    p.fragment.run(fctx2);
    check(fctx2.out.color[0][3] === 0.5, `frontFacing=false path: alpha 0.5 (got ${fctx2.out.color[0][3]})`);
  }
}

/* ================================================================== */
/* 2. Derivatives through linkProgram (dFdx/dFdy + implicit-LOD tex)   */
/* ================================================================== */

{
  const vs = compile(
    `attribute vec4 aPos;
     varying vec2 vUV;
     varying float v;
     void main() { vUV = aPos.xy; v = aPos.x; gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fs = compile(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     uniform sampler2D uTex;
     varying vec2 vUV;
     varying float v;
     void main() {
       vec4 t = texture2D(uTex, vUV);
       float c = clamp(v, 0.0, 1.0);
       float m = mix(c, 1.0 - c, 0.5);
       float d = dFdx(v) + dFdy(v);
       gl_FragColor = vec4(d, sin(m), t.x, t.z);
     }`,
    'FRAGMENT',
    100,
    ['GL_OES_standard_derivatives'],
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `derivative pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.fragment.usesDerivatives === true, `fragment.usesDerivatives === true through linkProgram`);

    /* Vertex: vUV = aPos.xy = [0.25, 0.5], v = 0.25 */
    const vctx = vertexCtx(p, {
      attribs: [new Float32Array([0.25, 0.5, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    const vg = vctx.out.varyings;
    check(
      vg[0] === 0.25 && vg[1] === 0.5 && vg[2] === 0.25,
      `vertex packs vUV=[0.25,0.5], v=0.25 (got [${Array.from(vg.slice(0, 3)).join(', ')}])`,
    );

    /* Fragment: dual mode — ctx.varyings[i] must carry ddx/ddy. */
    p.intStore[p.uniformMap.get('uTex')!.location] = 0;
    const fctx = fragmentCtx(
      p,
      [
        { v: new Float32Array([0.25, 0.5]), ddx: new Float32Array([2, 3]), ddy: new Float32Array([4, 5]) },
        { v: new Float32Array([0.25]), ddx: new Float32Array([2]), ddy: new Float32Array([3]) },
      ],
      { tex: makeTexStub() },
    );
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    near(c[0], 5, `dFdx(v) + dFdy(v) == 2 + 3 == 5`);
    // c = clamp(0.25,0,1) = 0.25; m = mix(0.25, 0.75, 0.5) = 0.5
    near(c[1], Math.sin(0.5), `sin(mix(clamp(v,0,1), 1-clamp(v,0,1), 0.5)) == sin(0.5)`);
    near(c[2], 0.25, `texture2D ran: sampled u == coord u`);
    near(c[3], 10 * 2 + 20 * 3 + 30 * 4 + 40 * 5, `texture2D routed coord duals into the gradient slot`);
  }
}

/* ================================================================== */
/* 3. UBO + dual mode + inlining composed                              */
/* ================================================================== */

{
  const vs = compile(
    `#version 300 es
     uniform Params { float k; float j; } p;
     in vec4 aPos;
     float scale(float x) { return x * p.k + p.j; }
     out float vX;
     void main() { vX = scale(aPos.x); gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     uniform Params { float k; float j; } p;
     in float vX;
     out vec4 oColor;
     float scale(float x) { return x * p.k + p.j; }
     void main() {
       float y = scale(vX);
       float d = dFdx(y);
       oColor = vec4(d, y, 0.0, 1.0);
     }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `UBO + dual + inline pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    uboDual = p;
    check(p.fragment.usesDerivatives === true, `fragment.usesDerivatives true (dFdx)`);
    check(
      p.uniformBlocks.length === 1 && p.uniformBlocks[0].name === 'Params' && p.uniformBlocks[0].index === 0 &&
        p.uniformBlocks[0].size === 16,
      `one block 'Params' index 0 size 16 (got ${JSON.stringify(p.uniformBlocks)})`,
    );
    const m = p.uniforms.filter((u) => u.blockIndex === 0);
    check(
      m.length === 2 && m.every((u) => u.location === -1) && m.map((u) => u.name).join(',') === 'p.k,p.j',
      `block members p.k/p.j in uniforms (location -1) (got ${JSON.stringify(m)})`,
    );

    // std140: k@float0, j@float1 (k=2, j=1).
    const store = new Float32Array([2, 1]);
    const vctx = vertexCtx(p, {
      blockStores: [store],
      attribs: [new Float32Array([1, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    near(vctx.out.varyings[0], 3, `vertex: vX = scale(1) = 1*2 + 1 = 3`);

    const fctx = fragmentCtx(
      p,
      [{ v: new Float32Array([3]), ddx: new Float32Array([3]), ddy: new Float32Array([0]) }],
      { blockStores: [store], blockIntStores: [] },
    );
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    // y = scale(vX) = 3*2+1 = 7; dFdx(y) = dFdx(vX)*k (block members constant) = 3*2 = 6.
    near(c[0], 6, `dual + block + inline: dFdx(scale(vX)) == vX.ddx * k == 6`);
    near(c[1], 7, `value: y == scale(3) == 7`);
  }
}

/* ================================================================== */
/* 4. GL_EXT_draw_buffers gl_FragData path                             */
/* ================================================================== */

{
  const vs = compile(
    `attribute vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fs = compile(
    `#extension GL_EXT_draw_buffers : require
     precision mediump float;
     void main() {
       gl_FragData[0] = vec4(1.0, 0.0, 0.0, 1.0);
       gl_FragData[1] = vec4(0.0, 1.0, 0.0, 1.0);
     }`,
    'FRAGMENT',
    100,
    ['GL_EXT_draw_buffers'],
  );
  const l = linkProgram(vs, fs, { limits: { maxDrawBuffers: 4 } });
  check(l.ok, `gl_FragData pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(
      p.fragment.outputs.length === 2 && p.fragment.outputs[0].location === 0 && p.fragment.outputs[1].location === 1 &&
        p.fragment.outputs[0].type === FLOAT_VEC4 && p.fragment.outputs[1].type === FLOAT_VEC4,
      `fragment.outputs [{0,FLOAT_VEC4},{1,FLOAT_VEC4}] (got ${JSON.stringify(p.fragment.outputs)})`,
    );
    const fctx = fragmentCtx(p, [], { out: { color: [new Float32Array(4), new Float32Array(4)], fragDepth: 0 } });
    p.fragment.run(fctx);
    const c0 = fctx.out.color[0];
    const c1 = fctx.out.color[1];
    check(
      c0[0] === 1 && c0[1] === 0 && c0[2] === 0 && c0[3] === 1,
      `gl_FragData[0] written [1,0,0,1] (got [${Array.from(c0).join(', ')}])`,
    );
    check(
      c1[0] === 0 && c1[1] === 1 && c1[2] === 0 && c1[3] === 1,
      `gl_FragData[1] written [0,1,0,1] (got [${Array.from(c1).join(', ')}])`,
    );
  }
}

/* ================================================================== */
/* 5. Error-path integration                                           */
/* ================================================================== */

{
  // (a) same uniform name, different types vs/fs → LINK error
  const vs = compile(`uniform vec4 uC; attribute vec4 aPos; void main(){ gl_Position = aPos + uC; }`, 'VERTEX', 100);
  const fs = compile(`precision mediump float; uniform float uC; void main(){ gl_FragColor = vec4(uC); }`, 'FRAGMENT', 100);
  const l = linkProgram(vs, fs);
  check(!l.ok && l.log.includes(`uniform 'uC' type conflict`), `uniform type conflict → link error (${l.ok ? 'linked!' : l.log})`);

  // (b) recursion → COMPILE error (semantics)
  const rRec = compileErr(
    `float f(float x) { return f(x); }
     attribute vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  check(
    !rRec.ok && rRec.errors.length > 0 && rRec.errors[0].message.includes('recursion is not allowed'),
    `recursion → compile error (got ${JSON.stringify(rRec.ok ? 'ok' : rRec.errors)})`,
  );

  // (c) gl_FragColor + gl_FragData both written → COMPILE error
  const rBoth = compileErr(
    `#extension GL_EXT_draw_buffers : require
     precision mediump float;
     void main() { gl_FragColor = vec4(1.0); gl_FragData[0] = vec4(1.0); }`,
    'FRAGMENT',
    100,
    ['GL_EXT_draw_buffers'],
  );
  check(
    !rBoth.ok && rBoth.errors.length > 0 && rBoth.errors[0].message.includes('cannot write both gl_FragColor and gl_FragData'),
    `gl_FragColor XOR gl_FragData → compile error (got ${JSON.stringify(rBoth.ok ? 'ok' : rBoth.errors)})`,
  );

  // (d) unknown extension → COMPILE error with a line number
  const rExt = compileErr(
    `#extension GL_FAKE : require
     precision mediump float;
     void main() { gl_FragColor = vec4(1.0); }`,
    'FRAGMENT',
    100,
  );
  check(
    !rExt.ok && rExt.errors.length > 0 && rExt.errors[0].line === 1 && rExt.errors[0].message.includes('GL_FAKE'),
    `unknown extension → compile error at line 1 (got ${JSON.stringify(rExt.ok ? 'ok' : rExt.errors)})`,
  );
}

/* ================================================================== */
/* 6. Minimal `void main() {}` pair                                    */
/* ================================================================== */

{
  const vs = compile(`void main() { }`, 'VERTEX', 100);
  const fs = compile(`precision mediump float; void main() { }`, 'FRAGMENT', 100);
  const l = linkProgram(vs, fs);
  check(l.ok, `minimal pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.attributes.length === 0 && p.varyings.length === 0 && p.uniforms.length === 0,
      `minimal pair has no active resources (got a=${p.attributes.length} v=${p.varyings.length} u=${p.uniforms.length})`);
    const vctx = vertexCtx(p);
    let vThrew = '';
    try {
      p.vertex.run(vctx);
    } catch (e) {
      vThrew = String(e);
    }
    check(vThrew === '', `minimal vertex run() doesn't throw (${vThrew || 'ok'})`);
    const fctx = fragmentCtx(p);
    let fThrew = '';
    try {
      p.fragment.run(fctx);
    } catch (e) {
      fThrew = String(e);
    }
    check(fThrew === '' && fctx.discarded === false, `minimal fragment run() doesn't throw (${fThrew || 'ok'})`);
  }
}

/* ================================================================== */
/* 7. discard through the pipeline                                     */
/* ================================================================== */

{
  const vs = compile(`attribute vec4 aPos; void main() { gl_Position = aPos; }`, 'VERTEX', 100);
  const fs = compile(
    `precision mediump float;
     void main() { if (gl_FrontFacing) discard; gl_FragColor = vec4(1.0); }`,
    'FRAGMENT',
    100,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `discard pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.usesFrontFacing === true, `discard variant: usesFrontFacing true`);
    const fctx = fragmentCtx(p, [], { frontFacing: true });
    p.fragment.run(fctx);
    check(fctx.discarded === true, `discard sets ctx.discarded (raster commits nothing)`);
    const fctx2 = fragmentCtx(p, [], { frontFacing: false });
    p.fragment.run(fctx2);
    check(
      fctx2.discarded === false && fctx2.out.color[0][0] === 1 && fctx2.out.color[0][3] === 1,
      `non-discard path: color written [1,...,1] (got [${Array.from(fctx2.out.color[0]).join(', ')}])`,
    );
  }
}

/* ================================================================== */
/* 8. Program contract sanity (check 1's program + check 3's UBO)      */
/* ================================================================== */

{
  const p = kitchenSink;
  check(p !== null, `check-1 program available for contract sanity`);
  if (p) {
    const um = p.uniformMap;
    // Struct-array uniform keys: bare name + '[0]' both map to the first leaf.
    check(um.get('u') !== undefined && um.get('u') === um.get('u[0]') && um.get('u[0]') === um.get('u[0].a'),
      `uniformMap 'u' and 'u[0]' → first leaf (struct array forms)`);
    check(
      um.get('u[0].a') !== undefined && um.get('u[0].b[0]') !== undefined && um.get('u[0].b[1]') !== undefined &&
        um.get('u[1].a') !== undefined && um.get('u[1].b[0]') !== undefined && um.get('u[1].b[1]') !== undefined &&
        um.get('uIdx') !== undefined,
      `uniformMap covers all struct-array leaves + uIdx (got [${[...um.keys()].join(', ')}])`,
    );
    // No varying/output paths in uniformMap.
    check(
      !um.has('vs.c') && !um.has('fs.c') && !um.has('vPlain') && !um.has('oColor'),
      `uniformMap excludes varying/output paths`,
    );

    // Varying order (already asserted in check 1, re-pin here against the info list).
    check(
      p.varyings.map((v) => v.name).join(',') === 'vs.c,vs.d,vPlain' &&
        p.varyings.map((v) => v.components).join(',') === '4,1,1',
      `program.varyings matches the packed order`,
    );

    // Store sizing: floatStore/intStore must cover every default-block uniform.
    // Packing is DENSE: a uniform's span is its raw component count (scalar 1,
    // vec2 2, vec4 4; matrices rows×cols; int compounds 1 int per component).
    const slotSpan = (u: UniformInfo): number => {
      if (u.integral || u.sampler) return u.components;
      if (u.type === 0x8b5a) return u.components * 2; // FLOAT_MAT2
      if (u.type === 0x8b5b) return u.components * 3; // FLOAT_MAT3
      if (u.type === 0x8b5c) return u.components * 4; // FLOAT_MAT4
      return u.components;
    };
    const fHigh = Math.max(0, ...p.uniforms.filter((u) => u.location >= 0 && !u.integral && !u.sampler)
      .map((u) => u.location + slotSpan(u)));
    const iHigh = Math.max(0, ...p.uniforms.filter((u) => u.location >= 0 && (u.integral || u.sampler))
      .map((u) => u.location + slotSpan(u)));
    check(p.floatStore.length >= fHigh, `floatStore sized >= high-water ${fHigh} (got ${p.floatStore.length})`);
    check(p.intStore.length >= iHigh, `intStore sized >= high-water ${iHigh} (got ${p.intStore.length})`);

    // Scratch: the vertex's local array must have charged the program scratch.
    check(p.scratchSize > 0, `local array → program.scratchSize > 0 (got ${p.scratchSize})`);
    check(p.intScratchSize >= 0, `intScratchSize >= 0 (got ${p.intScratchSize})`);
  }

  // Block members must NOT be reachable via uniformMap (getUniformLocation → null).
  const p3 = uboDual;
  check(p3 !== null, `check-3 program available for block-member contract sanity`);
  if (p3) {
    check(p3.uniformMap.has('p.k') === false && p3.uniformMap.has('p.j') === false && p3.uniformMap.has('k') === false,
      `UBO members absent from uniformMap (keys: [${[...p3.uniformMap.keys()].join(', ')}])`);
    check(p3.scratchSize >= 0 && p3.intScratchSize >= 0, `UBO+dual program scratch sizes sane (${p3.scratchSize}/${p3.intScratchSize})`);
  }
}

/* ================================================================== */
/* 9. Scalar-RHS broadcast on vector/matrix lvalues (assignment)       */
/* ================================================================== */
/* Regression: a scalar RHS of an assignment to a vector/matrix lvalue
 * (`x /= 2.0`, `m *= 2.0`, `x -= 1.0` in a for-update, `(y *= 2.0).x` in an
 * expression) must emit a per-component broadcast — EVERY lvalue slot gets
 * the converted scalar. Before the fix the scalar was emitted into slot 0
 * only and the remaining slots read `undefined.v`/`undefined.pre` at link
 * time ("codegen failed: Cannot read properties of undefined"). All five
 * compound ops (+=, -=, *=, /=, %=) + the plain `=` defensive path, statement
 * and expression contexts, ES 1.00 and ES 3.00, float and integral bases,
 * and dual mode are covered. */

{
  // ES 1.00: /= on a vec4 local (statement context).
  {
    const vs = compile(`attribute vec4 aPos; varying vec4 v; void main() { v = aPos; gl_Position = aPos; }`, 'VERTEX', 100);
    const fs = compile(
      `precision mediump float;
       varying vec4 v;
       void main() { vec4 x = v; x /= 2.0; gl_FragColor = x; }`,
      'FRAGMENT',
      100,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `scalar-broadcast vec4 /= links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const fctx = fragmentCtx(l.program, [{ v: new Float32Array([1, 2, 4, 8]) }]);
      l.program.fragment.run(fctx);
      const o = fctx.out.color[0];
      [0.5, 1, 2, 4].forEach((e, c) => near(o[c], e, `vec4 /= 2.0 → out[${c}]`));
    }
  }

  // ES 3.00: += on a vec4 local (statement context).
  {
    const vs = compile(`#version 300 es\nin vec4 aPos; out vec4 v; void main() { v = aPos; gl_Position = aPos; }`, 'VERTEX', 300);
    const fs = compile(
      `#version 300 es
       precision mediump float;
       in vec4 v;
       out vec4 c;
       void main() { vec4 x = v; x += 1.5; c = x; }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `scalar-broadcast vec4 += (ES3) links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const fctx = fragmentCtx(l.program, [{ v: new Float32Array([1, 2, 4, 8]) }]);
      l.program.fragment.run(fctx);
      const o = fctx.out.color[0];
      [2.5, 3.5, 5.5, 9.5].forEach((e, c) => near(o[c], e, `vec4 += 1.5 → out[${c}]`));
    }
  }

  // Matrix lvalues: mat2 *= and mat4 /= (scalar → every matrix slot).
  {
    const vs = compile(`attribute vec4 aPos; void main() { gl_Position = aPos; }`, 'VERTEX', 100);
    const fs = compile(
      `precision mediump float;
       void main() {
         mat2 m = mat2(1.0, 2.0, 3.0, 4.0);
         m *= 2.0;
         mat4 n = mat4(8.0);
         n /= 2.0;
         gl_FragColor = vec4(m[0][0], m[0][1], m[1][0], m[1][1]) +
                        vec4(n[0][0], n[1][1], n[2][2], n[3][3]) / 20.0;
       }`,
      'FRAGMENT',
      100,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `scalar-broadcast mat2 *= / mat4 /= links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const fctx = fragmentCtx(l.program);
      l.program.fragment.run(fctx);
      const o = fctx.out.color[0];
      [2.2, 4.2, 6.2, 8.2].forEach((e, c) => near(o[c], e, `mat *= scalar → out[${c}]`));
    }
  }

  // -= in a for-update slot (updateString path — non-comma update).
  {
    const vs = compile(`attribute vec4 aPos; varying vec4 v; void main() { v = aPos; gl_Position = aPos; }`, 'VERTEX', 100);
    const fs = compile(
      `precision mediump float;
       varying vec4 v;
       void main() {
         vec4 x = v;
         for (int i = 0; i < 3; x -= 1.0) { i++; }
         gl_FragColor = x;
       }`,
      'FRAGMENT',
      100,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `scalar-broadcast vec4 -= in for-update links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const fctx = fragmentCtx(l.program, [{ v: new Float32Array([1, 2, 4, 8]) }]);
      l.program.fragment.run(fctx);
      const o = fctx.out.color[0];
      [-2, -1, 1, 5].forEach((e, c) => near(o[c], e, `vec4 -= 1.0 ×3 → out[${c}]`));
    }
  }

  // *= in an EXPRESSION context (emitAssign path — the assignment's VALUE is
  // consumed: `(y *= 2.0).x`).
  {
    const vs = compile(`attribute vec4 aPos; varying vec4 v; void main() { v = aPos; gl_Position = aPos; }`, 'VERTEX', 100);
    const fs = compile(
      `precision mediump float;
       varying vec4 v;
       void main() {
         vec2 y = v.xy;
         float r = (y *= 2.0).x;
         gl_FragColor = vec4(r, y.y, 0.0, 1.0);
       }`,
      'FRAGMENT',
      100,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `scalar-broadcast vec2 *= in expression links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const fctx = fragmentCtx(l.program, [{ v: new Float32Array([1, 2, 4, 8]) }]);
      l.program.fragment.run(fctx);
      const o = fctx.out.color[0];
      [2, 4, 0, 1].forEach((e, c) => near(o[c], e, `(y *= 2.0).x → out[${c}]`));
    }
  }

  // ES 3.00 integral base: ivec4 %= int scalar (int broadcast).
  {
    const vs = compile(`#version 300 es\nin vec4 aPos; out vec4 v; void main() { v = aPos; gl_Position = aPos; }`, 'VERTEX', 300);
    const fs = compile(
      `#version 300 es
       precision mediump float;
       in vec4 v;
       out vec4 c;
       void main() {
         ivec4 x = ivec4(5, 7, 9, 11);
         x %= 3;
         c = vec4(x) / 10.0;
       }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `scalar-broadcast ivec4 %= links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const fctx = fragmentCtx(l.program);
      l.program.fragment.run(fctx);
      const o = fctx.out.color[0];
      [0.2, 0.1, 0, 0.2].forEach((e, c) => near(o[c], e, `ivec4 %= 3 → out[${c}]`));
    }
  }

  // DUAL MODE: += scalar on a varying-derived vec4, then dFdx reads the
  // broadcast result — every component's derivative plane must be updated
  // (the scalar contributes constant duals 0; the varying's ddx flows).
  {
    const vs = compile(`#version 300 es\nin vec4 aPos; out vec4 v; void main() { v = aPos; gl_Position = aPos; }`, 'VERTEX', 300);
    const fs = compile(
      `#version 300 es
       precision mediump float;
       in vec4 v;
       out vec4 c;
       void main() { vec4 x = v; x += 2.0; c = x + dFdx(x); }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `scalar-broadcast vec4 += (dual mode) links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      check(l.program.fragment.usesDerivatives === true, `dual broadcast variant uses derivatives`);
      const fctx = fragmentCtx(l.program, [{ v: new Float32Array([1, 2, 4, 8]), ddx: new Float32Array([1, 0, 0, 0]), ddy: new Float32Array([0, 0, 0, 0]) }]);
      l.program.fragment.run(fctx);
      const o = fctx.out.color[0];
      // c_c = (v_c + 2) + d(v_c + 2)/dx = v_c + 2 + ddx(v_c)
      [4, 4, 6, 10].forEach((e, c) => near(o[c], e, `dual x += 2.0 + dFdx → out[${c}]`));
    }
  }
}

/* 10. Postfix ++/-- semantics: result = OLD value, operand ±1          */
/* ================================================================== */

{
  /** Compile a 1.00 vertex shader (fragment = trivial pass-through pair),
   *  link, run once, return ctx.out.position. */
  const runVertexPos = (src: string, version: 100 | 300 = 100): Float32Array => {
    const vs = compile(src, 'VERTEX', version);
    const fs = compile(
      version === 100
        ? `precision mediump float; void main() { gl_FragColor = vec4(0.0); }`
        : `#version 300 es\nprecision mediump float; out vec4 o; void main() { o = vec4(0.0); }`,
      'FRAGMENT',
      version,
    );
    const l = linkProgram(vs, fs);
    check(l.ok, `postfix pair links (${l.ok ? '' : l.log})`);
    if (!l.ok) return new Float32Array(0);
    const p = l.program;
    const vctx = vertexCtx(p);
    p.vertex.run(vctx);
    return vctx.out.position;
  };

  // The exact CTS ogles operators repro: int k = m++; → k=23, m=24.
  {
    const pos = runVertexPos(`void main() { int m = 23; int k = m++; gl_Position = vec4(float(k), float(m), 0.0, 1.0); }`);
    check(pos[0] === 23 && pos[1] === 24, `postfix ++: k=23, m=24 (got ${pos[0]}, ${pos[1]})`);
  }
  {
    const pos = runVertexPos(`void main() { int m = 23; int k = ++m; gl_Position = vec4(float(k), float(m), 0.0, 1.0); }`);
    check(pos[0] === 24 && pos[1] === 24, `prefix ++: k=24, m=24 (got ${pos[0]}, ${pos[1]})`);
  }
  {
    const pos = runVertexPos(`void main() { int m = 23; int k = m--; gl_Position = vec4(float(k), float(m), 0.0, 1.0); }`);
    check(pos[0] === 23 && pos[1] === 22, `postfix --: k=23, m=22 (got ${pos[0]}, ${pos[1]})`);
  }
  {
    const pos = runVertexPos(`void main() { int m = 23; int k = --m; gl_Position = vec4(float(k), float(m), 0.0, 1.0); }`);
    check(pos[0] === 22 && pos[1] === 22, `prefix --: k=22, m=22 (got ${pos[0]}, ${pos[1]})`);
  }
  // Two postfixes in ONE expression: (m++) + (m++) = 23 + 24 = 47, m = 25.
  {
    const pos = runVertexPos(`void main() { int m = 23; int y = (m++) + (m++); gl_Position = vec4(float(y), float(m), 0.0, 1.0); }`);
    check(pos[0] === 47 && pos[1] === 25, `(m++)+(m++): y=47, m=25 (got ${pos[0]}, ${pos[1]})`);
  }
  // Postfix in an array index: a[i++] reads a[old i], then i += 1.
  {
    const pos = runVertexPos(`void main() {
      int a[3]; a[0] = 10; a[1] = 20; a[2] = 30;
      int i = 1;
      int v = a[i++];
      int w = a[i++];
      gl_Position = vec4(float(v), float(w), float(i), 1.0);
    }`);
    check(pos[0] === 20 && pos[1] === 30 && pos[2] === 3, `a[i++]: v=20, w=30, i=3 (got ${pos[0]}, ${pos[1]}, ${pos[2]})`);
  }
  // Postfix in a call argument.
  {
    const pos = runVertexPos(`int twice(int x) { return x * 2; }
      void main() { int m = 5; int r = twice(m++); gl_Position = vec4(float(r), float(m), 0.0, 1.0); }`);
    check(pos[0] === 10 && pos[1] === 6, `twice(m++): r=10, m=6 (got ${pos[0]}, ${pos[1]})`);
  }
  // Postfix in loop conditions (while + do-while bodies).
  {
    const pos = runVertexPos(`void main() {
      int s = 0; int i = 0;
      while (i++ < 4) s += i;
      gl_Position = vec4(float(s), float(i), 0.0, 1.0);
    }`);
    check(pos[0] === 10 && pos[1] === 5, `while (i++ < 4): s=10, i=5 (got ${pos[0]}, ${pos[1]})`);
  }
  {
    const pos = runVertexPos(`void main() {
      int s = 0; int i = 0;
      do { s += i; } while (i++ < 4);
      gl_Position = vec4(float(s), float(i), 0.0, 1.0);
    }`);
    check(pos[0] === 10 && pos[1] === 5, `do { s+=i; } while (i++ < 4): s=10, i=5 (got ${pos[0]}, ${pos[1]})`);
  }
  // Float postfix/prefix.
  {
    const pos = runVertexPos(`void main() { float t = 1.5; float f = t++; gl_Position = vec4(f, t, 0.0, 1.0); }`);
    check(pos[0] === 1.5 && pos[1] === 2.5, `float postfix ++: f=1.5, t=2.5 (got ${pos[0]}, ${pos[1]})`);
  }
  {
    const pos = runVertexPos(`void main() { float t = 1.5; float f = ++t; gl_Position = vec4(f, t, 0.0, 1.0); }`);
    check(pos[0] === 2.5 && pos[1] === 2.5, `float prefix ++: f=2.5, t=2.5 (got ${pos[0]}, ${pos[1]})`);
  }
  // uint postfix (ES 3.00).
  {
    const pos = runVertexPos(
      `#version 300 es
       void main() { uint m = 24u; uint k = m--; gl_Position = vec4(float(k), float(m), 0.0, 1.0); }`,
      300,
    );
    check(pos[0] === 24 && pos[1] === 23, `uint postfix --: k=24, m=23 (got ${pos[0]}, ${pos[1]})`);
  }
  // Postfix on a swizzle lvalue: result = old component values.
  {
    const pos = runVertexPos(`void main() { vec2 t = vec2(1.0, 2.0); vec2 f = t.xy++; gl_Position = vec4(f.x, f.y + t.x * 10.0 + t.y, 0.0, 1.0); }`);
    check(pos[0] === 1.0 && pos[1] === 2.0 + 2.0 * 10.0 + 3.0, `vec2 postfix swizzle: f=(1,2), t=(2,3) (got ${pos[0]}, ${pos[1]})`);
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`integration selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
