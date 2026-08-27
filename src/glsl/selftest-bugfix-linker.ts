/**
 * selftest-bugfix-linker.ts — regression pin for BUG 4: emitType's
 * float-store high-water underallocation for matrix ARRAYS.
 *
 * Root cause (src/glsl/linker.ts, emitType): the default-block float-store
 * high-water (floatMax) was sized for a matrix with `cols * rows` floats, but
 * matrix storage is COLUMN-MAJOR with vec4 columns — each matrix column
 * occupies 4 floats in the store (slot = cursor*4, column stride 4). A mat3
 * therefore needs 12 floats, not 9; with the wrong formula the
 * Float32Array(floatMax) came out short and getUniform read past the end →
 * NaN (diag page conformance2/uniforms/gl-uniform-arrays-sub-source.html:
 * mat3/mat2x3/mat3x2/mat4x2/mat4x3 arrays).
 *
 * This script mirrors that diag page's matrix-array uniform set:
 *   uniform mat3 u[2]; uniform mat2x3 v[1]; uniform mat3x2 w[1];
 *   uniform mat4x2 x[1]; uniform mat4x3 y[1];
 * and pins (a) each flattened uniform entry's location/type/size, (b) the
 * exact store high-water — the last leaf (y[0], 60 + 4*4 = 76 floats) must be
 * covered, pre-fix it stopped at 72 — and (c) an end-to-end vertex run that
 * reads the LAST column of every leaf (y[0][3][2] lives at float index 74,
 * beyond the pre-fix store → NaN propagates into gl_Position).
 *
 * Run: npx tsx src/glsl/selftest-bugfix-linker.ts
 * Prints "OK" and exits 0 on success.
 */
import { compileShader, linkProgram } from './compiler.js';
import type { LinkResult } from './compiler.js';
import type { Program } from './program.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function compile(src: string, type: 'VERTEX' | 'FRAGMENT', version: 100 | 300) {
  const r = compileShader(src, { type, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  return r.shader;
}

function vertexCtx(prog: Program, attribs: Float32Array): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { position: new Float32Array(4), pointSize: 0, varyings: new Float32Array(32) },
    attribs: [attribs],
    attribIndices: new Int32Array([0]),
    vertexId: 0,
    instanceId: 0,
  };
}

/* GL enums (types.ts MATRIX_ENUM). */
const FLOAT_MAT3 = 0x8b5b;
const FLOAT_MAT2x3 = 0x8b65;
const FLOAT_MAT3x2 = 0x8b67;
const FLOAT_MAT4x2 = 0x8b69;
const FLOAT_MAT4x3 = 0x8b6a;

/* ------------------------------------------------------------------ */
/* 1. ES 3.00 matrix-array set (diag-page mirror)                      */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     uniform mat3 u[2];
     uniform mat2x3 v[1];
     uniform mat3x2 w[1];
     uniform mat4x2 x[1];
     uniform mat4x3 y[1];
     in vec4 aPos;
     void main() {
       gl_Position = aPos + vec4(u[1][2][2] + u[0][2][2], v[0][1][2] + y[0][3][2], w[0][2][1] + x[0][3][1], 0.0);
     }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     out vec4 oColor;
     void main() { oColor = vec4(1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `matrix-array pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;

    // (a) Flattened getActiveUniform entries: arrays are ONE entry 'X[0]'
    // with size = array length; location = float index of the FIRST element.
    // Layout (declaration order, one vec4-slot cursor):
    //   u mat3[2]   cursor 0..6   (u[0]@0,  u[1]@12,  element 12 floats)
    //   v mat2x3[1] cursor 6..8   (v[0]@24,  element 8 floats)
    //   w mat3x2[1] cursor 8..11  (w[0]@32,  element 12 floats)
    //   x mat4x2[1] cursor 11..15 (x[0]@44,  element 16 floats)
    //   y mat4x3[1] cursor 15..19 (y[0]@60,  element 16 floats)
    const expectEntries: { name: string; location: number; type: number; size: number }[] = [
      { name: 'u[0]', location: 0, type: FLOAT_MAT3, size: 2 },
      { name: 'v[0]', location: 24, type: FLOAT_MAT2x3, size: 1 },
      { name: 'w[0]', location: 32, type: FLOAT_MAT3x2, size: 1 },
      { name: 'x[0]', location: 44, type: FLOAT_MAT4x2, size: 1 },
      { name: 'y[0]', location: 60, type: FLOAT_MAT4x3, size: 1 },
    ];
    for (const e of expectEntries) {
      const u = p.uniforms.find((x) => x.name === e.name);
      check(
        !!u && u.location === e.location && u.type === e.type && u.size === e.size,
        `uniform entry '${e.name}' @${e.location} type 0x${e.type.toString(16)} size ${e.size} (got ${JSON.stringify(u)})`,
      );
    }

    // (b) THE regression check: the float store (Float32Array(floatMax)) must
    // cover the LAST column of every matrix-array leaf. Element k of an array
    // sits at location + k*cols*4; each leaf occupies cols*4 floats (matrices
    // are never int-store). High-water = 76 (pre-fix cols*rows: 72).
    const leafEnds: { key: string; cols: number; end: number }[] = [
      { key: 'u[0]', cols: 3, end: 12 },
      { key: 'u[1]', cols: 3, end: 24 },
      { key: 'v[0]', cols: 2, end: 32 },
      { key: 'w[0]', cols: 3, end: 44 },
      { key: 'x[0]', cols: 4, end: 60 },
      { key: 'y[0]', cols: 4, end: 76 },
    ];
    let maxLeafEnd = 0;
    for (const e of leafEnds) {
      const info = p.uniformMap.get(e.key);
      check(!!info, `uniformMap has leaf '${e.key}'`);
      if (!info) continue;
      const end = info.location + e.cols * 4;
      maxLeafEnd = Math.max(maxLeafEnd, end);
      check(
        end === e.end && end <= p.floatStore.length,
        `leaf '${e.key}' covered: location ${info.location} + ${e.cols}*4 = ${end} <= floatStore.length ${p.floatStore.length} (expected end ${e.end})`,
      );
    }
    check(
      p.floatStore.length === maxLeafEnd && p.floatStore.length === 76,
      `floatStore.length ${p.floatStore.length} === matrix-array high-water 76 (pre-fix was 72 — NaN reads)`,
    );

    // (c) End-to-end: write a distinct value into the LAST column of every
    // leaf and run the vertex shader. y[0][3][2] lives at float index 74 —
    // past the pre-fix store (72) → out-of-bounds read → NaN in the result.
    const cells: { key: string; col: number; row: number; value: number }[] = [
      { key: 'u[1]', col: 2, row: 2, value: 1 },
      { key: 'u[0]', col: 2, row: 2, value: 0.5 },
      { key: 'v[0]', col: 1, row: 2, value: 2 },
      { key: 'w[0]', col: 2, row: 1, value: 3 },
      { key: 'x[0]', col: 3, row: 1, value: 4 },
      { key: 'y[0]', col: 3, row: 2, value: 5 },
    ];
    for (const c of cells) {
      const info = p.uniformMap.get(c.key)!;
      p.floatStore[info.location + c.col * 4 + c.row] = c.value;
    }
    const vctx = vertexCtx(p, new Float32Array([1, 2, 3, 4]));
    p.vertex.run(vctx);
    const pos = vctx.out.position;
    // x: 1 + (1 + 0.5) = 2.5; y: 2 + (2 + 5) = 9; z: 3 + (3 + 4) = 10; w: 4
    check(
      pos[0] === 2.5 && pos[1] === 9 && pos[2] === 10 && pos[3] === 4,
      `vertex run reads every matrix-array leaf column: position [2.5, 9, 10, 4] (got [${Array.from(pos).join(', ')}]${Number.isNaN(pos[1]) ? ' — NaN: float store too short' : ''})`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 2. ES 1.00 variant (matrices in the WebGL1 default block)           */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `uniform mat3 u[2];
     attribute vec4 aPos;
     varying vec4 vC;
     void main() { vC = vec4(u[1][2][2], 0.0, 0.0, 1.0); gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fs = compile(
    `precision mediump float;
     varying vec4 vC;
     void main() { gl_FragColor = vC; }`,
    'FRAGMENT',
    100,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `ES 1.00 mat3[2] pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // u mat3[2]: cursor 0..6; u[1]@12, element 12 floats → high-water 24
    // (pre-fix cols*rows: 2*9=18... high-water 21).
    const u1 = p.uniformMap.get('u[1]');
    check(!!u1 && u1.location === 12, `ES 1.00 'u[1]' located at 12 (got ${u1?.location})`);
    check(
      p.floatStore.length === 24,
      `ES 1.00 floatStore.length ${p.floatStore.length} === 24 (mat3[2] = 2*12 floats; pre-fix was 21)`,
    );
    p.floatStore[12 + 2 * 4 + 2] = 7; // u[1][2][2] — index 22, past pre-fix high-water 21
    const vctx = vertexCtx(p, new Float32Array([1, 1, 1, 1]));
    p.vertex.run(vctx);
    check(
      vctx.out.varyings[0] === 7 && !Number.isNaN(vctx.out.varyings[0]),
      `ES 1.00 vertex reads u[1][2][2] = 7 (got ${vctx.out.varyings[0]})`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`bugfix-linker selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
