/**
 * selftest-uint-varying.ts — bit-preserving UINT varying pack (T1-A573).
 *
 * UINT varyings pack their 32-bit VALUE's BIT PATTERN into the float32
 * vertex-record cell (`R.u2f`) instead of the float32-rounded value, so the
 * gl TF capture (draw.ts captureTransformFeedback — copies the record bits
 * RAW into the TF buffer) yields exact uint32s for getBufferSubData
 * (CTS get-buffer-sub-data-validity.html: srcData [1..8] read back through
 * a `flat out uint` transform-feedback varying). The fragment side and
 * vertex read-backs unpack via `R.f2u`. INT varyings stay float32-VALUE
 * packed (the int32 sign bit is the float32 sign bit — small negatives
 * would land on exponent-0xFF NaN bit patterns and corrupt in the
 * Float32Array store).
 *
 * Every check FAILS on the pre-fix code (the record holds float32-rounded
 * values; TF/fragment reads of values > 2^24 are garbage).
 *
 * Run: npx tsx src/glsl/codegen/selftest-uint-varying.ts
 * Prints "selftest-uint-varying: N checks" and exits 0 only when all pass.
 */
import { compileShader, linkProgram } from '../compiler.js';
import type { Program } from '../program.js';

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

function fragmentCtx(
  prog: Program,
  varyings: Record<string, unknown>[] = [],
  extra: Record<string, unknown> = {},
): any {
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

/** Link a VS/FS pair and run the vertex stage with one uint attribute value. */
function runVertex(
  vsSrc: string,
  fsSrc: string,
  a: number,
): { prog: Program; vctx: any; vg: Float32Array } {
  const vs = compile(vsSrc, 'VERTEX', 300);
  const fs = compile(fsSrc, 'FRAGMENT', 300);
  const l = linkProgram(vs, fs);
  if (!l.ok) throw new Error(`link failed: ${l.log}`);
  const prog = l.program;
  const vctx = vertexCtx(prog, {
    attribs: [new Uint32Array([a])],
    attribIndices: new Int32Array([0]),
  });
  prog.vertex.run(vctx);
  const vg = vctx.out.varyings as Float32Array;
  return { prog, vctx, vg };
}

const BIG = 0x04030201; // 67305985 > 2^24 — float32 rounding would corrupt it
const R = { f2u: (x: number) => new Uint32Array(new Float32Array([x]).buffer)[0] };

/* ------------------------------------------------------------------ */
/* 1. Bit-exact pack + TF-capture simulation (CTS get-buffer-sub-data) */
/* ------------------------------------------------------------------ */

{
  const fsSrc = `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = (v == ${BIG}u) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0); }`;
  const { prog, vg } = runVertex(
    `#version 300 es
     in uint a;
     flat out uint v;
     void main(){ v = a; gl_Position = vec4(0.0); }`,
    fsSrc,
    BIG,
  );
  check(prog.varyings.length === 1 && prog.varyings[0].flat === true && prog.varyings[0].components === 1,
    `varying flat flag + components (got ${JSON.stringify(prog.varyings)})`);
  check(R.f2u(vg[0]) === BIG,
    `uint varying packs BIT PATTERN 0x${BIG.toString(16)} (record float=${vg[0]}, f2u=${R.f2u(vg[0])})`);

  // Simulate draw.ts captureTransformFeedback: copy the record's float32
  // bits RAW into a byte buffer (little-endian platform — as CTS reads).
  const tfBytes = new Uint8Array(new Float32Array([vg[0]]).buffer);
  const expect = [1, 2, 3, 4]; // LE bytes of 0x04030201
  check(
    tfBytes.length === 4 && tfBytes[0] === expect[0] && tfBytes[1] === expect[1] &&
      tfBytes[2] === expect[2] && tfBytes[3] === expect[3],
    `TF capture bytes = [1,2,3,4] (got [${Array.from(tfBytes).join(',')}])`,
  );

  const fctx = fragmentCtx(prog, [{ v: vg.subarray(0, 1) }]);
  prog.fragment.run(fctx);
  check(fctx.out.color[0][0] === 1,
    `fragment reads the uint varying back bit-exact (color [${Array.from(fctx.out.color[0]).join(', ')}])`);
}

/* ------------------------------------------------------------------ */
/* 2. Vertex read-back + assignment-expression value                    */
/* ------------------------------------------------------------------ */

{
  const { prog, vctx } = runVertex(
    `#version 300 es
     in uint a;
     flat out uint v;
     void main(){
       v = a;
       // Vertex read-back of the packed varying must equal the source.
       if (v != a) { gl_Position = vec4(1.0); return; }
       // The assignment-expression VALUE is the assigned uint (unpacked).
       uint w = (v = ${BIG}u);
       gl_Position = vec4(w == ${BIG}u ? 2.0 : 1.0, 0.0, 0.0, 1.0);
     }`,
    `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = vec4(0.0, 0.0, 0.0, 1.0); }`,
    BIG,
  );
  check(vctx.out.position[0] === 2,
    `vertex read-back + assignment value correct (position.x=${vctx.out.position[0]})`);
}

/* ------------------------------------------------------------------ */
/* 3. Compound assign / ++/-- on a packed uint varying                  */
/* ------------------------------------------------------------------ */

{
  const { prog, vg } = runVertex(
    `#version 300 es
     in uint a;
     flat out uint v;
     void main(){ v = a; v += 1u; }`,
    `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    BIG,
  );
  check(R.f2u(vg[0]) === ((BIG + 1) & 0xffffffff),
    `'v += 1u' packs 0x${((BIG + 1) >>> 0).toString(16)} (got f2u=${R.f2u(vg[0])})`);

  const { prog: p2, vg: vg2 } = runVertex(
    `#version 300 es
     in uint a;
     flat out uint v;
     void main(){ v = a; v++; }`,
    `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    BIG,
  );
  check(R.f2u(vg2[0]) === ((BIG + 1) & 0xffffffff),
    `statement 'v++' packs 0x${((BIG + 1) >>> 0).toString(16)} (got f2u=${R.f2u(vg2[0])})`);

  const { prog: p3, vctx: vctx3, vg: vg3 } = runVertex(
    `#version 300 es
     in uint a;
     flat out uint v;
     void main(){ v = a; uint w = ++v; gl_Position = vec4(w == ${BIG + 1}u ? 2.0 : 1.0, 0.0, 0.0, 1.0); }`,
    `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    BIG,
  );
  check(R.f2u(vg3[0]) === ((BIG + 1) & 0xffffffff) && vctx3.out.position[0] === 2,
    `prefix '++v' value + pack (value w=${vctx3.out.position[0] === 2 ? 'new' : 'WRONG'}, f2u=${R.f2u(vg3[0])})`);
  void p2;
  void prog;
}

/* ------------------------------------------------------------------ */
/* 4. Small uint + uvec2 per-component pack + const-indexed array       */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     in uint a;
     flat out uint vSmall;
     flat out uvec2 vPair;
     flat out uint vArr[2];
     void main(){
       vSmall = 5u;
       vPair = uvec2(a, ${0x08070605}u);
       vArr[1] = a;
       gl_Position = vec4(0.0);
     }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     flat in uint vSmall;
     flat in uvec2 vPair;
     flat in uint vArr[2];
     out vec4 o;
     void main(){
       o = (vSmall == 5u && vPair.y == ${0x08070605}u && vArr[1] == ${BIG}u)
         ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
     }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `uvec2/array pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const prog = l.program;
    const vctx = vertexCtx(prog, { attribs: [new Uint32Array([BIG])], attribIndices: new Int32Array([0]) });
    prog.vertex.run(vctx);
    const vg = vctx.out.varyings;
    // Packed order: vSmall, vPair.x, vPair.y, vArr[0], vArr[1]
    check(R.f2u(vg[0]) === 5, `small uint 5 packs bit-exact (f2u=${R.f2u(vg[0])})`);
    check(R.f2u(vg[1]) === BIG && R.f2u(vg[2]) === 0x08070605,
      `uvec2 per-component bit-exact (f2u=[${R.f2u(vg[1])},${R.f2u(vg[2])}])`);
    check(R.f2u(vg[4]) === BIG, `const-indexed array element packs bit-exact (f2u=${R.f2u(vg[4])})`);
    const fctx = fragmentCtx(prog, [
      { v: vg.subarray(0, 1) },
      { v: vg.subarray(1, 3) },
      { v: vg.subarray(3, 5) },
    ]);
    prog.fragment.run(fctx);
    check(fctx.out.color[0][0] === 1,
      `fragment round-trip of small uint + uvec2 + array (color [${Array.from(fctx.out.color[0]).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* 5. INT varyings stay float32-VALUE packed (no NaN-bit regression)    */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     flat out int vI;
     void main(){ vI = -5; gl_Position = vec4(0.0); }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     flat in int vI;
     out vec4 o;
     void main(){ o = vec4(float(vI), 0.0, 0.0, 1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `int varying pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const prog = l.program;
    const vctx = vertexCtx(prog);
    prog.vertex.run(vctx);
    const vg = vctx.out.varyings;
    check(vg[0] === -5, `int varying keeps float32 VALUE -5 (got ${vg[0]})`);
    const fctx = fragmentCtx(prog, [{ v: vg.subarray(0, 1) }]);
    prog.fragment.run(fctx);
    check(fctx.out.color[0][0] === -5, `int varying fragment round-trip -5 (got ${fctx.out.color[0][0]})`);
  }
}

/* ------------------------------------------------------------------ */
/* 6. Dual-mode fragment (derivatives) still unpacks the uint varying   */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     in uint a;
     flat out uint v;
     out vec2 p;
     void main(){ v = a; p = vec2(0.5); gl_Position = vec4(0.0); }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     flat in uint v;
     in vec2 p;
     out vec4 o;
     void main(){ o = vec4(dFdx(p.x), v == ${BIG}u ? 1.0 : 0.0, 0.0, 1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `dual-mode pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const prog = l.program;
    const vctx = vertexCtx(prog, { attribs: [new Uint32Array([BIG])], attribIndices: new Int32Array([0]) });
    prog.vertex.run(vctx);
    const vg = vctx.out.varyings;
    const fctx = fragmentCtx(prog, [
      { v: vg.subarray(0, 1), ddx: new Float32Array(1), ddy: new Float32Array(1) },
      { v: vg.subarray(1, 3), ddx: new Float32Array(2), ddy: new Float32Array(2) },
    ]);
    prog.fragment.run(fctx);
    check(fctx.out.color[0][1] === 1,
      `dual-mode fragment unpacks the uint varying (color [${Array.from(fctx.out.color[0]).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-uint-varying: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
