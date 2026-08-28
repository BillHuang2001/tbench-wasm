/**
 * selftest-uint-varying.ts — bit-preserving INT/UINT varying pack
 * (T1-A573 uint, T1-A580 int).
 *
 * INT and UINT varyings pack their 32-bit VALUE's BIT PATTERN into the
 * float32 vertex-record cell (`R.u2f`) instead of the float32-rounded value:
 * values > 2^24 (e.g. gl_VertexID = 0x7FFFFFFD, TF capture of large uints)
 * would otherwise round in the record and corrupt fragment reads and vertex
 * read-backs (CTS vertex-id.html, get-buffer-sub-data-validity.html). The
 * fragment side and vertex read-backs unpack via `R.f2u` (uint32) / `R.f2i`
 * (int32 reinterpret) — the record's float32 bits are the true 32-bit
 * pattern. Writing a signed int to the Uint32Array view wraps via ToUint32
 * (two's-complement bits exact), so one `R.u2f` write serves both families.
 * Integral varyings are flat (GLSL ES 3.00 §4.3.6 — enforced at semantics;
 * VaryingInfo.flat is implied for integral types) so the raster copies cells
 * bit-exact with no interpolation. GLSL ES 1.00 cannot declare int/uint
 * varyings at all ('varying variables must have a float type'), so no v100
 * pin exists.
 *
 * NaN-pattern payloads (exponent 0xFF: 0x7F800000-0x7FFFFFFF,
 * 0xFF800000-0xFFFFFFFF) survive the Float32Array record round-trip
 * BIT-EXACT in V8 (Node + headless Chromium) — sections 7/8 pin the manager-
 * verified payload preservation (0x7FFFFFFD, 0x7FFFFFFF, 0xFFFFFFFF,
 * 0x80000000 → −0.0, 0xFFFFFFFB all bit-exact).
 *
 * Every INT check FAILS on the pre-T1-A580 code (the record held
 * float32-rounded values; fragment reads of |v| > 2^24 were garbage).
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

/** Link a VS/FS pair and run the vertex stage with one INT attribute value. */
function runVertexInt(
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
    attribs: [new Int32Array([a])],
    attribIndices: new Int32Array([0]),
  });
  prog.vertex.run(vctx);
  const vg = vctx.out.varyings as Float32Array;
  return { prog, vctx, vg };
}

const BIG = 0x04030201; // 67305985 > 2^24 — float32 rounding would corrupt it
const R = {
  f2u: (x: number) => new Uint32Array(new Float32Array([x]).buffer)[0],
  f2i: (x: number) => new Int32Array(new Float32Array([x]).buffer)[0],
};

/* ------------------------------------------------------------------ */
/* 1. Bit-exact pack + TF-capture simulation (CTS get-buffer-sub-data) */
/* ------------------------------------------------------------------ */

{
  const fsSrc = `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = (v == ${BIG}u) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0); }`;
  const { prog, vg } = (() => {
    const vs = compile(
      `#version 300 es
       in uint a;
       flat out uint v;
       void main(){ v = a; gl_Position = vec4(0.0); }`,
      'VERTEX',
      300,
    );
    const fs = compile(fsSrc, 'FRAGMENT', 300);
    const l = linkProgram(vs, fs);
    if (!l.ok) throw new Error(`link failed: ${l.log}`);
    const prog = l.program;
    const vctx = vertexCtx(prog, {
      attribs: [new Uint32Array([BIG])],
      attribIndices: new Int32Array([0]),
    });
    prog.vertex.run(vctx);
    return { prog, vg: vctx.out.varyings as Float32Array };
  })();
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
  const { prog, vctx } = runVertexInt(
    `#version 300 es
     in int a;
     flat out uint v;
     void main(){
       v = uint(a);
       // Vertex read-back of the packed varying must equal the source.
       if (v != uint(a)) { gl_Position = vec4(1.0); return; }
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
  void prog;
}

/* ------------------------------------------------------------------ */
/* 3. Compound assign / ++/-- on a packed uint varying                  */
/* ------------------------------------------------------------------ */

{
  const { prog, vg } = runVertexInt(
    `#version 300 es
     in int a;
     flat out uint v;
     void main(){ v = uint(a); v += 1u; }`,
    `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    BIG,
  );
  check(R.f2u(vg[0]) === ((BIG + 1) & 0xffffffff),
    `'v += 1u' packs 0x${((BIG + 1) >>> 0).toString(16)} (got f2u=${R.f2u(vg[0])})`);

  const { prog: p2, vg: vg2 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out uint v;
     void main(){ v = uint(a); v++; }`,
    `#version 300 es
     precision mediump float;
     flat in uint v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    BIG,
  );
  check(R.f2u(vg2[0]) === ((BIG + 1) & 0xffffffff),
    `statement 'v++' packs 0x${((BIG + 1) >>> 0).toString(16)} (got f2u=${R.f2u(vg2[0])})`);

  const { prog: p3, vctx: vctx3, vg: vg3 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out uint v;
     void main(){ v = uint(a); uint w = ++v; gl_Position = vec4(w == ${BIG + 1}u ? 2.0 : 1.0, 0.0, 0.0, 1.0); }`,
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
/* 5. INT varying bit-packs its two's-complement bits (T1-A580)         */
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
    // -5 → 0xFFFFFFFB (a NaN-pattern float) — bit-exact in the V8 store.
    check(R.f2u(vg[0]) === 0xfffffffb,
      `int varying packs BIT PATTERN of -5 (0xFFFFFFFB) (record float=${vg[0]}, f2u=${R.f2u(vg[0])})`);
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
/* 7. THE vertex-id value: int 2147483645 (0x7FFFFFFD) bit-exact pack   */
/*    + TF-capture byte simulation (raw LE bytes of the int32)          */
/* ------------------------------------------------------------------ */

{
  const { prog, vg } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; gl_Position = vec4(0.0); }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = (v == 2147483645) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0); }`,
    2147483645,
  );
  check(prog.varyings[0].flat === true && prog.varyings[0].components === 1,
    `int varying flat flag + components (got ${JSON.stringify(prog.varyings)})`);
  check(R.f2u(vg[0]) === 0x7ffffffd,
    `int varying packs 2147483645 BIT PATTERN 0x7FFFFFFD (record float=${vg[0]}, f2u=${R.f2u(vg[0])})`);

  // TF capture byte simulation: the record cell's float32 bits ARE the int32
  // bits (LE byte order — as CTS getBufferSubData reads).
  const tfBytes = new Uint8Array(new Float32Array([vg[0]]).buffer);
  const expect = [0xfd, 0xff, 0xff, 0x7f]; // LE bytes of int32 0x7FFFFFFD
  check(
    tfBytes[0] === expect[0] && tfBytes[1] === expect[1] &&
      tfBytes[2] === expect[2] && tfBytes[3] === expect[3],
    `int TF capture bytes = [FD,FF,FF,7F] (got [${Array.from(tfBytes).map((b) => b.toString(16)).join(',')}])`,
  );

  const fctx = fragmentCtx(prog, [{ v: vg.subarray(0, 1) }]);
  prog.fragment.run(fctx);
  check(fctx.out.color[0][0] === 1,
    `fragment reads int 2147483645 back bit-exact (color [${Array.from(fctx.out.color[0]).join(', ')}])`);
}

/* ------------------------------------------------------------------ */
/* 8. INT value matrix: record bits + vertex read-back + assignment     */
/*    value + fragment round-trip for NaN-pattern / edge values         */
/* ------------------------------------------------------------------ */

{
  // 2147483647 (0x7FFFFFFF), -1 (0xFFFFFFFF), -2147483648 (0x80000000 →
  // float -0.0), -5 (0xFFFFFFFB), 2147483645 (0x7FFFFFFD).
  const values: { v: number; bits: number; fsLit: string }[] = [
    { v: 2147483647, bits: 0x7fffffff, fsLit: '2147483647' },
    { v: -1, bits: 0xffffffff, fsLit: '-1' },
    { v: -2147483648, bits: 0x80000000, fsLit: '(-2147483647 - 1)' },
    { v: -5, bits: 0xfffffffb, fsLit: '-5' },
    { v: 2147483645, bits: 0x7ffffffd, fsLit: '2147483645' },
  ];
  for (const { v, bits, fsLit } of values) {
    const { prog, vctx, vg } = runVertexInt(
      `#version 300 es
       in int a;
       flat out int v;
       void main(){
         v = a;
         // Vertex read-back of the packed varying must equal the source.
         if (v != a) { gl_Position = vec4(1.0); return; }
         // The assignment-expression VALUE is the assigned int (unpacked).
         int w = (v = a);
         gl_Position = vec4(w == a ? 2.0 : 1.0, 0.0, 0.0, 1.0);
       }`,
      `#version 300 es
       precision mediump float;
       flat in int v;
       out vec4 o;
       void main(){ o = (v == ${fsLit}) ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0); }`,
      v,
    );
    check(R.f2u(vg[0]) === bits,
      `int ${v} packs bits 0x${bits.toString(16)} (record float=${vg[0]}, f2u=${R.f2u(vg[0])})`);
    check(vctx.out.position[0] === 2,
      `int ${v} vertex read-back + assignment value correct (position.x=${vctx.out.position[0]})`);
    const fctx = fragmentCtx(prog, [{ v: vg.subarray(0, 1) }]);
    prog.fragment.run(fctx);
    check(fctx.out.color[0][0] === 1,
      `int ${v} fragment round-trip bit-exact (color [${Array.from(fctx.out.color[0]).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* 9. Compound ops / ++/-- on a packed INT varying (int vs uint ops)    */
/* ------------------------------------------------------------------ */

{
  // `+=` (bit-identical to uint)
  const { vg: vg1 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; v += 1; }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    -5,
  );
  check(R.f2u(vg1[0]) === 0xfffffffc,
    `int 'v += 1' on -5 packs 0xFFFFFFFC (got f2u=${R.f2u(vg1[0])})`);

  // `*=` via Math.imul (large operands would lose low bits with plain JS `*`)
  const { vg: vg2 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; v *= 3; }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    BIG,
  );
  check(R.f2u(vg2[0]) === 0x0c090603,
    `int 'v *= 3' on 0x04030201 uses Math.imul → 0x0C090603 (got f2u=${R.f2u(vg2[0])})`);

  // `>>=` ARITHMETIC shift (uint would be logical): -7 >> 2 = -2
  const { vg: vg3 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; v >>= 2; }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    -7,
  );
  check(R.f2i(vg3[0]) === -2,
    `int 'v >>= 2' on -7 is ARITHMETIC → -2 (got f2i=${R.f2i(vg3[0])})`);

  // `/=` truncates toward zero: -7 / 3 = -2 (uint would floor)
  const { vg: vg4 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; v /= 3; }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    -7,
  );
  check(R.f2i(vg4[0]) === -2,
    `int 'v /= 3' on -7 truncates toward zero → -2 (got f2i=${R.f2i(vg4[0])})`);

  // `%=` sign-of-dividend: -7 % 3 = -1
  const { vg: vg5 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; v %= 3; }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    -7,
  );
  check(R.f2i(vg5[0]) === -1,
    `int 'v %= 3' on -7 keeps the dividend sign → -1 (got f2i=${R.f2i(vg5[0])})`);

  // Statement `v++` on 2147483645 → 2147483646 (0x7FFFFFFE)
  const { vg: vg6 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; v++; }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    2147483645,
  );
  check(R.f2u(vg6[0]) === 0x7ffffffe,
    `statement 'v++' on 2147483645 packs 0x7FFFFFFE (got f2u=${R.f2u(vg6[0])})`);

  // Prefix `++v` value (new value) + pack
  const { vctx: vctx7, vg: vg7 } = runVertexInt(
    `#version 300 es
     in int a;
     flat out int v;
     void main(){ v = a; int w = ++v; gl_Position = vec4(w == a + 1 ? 2.0 : 1.0, 0.0, 0.0, 1.0); }`,
    `#version 300 es
     precision mediump float;
     flat in int v;
     out vec4 o;
     void main(){ o = vec4(0.0); }`,
    2147483645,
  );
  check(R.f2u(vg7[0]) === 0x7ffffffe && vctx7.out.position[0] === 2,
    `prefix '++v' value (new) + pack on 2147483645 (value w=${vctx7.out.position[0] === 2 ? 'new' : 'WRONG'}, f2u=${R.f2u(vg7[0])})`);
}

/* ------------------------------------------------------------------ */
/* 10. ivec2 per-component pack + const-indexed int array element       */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     in int a;
     flat out ivec2 vPair;
     flat out int vArr[2];
     void main(){
       vPair = ivec2(a, -1);
       vArr[1] = -2147483648;
       gl_Position = vec4(0.0);
     }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     flat in ivec2 vPair;
     flat in int vArr[2];
     out vec4 o;
     void main(){
       o = (vPair.x == 2147483645 && vPair.y == -1 && vArr[1] == -2147483648)
         ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
     }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `ivec2/array pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const prog = l.program;
    const vctx = vertexCtx(prog, { attribs: [new Int32Array([2147483645])], attribIndices: new Int32Array([0]) });
    prog.vertex.run(vctx);
    const vg = vctx.out.varyings;
    // Packed order: vPair.x, vPair.y, vArr[0], vArr[1]
    check(R.f2u(vg[0]) === 0x7ffffffd && R.f2u(vg[1]) === 0xffffffff,
      `ivec2 per-component bit-exact (f2u=[${R.f2u(vg[0])},${R.f2u(vg[1])}])`);
    check(R.f2u(vg[3]) === 0x80000000,
      `const-indexed int array element packs -2147483648 bits (f2u=${R.f2u(vg[3])})`);
    const fctx = fragmentCtx(prog, [
      { v: vg.subarray(0, 2) },
      { v: vg.subarray(2, 4) },
    ]);
    prog.fragment.run(fctx);
    check(fctx.out.color[0][0] === 1,
      `fragment round-trip of ivec2 + array (color [${Array.from(fctx.out.color[0]).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* 11. Dual-mode fragment (derivatives) still unpacks the INT varying   */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     in int a;
     flat out int v;
     out vec2 p;
     void main(){ v = a; p = vec2(0.5); gl_Position = vec4(0.0); }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     flat in int v;
     in vec2 p;
     out vec4 o;
     void main(){ o = vec4(dFdx(p.x), v == 2147483645 ? 1.0 : 0.0, 0.0, 1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `dual-mode int pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const prog = l.program;
    const vctx = vertexCtx(prog, { attribs: [new Int32Array([2147483645])], attribIndices: new Int32Array([0]) });
    prog.vertex.run(vctx);
    const vg = vctx.out.varyings;
    const fctx = fragmentCtx(prog, [
      { v: vg.subarray(0, 1), ddx: new Float32Array(1), ddy: new Float32Array(1) },
      { v: vg.subarray(1, 3), ddx: new Float32Array(2), ddy: new Float32Array(2) },
    ]);
    prog.fragment.run(fctx);
    check(fctx.out.color[0][1] === 1,
      `dual-mode fragment unpacks the int varying (color [${Array.from(fctx.out.color[0]).join(', ')}])`);
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
