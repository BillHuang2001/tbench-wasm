/**
 * selftest-link.ts — linker checks (task E5a): `linkProgram` compiles a
 * vertex+fragment pair into an executable `Program` (uniform merge +
 * vec4-slot layout, varying match + pack, attribute/output location
 * assignment, limit checks, JS codegen) and the linked stages RUN against
 * hand-built exec ctxs (same ctx shapes as selftest-codegen-stage.ts).
 *
 * Coverage (one block per check):
 *   1. trivial ES 1.00 pair: attrib vec4 + uniform vec4 + varying +
 *      gl_FragColor — link, run vertex+fragment, verify position/color, then
 *      write uniforms via floatStore[location + c] and re-run.
 *   2. ES 3.00 pair: in/out, layout(location=2) attrib (verify location 2),
 *      out vec4, gl_FragDepth write.
 *   3. struct uniform + struct-array uniform + struct varying round-trip
 *      (incl. DYNAMIC struct-array element index); uniformMap keys
 *      'u' → first element, 'u[0]' → first element, 'u.m', 'u[0].m'.
 *   4. link errors: varying name mismatch / type mismatch / flat mismatch,
 *      uniform type conflict, duplicate attribute location,
 *      maxVaryingVectors exceeded (opts.limits).
 *   5. bindAttribLocation (opts.attribBindings) → location 3; unbound
 *      attribute auto-assigns first-free.
 *   6. user function + local array through the linked program (scratchSize>0).
 *   7. ES 3.00 flat ivec4 varying round-trip (small values).
 *   8. UBO declared → links with std140 block metadata (E5b1 lifted the
 *      'uniform blocks not supported' deferral).
 *   9. std140 layout + full run: {float a; vec3 b; mat4 c; float d[3]; int e;}
 *      (a@0, b@16, c@32, d@96, e@108, size 112), float + int block stores,
 *      UBO members in Program.uniforms (location -1) but NOT uniformMap.
 *  10. same block in both stages → ONE shared index + run; layout mismatch
 *      (float vs vec2 member) → link error.
 *  11. instance-less block accessed by bare member name.
 *  12. arrayed block `uniform B { vec4 v; } b[2]` with DYNAMIC instance index
 *      b[i].v (i uniform; i=0 and i=1 verify the blockStride path).
 *  13. nested struct members + member array inside a struct:
 *      {S1 {vec2 x; float y;} s; S2 {vec2 x; float y[2];} t;} — x@0, y@8,
 *      t@16, t.x@16, t.y@24, size 48.
 *  14. limits: maxUniformBlockSize / maxVertexUniformBlocks /
 *      maxCombinedUniformBlocks exceeded → link errors.
 *  15. struct MEMBER arrays stay whole in the active-uniform layout (CTS
 *      shader-with-array-of-structs-containing-arrays.html): `my_struct
 *      {vec4 color1[2]; vec4 color2[2];} u_colors[2]` → 4 entries
 *      'u_colors[i].colorN[0]' size 2; uniformMap aliases '<p>.m' + '<p>.m[k]'
 *      → the leaf; round-trip run.
 *  16. struct-ELEMENT member arrays still expand per element ('u.a[0].c',
 *      'u.a[1].c' — a struct-array member has no GLenum for one entry).
 *  17. UBO arrays of structs (uniform-blocks-with-arrays.html crash pin):
 *      struct members flatten per element ('d.s.a', 'd.s.b'), arrays of
 *      structs expand EVERY element ('lights[0].intensity'/'lights[1].intensity'
 *      @0/@16, size 1), structs containing arrays recurse
 *      ('lights[0].intensity[0]' size 3 stride 16, offsets 0/48), and the
 *      arrayed-block variant ('ld[0].lights[0].intensity' ... @0/16/32/48).
 *  18. gl_DepthRange builtin uniform reflection: usage-gated active-uniform
 *      entries ('gl_DepthRange.near/far/diff', GL_FLOAT, size 1) backed by 3
 *      real float-store slots appended after user uniforms (1.00 vertex /
 *      3.00 both-stages / 1.00 fragment-only / not-used cases).
 *
 * Run: npx tsx src/glsl/selftest-link.ts
 * Prints "OK" and exits 0 on success.
 */
import { compileShader, linkProgram } from './compiler.js';
import { collectStructNames } from './linker.js';
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

/** Error-path helper: the log string of a failed link ('' when ok). */
function logOf(l: LinkResult): string {
  return l.ok ? '' : l.log;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

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
  varyings: Float32Array[] = [],
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
    varyings: varyings.map((v) => ({ v })),
    ...extra,
  };
}

const FLOAT_VEC4 = 0x8b52; // 35666
const FLOAT_VEC3 = 0x8b51; // 35665
const FLOAT_VEC2 = 0x8b50; // 35664
const FLOAT_MAT4 = 0x8b5c; // 35676
const FLOAT = 0x1406; // 5126
const INT = 0x1404; // 5124

/* ------------------------------------------------------------------ */
/* 1. Trivial ES 1.00 pair + uniform write re-run                      */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `attribute vec4 aPos;
     uniform vec4 uColor;
     varying vec4 vColor;
     void main() { vColor = uColor; gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fs = compile(
    `precision mediump float;
     varying vec4 vColor;
     void main() { gl_FragColor = vColor; }`,
    'FRAGMENT',
    100,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `trivial pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.attributes.length === 1 && p.attributes[0].name === 'aPos' && p.attributes[0].location === 0,
      `attribute 'aPos' auto-located at 0 (got ${JSON.stringify(p.attributes)})`);
    check(p.uniforms.length === 1 && p.uniforms[0].name === 'uColor' && p.uniforms[0].type === FLOAT_VEC4 && p.uniforms[0].size === 1,
      `uniform 'uColor' one entry, FLOAT_VEC4, size 1 (got ${JSON.stringify(p.uniforms)})`);
    check(p.varyings.length === 1 && p.varyings[0].name === 'vColor' && p.varyings[0].components === 4,
      `varying 'vColor' packed (got ${JSON.stringify(p.varyings)})`);

    const uLoc = p.uniformMap.get('uColor')!.location;
    p.floatStore[uLoc + 0] = 0.25;
    p.floatStore[uLoc + 1] = 0.5;
    p.floatStore[uLoc + 2] = 0.75;
    p.floatStore[uLoc + 3] = 1.0;
    const vctx = vertexCtx(p, { attribs: [new Float32Array([1, 2, 3, 4])], attribIndices: new Int32Array([0]) });
    p.vertex.run(vctx);
    const pos = vctx.out.position;
    check(pos[0] === 1 && pos[1] === 2 && pos[2] === 3 && pos[3] === 4,
      `vertex run: position [1,2,3,4] (got [${Array.from(pos).join(', ')}])`);
    const vg = vctx.out.varyings;
    check(vg[0] === 0.25 && vg[1] === 0.5 && vg[2] === 0.75 && vg[3] === 1,
      `vertex packs vColor (got [${Array.from(vg.slice(0, 4)).join(', ')}])`);

    const fctx = fragmentCtx(p, [vg]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 0.25 && c[1] === 0.5 && c[2] === 0.75 && c[3] === 1,
      `fragment run: color [0.25,0.5,0.75,1] (got [${Array.from(c).join(', ')}])`);

    // Uniform write via floatStore[location+c], then re-run both stages.
    p.floatStore[uLoc + 0] = 2;
    p.floatStore[uLoc + 1] = 2;
    p.floatStore[uLoc + 2] = 2;
    p.floatStore[uLoc + 3] = 2;
    p.vertex.run(vctx);
    const vg2 = vctx.out.varyings;
    check(vg2[0] === 2 && vg2[1] === 2 && vg2[2] === 2 && vg2[3] === 2,
      `uniform write re-run: vColor [2,2,2,2] (got [${Array.from(vg2.slice(0, 4)).join(', ')}])`);
    const fctx2 = fragmentCtx(p, [vg2]);
    p.fragment.run(fctx2);
    check(fctx2.out.color[0][0] === 2 && fctx2.out.color[0][1] === 2,
      `uniform write re-run: fragment color [2,2,...] (got [${Array.from(fctx2.out.color[0]).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* 2. ES 3.00 pair: layout(location=2) attrib, out, gl_FragDepth       */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     layout(location = 2) in vec4 aPos;
     out vec4 vColor;
     void main() { vColor = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     in vec4 vColor;
     out vec4 oColor;
     void main() { oColor = vColor; gl_FragDepth = 0.5; }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `300 pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.attributes.length === 1 && p.attributes[0].location === 2,
      `layout(location=2) attrib → location 2 (got ${JSON.stringify(p.attributes)})`);
    check(p.fragment.usesFragDepth === true, `fragment.usesFragDepth set`);
    check(p.fragment.outputs.length === 1 && p.fragment.outputs[0].location === 0 && p.fragment.outputs[0].type === FLOAT_VEC4,
      `fragment outputs [{0, FLOAT_VEC4}] (got ${JSON.stringify(p.fragment.outputs)})`);
    check(p.varyings.length === 1 && p.varyings[0].name === 'vColor' && p.varyings[0].components === 4,
      `varying 'vColor' matched + packed (got ${JSON.stringify(p.varyings)})`);

    const vctx = vertexCtx(p, {
      attribs: [undefined, undefined, new Float32Array([1, 2, 3, 4])],
      attribIndices: new Int32Array([0, 0, 0]),
    });
    p.vertex.run(vctx);
    const pos = vctx.out.position;
    check(pos[0] === 1 && pos[1] === 2 && pos[2] === 3 && pos[3] === 4,
      `300 vertex run: position [1,2,3,4] (got [${Array.from(pos).join(', ')}])`);

    const fctx = fragmentCtx(p, [vctx.out.varyings]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 1 && c[1] === 2 && c[2] === 3 && c[3] === 4,
      `300 fragment run: color [1,2,3,4] (got [${Array.from(c).join(', ')}])`);
    check(fctx.out.fragDepth === 0.5, `gl_FragDepth write 0.5 (got ${fctx.out.fragDepth})`);
  }
}

/* ------------------------------------------------------------------ */
/* 3. Struct uniform + struct-array uniform + struct varying           */
/*    round-trip (dynamic index); uniformMap keys                      */
/* ------------------------------------------------------------------ */

{
  // Struct varyings are ES 3.00-only (ES 1.00 requires float varyings), so
  // this pair is version 300. Struct uniforms work in both.
  const vs = compile(
    `#version 300 es
     struct S { vec4 a; float b; };
     uniform S u;
     uniform S uArr[2];
     out S vS;
     in vec4 aPos;
     void main() {
       int i = 1;
       vS.a = u.a + uArr[i].a;
       vS.b = u.b + uArr[i].b;
       gl_Position = aPos;
     }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     struct S { vec4 a; float b; };
     in S vS;
     out vec4 oColor;
     void main() { oColor = vS.a + vec4(vS.b); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `struct pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    const um = p.uniformMap;
    // uniformMap: 'u' → first leaf; 'u.m' leaf; struct array 'uArr[0]' → first leaf.
    check(um.get('u') !== undefined && um.get('u') === um.get('u.a'), `uniformMap 'u' → first element info`);
    check(um.get('u.a') !== undefined && um.get('u.a')!.location === 0, `uniformMap 'u.a' at location 0 (got ${um.get('u.a')?.location})`);
    check(um.get('u.b') !== undefined && um.get('u.b')!.location === 4, `uniformMap 'u.b' at location 4 (got ${um.get('u.b')?.location})`);
    check(um.get('uArr[0]') !== undefined && um.get('uArr[0]') === um.get('uArr[0].a'), `uniformMap 'uArr[0]' → first element info`);
    check(um.get('uArr') !== undefined && um.get('uArr') === um.get('uArr[0].a'), `uniformMap 'uArr' → first element info`);
    check(um.get('uArr[1].a') !== undefined && um.get('uArr[1].a')!.location === 16,
      `uniformMap 'uArr[1].a' at location 16 (got ${um.get('uArr[1].a')?.location})`);
    check(um.get('uArr[1].b') !== undefined && um.get('uArr[1].b')!.location === 20,
      `uniformMap 'uArr[1].b' at location 20 (got ${um.get('uArr[1].b')?.location})`);
    // Struct varyings flatten to per-member leaves.
    check(p.varyings.length === 2 && p.varyings[0].name === 'vS.a' && p.varyings[0].components === 4 &&
      p.varyings[1].name === 'vS.b' && p.varyings[1].components === 1,
      `struct varying flattened to vS.a(4)+vS.b(1) (got ${JSON.stringify(p.varyings)})`);

    const set4 = (key: string, v: [number, number, number, number]) => {
      const loc = um.get(key)!.location;
      p.floatStore[loc + 0] = v[0];
      p.floatStore[loc + 1] = v[1];
      p.floatStore[loc + 2] = v[2];
      p.floatStore[loc + 3] = v[3];
    };
    const set1 = (key: string, v: number) => {
      p.floatStore[um.get(key)!.location] = v;
    };
    set4('u.a', [1, 2, 3, 4]);
    set1('u.b', 0.5);
    set4('uArr[0].a', [0, 0, 0, 0]);
    set1('uArr[0].b', 0);
    set4('uArr[1].a', [5, 5, 5, 5]);
    set1('uArr[1].b', 0.5);

    const vctx = vertexCtx(p, { attribs: [new Float32Array([0, 0, 0, 1])], attribIndices: new Int32Array([0]) });
    p.vertex.run(vctx);
    const vg = vctx.out.varyings;
    // vS.a = [1,2,3,4] + uArr[1].a = [6,7,8,9]; vS.b = 0.5 + 0.5 = 1.0
    check(vg[0] === 6 && vg[1] === 7 && vg[2] === 8 && vg[3] === 9 && vg[4] === 1,
      `struct vertex: vS.a [6,7,8,9], vS.b 1.0 (got [${Array.from(vg.slice(0, 5)).join(', ')}])`);

    // One ctx.varyings entry per varying index: vS.a → 0, vS.b → 1.
    const fctx = fragmentCtx(p, [vg.slice(0, 4), vg.slice(4, 5)]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 7 && c[1] === 8 && c[2] === 9 && c[3] === 10,
      `struct round-trip: color [7,8,9,10] (got [${Array.from(c).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* 4. Link errors                                                      */
/* ------------------------------------------------------------------ */

{
  // varying name mismatch
  const vs = compile(`attribute vec4 aPos; varying vec4 vA; void main(){ vA = aPos; gl_Position = aPos; }`, 'VERTEX', 100);
  const fs = compile(`precision mediump float; varying vec4 vB; void main(){ gl_FragColor = vB; }`, 'FRAGMENT', 100);
  const l1 = linkProgram(vs, fs);
  check(!l1.ok && typeof l1.log === 'string' && l1.log.length > 0 && l1.log.includes(`varying 'vB' not matched`),
    `varying name mismatch error (${l1.ok ? 'linked!' : l1.log})`);

  // varying type mismatch
  const vs2 = compile(`attribute vec4 aPos; varying vec4 vA; void main(){ vA = aPos; gl_Position = aPos; }`, 'VERTEX', 100);
  const fs2 = compile(`precision mediump float; varying vec2 vA; void main(){ gl_FragColor = vec4(vA, 0.0, 1.0); }`, 'FRAGMENT', 100);
  const l2 = linkProgram(vs2, fs2);
  check(!l2.ok && l2.log.includes('type mismatch'), `varying type mismatch error (${logOf(l2)})`);

  // flat qualifier mismatch (ES 3.00)
  const vs3 = compile(`#version 300 es
    flat out float vF; void main(){ vF = 1.0; gl_Position = vec4(0.0); }`, 'VERTEX', 300);
  const fs3 = compile(`#version 300 es
    precision mediump float; in float vF; out vec4 o; void main(){ o = vec4(vF); }`, 'FRAGMENT', 300);
  const l3 = linkProgram(vs3, fs3);
  check(!l3.ok && l3.log.includes('flat qualifier mismatch'), `flat mismatch error (${logOf(l3)})`);

  // uniform type conflict vs/fs
  const vs4 = compile(`uniform vec4 uC; attribute vec4 aPos; void main(){ gl_Position = aPos + uC; }`, 'VERTEX', 100);
  const fs4 = compile(`precision mediump float; uniform float uC; void main(){ gl_FragColor = vec4(uC); }`, 'FRAGMENT', 100);
  const l4 = linkProgram(vs4, fs4);
  check(!l4.ok && l4.log.includes(`uniform 'uC' type conflict`), `uniform type conflict error (${logOf(l4)})`);

  // duplicate attribute location
  const vs5 = compile(`#version 300 es
    layout(location=0) in vec4 aA; layout(location=0) in vec4 aB;
    void main(){ gl_Position = aA + aB; }`, 'VERTEX', 300);
  const fs5 = compile(`#version 300 es
    precision mediump float; out vec4 o; void main(){ o = vec4(0.0); }`, 'FRAGMENT', 300);
  const l5 = linkProgram(vs5, fs5);
  check(!l5.ok && l5.log.includes('conflicts with'), `duplicate attribute location error (${logOf(l5)})`);

  // maxVaryingVectors exceeded via opts.limits
  const vs6 = compile(`attribute vec4 aPos; varying vec4 vA; varying vec4 vB; varying vec4 vC;
    void main(){ vA = vB = vC = aPos; gl_Position = aPos; }`, 'VERTEX', 100);
  const fs6 = compile(`precision mediump float; varying vec4 vA; varying vec4 vB; varying vec4 vC;
    void main(){ gl_FragColor = vA + vB + vC; }`, 'FRAGMENT', 100);
  const l6 = linkProgram(vs6, fs6, { limits: { maxVaryingVectors: 2 } });
  check(!l6.ok && l6.log.includes('too many varying vectors'), `maxVaryingVectors limit error (${logOf(l6)})`);
}

/* ------------------------------------------------------------------ */
/* 5. bindAttribLocation + first-free auto assignment                  */
/* ------------------------------------------------------------------ */

{
  const vs = compile(`attribute vec4 aPos; attribute float aScale;
    void main(){ gl_Position = aPos + vec4(aScale); }`, 'VERTEX', 100);
  const fs = compile(`precision mediump float; void main(){ gl_FragColor = vec4(1.0); }`, 'FRAGMENT', 100);
  const l = linkProgram(vs, fs, { attribBindings: new Map([['aPos', 3]]) });
  check(l.ok, `bindAttribLocation pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.attributes.length === 2 && p.attributes[0].name === 'aPos' && p.attributes[0].location === 3,
      `bindAttribLocation aPos → 3 (got ${JSON.stringify(p.attributes)})`);
    check(p.attributes[1].name === 'aScale' && p.attributes[1].location === 0,
      `unbound aScale auto-assigns first-free 0 (got ${JSON.stringify(p.attributes)})`);
  }
}

/* ------------------------------------------------------------------ */
/* 6. User function + local array (scratchSize > 0)                    */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `float sum4(float a[4]) { float s = 0.0; for (int i = 0; i < 4; i++) s += a[i]; return s; }
     attribute vec4 aPos;
     void main() {
       float a[4]; a[0] = 1.0; a[1] = 2.0; a[2] = 3.0; a[3] = 4.0;
       gl_Position = vec4(sum4(a), 0.0, 0.0, 1.0);
     }`,
    'VERTEX',
    100,
  );
  const fs = compile(`precision mediump float; void main(){ gl_FragColor = vec4(1.0); }`, 'FRAGMENT', 100);
  const l = linkProgram(vs, fs);
  check(l.ok, `user-fn pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.scratchSize > 0, `local array → scratchSize > 0 (got ${p.scratchSize})`);
    const vctx = vertexCtx(p);
    p.vertex.run(vctx);
    check(vctx.out.position[0] === 10, `sum4(1..4) === 10 (got ${vctx.out.position[0]})`);
  }
}

/* ------------------------------------------------------------------ */
/* 7. ES 3.00 flat ivec4 varying round-trip (small values)             */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     flat out ivec4 vI;
     in vec4 aPos;
     void main(){ vI = ivec4(1, -2, 3, -4); gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     flat in ivec4 vI;
     out vec4 o;
     void main(){ o = vec4(vI); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `ivec4 flat pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.varyings.length === 1 && p.varyings[0].flat === true && p.varyings[0].components === 4,
      `varying flat flag + components (got ${JSON.stringify(p.varyings)})`);
    const vctx = vertexCtx(p, { attribs: [new Float32Array([0, 0, 0, 1])], attribIndices: new Int32Array([0]) });
    p.vertex.run(vctx);
    const vg = vctx.out.varyings;
    check(vg[0] === 1 && vg[1] === -2 && vg[2] === 3 && vg[3] === -4,
      `ivec4 packed [1,-2,3,-4] (got [${Array.from(vg.slice(0, 4)).join(', ')}])`);
    const fctx = fragmentCtx(p, [vg]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 1 && c[1] === -2 && c[2] === 3 && c[3] === -4,
      `ivec4 round-trip color [1,-2,3,-4] (got [${Array.from(c).join(', ')}])`);
  }
}

/* ------------------------------------------------------------------ */
/* 8. UBO declared → links with std140 metadata (E5b1)                 */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     uniform Blocks { vec4 u; } b;
     void main(){ gl_Position = b.u; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float; out vec4 o; void main(){ o = vec4(1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `UBO pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(
      p.uniformBlocks.length === 1 && p.uniformBlocks[0].name === 'Blocks' && p.uniformBlocks[0].index === 0 &&
        p.uniformBlocks[0].size === 16 && p.uniformBlocks[0].activeUniforms.length === 1 &&
        p.uniformBlocks[0].activeUniforms[0].name === 'b.u' && p.uniformBlocks[0].activeUniforms[0].offset === 0 &&
        p.uniformBlocks[0].activeUniforms[0].type === FLOAT_VEC4,
      `UBO metadata: one block 'Blocks' index 0 size 16, member b.u@0 FLOAT_VEC4 (got ${JSON.stringify(p.uniformBlocks)})`,
    );
    check(p.uniformMap.has('b.u') === false && p.uniformMap.has('u') === false,
      `UBO members absent from uniformMap (getUniformLocation → null)`);
    const m = p.uniforms.find((u) => u.name === 'b.u');
    check(m !== undefined && m.blockIndex === 0 && m.location === -1,
      `UBO member in Program.uniforms with blockIndex 0, location -1 (got ${JSON.stringify(m)})`);
  }
}

/* ------------------------------------------------------------------ */
/* 9. UBO std140 layout + full read run (float + int block stores)     */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     uniform Blocks { highp float a; highp vec3 b; highp mat4 c; highp float d[3]; highp int e; } blk;
     in vec4 aPos;
     void main(){ gl_Position = aPos + vec4(blk.a, blk.b.x, blk.d[0], 0.0); }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     uniform Blocks { highp float a; highp vec3 b; highp mat4 c; highp float d[3]; highp int e; } blk;
     out vec4 o;
     void main(){ o = vec4(blk.a + blk.b.x + blk.c[0][0] + blk.d[0] + blk.d[1] + blk.d[2], blk.b.y, blk.b.z, float(blk.e)); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `std140 block links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // std140 (computed by hand): a float@0 (4B) | b vec3@16 (align 16, 12B) |
    // c mat4@32 (align 16, 64B, column stride 16) | d float[3]@96 (align 4,
    // stride 4, 12B) | e int@108 → total 112, block size = roundUp(112,16)=112.
    check(p.uniformBlocks.length === 1 && p.uniformBlocks[0].size === 112,
      `block size 112 (got ${JSON.stringify(p.uniformBlocks.map((b) => b.size))})`);
    const m = new Map(p.uniformBlocks[0].activeUniforms.map((u) => [u.name, u]));
    check(
      m.get('blk.a')?.offset === 0 && m.get('blk.b')?.offset === 16 && m.get('blk.c')?.offset === 32 &&
        m.get('blk.d[0]')?.offset === 96 && m.get('blk.e')?.offset === 108,
      `member offsets a@0 b@16 c@32 d@96 e@108 (got ${JSON.stringify([...m].map(([k, v]) => [k, v.offset]))})`,
    );
    check(
      m.get('blk.b')?.type === FLOAT_VEC3 && m.get('blk.c')?.type === FLOAT_MAT4 && m.get('blk.c')?.matrixStride === 16 &&
        m.get('blk.d[0]')?.size === 3 && m.get('blk.d[0]')?.arrayStride === 4 &&
        m.get('blk.e')?.type === INT && m.get('blk.e')?.size === 1,
      `member types/sizes: b FLOAT_VEC3, c FLOAT_MAT4 matrixStride 16, d[3] size 3 stride 4, e INT (got ${JSON.stringify([...m].map(([k, v]) => [k, v.type, v.size, v.arrayStride, v.matrixStride]))})`,
    );
    const um = p.uniforms.filter((u) => u.blockIndex === 0);
    check(
      um.length === 5 && um.every((u) => u.location === -1) &&
        um.map((u) => u.name).join(',') === 'blk.a,blk.b,blk.c,blk.d[0],blk.e',
      `block members in Program.uniforms (location -1, blockIndex 0) (got ${JSON.stringify(um)})`,
    );
    check(p.uniformMap.has('blk.a') === false && p.uniformMap.has('blk.d[0]') === false,
      `block members NOT in uniformMap (got ${JSON.stringify([...p.uniformMap.keys()])})`);

    // Block stores are FLOAT-indexed (generated code reads byteOffset/4):
    // a@float 0 | b@4..6 | c@8..23 | d@24..26 | e@float 27 (INT store).
    const store = new Float32Array(28); // 112 bytes / 4
    store[0] = 1.5; // blk.a
    store[4] = 2; store[5] = 3; store[6] = 4; // blk.b
    store[8] = 0.5; // blk.c[0][0]
    store[24] = 1; store[25] = 2; store[26] = 3; // blk.d
    const istore = new Int32Array(28);
    istore[27] = 7; // blk.e
    const vctx = vertexCtx(p, {
      blockStores: [store],
      blockIntStores: [istore],
      attribs: [new Float32Array([0, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    check(
      vctx.out.position[0] === 1.5 && vctx.out.position[1] === 2 && vctx.out.position[2] === 1,
      `vertex reads block: position [1.5,2,1,...] (got [${Array.from(vctx.out.position).join(', ')}])`,
    );
    const fctx = fragmentCtx(p, [], { blockStores: [store], blockIntStores: [istore] });
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(
      c[0] === 10 && c[1] === 3 && c[2] === 4 && c[3] === 7,
      `fragment reads block: color [10,3,4,7] (got [${Array.from(c).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 10. Same block in both stages → ONE shared index; mismatch → error  */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     uniform Shared { highp vec4 v; } s;
     in vec4 aPos;
     void main(){ gl_Position = aPos + s.v; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     uniform Shared { highp vec4 v; } s;
     out vec4 o;
     void main(){ o = s.v; }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `shared block links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(p.uniformBlocks.length === 1 && p.uniformBlocks[0].index === 0,
      `shared block: ONE entry index 0 (got ${JSON.stringify(p.uniformBlocks)})`);
    const store = new Float32Array([3, 4, 5, 6]);
    const vctx = vertexCtx(p, {
      blockStores: [store],
      attribs: [new Float32Array([0, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    check(vctx.out.position[0] === 3 && vctx.out.position[1] === 4,
      `vertex reads shared block (got [${Array.from(vctx.out.position).join(', ')}])`);
    const fctx = fragmentCtx(p, [], { blockStores: [store] });
    p.fragment.run(fctx);
    check(fctx.out.color[0][0] === 3 && fctx.out.color[0][2] === 5,
      `fragment reads shared block (got [${Array.from(fctx.out.color[0]).join(', ')}])`);
  }

  // Same block NAME but different member types (float vs vec2 — both at
  // offset 0, so the offsets alone would not catch it) → link error.
  const vs2 = compile(
    `#version 300 es
     uniform Shared { float v; } s;
     void main(){ gl_Position = vec4(s.v); }`,
    'VERTEX',
    300,
  );
  const fs2 = compile(
    `#version 300 es
     precision mediump float;
     uniform Shared { vec2 v; } s;
     out vec4 o;
     void main(){ o = vec4(s.v, 0.0, 1.0); }`,
    'FRAGMENT',
    300,
  );
  const l2 = linkProgram(vs2, fs2);
  check(!l2.ok && l2.log.includes(`uniform block 'Shared' layout mismatch`),
    `layout mismatch link error (${logOf(l2)})`);
}

/* ------------------------------------------------------------------ */
/* 11. Instance-less block — bare member name access                   */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     uniform Inst { highp float f; };
     in vec4 aPos;
     void main(){ gl_Position = aPos + vec4(f); }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     uniform Inst { highp float f; };
     out vec4 o;
     void main(){ o = vec4(f, 0.0, 0.0, 1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `instance-less block links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(
      p.uniformBlocks.length === 1 && p.uniformBlocks[0].name === 'Inst' &&
        p.uniformBlocks[0].activeUniforms[0].name === 'f' && p.uniformBlocks[0].activeUniforms[0].offset === 0,
      `instance-less: block 'Inst', bare member 'f' (got ${JSON.stringify(p.uniformBlocks)})`,
    );
    const store = new Float32Array([2.5]);
    const vctx = vertexCtx(p, {
      blockStores: [store],
      attribs: [new Float32Array([0, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    check(vctx.out.position[0] === 2.5, `bare member read in vs (got ${vctx.out.position[0]})`);
    const fctx = fragmentCtx(p, [], { blockStores: [store] });
    p.fragment.run(fctx);
    check(fctx.out.color[0][0] === 2.5, `bare member read in fs (got ${fctx.out.color[0][0]})`);
  }
}

/* ------------------------------------------------------------------ */
/* 12. Arrayed block + DYNAMIC instance index (blockStride path)       */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     uniform B { vec4 v; } b[2];
     uniform int i;
     in vec4 aPos;
     void main(){ gl_Position = aPos + b[i].v; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float; out vec4 o; void main(){ o = vec4(1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `arrayed block links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // One UniformBlockInfo PER ELEMENT ('b[0]','b[1]'), shared index 0,
    // size = one instance (16 bytes).
    check(
      p.uniformBlocks.length === 2 && p.uniformBlocks[0].name === 'b[0]' && p.uniformBlocks[1].name === 'b[1]' &&
        p.uniformBlocks[0].index === 0 && p.uniformBlocks[1].index === 0 && p.uniformBlocks[0].size === 16,
      `arrayed block: 'b[0]','b[1]' shared index 0 size 16 (got ${JSON.stringify(p.uniformBlocks)})`,
    );
    const iLoc = p.uniformMap.get('i')!.location;
    // Two 16-byte instances: [1,0,0,0] and [2,0,0,0] in float index space.
    const store = new Float32Array([1, 0, 0, 0, 2, 0, 0, 0]);
    const vctx = vertexCtx(p, {
      blockStores: [store],
      attribs: [new Float32Array([0, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.intStore[iLoc] = 0;
    p.vertex.run(vctx);
    check(vctx.out.position[0] === 1, `dynamic instance i=0 → v=(1,...) (got ${vctx.out.position[0]})`);
    p.intStore[iLoc] = 1;
    p.vertex.run(vctx);
    check(vctx.out.position[0] === 2, `dynamic instance i=1 → v=(2,...) (got ${vctx.out.position[0]})`);
  }
}

/* ------------------------------------------------------------------ */
/* 13. Nested struct members + member array inside a struct            */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     struct S1 { vec2 x; float y; };
     struct S2 { vec2 x; float y[2]; };
     uniform B { S1 s; S2 t; };
     in vec4 aPos;
     void main(){ gl_Position = aPos + vec4(s.x.x, s.y, t.y[0], t.y[1]); }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     struct S1 { vec2 x; float y; };
     struct S2 { vec2 x; float y[2]; };
     uniform B { S1 s; S2 t; };
     out vec4 o;
     void main(){ o = vec4(s.x.x + s.y + t.y[0] + t.y[1], s.x.y, t.x.x, t.x.y); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `nested-struct block links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // std140: s (S1)@0 — x@0 (vec2, 8B), y@8 (4B); S1 size roundUp(12,8)=16.
    // t (S2): align 8 → @16 — x@16 (8B), y float[2]@24 (stride 4, 8B);
    // S2 size roundUp(16,8)=16. Block: 16+16=32 → roundUp(32,16)=32.
    check(p.uniformBlocks.length === 1 && p.uniformBlocks[0].size === 32,
      `nested-struct block size 32 (got ${JSON.stringify(p.uniformBlocks.map((b) => b.size))})`);
    const m = new Map(p.uniformBlocks[0].activeUniforms.map((u) => [u.name, u]));
    check(
      m.get('s.x')?.offset === 0 && m.get('s.y')?.offset === 8 && m.get('t.x')?.offset === 16 && m.get('t.y[0]')?.offset === 24,
      `nested offsets s.x@0 s.y@8 t.x@16 t.y@24 (got ${JSON.stringify([...m].map(([k, v]) => [k, v.offset]))})`,
    );
    check(
      m.get('t.y[0]')?.size === 2 && m.get('t.y[0]')?.arrayStride === 4 && m.get('s.x')?.type === FLOAT_VEC2 && m.get('t.y[0]')?.type === FLOAT,
      `nested types/sizes: t.y float[2] stride 4, s.x vec2 (got ${JSON.stringify([...m].map(([k, v]) => [k, v.type, v.size, v.arrayStride]))})`,
    );
    // float space: s.x@0,1 | s.y@2 | t.x@4,5 | t.y[0]@6 | t.y[1]@7 (48B/4=12).
    const store = new Float32Array(12);
    store[0] = 1; store[1] = 2; // s.x
    store[2] = 3; // s.y
    store[4] = 4; store[5] = 5; // t.x
    store[6] = 6; store[7] = 7; // t.y
    const vctx = vertexCtx(p, {
      blockStores: [store],
      attribs: [new Float32Array([0, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    check(
      vctx.out.position[0] === 1 && vctx.out.position[1] === 3 && vctx.out.position[2] === 6 && vctx.out.position[3] === 8,
      `vertex reads nested members (got [${Array.from(vctx.out.position).join(', ')}])`,
    );
    const fctx = fragmentCtx(p, [], { blockStores: [store] });
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(
      c[0] === 17 && c[1] === 2 && c[2] === 4 && c[3] === 5,
      `fragment reads nested members: color [17,2,4,5] (got [${Array.from(c).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 14. Block limits: maxUniformBlockSize / per-stage / combined        */
/* ------------------------------------------------------------------ */

{
  // maxUniformBlockSize: vec4 v[8] = 128 bytes > 16.
  const vs = compile(
    `#version 300 es
     uniform Big { vec4 v[8]; } b;
     void main(){ gl_Position = b.v[0]; }`,
    'VERTEX',
    300,
  );
  const fs = compile(`#version 300 es
     precision mediump float; out vec4 o; void main(){ o = vec4(1.0); }`, 'FRAGMENT', 300);
  const l = linkProgram(vs, fs, { limits: { maxUniformBlockSize: 16 } });
  check(!l.ok && l.log.includes(`uniform block 'Big' exceeds maxUniformBlockSize`),
    `maxUniformBlockSize limit (${logOf(l)})`);

  // maxVertexUniformBlocks: two blocks in the VS, max 1.
  const vs2 = compile(
    `#version 300 es
     uniform A { highp vec4 a; } ba;
     uniform C { vec4 c; } bc;
     void main(){ gl_Position = ba.a + bc.c; }`,
    'VERTEX',
    300,
  );
  const fs2 = compile(`#version 300 es
     precision mediump float; out vec4 o; void main(){ o = vec4(1.0); }`, 'FRAGMENT', 300);
  const l2 = linkProgram(vs2, fs2, { limits: { maxVertexUniformBlocks: 1 } });
  check(!l2.ok && l2.log.includes('too many vertex uniform blocks'),
    `maxVertexUniformBlocks limit (${logOf(l2)})`);

  // maxFragmentUniformBlocks: one block in each stage, max fragment 0.
  const vs3 = compile(
    `#version 300 es
     uniform A { highp vec4 a; } ba;
     void main(){ gl_Position = ba.a; }`,
    'VERTEX',
    300,
  );
  const fs3 = compile(`#version 300 es
     precision mediump float;
     uniform A { highp vec4 a; } ba;
     out vec4 o; void main(){ o = ba.a; }`, 'FRAGMENT', 300);
  const l3 = linkProgram(vs3, fs3, { limits: { maxFragmentUniformBlocks: 0 } });
  check(!l3.ok && l3.log.includes('too many fragment uniform blocks'),
    `maxFragmentUniformBlocks limit (${logOf(l3)})`);

  // maxCombinedUniformBlocks: two blocks (vs+fs distinct), max combined 1.
  const vs4 = compile(
    `#version 300 es
     uniform A { highp vec4 a; } ba;
     void main(){ gl_Position = ba.a; }`,
    'VERTEX',
    300,
  );
  const fs4 = compile(`#version 300 es
     precision mediump float;
     uniform D { vec4 d; } bd;
     out vec4 o; void main(){ o = bd.d; }`, 'FRAGMENT', 300);
  const l4 = linkProgram(vs4, fs4, { limits: { maxCombinedUniformBlocks: 1 } });
  check(!l4.ok && l4.log.includes('too many uniform blocks'),
    `maxCombinedUniformBlocks limit (${logOf(l4)})`);
}

/* ------------------------------------------------------------------ */
/* 15. Varying interface blocks (E5b2): matching + packing             */
/* ------------------------------------------------------------------ */

{
  // Different instance names between stages must still link (matching is by
  // (blockName, memberName)); the layout carries BOTH stages' keys.
  const vs = compile(
    `#version 300 es
     out VS_OUT { vec4 c; } a;
     in vec4 aPos;
     void main() { a.c = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; } b;
     out vec4 o;
     void main() { o = b.c; }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `diff-instance block pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(
      p.varyings.length === 1 && p.varyings[0].name === 'a.c' && p.varyings[0].components === 4,
      `block varying 'a.c' packed (got ${JSON.stringify(p.varyings)})`,
    );
    const vctx = vertexCtx(p, { attribs: [new Float32Array([1, 2, 3, 4])], attribIndices: new Int32Array([0]) });
    p.vertex.run(vctx);
    const vg = vctx.out.varyings;
    check(vg[0] === 1 && vg[1] === 2 && vg[2] === 3 && vg[3] === 4,
      `vertex packs a.c (got [${Array.from(vg.slice(0, 4)).join(', ')}])`);
    const fctx = fragmentCtx(p, [vg]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 1 && c[1] === 2 && c[2] === 3 && c[3] === 4,
      `fragment reads b.c → color [1,2,3,4] (got [${Array.from(c).join(', ')}])`);
  }

  // Two-member block: cumulative offset packing (vec4 c; float d;).
  const vs2 = compile(
    `#version 300 es
     out VS_OUT { vec4 c; float d; } a;
     in vec4 aPos;
     void main() { a.c = aPos; a.d = aPos.x; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs2 = compile(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; float d; } b;
     out vec4 o;
     void main() { o = b.c + vec4(b.d); }`,
    'FRAGMENT',
    300,
  );
  const l2 = linkProgram(vs2, fs2);
  check(l2.ok, `two-member block links (${l2.ok ? '' : l2.log})`);
  if (l2.ok) {
    const p = l2.program;
    check(
      p.varyings.length === 2 && p.varyings[0].name === 'a.c' && p.varyings[0].components === 4 &&
        p.varyings[1].name === 'a.d' && p.varyings[1].components === 1,
      `two-member leaves a.c(4) a.d(1) (got ${JSON.stringify(p.varyings)})`,
    );
    const vctx = vertexCtx(p, { attribs: [new Float32Array([1, 2, 3, 4])], attribIndices: new Int32Array([0]) });
    p.vertex.run(vctx);
    check(vctx.out.varyings[4] === 1, `a.d packed at offset 4 (got [${Array.from(vctx.out.varyings.slice(0, 6)).join(', ')}])`);
    const fctx = fragmentCtx(p, [vctx.out.varyings.slice(0, 4), vctx.out.varyings.slice(4, 5)]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 2 && c[1] === 3 && c[2] === 4 && c[3] === 5,
      `two-member round-trip [2,3,4,5] (got [${Array.from(c).join(', ')}])`);
  }

  // Instance-less block: bare member names in both stages.
  const vs3 = compile(
    `#version 300 es
     out VS_OUT { vec4 c; };
     in vec4 aPos;
     void main() { c = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs3 = compile(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; };
     out vec4 o;
     void main() { o = c; }`,
    'FRAGMENT',
    300,
  );
  const l3 = linkProgram(vs3, fs3);
  check(l3.ok, `instance-less block links (${l3.ok ? '' : l3.log})`);
  if (l3.ok) {
    const p = l3.program;
    check(p.varyings.length === 1 && p.varyings[0].name === 'c', `bare member 'c' packed (got ${JSON.stringify(p.varyings)})`);
    const vctx = vertexCtx(p, { attribs: [new Float32Array([5, 6, 7, 8])], attribIndices: new Int32Array([0]) });
    p.vertex.run(vctx);
    const fctx = fragmentCtx(p, [vctx.out.varyings]);
    p.fragment.run(fctx);
    check(fctx.out.color[0][0] === 5 && fctx.out.color[0][3] === 8,
      `instance-less round-trip [5,..,8] (got [${Array.from(fctx.out.color[0]).join(', ')}])`);
  }

  // flat mismatch between stages → link error.
  const vs4 = compile(
    `#version 300 es
     out VS_OUT { flat vec4 c; } a;
     in vec4 aPos;
     void main() { a.c = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs4 = compile(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; } b;
     out vec4 o;
     void main() { o = b.c; }`,
    'FRAGMENT',
    300,
  );
  const l4 = linkProgram(vs4, fs4);
  check(!l4.ok && l4.log.includes('flat qualifier mismatch'), `block flat mismatch error (${logOf(l4)})`);

  // Fragment block member absent from the vertex block → link error.
  const vs5 = compile(
    `#version 300 es
     out VS_OUT { vec4 c; } a;
     in vec4 aPos;
     void main() { a.c = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs5 = compile(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; vec2 t; } b;
     out vec4 o;
     void main() { o = b.c + vec4(b.t, 0.0, 0.0); }`,
    'FRAGMENT',
    300,
  );
  const l5 = linkProgram(vs5, fs5);
  check(!l5.ok && l5.log.includes('not matched'), `fs-only block member error (${logOf(l5)})`);

  // Mismatched member types → link error.
  const vs6 = compile(
    `#version 300 es
     out VS_OUT { vec4 c; } a;
     in vec4 aPos;
     void main() { a.c = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs6 = compile(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec2 c; } b;
     out vec4 o;
     void main() { o = vec4(b.c, 0.0, 1.0); }`,
    'FRAGMENT',
    300,
  );
  const l6 = linkProgram(vs6, fs6);
  check(!l6.ok && l6.log.includes('type mismatch'), `block member type mismatch error (${logOf(l6)})`);

  // Struct member inside a block: flattened per sub-member leaf.
  const vs7 = compile(
    `#version 300 es
     struct S { vec2 x; float y; };
     out VS_OUT { S s; } a;
     in vec4 aPos;
     void main() { a.s.x = aPos.xy; a.s.y = aPos.z; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs7 = compile(
    `#version 300 es
     precision mediump float;
     struct S { vec2 x; float y; };
     in VS_OUT { S s; } b;
     out vec4 o;
     void main() { o = vec4(b.s.x.x, b.s.x.y, b.s.y, 1.0); }`,
    'FRAGMENT',
    300,
  );
  const l7 = linkProgram(vs7, fs7);
  check(l7.ok, `struct-member block links (${l7.ok ? '' : l7.log})`);
  if (l7.ok) {
    const p = l7.program;
    check(
      p.varyings.length === 2 && p.varyings[0].name === 'a.s.x' && p.varyings[1].name === 'a.s.y',
      `struct member flattened (got ${JSON.stringify(p.varyings)})`,
    );
    const vctx = vertexCtx(p, { attribs: [new Float32Array([1, 2, 3, 4])], attribIndices: new Int32Array([0]) });
    p.vertex.run(vctx);
    const vg = vctx.out.varyings;
    const fctx = fragmentCtx(p, [vg.slice(0, 2), vg.slice(2, 3)]);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 1 && c[1] === 2 && c[2] === 3,
      `struct-member round-trip [1,2,3] (got [${Array.from(c).join(', ')}])`);
  }

  // Arrayed varying blocks: the codegen walker cannot resolve element
  // indices on member descent → clear link rejection.
  const vs8 = compile(
    `#version 300 es
     out VS_OUT { vec4 c; } a[2];
     in vec4 aPos;
     void main() { a[0].c = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs8 = compile(
    `#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; } b[2];
     out vec4 o;
     void main() { o = b[0].c; }`,
    'FRAGMENT',
    300,
  );
  const l8 = linkProgram(vs8, fs8);
  check(!l8.ok && l8.log.includes('arrayed varying interface blocks not supported'),
    `arrayed block rejected (${logOf(l8)})`);
}

/* ------------------------------------------------------------------ */
/* 16. Transform feedback (E5b2)                                       */
/* ------------------------------------------------------------------ */

{
  // Valid SEPARATE_ATTRIBS: two varyings captured with metadata.
  const vs = compile(
    `#version 300 es
     out vec4 vColor;
     out float vScale;
     in vec4 aPos;
     void main() { vColor = aPos; vScale = aPos.x; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     in vec4 vColor;
     in float vScale;
     out vec4 o;
     void main() { o = vColor + vec4(vScale); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs, { transformFeedback: { varyings: ['vColor', 'vScale'], bufferMode: 'SEPARATE_ATTRIBS' } });
  check(l.ok, `TF SEPARATE_ATTRIBS links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(
      p.transformFeedbackVaryings.length === 2 && p.transformFeedbackVaryings[0].name === 'vColor' &&
        p.transformFeedbackVaryings[0].type === FLOAT_VEC4 && p.transformFeedbackVaryings[0].size === 1 &&
        p.transformFeedbackVaryings[1].name === 'vScale' && p.transformFeedbackVaryings[1].type === FLOAT,
      `TF varyings metadata (got ${JSON.stringify(p.transformFeedbackVaryings)})`,
    );
  }

  // 'gl_Position' is capturable.
  const vs2 = compile(
    `#version 300 es
     in vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs2 = compile(`#version 300 es
     precision mediump float; out vec4 o; void main() { o = vec4(1.0); }`, 'FRAGMENT', 300);
  const l2 = linkProgram(vs2, fs2, { transformFeedback: { varyings: ['gl_Position'], bufferMode: 'SEPARATE_ATTRIBS' } });
  check(l2.ok && l2.program.transformFeedbackVaryings.length === 1 && l2.program.transformFeedbackVaryings[0].name === 'gl_Position',
    `TF gl_Position capturable (${l2.ok ? '' : l2.log})`);

  // Unknown varying name → link error.
  const l3 = linkProgram(vs2, fs2, { transformFeedback: { varyings: ['nope'], bufferMode: 'SEPARATE_ATTRIBS' } });
  check(!l3.ok && l3.log.includes(`transform feedback varying 'nope'`), `TF invalid name error (${logOf(l3)})`);

  // INTERLEAVED limit: vec4 varying = 4 components > 2.
  const vs4 = compile(
    `#version 300 es
     out vec4 vColor;
     in vec4 aPos;
     void main() { vColor = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs4 = compile(`#version 300 es
     precision mediump float;
     in vec4 vColor;
     out vec4 o;
     void main() { o = vColor; }`, 'FRAGMENT', 300);
  const l4 = linkProgram(vs4, fs4, {
    transformFeedback: { varyings: ['vColor'], bufferMode: 'INTERLEAVED_ATTRIBS' },
    limits: { maxTransformFeedbackInterleavedComponents: 2 },
  });
  check(!l4.ok && l4.log.includes('maxTransformFeedbackInterleavedComponents'),
    `TF interleaved limit error (${logOf(l4)})`);

  // SEPARATE count limit: 2 varyings, max 1 attrib.
  const vs5 = compile(
    `#version 300 es
     out vec4 vColor;
     out float vScale;
     in vec4 aPos;
     void main() { vColor = aPos; vScale = aPos.x; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs5 = compile(`#version 300 es
     precision mediump float;
     in vec4 vColor;
     in float vScale;
     out vec4 o;
     void main() { o = vColor + vec4(vScale); }`, 'FRAGMENT', 300);
  const l5 = linkProgram(vs5, fs5, {
    transformFeedback: { varyings: ['vColor', 'vScale'], bufferMode: 'SEPARATE_ATTRIBS' },
    limits: { maxTransformFeedbackSeparateAttribs: 1 },
  });
  check(!l5.ok && l5.log.includes('maxTransformFeedbackSeparateAttribs'), `TF separate count limit error (${logOf(l5)})`);

  // SEPARATE components limit: vec4 = 4 components > 2.
  const l6 = linkProgram(vs4, fs4, {
    transformFeedback: { varyings: ['vColor'], bufferMode: 'SEPARATE_ATTRIBS' },
    limits: { maxTransformFeedbackSeparateComponents: 2 },
  });
  check(!l6.ok && l6.log.includes('maxTransformFeedbackSeparateComponents'),
    `TF separate components limit error (${logOf(l6)})`);

  // TF on an ES 1.00 program → link error.
  const vs7 = compile(`attribute vec4 aPos; varying vec4 vC; void main(){ vC = aPos; gl_Position = aPos; }`, 'VERTEX', 100);
  const fs7 = compile(`precision mediump float; varying vec4 vC; void main(){ gl_FragColor = vC; }`, 'FRAGMENT', 100);
  const l7 = linkProgram(vs7, fs7, { transformFeedback: { varyings: ['vC'], bufferMode: 'SEPARATE_ATTRIBS' } });
  check(!l7.ok && l7.log.includes('requires GLSL ES 3.00'), `TF on ES 1.00 error (${logOf(l7)})`);

  // Block member by bare member name and by full instance path.
  const vs8 = compile(
    `#version 300 es
     out VS_OUT { vec4 c; } a;
     in vec4 aPos;
     void main() { a.c = aPos; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs8 = compile(`#version 300 es
     precision mediump float;
     in VS_OUT { vec4 c; } b;
     out vec4 o;
     void main() { o = b.c; }`, 'FRAGMENT', 300);
  const l8 = linkProgram(vs8, fs8, { transformFeedback: { varyings: ['c'], bufferMode: 'SEPARATE_ATTRIBS' } });
  check(l8.ok && l8.program.transformFeedbackVaryings.length === 1 && l8.program.transformFeedbackVaryings[0].name === 'c',
    `TF block member by bare name (${l8.ok ? '' : l8.log})`);
  const l8b = linkProgram(vs8, fs8, { transformFeedback: { varyings: ['a.c'], bufferMode: 'SEPARATE_ATTRIBS' } });
  check(l8b.ok && l8b.program.transformFeedbackVaryings.length === 1 && l8b.program.transformFeedbackVaryings[0].name === 'a.c',
    `TF block member by full path (${l8b.ok ? '' : l8b.log})`);
}

/* ------------------------------------------------------------------ */
/* 17. Sampler explicit-binding conflicts (E5b2)                       */
/* ------------------------------------------------------------------ */

{
  // Two ACTIVE samplers of DIFFERENT types with the SAME explicit binding.
  const vs = compile(
    `#version 300 es
     layout(binding = 0) uniform sampler2D a;
     in vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     layout(binding = 0) uniform samplerCube b;
     out vec4 o;
     void main() { o = texture(b, vec3(0.0)); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(!l.ok && l.log.includes(`sampler binding conflict: units 0`), `sampler binding conflict error (${logOf(l)})`);

  // Different explicit bindings → links fine.
  const vs2 = compile(
    `#version 300 es
     layout(binding = 0) uniform sampler2D a;
     in vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs2 = compile(
    `#version 300 es
     precision mediump float;
     layout(binding = 1) uniform samplerCube b;
     out vec4 o;
     void main() { o = texture(b, vec3(0.0)); }`,
    'FRAGMENT',
    300,
  );
  const l2 = linkProgram(vs2, fs2);
  check(l2.ok, `different sampler bindings link (${l2.ok ? '' : l2.log})`);

  // Active sampler count > maxCombinedTextureImageUnits → link error.
  const vs3 = compile(
    `#version 300 es
     uniform sampler2D a;
     in vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs3 = compile(
    `#version 300 es
     precision mediump float;
     uniform sampler2D b;
     out vec4 o;
     void main() { o = texture(b, vec2(0.0)); }`,
    'FRAGMENT',
    300,
  );
  const l3 = linkProgram(vs3, fs3, { limits: { maxCombinedTextureImageUnits: 1 } });
  check(!l3.ok && l3.log.includes('too many texture units'), `sampler count limit error (${logOf(l3)})`);

  // Default-0 samplers (no explicit binding) of different types → NO error
  // (WebGL practice; only explicit bindings are conflict-checked).
  const vs4 = compile(
    `#version 300 es
     uniform sampler2D a;
     in vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs4 = compile(
    `#version 300 es
     precision mediump float;
     uniform samplerCube b;
     out vec4 o;
     void main() { o = texture(b, vec3(0.0)); }`,
    'FRAGMENT',
    300,
  );
  const l4 = linkProgram(vs4, fs4);
  check(l4.ok, `default-0 samplers of different types link (${l4.ok ? '' : l4.log})`);
}

/* ------------------------------------------------------------------ */
/* 15. collectStructNames (pure AST walker — no codegen dependency)     */
/* ------------------------------------------------------------------ */

function structNames(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT'): string[] {
  const r = compileShader(src, { type, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  return collectStructNames(r.shader.ast);
}

{
  // Top-level bare struct-decl + struct-with-declarators + local structs in
  // nested blocks; source order. Within-stage dedup via the same struct name
  // declared in two DIFFERENT function bodies (sibling scopes — legal GLSL).
  const vs = structNames(
    `#version 300 es
     struct A { vec2 a; };
     struct B { float b; } bVar;
     uniform struct C { float c; } cVar;
     void f() {
       struct D { float d; };
       if (true) { struct E { float e; }; }
       for (int i = 0; i < 1; i++) { struct F { float f; }; }
       while (false) { struct G { float g; }; }
       do { struct H { float h; }; } while (false);
       switch (1) { default: { struct I { float i; }; } }
     }
     void g() { struct D { float d; }; } // same name, sibling scope → dedup
     `,
    300,
    'VERTEX',
  );
  check(
    JSON.stringify(vs) === JSON.stringify(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']),
    `collectStructNames vertex: got ${JSON.stringify(vs)}`,
  );

  // Struct declared in only ONE stage still lands in the (union) layout.
  const fs = structNames(
    `#version 300 es
     precision mediump float;
     struct Foo { vec2 a; float b; };
     out vec4 o;
     void main() { o = vec4(0.0); }`,
    300,
    'FRAGMENT',
  );
  check(JSON.stringify(fs) === JSON.stringify(['Foo']), `collectStructNames fragment: got ${JSON.stringify(fs)}`);

  // Sanity: struct decls (incl. a struct in only one stage) do not break the
  // link. These shaders do NOT call a struct constructor, so this check does
  // not depend on codegen struct-ctor dispatch (covered by codegen's own
  // selftests once env.structNames seeding lands).
  const vs2 = compile(
    `#version 300 es
     struct Shared { vec2 a; };
     struct VsOnly { float x; };
     in vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs2 = compile(
    `#version 300 es
     precision mediump float;
     struct Shared { vec2 a; };
     out vec4 o;
     void main() { o = vec4(0.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs2, fs2);
  check(l.ok, `struct-decl pair links without ctors (${l.ok ? '' : l.log})`);
}

/* ------------------------------------------------------------------ */
/* 18. gl_DepthRange builtin uniform reflection                        */
/* ------------------------------------------------------------------ */

{
  // floatStore length of the (a) using program — compared in (d) for the
  // store-size accounting check (blocks are sibling scopes).
  let usingFloatLen = -1;

  // (a) ES 1.00 VERTEX-stage use: the shader reads all three members. The
  // linker must expose them as active uniforms (GL_FLOAT, size 1) with REAL
  // float-store slots appended after the (empty) user-uniform block, and the
  // linked program must RUN (codegen reads ctx.depthRange, unchanged).
  const vs = compile(
    `attribute vec4 aPos;
     varying float vNear;
     varying float vFar;
     varying float vDiff;
     void main() {
       vNear = gl_DepthRange.near;
       vFar = gl_DepthRange.far;
       vDiff = gl_DepthRange.diff;
       gl_Position = aPos;
     }`,
    'VERTEX',
    100,
  );
  const fs = compile(
    `precision mediump float;
     varying float vNear;
     varying float vFar;
     varying float vDiff;
     void main() { gl_FragColor = vec4(vNear, vFar, vDiff, 1.0); }`,
    'FRAGMENT',
    100,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `1.00 gl_DepthRange vertex pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // Exactly the 3 builtin entries (no user uniforms): names, GL_FLOAT, size
    // 1, components 1, non-integral, default block, non-sampler.
    check(
      p.uniforms.length === 3 &&
        JSON.stringify(p.uniforms.map((u) => u.name)) ===
          JSON.stringify(['gl_DepthRange.near', 'gl_DepthRange.far', 'gl_DepthRange.diff']) &&
        p.uniforms.every((u) => u.type === FLOAT && u.size === 1 && u.components === 1 && !u.integral && u.blockIndex === -1 && !u.sampler),
      `3 gl_DepthRange entries with correct names/type/size/components/integral/blockIndex (got ${JSON.stringify(p.uniforms)})`,
    );
    check(
      p.uniformMap.has('gl_DepthRange.near') && p.uniformMap.has('gl_DepthRange.far') && p.uniformMap.has('gl_DepthRange.diff'),
      `uniformMap has the 3 member names (keys: ${JSON.stringify([...p.uniformMap.keys()])})`,
    );
    const s = p.depthRangeSlots;
    check(s !== null && s[0] < s[1] && s[1] < s[2], `depthRangeSlots non-null ascending (got ${JSON.stringify(s)})`);
    if (s !== null) {
      check(p.floatStore.length >= s[2] + 1, `floatStore sized for the last slot (len ${p.floatStore.length}, need ${s[2] + 1})`);
      check(p.floatStore[s[0]] === 0 && p.floatStore[s[1]] === 0 && p.floatStore[s[2]] === 0,
        'the 3 slots are zero-initialized');
      // uniformMap entries point at the slots (gl: getUniformLocation → find by
      // name → read floatStore[location]).
      check(
        p.uniformMap.get('gl_DepthRange.near')!.location === s[0] &&
          p.uniformMap.get('gl_DepthRange.far')!.location === s[1] &&
          p.uniformMap.get('gl_DepthRange.diff')!.location === s[2],
        `uniformMap locations == depthRangeSlots (near ${p.uniformMap.get('gl_DepthRange.near')!.location}, far ${p.uniformMap.get('gl_DepthRange.far')!.location}, diff ${p.uniformMap.get('gl_DepthRange.diff')!.location})`,
      );
      // Slot write-through (simulates gl/ writing the current depth range on
      // link/adopt and on glDepthRangef; getUniform reads these indices).
      // Values chosen exactly representable in float32 (0.1/0.9 are not).
      p.floatStore[s[0]] = 0.5;
      p.floatStore[s[1]] = 1.0;
      p.floatStore[s[2]] = 0.25;
      check(
        p.floatStore[s[0]] === 0.5 && p.floatStore[s[1]] === 1.0 && p.floatStore[s[2]] === 0.25,
        'write-through: values sit at the depthRangeSlots indices',
      );
    }
    // Run: codegen still lowers member reads to ctx.depthRange (draw-time
    // state), so the program executes with a ctx-provided depthRange.
    const dr = new Float32Array([0.25, 0.75, 0.5]);
    const vctx = vertexCtx(p, { attribs: [new Float32Array([1, 2, 3, 4])], attribIndices: new Int32Array([0]), depthRange: dr });
    p.vertex.run(vctx);
    const vg = vctx.out.varyings;
    check(vg[0] === 0.25 && vg[1] === 0.75 && vg[2] === 0.5,
      `vertex run reads gl_DepthRange.near/far/diff (got [${Array.from(vg.slice(0, 3)).join(', ')}])`);
    const fctx = fragmentCtx(p, [vg.slice(0, 1), vg.slice(1, 2), vg.slice(2, 3)], { depthRange: dr });
    p.fragment.run(fctx);
    check(
      fctx.out.color[0][0] === 0.25 && fctx.out.color[0][1] === 0.75 && fctx.out.color[0][2] === 0.5,
      `fragment run: color [0.25,0.75,0.5,...] (got [${Array.from(fctx.out.color[0]).join(', ')}])`,
    );
    usingFloatLen = p.floatStore.length;
  }

  // (b) ES 3.00 with BOTH stages using the builtin (vertex reads diff,
  // fragment reads near) — one shared set of 3 entries per program.
  const vs3 = compile(
    `#version 300 es
     in vec4 aPos;
     out float vD;
     void main() { vD = gl_DepthRange.diff; gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fs3 = compile(
    `#version 300 es
     precision mediump float;
     in float vD;
     out vec4 o;
     void main() { o = vec4(gl_DepthRange.near + vD); }`,
    'FRAGMENT',
    300,
  );
  const l3 = linkProgram(vs3, fs3);
  check(l3.ok, `3.00 both-stage gl_DepthRange pair links (${l3.ok ? '' : l3.log})`);
  if (l3.ok) {
    const p = l3.program;
    check(
      p.uniforms.length === 3 && p.uniforms.every((u) => u.type === FLOAT && u.size === 1) &&
        p.depthRangeSlots !== null,
      `3.00 both-stage use → 3 entries + slots (got ${JSON.stringify(p.uniforms.map((u) => u.name))}, slots ${JSON.stringify(p.depthRangeSlots)})`,
    );
    const dr = new Float32Array([0.25, 0.75, 0.5]);
    const vctx = vertexCtx(p, { attribs: [new Float32Array([1, 2, 3, 4])], attribIndices: new Int32Array([0]), depthRange: dr });
    p.vertex.run(vctx);
    const fctx = fragmentCtx(p, [vctx.out.varyings], { depthRange: dr });
    p.fragment.run(fctx);
    check(fctx.out.color[0][0] === 0.75, `3.00 run: near + diff = 0.75 (got ${fctx.out.color[0][0]})`);
  }

  // (c) FRAGMENT-only use (1.00): entries present, program runs.
  const vsc = compile(
    `attribute vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fsc = compile(
    `precision mediump float;
     void main() { gl_FragColor = vec4(gl_DepthRange.far); }`,
    'FRAGMENT',
    100,
  );
  const lc = linkProgram(vsc, fsc);
  check(lc.ok, `fragment-only gl_DepthRange pair links (${lc.ok ? '' : lc.log})`);
  if (lc.ok) {
    const p = lc.program;
    check(
      p.uniforms.length === 3 && p.uniforms[1].name === 'gl_DepthRange.far' && p.depthRangeSlots !== null,
      `fragment-only use → 3 entries + slots (got ${JSON.stringify(p.uniforms.map((u) => u.name))})`,
    );
    const fctx = fragmentCtx(p, [], { depthRange: new Float32Array([0.25, 0.75, 0.5]) });
    p.fragment.run(fctx);
    check(fctx.out.color[0][0] === 0.75, `fragment-only run: gl_DepthRange.far = 0.75 (got ${fctx.out.color[0][0]})`);
  }

  // (d) NOT using gl_DepthRange: no entries, no slots, store unaffected —
  // uniform enumeration identical to before (empty here).
  const vsd = compile(
    `attribute vec4 aPos;
     varying float vNear;
     varying float vFar;
     varying float vDiff;
     void main() {
       vNear = 0.1;
       vFar = 0.9;
       vDiff = 0.8;
       gl_Position = aPos;
     }`,
    'VERTEX',
    100,
  );
  const fsd = compile(
    `precision mediump float;
     varying float vNear;
     varying float vFar;
     varying float vDiff;
     void main() { gl_FragColor = vec4(vNear, vFar, vDiff, 1.0); }`,
    'FRAGMENT',
    100,
  );
  const ld = linkProgram(vsd, fsd);
  check(ld.ok, `non-using pair links (${ld.ok ? '' : ld.log})`);
  if (ld.ok) {
    const p = ld.program;
    check(p.uniforms.length === 0, `no gl_DepthRange entries when unused (got ${JSON.stringify(p.uniforms)})`);
    check(p.depthRangeSlots === null, `depthRangeSlots === null when unused (got ${JSON.stringify(p.depthRangeSlots)})`);
    check(![...p.uniformMap.keys()].some((k) => k.startsWith('gl_DepthRange')), `uniformMap has no gl_DepthRange names when unused`);
    // Store-size accounting: the using program (a) has exactly 3 more floats
    // than this otherwise-uniform-identical non-using program.
    if (usingFloatLen >= 0) {
      check(
        usingFloatLen === p.floatStore.length + 3,
        `depthRange program floatStore is 3 floats larger (using ${usingFloatLen} vs not-using ${p.floatStore.length})`,
      );
    }
  }
}

/* ------------------------------------------------------------------ */
/* 15. Struct MEMBER arrays stay whole in the active-uniform layout     */
/*     (CTS shader-with-array-of-structs-containing-arrays.html)        */
/* ------------------------------------------------------------------ */
/* PIN: `my_struct { vec4 color1[2]; vec4 color2[2]; } u_colors[2];`
 * reports 4 getActiveUniform entries ('u_colors[i].colorN[0]', size 2) —
 * only the TOP-LEVEL array expands; member arrays stay whole. Pre-fix the
 * linker expanded member arrays per element → 8 entries of size 1 and the
 * uniformMap lacked the bare '<p>.m' / '<p>.m[k]' keys (getUniformLocation
 * returned null → the CTS page's loc00-11 checks failed). */

{
  const vs = compile(
    `attribute vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fs = compile(
    `precision mediump float;
     struct my_struct {
       vec4 color1[2];
       vec4 color2[2];
     };
     uniform my_struct u_colors[2];
     void main(void) {
       gl_FragColor = u_colors[0].color1[0] + u_colors[0].color2[0] +
                      u_colors[1].color1[1] + u_colors[1].color2[1];
     }`,
    'FRAGMENT',
    100,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `member-array struct pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // 4 active uniforms (was 8 pre-fix), one per member array, in walk order
    // (top-level element 0 then 1, member declaration order).
    const expect = [
      { name: 'u_colors[0].color1[0]', loc: 0 },
      { name: 'u_colors[0].color2[0]', loc: 8 },
      { name: 'u_colors[1].color1[0]', loc: 16 },
      { name: 'u_colors[1].color2[0]', loc: 24 },
    ];
    check(
      p.uniforms.length === 4 && p.uniforms.map((u) => u.name).join('|') === expect.map((e) => e.name).join('|'),
      `4 active uniforms in walk order (got ${JSON.stringify(p.uniforms.map((u) => u.name))})`,
    );
    for (let i = 0; i < expect.length; i++) {
      const u = p.uniforms[i];
      check(
        u.name === expect[i].name && u.location === expect[i].loc && u.type === FLOAT_VEC4 &&
          u.size === 2 && u.components === 4 && u.blockIndex === -1,
        `uniform[${i}] '${expect[i].name}' @${expect[i].loc} FLOAT_VEC4 size 2 (got ${JSON.stringify(u)})`,
      );
    }
    // uniformMap: bare root + '[0]' → first leaf; every member-array alias
    // ('<p>.m' bare and '<p>.m[k]' k < size) → the SAME leaf; out-of-range
    // '<p>.m[2]' and unknown paths absent.
    const um = p.uniformMap;
    check(
      um.get('u_colors') === um.get('u_colors[0]') && um.get('u_colors[0]') === um.get('u_colors[0].color1[0]'),
      `uniformMap 'u_colors'/'u_colors[0]' → first leaf`,
    );
    const aliases: [string, string][] = [
      ['u_colors[0].color1', 'u_colors[0].color1[0]'],
      ['u_colors[0].color1[1]', 'u_colors[0].color1[0]'],
      ['u_colors[0].color2', 'u_colors[0].color2[0]'],
      ['u_colors[0].color2[1]', 'u_colors[0].color2[0]'],
      ['u_colors[1].color1', 'u_colors[1].color1[0]'],
      ['u_colors[1].color1[1]', 'u_colors[1].color1[0]'],
      ['u_colors[1].color2', 'u_colors[1].color2[0]'],
      ['u_colors[1].color2[1]', 'u_colors[1].color2[0]'],
    ];
    for (const [key, leaf] of aliases) {
      const info = um.get(key);
      check(info !== undefined && info === um.get(leaf), `uniformMap '${key}' → leaf '${leaf}'`);
    }
    check(
      um.has('u_colors[0].color1[2]') === false && um.has('u_colors[2]') === false && um.has('u_colors[0].color3') === false,
      `out-of-range/unknown member-array keys absent`,
    );
    // Round-trip: write one vec4 per member-array element (emulating gl's
    // getUniformLocation: element k of a leaf starts at location + k*stride)
    // and run the fragment — mirrors the CTS page's yellow-draw (which failed
    // pre-fix because loc00-11 were null).
    const write = (key: string, v: [number, number, number, number]) => {
      const info = um.get(key)!;
      const m = /\[(\d+)\]$/.exec(key);
      const elem = m !== null && info.size > 1 ? parseInt(m[1], 10) : 0;
      const loc = info.location + elem * 4; // vec4 element stride 4
      p.floatStore[loc + 0] = v[0];
      p.floatStore[loc + 1] = v[1];
      p.floatStore[loc + 2] = v[2];
      p.floatStore[loc + 3] = v[3];
    };
    write('u_colors[0].color1[0]', [1, 0, 0, 0]);
    write('u_colors[0].color2[0]', [0, 1, 0, 0]);
    write('u_colors[1].color1[1]', [0, 0, 1, 0]);
    write('u_colors[1].color2[1]', [0, 0, 0, 1]);
    const fctx = fragmentCtx(p);
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(
      c[0] === 1 && c[1] === 1 && c[2] === 1 && c[3] === 1,
      `member-array round-trip: color [1,1,1,1] (got [${Array.from(c).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 16. Struct-element MEMBER arrays still expand per element            */
/*     (struct-array members have no GLenum — one entry would throw)    */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `attribute vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fs = compile(
    `precision mediump float;
     struct A { vec4 c; };
     struct B { A a[2]; };
     uniform B u;
     void main() { gl_FragColor = u.a[0].c + u.a[1].c; }`,
    'FRAGMENT',
    100,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `struct-array member pair links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    check(
      p.uniforms.length === 2 && p.uniforms[0].name === 'u.a[0].c' && p.uniforms[0].size === 1 &&
        p.uniforms[1].name === 'u.a[1].c' && p.uniforms[1].size === 1,
      `struct-array member expands to 'u.a[0].c'/'u.a[1].c' size 1 (got ${JSON.stringify(p.uniforms.map((u) => u.name))})`,
    );
    check(
      p.uniformMap.get('u.a[0].c') !== undefined && p.uniformMap.get('u.a[1].c') !== undefined &&
        p.uniformMap.get('u') === p.uniformMap.get('u.a[0].c'),
      `struct-array member leaves in uniformMap`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 17. UBO arrays of structs (uniform-blocks-with-arrays.html crash    */
/*     pin): struct members flatten per element; arrays of structs     */
/*     expand EVERY element; structs containing arrays recurse;        */
/*     arrayed-block variant. Before the collectBlockLeaves fix the    */
/*     leaf for `light_t lights[2]` carried the STRUCT type →          */
/*     memberInfo's toGLenum threw 'struct type light_t has no         */
/*     GLenum' → linkProgram threw → CTS page 'Loading program         */
/*     failed'.                                                        */
/* ------------------------------------------------------------------ */

{
  // (a) NON-array struct member: flattened 'd.s.a'/'d.s.b' leaves.
  const vsa = compile(
    `#version 300 es
     struct Pair { vec2 a; float b; };
     uniform Data { Pair s; } d;
     in vec4 aPos;
     void main(){ gl_Position = aPos + vec4(d.s.a.x, d.s.a.y, d.s.b, 0.0); }`,
    'VERTEX',
    300,
  );
  const fsa = compile(`#version 300 es
     precision mediump float; out vec4 o; void main(){ o = vec4(1.0); }`, 'FRAGMENT', 300);
  const la = linkProgram(vsa, fsa);
  check(la.ok, `(a) struct-member block links (${la.ok ? '' : la.log})`);
  if (la.ok) {
    const p = la.program;
    check(p.uniformBlocks.length === 1 && p.uniformBlocks[0].size === 16,
      `(a) block size 16 (got ${JSON.stringify(p.uniformBlocks.map((b) => b.size))})`);
    const m = new Map(p.uniformBlocks[0].activeUniforms.map((u) => [u.name, u]));
    check(
      m.get('d.s.a')?.offset === 0 && m.get('d.s.a')?.type === FLOAT_VEC2 && m.get('d.s.a')?.size === 1 &&
        m.get('d.s.b')?.offset === 8 && m.get('d.s.b')?.type === FLOAT && m.get('d.s.b')?.size === 1,
      `(a) flattened members d.s.a@0 vec2, d.s.b@8 float (got ${JSON.stringify([...m].map(([k, v]) => [k, v.offset, v.type, v.size]))})`,
    );
    const store = new Float32Array(4);
    store[0] = 1; store[1] = 2; store[2] = 3; // d.s.a, d.s.b
    const vctx = vertexCtx(p, {
      blockStores: [store],
      attribs: [new Float32Array([0, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.vertex.run(vctx);
    check(vctx.out.position[0] === 1 && vctx.out.position[1] === 2 && vctx.out.position[2] === 3,
      `(a) vertex reads struct members (got [${Array.from(vctx.out.position).join(', ')}])`);
  }

  // (b) EXACT crash scenario: array of structs in an instance-less block.
  const vsb = compile(
    `#version 300 es
     in vec4 aPos;
     void main(){ gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fsb = compile(
    `#version 300 es
     precision highp float;
     out vec4 o;
     struct light_t { vec4 intensity; };
     layout(std140) uniform lightData { light_t lights[2]; };
     void main(){ o = lights[0].intensity + lights[1].intensity; }`,
    'FRAGMENT',
    300,
  );
  const lb = linkProgram(vsb, fsb);
  check(lb.ok, `(b) array-of-structs block links (${lb.ok ? '' : lb.log})`);
  if (lb.ok) {
    const p = lb.program;
    check(p.uniformBlocks.length === 1 && p.uniformBlocks[0].name === 'lightData' && p.uniformBlocks[0].size === 32,
      `(b) block 'lightData' size 32 (got ${JSON.stringify(p.uniformBlocks)})`);
    const m = new Map(p.uniformBlocks[0].activeUniforms.map((u) => [u.name, u]));
    check(
      m.get('lights[0].intensity')?.offset === 0 && m.get('lights[0].intensity')?.type === FLOAT_VEC4 &&
        m.get('lights[0].intensity')?.size === 1 && m.get('lights[0].intensity')?.arrayStride === 0 &&
        m.get('lights[1].intensity')?.offset === 16 && m.get('lights[1].intensity')?.type === FLOAT_VEC4 &&
        m.get('lights[1].intensity')?.size === 1 && m.get('lights[1].intensity')?.arrayStride === 0,
      `(b) per-element leaves lights[0].intensity@0 / lights[1].intensity@16 size 1 (got ${JSON.stringify([...m].map(([k, v]) => [k, v.offset, v.type, v.size, v.arrayStride]))})`,
    );
    const um = p.uniforms.filter((u) => u.blockIndex === 0);
    check(
      um.length === 2 && um.every((u) => u.location === -1) &&
        um.map((u) => u.name).join(',') === 'lights[0].intensity,lights[1].intensity',
      `(b) Program.uniforms block members (got ${JSON.stringify(um)})`,
    );
    const store = new Float32Array(8); // 32 bytes
    store[0] = 1; store[1] = 2; store[2] = 3; store[3] = 4; // lights[0].intensity
    store[4] = 10; store[5] = 20; store[6] = 30; store[7] = 40; // lights[1].intensity
    const fctx = fragmentCtx(p, [], { blockStores: [store] });
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 11 && c[1] === 22 && c[2] === 33 && c[3] === 44,
      `(b) fragment reads both elements: [11,22,33,44] (got [${Array.from(c).join(', ')}])`);
  }

  // (c) array of structs containing ARRAYS: nested member arrays recurse.
  const vsc = compile(
    `#version 300 es
     in vec4 aPos;
     void main(){ gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fsc = compile(
    `#version 300 es
     precision highp float;
     out vec4 o;
     struct light_t { vec4 intensity[3]; };
     layout(std140) uniform lightData { light_t lights[2]; };
     void main(){ o = lights[0].intensity[1] + lights[1].intensity[2]; }`,
    'FRAGMENT',
    300,
  );
  const lc = linkProgram(vsc, fsc);
  check(lc.ok, `(c) structs-containing-arrays block links (${lc.ok ? '' : lc.log})`);
  if (lc.ok) {
    const p = lc.program;
    check(p.uniformBlocks.length === 1 && p.uniformBlocks[0].size === 96,
      `(c) block size 96 (got ${JSON.stringify(p.uniformBlocks.map((b) => b.size))})`);
    const m = new Map(p.uniformBlocks[0].activeUniforms.map((u) => [u.name, u]));
    check(
      m.get('lights[0].intensity[0]')?.offset === 0 && m.get('lights[0].intensity[0]')?.size === 3 &&
        m.get('lights[0].intensity[0]')?.arrayStride === 16 && m.get('lights[0].intensity[0]')?.type === FLOAT_VEC4 &&
        m.get('lights[1].intensity[0]')?.offset === 48 && m.get('lights[1].intensity[0]')?.size === 3 &&
        m.get('lights[1].intensity[0]')?.arrayStride === 16 && m.get('lights[1].intensity[0]')?.type === FLOAT_VEC4,
      `(c) nested leaves lights[0].intensity[0]@0 / lights[1].intensity[0]@48 size 3 stride 16 (got ${JSON.stringify([...m].map(([k, v]) => [k, v.offset, v.size, v.arrayStride]))})`,
    );
    const store = new Float32Array(24); // 96 bytes
    store[4] = 1; // lights[0].intensity[1] (offset 16)
    store[20] = 2; // lights[1].intensity[2] (offset 80)
    const fctx = fragmentCtx(p, [], { blockStores: [store] });
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(c[0] === 3 && c[1] === 0 && c[2] === 0 && c[3] === 0,
      `(c) const-indexed reads: [3,0,0,0] (got [${Array.from(c).join(', ')}])`);
  }

  // (d) ARRAYED block containing an array of structs: per-element groups.
  const vsd = compile(
    `#version 300 es
     in vec4 aPos;
     void main(){ gl_Position = aPos; }`,
    'VERTEX',
    300,
  );
  const fsd = compile(
    `#version 300 es
     precision highp float;
     out vec4 o;
     struct light_t { vec4 intensity; };
     uniform lightData { light_t lights[2]; } ld[2];
     void main(){ o = ld[0].lights[1].intensity + ld[1].lights[0].intensity; }`,
    'FRAGMENT',
    300,
  );
  const ldd = linkProgram(vsd, fsd);
  check(ldd.ok, `(d) arrayed block links (${ldd.ok ? '' : ldd.log})`);
  if (ldd.ok) {
    const p = ldd.program;
    check(
      p.uniformBlocks.length === 2 && p.uniformBlocks[0].name === 'ld[0]' && p.uniformBlocks[1].name === 'ld[1]' &&
        p.uniformBlocks[0].index === 0 && p.uniformBlocks[1].index === 0 && p.uniformBlocks[0].size === 32,
      `(d) 'ld[0]','ld[1]' shared index 0 size 32 (got ${JSON.stringify(p.uniformBlocks)})`,
    );
    const m0 = new Map(p.uniformBlocks[0].activeUniforms.map((u) => [u.name, u]));
    const m1 = new Map(p.uniformBlocks[1].activeUniforms.map((u) => [u.name, u]));
    check(
      m0.get('ld[0].lights[0].intensity')?.offset === 0 && m0.get('ld[0].lights[1].intensity')?.offset === 16 &&
        m1.get('ld[1].lights[0].intensity')?.offset === 32 && m1.get('ld[1].lights[1].intensity')?.offset === 48,
      `(d) per-element leaves @0/16/32/48 (got ${JSON.stringify([...m0, ...m1].map(([k, v]) => [k, v.offset]))})`,
    );
    const um = p.uniforms.filter((u) => u.blockIndex === 0);
    check(
      um.length === 4 && um.map((u) => u.name).join(',') ===
        'ld[0].lights[0].intensity,ld[0].lights[1].intensity,ld[1].lights[0].intensity,ld[1].lights[1].intensity',
      `(d) Program.uniforms block members (got ${JSON.stringify(um.map((u) => u.name))})`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 19. struct with an ARRAY MEMBER passed by value to a user function  */
/*     (pinned regression for CTS conformance/glsl/bugs/sampler-array- */
/*     struct-function-arg.html). Before the `array` case in           */
/*     flatNames's struct recursion (env.ts), makeParamLocal threw     */
/*     "codegen: array member 'arg$c0__sam' inside a flat struct is    */
/*     unsupported" → the linker caught it and the LINK failed (page:  */
/*     1 PASS / 1 FAIL). The array case flattens per element           */
/*     (arg__sam__0, arg__sam__1, ...), matching flatComponents, so    */
/*     the inliner's per-component binding copies the uniform struct's */
/*     sampler units into the param locals and arg.sam[0] reads the    */
/*     right unit.                                                     */
/* ------------------------------------------------------------------ */

{
  const vsa = compile(
    `attribute vec4 aPos;
     void main() { gl_Position = aPos; }`,
    'VERTEX',
    100,
  );
  const fsa = compile(
    `precision mediump float;
     struct S { sampler2D sam[2]; };
     uniform S uni;
     vec4 useSampler(S arg) { return texture2D(arg.sam[0], vec2(0.0, 0.0)); }
     void main() { gl_FragColor = vec4(useSampler(uni)); }`,
    'FRAGMENT',
    100,
  );
  const la = linkProgram(vsa, fsa);
  check(la.ok, `struct-with-array-member by-value param pair links (${la.ok ? '' : la.log})`);
  if (la.ok) {
    const p = la.program;
    // Member arrays stay WHOLE in the reflection (bf1c78c semantics, CTS
    // shader-with-array-of-structs-containing-arrays.html): one entry
    // 'uni.sam[0]' with size = 2; the second element is reachable via the
    // uniformMap alias 'uni.sam[1]' → location+1.
    const sam0 = p.uniforms.find((u) => u.name === 'uni.sam[0]');
    check(
      !!sam0 && sam0.size === 2 && sam0.location >= 0 && p.uniforms.length === 1,
      `member sampler array stays whole: one 'uni.sam[0]' size 2 (got ${JSON.stringify(p.uniforms.map((u) => ({ n: u.name, s: u.size, l: u.location })))})`,
    );
    const alias = p.uniformMap.get('uni.sam[1]');
    check(
      alias !== undefined && alias.name === 'uni.sam[0]',
      `uniformMap alias 'uni.sam[1]' → entry 'uni.sam[0]' (got ${alias === undefined ? 'undefined' : alias.name})`,
    );
    if (sam0) {
      p.intStore[sam0.location] = 5;
      p.intStore[sam0.location + 1] = 9;
      let seenUnit = -1;
      const tex: any = { out: new Float32Array(4) };
      tex.sample2D = (unit: number): boolean => {
        seenUnit = unit;
        tex.out[0] = 1;
        tex.out[1] = 0.5;
        tex.out[2] = 0.25;
        tex.out[3] = 1;
        return true;
      };
      const fctx = fragmentCtx(p, [], { tex });
      p.fragment.run(fctx);
      check(
        seenUnit === 5,
        `arg.sam[0] bound the right sampler unit (got ${seenUnit}, want 5)`,
      );
      check(
        fctx.out.color[0][0] === 1 && fctx.out.color[0][1] === 0.5,
        `useSampler result color from tex.out (got [${Array.from(fctx.out.color[0]).join(', ')}])`,
      );
    }
  }
}

/* ------------------------------------------------------------------ */
/* Fragment output layout (ES 3.00): arrays, scalars, locations        */
/* ------------------------------------------------------------------ */

{
  const LIM = { maxDrawBuffers: 8 };
  const vs = compile('#version 300 es\nvoid main() { gl_Position = vec4(0,0,0,1); }', 'VERTEX', 300);

  // ARRAY output: 8 slots at 0..7 (draw-buffers MRT style, no explicit
  // location — the ONLY output variable → base 0).
  {
    const fs = compile(
      `#version 300 es
       precision mediump float;
       out vec4 my_FragData[8];
       void main() { my_FragData[0] = vec4(1.0); my_FragData[7] = vec4(2.0); }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs, { limits: LIM });
    check(l.ok, `8-slot array output links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const outs = l.program.fragment.outputs;
      check(
        outs.length === 8 && outs.every((o, i) => o.location === i && o.type === FLOAT_VEC4),
        `array output → 8 per-slot entries 0..7 FLOAT_VEC4 (got ${JSON.stringify(outs)})`,
      );
      const fctx = fragmentCtx(l.program);
      fctx.out.color = Array.from({ length: 8 }, () => new Float32Array(4));
      l.program.fragment.run(fctx);
      check(
        fctx.out.color[0][0] === 1 && fctx.out.color[7][0] === 2 && fctx.out.color[3][0] === 0,
        `array output run writes slots 0 and 7 (got ${fctx.out.color[0][0]},${fctx.out.color[7][0]})`,
      );
    }
  }

  // Explicit layout(location=2) on an array → slots 2,3,4.
  {
    const fs = compile(
      `#version 300 es
       precision mediump float;
       layout(location = 2) out vec4 a[3];
       void main() { a[1] = vec4(1.0); }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs, { limits: LIM });
    check(l.ok, `explicit-location array links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const outs = l.program.fragment.outputs;
      check(
        outs.length === 3 && outs[0].location === 2 && outs[1].location === 3 && outs[2].location === 4,
        `explicit array base 2 → slots 2,3,4 (got ${JSON.stringify(outs)})`,
      );
    }
  }

  // Non-contiguous explicit scalar locations (gl-get-frag-data-location).
  {
    const fs = compile(
      `#version 300 es
       precision mediump float;
       layout(location = 2) out vec4 fragColor0;
       layout(location = 0) out vec4 fragColor1;
       void main() { fragColor0 = vec4(0,1,0,1); fragColor1 = vec4(1,0,0,1); }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs, { limits: LIM });
    check(l.ok, `explicit non-contiguous scalar outputs link (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const outs = l.program.fragment.outputs;
      check(
        outs.length === 2 && outs[0].location === 2 && outs[1].location === 0,
        `scalar outputs at 2 and 0 (got ${JSON.stringify(outs)})`,
      );
      const fctx = fragmentCtx(l.program);
      fctx.out.color = Array.from({ length: 3 }, () => new Float32Array(4));
      l.program.fragment.run(fctx);
      check(
        fctx.out.color[2][1] === 1 && fctx.out.color[0][0] === 1,
        `run writes color[2] and color[0] (got ${fctx.out.color[2][1]},${fctx.out.color[0][0]})`,
      );
    }
  }

  // Scalar INT output (vertex-id): type GL_INT, run writes the int value.
  {
    const vsI = compile(
      `#version 300 es
       flat out highp int vVertexID;
       void main() { vVertexID = gl_VertexID; gl_Position = vec4(0,0,0,1); }`,
      'VERTEX',
      300,
    );
    const fs = compile(
      `#version 300 es
       flat in highp int vVertexID;
       out highp int oVertexID;
       void main() { oVertexID = vVertexID; }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vsI, fs, { limits: LIM });
    check(l.ok, `scalar int output links (${l.ok ? '' : l.log})`);
    if (l.ok) {
      const p = l.program;
      check(
        p.fragment.outputs.length === 1 && p.fragment.outputs[0].location === 0 && p.fragment.outputs[0].type === INT,
        `int output → [{0, GL_INT}] (got ${JSON.stringify(p.fragment.outputs)})`,
      );
      const vctx = vertexCtx(p, { vertexId: 12345 });
      p.vertex.run(vctx);
      const fctx = fragmentCtx(p, [vctx.out.varyings]);
      p.fragment.run(fctx);
      check(fctx.out.color[0][0] === 12345, `int output run writes 12345 (got ${fctx.out.color[0][0]})`);
    }
  }

  // Location conflicts are link errors (explicit duplicate + array overlap).
  {
    const fsDup = compile(
      `#version 300 es
       precision mediump float;
       layout(location = 0) out vec4 a;
       layout(location = 0) out vec4 b;
       void main() { a = vec4(1.0); b = vec4(2.0); }`,
      'FRAGMENT',
      300,
    );
    const ld = linkProgram(vs, fsDup, { limits: LIM });
    check(!ld.ok && ld.log.includes('conflicts with another output'), `duplicate location 0 → link error (${ld.ok ? '' : ld.log})`);

    const fsOver = compile(
      `#version 300 es
       precision mediump float;
       layout(location = 0) out vec4 a[2];
       layout(location = 1) out vec4 b;
       void main() { a[0] = vec4(1.0); b = vec4(2.0); }`,
      'FRAGMENT',
      300,
    );
    const lo = linkProgram(vs, fsOver, { limits: LIM });
    check(!lo.ok && lo.log.includes('conflicts with another output'), `array/slot overlap → link error (${lo.ok ? '' : lo.log})`);
  }

  // Multiple output variables without explicit locations → link error
  // (WEBGL_blend_func_extended 'locations300' expectation).
  {
    const fs = compile(
      `#version 300 es
       precision mediump float;
       out vec4 color0;
       out vec4 color1;
       void main() { color0 = vec4(1.0); color1 = vec4(2.0); }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs, { limits: LIM });
    check(!l.ok && l.log.includes("must declare layout(location=)"), `multi-output without locations → link error (${l.ok ? '' : l.log})`);
  }

  // maxDrawBuffers bound check on array expansion.
  {
    const fs = compile(
      `#version 300 es
       precision mediump float;
       layout(location = 6) out vec4 a[3];
       void main() { a[0] = vec4(1.0); }`,
      'FRAGMENT',
      300,
    );
    const l = linkProgram(vs, fs, { limits: { maxDrawBuffers: 8 } });
    check(!l.ok && l.log.includes('exceeds maxDrawBuffers'), `array expansion past maxDrawBuffers → link error (${l.ok ? '' : l.log})`);
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`linker selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
