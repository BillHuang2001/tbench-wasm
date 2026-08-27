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
 *   8. UBO declared → {ok:false} 'linker: uniform blocks not supported'
 *      (deferred to a later task — documents the deferral).
 *
 * Run: npx tsx src/glsl/selftest-link.ts
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
const FLOAT = 0x1406; // 5126

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
/* 8. UBO declared → deferred (not supported)                          */
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
  check(!l.ok && l.log.includes('uniform blocks') && l.log.includes('not supported'),
    `UBO deferred with 'linker: uniform blocks not supported' (${logOf(l)})`);
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
