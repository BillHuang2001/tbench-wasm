/**
 * selftest-ubo-array.ts — per-element uniform-block indices for ARRAYED
 * uniform blocks (`uniform UBO { vec4 v; } blocks[2];`).
 *
 * GLES 3.0 / WebGL2 treat every element of an arrayed block as its OWN
 * uniform block: glGetUniformBlockIndex('UBO[0]') and 'UBO[1]' return
 * DISTINCT consecutive indices, each element binds to its own binding point /
 * buffer range (CTS conformance2/buffers/uniform-buffers.html
 * runNamedArrayDrawTest), and member offsets are ELEMENT-LOCAL. The linker
 * must therefore emit one UNIQUE index per element and codegen must address
 * ctx.blockStores[elementIndex] (constant element accesses bake the element's
 * own index; dynamic `b[i]` strides the STORE array by 1).
 *
 * Every check FAILS on the pre-fix linker (one shared index for all elements
 * + byte-stride dynamic addressing): the metadata checks see index 0 for
 * 'b[1]', and the run pins read blockStores[0] out of bounds (NaN).
 *
 * Run: npx tsx src/glsl/codegen/selftest-ubo-array.ts
 * Prints "selftest-ubo-array: N checks" and exits 0 only when all pass.
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

function fragmentCtx(prog: Program, extra: Record<string, unknown> = {}): any {
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
    varyings: [],
    ...extra,
  };
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

const vsTrivial = compile(
  `#version 300 es
   void main() { gl_Position = vec4(0.0); }`,
  'VERTEX',
  300,
);

/* ------------------------------------------------------------------ */
/* 1. Const element accesses read their OWN store; per-element         */
/*    metadata (unique indices, element-local member offsets).         */
/* ------------------------------------------------------------------ */

{
  const fs = compile(
    `#version 300 es
     precision mediump float;
     uniform UBO { vec4 v; } blocks[2];
     layout(location=0) out vec4 o;
     void main() { o = blocks[0].v + blocks[1].v; }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vsTrivial, fs);
  check(l.ok, `arrayed block links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // One UNIQUE index per element; size = ONE instance; member offsets
    // element-local (0 for both elements' 'v').
    check(
      p.uniformBlocks.length === 2 && p.uniformBlocks[0].name === 'blocks[0]' &&
        p.uniformBlocks[1].name === 'blocks[1]' && p.uniformBlocks[0].index === 0 &&
        p.uniformBlocks[1].index === 1 && p.uniformBlocks[0].size === 16 &&
        p.uniformBlocks[1].size === 16,
      `per-element infos: 'blocks[0]' index 0, 'blocks[1]' index 1, size 16 (got ${JSON.stringify(p.uniformBlocks)})`,
    );
    const m0 = p.uniformBlocks[0].activeUniforms[0];
    const m1 = p.uniformBlocks[1].activeUniforms[0];
    check(
      m0 !== undefined && m1 !== undefined && m0.name === 'blocks[0].v' && m0.offset === 0 &&
        m1.name === 'blocks[1].v' && m1.offset === 0,
      `element-local member offsets (got ${JSON.stringify([m0, m1])})`,
    );
    const u0 = p.uniforms.find((u) => u.name === 'blocks[0].v');
    const u1 = p.uniforms.find((u) => u.name === 'blocks[1].v');
    check(
      u0 !== undefined && u1 !== undefined && u0.blockIndex === 0 && u1.blockIndex === 1 &&
        u0.location === -1 && u1.location === -1,
      `Program.uniforms blockIndex per element (got ${JSON.stringify([u0, u1])})`,
    );
    // Distinct stores: element 0 red, element 1 blue.
    const red = new Float32Array([1, 0, 0, 1]);
    const blue = new Float32Array([0, 0, 1, 1]);
    const fctx = fragmentCtx(p, { blockStores: [red, blue] });
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(
      c[0] === 1 && c[1] === 0 && c[2] === 1 && c[3] === 2,
      `blocks[0].v + blocks[1].v reads BOTH stores → (1,0,1,2) (got [${Array.from(c).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 2. DYNAMIC element index `blocks[i].v` strides the STORE array.     */
/* ------------------------------------------------------------------ */

{
  const fs = compile(
    `#version 300 es
     precision mediump float;
     uniform UBO { vec4 v; } blocks[2];
     uniform int i;
     layout(location=0) out vec4 o;
     void main() { o = blocks[i].v; }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vsTrivial, fs);
  check(l.ok, `dynamic element index links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    const iLoc = p.uniformMap.get('i')!.location;
    const red = new Float32Array([1, 0, 0, 1]);
    const blue = new Float32Array([0, 0, 1, 1]);
    const fctx = fragmentCtx(p, { blockStores: [red, blue] });
    p.intStore[iLoc] = 0;
    p.fragment.run(fctx);
    let c = fctx.out.color[0];
    check(
      c[0] === 1 && c[2] === 0,
      `dynamic i=0 → element 0's store (got [${Array.from(c).join(', ')}])`,
    );
    p.intStore[iLoc] = 1;
    p.fragment.run(fctx);
    c = fctx.out.color[0];
    check(
      c[0] === 0 && c[2] === 1,
      `dynamic i=1 → element 1's store (got [${Array.from(c).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 3. THREE-element block (base + 2): const element 2 + dynamic 2,     */
/*    plus a second block AFTER it keeps the next dense index.         */
/* ------------------------------------------------------------------ */

{
  const fs = compile(
    `#version 300 es
     precision mediump float;
     uniform UBO { vec4 v; } blocks[3];
     uniform Other { float x; } other;
     uniform int i;
     layout(location=0) out vec4 o;
     void main() { o = blocks[2].v + blocks[i].v + vec4(other.x); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vsTrivial, fs);
  check(l.ok, `3-element arrayed block + following block links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    // 'blocks' occupies indices 0..2; 'Other' gets the NEXT index (3).
    check(
      p.uniformBlocks.length === 4 &&
        p.uniformBlocks.map((b) => b.name).join(',') === 'blocks[0],blocks[1],blocks[2],Other' &&
        p.uniformBlocks.map((b) => b.index).join(',') === '0,1,2,3' &&
        p.uniformBlocks[3].size === 16,
      `dense indices across elements + following block (got ${JSON.stringify(p.uniformBlocks.map((b) => ({ n: b.name, i: b.index, s: b.size })))})`,
    );
    const iLoc = p.uniformMap.get('i')!.location;
    const stores = [
      new Float32Array([1, 0, 0, 1]),
      new Float32Array([0, 1, 0, 1]),
      new Float32Array([0, 0, 1, 1]),
    ];
    const otherStore = new Float32Array([0.5, 0, 0, 0]);
    const fctx = fragmentCtx(p, { blockStores: [...stores, otherStore] });
    p.intStore[iLoc] = 2;
    p.fragment.run(fctx);
    const c = fctx.out.color[0];
    check(
      c[0] === 0.5 && c[1] === 0.5 && c[2] === 2.5 && c[3] === 2.5,
      `blocks[2].v + blocks[i=2].v + vec4(other.x)=splat(0.5) → (0.5,0.5,2.5,2.5) (got [${Array.from(c).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 4. VERTEX-stage arrayed block + member array inside each element    */
/*    (`blocks[1].m[1]` const element, `blocks[i].m[0]` dynamic).      */
/* ------------------------------------------------------------------ */

{
  const vs = compile(
    `#version 300 es
     uniform UBO { vec4 v; float m[2]; } blocks[2];
     uniform int i;
     in vec4 aPos;
     void main() {
       gl_Position = aPos + vec4(blocks[1].m[1] + blocks[i].m[0] + blocks[0].v.x, 0.0, 0.0, 0.0);
     }`,
    'VERTEX',
    300,
  );
  const fs = compile(
    `#version 300 es
     precision mediump float;
     layout(location=0) out vec4 o;
     void main() { o = vec4(1.0); }`,
    'FRAGMENT',
    300,
  );
  const l = linkProgram(vs, fs);
  check(l.ok, `vertex arrayed block with member arrays links (${l.ok ? '' : l.log})`);
  if (l.ok) {
    const p = l.program;
    const iLoc = p.uniformMap.get('i')!.location;
    // Element 0: v=(1,0,0,0), m=[10, 11]; element 1: v=(0,0,0,0), m=[20, 21].
    // blocks[1].m[1] = 21, blocks[i=1].m[0] = 20, blocks[0].v = (1,0,0,0).
    const e0 = new Float32Array([1, 0, 0, 0, 10, 11]);
    const e1 = new Float32Array([0, 0, 0, 0, 20, 21]);
    const vctx = vertexCtx(p, {
      blockStores: [e0, e1],
      attribs: [new Float32Array([0, 0, 0, 1])],
      attribIndices: new Int32Array([0]),
    });
    p.intStore[iLoc] = 1;
    p.vertex.run(vctx);
    const pos = vctx.out.position;
    check(
      pos[0] === 42 && pos[1] === 0 && pos[2] === 0 && pos[3] === 1,
      `vertex: blocks[1].m[1](21) + blocks[i=1].m[0](20) + blocks[0].v.x(1) → x=42 (got [${Array.from(pos).join(', ')}])`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-ubo-array: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
