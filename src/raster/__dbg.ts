/**
 * TEMPORARY debug script — deleted before commit.
 * Replicates Test B (shared diagonal) step by step to locate the
 * triangle-coverage failure.
 */
import {
  draw, createSurface, getFormat, RasterProgram, DrawCall, Surface,
  VARYINGS_OFFSET, RGBA8, TRIANGLES, LEQUAL, FUNC_ADD, ONE,
} from './index';
import { clipPrimitive, applyViewportTransform } from './clip';
import { rasterizeTriangle, signedArea2, edgeFunction } from './triangles';
import { createRasterState } from './rasterizer';
import { runFragment } from './fragment-ops';

function prog(run: (ctx: any) => void, opts: any = {}): RasterProgram {
  return {
    varyings: opts.varyings ?? [],
    fragment: {
      run, usesDerivatives: opts.usesDerivatives ?? false,
      usesFragDepth: opts.usesFragDepth ?? false,
      outputs: [{ location: 0, type: RGBA8 }],
    },
  } as any;
}

function dc(partial: any): DrawCall {
  const base: any = {
    mode: TRIANGLES, count: 0, first: 0, instanceCount: 1,
    vertices: new Float32Array(0), vertexStride: 5, varyingsOffset: VARYINGS_OFFSET,
    program: prog(() => {}), fb: null!, viewport: { x: 0, y: 0, w: 4, h: 4 },
    depthRange: { near: 0, far: 1 }, scissor: { enabled: false, x: 0, y: 0, w: 0, h: 0 },
    cull: { enabled: false, face: 0x0405, frontFace: 0x0901 },
    polygonOffset: { enabled: false, factor: 0, units: 0 }, dither: false,
    colorMask: [[true, true, true, true]], blend: { enabled: false, srcRGB: ONE, dstRGB: 0, srcAlpha: ONE, dstAlpha: 0, eqRGB: FUNC_ADD, eqAlpha: FUNC_ADD, color: [0, 0, 0, 0] },
    depthTest: { enabled: false, func: LEQUAL }, depthMask: true,
    stencilTest: { enabled: false, front: { func: 0x0207, ref: 0, valueMask: 0xff, writeMask: 0xff, fail: 0x1e00, zfail: 0x1e00, zpass: 0x1e00 }, back: { func: 0x0207, ref: 0, valueMask: 0xff, writeMask: 0xff, fail: 0x1e00, zfail: 0x1e00, zpass: 0x1e00 } },
    sampleCoverage: { enabled: false, value: 1, invert: false },
    rasterizerDiscard: false, lineWidth: 1, textures: [], drawBuffers: [0],
    uniforms: new Float32Array(0),
  };
  return Object.assign(base, partial) as DrawCall;
}

function winVert(x: number, y: number, z = 0, w = 1, vary: number[] = [], ps = 1): number[] {
  return [x / 2 - 1, y / 2 - 1, z, w, ps, ...vary];
}

function px(s: Surface, x: number, y: number): number[] {
  const d = s.data as Uint8Array;
  const o = (y * s.width + x) * 4;
  return [d[o], d[o + 1], d[o + 2], d[o + 3]];
}

const s = createSurface(RGBA8, 2, 2);
const fb = { color: [s], depth: null, stencil: null, width: 2, height: 2, samples: 1 };
const v1 = [winVert(0, 0), winVert(2, 0), winVert(0, 2)].flat();
console.log('clip verts T1:', JSON.stringify(Array.from(v1)));

// ---- Step 1: reproduce emitTriangle's clip+viewport manually
const stride = 5;
const primBuf = new Float32Array(3 * stride);
primBuf.set(v1);
const clipA = new Float32Array(7 * stride);
const clipB = new Float32Array(7 * stride);
const nv = clipPrimitive(primBuf, 0, stride, 3, clipA, clipB, 0);
console.log('clipPrimitive nv =', nv);
applyViewportTransform(clipB, 0, stride, nv, { x: 0, y: 0, w: 4, h: 4 }, { near: 0, far: 1 });
const recs: number[][] = [];
for (let k = 0; k < nv; k++) {
  recs.push([clipB[k * stride], clipB[k * stride + 1], clipB[k * stride + 2], clipB[k * stride + 3]]);
}
console.log('window records:', JSON.stringify(recs));
console.log('signedArea2(0,1,2):', signedArea2(clipB, 0, 1, 2, stride));

// ---- Step 2: per-pixel inside math replication for the fan triangle (0,1,2)
const d = dc({
  count: 3, vertices: new Float32Array(v1),
  program: prog((ctx) => ctx.out.color[0].set([255, 0, 0, 255])), fb,
});
const rs = createRasterState(d);
console.log('totalVaryComponents:', rs.totalVaryComponents);

for (let k = 1; k + 1 < nv; k++) {
  const [x0, y0] = [clipB[0 * stride], clipB[0 * stride + 1]];
  const [x1, y1] = [clipB[k * stride], clipB[k * stride + 1]];
  const [x2, y2] = [clipB[(k + 1) * stride], clipB[(k + 1) * stride + 1]];
  let a2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  console.log(`--- fan tri (0,${k},${k + 1}) area2=${a2}`);
  if (a2 < 0) { /* swap not needed for math here */ }
  for (let py = 0; py < 2; py++) {
    for (let pxx = 0; pxx < 2; pxx++) {
      const cx = pxx + 0.5, cy = py + 0.5;
      const e0 = (x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0);
      const e1 = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1);
      const e2 = (x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2);
      const tl0 = y1 < y0 || (y1 === y0 && x1 < x0);
      const tl1 = y2 < y1 || (y2 === y1 && x2 < x1);
      const tl2 = y0 < y2 || (y0 === y2 && x0 < x2);
      const inside = (e0 > 0 || (e0 === 0 && tl0)) && (e1 > 0 || (e1 === 0 && tl1)) && (e2 > 0 || (e2 === 0 && tl2));
      console.log(`  px(${pxx},${py}) e=(${e0.toFixed(2)},${e1.toFixed(2)},${e2.toFixed(2)}) tl=(${tl0},${tl1},${tl2}) inside=${inside}`);
    }
  }
}

// ---- Step 3: call the real rasterizeTriangle on the clipped records
rasterizeTriangle(clipB, 0, 1, 2, stride, rs);
console.log('after rasterizeTriangle (0,1,2):');
for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) console.log(`  px(${x},${y}) =`, JSON.stringify(px(s, x, y)));

// ---- Step 4: full draw() comparison
const s2 = createSurface(RGBA8, 2, 2);
const fb2 = { color: [s2], depth: null, stencil: null, width: 2, height: 2, samples: 1 };
draw(dc({ count: 3, vertices: new Float32Array(v1), program: prog((ctx) => ctx.out.color[0].set([255, 0, 0, 255])), fb: fb2 }));
console.log('after full draw():');
for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) console.log(`  px(${x},${y}) =`, JSON.stringify(px(s2, x, y)));
