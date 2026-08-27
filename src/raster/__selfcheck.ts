/**
 * TEMPORARY integration self-test for src/raster — DELETED before committing.
 * Exercises the full draw pipeline: rasterizer → clip → viewport → triangles/
 * lines/points → fragment-ops → surfaces, plus sampler, occlusion counting,
 * instancing, blend, depth.
 */
import {
  draw, createSurface, getFormat,
  RasterProgram, DrawCall, Surface, TextureImage, SamplerState,
  VARYINGS_OFFSET, RGBA8, TRIANGLES, LINES, POINTS, LEQUAL, FUNC_ADD, ONE,
  SRC_ALPHA, ONE_MINUS_SRC_ALPHA, REPEAT, NEAREST, TEXTURE_2D,
} from './index';

let failures = 0;
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  PASS ${name}`);
  else { failures++; console.log(`  FAIL ${name} ${detail}`); }
}
function px(s: Surface, x: number, y: number): number[] {
  const d = s.data as Uint8Array;
  const o = (y * s.width + x) * 4;
  return [d[o], d[o + 1], d[o + 2], d[o + 3]];
}
function near(a: number[], b: number[], tol = 2): boolean {
  return a.every((v, i) => Math.abs(v - b[i]) <= tol);
}

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
  // viewport {0,0,4,4}: clip = win/2 - 1 (for the default 4x4 viewport)
  return [x / 2 - 1, y / 2 - 1, z, w, ps, ...vary];
}

// ---------------------------------------------------------------- Test A: triangle fill
{
  const s = createSurface(RGBA8, 4, 4);
  const v = [winVert(0, 0), winVert(4, 0), winVert(2, 4)].flat();
  const d = dc({ count: 3, vertices: new Float32Array(v), program: prog((ctx) => ctx.out.color[0].set([255, 0, 0, 255])), fb: { color: [s], depth: null, stencil: null, width: 4, height: 4, samples: 1 } });
  draw(d);
  const covered: [number, number][] = [[0, 0], [1, 0], [2, 0], [3, 0], [1, 1], [2, 1], [3, 1], [2, 2], [3, 2], [2, 3], [3, 3]];
  const empty: [number, number][] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]];
  let ok = covered.every(([x, y]) => near(px(s, x, y), [255, 0, 0, 255])) && empty.every(([x, y]) => near(px(s, x, y), [0, 0, 0, 0]));
  console.log('Test A: triangle fill (top-left rule)');
  check('covered set exact (11 px)', ok);
}

// ------------------------------------------------------- Test B: shared diagonal (crack-free)
{
  const s = createSurface(RGBA8, 2, 2);
  const v1 = [winVert(0, 0), winVert(2, 0), winVert(0, 2)].flat();
  const v2 = [winVert(2, 0), winVert(2, 2), winVert(0, 2)].flat();
  const fb = { color: [s], depth: null, stencil: null, width: 2, height: 2, samples: 1 };
  draw(dc({ count: 3, vertices: new Float32Array(v1), program: prog((ctx) => ctx.out.color[0].set([255, 0, 0, 255])), fb }));
  draw(dc({ count: 3, vertices: new Float32Array(v2), program: prog((ctx) => ctx.out.color[0].set([0, 255, 0, 255])), fb }));
  const a = px(s, 0, 0), b = px(s, 1, 0), c = px(s, 0, 1), d = px(s, 1, 1);
  const ok = near(a, [255, 0, 0, 255]) && near(b, [0, 255, 0, 255]) && near(c, [0, 255, 0, 255]) && near(d, [0, 255, 0, 255]);
  console.log('Test B: shared diagonal crack-free (each px exactly one tri)');
  check('T1=(0,0), T2=rest, no overlap', ok, JSON.stringify([a, b, c, d]));
}

// ------------------------------------------------------- Test C: depth test + mask
{
  const s = createSurface(RGBA8, 2, 2);
  const ds = createSurface(0x81a5 /*DEPTH_COMPONENT16*/, 2, 2);
  const fb = { color: [s], depth: ds, stencil: null, width: 2, height: 2, samples: 1 };
  const far = [winVert(-1, -1, 0.75), winVert(3, -1, 0.75), winVert(-1, 3, 0.75)].flat();
  const nearV = [winVert(-1, -1, 0.25), winVert(3, -1, 0.25), winVert(-1, 3, 0.25)].flat();
  const pFar = prog((ctx) => ctx.out.color[0].set([255, 0, 0, 255]));
  const pNear = prog((ctx) => ctx.out.color[0].set([0, 255, 0, 255]));
  draw(dc({ count: 3, vertices: new Float32Array(far), program: pFar, fb, depthTest: { enabled: true, func: LEQUAL } }));
  draw(dc({ count: 3, vertices: new Float32Array(nearV), program: pNear, fb, depthTest: { enabled: true, func: LEQUAL } }));
  const dep = ds.data as Float32Array;
  let ok = true;
  for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
    if (!near(px(s, x, y), [0, 255, 0, 255])) ok = false;
    if (Math.abs(dep[y * 2 + x] - 0.25) > 0.001) ok = false;
  }
  console.log('Test C: depth test LEQUAL keeps near, depth written');
  check('all green + depth 0.25', ok, `depth=${JSON.stringify(Array.from(dep))}`);
}

// ------------------------------------------------------- Test D: occlusion counting
{
  const s = createSurface(RGBA8, 2, 2);
  const fb = { color: [s], depth: null, stencil: null, width: 2, height: 2, samples: 1 };
  const v = [winVert(0, 0), winVert(1, 0), winVert(0, 1)].flat();
  const count = { value: 0 };
  draw(dc({ count: 3, vertices: new Float32Array(v), program: prog((ctx) => ctx.out.color[0].set([1, 1, 1, 255])), fb, sampleCountRef: count }));
  console.log('Test D: occlusion sampleCountRef');
  check('counts exactly 1 passing sample', count.value === 1, `got ${count.value}`);
  const count2 = { value: 0 };
  draw(dc({ count: 3, vertices: new Float32Array(v), program: prog((ctx) => ctx.out.color[0].set([1, 1, 1, 255])), fb, sampleCountRef: count2, rasterizerDiscard: true }));
  check('rasterizerDiscard counts 0', count2.value === 0, `got ${count2.value}`);
}

// ------------------------------------------------------- Test E: textured triangle (varying + sampler)
{
  const s = createSurface(RGBA8, 4, 4);
  const texData = new Uint8Array([0, 0, 255, 255, 255, 255, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255]);
  const img: TextureImage = {
    target: TEXTURE_2D, internalFormat: RGBA8, info: getFormat(RGBA8)!,
    width: 2, height: 2, depth: 1,
    levels: [{ width: 2, height: 2, depth: 1, data: [texData] }],
    baseLevel: 0, maxLevel: 0, immutable: false, complete: true,
  };
  const state: SamplerState = { minFilter: NEAREST, magFilter: NEAREST, wrapS: REPEAT, wrapT: REPEAT, wrapR: REPEAT, minLod: -1000, maxLod: 1000, compareMode: 0, compareFunc: LEQUAL, maxAnisotropy: 1 };
  const varyings = [{ name: 'v_uv', type: 0x8b50, components: 2, flat: false }];
  const v = [winVert(0, 0, 0, 1, [0, 0]), winVert(4, 0, 0, 1, [1, 0]), winVert(0, 4, 0, 1, [0, 1])].flat();
  const program = prog((ctx: any) => {
    ctx.tex.sample2D(0, ctx.varyings[0].v[0], ctx.varyings[0].v[1], 0, 0, 0, 0, 0);
    const c = ctx.tex.out;
    ctx.out.color[0].set([c[0] * 255, c[1] * 255, c[2] * 255, 255]);
  }, { varyings });
  const d = dc({
    count: 3, vertexStride: 7, vertices: new Float32Array(v), program,
    textures: [{ img, state }],
    fb: { color: [s], depth: null, stencil: null, width: 4, height: 4, samples: 1 },
  });
  draw(d);
  // Pixel (1,1) center (1.5,1.5): u = 0.375, v = 0.375 → texel (0,0).
  // Texture rows: row0 (bottom) = [blue, white], row1 (top) = [green, red].
  // Row 0 = BOTTOM: u=0.375 → x=0; v=0.375 → y=0 → bottom-left texel = BLUE.
  const p = px(s, 1, 1);
  const ok = near(p, [0, 0, 255, 255]);
  console.log('Test E: textured triangle via varyings + TextureEnv');
  check('pixel(1,1) = blue texel', ok, JSON.stringify(p));
}

// ------------------------------------------------------- Test F: derivatives (quad path)
{
  const s = createSurface(RGBA8, 4, 4);
  const varyings = [{ name: 'v_x', type: 0x8b50, components: 2, flat: false }];
  const v = [winVert(0, 0, 0, 1, [0, 0]), winVert(4, 0, 0, 1, [4, 0]), winVert(0, 4, 0, 1, [0, 4])].flat();
  const program = prog((ctx: any) => {
    const dx = ctx.varyings[0].ddx[0];
    ctx.out.color[0].set([dx * 255, 0, 0, 255]);
  }, { varyings, usesDerivatives: true });
  const d = dc({
    count: 3, vertexStride: 7, vertices: new Float32Array(v), program,
    fb: { color: [s], depth: null, stencil: null, width: 4, height: 4, samples: 1 },
  });
  draw(d);
  const p = px(s, 1, 1);
  const ok = Math.abs(p[0] - 255) <= 3;
  console.log('Test F: 2x2 quad path with derivatives (dFdx(v_x) = 1)');
  check('red ≈ 255', ok, JSON.stringify(p));
}

// ------------------------------------------------------- Test G: lines (diamond-exit)
{
  const s = createSurface(RGBA8, 4, 4);
  const fb = { color: [s], depth: null, stencil: null, width: 4, height: 4, samples: 1 };
  const p = prog((ctx) => ctx.out.color[0].set([255, 255, 255, 255]));
  const v1 = [winVert(0, 1.5), winVert(4, 1.5)].flat();
  draw(dc({ mode: LINES, count: 2, vertices: new Float32Array(v1), program: p, fb }));
  let ok = true;
  for (let x = 0; x < 4; x++) if (!near(px(s, x, 1), [255, 255, 255, 255])) ok = false;
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) if (y !== 1 && !near(px(s, x, y), [0, 0, 0, 0])) ok = false;
  const s2 = createSurface(RGBA8, 4, 4);
  const fb2 = { color: [s2], depth: null, stencil: null, width: 4, height: 4, samples: 1 };
  const v2 = [winVert(0, 1.0), winVert(4, 1.0)].flat();
  draw(dc({ mode: LINES, count: 2, vertices: new Float32Array(v2), program: p, fb: fb2 }));
  let ok2 = true;
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) if (!near(px(s2, x, y), [0, 0, 0, 0])) ok2 = false;
  console.log('Test G: line diamond-exit');
  check('y=1.5 covers row 1 only', ok);
  check('y=1.0 covers nothing', ok2);
}

// ------------------------------------------------------- Test H: points
{
  const s = createSurface(RGBA8, 4, 4);
  const fb = { color: [s], depth: null, stencil: null, width: 4, height: 4, samples: 1 };
  const p = prog((ctx) => ctx.out.color[0].set([255, 255, 255, 255]));
  const v = [winVert(1.5, 1.5, 0, 1, [], 1)].flat();
  draw(dc({ mode: POINTS, count: 1, vertices: new Float32Array(v), program: p, fb }));
  let ok = near(px(s, 1, 1), [255, 255, 255, 255]);
  for (let y = 0; y < 4 && ok; y++) for (let x = 0; x < 4; x++) if (!(x === 1 && y === 1) && !near(px(s, x, y), [0, 0, 0, 0])) ok = false;
  const s2 = createSurface(RGBA8, 4, 4);
  const fb2 = { color: [s2], depth: null, stencil: null, width: 4, height: 4, samples: 1 };
  const v2 = [winVert(2, 2, 0, 1, [], 2000)].flat();
  draw(dc({ mode: POINTS, count: 1, vertices: new Float32Array(v2), program: p, fb: fb2 }));
  const ok2 = near(px(s2, 1, 1), [255, 255, 255, 255]);
  console.log('Test H: points (size 1 coverage + size clamp)');
  check('size-1 point covers exactly (1,1)', ok);
  check('size 2000 clamped, no crash, center covered', ok2);
}

// ------------------------------------------------------- Test I: blend
{
  const s = createSurface(RGBA8, 2, 2);
  const fb = { color: [s], depth: null, stencil: null, width: 2, height: 2, samples: 1 };
  const v = [winVert(-1, -1), winVert(3, -1), winVert(-1, 3)].flat();
  draw(dc({ count: 3, vertices: new Float32Array(v), program: prog((ctx) => ctx.out.color[0].set([255, 0, 0, 255])), fb }));
  draw(dc({
    count: 3, vertices: new Float32Array(v),
    program: prog((ctx) => ctx.out.color[0].set([0, 1, 0, 0.5])),
    fb, blend: { enabled: true, srcRGB: SRC_ALPHA, dstRGB: ONE_MINUS_SRC_ALPHA, srcAlpha: ONE, dstAlpha: 0, eqRGB: FUNC_ADD, eqAlpha: FUNC_ADD, color: [0, 0, 0, 0] },
  }));
  const p = px(s, 0, 0);
  const ok = near(p, [127, 127, 0], 3);
  console.log('Test I: blending SRC_ALPHA/ONE_MINUS_SRC_ALPHA');
  check('(127,127,0) over red', ok, JSON.stringify(p));
}

// ------------------------------------------------------- Test J: instancing
{
  const s = createSurface(RGBA8, 4, 4);
  const fb = { color: [s], depth: null, stencil: null, width: 4, height: 4, samples: 1 };
  const v = [
    winVert(0, 0), winVert(1, 0), winVert(0, 1),
    winVert(2, 2), winVert(3, 2), winVert(2, 3),
  ].flat();
  const d = dc({
    count: 3, instanceCount: 2, vertices: new Float32Array(v),
    program: prog((ctx) => ctx.out.color[0].set([255, 255, 255, 255])), fb,
  });
  draw(d);
  const ok = near(px(s, 0, 0), [255, 255, 255, 255]) && near(px(s, 2, 2), [255, 255, 255, 255]) && near(px(s, 1, 1), [0, 0, 0, 0]);
  console.log('Test J: instancing (2 instances)');
  check('both instances drawn at their offsets', ok);
}

// ------------------------------------------------------- Test K: scissor + colorMask
{
  const s = createSurface(RGBA8, 4, 4);
  const fb = { color: [s], depth: null, stencil: null, width: 4, height: 4, samples: 1 };
  const v = [winVert(-1, -1), winVert(3, -1), winVert(-1, 3)].flat();
  draw(dc({
    count: 3, vertices: new Float32Array(v), program: prog((ctx) => ctx.out.color[0].set([255, 255, 255, 255])), fb,
    scissor: { enabled: true, x: 1, y: 1, w: 2, h: 2 },
  }));
  let ok = near(px(s, 1, 1), [255, 255, 255, 255]) && near(px(s, 2, 1), [255, 255, 255, 255]) && near(px(s, 0, 0), [0, 0, 0, 0]);
  const s2 = createSurface(RGBA8, 2, 2);
  const fb2 = { color: [s2], depth: null, stencil: null, width: 2, height: 2, samples: 1 };
  const v2 = [winVert(-1, -1), winVert(3, -1), winVert(-1, 3)].flat();
  draw(dc({ count: 3, vertices: new Float32Array(v2), program: prog((ctx) => ctx.out.color[0].set([255, 255, 255, 255])), fb: fb2 }));
  draw(dc({ count: 3, vertices: new Float32Array(v2), program: prog((ctx) => ctx.out.color[0].set([0, 255, 0, 255])), fb: fb2, colorMask: [[false, true, false, false]] }));
  const p = px(s2, 0, 0);
  const ok2 = near(p, [255, 255, 255, 255]);
  console.log('Test K: scissor rect + colorMask read-modify-write');
  check('scissor limits writes', ok);
  check('green write masked out (stays white)', ok2, JSON.stringify(p));
}

// ------------------------------------------------------- Test L: stencil ops
{
  const s = createSurface(RGBA8, 2, 2);
  const ss = createSurface(0x8d48 /*STENCIL_INDEX8*/, 2, 2);
  const fb = { color: [s], depth: null, stencil: ss, width: 2, height: 2, samples: 1 };
  const v = [winVert(-1, -1), winVert(3, -1), winVert(-1, 3)].flat();
  draw(dc({
    count: 3, vertices: new Float32Array(v), program: prog((ctx) => ctx.out.color[0].set([1, 1, 1, 1])), fb,
    stencilTest: { enabled: true, front: { func: 0x0207, ref: 7, valueMask: 0xff, writeMask: 0xff, fail: 0x1e00, zfail: 0x1e00, zpass: 0x1e01 }, back: { func: 0x0207, ref: 7, valueMask: 0xff, writeMask: 0xff, fail: 0x1e00, zfail: 0x1e00, zpass: 0x1e01 } },
  }));
  const st = ss.data as Uint8Array;
  let ok = st[0] === 7 && st[3] === 7;
  draw(dc({
    count: 3, vertices: new Float32Array(v), program: prog((ctx) => ctx.out.color[0].set([0, 255, 0, 255])), fb,
    stencilTest: { enabled: true, front: { func: 0x0202, ref: 7, valueMask: 0xff, writeMask: 0xff, fail: 0x1e00, zfail: 0x1e00, zpass: 0x8507 }, back: { func: 0x0202, ref: 7, valueMask: 0xff, writeMask: 0xff, fail: 0x1e00, zfail: 0x1e00, zpass: 0x8507 } },
  }));
  ok = ok && near(px(s, 0, 0), [0, 255, 0, 255]) && st[0] === 8;
  console.log('Test L: stencil REPLACE then EQUAL+INCR_WRAP');
  check('stencil 7 written, EQUAL passes, INCR_WRAP → 8', ok, `stencil=${JSON.stringify(Array.from(st))}`);
}

if (failures === 0) console.log('\nALL SELF-CHECKS PASSED');
else console.log(`\n${failures} SELF-CHECKS FAILED`);
process.exit(failures === 0 ? 0 : 1);
