/**
 * End-to-end tests of the REAL raster draw pipeline (src/raster `draw()`):
 * triangle fill (top-left rule), depth testing + depth mask, occlusion
 * counting (sampleCountRef), diamond-exit line rasterization, blending math,
 * instancing addressing, scissor + colorMask, and stencil ops. These areas
 * previously had NO unit coverage (the temporary src/raster/__selfcheck.ts
 * was dead code whose stale expectations never matched the pipeline; its
 * behaviors were re-verified against the current contracts before writing
 * these tests).
 *
 * Pipeline conventions relied on here (pinned in src/raster/CONTEXT.md):
 * - Vertex records are CLIP-space [x,y,z,w,pointSize,varyings...]; with the
 *   default viewport {0,0,4,4}, clip = win/2 − 1 (see winVert below).
 * - winZ = near + (far−near)·(z/w·0.5+0.5) — depth is stored as Float32.
 * - Fragment colors are normalized floats; 8-bit targets encode via
 *   Math.round(c·255) (formats.ts P_U8).
 * - Depth/stencil surfaces are never auto-cleared by the pipeline — fill the
 *   depth Float32Array with 1.0 to "clear" it.
 * - Surface rows: row 0 = BOTTOM (window y); px(x, y) = data[(y·w+x)·4..].
 */
import { describe, it, expect } from "vitest";
import {
  draw,
  createSurface,
  getDepthData,
  getStencilData,
  VARYINGS_OFFSET,
  type RasterProgram,
  type DrawCall,
  type FragmentExecCtx,
  type FramebufferTarget,
  type Surface,
} from "../../src/raster/index";
import { GL, expectArrayClose } from "./helpers";

/* ================================================================== */
/* Harness (mirrors the deleted src/raster/__selfcheck.ts)             */
/* ================================================================== */

/** Builds a minimal RasterProgram with the given fragment function. */
function prog(
  run: (ctx: FragmentExecCtx) => void,
  opts: {
    varyings?: RasterProgram["varyings"];
    usesDerivatives?: boolean;
    usesFragDepth?: boolean;
  } = {},
): RasterProgram {
  return {
    varyings: opts.varyings ?? [],
    fragment: {
      run,
      usesDerivatives: opts.usesDerivatives ?? false,
      usesFragDepth: opts.usesFragDepth ?? false,
      outputs: [{ location: 0, type: GL.RGBA8 }],
    },
  } as RasterProgram;
}

/** Builds a DrawCall with spec-default state; `partial` overrides it. */
function dc(partial: Partial<DrawCall>): DrawCall {
  const stencilFace = {
    func: GL.ALWAYS, ref: 0, valueMask: 0xff, writeMask: 0xff,
    fail: GL.KEEP, zfail: GL.KEEP, zpass: GL.KEEP,
  };
  const base: DrawCall = {
    mode: GL.TRIANGLES, count: 0, first: 0, instanceCount: 1,
    vertices: new Float32Array(0), vertexStride: VARYINGS_OFFSET,
    varyingsOffset: VARYINGS_OFFSET,
    program: prog(() => {}), fb: null!, // every test sets fb
    viewport: { x: 0, y: 0, w: 4, h: 4 },
    depthRange: { near: 0, far: 1 },
    scissor: { enabled: false, x: 0, y: 0, w: 0, h: 0 },
    cull: { enabled: false, face: GL.BACK, frontFace: GL.CCW },
    polygonOffset: { enabled: false, factor: 0, units: 0 },
    dither: false,
    colorMask: [[true, true, true, true]],
    blend: {
      enabled: false, srcRGB: GL.ONE, dstRGB: GL.ZERO,
      srcAlpha: GL.ONE, dstAlpha: GL.ZERO,
      eqRGB: GL.FUNC_ADD, eqAlpha: GL.FUNC_ADD,
      color: [0, 0, 0, 0],
    },
    depthTest: { enabled: false, func: GL.LEQUAL },
    depthMask: true,
    stencilTest: { enabled: false, front: stencilFace, back: stencilFace },
    sampleCoverage: { enabled: false, value: 1, invert: false },
    rasterizerDiscard: false, lineWidth: 1,
    textures: [], drawBuffers: [0], uniforms: new Float32Array(0),
  };
  return { ...base, ...partial } as DrawCall;
}

/** One clip-space vertex record for the default 4×4 viewport
 *  (clip = win/2 − 1). `z` is the CLIP z (winZ = z·0.5+0.5 for near=0,far=1). */
function winVert(x: number, y: number, z = 0, w = 1): number[] {
  return [x / 2 - 1, y / 2 - 1, z, w, 1];
}

/** Packs vertex records into the draw's Float32Array. */
function verts(...records: number[][]): Float32Array {
  return new Float32Array(records.flat());
}

/** Reads pixel (x, y) as [r, g, b, a] (row 0 = BOTTOM). */
function px(s: Surface, x: number, y: number): number[] {
  const d = s.data as Uint8Array;
  const o = (y * s.width + x) * 4;
  return [d[o], d[o + 1], d[o + 2], d[o + 3]];
}

/** Builds a FramebufferTarget for a color surface (+ optional depth/stencil). */
function fb(
  color: Surface,
  depth: Surface | null = null,
  stencil: Surface | null = null,
): FramebufferTarget {
  return {
    color: [color], depth, stencil,
    width: color.width, height: color.height, samples: 1,
  };
}

/** Asserts every pixel of a 4×4 surface equals `expected`. */
function expectAllPx(s: Surface, expected: number[]): void {
  for (let y = 0; y < s.height; y++) {
    for (let x = 0; x < s.width; x++) {
      expect(px(s, x, y), `pixel (${x},${y})`).toEqual(expected);
    }
  }
}

/**
 * Full-viewport triangle: window (−2,−2),(6,−2),(2,10) → clip
 * (−2,−2),(2,−2),(0,4). All 16 pixel centers are strictly inside (hypotenuse
 * 3x+y ≤ 4 in clip space; the nearest center (0.75,0.75) has 3x+y = 3), and
 * no center lies on any edge, so the top-left rule is not exercised here.
 */
function fullScreenTriangle(z = 0): Float32Array {
  return verts(winVert(-2, -2, z), winVert(6, -2, z), winVert(2, 10, z));
}

/* ================================================================== */
/* 1. Triangle fill (top-left rule)                                    */
/* ================================================================== */

describe("triangle fill (top-left rule)", () => {
  it("covers exactly the 8 interior pixels of window triangle (0,0),(4,0),(2,4)", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    const d = dc({
      count: 3,
      vertices: verts(winVert(0, 0), winVert(4, 0), winVert(2, 4)),
      program: prog((ctx) => { ctx.out.color[0].set([1, 0, 0, 1]); }),
      fb: fb(s),
    });
    draw(d);
    // Interior: y ≥ 0, y ≤ 2x, y ≤ 8−2x (window). No pixel center lies on an
    // edge (the edge lines pass between centers), so this is pure interior
    // coverage — the top-left rule is not needed to decide any pixel here.
    const covered: [number, number][] = [
      [0, 0], [1, 0], [2, 0], [3, 0],
      [1, 1], [2, 1],
      [1, 2], [2, 2],
    ];
    for (const [x, y] of covered) {
      expect(px(s, x, y), `pixel (${x},${y})`).toEqual([255, 0, 0, 255]);
    }
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        if (!covered.some(([cx, cy]) => cx === x && cy === y)) {
          expect(px(s, x, y), `pixel (${x},${y}) must stay clear`)
            .toEqual([0, 0, 0, 0]);
        }
      }
    }
  });
});

/* ================================================================== */
/* 2. Depth test (LEQUAL) + depth mask                                 */
/* ================================================================== */

describe("depth test (LEQUAL) + depth writes", () => {
  it("writes 0.625, rejects a farther draw, accepts a nearer one and overwrites", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    const ds = createSurface(GL.DEPTH_COMPONENT16, 4, 4);
    const depth = getDepthData(ds);
    depth.fill(1.0); // the pipeline does NOT auto-clear — clear by hand
    const target = fb(s, ds);
    const depthState = { enabled: true, func: GL.LEQUAL };

    // winZ = near + (far−near)·(z/w·0.5+0.5): clip z 0.25 → 0.625,
    // 0.75 → 0.875, −0.5 → 0.25 (near=0, far=1).
    draw(dc({
      count: 3, vertices: fullScreenTriangle(0.25),
      program: prog((ctx) => { ctx.out.color[0].set([1, 0, 0, 1]); }),
      fb: target, depthTest: depthState, depthMask: true,
    }));
    expectAllPx(s, [255, 0, 0, 255]);
    expectArrayClose(depth, new Float32Array(16).fill(0.625), 1e-5);

    // Farther triangle: 0.875 > 0.625 → fails LEQUAL, no color/depth change.
    draw(dc({
      count: 3, vertices: fullScreenTriangle(0.75),
      program: prog((ctx) => { ctx.out.color[0].set([0, 1, 0, 1]); }),
      fb: target, depthTest: depthState, depthMask: true,
    }));
    expectAllPx(s, [255, 0, 0, 255]);
    expectArrayClose(depth, new Float32Array(16).fill(0.625), 1e-5);

    // Nearer triangle: 0.25 ≤ 0.625 → passes, overwrites color AND depth.
    draw(dc({
      count: 3, vertices: fullScreenTriangle(-0.5),
      program: prog((ctx) => { ctx.out.color[0].set([0, 0, 1, 1]); }),
      fb: target, depthTest: depthState, depthMask: true,
    }));
    expectAllPx(s, [0, 0, 255, 255]);
    expectArrayClose(depth, new Float32Array(16).fill(0.25), 1e-5);
  });
});

/* ================================================================== */
/* 3. Occlusion counting (sampleCountRef)                              */
/* ================================================================== */

describe("occlusion counting (sampleCountRef)", () => {
  // Window triangle (0,0),(2,0),(0,2) → clip (−1,−1),(0,−1),(−1,0). Its
  // hypotenuse is x+y = 2 (window): pixel (0,0)'s center (0.5,0.5) is
  // strictly inside, while (1,0) and (0,1) centers lie exactly ON the
  // up-left hypotenuse, which the pinned top-left convention excludes
  // (edge (2,0)→(0,2) is neither top nor left) — so exactly 1 sample.
  const tri = () => verts(winVert(0, 0), winVert(2, 0), winVert(0, 2));

  it("counts exactly one passing sample", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    const count = { value: 0 };
    draw(dc({
      count: 3, vertices: tri(),
      program: prog((ctx) => { ctx.out.color[0].set([1, 1, 1, 1]); }),
      fb: fb(s), sampleCountRef: count,
    }));
    expect(count.value).toBe(1);
  });

  it("counts nothing when rasterizerDiscard is set", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    const count = { value: 0 };
    draw(dc({
      count: 3, vertices: tri(),
      program: prog((ctx) => { ctx.out.color[0].set([1, 1, 1, 1]); }),
      fb: fb(s), sampleCountRef: count, rasterizerDiscard: true,
    }));
    expect(count.value).toBe(0);
    expectAllPx(s, [0, 0, 0, 0]); // and nothing was drawn
  });
});

/* ================================================================== */
/* 4. Line rasterization (diamond-exit)                                */
/* ================================================================== */

describe("line rasterization (diamond-exit)", () => {
  const p = prog((ctx) => { ctx.out.color[0].set([1, 1, 1, 1]); });

  it("covers exactly (0,1),(1,1),(2,1) for the half-integer segment (0,1.5)→(4,1.5)", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    draw(dc({
      mode: GL.LINES, count: 2,
      vertices: verts(winVert(0, 1.5), winVert(4, 1.5)),
      program: p, fb: fb(s),
    }));
    // The segment passes through the diamonds of pixels (0..2, 1) and EXITS
    // each before its end; pixel (3,1)'s diamond is only reached at the
    // segment's very END (its right vertex) — the half-open [start, end)
    // diamond-exit convention excludes the endpoint, so (3,1) stays clear.
    for (let x = 0; x < 3; x++) {
      expect(px(s, x, 1), `pixel (${x},1)`).toEqual([255, 255, 255, 255]);
    }
    expect(px(s, 3, 1)).toEqual([0, 0, 0, 0]);
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        if (y !== 1 || x > 2) expect(px(s, x, y), `pixel (${x},${y})`).toEqual([0, 0, 0, 0]);
      }
    }
  });

  it("covers nothing for the tangent segment y = 1.0", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    draw(dc({
      mode: GL.LINES, count: 2,
      vertices: verts(winVert(0, 1), winVert(4, 1)),
      program: p, fb: fb(s),
    }));
    // y = 1.0 only touches the diamonds' bottom/top vertices (tangent):
    // f(t) = |u|+|v| ≥ 0.5 everywhere → no fragment.
    expectAllPx(s, [0, 0, 0, 0]);
  });
});

/* ================================================================== */
/* 5. Blending                                                        */
/* ================================================================== */

describe("blending (SRC_ALPHA / ONE_MINUS_SRC_ALPHA, FUNC_ADD)", () => {
  it("computes src·As + dst·(1−As) per channel with 8-bit rounding", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    const target = fb(s);
    const v = fullScreenTriangle();
    // Base: opaque red (blend disabled).
    draw(dc({
      count: 3, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([1, 0, 0, 1]); }),
      fb: target,
    }));
    // Source (0, 1, 0, 128/255) — alpha EXACTLY 128/255, not 0.5, so the
    // blend factors are exact rationals.
    draw(dc({
      count: 3, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([0, 1, 0, 128 / 255]); }),
      fb: target,
      blend: {
        enabled: true,
        srcRGB: GL.SRC_ALPHA, dstRGB: GL.ONE_MINUS_SRC_ALPHA,
        srcAlpha: GL.SRC_ALPHA, dstAlpha: GL.ONE_MINUS_SRC_ALPHA,
        eqRGB: GL.FUNC_ADD, eqAlpha: GL.FUNC_ADD,
        color: [0, 0, 0, 0],
      },
    }));
    // Hand-computed from the GL blend equation (As = 128/255, dst = red):
    //   R = 0·As + 1·(1−As) = 127/255       → round(127)  = 127
    //   G = 1·As + 0·(1−As) = 128/255       → round(128)  = 128
    //   B = 0
    //   A = As·As + 1·(1−As) = 48769/65025  → round(191.25) = 191
    // NOTE: an earlier investigation claimed the pipeline emits
    // [128,128,0,128] here; that value does not satisfy the GL equation
    // (alpha would be ≈191, red 127). This assertion is the spec-correct
    // one — if it fails, that is a real pipeline bug, not a test bug.
    expectArrayClose(px(s, 0, 0), [127, 128, 0, 191], 3);
  });
});

/* ================================================================== */
/* 6. Instancing                                                      */
/* ================================================================== */

describe("instancing (vertex addressing first + i·count + j)", () => {
  it("draws both instances at their offsets from one 6-record buffer", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    // Instance 0 → records 0..2: window (0,0),(2,0),(0,2) covers exactly
    // pixel (0,0) (see occlusion test); instance 1 → records 3..5: window
    // (2,2),(4,2),(2,4) covers exactly pixel (2,2).
    const v = verts(
      winVert(0, 0), winVert(2, 0), winVert(0, 2),
      winVert(2, 2), winVert(4, 2), winVert(2, 4),
    );
    draw(dc({
      count: 3, instanceCount: 2, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([1, 1, 1, 1]); }),
      fb: fb(s),
    }));
    expect(px(s, 0, 0)).toEqual([255, 255, 255, 255]);
    expect(px(s, 2, 2)).toEqual([255, 255, 255, 255]);
    expect(px(s, 1, 1)).toEqual([0, 0, 0, 0]);
  });
});

/* ================================================================== */
/* 7. Scissor + colorMask                                             */
/* ================================================================== */

describe("scissor + colorMask", () => {
  it("limits writes to the scissor rectangle", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    draw(dc({
      count: 3, vertices: fullScreenTriangle(),
      program: prog((ctx) => { ctx.out.color[0].set([1, 1, 1, 1]); }),
      fb: fb(s),
      scissor: { enabled: true, x: 1, y: 1, w: 2, h: 2 },
    }));
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const inRect = x >= 1 && x <= 2 && y >= 1 && y <= 2;
        expect(px(s, x, y), `pixel (${x},${y})`)
          .toEqual(inRect ? [255, 255, 255, 255] : [0, 0, 0, 0]);
      }
    }
  });

  it("read-modify-write: masked channels keep their current value", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    const target = fb(s);
    const v = fullScreenTriangle();
    draw(dc({
      count: 3, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([1, 0, 0, 1]); }),
      fb: target,
    }));
    // Only the green channel is unmasked: writing green over red keeps the
    // red channel (glColorMask: false = no change to that channel).
    draw(dc({
      count: 3, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([0, 1, 0, 1]); }),
      fb: target,
      colorMask: [[false, true, false, false]],
    }));
    expectAllPx(s, [255, 255, 0, 255]);
    // The mask literal from the original task sheet, [true,false,true,true],
    // WRITES red/blue/alpha from the source (mask=true = channel takes the
    // source value) — a green write then zeroes red and blue, and only green
    // survives from the previous pass. The task's "stays red" expectation was
    // inconsistent with its own mask; this is the spec-correct behavior.
    draw(dc({
      count: 3, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([0, 1, 0, 1]); }),
      fb: target,
      colorMask: [[true, false, true, true]],
    }));
    expectAllPx(s, [0, 255, 0, 255]);
  });
});

/* ================================================================== */
/* 8. Stencil ops                                                     */
/* ================================================================== */

describe("stencil ops", () => {
  const face = (func: number, ref: number, zpass: number) => ({
    func, ref, valueMask: 0xff, writeMask: 0xff,
    fail: GL.KEEP, zfail: GL.KEEP, zpass,
  });

  it("REPLACE ref 7, then EQUAL ref 7 with INCR_WRAP → 8", () => {
    const s = createSurface(GL.RGBA8, 4, 4);
    const ss = createSurface(GL.STENCIL_INDEX8, 4, 4);
    const target = fb(s, null, ss);
    const st = getStencilData(ss);
    const v = fullScreenTriangle();

    // (a) ALWAYS + zpass REPLACE ref 7 → stencil 7 at every covered pixel.
    draw(dc({
      count: 3, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([1, 1, 1, 1]); }),
      fb: target,
      stencilTest: {
        enabled: true,
        front: face(GL.ALWAYS, 7, GL.REPLACE),
        back: face(GL.ALWAYS, 7, GL.REPLACE),
      },
    }));
    for (let i = 0; i < st.length; i++) {
      expect(st[i], `stencil index ${i} after REPLACE`).toBe(7);
    }
    expectAllPx(s, [255, 255, 255, 255]);

    // (b) EQUAL ref 7 + zpass INCR_WRAP → passes where stencil == 7,
    //     stencil becomes 8, color written.
    draw(dc({
      count: 3, vertices: v,
      program: prog((ctx) => { ctx.out.color[0].set([0, 1, 0, 1]); }),
      fb: target,
      stencilTest: {
        enabled: true,
        front: face(GL.EQUAL, 7, GL.INCR_WRAP),
        back: face(GL.EQUAL, 7, GL.INCR_WRAP),
      },
    }));
    for (let i = 0; i < st.length; i++) {
      expect(st[i], `stencil index ${i} after INCR_WRAP`).toBe(8);
    }
    expectAllPx(s, [0, 255, 0, 255]);
  });
});
