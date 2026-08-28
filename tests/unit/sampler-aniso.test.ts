/**
 * N-tap anisotropic filtering (src/raster/sampler-aniso.ts) —
 * EXT_texture_filter_anisotropic unit coverage: the gate (2D + mipmap
 * minification + TEXTURE_MAX_ANISOTROPY_EXT > 1), pow2ceilN rounding, N-tap
 * averaging along the MAJOR footprint axis, tap placement math, per-tap
 * trilinear blending (same level pair + same f for every tap), the isotropic
 * n = 1 single centered tap, degenerate-footprint N = pow2ceil(maxAniso), and
 * module-scratch / out-buffer hygiene across sequential calls.
 *
 * Reconstructed from the wave-10 temporary harness (sampler-aniso landed in
 * commit 8c7e34b); design rationale is in the module header, lines 1-52.
 * The module is NOT re-exported by src/raster/index.ts — it is imported
 * directly. `anisoTapParams` is module-level, so every test sets it before
 * calling and an afterEach restores the inactive default.
 */
import { describe, it, expect, afterEach } from "vitest";
import {
  anisoGate,
  pow2ceilN,
  anisoTapParams,
  sample2DAnisoTaps,
  type AnisoFilter2D,
} from "../../src/raster/sampler-aniso";
import type { SamplerState, TextureImage } from "../../src/raster/types";
import { getFormat } from "../../src/raster/formats";
import { GL, expectArrayClose } from "./helpers";

/* ================================================================== */
/* Harness                                                             */
/* ================================================================== */

/** One recorded filter2D invocation. */
interface TapCall {
  level: number;
  u: number;
  v: number;
}

/** Minimal TextureImage — sampler-aniso only reads target/width/height. */
function makeImage(width: number, height: number, target: GLenum = GL.TEXTURE_2D): TextureImage {
  return {
    target,
    internalFormat: GL.RGBA8,
    info: getFormat(GL.RGBA8)!,
    width,
    height,
    depth: 1,
    levels: [],
    baseLevel: 0,
    maxLevel: 1000,
    immutable: false,
    complete: true,
  };
}

/** SamplerState with spec-default fields; `partial` overrides them. */
function makeState(partial: Partial<SamplerState> = {}): SamplerState {
  return {
    minFilter: GL.LINEAR_MIPMAP_LINEAR,
    magFilter: GL.LINEAR,
    wrapS: GL.REPEAT,
    wrapT: GL.REPEAT,
    wrapR: GL.REPEAT,
    minLod: -1000,
    maxLod: 1000,
    compareMode: GL.NONE,
    compareFunc: GL.LEQUAL,
    maxAnisotropy: 1,
    ...partial,
  };
}

/**
 * Structural mirror of sampler.ts's SamplePlan (not exported from
 * sampler-aniso.ts; the per-tap filter only reads plan.u8Fast, but the full
 * shape keeps the stub assignable to AnisoFilter2D under contravariance).
 */
function makePlan() {
  return {
    base: 0,
    hi: 0,
    single: false,
    oneTexel: false,
    posFilter: GL.LINEAR,
    mipMin: false,
    u8Fast: false,
  };
}

/**
 * Recording stub for the per-tap filter: pushes (level, u, v) and writes
 * colorOf(level) — a constant per level — into out[0..3].
 */
function makeFilter2D(
  calls: TapCall[],
  colorOf: (level: number) => readonly [number, number, number, number],
): AnisoFilter2D {
  return (_img, _state, _plan, _filter, level, u, v, _layer, _shadow, _refQ, out) => {
    calls.push({ level, u, v });
    const c = colorOf(level);
    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
    out[3] = c[3];
  };
}

afterEach(() => {
  anisoTapParams.n = 0;
  anisoTapParams.majorU = true;
});

/* ================================================================== */
/* anisoGate                                                           */
/* ================================================================== */

describe("anisoGate", () => {
  it("is false when maxAnisotropy <= 1", () => {
    const img = makeImage(8, 8);
    expect(anisoGate(img, makeState({ maxAnisotropy: 0 }))).toBe(false);
    expect(anisoGate(img, makeState({ maxAnisotropy: 1 }))).toBe(false);
  });

  it("is false for non-2D targets (cube / 3D / 2D_ARRAY)", () => {
    const state = makeState({ maxAnisotropy: 8 });
    expect(anisoGate(makeImage(8, 8, GL.TEXTURE_CUBE_MAP), state)).toBe(false);
    expect(anisoGate(makeImage(8, 8, GL.TEXTURE_3D), state)).toBe(false);
    expect(anisoGate(makeImage(8, 8, GL.TEXTURE_2D_ARRAY), state)).toBe(false);
  });

  it("is false for non-mipmap min filters (magnification has no anisotropy)", () => {
    const img = makeImage(8, 8);
    expect(anisoGate(img, makeState({ maxAnisotropy: 8, minFilter: GL.NEAREST }))).toBe(false);
    expect(anisoGate(img, makeState({ maxAnisotropy: 8, minFilter: GL.LINEAR }))).toBe(false);
  });

  it("is true for 2D + maxAniso > 1 + each mipmap min filter (0x2700..0x2703)", () => {
    const img = makeImage(8, 8);
    for (const f of [
      GL.NEAREST_MIPMAP_NEAREST,
      GL.LINEAR_MIPMAP_NEAREST,
      GL.NEAREST_MIPMAP_LINEAR,
      GL.LINEAR_MIPMAP_LINEAR,
    ]) {
      expect(
        anisoGate(img, makeState({ maxAnisotropy: 2, minFilter: f })),
        `minFilter 0x${f.toString(16)}`,
      ).toBe(true);
    }
  });
});

/* ================================================================== */
/* pow2ceilN                                                           */
/* ================================================================== */

describe("pow2ceilN", () => {
  it("rounds up to the nearest power of two (pin maxAniso=3 → N=4)", () => {
    const cases: Array<[number, number]> = [
      [1, 1],
      [2, 2],
      [3, 4],
      [4, 4],
      [5, 8],
      [8, 8],
      [16, 16],
    ];
    for (const [x, want] of cases) {
      expect(pow2ceilN(x), `pow2ceilN(${x})`).toBe(want);
    }
  });
});

/* ================================================================== */
/* sample2DAnisoTaps                                                   */
/* ================================================================== */

describe("sample2DAnisoTaps", () => {
  it("averages N identical taps into out (N equally weighted)", () => {
    const calls: TapCall[] = [];
    const color = [0.25, 0.5, 0.75, 1] as const;
    const filter2D = makeFilter2D(calls, () => color);
    anisoTapParams.n = 4;
    anisoTapParams.majorU = true;
    const out = new Float32Array(4);
    sample2DAnisoTaps(filter2D, makeImage(8, 8), makeState(), makePlan(), GL.LINEAR,
      1, -1, 0, 0.5, 0.25, 0, false, 0, 1, out);
    expectArrayClose(out, color);
    expect(calls).toHaveLength(4);
    for (const c of calls) expect(c.level).toBe(1);
  });

  it("places taps along the major axis, centered on the sample UV", () => {
    const calls: TapCall[] = [];
    const filter2D = makeFilter2D(calls, () => [0, 0, 0, 1] as const);
    const out = new Float32Array(4);
    anisoTapParams.n = 4;

    // majorU: step along u; lambda = 1, width = 8 → scale = 2/8 = 0.25;
    // uu = 0.5 + {-0.375, -0.125, 0.125, 0.375} · 0.25.
    anisoTapParams.majorU = true;
    sample2DAnisoTaps(filter2D, makeImage(8, 8), makeState(), makePlan(), GL.LINEAR,
      0, -1, 0, 0.5, 0.25, 0, false, 0, 1, out);
    const expectedU = [0.40625, 0.46875, 0.53125, 0.59375];
    calls.forEach((c, i) => {
      expect(c.u, `tap ${i} u`).toBeCloseTo(expectedU[i], 10);
      expect(c.v, `tap ${i} v`).toBeCloseTo(0.25, 10);
    });

    // majorU=false: step along v; scale uses img.height (4) → 2/4 = 0.5;
    // vv = 0.25 + {-0.375, -0.125, 0.125, 0.375} · 0.5; u unchanged.
    calls.length = 0;
    anisoTapParams.majorU = false;
    sample2DAnisoTaps(filter2D, makeImage(8, 4), makeState(), makePlan(), GL.LINEAR,
      0, -1, 0, 0.5, 0.25, 0, false, 0, 1, out);
    const expectedV = [0.0625, 0.1875, 0.3125, 0.4375];
    calls.forEach((c, i) => {
      expect(c.v, `tap ${i} v`).toBeCloseTo(expectedV[i], 10);
      expect(c.u, `tap ${i} u`).toBeCloseTo(0.5, 10);
    });
  });

  it("trilinear: blends the SAME level pair with the SAME f for every tap", () => {
    const calls: TapCall[] = [];
    const c0 = [0.2, 0.4, 0.6, 0.8] as const;
    const c1 = [0.8, 0.6, 0.4, 0.2] as const;
    const filter2D = makeFilter2D(calls, (level) => (level === 2 ? c0 : c1));
    anisoTapParams.n = 4;
    anisoTapParams.majorU = true;
    const out = new Float32Array(4);
    sample2DAnisoTaps(filter2D, makeImage(8, 8), makeState(), makePlan(), GL.LINEAR,
      2, 3, 0.5, 0.5, 0.5, 0, false, 0, 1, out);
    // Per tap: 0.5·(c0 + c1) = [0.5, 0.5, 0.5, 0.5]; averaging identical
    // tap results leaves it unchanged.
    expectArrayClose(out, [0.5, 0.5, 0.5, 0.5]);
    expect(calls).toHaveLength(8); // 4 taps × 2 levels
    for (let i = 0; i < 4; i++) {
      expect(calls[2 * i].level, `tap ${i} level0`).toBe(2);
      expect(calls[2 * i + 1].level, `tap ${i} level1`).toBe(3);
      // Same UV for both levels of a tap (same footprint position).
      expect(calls[2 * i].u).toBe(calls[2 * i + 1].u);
      expect(calls[2 * i].v).toBe(calls[2 * i + 1].v);
    }
  });

  it("isotropic n = 1: single centered tap (offset exactly 0)", () => {
    const calls: TapCall[] = [];
    const filter2D = makeFilter2D(calls, () => [0.1, 0.2, 0.3, 1] as const);
    anisoTapParams.n = 1;
    anisoTapParams.majorU = true;
    const out = new Float32Array(4);
    sample2DAnisoTaps(filter2D, makeImage(8, 8), makeState(), makePlan(), GL.LINEAR,
      0, -1, 0, 0.5, 0.25, 0, false, 0, 1, out);
    expectArrayClose(out, [0.1, 0.2, 0.3, 1]);
    expect(calls).toHaveLength(1);
    expect(calls[0].u).toBe(0.5);
    expect(calls[0].v).toBe(0.25);
  });

  it("degenerate footprint: n = pow2ceil(maxAniso) still averages n taps", () => {
    for (const maxAniso of [3, 5, 6]) {
      const n = pow2ceilN(maxAniso); // 4, 8, 8
      const calls: TapCall[] = [];
      const color = [0.3, 0.1, 0.9, 1] as const;
      const filter2D = makeFilter2D(calls, () => color);
      anisoTapParams.n = n;
      anisoTapParams.majorU = false; // degenerate v-axis → step along v
      const out = new Float32Array(4);
      sample2DAnisoTaps(filter2D, makeImage(8, 8), makeState(), makePlan(), GL.LINEAR,
        0, -1, 0, 0.5, 0.5, 0, false, 0, 1, out);
      expectArrayClose(out, color);
      expect(calls, `maxAniso ${maxAniso} tap count`).toHaveLength(n);
    }
  });

  it("sequential reuse: independent results into the same out buffer", () => {
    const callsA: TapCall[] = [];
    const callsB: TapCall[] = [];
    const colorA = [0.9, 0.8, 0.7, 0.6] as const;
    const colorB = [0.1, 0.2, 0.3, 0.4] as const;
    const out = new Float32Array([9, 9, 9, 9]); // junk pre-fill

    anisoTapParams.n = 2;
    anisoTapParams.majorU = true;
    sample2DAnisoTaps(makeFilter2D(callsA, () => colorA), makeImage(8, 8),
      makeState(), makePlan(), GL.LINEAR, 0, -1, 0, 0.5, 0.25, 0, false, 0, 1, out);
    expectArrayClose(out, colorA); // all 4 channels fully overwritten
    expect(callsA).toHaveLength(2);
    expect(callsA.every((c) => c.level === 0)).toBe(true);

    // Second call with different params + different color into the SAME out:
    // module-level scratch (mipA/mipB, tap params) must not leak.
    anisoTapParams.n = 4;
    anisoTapParams.majorU = false;
    sample2DAnisoTaps(makeFilter2D(callsB, () => colorB), makeImage(8, 4),
      makeState(), makePlan(), GL.LINEAR, 1, -1, 0, 0.5, 0.25, 0, false, 0, 1, out);
    expectArrayClose(out, colorB);
    expect(callsB).toHaveLength(4);
    expect(callsB.every((c) => c.level === 1)).toBe(true);
  });
});