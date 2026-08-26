/**
 * Unit tests for the PURE math parts of the rasterizer (src/raster) —
 * edge functions, clip-space interpolation, perspective-correct varying
 * interpolation. Written against the contracts in src/CONTEXT.md §2
 * (homogeneous clipping "interpolating position + varyings linearly in clip
 * space"; perspective-correct varying interpolation). Fails (module not
 * found) until src/raster lands; these tests are then the executable spec.
 *
 * Assumed imports from `../../src/raster/index`:
 * - `edgeFunction(a, b, c) → number` — signed area of edge a→b at point c
 *   (positive = inside for CCW winding): (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x)
 * - `clipInterpolate(a, b, t) → same-length array` — linear interpolation of
 *   clip-space attributes (position + varyings) at parameter t
 * - `perspectiveCorrect(s, t, w, va, vb, vc) → number` —
 *   (s*va/w0 + t*vb/w1 + (1-s-t)*vc/w2) / (s/w0 + t/w1 + (1-s-t)/w2)
 * If the actual module exports differ, update ONLY the call sites below.
 */
import { describe, it, expect } from "vitest";
import { edgeFunction, clipInterpolate, perspectiveCorrect } from "../../src/raster/index";
import { expectArrayClose } from "./helpers";

describe("edgeFunction", () => {
  it("is positive for points inside a CCW triangle edge", () => {
    // Edge (0,0)→(1,0), point (0.5, 0.5): left of the edge (CCW interior).
    expect(edgeFunction([0, 0], [1, 0], [0.5, 0.5])).toBeGreaterThan(0);
  });

  it("is negative for points outside", () => {
    expect(edgeFunction([0, 0], [1, 0], [0.5, -0.5])).toBeLessThan(0);
  });

  it("is ~0 for points on the edge", () => {
    expect(edgeFunction([0, 0], [1, 0], [0.5, 0])).toBeCloseTo(0, 10);
    expect(edgeFunction([0, 0], [1, 0], [0, 0])).toBeCloseTo(0, 10);
  });

  it("classifies points relative to the triangle (0,0),(1,0),(0,1)", () => {
    // Interior point: all three edges positive.
    const e1 = edgeFunction([0, 0], [1, 0], [0.25, 0.25]);
    const e2 = edgeFunction([1, 0], [0, 1], [0.25, 0.25]);
    const e3 = edgeFunction([0, 1], [0, 0], [0.25, 0.25]);
    expect(e1).toBeGreaterThan(0);
    expect(e2).toBeGreaterThan(0);
    expect(e3).toBeGreaterThan(0);
    // Exterior point: at least one edge negative.
    const o1 = edgeFunction([0, 0], [1, 0], [0.75, 0.75]);
    const o2 = edgeFunction([1, 0], [0, 1], [0.75, 0.75]);
    const o3 = edgeFunction([0, 1], [0, 0], [0.75, 0.75]);
    expect(Math.min(o1, o2, o3)).toBeLessThan(0);
  });

  it("has the expected signed area magnitude", () => {
    // Edge (0,0)→(2,0) at (1,1): signed area = 2*1 = 2.
    expect(edgeFunction([0, 0], [2, 0], [1, 1])).toBeCloseTo(2, 10);
    // Swapping endpoints flips the sign.
    expect(edgeFunction([2, 0], [0, 0], [1, 1])).toBeCloseTo(-2, 10);
  });
});

describe("clipInterpolate", () => {
  it("interpolates linearly between clip-space attribute records", () => {
    expectArrayClose(clipInterpolate([0, 0, 0, 1], [1, 1, 1, 1], 0.5), [0.5, 0.5, 0.5, 1]);
  });

  it("handles endpoints (t=0 and t=1) exactly", () => {
    const a = [0, 0, 0, 1];
    const b = [1, 1, 1, 1];
    expectArrayClose(clipInterpolate(a, b, 0), a);
    expectArrayClose(clipInterpolate(a, b, 1), b);
  });

  it("interpolates long varying records too", () => {
    const a = [0, 0, 0, 1, 1, 2, 3, 4];
    const b = [2, 2, 2, 1, 3, 4, 5, 6];
    expectArrayClose(clipInterpolate(a, b, 0.25), [0.5, 0.5, 0.5, 1, 1.5, 2.5, 3.5, 4.5]);
  });
});

describe("perspectiveCorrect", () => {
  it("reduces to barycentric mix when all w are equal", () => {
    const w = [1, 1, 1];
    // s=0.25, t=0.25 → weight of vc = 0.5
    expect(perspectiveCorrect(0.25, 0.25, w, 1, 2, 3)).toBeCloseTo(0.25 * 1 + 0.25 * 2 + 0.5 * 3, 10);
  });

  it("applies perspective correction with unequal w", () => {
    const w = [1, 2, 4];
    const s = 0.25;
    const t = 0.25;
    const va = 1;
    const vb = 2;
    const vc = 3;
    const expected = (s * va / w[0] + t * vb / w[1] + (1 - s - t) * vc / w[2]) /
      (s / w[0] + t / w[1] + (1 - s - t) / w[2]);
    // (0.25 + 0.25 + 0.375) / (0.25 + 0.125 + 0.125) = 0.875 / 0.5 = 1.75
    expect(expected).toBeCloseTo(1.75, 10);
    expect(perspectiveCorrect(s, t, w, va, vb, vc)).toBeCloseTo(expected, 10);
  });

  it("degenerates gracefully at a vertex (s=t=0 → vc)", () => {
    expect(perspectiveCorrect(0, 0, [1, 2, 4], 1, 2, 3)).toBeCloseTo(3, 10);
  });
});
