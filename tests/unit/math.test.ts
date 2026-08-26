/**
 * Unit tests for src/util/math (vector/matrix helpers) — written against the
 * contracts in src/CONTEXT.md (util/ → "math (matrices/vectors for glsl
 * codegen support and raster)"). Fails (module not found) until src/util/math
 * lands; these tests are then the executable spec.
 *
 * Assumed import: `../../src/util/math` with gl-matrix-style out-param
 * helpers: `vec3.{dot,cross,length,normalize,add,sub,scale}` and
 * `mat4.{identity,multiply,invert,perspective,translate}` operating on
 * column-major 16-element matrices. If the actual module exports differ,
 * update ONLY the call sites below.
 */
import { describe, it, expect } from "vitest";
import * as math from "../../src/util/math";
import { expectArrayClose } from "./helpers";

/** Column-major 4x4 translation matrix. */
function translation(tx: number, ty: number, tz: number): number[] {
  const m = new Array(16).fill(0);
  m[0] = m[5] = m[10] = m[15] = 1;
  m[12] = tx;
  m[13] = ty;
  m[14] = tz;
  return m;
}

/** Column-major 4x4 scale matrix. */
function scale(sx: number, sy: number, sz: number): number[] {
  const m = new Array(16).fill(0);
  m[0] = sx;
  m[5] = sy;
  m[10] = sz;
  m[15] = 1;
  return m;
}

/** Applies a column-major 4x4 matrix to a homogeneous point. */
function applyMat4(m: ArrayLike<number>, v: [number, number, number, number]): number[] {
  const [x, y, z, w] = v;
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12] * w,
    m[1] * x + m[5] * y + m[9] * z + m[13] * w,
    m[2] * x + m[6] * y + m[10] * z + m[14] * w,
    m[3] * x + m[7] * y + m[11] * z + m[15] * w,
  ];
}

describe("vec3", () => {
  it("dot", () => {
    expect(math.vec3.dot([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it("cross", () => {
    const out = new Float64Array(3);
    math.vec3.cross(out, [1, 0, 0], [0, 1, 0]);
    expectArrayClose(out, [0, 0, 1]);
  });

  it("length", () => {
    expect(math.vec3.length([3, 4, 0])).toBeCloseTo(5, 10);
  });

  it("normalize", () => {
    const out = new Float64Array(3);
    math.vec3.normalize(out, [3, 4, 0]);
    expectArrayClose(out, [0.6, 0.8, 0], 1e-6);
  });

  it("add / sub / scale", () => {
    const out = new Float64Array(3);
    math.vec3.add(out, [1, 2, 3], [10, 20, 30]);
    expectArrayClose(out, [11, 22, 33]);
    math.vec3.sub(out, [1, 2, 3], [10, 20, 30]);
    expectArrayClose(out, [-9, -18, -27]);
    math.vec3.scale(out, [1, 2, 3], 2);
    expectArrayClose(out, [2, 4, 6]);
  });
});

describe("mat4", () => {
  it("identity", () => {
    const out = new Float64Array(16);
    math.mat4.identity(out);
    for (let i = 0; i < 16; i++) {
      expect(out[i], `element ${i}`).toBe(i % 5 === 0 ? 1 : 0);
    }
  });

  it("multiply composes translations", () => {
    const out = new Float64Array(16);
    // T(1,2,3) * T(4,5,6): apply T(4,5,6) first, then T(1,2,3).
    math.mat4.multiply(out, translation(1, 2, 3), translation(4, 5, 6));
    const p = applyMat4(out, [0, 0, 0, 1]);
    expectArrayClose(p, [5, 7, 9, 1]);
  });

  it("invert undoes translate*scale", () => {
    const m = new Float64Array(16);
    math.mat4.multiply(m, translation(1, 2, 3), scale(2, 2, 2));
    const inv = new Float64Array(16);
    math.mat4.invert(inv, m);
    // M maps (1,1,1) → scale → (2,2,2) → translate → (3,4,5); inverse maps back.
    const p = applyMat4(inv, [3, 4, 5, 1]);
    expectArrayClose(p, [1, 1, 1, 1], 1e-6);
  });

  it("perspective maps near plane to NDC z=-1 and far plane to NDC z=+1", () => {
    const near = 0.1;
    const far = 100;
    const out = new Float64Array(16);
    math.mat4.perspective(out, Math.PI / 3, 1.5, near, far);

    const nearClip = applyMat4(out, [0, 0, -near, 1]);
    expect(nearClip[3]).toBeGreaterThan(0);
    expect(nearClip[2] / nearClip[3]).toBeCloseTo(-1, 5);

    const farClip = applyMat4(out, [0, 0, -far, 1]);
    expect(farClip[3]).toBeGreaterThan(0);
    expect(farClip[2] / farClip[3]).toBeCloseTo(1, 5);
  });

  it("translate moves a point", () => {
    const out = new Float64Array(16);
    math.mat4.translate(out, math.mat4.identity(new Float64Array(16)), [1, 2, 3]);
    const p = applyMat4(out, [0, 0, 0, 1]);
    expectArrayClose(p, [1, 2, 3, 1]);
  });
});
