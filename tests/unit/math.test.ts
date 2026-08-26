/**
 * Unit tests for src/util/math (vector/matrix helpers) — written against the
 * actual `src/util/math` module API (flat top-level functions; no namespaces).
 * Fails at runtime ("not implemented") until src/util/math lands; these tests
 * are then the executable spec.
 *
 * API conventions (src/util/math.ts):
 *  - Constructors: `vec2/vec3/vec4(x, y, z, w)` and `mat2/mat3/mat4(s)` return
 *    Float32Array (matrices column-major with diagonal `s`; identity at the
 *    default `s = 1`).
 *  - Vector/matrix ops are flat functions with an optional trailing `out`
 *    Float32Array of the exact expected length; inputs must also be
 *    Float32Array (number[] literals are type errors).
 *  - No namespaced `vec3.`/`mat4.` objects; no perspective/translate helpers —
 *    transforms are composed via `mat4Mul` and applied via `mat4MulVec4`.
 */
import { describe, it, expect } from "vitest";
import * as math from "../../src/util/math";
import { expectArrayClose } from "./helpers";

/** Column-major 4x4 translation matrix (plain array; wrap in Float32Array at the call site). */
function translation(tx: number, ty: number, tz: number): number[] {
  const m = new Array(16).fill(0);
  m[0] = m[5] = m[10] = m[15] = 1;
  m[12] = tx;
  m[13] = ty;
  m[14] = tz;
  return m;
}

/** Column-major 4x4 scale matrix (plain array; wrap in Float32Array at the call site). */
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
    expect(math.dot(math.vec3(1, 2, 3), math.vec3(4, 5, 6))).toBe(32);
  });

  it("cross", () => {
    const out = new Float32Array(3);
    math.cross(math.vec3(1, 0, 0), math.vec3(0, 1, 0), out);
    expectArrayClose(out, [0, 0, 1]);
  });

  it("length", () => {
    expect(math.length(math.vec3(3, 4, 0))).toBeCloseTo(5, 10);
  });

  it("normalize", () => {
    const out = new Float32Array(3);
    math.normalize(math.vec3(3, 4, 0), out);
    expectArrayClose(out, [0.6, 0.8, 0], 1e-6);
  });

  it("add / sub / scale", () => {
    const out = new Float32Array(3);
    math.vecAdd(math.vec3(1, 2, 3), math.vec3(10, 20, 30), out);
    expectArrayClose(out, [11, 22, 33]);
    math.vecSub(math.vec3(1, 2, 3), math.vec3(10, 20, 30), out);
    expectArrayClose(out, [-9, -18, -27]);
    math.vecScale(math.vec3(1, 2, 3), 2, out);
    expectArrayClose(out, [2, 4, 6]);
  });
});

describe("mat4", () => {
  it("identity", () => {
    const out = new Float32Array(16);
    math.mat4Identity(out);
    for (let i = 0; i < 16; i++) {
      expect(out[i], `element ${i}`).toBe(i % 5 === 0 ? 1 : 0);
    }
  });

  it("multiply composes translations", () => {
    const out = new Float32Array(16);
    // T(1,2,3) * T(4,5,6): apply T(4,5,6) first, then T(1,2,3).
    math.mat4Mul(
      new Float32Array(translation(1, 2, 3)),
      new Float32Array(translation(4, 5, 6)),
      out,
    );
    const p = applyMat4(out, [0, 0, 0, 1]);
    expectArrayClose(p, [5, 7, 9, 1]);
  });

  it("invert undoes translate*scale", () => {
    const m = new Float32Array(16);
    math.mat4Mul(new Float32Array(translation(1, 2, 3)), new Float32Array(scale(2, 2, 2)), m);
    const inv = new Float32Array(16);
    math.mat4Inverse(m, inv);
    // M maps (1,1,1) → scale → (2,2,2) → translate → (3,4,5); inverse maps back.
    const p = applyMat4(inv, [3, 4, 5, 1]);
    expectArrayClose(p, [1, 1, 1, 1], 1e-6);
  });

  it("mat4MulVec4 transforms a point", () => {
    // T(1,2,3) * (0,0,0,1) → (1,2,3,1).
    const t = new Float32Array(translation(1, 2, 3));
    const p = math.mat4MulVec4(t, math.vec4(0, 0, 0, 1));
    expectArrayClose(p, [1, 2, 3, 1]);
  });

  it("determinant of a scale matrix is the product of the scales", () => {
    const m = new Float32Array(scale(2, 3, 4));
    expect(math.mat4Determinant(m)).toBeCloseTo(24, 5);
  });

  it("transpose swaps column/row storage", () => {
    const t = new Float32Array(translation(1, 2, 3));
    const out = new Float32Array(16);
    math.mat4Transpose(t, out);
    // Column-major T: translation lives in column 3 (elements 12,13,14);
    // transposed it moves to row 3 (elements 3,7,11).
    expectArrayClose(out, [1, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1]);
  });
});
