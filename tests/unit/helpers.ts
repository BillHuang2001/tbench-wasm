/**
 * Shared test utilities for tests/unit.
 *
 * GL constant values are spec-fixed (WebGL 1.0/2.0 / GLES2/3.0) and are defined
 * here so unit tests never depend on a src/gl constants module. They are
 * numeric literals per the spec — do not "fix" them to match src/gl if they
 * differ; the spec wins.
 */
import { expect } from "vitest";

export const GL = {
  // Blend / depth / stencil / cull state
  ZERO: 0x0000,
  ONE: 0x0001,
  SRC_ALPHA: 0x0302,
  ONE_MINUS_SRC_ALPHA: 0x0303,
  FUNC_ADD: 0x8006,
  FUNC_SUBTRACT: 0x800a,
  LESS: 0x0201,
  ALWAYS: 0x0207,
  KEEP: 0x1e00,
  BACK: 0x0405,
  CCW: 0x0901,
  TEXTURE0: 0x84c0,

  // Pixel formats (internal & source)
  DEPTH_COMPONENT: 0x1902,
  ALPHA: 0x1906,
  RGB: 0x1907,
  RGBA: 0x1908,
  LUMINANCE: 0x1909,
  LUMINANCE_ALPHA: 0x190a,
  DEPTH_STENCIL: 0x84f9,
  R8: 0x8229,
  R16F: 0x822d,
  R32F: 0x822e,
  R32I: 0x8235,
  R32UI: 0x8236,
  RGBA8: 0x8058,
  RGB8: 0x8051,
  RGBA4: 0x8056,
  RGB5_A1: 0x8057,
  RGB565: 0x8d62,
  RGB10_A2: 0x8059,
  RGBA16F: 0x881a,
  RGBA32F: 0x8814,
  SRGB8_ALPHA8: 0x8c43,
  DEPTH_COMPONENT16: 0x81a5,
  DEPTH_COMPONENT24: 0x81a6,
  DEPTH_COMPONENT32F: 0x8cac,
  DEPTH24_STENCIL8: 0x88f0,

  // Data types (texImage2D / readPixels)
  UNSIGNED_BYTE: 0x1401,
  FLOAT: 0x1406,
} as const;

/**
 * Compares two indexable number sequences element-wise with a tolerance.
 * NaN / ±Infinity must match exactly. Falls through to vitest's expect so
 * failures report the offending index.
 */
export function expectArrayClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  epsilon = 1e-5,
): void {
  expect(actual.length, "array length").toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const e = expected[i];
    if (Number.isFinite(e)) {
      expect(
        Math.abs(a - e),
        `element ${i} (got ${a}, want ${e} ± ${epsilon})`,
      ).toBeLessThanOrEqual(epsilon);
    } else {
      expect(a, `element ${i} (got ${a}, want ${e})`).toBe(e);
    }
  }
}
