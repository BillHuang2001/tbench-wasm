/**
 * Unit tests for the shared pixel-format registry (src/raster/formats.ts) —
 * written against the FINAL contract in src/CONTEXT.md §3. Fails (module not
 * found) until src/raster/formats.ts lands; these tests are then the
 * executable spec.
 *
 * Assumed imports from `../../src/raster/formats`:
 * - `getFormat(internalFormat: number) → PixelFormat | undefined` where
 *   PixelFormat = `{ bytesPerPixel, components, decode(data, off) → [r,g,b,a],
 *   encode(data, off, r, g, b, a), isDepth, isStencil, isFloat, isSigned,
 *   isInteger, normalized }` (contract §3). decode returns floats 0..1 for
 *   normalized formats, raw values for float/integer formats, and fills a=1
 *   for formats without alpha. encode takes floats 0..1 and quantizes.
 * - `convertPixels(srcFormat, dstFormat, src, dst, width, height, opts?)` with
 *   `opts: { flipY?: boolean; premultiplyAlpha?: boolean }` — texImage2D-style
 *   source-format conversion (contract §3 "source-format conversion tables").
 * If the actual module exports differ, update ONLY the call sites below.
 */
import { describe, it, expect } from "vitest";
import { getFormat, convertPixels } from "../../src/raster/formats";
import { GL, expectArrayClose } from "./helpers";

function fmt(internalFormat: number) {
  const f = getFormat(internalFormat);
  expect(f, `getFormat(0x${internalFormat.toString(16)}) returned undefined`).toBeTruthy();
  return f!;
}

describe("format metadata", () => {
  it("exposes bytesPerPixel / components for byte-aligned normalized formats", () => {
    const cases: [number, number, number][] = [
      [GL.RGBA8, 4, 4],
      [GL.RGB8, 3, 3],
      [GL.RGBA4, 2, 4],
      [GL.RGB5_A1, 2, 4],
      [GL.RGB565, 2, 3],
      [GL.LUMINANCE, 1, 1],
      [GL.LUMINANCE_ALPHA, 2, 2],
      [GL.ALPHA, 1, 1],
      [GL.DEPTH_COMPONENT16, 2, 1],
      [GL.R16F, 2, 1],
      [GL.RGBA32F, 16, 4],
      [GL.R32I, 4, 1],
      [GL.R32UI, 4, 1],
    ];
    for (const [internalFormat, bytesPerPixel, components] of cases) {
      const f = fmt(internalFormat);
      expect(f.bytesPerPixel, `bpp of 0x${internalFormat.toString(16)}`).toBe(bytesPerPixel);
      expect(f.components, `components of 0x${internalFormat.toString(16)}`).toBe(components);
    }
  });

  it("classifies normalized vs float vs integer vs depth formats", () => {
    const normalized: number[] = [GL.RGBA8, GL.RGB565, GL.LUMINANCE, GL.SRGB8_ALPHA8];
    for (const internalFormat of normalized) {
      const f = fmt(internalFormat);
      expect(f.normalized).toBe(true);
      expect(f.isFloat).toBe(false);
      expect(f.isInteger).toBe(false);
      expect(f.isDepth).toBe(false);
      expect(f.isStencil).toBe(false);
    }
    for (const internalFormat of [GL.R16F, GL.RGBA16F, GL.RGBA32F, GL.DEPTH_COMPONENT32F]) {
      const f = fmt(internalFormat);
      expect(f.isFloat, `isFloat of 0x${internalFormat.toString(16)}`).toBe(true);
      expect(f.normalized).toBe(false);
      expect(f.isInteger).toBe(false);
    }
    for (const internalFormat of [GL.DEPTH_COMPONENT16, GL.DEPTH_COMPONENT24]) {
      expect(fmt(internalFormat).isDepth).toBe(true);
    }
    expect(fmt(GL.DEPTH24_STENCIL8).isDepth).toBe(true);
    expect(fmt(GL.DEPTH24_STENCIL8).isStencil).toBe(true);

    const r32i = fmt(GL.R32I);
    expect(r32i.isInteger).toBe(true);
    expect(r32i.isSigned).toBe(true);
    const r32ui = fmt(GL.R32UI);
    expect(r32ui.isInteger).toBe(true);
    expect(r32ui.isSigned).toBe(false);
  });
});

describe("encode/decode round-trips (normalized formats)", () => {
  it("RGBA8", () => {
    const f = fmt(GL.RGBA8);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.5, 0.25, 1.0, 1.0);
    expectArrayClose(f.decode(data, 0), [0.5, 0.25, 1.0, 1.0], 0.005);
  });

  it("RGB565 within 5/6-bit quantization", () => {
    const f = fmt(GL.RGB565);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.5, 0.25, 1.0, 1.0);
    expectArrayClose(f.decode(data, 0), [0.5, 0.25, 1.0, 1.0], 0.02);
  });

  it("RGBA4 within 4-bit quantization", () => {
    const f = fmt(GL.RGBA4);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.5, 0.25, 1.0, 0.75);
    expectArrayClose(f.decode(data, 0), [0.5, 0.25, 1.0, 0.75], 0.04);
  });

  it("RGB5_A1 within 5-bit quantization, alpha is binary", () => {
    const f = fmt(GL.RGB5_A1);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.5, 0.25, 1.0, 1.0);
    const rgba = f.decode(data, 0);
    expectArrayClose(rgba, [0.5, 0.25, 1.0, 1.0], 0.02);
    expect(rgba[3]).toBeCloseTo(1.0, 5);
  });

  it("LUMINANCE keeps R as luma and forces alpha to 1", () => {
    const f = fmt(GL.LUMINANCE);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.3, 0.9, 0.1, 0.7);
    expectArrayClose(f.decode(data, 0), [0.3, 0.3, 0.3, 1.0], 0.005);
  });

  it("LUMINANCE_ALPHA keeps alpha", () => {
    const f = fmt(GL.LUMINANCE_ALPHA);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.3, 0.9, 0.1, 0.7);
    expectArrayClose(f.decode(data, 0), [0.3, 0.3, 0.3, 0.7], 0.005);
  });
});

describe("encode/decode round-trips (depth formats)", () => {
  it("DEPTH_COMPONENT16", () => {
    const f = fmt(GL.DEPTH_COMPONENT16);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.5, 0, 0, 1);
    expect(f.decode(data, 0)[0]).toBeCloseTo(0.5, 4);
  });

  it("DEPTH_COMPONENT24", () => {
    const f = fmt(GL.DEPTH_COMPONENT24);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.5, 0, 0, 1);
    expect(f.decode(data, 0)[0]).toBeCloseTo(0.5, 6);
  });
});

describe("encode/decode round-trips (float formats)", () => {
  it("R16F half-float: exact values round-trip, 0.1 within half precision", () => {
    const f = fmt(GL.R16F);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.5, 0, 0, 1);
    expect(f.decode(data, 0)[0]).toBeCloseTo(0.5, 5);
    f.encode(data, 0, 0.1, 0, 0, 1);
    expect(f.decode(data, 0)[0]).toBeCloseTo(0.1, 3);
  });

  it("RGBA32F full-float round-trip", () => {
    const f = fmt(GL.RGBA32F);
    const data = new Uint8Array(f.bytesPerPixel);
    f.encode(data, 0, 0.1, 0.2, 0.3, 0.4);
    expectArrayClose(f.decode(data, 0), [0.1, 0.2, 0.3, 0.4], 1e-6);
  });
});

describe("convertPixels (texImage2D source-format conversion)", () => {
  it("RGBA → LUMINANCE uses the R channel as luma", () => {
    const src = new Uint8Array([77, 230, 26, 178]); // (0.3, 0.9, 0.1, 0.7)
    const dst = new Uint8Array(1);
    convertPixels(GL.RGBA, GL.LUMINANCE, src, dst, 1, 1);
    expect(dst[0]).toBe(77);
  });

  it("RGB → RGBA8 fills alpha with 255", () => {
    const src = new Uint8Array([10, 20, 30]);
    const dst = new Uint8Array(4);
    convertPixels(GL.RGB, GL.RGBA8, src, dst, 1, 1);
    expectArrayClose(dst, [10, 20, 30, 255]);
  });

  it("RGBA8 → RGB565 quantizes to 5/6 bits", () => {
    const src = new Uint8Array([255, 128, 64, 255]); // (1.0, 0.5, 0.25, 1.0)
    const dst = new Uint8Array(2);
    convertPixels(GL.RGBA, GL.RGB565, src, dst, 1, 1);
    // RGB565 decode: r5 g6 b5 → approx (1.0, 0.5, 0.25)
    const r = (dst[1] >> 3) / 31;
    const g = (((dst[0] >> 5) | (dst[1] << 3)) & 0x3f) / 63;
    const b = (dst[0] & 0x1f) / 31;
    expect(r).toBeCloseTo(1.0, 2);
    expect(g).toBeCloseTo(0.5, 2);
    expect(b).toBeCloseTo(0.25, 2);
  });

  it("flipY reverses rows", () => {
    // 2 rows x 1 pixel RGBA8: row0=(10,20,30,255), row1=(40,50,60,255)
    const src = new Uint8Array([10, 20, 30, 255, 40, 50, 60, 255]);
    const dst = new Uint8Array(8);
    convertPixels(GL.RGBA, GL.RGBA8, src, dst, 1, 2, { flipY: true });
    expectArrayClose(dst.slice(0, 4), [40, 50, 60, 255]);
    expectArrayClose(dst.slice(4, 8), [10, 20, 30, 255]);
  });
});
