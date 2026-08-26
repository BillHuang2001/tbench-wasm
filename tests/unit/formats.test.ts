/**
 * Unit tests for the shared pixel-format registry (src/raster/formats.ts) —
 * written against the REAL API in src/raster/formats.ts (verified at rewrite;
 * do not re-derive from src/CONTEXT.md §3).
 *
 * API under test:
 * - `getFormat(format: GLenum) → PixelFormatInfo | null` (null, not
 *   undefined) with `{ format, components, bytesPerPixel, storage:
 *   StorageKind, isColor, isDepth, isStencil, isFloat, isSigned, isInteger,
 *   isSRGB, normalized, decode(data, byteOffset, out) → void, encode(data,
 *   byteOffset, r, g, b, a) }`. decode fills the caller's `out` Float32Array
 *   (length ≥ 4) — it does NOT return an array. Normalized formats decode to
 *   0..1 (snorm −1..1), float formats raw, integer formats raw; formats
 *   without alpha decode a=1. encode takes floats 0..1 (raw for float/int)
 *   and quantizes per GLES3 rules.
 * - Upload conversion is per-texel: `getTexImageConverter(srcFormat,
 *   srcType, internalFormat) → TexelConverter | null` with `{ srcComponents,
 *   read(src, byteOffset, out), write(dst, byteOffset, r, g, b, a) }`, plus
 *   `getTexelReader` / `getTexelWriter` / `getPackConverter`. There is NO
 *   `convertPixels`: row flipping / UNPACK_ALIGNMENT is gl/'s concern, so the
 *   old flipY test was dropped (per formats.ts JSDoc — converters are
 *   per-texel).
 * - Storage decisions (formats.ts JSDoc): normalized 8-bit → u8/i8, 16-bit →
 *   u16/i16 (565/4444/5551); ALL float formats and depth are stored f32, so
 *   e.g. R16F bpp=4 and DEPTH_COMPONENT16/24 bpp=4 (not 2 as in the old
 *   half-float / 16-bit-depth assumption).
 *
 * Runtime status at rewrite: `getFormat` is implemented but the FORMATS
 * registry starts EMPTY (Phase 2 calls defineFormat() for every entry in
 * ALL_INTERNAL_FORMATS) and the converters are stubs that throw — so the
 * metadata / round-trip / converter tests fail at runtime until that lands
 * (their failures are the executable spec). The numeric helpers
 * (halfToFloat/floatToHalf/sRGBToLinear/linearToSRGB/packDepth24Stencil/
 * unpackDepth24) are implemented and PASS today.
 */
import { describe, it, expect } from "vitest";
import {
  getFormat,
  getTexImageConverter,
  halfToFloat,
  floatToHalf,
  sRGBToLinear,
  linearToSRGB,
  packDepth24Stencil,
  unpackDepth24,
  type PixelFormatInfo,
} from "../../src/raster/formats";
import { GL, expectArrayClose } from "./helpers";

function fmt(internalFormat: number): PixelFormatInfo {
  const f = getFormat(internalFormat);
  expect(f, `getFormat(0x${internalFormat.toString(16)}) returned null`).toBeTruthy();
  return f!;
}

/** Allocates a surface buffer matching the format's storage class. */
function surface(f: PixelFormatInfo): ArrayBufferView {
  const n = f.bytesPerPixel;
  switch (f.storage) {
    case "u8": return new Uint8Array(n);
    case "i8": return new Int8Array(n);
    case "u16": return new Uint16Array(n / 2);
    case "i16": return new Int16Array(n / 2);
    case "u32": return new Uint32Array(n / 4);
    case "i32": return new Int32Array(n / 4);
    case "f32": return new Float32Array(n / 4);
    case "f16": return new Uint16Array(n / 2);
  }
}

/** Decodes one texel at offset 0 into a fresh [r,g,b,a] Float32Array. */
function decodeTexel(f: PixelFormatInfo, data: ArrayBufferView): Float32Array {
  const out = new Float32Array(4);
  f.decode(data, 0, out);
  return out;
}

describe("format metadata", () => {
  it("exposes bytesPerPixel / components per the storage decisions", () => {
    const cases: [number, number, number][] = [
      [GL.RGBA8, 4, 4],
      [GL.RGB8, 3, 3],
      [GL.RGBA4, 2, 4],
      [GL.RGB5_A1, 2, 4],
      [GL.RGB565, 2, 3],
      [GL.LUMINANCE, 1, 1],
      [GL.LUMINANCE_ALPHA, 2, 2],
      [GL.ALPHA, 1, 1],
      [GL.R8, 1, 1],
      [GL.R32I, 4, 1],
      [GL.R32UI, 4, 1],
      // float formats are stored f32 (4 bytes/channel, not 2 for 16F)
      [GL.R16F, 4, 1],
      [GL.RGBA32F, 16, 4],
      // depth is stored f32 (0..1)
      [GL.DEPTH_COMPONENT16, 4, 1],
      [GL.DEPTH_COMPONENT24, 4, 1],
      [GL.DEPTH_COMPONENT32F, 4, 1],
    ];
    for (const [internalFormat, bytesPerPixel, components] of cases) {
      const f = fmt(internalFormat);
      expect(f.bytesPerPixel, `bpp of 0x${internalFormat.toString(16)}`).toBe(bytesPerPixel);
      expect(f.components, `components of 0x${internalFormat.toString(16)}`).toBe(components);
    }
  });

  it("assigns storage classes per the storage decisions", () => {
    const cases: [number, string][] = [
      [GL.RGBA8, "u8"],
      [GL.LUMINANCE, "u8"],
      [GL.RGB565, "u16"],
      [GL.RGBA4, "u16"],
      [GL.RGB5_A1, "u16"],
      [GL.R32I, "i32"],
      [GL.R32UI, "u32"],
      [GL.R16F, "f32"],
      [GL.RGBA32F, "f32"],
      [GL.DEPTH_COMPONENT16, "f32"],
      [GL.DEPTH_COMPONENT24, "f32"],
    ];
    for (const [internalFormat, storage] of cases) {
      expect(fmt(internalFormat).storage, `storage of 0x${internalFormat.toString(16)}`).toBe(storage);
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
      expect(f.isColor).toBe(true);
      expect(f.isSRGB).toBe(internalFormat === GL.SRGB8_ALPHA8);
    }
    for (const internalFormat of [GL.R16F, GL.RGBA16F, GL.RGBA32F, GL.DEPTH_COMPONENT32F]) {
      const f = fmt(internalFormat);
      expect(f.isFloat, `isFloat of 0x${internalFormat.toString(16)}`).toBe(true);
      expect(f.normalized).toBe(false);
      expect(f.isInteger).toBe(false);
    }
    for (const internalFormat of [GL.DEPTH_COMPONENT16, GL.DEPTH_COMPONENT24]) {
      expect(fmt(internalFormat).isDepth).toBe(true);
      expect(fmt(internalFormat).isColor).toBe(false);
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
    const data = surface(f);
    f.encode(data, 0, 0.5, 0.25, 1.0, 1.0);
    expectArrayClose(decodeTexel(f, data), [0.5, 0.25, 1.0, 1.0], 0.005);
  });

  it("RGB565 within 5/6-bit quantization", () => {
    const f = fmt(GL.RGB565);
    const data = surface(f);
    f.encode(data, 0, 0.5, 0.25, 1.0, 1.0);
    expectArrayClose(decodeTexel(f, data), [0.5, 0.25, 1.0, 1.0], 0.02);
  });

  it("RGBA4 within 4-bit quantization", () => {
    const f = fmt(GL.RGBA4);
    const data = surface(f);
    f.encode(data, 0, 0.5, 0.25, 1.0, 0.75);
    expectArrayClose(decodeTexel(f, data), [0.5, 0.25, 1.0, 0.75], 0.04);
  });

  it("RGB5_A1 within 5-bit quantization, alpha is binary", () => {
    const f = fmt(GL.RGB5_A1);
    const data = surface(f);
    f.encode(data, 0, 0.5, 0.25, 1.0, 1.0);
    const rgba = decodeTexel(f, data);
    expectArrayClose(rgba, [0.5, 0.25, 1.0, 1.0], 0.02);
    expect(rgba[3]).toBeCloseTo(1.0, 5);
  });

  it("LUMINANCE keeps R as luma and forces alpha to 1", () => {
    const f = fmt(GL.LUMINANCE);
    const data = surface(f);
    f.encode(data, 0, 0.3, 0.9, 0.1, 0.7);
    expectArrayClose(decodeTexel(f, data), [0.3, 0.3, 0.3, 1.0], 0.005);
  });

  it("LUMINANCE_ALPHA keeps alpha", () => {
    const f = fmt(GL.LUMINANCE_ALPHA);
    const data = surface(f);
    f.encode(data, 0, 0.3, 0.9, 0.1, 0.7);
    expectArrayClose(decodeTexel(f, data), [0.3, 0.3, 0.3, 0.7], 0.005);
  });
});

describe("encode/decode round-trips (depth formats)", () => {
  it("DEPTH_COMPONENT16 (f32 surface storage)", () => {
    const f = fmt(GL.DEPTH_COMPONENT16);
    const data = surface(f); // Float32Array(1) — depth is stored f32
    f.encode(data, 0, 0.5, 0, 0, 1);
    expect(decodeTexel(f, data)[0]).toBeCloseTo(0.5, 4);
  });

  it("DEPTH_COMPONENT24 (f32 surface storage)", () => {
    const f = fmt(GL.DEPTH_COMPONENT24);
    const data = surface(f); // Float32Array(1)
    f.encode(data, 0, 0.5, 0, 0, 1);
    expect(decodeTexel(f, data)[0]).toBeCloseTo(0.5, 6);
  });
});

describe("encode/decode round-trips (float formats)", () => {
  it("R16F: values round-trip via f32 surface storage", () => {
    const f = fmt(GL.R16F);
    const data = surface(f); // Float32Array(1) — no half conversion in the surface
    f.encode(data, 0, 0.5, 0, 0, 1);
    expect(decodeTexel(f, data)[0]).toBeCloseTo(0.5, 5);
    f.encode(data, 0, 0.1, 0, 0, 1);
    expect(decodeTexel(f, data)[0]).toBeCloseTo(0.1, 3);
  });

  it("RGBA32F full-float round-trip", () => {
    const f = fmt(GL.RGBA32F);
    const data = surface(f); // Float32Array(4)
    f.encode(data, 0, 0.1, 0.2, 0.3, 0.4);
    expectArrayClose(decodeTexel(f, data), [0.1, 0.2, 0.3, 0.4], 1e-6);
  });
});

describe("getTexImageConverter (per-texel texImage2D upload conversion)", () => {
  // NOTE: the old `convertPixels` flipY test was dropped — row flipping and
  // UNPACK_ALIGNMENT are gl/'s concern (formats.ts JSDoc); converters are
  // per-texel.

  it("RGBA → LUMINANCE uses the R channel as luma", () => {
    const conv = getTexImageConverter(GL.RGBA, GL.UNSIGNED_BYTE, GL.LUMINANCE);
    expect(conv, "converter for (RGBA, UNSIGNED_BYTE, LUMINANCE)").toBeTruthy();
    const src = new Uint8Array([77, 230, 26, 178]); // (0.3, 0.9, 0.1, 0.7)
    const dst = new Uint8Array(1);
    const tmp = new Float32Array(4);
    conv!.read(src, 0, tmp);
    conv!.write(dst, 0, tmp[0], tmp[1], tmp[2], tmp[3]);
    expect(dst[0]).toBe(77);
  });

  it("RGB → RGBA8 fills alpha with 255", () => {
    const conv = getTexImageConverter(GL.RGB, GL.UNSIGNED_BYTE, GL.RGBA8);
    expect(conv, "converter for (RGB, UNSIGNED_BYTE, RGBA8)").toBeTruthy();
    const src = new Uint8Array([10, 20, 30]);
    const dst = new Uint8Array(4);
    const tmp = new Float32Array(4);
    conv!.read(src, 0, tmp);
    conv!.write(dst, 0, tmp[0], tmp[1], tmp[2], tmp[3]);
    expectArrayClose(dst, [10, 20, 30, 255]);
  });

  it("RGBA → RGB565 quantizes to 5/6 bits", () => {
    const conv = getTexImageConverter(GL.RGBA, GL.UNSIGNED_BYTE, GL.RGB565);
    expect(conv, "converter for (RGBA, UNSIGNED_BYTE, RGB565)").toBeTruthy();
    const src = new Uint8Array([255, 128, 64, 255]); // (1.0, 0.5, 0.25, 1.0)
    const dst = new Uint16Array(1);
    const tmp = new Float32Array(4);
    conv!.read(src, 0, tmp);
    conv!.write(dst, 0, tmp[0], tmp[1], tmp[2], tmp[3]);
    // decode the packed texel through the registry instead of hand bit-math
    expectArrayClose(decodeTexel(fmt(GL.RGB565), dst), [1.0, 0.5, 0.25, 1.0], 0.02);
  });
});

describe("implemented numeric helpers", () => {
  it("halfToFloat/floatToHalf round-trip within half precision", () => {
    for (const v of [0.5, -2.0, 1.0, 100.0, 0.1]) {
      const round = halfToFloat(floatToHalf(v));
      // half has 10 mantissa bits → relative error ≤ 2^-11
      expect(Math.abs(round - v), `round-trip of ${v}`).toBeLessThanOrEqual(Math.abs(v) * 2 ** -11 + 1e-7);
    }
  });

  it("half-float special values", () => {
    expect(floatToHalf(Infinity)).toBe(0x7c00);
    expect(floatToHalf(-Infinity)).toBe(0xfc00);
    expect(floatToHalf(0)).toBe(0);
    expect(floatToHalf(-0)).toBe(0x8000);
    expect(floatToHalf(NaN)).toBe(0x7e00);
    expect(halfToFloat(0x3c00)).toBe(1.0); // +1.0
    expect(halfToFloat(0xbc00)).toBe(-1.0); // −1.0
    expect(halfToFloat(0x7c00)).toBe(Infinity);
    expect(halfToFloat(0x0000)).toBe(0);
  });

  it("sRGBToLinear/linearToSRGB endpoints and round-trip", () => {
    expect(sRGBToLinear(0)).toBe(0);
    expect(sRGBToLinear(1)).toBe(1);
    expect(sRGBToLinear(0.04045)).toBeCloseTo(0.04045 / 12.92, 10);
    expect(linearToSRGB(0.0031308)).toBeCloseTo(0.0031308 * 12.92, 10);
    for (const c of [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]) {
      expect(linearToSRGB(sRGBToLinear(c)), `sRGB round-trip of ${c}`).toBeCloseTo(c, 5);
    }
  });

  it("packDepth24Stencil / unpackDepth24", () => {
    expect(packDepth24Stencil(0, 0)).toBe(0);
    expect(packDepth24Stencil(1, 255) >>> 0).toBe(0xffffffff);
    const packed = packDepth24Stencil(0.5, 7);
    expect((packed >>> 8) & 0xffffff).toBe(0x800000); // round(0.5 * 0xffffff)
    expect(packed & 0xff).toBe(7);
    expect(unpackDepth24(packed)).toBeCloseTo(0.5, 5);
    expect(unpackDepth24(0)).toBe(0);
    expect(unpackDepth24(0xffffffff)).toBe(1);
  });
});
