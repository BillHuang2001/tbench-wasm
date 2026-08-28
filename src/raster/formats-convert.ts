/**
 * formats-convert.ts — texImage2D upload conversion + readPixels pack
 * conversion for the pixel-format registry. Sibling of formats.ts (the
 * registry), which re-exports the public converter API from here.
 *
 * Upload converters: `getTexImageConverter(srcFormat, srcType, internalFormat)`
 * (per-texel read in the DESTINATION domain + write via the dst format's
 * encode), `getTexelReader`/`getTexelWriter` single-side conveniences, and
 * `convertPixels` (row-wise Uint8Array → dst-format conversion for gl/
 * texImage2D, with flipY / premultiplyAlpha).
 *
 * Pack converters: `getPackConverter(internalFormat, packFormat, packType)`
 * (per-texel surface-storage → packed readPixels bytes; color, depth/stencil
 * and integer variants; integer reads are bit-exact — values travel as JS
 * doubles, never through the Float32 scratch). DEPTH_STENCIL pack
 * (UNSIGNED_INT_24_8) takes the Surface object because the stencil plane
 * lives in `stencilData`.
 *
 * No per-texel allocation: module-scratch buffers only (`_r4`, `_scratchBuf`).
 * Row padding (UNPACK_ALIGNMENT / PACK_ALIGNMENT) is gl/'s concern.
 */

import type { GLenum } from './gl-enums';
import {
  RED, RG, RGB, RGBA, LUMINANCE, LUMINANCE_ALPHA, ALPHA, BGRA,
  DEPTH_COMPONENT, DEPTH_STENCIL, STENCIL_INDEX,
  BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, INT, UNSIGNED_INT,
  FLOAT, HALF_FLOAT, HALF_FLOAT_OES,
  UNSIGNED_SHORT_5_6_5, UNSIGNED_SHORT_4_4_4_4, UNSIGNED_SHORT_5_5_5_1,
  UNSIGNED_INT_2_10_10_10_REV, UNSIGNED_INT_10F_11F_11F_REV,
  UNSIGNED_INT_5_9_9_9_REV, UNSIGNED_INT_24_8, FLOAT_32_UNSIGNED_INT_24_8_REV,
  R8, R8_SNORM, R8UI, R8I, R16F, R32F, R16UI, R16I, R32UI, R32I,
  RG8, RG8_SNORM, RG8UI, RG8I, RG16F, RG32F, RG16UI, RG16I, RG32UI, RG32I,
  RGB8, RGB8_SNORM, RGB8UI, RGB8I, RGB565, RGB16F, RGB32F,
  RGB16UI, RGB16I, RGB32UI, RGB32I,
  RGB10_A2, RGB10_A2UI, R11F_G11F_B10F, RGB9_E5, SRGB8, SRGB8_ALPHA8,
  RGBA8, RGBA8_SNORM, RGBA8UI, RGBA8I, RGBA4, RGB5_A1,
  RGBA16F, RGBA32F, RGBA16UI, RGBA16I, RGBA32UI, RGBA32I,
  DEPTH_COMPONENT16, DEPTH_COMPONENT24, DEPTH_COMPONENT32F,
  DEPTH24_STENCIL8, DEPTH32F_STENCIL8,
} from './gl-enums';
import {
  getFormat, clamp, FILL_ZERO, FILL_ONE,
  readU32At, readCompAt, packDepth24Stencil, halfToFloat, floatToHalf,
  unpack11, unpack10, pack11, pack10, pack9E5,
  P_I_U8, P_I_I8, P_I_U16, P_I_I16, P_I_U32, P_I_I32,
} from './formats';
import type { PixelFormatInfo, PerCompParams, StorageKind } from './formats';


/* ================================================================== */
/* texImage2D / texSubImage2D upload conversion                        */
/* ================================================================== */

/** Reads one source texel (srcFormat/srcType) into `out` as [r,g,b,a]. */
export interface TexelReader {
  /** Component count of the source format (1..4). */
  components: number;
  read(src: ArrayBufferView, byteOffset: number, out: Float32Array): void;
}

/** Writes one texel into a destination format (internalFormat). */
export interface TexelWriter {
  write(dst: ArrayBufferView, byteOffset: number, r: number, g: number, b: number, a: number): void;
}

/**
 * Combined per-texel converter for uploads. `read` produces components in the
 * DESTINATION domain: normalized dst → 0..1 floats; float dst → raw floats;
 * integer dst → raw ints (so UNSIGNED_BYTE sources for integer targets yield
 * 0..255, not normalized). `write` component-maps (via the dst format's own
 * encode: LUMINANCE takes r, ALPHA takes a, LA takes (r,a), RGB drops a,
 * RED takes r, RG takes r,g, depth takes r) and converts.
 * Row padding (UNPACK_ALIGNMENT) is gl/'s concern — converters are
 * per-texel, called row by row.
 */
export interface TexelConverter {
  srcComponents: number;
  read(src: ArrayBufferView, byteOffset: number, out: Float32Array): void;
  write(dst: ArrayBufferView, byteOffset: number, r: number, g: number, b: number, a: number): void;
}

/** Source texel layout: stored component count + decode expansion. */
interface SrcLayout {
  n: number;
  exp: number[];
}

function srcLayout(format: GLenum): SrcLayout | null {
  switch (format) {
    case RED: return { n: 1, exp: [0, FILL_ZERO, FILL_ZERO, FILL_ONE] };
    case RG: return { n: 2, exp: [0, 1, FILL_ZERO, FILL_ONE] };
    case RGB: return { n: 3, exp: [0, 1, 2, FILL_ONE] };
    case RGBA: return { n: 4, exp: [0, 1, 2, 3] };
    case LUMINANCE: return { n: 1, exp: [0, 0, 0, FILL_ONE] };
    case LUMINANCE_ALPHA: return { n: 2, exp: [0, 0, 0, 1] };
    case ALPHA: return { n: 1, exp: [FILL_ZERO, FILL_ZERO, FILL_ZERO, 0] };
    case DEPTH_COMPONENT: return { n: 1, exp: [0, 0, 0, FILL_ONE] };
    case DEPTH_STENCIL: return { n: 1, exp: [0, 0, 0, FILL_ONE] }; // depth only; stencil plane is gl/'s concern
    case BGRA: return { n: 4, exp: [2, 1, 0, 3] }; // swizzle BGR→RGB
    default: return null;
  }
}

/** Module-scratch float4 (never escapes this module). */
const _r4 = new Float32Array(4);

type RawReader = (src: ArrayBufferView, off: number, out: Float32Array) => void;

/**
 * Raw component reader for a source (srcType). `norm` selects the
 * normalized domain (value / max) vs raw values (floats and depth sources are
 * always raw/normalized-to-0..1 respectively, regardless of `norm`).
 */
function makeTypeReader(srcType: GLenum, norm: boolean): RawReader | null {
  switch (srcType) {
    case UNSIGNED_BYTE: {
      const d = norm ? 255 : 1;
      return (src, off, out) => {
        const a = src as Uint8Array;
        out[0] = a[off] / d; out[1] = a[off + 1] / d; out[2] = a[off + 2] / d; out[3] = a[off + 3] / d;
      };
    }
    case BYTE: {
      const d = norm ? 127 : 1;
      return (src, off, out) => {
        const a = src as Int8Array;
        out[0] = a[off] / d; out[1] = a[off + 1] / d; out[2] = a[off + 2] / d; out[3] = a[off + 3] / d;
      };
    }
    case UNSIGNED_SHORT: {
      const d = norm ? 65535 : 1;
      return (src, off, out) => {
        const a = src as Uint16Array;
        const i = off >> 1;
        out[0] = a[i] / d; out[1] = a[i + 1] / d; out[2] = a[i + 2] / d; out[3] = a[i + 3] / d;
      };
    }
    case SHORT: {
      const d = norm ? 32767 : 1;
      return (src, off, out) => {
        const a = src as Int16Array;
        const i = off >> 1;
        out[0] = a[i] / d; out[1] = a[i + 1] / d; out[2] = a[i + 2] / d; out[3] = a[i + 3] / d;
      };
    }
    case UNSIGNED_INT: {
      const d = norm ? 4294967295 : 1;
      return (src, off, out) => {
        const a = src as Uint32Array;
        const i = off >> 2;
        out[0] = a[i] / d; out[1] = a[i + 1] / d; out[2] = a[i + 2] / d; out[3] = a[i + 3] / d;
      };
    }
    case INT: {
      const d = norm ? 2147483647 : 1;
      return (src, off, out) => {
        const a = src as Int32Array;
        const i = off >> 2;
        out[0] = a[i] / d; out[1] = a[i + 1] / d; out[2] = a[i + 2] / d; out[3] = a[i + 3] / d;
      };
    }
    case FLOAT:
      return (src, off, out) => {
        const a = src as Float32Array;
        const i = off >> 2;
        out[0] = a[i]; out[1] = a[i + 1]; out[2] = a[i + 2]; out[3] = a[i + 3];
      };
    case HALF_FLOAT:
    case HALF_FLOAT_OES:
      return (src, off, out) => {
        const a = src as Uint16Array;
        const i = off >> 1;
        out[0] = halfToFloat(a[i]); out[1] = halfToFloat(a[i + 1]);
        out[2] = halfToFloat(a[i + 2]); out[3] = halfToFloat(a[i + 3]);
      };
    case UNSIGNED_SHORT_5_6_5:
      return (src, off, out) => {
        const v = (src as Uint16Array)[off >> 1];
        out[0] = ((v >> 11) & 0x1f) / (norm ? 31 : 1);
        out[1] = ((v >> 5) & 0x3f) / (norm ? 63 : 1);
        out[2] = (v & 0x1f) / (norm ? 31 : 1);
        out[3] = 1;
      };
    case UNSIGNED_SHORT_4_4_4_4:
      return (src, off, out) => {
        const v = (src as Uint16Array)[off >> 1];
        out[0] = ((v >> 12) & 0xf) / (norm ? 15 : 1);
        out[1] = ((v >> 8) & 0xf) / (norm ? 15 : 1);
        out[2] = ((v >> 4) & 0xf) / (norm ? 15 : 1);
        out[3] = (v & 0xf) / (norm ? 15 : 1);
      };
    case UNSIGNED_SHORT_5_5_5_1:
      return (src, off, out) => {
        const v = (src as Uint16Array)[off >> 1];
        out[0] = ((v >> 11) & 0x1f) / (norm ? 31 : 1);
        out[1] = ((v >> 6) & 0x1f) / (norm ? 31 : 1);
        out[2] = ((v >> 1) & 0x1f) / (norm ? 31 : 1);
        out[3] = v & 1;
      };
    case UNSIGNED_INT_2_10_10_10_REV:
      return (src, off, out) => {
        const v = (src as Uint32Array)[off >> 2];
        out[0] = (v & 0x3ff) / (norm ? 1023 : 1);
        out[1] = ((v >> 10) & 0x3ff) / (norm ? 1023 : 1);
        out[2] = ((v >> 20) & 0x3ff) / (norm ? 1023 : 1);
        out[3] = ((v >> 30) & 0x3) / (norm ? 3 : 1);
      };
    case UNSIGNED_INT_10F_11F_11F_REV:
      return (src, off, out) => {
        const v = (src as Uint32Array)[off >> 2];
        out[0] = unpack11((v >> 21) & 0x7ff);
        out[1] = unpack11((v >> 10) & 0x7ff);
        out[2] = unpack10(v & 0x3ff);
        out[3] = 1;
      };
    case UNSIGNED_INT_5_9_9_9_REV:
      return (src, off, out) => {
        const v = (src as Uint32Array)[off >> 2];
        const scale = 2 ** (((v >> 27) & 0x1f) - 24);
        out[0] = ((v >> 18) & 0x1ff) * scale;
        out[1] = ((v >> 9) & 0x1ff) * scale;
        out[2] = (v & 0x1ff) * scale;
        out[3] = 1;
      };
    case UNSIGNED_INT_24_8: // depth always normalizes to [0,1]
      return (src, off, out) => {
        const v = (src as Uint32Array)[off >> 2];
        const d = ((v >>> 8) & 0xffffff) / 0xffffff;
        out[0] = d; out[1] = d; out[2] = d; out[3] = 1;
      };
    case FLOAT_32_UNSIGNED_INT_24_8_REV: // depth f32 (little-endian) at off
      return (src, off, out) => {
        const bpe = (src as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
        const d = readF32Bytes(src, off, bpe);
        out[0] = d; out[1] = d; out[2] = d; out[3] = 1;
      };
    default:
      return null;
  }
}

/* Scratch for byte-level reads (FLOAT_32_UNSIGNED_INT_24_8_REV) — no allocation. */
const _scratchBuf = new ArrayBuffer(8);
const _sbU8 = new Uint8Array(_scratchBuf);
const _sbU16 = new Uint16Array(_scratchBuf);
const _sbU32 = new Uint32Array(_scratchBuf);
const _sbDV = new DataView(_scratchBuf);

/** Copies the 4 source bytes at `off` into the scratch and returns the f32 they encode. */
function readF32Bytes(src: ArrayBufferView, off: number, bpe: number): number {
  if (bpe === 1) {
    const a = src as Uint8Array;
    _sbU8[0] = a[off]; _sbU8[1] = a[off + 1]; _sbU8[2] = a[off + 2]; _sbU8[3] = a[off + 3];
  } else if (bpe === 2) {
    const a = src as Uint16Array;
    const i = off >> 1;
    _sbU16[0] = a[i]; _sbU16[1] = a[i + 1];
  } else {
    const a = src as Uint32Array;
    _sbU32[0] = a[off >> 2];
  }
  return _sbDV.getFloat32(0, true);
}

/**
 * Builds the read side of an upload converter: raw type read + per-format
 * expansion, in the chosen domain. Returns null for unknown formats/types.
 */
function buildTexelRead(
  srcFormat: GLenum, srcType: GLenum, norm: boolean,
): { n: number; read: (src: ArrayBufferView, off: number, out: Float32Array) => void } | null {
  const layout = srcLayout(srcFormat);
  const raw = makeTypeReader(srcType, norm);
  if (!layout || !raw) return null;
  const exp = layout.exp;
  return {
    n: layout.n,
    read(src, off, out) {
      raw(src, off, _r4);
      for (let k = 0; k < 4; k++) {
        const e = exp[k];
        out[k] = e < 0 ? (e === FILL_ONE ? 1 : 0) : _r4[e];
      }
    },
  };
}

/** Valid dst internal formats per (srcFormat, srcType) — GLES 3.0 Table 3.2
 *  (unsized dsts included: they are storage-backed; version gating is gl/'s
 *  job — WebGL1 restricts LUMINANCE/LA/ALPHA to UNSIGNED_BYTE and has no
 *  RED/RG sources). */
function texImageDstAllowed(srcFormat: GLenum, srcType: GLenum): ReadonlySet<number> | null {
  switch (srcFormat) {
    case RED:
      switch (srcType) {
        case UNSIGNED_BYTE: return new Set([R8, R8UI]);
        case BYTE: return new Set([R8_SNORM, R8I]);
        case UNSIGNED_SHORT: return new Set([R16UI]);
        case SHORT: return new Set([R16I]);
        case UNSIGNED_INT: return new Set([R32UI]);
        case INT: return new Set([R32I]);
        case HALF_FLOAT: return new Set([R16F]);
        case FLOAT: return new Set([R32F]);
        default: return null;
      }
    case RG:
      switch (srcType) {
        case UNSIGNED_BYTE: return new Set([RG8, RG8UI]);
        case BYTE: return new Set([RG8_SNORM, RG8I]);
        case UNSIGNED_SHORT: return new Set([RG16UI]);
        case SHORT: return new Set([RG16I]);
        case UNSIGNED_INT: return new Set([RG32UI]);
        case INT: return new Set([RG32I]);
        case HALF_FLOAT: return new Set([RG16F]);
        case FLOAT: return new Set([RG32F]);
        default: return null;
      }
    case RGB:
      switch (srcType) {
        case UNSIGNED_BYTE: return new Set([RGB8, RGB8UI, SRGB8, RGB, RGBA8, RGBA4, RGB5_A1, RGB565, SRGB8_ALPHA8, RGBA]);
        case BYTE: return new Set([RGB8_SNORM, RGB8I]);
        case UNSIGNED_SHORT_5_6_5: return new Set([RGB565, RGB]);
        case UNSIGNED_INT_2_10_10_10_REV: return new Set([RGB10_A2, RGB10_A2UI]);
        case UNSIGNED_SHORT: return new Set([RGB16UI]);
        case SHORT: return new Set([RGB16I]);
        case UNSIGNED_INT: return new Set([RGB32UI]);
        case INT: return new Set([RGB32I]);
        case HALF_FLOAT: return new Set([RGB16F]);
        case FLOAT: return new Set([RGB32F]);
        case UNSIGNED_INT_10F_11F_11F_REV: return new Set([R11F_G11F_B10F]);
        case UNSIGNED_INT_5_9_9_9_REV: return new Set([RGB9_E5]);
        default: return null;
      }
    case RGBA:
      switch (srcType) {
        case UNSIGNED_BYTE: return new Set([RGBA8, RGBA8UI, SRGB8_ALPHA8, RGBA, LUMINANCE, LUMINANCE_ALPHA, ALPHA, RGB565, RGB, RGB5_A1, RGBA4]);
        case BYTE: return new Set([RGBA8_SNORM, RGBA8I]);
        case UNSIGNED_SHORT_4_4_4_4: return new Set([RGBA4, RGBA]);
        case UNSIGNED_SHORT_5_5_5_1: return new Set([RGB5_A1, RGBA]);
        case UNSIGNED_INT_2_10_10_10_REV: return new Set([RGB10_A2, RGB10_A2UI]);
        case UNSIGNED_SHORT: return new Set([RGBA16UI]);
        case SHORT: return new Set([RGBA16I]);
        case UNSIGNED_INT: return new Set([RGBA32UI]);
        case INT: return new Set([RGBA32I]);
        case HALF_FLOAT: return new Set([RGBA16F]);
        case FLOAT: return new Set([RGBA32F]);
        default: return null;
      }
    case LUMINANCE:
      return srcType === UNSIGNED_BYTE ? new Set([LUMINANCE]) : null;
    case LUMINANCE_ALPHA:
      return srcType === UNSIGNED_BYTE ? new Set([LUMINANCE_ALPHA]) : null;
    case ALPHA:
      return srcType === UNSIGNED_BYTE ? new Set([ALPHA]) : null;
    case DEPTH_COMPONENT:
      switch (srcType) {
        case UNSIGNED_SHORT: return new Set([DEPTH_COMPONENT16, DEPTH_COMPONENT]);
        case UNSIGNED_INT: return new Set([DEPTH_COMPONENT24, DEPTH_COMPONENT]);
        case FLOAT: return new Set([DEPTH_COMPONENT32F, DEPTH_COMPONENT]);
        default: return null;
      }
    case DEPTH_STENCIL:
      switch (srcType) {
        case UNSIGNED_INT_24_8: return new Set([DEPTH24_STENCIL8, DEPTH_STENCIL]);
        case FLOAT_32_UNSIGNED_INT_24_8_REV: return new Set([DEPTH32F_STENCIL8, DEPTH_STENCIL]);
        default: return null;
      }
    case BGRA:
      return srcType === UNSIGNED_BYTE ? new Set([RGBA8, SRGB8_ALPHA8, RGBA]) : null;
    default:
      return null;
  }
}

/** Builds a converter, or null for an invalid (srcFormat, srcType, internalFormat) combo. */
export function getTexImageConverter(
  srcFormat: GLenum, srcType: GLenum, internalFormat: GLenum,
): TexelConverter | null {
  const dst = getFormat(internalFormat);
  const allowed = texImageDstAllowed(srcFormat, srcType);
  if (!dst || !allowed || !allowed.has(internalFormat)) return null;
  // Domain: normalized (incl. sRGB) dst → normalized reads; float/integer dst → raw.
  const norm = !dst.isFloat && !dst.isInteger;
  const r = buildTexelRead(srcFormat, srcType, norm);
  if (!r) return null;
  return {
    srcComponents: r.n,
    read: r.read,
    write(dstBuf, byteOffset, rr, gg, bb, aa) {
      dst.encode(dstBuf, byteOffset, rr, gg, bb, aa);
    },
  };
}

/** Convenience for code that reads raw source texels only (source's own natural values). */
export function getTexelReader(srcFormat: GLenum, srcType: GLenum): TexelReader | null {
  if (!texImageDstAllowed(srcFormat, srcType)) return null;
  const r = buildTexelRead(srcFormat, srcType, false);
  if (!r) return null;
  return { components: r.n, read: r.read };
}

/** Convenience for code that writes destination texels only. */
export function getTexelWriter(internalFormat: GLenum): TexelWriter | null {
  const dst = getFormat(internalFormat);
  if (!dst) return null;
  return {
    write(dstBuf, byteOffset, r, g, b, a) {
      dst.encode(dstBuf, byteOffset, r, g, b, a);
    },
  };
}

/**
 * texImage2D-style row-wise source conversion: `src` is a Uint8Array of
 * tightly packed `srcFormat` texels (srcType = UNSIGNED_BYTE), row 0 first.
 * Each source texel is read as 0..255 values with standard component
 * expansion (ALPHA→(0,0,0,a), LUMINANCE→(l,l,l,1), LA→(l,l,l,a)), optionally
 * premultiplied (rgb *= a/255) and flipped (src row 0 → dst row height−1),
 * then written via the dst format's encode (dst may be unsized, e.g.
 * LUMINANCE). Width/height ≥ 1.
 */
export function convertPixels(
  srcFormat: GLenum,
  dstFormat: GLenum,
  src: Uint8Array,
  dst: ArrayBufferView,
  width: number,
  height: number,
  opts?: { flipY?: boolean; premultiplyAlpha?: boolean },
): void {
  const layout = srcLayout(srcFormat);
  const dstInfo = getFormat(dstFormat);
  if (!layout || !dstInfo) {
    throw new Error(
      `convertPixels: unsupported source format 0x${srcFormat.toString(16)}` +
      ` or destination format 0x${dstFormat.toString(16)}`,
    );
  }
  // Domain matches the destination (same rule as getTexImageConverter):
  // normalized dst → 0..1 reads (byte-exact round-trip for u8 dsts);
  // float/integer dst → raw 0..255 reads.
  const norm = !dstInfo.isFloat && !dstInfo.isInteger;
  const reader = buildTexelRead(srcFormat, UNSIGNED_BYTE, norm)!;
  const flipY = opts?.flipY ?? false;
  const premul = opts?.premultiplyAlpha ?? false;
  const srcBpp = layout.n;
  const dstBpp = dstInfo.bytesPerPixel;
  const out = new Float32Array(4);
  for (let y = 0; y < height; y++) {
    const srcRow = y * width * srcBpp;
    const dstRow = (flipY ? height - 1 - y : y) * width * dstBpp;
    for (let x = 0; x < width; x++) {
      reader.read(src, srcRow + x * srcBpp, out);
      if (premul) {
        const f = norm ? out[3] : out[3] / 255;
        out[0] *= f; out[1] *= f; out[2] *= f;
      }
      dstInfo.encode(dst, dstRow + x * dstBpp, out[0], out[1], out[2], out[3]);
    }
  }
}

/* ================================================================== */
/* readPixels pack conversion                                          */
/* ================================================================== */

/**
 * Converts one surface texel (internalFormat storage) into packed
 * (packFormat, packType) bytes for readPixels. Handles component mapping
 * (RGBA→RGB, luminance, depth (d,d,d,1), DEPTH_STENCIL packing: depth in the
 * high 24 bits + stencil in the low 8), normalization (float→UNSIGNED_BYTE
 * etc.) and integer variants. Row padding (PACK_ALIGNMENT) is gl/'s concern.
 * NOTE for DEPTH_STENCIL pack (UNSIGNED_INT_24_8): `src` must be the Surface
 * object (depth plane in `data`, stencil plane in `stencilData`), because the
 * stencil plane is stored separately.
 */
export interface PackConverter {
  convert(
    src: ArrayBufferView, srcByteOffset: number,
    dst: ArrayBufferView, dstByteOffset: number,
  ): void;
}

/* readPixels integer pack formats (spec-fixed; only interpreted here, so kept
 * local rather than added to gl-enums.ts). */
const RED_INTEGER = 0x8d94;
const RG_INTEGER = 0x8228;
const RGB_INTEGER = 0x8d98;
const RGBA_INTEGER = 0x8d99;

const INTEGER_PACK_FORMATS = new Set([RED_INTEGER, RG_INTEGER, RGB_INTEGER, RGBA_INTEGER]);

function integerPackComponents(packFormat: GLenum): number {
  return packFormat === RED_INTEGER ? 1
    : packFormat === RG_INTEGER ? 2
    : packFormat === RGB_INTEGER ? 3 : 4;
}

function isIntegerPackType(packType: GLenum): boolean {
  return packType === BYTE || packType === UNSIGNED_BYTE || packType === SHORT ||
    packType === UNSIGNED_SHORT || packType === INT || packType === UNSIGNED_INT;
}

type PackConvertFn = (
  src: ArrayBufferView, srcByteOffset: number,
  dst: ArrayBufferView, dstByteOffset: number,
) => void;

/**
 * Per-component read/write params for an integer storage class. Resolved
 * lazily (not a top-level table) because formats-convert ↔ formats is a
 * circular import: formats.ts bindings are only safe to touch at call time.
 */
function intParams(storage: StorageKind): PerCompParams {
  switch (storage) {
    case 'u8': return P_I_U8;
    case 'i8': return P_I_I8;
    case 'u16': return P_I_U16;
    case 'i16': return P_I_I16;
    case 'u32': return P_I_U32;
    default: return P_I_I32; // i32 (f32/f16 are never integer formats)
  }
}

/**
 * Integer attachment → integer pack (raw, bit-exact: values travel as JS
 * doubles — all 32-bit ints are exactly representable — and never pass
 * through the Float32 scratch).
 */
function buildIntegerPack(info: PixelFormatInfo, packFormat: GLenum, packType: GLenum): PackConvertFn | null {
  const n = integerPackComponents(packFormat);
  const p = intParams(info.storage);
  // Packed one-texel formats (RGB10_A2UI) read via their u32 layout; the rest
  // are per-component at stride p.bpe.
  const packed = info.bytesPerPixel !== p.bpe * info.components;
  const readK = packed
    ? (src: ArrayBufferView, off: number, k: number) => {
        const v = readU32At(src, off);
        return k === 3 ? v & 0x3 : (v >>> (22 - k * 10)) & 0x3ff;
      }
    : (src: ArrayBufferView, off: number, k: number) => readCompAt(src, off + k * p.bpe, p);
  switch (packType) {
    case BYTE:
      return (src, so, dst, do_) => {
        const d = dst as Int8Array;
        for (let k = 0; k < n; k++) d[do_ + k] = readK(src, so, k);
      };
    case UNSIGNED_BYTE:
      return (src, so, dst, do_) => {
        const d = dst as Uint8Array;
        for (let k = 0; k < n; k++) d[do_ + k] = readK(src, so, k);
      };
    case SHORT:
      return (src, so, dst, do_) => {
        const d = dst as Int16Array;
        const i = do_ >> 1;
        for (let k = 0; k < n; k++) d[i + k] = readK(src, so, k);
      };
    case UNSIGNED_SHORT:
      return (src, so, dst, do_) => {
        const d = dst as Uint16Array;
        const i = do_ >> 1;
        for (let k = 0; k < n; k++) d[i + k] = readK(src, so, k);
      };
    case INT:
      return (src, so, dst, do_) => {
        const d = dst as Int32Array;
        const i = do_ >> 2;
        for (let k = 0; k < n; k++) d[i + k] = readK(src, so, k);
      };
    case UNSIGNED_INT:
      return (src, so, dst, do_) => {
        const d = dst as Uint32Array;
        const i = do_ >> 2;
        for (let k = 0; k < n; k++) d[i + k] = readK(src, so, k);
      };
    default:
      return null;
  }
}

/**
 * Normalized/float color attachment → color pack. `snorm` remaps snorm
 * surfaces to 0..1 for unorm pack outputs. `uq(bits)` quantizes a 0..1
 * component to `bits` bits (works for unorm and float surfaces alike).
 */
function buildColorPack(info: PixelFormatInfo, packFormat: GLenum, packType: GLenum): PackConvertFn | null {
  const decode = info.decode;
  const snorm = info.isSigned;
  const uq = (bits: number) => (snorm
    ? (c: number) => Math.round(clamp((c + 1) / 2, 0, 1) * ((1 << bits) - 1))
    : (c: number) => Math.round(clamp(c, 0, 1) * ((1 << bits) - 1)));
  switch (packFormat) {
    case RGBA:
      switch (packType) {
        case UNSIGNED_BYTE: {
          const q = uq(8);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Uint8Array;
            d[do_] = q(_r4[0]); d[do_ + 1] = q(_r4[1]); d[do_ + 2] = q(_r4[2]); d[do_ + 3] = q(_r4[3]);
          };
        }
        case UNSIGNED_SHORT_4_4_4_4: {
          const q = uq(4);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            (dst as Uint16Array)[do_ >> 1] =
              (q(_r4[0]) << 12) | (q(_r4[1]) << 8) | (q(_r4[2]) << 4) | q(_r4[3]);
          };
        }
        case UNSIGNED_SHORT_5_5_5_1: {
          const q = uq(5);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            (dst as Uint16Array)[do_ >> 1] =
              (q(_r4[0]) << 11) | (q(_r4[1]) << 6) | (q(_r4[2]) << 1) | (_r4[3] >= 0.5 ? 1 : 0);
          };
        }
        case HALF_FLOAT:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Uint16Array;
            const i = do_ >> 1;
            d[i] = floatToHalf(_r4[0]); d[i + 1] = floatToHalf(_r4[1]);
            d[i + 2] = floatToHalf(_r4[2]); d[i + 3] = floatToHalf(_r4[3]);
          };
        case FLOAT:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Float32Array;
            const i = do_ >> 2;
            d[i] = _r4[0]; d[i + 1] = _r4[1]; d[i + 2] = _r4[2]; d[i + 3] = _r4[3];
          };
        case UNSIGNED_INT_10F_11F_11F_REV:
          // RGB packed as 11/11/10-bit floats (R top); alpha dropped.
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            (dst as Uint32Array)[do_ >> 2] =
              (pack11(_r4[0]) << 21) | (pack11(_r4[1]) << 10) | pack10(_r4[2]);
          };
        case UNSIGNED_INT_5_9_9_9_REV:
          // RGB packed as shared-exponent 9/9/9/5 (R top); alpha dropped.
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            (dst as Uint32Array)[do_ >> 2] = pack9E5(_r4[0], _r4[1], _r4[2]);
          };
        default:
          return null;
      }
    case RGB:
      switch (packType) {
        case UNSIGNED_BYTE: {
          const q = uq(8);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Uint8Array;
            d[do_] = q(_r4[0]); d[do_ + 1] = q(_r4[1]); d[do_ + 2] = q(_r4[2]);
          };
        }
        case UNSIGNED_SHORT_5_6_5: {
          const q5 = uq(5);
          const q6 = uq(6);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            (dst as Uint16Array)[do_ >> 1] =
              (q5(_r4[0]) << 11) | (q6(_r4[1]) << 5) | q5(_r4[2]);
          };
        }
        case HALF_FLOAT:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Uint16Array;
            const i = do_ >> 1;
            d[i] = floatToHalf(_r4[0]); d[i + 1] = floatToHalf(_r4[1]); d[i + 2] = floatToHalf(_r4[2]);
          };
        case FLOAT:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Float32Array;
            const i = do_ >> 2;
            d[i] = _r4[0]; d[i + 1] = _r4[1]; d[i + 2] = _r4[2];
          };
        case UNSIGNED_INT_10F_11F_11F_REV:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            (dst as Uint32Array)[do_ >> 2] =
              (pack11(_r4[0]) << 21) | (pack11(_r4[1]) << 10) | pack10(_r4[2]);
          };
        case UNSIGNED_INT_5_9_9_9_REV:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            (dst as Uint32Array)[do_ >> 2] = pack9E5(_r4[0], _r4[1], _r4[2]);
          };
        default:
          return null;
      }
    case RED:
    case RG: {
      const n = packFormat === RED ? 1 : 2;
      switch (packType) {
        case UNSIGNED_BYTE: {
          const q = uq(8);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Uint8Array;
            for (let k = 0; k < n; k++) d[do_ + k] = q(_r4[k]);
          };
        }
        case HALF_FLOAT:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Uint16Array;
            const i = do_ >> 1;
            for (let k = 0; k < n; k++) d[i + k] = floatToHalf(_r4[k]);
          };
        case FLOAT:
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Float32Array;
            const i = do_ >> 2;
            for (let k = 0; k < n; k++) d[i + k] = _r4[k];
          };
        case UNSIGNED_INT: {
          const q = snorm
            ? (c: number) => Math.round(clamp((c + 1) / 2, 0, 1) * 4294967295)
            : info.isFloat
              ? (c: number) => Math.round(clamp(c, 0, 4294967295))
              : (c: number) => Math.round(clamp(c, 0, 1) * 4294967295);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Uint32Array;
            const i = do_ >> 2;
            for (let k = 0; k < n; k++) d[i + k] = q(_r4[k]);
          };
        }
        case INT: {
          const q = snorm
            ? (c: number) => Math.round(clamp(c, -1, 1) * 2147483647)
            : info.isFloat
              ? (c: number) => Math.round(clamp(c, -2147483648, 2147483647))
              : (c: number) => Math.round(clamp(c, 0, 1) * 2147483647);
          return (src, so, dst, do_) => {
            decode(src, so, _r4);
            const d = dst as Int32Array;
            const i = do_ >> 2;
            for (let k = 0; k < n; k++) d[i + k] = q(_r4[k]);
          };
        }
        default:
          return null;
      }
    }
    case LUMINANCE:
    case LUMINANCE_ALPHA:
    case ALPHA: {
      if (packType !== UNSIGNED_BYTE) return null;
      const n = packFormat === LUMINANCE ? 1 : packFormat === LUMINANCE_ALPHA ? 2 : 1;
      const take = packFormat === ALPHA ? 3 : 0; // LUMINANCE takes r; ALPHA takes a
      const q = uq(8);
      return (src, so, dst, do_) => {
        decode(src, so, _r4);
        const d = dst as Uint8Array;
        d[do_] = q(_r4[take]);
        if (n === 2) d[do_ + 1] = q(_r4[3]);
      };
    }
    default:
      return null;
  }
}

function buildDepthPack(info: PixelFormatInfo, packFormat: GLenum, packType: GLenum): PackConvertFn | null {
  if (info.isDepth && info.isStencil) {
    // DEPTH_STENCIL surfaces: src must be the Surface (stencilData plane).
    if (packFormat === DEPTH_STENCIL && packType === UNSIGNED_INT_24_8) {
      return (src, so, dst, do_) => {
        const surf = src as unknown as { data: ArrayBufferView; stencilData?: Uint8Array };
        const stencil = surf.stencilData ? surf.stencilData[so >> 2] : 0;
        const v = packDepth24Stencil((surf.data as Float32Array)[so >> 2], stencil);
        const d = dst as Uint8Array;
        d[do_] = v & 0xff;
        d[do_ + 1] = (v >>> 8) & 0xff;
        d[do_ + 2] = (v >>> 16) & 0xff;
        d[do_ + 3] = (v >>> 24) & 0xff;
      };
    }
    return null;
  }
  if (info.isDepth) {
    switch (packFormat) {
      case DEPTH_COMPONENT:
        switch (packType) {
          case UNSIGNED_SHORT:
            return (src, so, dst, do_) => {
              (dst as Uint16Array)[do_ >> 1] =
                Math.round(clamp((src as Float32Array)[so >> 2], 0, 1) * 0xffff);
            };
          case UNSIGNED_INT:
            return (src, so, dst, do_) => {
              (dst as Uint32Array)[do_ >> 2] =
                Math.round(clamp((src as Float32Array)[so >> 2], 0, 1) * 0xffffff);
            };
          case FLOAT:
            return (src, so, dst, do_) => {
              (dst as Float32Array)[do_ >> 2] = (src as Float32Array)[so >> 2];
            };
          default:
            return null;
        }
      default:
        return null;
    }
  }
  if (info.isStencil) {
    if (packFormat === STENCIL_INDEX && packType === UNSIGNED_BYTE) {
      return (src, so, dst, do_) => {
        (dst as Uint8Array)[do_] = (src as Uint8Array)[so];
      };
    }
    return null;
  }
  return null;
}

export function getPackConverter(
  internalFormat: GLenum, packFormat: GLenum, packType: GLenum,
): PackConverter | null {
  const info = getFormat(internalFormat);
  if (!info) return null;
  let convert: PackConvertFn | null = null;
  if (info.isDepth || info.isStencil) {
    convert = buildDepthPack(info, packFormat, packType);
  } else if (info.isInteger) {
    // Integer attachments only pack to integer pack formats/types (raw ints).
    if (INTEGER_PACK_FORMATS.has(packFormat) && isIntegerPackType(packType)) {
      convert = buildIntegerPack(info, packFormat, packType);
    }
  } else if (!INTEGER_PACK_FORMATS.has(packFormat)) {
    // Non-integer pack formats always take the normalized color path;
    // buildColorPack's per-format switches return null for unsupported
    // pack types (e.g. RGBA + BYTE), so no invalid converter is produced.
    convert = buildColorPack(info, packFormat, packType);
  }
  return convert ? { convert } : null;
}
