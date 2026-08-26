/**
 * formats.ts — THE single shared pixel-format registry (contract §3).
 *
 * Every WebGL 1.0/2.0 internal format (sized + unsized, color/depth/stencil,
 * normalized/integer/float/sRGB) has one `PixelFormatInfo` entry here,
 * registered via defineFormat() into the FORMATS table. gl/ uses this module
 * to allocate surfaces, validate renderbuffer/texture formats, convert
 * texImage2D uploads, and pack readPixels; raster uses it to know how to
 * write surfaces; present/ uses it to present the drawing buffer.
 *
 * Storage representation decisions (MANDATORY — see Surface in types.ts):
 *  - Normalized 8-bit: Uint8Array / Int8Array (signed); 16-bit: Uint16Array
 *    / Int16Array (565/4444/5551 use Uint16Array).
 *  - All float formats (R16F..RGBA32F, R11F_G11F_B10F, RGB9_E5) are stored
 *    as Float32Array — native float math in the hot sampling path; the packed
 *    GL forms exist only at the upload/readPixels boundaries.
 *  - Depth: Float32Array (0..1) for DEPTH_COMPONENT16/24/32F and the depth
 *    plane of DEPTH*_STENCIL* (stencil plane = Uint8Array, split).
 *  - `decode`/`encode` work on the SURFACE representation and never allocate
 *    (results written into the caller's `out`). They are correctness paths
 *    (clear, readPixels, copy, blit); the texture sampler reads raw typed
 *    arrays directly with format-class fast paths.
 */

import type { GLenum } from './gl-enums';
import {
  RGBA8, RGB8, RGBA, RGB, RGBA4, RGB5_A1, RGB565,
  LUMINANCE, LUMINANCE_ALPHA, ALPHA, RED, RG,
  R8, R8_SNORM, R16F, R32F, R8UI, R8I, R16UI, R16I, R32UI, R32I,
  RG8, RG8_SNORM, RG16F, RG32F, RG8UI, RG8I, RG16UI, RG16I, RG32UI, RG32I,
  RGB8_SNORM, RGB16F, RGB32F, RGB8UI, RGB8I, RGB16UI, RGB16I, RGB32UI, RGB32I,
  RGBA8_SNORM, RGBA16F, RGBA32F, RGBA8UI, RGBA8I, RGBA16UI, RGBA16I,
  RGBA32UI, RGBA32I,
  RGB10_A2, RGB10_A2UI, R11F_G11F_B10F, RGB9_E5, SRGB8, SRGB8_ALPHA8,
  DEPTH_COMPONENT16, DEPTH_COMPONENT24, DEPTH_COMPONENT32F,
  DEPTH24_STENCIL8, DEPTH32F_STENCIL8, STENCIL_INDEX8,
  DEPTH_COMPONENT, DEPTH_STENCIL,
} from './gl-enums';

/** How a format's texels are stored in Surface.data. */
export type StorageKind =
  | 'u8' | 'i8'          // Uint8Array / Int8Array
  | 'u16' | 'i16'        // Uint16Array / Int16Array (incl. 565/4444/5551)
  | 'u32' | 'i32'        // Uint32Array / Int32Array (incl. RGB10_A2, RGB10_A2UI)
  | 'f32'                // Float32Array (all float formats incl. 16F, R11F, RGB9_E5)
  | 'f16';               // Uint16Array of half-float bits (reserved; not used today)

/** One registry entry — one internal format. */
export interface PixelFormatInfo {
  /** Internal format GLenum (registry key). */
  format: GLenum;
  /** 1..4 components in the RGBA expansion. */
  components: number;
  /** Bytes per texel in the surface representation (data.byteLength / (w*h*d)). */
  bytesPerPixel: number;
  /** Surface storage class (see StorageKind). */
  storage: StorageKind;

  isColor: boolean;
  isDepth: boolean;
  isStencil: boolean;
  /** Floating-point channels (float storage or shared-exponent). */
  isFloat: boolean;
  /** Signed normalized or signed integer. */
  isSigned: boolean;
  /** Non-normalized integer channels (sampled raw, not filtered). */
  isInteger: boolean;
  isSRGB: boolean;
  /** Normalized fixed-point (unorm or snorm). */
  normalized: boolean;

  /**
   * Decode one texel at byte offset into `out` as [r,g,b,a] floats:
   * normalized formats → 0..1 (snorm −1..1); float → raw; integer → raw int
   * values (exact up to 2^24). Depth texels decode as (d,d,d,1).
   * `out` must have length ≥ 4.
   */
  decode(data: ArrayBufferView, byteOffset: number, out: Float32Array): void;
  /**
   * Encode float [r,g,b,a] (0..1 for normalized, raw for float/integer) into
   * the texel at byte offset. Conversion rules per GLES 3.0: clamp to range,
   * round to nearest.
   */
  encode(data: ArrayBufferView, byteOffset: number, r: number, g: number, b: number, a: number): void;
}

/** Registry table (internal). Populated by defineFormat(). */
const FORMATS = new Map<number, PixelFormatInfo>();

/** Registers (or replaces) a format entry. Called at module init and by tests. */
export function defineFormat(entry: PixelFormatInfo): void {
  FORMATS.set(entry.format, entry);
}

/** Looks up a format descriptor; null for unknown/unsupported formats. */
export function getFormat(format: GLenum): PixelFormatInfo | null {
  return FORMATS.get(format) ?? null;
}

export function isColorFormat(format: GLenum): boolean {
  return FORMATS.get(format)?.isColor ?? false;
}
export function isDepthFormat(format: GLenum): boolean {
  return FORMATS.get(format)?.isDepth ?? false;
}
export function isStencilFormat(format: GLenum): boolean {
  return FORMATS.get(format)?.isStencil ?? false;
}
export function isDepthStencilFormat(format: GLenum): boolean {
  const f = FORMATS.get(format);
  return f != null && f.isDepth && f.isStencil;
}

/**
 * Renderbuffer format validation (renderbufferStorage). WebGL1: RGBA4, RGB565,
 * RGB5_A1, DEPTH_COMPONENT16, STENCIL_INDEX8, DEPTH_STENCIL. WebGL2: the ES3
 * color-renderable set (R8, RG8, RGBA8, RGB10_A2, RGBA4, RGB5_A1, RGB565,
 * R16F/RG16F/RGBA16F, R32F/RG32F/RGBA32F via EXT_color_buffer_float,
 * SRGB8_ALPHA8, integer R8I..RGBA32I/UI) + depth/stencil formats.
 */
export function isValidRenderbufferFormat(format: GLenum, version: 1 | 2): boolean {
  throw new Error('not implemented: isValidRenderbufferFormat');
}

/**
 * Texture internal format validation for texImage2D/texStorage. WebGL1
 * accepts the unsized formats (RGBA/RGB/LUMINANCE/LUMINANCE_ALPHA/ALPHA +
 * depth); WebGL2 requires sized formats (and rejects unsized).
 */
export function isValidTextureInternalFormat(format: GLenum, version: 1 | 2): boolean {
  throw new Error('not implemented: isValidTextureInternalFormat');
}

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
 * 0..255, not normalized). `write` component-maps (LUMINANCE→(l,l,l,1),
 * ALPHA→(0,0,0,a), BGRA swizzle, RGB↔RGBA padding) and converts.
 * Row padding (UNPACK_ALIGNMENT) is gl/'s concern — converters are
 * per-texel, called row by row.
 */
export interface TexelConverter {
  srcComponents: number;
  read(src: ArrayBufferView, byteOffset: number, out: Float32Array): void;
  write(dst: ArrayBufferView, byteOffset: number, r: number, g: number, b: number, a: number): void;
}

/** Builds a converter, or null for an invalid (srcFormat, srcType, internalFormat) combo. */
export function getTexImageConverter(
  srcFormat: GLenum, srcType: GLenum, internalFormat: GLenum,
): TexelConverter | null {
  throw new Error('not implemented: getTexImageConverter');
}

/** Convenience for code that reads raw source texels only. */
export function getTexelReader(srcFormat: GLenum, srcType: GLenum): TexelReader | null {
  throw new Error('not implemented: getTexelReader');
}

/** Convenience for code that writes destination texels only. */
export function getTexelWriter(internalFormat: GLenum): TexelWriter | null {
  throw new Error('not implemented: getTexelWriter');
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
 */
export interface PackConverter {
  convert(
    src: ArrayBufferView, srcByteOffset: number,
    dst: ArrayBufferView, dstByteOffset: number,
  ): void;
}

export function getPackConverter(
  internalFormat: GLenum, packFormat: GLenum, packType: GLenum,
): PackConverter | null {
  throw new Error('not implemented: getPackConverter');
}

/* ================================================================== */
/* Numeric conversion helpers (boundary paths only — never hot)        */
/* ================================================================== */

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
const _halfScratch = new DataView(new ArrayBuffer(4));

/** IEEE half → float. */
export function halfToFloat(h: number): number {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return m === 0 ? s * 0 : s * (m / 1024) * 2 ** -14;
  if (e === 31) return m === 0 ? s * Infinity : NaN;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Float → IEEE half (round to nearest even). */
export function floatToHalf(f: number): number {
  if (f !== f) return 0x7e00; // NaN
  if (f === Infinity) return 0x7c00;
  if (f === -Infinity) return 0xfc00;
  if (f === 0) return 1 / f < 0 ? 0x8000 : 0;
  _halfScratch.setFloat32(0, f);
  const bits = _halfScratch.getUint32(0);
  const sign = (bits >>> 16) & 0x8000;
  let exp = ((bits >>> 23) & 0xff) - 127 + 15;
  let mant = bits & 0x7fffff;
  if (exp >= 31) return sign | 0x7c00; // overflow → inf
  if (exp <= 0) {
    // Subnormal or flush-to-zero.
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let half = mant >> shift;
    const rem = mant & ((1 << shift) - 1);
    if (rem > 1 << (shift - 1) || (rem === 1 << (shift - 1) && (half & 1))) half++;
    return sign | half;
  }
  let half = ((exp << 10) | (mant >> 13)) & 0x7fff;
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++;
  return sign | half;
}

/** sRGB-encoded → linear (used by the sampler and sRGB framebuffer blending). */
export function sRGBToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Linear → sRGB-encoded. */
export function linearToSRGB(c: number): number {
  return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** Packs float depth (0..1) + stencil into a DEPTH_STENCIL uint32 (depth: high 24 bits). */
export function packDepth24Stencil(depth: number, stencil: number): number {
  const d = Math.min(0xffffff, Math.max(0, Math.round(depth * 0xffffff)));
  return (d << 8) | (stencil & 0xff);
}

/** Unpacks float depth (0..1) from a packed DEPTH_STENCIL uint32. */
export function unpackDepth24(value: number): number {
  return ((value >>> 8) & 0xffffff) / 0xffffff;
}

/**
 * NOTE FOR IMPLEMENTERS: defineFormat() must be called for every format in
 * the list below (see imports). WebGL1 unsized formats (RGBA/RGB/LUMINANCE/
 * LUMINANCE_ALPHA/ALPHA, DEPTH_COMPONENT, DEPTH_STENCIL) get concrete sized
 * storage entries (e.g. RGBA → RGBA8 storage semantics) while keeping their
 * own GLenum keys, so getFormat() resolves both unsized and sized keys.
 */
export const ALL_INTERNAL_FORMATS: readonly GLenum[] = [
  // WebGL1 unsized (storage-backed)
  RED, RG, RGB, RGBA, LUMINANCE, LUMINANCE_ALPHA, ALPHA,
  DEPTH_COMPONENT, DEPTH_STENCIL,
  // WebGL1 sized
  RGBA4, RGB5_A1, RGB565, DEPTH_COMPONENT16, STENCIL_INDEX8, DEPTH24_STENCIL8,
  // WebGL2 R
  R8, R8_SNORM, R16F, R32F, R8UI, R8I, R16UI, R16I, R32UI, R32I,
  // WebGL2 RG
  RG8, RG8_SNORM, RG16F, RG32F, RG8UI, RG8I, RG16UI, RG16I, RG32UI, RG32I,
  // WebGL2 RGB
  RGB8, RGB8_SNORM, RGB16F, RGB32F, RGB8UI, RGB8I, RGB16UI, RGB16I, RGB32UI, RGB32I,
  // WebGL2 RGBA
  RGBA8, RGBA8_SNORM, RGBA16F, RGBA32F, RGBA8UI, RGBA8I, RGBA16UI, RGBA16I,
  RGBA32UI, RGBA32I,
  // WebGL2 packed / special
  RGB10_A2, RGB10_A2UI, R11F_G11F_B10F, RGB9_E5, SRGB8, SRGB8_ALPHA8,
  // WebGL2 depth
  DEPTH_COMPONENT24, DEPTH_COMPONENT32F, DEPTH32F_STENCIL8,
];
