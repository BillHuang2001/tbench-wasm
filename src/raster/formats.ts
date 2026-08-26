/**
 * formats.ts — THE single shared pixel-format registry (contract §3).
 *
 * Every WebGL 1.0/2.0 internal format (sized + unsized, color/depth/stencil,
 * normalized/integer/float/sRGB) has one `PixelFormatInfo` entry here,
 * registered via defineFormat() into the FORMATS table. gl/ uses this module
 * to allocate surfaces, validate renderbuffer/texture formats, convert
 * texImage2D uploads, and pack readPixels; raster uses it to know how to
 * write surfaces; present/ uses it to present the drawing buffer.
 * The upload/pack converters live in the sibling module formats-convert.ts
 * (re-exported below); this file holds the registry, validation tables and
 * the byte-aware decode/encode machinery.
 *
 * Storage representation decisions (MANDATORY — see Surface in types.ts):
 *  - Normalized 8-bit: Uint8Array / Int8Array (signed); 16-bit: Uint16Array
 *    / Int16Array (565/4444/5551 pack one texel per Uint16Array element).
 *  - All float formats (R16F..RGBA32F, R11F_G11F_B10F, RGB9_E5) are stored
 *    as Float32Array — native float math in the hot sampling path; the packed
 *    GL forms exist only at the upload/readPixels boundaries. R16F bpp = 4,
 *    RGBA16F bpp = 16, R11F_G11F_B10F = 12, RGB9_E5 = 12.
 *  - Depth: Float32Array (0..1) for DEPTH_COMPONENT16/24/32F and the depth
 *    plane of DEPTH*_STENCIL* (stencil plane = Uint8Array, split).
 *  - WebGL1 unsized formats get concrete sized storage entries under their
 *    OWN GLenum keys: RED→R8 storage, RG→RG8, RGB→RGB8, RGBA→RGBA8,
 *    LUMINANCE→1×u8 (decode (l,l,l,1)), LUMINANCE_ALPHA→2×u8, ALPHA→1×u8
 *    (decode (0,0,0,a)), DEPTH_COMPONENT→f32 depth, DEPTH_STENCIL→f32 depth
 *    + u8 stencil.
 *  - `decode`/`encode` work on the SURFACE representation and never allocate
 *    when `out` is provided (correctness paths: clear, readPixels, copy,
 *    blit); the texture sampler reads raw typed arrays directly with
 *    format-class fast paths.
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
   * Decode one texel at byte offset as [r,g,b,a] floats:
   * normalized formats → 0..1 (snorm −1..1); float → raw; integer → raw int
   * values (exact for all 32-bit ints — JS numbers are doubles). Depth texels
   * decode as (d,d,d,1). Writes into `out` (length ≥ 4) when provided (hot
   * path, no allocation); otherwise allocates a fresh Float32Array(4). Always
   * returns the out array.
   */
  decode(data: ArrayBufferView, byteOffset: number, out?: Float32Array): Float32Array;
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

/* ================================================================== */
/* Shared encode/decode factories (per storage class, NOT per format)  */
/* ================================================================== */

type DecodeFn = (data: ArrayBufferView, byteOffset: number, out: Float32Array) => void;
type EncodeFn = (
  data: ArrayBufferView, byteOffset: number, r: number, g: number, b: number, a: number,
) => void;

/** Decode expansion markers: out[k] = stored[exp[k]], or fill 0 / fill 1. */
export const FILL_ZERO = -1;
export const FILL_ONE = -2;

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/* --- module-scratch for bit reinterpretation (no per-texel allocation) --- */
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
const _halfScratch = new DataView(new ArrayBuffer(4));

/**
 * Byte-aware accessors. `data` is either the surface's OWN typed array
 * (element size == access width; direct element access) or a plain byte
 * container (Uint8Array; assemble/disassemble little-endian bytes). Callers
 * (gl/, tests) pass either; both must work. `off` is always a BYTE offset.
 */
interface AnyArr { BYTES_PER_ELEMENT: number; [i: number]: number }

export function readU16At(data: ArrayBufferView, off: number): number {
  const d = data as unknown as AnyArr;
  return d.BYTES_PER_ELEMENT === 2 ? d[off >> 1] : (d[off] | (d[off + 1] << 8));
}
export function writeU16At(data: ArrayBufferView, off: number, v: number): void {
  const d = data as unknown as AnyArr;
  if (d.BYTES_PER_ELEMENT === 2) d[off >> 1] = v;
  else { d[off] = v & 0xff; d[off + 1] = (v >> 8) & 0xff; }
}
export function readU32At(data: ArrayBufferView, off: number): number {
  const d = data as unknown as AnyArr;
  if (d.BYTES_PER_ELEMENT === 4) return d[off >> 2] >>> 0;
  return (d[off] | (d[off + 1] << 8) | (d[off + 2] << 16) | (d[off + 3] << 24)) >>> 0;
}
export function writeU32At(data: ArrayBufferView, off: number, v: number): void {
  const d = data as unknown as AnyArr;
  if (d.BYTES_PER_ELEMENT === 4) { d[off >> 2] = v >>> 0; return; }
  d[off] = v & 0xff; d[off + 1] = (v >>> 8) & 0xff;
  d[off + 2] = (v >>> 16) & 0xff; d[off + 3] = (v >>> 24) & 0xff;
}
export function readF32At(data: ArrayBufferView, off: number): number {
  const d = data as unknown as AnyArr;
  if (d.BYTES_PER_ELEMENT === 4) return d[off >> 2];
  _u32[0] = readU32At(data, off);
  return _f32[0];
}
export function writeF32At(data: ArrayBufferView, off: number, v: number): void {
  const d = data as unknown as AnyArr;
  if (d.BYTES_PER_ELEMENT === 4) { d[off >> 2] = v; return; }
  _f32[0] = v;
  writeU32At(data, off, _u32[0]);
}
function readU8At(data: ArrayBufferView, off: number): number {
  const d = data as unknown as AnyArr;
  return d.BYTES_PER_ELEMENT === 1 ? d[off] : (d[off / d.BYTES_PER_ELEMENT] >>> ((off % d.BYTES_PER_ELEMENT) * 8)) & 0xff;
}
function writeU8At(data: ArrayBufferView, off: number, v: number): void {
  const d = data as unknown as AnyArr;
  if (d.BYTES_PER_ELEMENT === 1) { d[off] = v & 0xff; return; }
  const i = off / d.BYTES_PER_ELEMENT;
  const shift = (off % d.BYTES_PER_ELEMENT) * 8;
  d[i] = (d[i] & ~(0xff << shift)) | ((v & 0xff) << shift);
}

/** Standard decode expansion: (c0..cN-1), missing g/b → 0, missing a → 1. */
function stdExp(n: number): number[] {
  const exp = [0, 1, 2, 3].slice(0, n);
  for (let i = n; i < 4; i++) exp.push(i === 3 ? FILL_ONE : FILL_ZERO);
  return exp;
}

/** Standard encode selection: first N of (r,g,b,a). */
function stdSel(n: number): number[] {
  return [0, 1, 2, 3].slice(0, n);
}

/** Per-component read/write parameters (one per storage class — NOT per format). */
export interface PerCompParams {
  bpe: number;              // bytes per element (storage width)
  scale: number;            // decode divisor (0..1 normalization)
  encScale: number;         // encode multiplier
  min: number; max: number; // encode clamp range
  round: boolean;           // false = raw float passthrough
  signed: boolean;          // signed storage (i8/i16/i32)
  float: boolean;           // f32 storage (raw float component)
}

const P_U8 =   { bpe: 1, scale: 255, encScale: 255, min: 0, max: 1, round: true, signed: false, float: false };
const P_I8 =   { bpe: 1, scale: 127, encScale: 127, min: -1, max: 1, round: true, signed: true, float: false };
const P_U16 =  { bpe: 2, scale: 65535, encScale: 65535, min: 0, max: 1, round: true, signed: false, float: false };
const P_I16 =  { bpe: 2, scale: 32767, encScale: 32767, min: -1, max: 1, round: true, signed: true, float: false };
const P_U32 =  { bpe: 4, scale: 4294967295, encScale: 4294967295, min: 0, max: 1, round: true, signed: false, float: false };
const P_I32 =  { bpe: 4, scale: 2147483647, encScale: 2147483647, min: -1, max: 1, round: true, signed: true, float: false };
const P_F32 =  { bpe: 4, scale: 1, encScale: 1, min: -Infinity, max: Infinity, round: false, signed: false, float: true };
export const P_I_U8 =  { bpe: 1, scale: 1, encScale: 1, min: 0, max: 255, round: true, signed: false, float: false };
export const P_I_I8 =  { bpe: 1, scale: 1, encScale: 1, min: -128, max: 127, round: true, signed: true, float: false };
export const P_I_U16 = { bpe: 2, scale: 1, encScale: 1, min: 0, max: 65535, round: true, signed: false, float: false };
export const P_I_I16 = { bpe: 2, scale: 1, encScale: 1, min: -32768, max: 32767, round: true, signed: true, float: false };
export const P_I_U32 = { bpe: 4, scale: 1, encScale: 1, min: 0, max: 4294967295, round: true, signed: false, float: false };
export const P_I_I32 = { bpe: 4, scale: 1, encScale: 1, min: -2147483648, max: 2147483647, round: true, signed: true, float: false };

/** Reads one stored component (exact JS number — doubles hold all 32-bit ints). */
export function readCompAt(data: ArrayBufferView, off: number, p: PerCompParams): number {
  const d = data as unknown as AnyArr;
  const bpe = p.bpe;
  if (d.BYTES_PER_ELEMENT === bpe) {
    const v = d[off / bpe];
    if (p.float) return v;
    if (bpe === 1) return p.signed ? (v << 24 >> 24) : v;
    if (bpe === 2) return p.signed ? (v << 16 >> 16) : v;
    return p.signed ? (v | 0) : v >>> 0;
  }
  // Byte container (Uint8Array): assemble little-endian.
  if (p.float) return readF32At(data, off);
  if (bpe === 1) return p.signed ? (d[off] << 24 >> 24) : d[off];
  if (bpe === 2) {
    const v = d[off] | (d[off + 1] << 8);
    return p.signed ? (v << 16 >> 16) : v;
  }
  const v = (d[off] | (d[off + 1] << 8) | (d[off + 2] << 16) | (d[off + 3] << 24)) >>> 0;
  return p.signed ? (v | 0) : v;
}

/** Writes one stored component (value already quantized to the target range). */
function writeCompAt(data: ArrayBufferView, off: number, v: number, p: PerCompParams): void {
  const d = data as unknown as AnyArr;
  const bpe = p.bpe;
  if (d.BYTES_PER_ELEMENT === bpe) { d[off / bpe] = v; return; }
  if (p.float) { _f32[0] = v; writeU32At(data, off, _u32[0]); return; }
  if (bpe === 1) { d[off] = v & 0xff; return; }
  if (bpe === 2) { d[off] = v & 0xff; d[off + 1] = (v >> 8) & 0xff; return; }
  writeU32At(data, off, v >>> 0);
}

/**
 * Generic per-component decode/encode pair for u8/i8/u16/i16/u32/i32/f32
 * storages. `sel[j]` = which of (r,g,b,a) feeds stored component j;
 * `exp[k]` = which stored component feeds out[k] (or 0/1 fill).
 */
function makePerCompFns(
  p: PerCompParams,
  n: number, sel: number[], exp: number[],
): { decode: DecodeFn; encode: EncodeFn } {
  const decode: DecodeFn = (data, byteOffset, out) => {
    for (let k = 0; k < 4; k++) {
      const e = exp[k];
      out[k] = e < 0 ? (e === FILL_ONE ? 1 : 0) : readCompAt(data, byteOffset + e * p.bpe, p) / p.scale;
    }
  };
  const encode: EncodeFn = (data, byteOffset, r, g, b, a) => {
    const src = [r, g, b, a];
    for (let j = 0; j < n; j++) {
      const v = src[sel[j]];
      writeCompAt(data, byteOffset + j * p.bpe, p.round ? Math.round(clamp(v, p.min, p.max) * p.encScale) : v, p);
    }
  };
  return { decode, encode };
}

/** Per-component pair with standard layout (first N comps, 0/1 fills). */
function makeStd(p: PerCompParams, n: number): { decode: DecodeFn; encode: EncodeFn } {
  return makePerCompFns(p, n, stdSel(n), stdExp(n));
}

/** Packed one-texel-per-element formats (565/4444/5551 in u16; 10_A2 in u32). */
function makePackedU16(
  dec: (v: number, out: Float32Array) => void,
  enc: (r: number, g: number, b: number, a: number) => number,
): { decode: DecodeFn; encode: EncodeFn } {
  return {
    decode(data, byteOffset, out) {
      dec(readU16At(data, byteOffset), out);
    },
    encode(data, byteOffset, r, g, b, a) {
      writeU16At(data, byteOffset, enc(r, g, b, a) & 0xffff);
    },
  };
}

function makePackedU32(
  dec: (v: number, out: Float32Array) => void,
  enc: (r: number, g: number, b: number, a: number) => number,
): { decode: DecodeFn; encode: EncodeFn } {
  return {
    decode(data, byteOffset, out) {
      dec(readU32At(data, byteOffset), out);
    },
    encode(data, byteOffset, r, g, b, a) {
      writeU32At(data, byteOffset, enc(r, g, b, a) >>> 0);
    },
  };
}

const FNS_565 = makePackedU16(
  (v, out) => {
    out[0] = ((v >> 11) & 0x1f) / 31;
    out[1] = ((v >> 5) & 0x3f) / 63;
    out[2] = (v & 0x1f) / 31;
    out[3] = 1;
  },
  (r, g, b) =>
    (Math.round(clamp(r, 0, 1) * 31) << 11) |
    (Math.round(clamp(g, 0, 1) * 63) << 5) |
    Math.round(clamp(b, 0, 1) * 31),
);

const FNS_4444 = makePackedU16(
  (v, out) => {
    out[0] = ((v >> 12) & 0xf) / 15;
    out[1] = ((v >> 8) & 0xf) / 15;
    out[2] = ((v >> 4) & 0xf) / 15;
    out[3] = (v & 0xf) / 15;
  },
  (r, g, b, a) =>
    (Math.round(clamp(r, 0, 1) * 15) << 12) |
    (Math.round(clamp(g, 0, 1) * 15) << 8) |
    (Math.round(clamp(b, 0, 1) * 15) << 4) |
    Math.round(clamp(a, 0, 1) * 15),
);

const FNS_5551 = makePackedU16(
  (v, out) => {
    out[0] = ((v >> 11) & 0x1f) / 31;
    out[1] = ((v >> 6) & 0x1f) / 31;
    out[2] = ((v >> 1) & 0x1f) / 31;
    out[3] = v & 1;
  },
  (r, g, b, a) =>
    (Math.round(clamp(r, 0, 1) * 31) << 11) |
    (Math.round(clamp(g, 0, 1) * 31) << 6) |
    (Math.round(clamp(b, 0, 1) * 31) << 1) |
    (a >= 0.5 ? 1 : 0),
);

const FNS_10A2 = makePackedU32(
  (v, out) => {
    out[0] = ((v >> 22) & 0x3ff) / 1023;
    out[1] = ((v >> 12) & 0x3ff) / 1023;
    out[2] = ((v >> 2) & 0x3ff) / 1023;
    out[3] = (v & 0x3) / 3;
  },
  (r, g, b, a) =>
    (Math.round(clamp(r, 0, 1) * 1023) << 22) |
    (Math.round(clamp(g, 0, 1) * 1023) << 12) |
    (Math.round(clamp(b, 0, 1) * 1023) << 2) |
    Math.round(clamp(a, 0, 1) * 3),
);

const FNS_10A2UI = makePackedU32(
  (v, out) => {
    out[0] = (v >> 22) & 0x3ff;
    out[1] = (v >> 12) & 0x3ff;
    out[2] = (v >> 2) & 0x3ff;
    out[3] = v & 0x3;
  },
  (r, g, b, a) =>
    (Math.round(clamp(r, 0, 1023)) << 22) |
    (Math.round(clamp(g, 0, 1023)) << 12) |
    (Math.round(clamp(b, 0, 1023)) << 2) |
    Math.round(clamp(a, 0, 3)),
);

/* --- 11/11/10-bit float packing (R11F_G11F_B10F storage) --- */

export function unpack11(v: number): number {
  const e = (v >> 6) & 0x1f;
  const m = v & 0x3f;
  if (e === 0) return m === 0 ? 0 : (m / 64) * 2 ** -14;
  if (e === 31) return m === 0 ? Infinity : NaN;
  return (1 + m / 64) * 2 ** (e - 15);
}

export function unpack10(v: number): number {
  const e = (v >> 5) & 0x1f;
  const m = v & 0x1f;
  if (e === 0) return m === 0 ? 0 : (m / 32) * 2 ** -14;
  if (e === 31) return m === 0 ? Infinity : NaN;
  return (1 + m / 32) * 2 ** (e - 15);
}

/** Float → 11-bit float bits (5-bit exp bias 15, 6-bit mantissa; round-half-even). */
function floatToFloat11(f: number): number {
  if (f !== f || f <= 0) return 0; // NaN / negatives clamp to 0
  if (f === Infinity) return 0x7c0;
  _f32[0] = f;
  const bits = _u32[0];
  const exp = ((bits >>> 23) & 0xff) - 127 + 15;
  const mant = bits & 0x7fffff;
  if (exp >= 31) return 0x7c0; // overflow → inf
  if (exp <= 0) {
    if (exp < -6) return 0;
    const shift = 18 - exp;
    let v = (mant | 0x800000) >> shift;
    const rem = (mant | 0x800000) & ((1 << shift) - 1);
    if (rem > 1 << (shift - 1) || (rem === 1 << (shift - 1) && (v & 1))) v++;
    return v;
  }
  let v = (exp << 6) | (mant >> 17);
  const rem = mant & 0x1ffff;
  if (rem > 0x10000 || (rem === 0x10000 && (v & 1))) v++;
  return v;
}

/** Float → 10-bit float bits (5-bit exp bias 15, 5-bit mantissa; round-half-even). */
function floatToFloat10(f: number): number {
  if (f !== f || f <= 0) return 0;
  if (f === Infinity) return 0x7c0;
  _f32[0] = f;
  const bits = _u32[0];
  const exp = ((bits >>> 23) & 0xff) - 127 + 15;
  const mant = bits & 0x7fffff;
  if (exp >= 31) return 0x7c0;
  if (exp <= 0) {
    if (exp < -5) return 0;
    const shift = 19 - exp;
    let v = (mant | 0x800000) >> shift;
    const rem = (mant | 0x800000) & ((1 << shift) - 1);
    if (rem > 1 << (shift - 1) || (rem === 1 << (shift - 1) && (v & 1))) v++;
    return v;
  }
  let v = (exp << 5) | (mant >> 18);
  const rem = mant & 0x3ffff;
  if (rem > 0x20000 || (rem === 0x20000 && (v & 1))) v++;
  return v;
}

const FNS_111110: { decode: DecodeFn; encode: EncodeFn } = {
  decode(data, byteOffset, out) {
    out[0] = readF32At(data, byteOffset);
    out[1] = readF32At(data, byteOffset + 4);
    out[2] = readF32At(data, byteOffset + 8);
    out[3] = 1;
  },
  encode(data, byteOffset, r, g, b) {
    writeF32At(data, byteOffset, unpack11(floatToFloat11(r)));
    writeF32At(data, byteOffset + 4, unpack11(floatToFloat11(g)));
    writeF32At(data, byteOffset + 8, unpack10(floatToFloat10(b)));
  },
};

/* --- shared-exponent 9_9_9_5 packing (RGB9_E5 storage) --- */

const FNS_9E5: { decode: DecodeFn; encode: EncodeFn } = {
  decode(data, byteOffset, out) {
    out[0] = readF32At(data, byteOffset);
    out[1] = readF32At(data, byteOffset + 4);
    out[2] = readF32At(data, byteOffset + 8);
    out[3] = 1;
  },
  encode(data, byteOffset, r, g, b) {
    const maxC = Math.max(r, g, b);
    let rm = 0, gm = 0, bm = 0, scale = 0;
    if (maxC > 2 ** -25) {
      const exp = Math.max(0, Math.floor(Math.log2(maxC)) + 16);
      scale = 2 ** (exp - 24);
      rm = clamp(Math.round(r / scale), 0, 511);
      gm = clamp(Math.round(g / scale), 0, 511);
      bm = clamp(Math.round(b / scale), 0, 511);
    }
    writeF32At(data, byteOffset, rm * scale);
    writeF32At(data, byteOffset + 4, gm * scale);
    writeF32At(data, byteOffset + 8, bm * scale);
  },
};

/* --- depth / depth-stencil / stencil --- */

/** Depth plane: decode (d,d,d,1); encode writes r → depth. Also used for the
 *  depth plane of DEPTH*_STENCIL* formats (stencil plane written separately
 *  via Surface.stencilData). */
const FNS_DEPTH: { decode: DecodeFn; encode: EncodeFn } = {
  decode(data, byteOffset, out) {
    const d = readF32At(data, byteOffset);
    out[0] = d; out[1] = d; out[2] = d; out[3] = 1;
  },
  encode(data, byteOffset, r) {
    writeF32At(data, byteOffset, r);
  },
};

const FNS_STENCIL: { decode: DecodeFn; encode: EncodeFn } = {
  decode(data, byteOffset, out) {
    const s = readU8At(data, byteOffset);
    out[0] = s; out[1] = s; out[2] = s; out[3] = 1;
  },
  encode(data, byteOffset, r) {
    writeU8At(data, byteOffset, Math.round(clamp(r, 0, 255)));
  },
};

/* ================================================================== */
/* Registry population — one entry per format in ALL_INTERNAL_FORMATS   */
/* ================================================================== */

/** bytesPerPixel per storage class (per component). */
function bppFor(storage: StorageKind, components: number): number {
  switch (storage) {
    case 'u8': case 'i8': return components;
    case 'u16': case 'i16': case 'f16': return components * 2;
    default: return components * 4; // u32, i32, f32
  }
}

interface RegSpec {
  format: GLenum;
  components: number;
  storage: StorageKind;
  /** Override for PACKED formats (one element per texel): 565/4444/5551 → 2,
   *  RGB10_A2/RGB10_A2UI → 4. Default: per-component bppFor(). */
  bytesPerPixel?: number;
  isColor?: boolean;
  isDepth?: boolean;
  isStencil?: boolean;
  isFloat?: boolean;
  isSigned?: boolean;
  isInteger?: boolean;
  isSRGB?: boolean;
  normalized?: boolean;
  decode: DecodeFn;
  encode: EncodeFn;
}

function reg(spec: RegSpec): void {
  // Wrap the void DecodeFn into the public signature: write into `out` when
  // provided (no allocation), otherwise allocate a fresh Float32Array(4).
  const decode = (data: ArrayBufferView, byteOffset: number, out?: Float32Array): Float32Array => {
    const o = out ?? new Float32Array(4);
    spec.decode(data, byteOffset, o);
    return o;
  };
  defineFormat({
    format: spec.format,
    components: spec.components,
    bytesPerPixel: spec.bytesPerPixel ?? bppFor(spec.storage, spec.components),
    storage: spec.storage,
    isColor: spec.isColor ?? false,
    isDepth: spec.isDepth ?? false,
    isStencil: spec.isStencil ?? false,
    isFloat: spec.isFloat ?? false,
    isSigned: spec.isSigned ?? false,
    isInteger: spec.isInteger ?? false,
    isSRGB: spec.isSRGB ?? false,
    normalized: spec.normalized ?? false,
    decode,
    encode: spec.encode,
  });
}

/** Color format with standard per-component layout. */
function regColor(
  format: GLenum, storage: StorageKind, p: PerCompParams, n: number,
  extra: Partial<Omit<RegSpec, 'format' | 'components' | 'storage' | 'decode' | 'encode'>> = {},
): void {
  reg({ format, components: n, storage, isColor: true, ...extra, ...makeStd(p, n) });
}

/** Color format with a custom decode/encode pair. */
function regColorFns(
  format: GLenum, storage: StorageKind, n: number,
  fns: { decode: DecodeFn; encode: EncodeFn },
  extra: Partial<Omit<RegSpec, 'format' | 'components' | 'storage' | 'decode' | 'encode'>> = {},
): void {
  reg({ format, components: n, storage, isColor: true, ...extra, ...fns });
}
// --- WebGL1 unsized (storage-backed) ---
regColor(RED, 'u8', P_U8, 1, { normalized: true });
regColor(RG, 'u8', P_U8, 2, { normalized: true });
regColor(RGB, 'u8', P_U8, 3, { normalized: true });
regColor(RGBA, 'u8', P_U8, 4, { normalized: true });
// LUMINANCE (l,l,l,1) / LUMINANCE_ALPHA (l,l,l,a) / ALPHA (0,0,0,a)
regColorFns(LUMINANCE, 'u8', 1, makePerCompFns(P_U8, 1, [0], [0, 0, 0, FILL_ONE]), { normalized: true });
regColorFns(LUMINANCE_ALPHA, 'u8', 2, makePerCompFns(P_U8, 2, [0, 3], [0, 0, 0, 1]), { normalized: true });
regColorFns(ALPHA, 'u8', 1, makePerCompFns(P_U8, 1, [3], [FILL_ZERO, FILL_ZERO, FILL_ZERO, 0]), { normalized: true });
reg({ format: DEPTH_COMPONENT, components: 1, storage: 'f32', isDepth: true, ...FNS_DEPTH });
reg({ format: DEPTH_STENCIL, components: 1, storage: 'f32', isDepth: true, isStencil: true, ...FNS_DEPTH });

// --- WebGL1 sized ---
regColorFns(RGBA4, 'u16', 4, FNS_4444, { normalized: true, bytesPerPixel: 2 });
regColorFns(RGB5_A1, 'u16', 4, FNS_5551, { normalized: true, bytesPerPixel: 2 });
regColorFns(RGB565, 'u16', 3, FNS_565, { normalized: true, bytesPerPixel: 2 });
reg({ format: DEPTH_COMPONENT16, components: 1, storage: 'f32', isDepth: true, ...FNS_DEPTH });
reg({ format: STENCIL_INDEX8, components: 1, storage: 'u8', isStencil: true, ...FNS_STENCIL });
reg({ format: DEPTH24_STENCIL8, components: 1, storage: 'f32', isDepth: true, isStencil: true, ...FNS_DEPTH });

// --- WebGL2 R ---
regColor(R8, 'u8', P_U8, 1, { normalized: true });
regColor(R8_SNORM, 'i8', P_I8, 1, { normalized: true, isSigned: true });
regColor(R16F, 'f32', P_F32, 1, { isFloat: true });
regColor(R32F, 'f32', P_F32, 1, { isFloat: true });
regColor(R8UI, 'u8', P_I_U8, 1, { isInteger: true });
regColor(R8I, 'i8', P_I_I8, 1, { isInteger: true, isSigned: true });
regColor(R16UI, 'u16', P_I_U16, 1, { isInteger: true });
regColor(R16I, 'i16', P_I_I16, 1, { isInteger: true, isSigned: true });
regColor(R32UI, 'u32', P_I_U32, 1, { isInteger: true });
regColor(R32I, 'i32', P_I_I32, 1, { isInteger: true, isSigned: true });

// --- WebGL2 RG ---
regColor(RG8, 'u8', P_U8, 2, { normalized: true });
regColor(RG8_SNORM, 'i8', P_I8, 2, { normalized: true, isSigned: true });
regColor(RG16F, 'f32', P_F32, 2, { isFloat: true });
regColor(RG32F, 'f32', P_F32, 2, { isFloat: true });
regColor(RG8UI, 'u8', P_I_U8, 2, { isInteger: true });
regColor(RG8I, 'i8', P_I_I8, 2, { isInteger: true, isSigned: true });
regColor(RG16UI, 'u16', P_I_U16, 2, { isInteger: true });
regColor(RG16I, 'i16', P_I_I16, 2, { isInteger: true, isSigned: true });
regColor(RG32UI, 'u32', P_I_U32, 2, { isInteger: true });
regColor(RG32I, 'i32', P_I_I32, 2, { isInteger: true, isSigned: true });

// --- WebGL2 RGB ---
regColor(RGB8, 'u8', P_U8, 3, { normalized: true });
regColor(RGB8_SNORM, 'i8', P_I8, 3, { normalized: true, isSigned: true });
regColor(RGB16F, 'f32', P_F32, 3, { isFloat: true });
regColor(RGB32F, 'f32', P_F32, 3, { isFloat: true });
regColor(RGB8UI, 'u8', P_I_U8, 3, { isInteger: true });
regColor(RGB8I, 'i8', P_I_I8, 3, { isInteger: true, isSigned: true });
regColor(RGB16UI, 'u16', P_I_U16, 3, { isInteger: true });
regColor(RGB16I, 'i16', P_I_I16, 3, { isInteger: true, isSigned: true });
regColor(RGB32UI, 'u32', P_I_U32, 3, { isInteger: true });
regColor(RGB32I, 'i32', P_I_I32, 3, { isInteger: true, isSigned: true });

// --- WebGL2 RGBA ---
regColor(RGBA8, 'u8', P_U8, 4, { normalized: true });
regColor(RGBA8_SNORM, 'i8', P_I8, 4, { normalized: true, isSigned: true });
regColor(RGBA16F, 'f32', P_F32, 4, { isFloat: true });
regColor(RGBA32F, 'f32', P_F32, 4, { isFloat: true });
regColor(RGBA8UI, 'u8', P_I_U8, 4, { isInteger: true });
regColor(RGBA8I, 'i8', P_I_I8, 4, { isInteger: true, isSigned: true });
regColor(RGBA16UI, 'u16', P_I_U16, 4, { isInteger: true });
regColor(RGBA16I, 'i16', P_I_I16, 4, { isInteger: true, isSigned: true });
regColor(RGBA32UI, 'u32', P_I_U32, 4, { isInteger: true });
regColor(RGBA32I, 'i32', P_I_I32, 4, { isInteger: true, isSigned: true });

// --- WebGL2 packed / special ---
regColorFns(RGB10_A2, 'u32', 4, FNS_10A2, { normalized: true, bytesPerPixel: 4 });
regColorFns(RGB10_A2UI, 'u32', 4, FNS_10A2UI, { isInteger: true, bytesPerPixel: 4 });
regColorFns(R11F_G11F_B10F, 'f32', 3, FNS_111110, { isFloat: true });
regColorFns(RGB9_E5, 'f32', 3, FNS_9E5, { isFloat: true });
regColor(SRGB8, 'u8', P_U8, 3, { normalized: true, isSRGB: true });
regColor(SRGB8_ALPHA8, 'u8', P_U8, 4, { normalized: true, isSRGB: true });

// --- WebGL2 depth ---
reg({ format: DEPTH_COMPONENT24, components: 1, storage: 'f32', isDepth: true, ...FNS_DEPTH });
reg({ format: DEPTH_COMPONENT32F, components: 1, storage: 'f32', isDepth: true, isFloat: true, ...FNS_DEPTH });
reg({ format: DEPTH32F_STENCIL8, components: 1, storage: 'f32', isDepth: true, isStencil: true, isFloat: true, ...FNS_DEPTH });

/* ================================================================== */
/* Validation tables                                                   */
/* ================================================================== */

/**
 * Renderbuffer format validation (renderbufferStorage). WebGL1: RGBA4, RGB565,
 * RGB5_A1, DEPTH_COMPONENT16, STENCIL_INDEX8, DEPTH_STENCIL. WebGL2: the ES3
 * color-renderable set (R8, RG8, RGBA8, RGB10_A2, RGB10_A2UI, RGBA4, RGB5_A1,
 * RGB565, SRGB8_ALPHA8, R16F/RG16F/RGBA16F, integer R8I..RGBA32I/UI) +
 * depth/stencil formats. NOTE: R32F/RG32F/RGBA32F/R11F_G11F_B10F/RGB9_E5 are
 * ES3+EXT_color_buffer_float additions — gl/ must gate them on the extension.
 * NOT renderable: RGB8, RGB16F, RGB32F, SRGB8.
 */
const RENDERBUFFER_FORMATS_1 = new Set<number>([
  RGBA4, RGB565, RGB5_A1, DEPTH_COMPONENT16, STENCIL_INDEX8, DEPTH_STENCIL,
]);
const RENDERBUFFER_FORMATS_2 = new Set<number>([
  R8, RG8, RGBA8, RGB10_A2, RGB10_A2UI, RGBA4, RGB5_A1, RGB565, SRGB8_ALPHA8,
  R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F, R11F_G11F_B10F, RGB9_E5,
  R8I, R8UI, RG8I, RG8UI, RGBA8I, RGBA8UI,
  R16I, R16UI, RG16I, RG16UI, RGBA16I, RGBA16UI,
  R32I, R32UI, RG32I, RG32UI, RGB32I, RGB32UI, RGBA32I, RGBA32UI,
  DEPTH_COMPONENT16, DEPTH_COMPONENT24, DEPTH_COMPONENT32F,
  DEPTH24_STENCIL8, DEPTH32F_STENCIL8, STENCIL_INDEX8,
]);

export function isValidRenderbufferFormat(format: GLenum, version: 1 | 2): boolean {
  return version === 1
    ? RENDERBUFFER_FORMATS_1.has(format)
    : RENDERBUFFER_FORMATS_2.has(format);
}

/**
 * Texture internal format validation for texImage2D/texStorage. WebGL1
 * accepts ONLY the unsized formats (RGBA/RGB/LUMINANCE/LUMINANCE_ALPHA/ALPHA +
 * DEPTH_COMPONENT/DEPTH_STENCIL; RED/RG are not valid WebGL1 internal
 * formats); WebGL2 requires sized formats (every other entry in
 * ALL_INTERNAL_FORMATS) and rejects unsized.
 */
const UNSIZED_INTERNAL = new Set<number>([
  RED, RG, RGB, RGBA, LUMINANCE, LUMINANCE_ALPHA, ALPHA, DEPTH_COMPONENT, DEPTH_STENCIL,
]);
const WEBGL1_TEXTURE_FORMATS = new Set<number>([
  RGB, RGBA, LUMINANCE, LUMINANCE_ALPHA, ALPHA, DEPTH_COMPONENT, DEPTH_STENCIL,
]);

export function isValidTextureInternalFormat(format: GLenum, version: 1 | 2): boolean {
  if (version === 1) return WEBGL1_TEXTURE_FORMATS.has(format);
  return !UNSIZED_INTERNAL.has(format) && ALL_INTERNAL_FORMATS.includes(format);
}

/* ================================================================== */
/* Converter API (implemented in formats-convert.ts)                   */
/* ================================================================== */
export {
  getTexImageConverter, getTexelReader, getTexelWriter, convertPixels,
  getPackConverter,
} from './formats-convert';
export type {
  TexelReader, TexelWriter, TexelConverter, PackConverter,
} from './formats-convert';


/* ================================================================== */
/* Numeric conversion helpers (boundary paths only — never hot)        */
/* ================================================================== */

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
 * Every format that must be registered (see imports). WebGL1 unsized formats
 * (RGBA/RGB/LUMINANCE/LUMINANCE_ALPHA/ALPHA, DEPTH_COMPONENT, DEPTH_STENCIL)
 * get concrete sized storage entries while keeping their own GLenum keys, so
 * getFormat() resolves both unsized and sized keys.
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