/**
 * src/gl/teximage.ts — texture upload engine (internal; api/teximage.ts delegates).
 *
 * Handles ALL texImage* / texSubImage* / texStorage* / copyTex* / compressedTex* / generateMipmap
 * storage mutations of WebGLTexture:
 *  - Level allocation + per-texel source→storage conversion (UNPACK_ALIGNMENT row
 *    padding, WebGL2 ROW_LENGTH/SKIP_*, UNPACK_FLIP_Y, DOM premultiply).
 *  - PIXEL_UNPACK_BUFFER offset path (WebGL2: pixels is a byte offset).
 *  - DOM source decode via present/ (stubbed → zero-fill fallback, documented gap).
 *  - texStorage immutability, cube-face handling, depth-stencil split planes,
 *    generateMipmap 2×2 box downsample, copyTex* from the read framebuffer
 *    (framebuffer-util is stubbed → defensive no-op).
 *
 * Storage conventions (contract §3): normalized 8-bit → Uint8Array/Int8Array;
 * 16-bit → Uint16Array/Int16Array; float formats → Float32Array (incl. 16F,
 * R11F_G11F_B10F, RGB9_E5); depth → Float32Array; DEPTH*_STENCIL* → split
 * data (Float32Array) + stencilData (Uint8Array); integer formats → natural
 * typed arrays. Unsized internal formats keep their user GLenum in
 * `_image.internalFormat` but allocate sized storage.
 *
 * The per-format registry below is LOCAL (comment: replace with raster/formats
 * when it lands) — `getFormat()` from raster is consulted first and falls back
 * to these descriptors. Completeness metadata (`_image.complete`,
 * `baseLevel`/`maxLevel`, dims) is recomputed after every mutation.
 */

import type { WebGLRenderingContext } from './webgl1';
import type { WebGLTexture, TextureLevel } from './objects';
import type { GLenum, GLint, GLsizei, TexImageSource } from './types';
import { C, CExt } from './constants';
import { getFormat, halfToFloat, type PixelFormatInfo, type StorageKind } from '../raster';
import { resolveReadSurface } from './framebuffer-util';
import { decodeImageSource } from '../present';
import type { Surface } from '../raster';

export type TexImageSourceArg = ArrayBufferView | number | TexImageSource | null;

// ---------------------------------------------------------------------------
// Local storage-format registry (per-texel decode/encode on the SURFACE
// representation; see header). TODO: replace with raster/formats when it lands.
// ---------------------------------------------------------------------------

/** Component mapping of a color format (how decode fills [r,g,b,a]). */
type ChannelMode = 'rgba' | 'rgb' | 'rg' | 'red' | 'luminance' | 'alpha' | 'lumalpha' | 'packed';

interface FormatSpec {
  /** Storage GLenum (sized key). */
  format: GLenum;
  components: number;
  bytesPerPixel: number;
  storage: StorageKind;
  bytesPerElement: number;
  ctor: new (length: number) => ArrayBufferView;
  isColor: boolean;
  isDepth: boolean;
  isStencil: boolean;
  isFloat: boolean;
  isSigned: boolean;
  isInteger: boolean;
  isSRGB: boolean;
  normalized: boolean;
  /** Unpack one texel (byte offset) → [r,g,b,a] in surface domain (0..1 norm / raw). */
  unpack: (data: ArrayBufferView, off: number, out: Float32Array) => void;
  /** Pack [r,g,b,a] (surface domain) into one texel at byte offset. */
  pack: (data: ArrayBufferView, off: number, r: number, g: number, b: number, a: number) => void;
}

const clamp01 = (v: number): number => (v < 0 ? 0 : v > 1 ? 1 : v);
const roundClamp = (v: number): number => Math.round(clamp01(v) * 255);

/** Generic normalized unpack for u8/u16/u32 (and i8/i16/i32 snorm) storage. */
function normUnpack(storage: StorageKind, comps: number, mode: ChannelMode): FormatSpec['unpack'] {
  const div =
    storage === 'u8' ? 255
    : storage === 'i8' ? 127
    : storage === 'u16' ? 65535
    : storage === 'i16' ? 32767
    : storage === 'u32' ? 4294967295
    : 2147483647;
  const step = storage === 'u8' || storage === 'i8' ? 1 : 2;
  return (data, off, out) => {
    const e = off / step;
    const v = (data as unknown as { [i: number]: number })[e];
    const c = v / div;
    switch (mode) {
      case 'rgba': out[0] = c; out[1] = (data as unknown as { [i: number]: number })[e + 1] / div; out[2] = (data as unknown as { [i: number]: number })[e + 2] / div; out[3] = (data as unknown as { [i: number]: number })[e + 3] / div; break;
      case 'rgb': out[0] = c; out[1] = (data as unknown as { [i: number]: number })[e + 1] / div; out[2] = (data as unknown as { [i: number]: number })[e + 2] / div; out[3] = 1; break;
      case 'rg': out[0] = c; out[1] = (data as unknown as { [i: number]: number })[e + 1] / div; out[2] = 0; out[3] = 1; break;
      case 'red': out[0] = c; out[1] = 0; out[2] = 0; out[3] = 1; break;
      case 'luminance': out[0] = c; out[1] = c; out[2] = c; out[3] = 1; break;
      case 'alpha': out[0] = 0; out[1] = 0; out[2] = 0; out[3] = c; break;
      case 'lumalpha': out[0] = c; out[1] = c; out[2] = c; out[3] = (data as unknown as { [i: number]: number })[e + 1] / div; break;
      default: out[0] = out[1] = out[2] = 0; out[3] = 1;
    }
    void comps;
  };
}

/** Generic integer unpack (raw values). */
function intUnpack(storage: StorageKind, comps: number, mode: ChannelMode): FormatSpec['unpack'] {
  const step = storage === 'u8' || storage === 'i8' ? 1 : storage === 'u16' || storage === 'i16' ? 2 : 4;
  return (data, off, out) => {
    const e = off / step;
    const d = data as unknown as { [i: number]: number };
    const v = d[e];
    switch (mode) {
      case 'rgba': out[0] = v; out[1] = d[e + 1]; out[2] = d[e + 2]; out[3] = d[e + 3]; break;
      case 'rgb': out[0] = v; out[1] = d[e + 1]; out[2] = d[e + 2]; out[3] = 1; break;
      case 'rg': out[0] = v; out[1] = d[e + 1]; out[2] = 0; out[3] = 1; break;
      case 'red': out[0] = v; out[1] = 0; out[2] = 0; out[3] = 1; break;
      default: out[0] = out[1] = out[2] = 0; out[3] = 1;
    }
    void comps;
  };
}

/** Generic float unpack (raw). */
function floatUnpack(comps: number, mode: ChannelMode): FormatSpec['unpack'] {
  return (data, off, out) => {
    const e = off / 4;
    const d = data as unknown as { [i: number]: number };
    const v = d[e];
    switch (mode) {
      case 'rgba': out[0] = v; out[1] = d[e + 1]; out[2] = d[e + 2]; out[3] = d[e + 3]; break;
      case 'rgb': out[0] = v; out[1] = d[e + 1]; out[2] = d[e + 2]; out[3] = 1; break;
      case 'rg': out[0] = v; out[1] = d[e + 1]; out[2] = 0; out[3] = 1; break;
      case 'red': out[0] = v; out[1] = 0; out[2] = 0; out[3] = 1; break;
      default: out[0] = out[1] = out[2] = 0; out[3] = 1;
    }
    void comps;
  };
}

/** Generic normalized pack for u8/u16/u32 (and snorm). */
function normPack(storage: StorageKind, comps: number): FormatSpec['pack'] {
  const step = storage === 'u8' || storage === 'i8' ? 1 : 2;
  return (data, off, r, g, b, a) => {
    const e = off / step;
    const d = data as unknown as { [i: number]: number };
    if (storage === 'i8' || storage === 'i16' || storage === 'i32') {
      d[e] = Math.round(clamp01(r) * (storage === 'i8' ? 127 : storage === 'i16' ? 32767 : 2147483647));
      if (comps > 1) d[e + 1] = Math.round(clamp01(g) * (storage === 'i8' ? 127 : storage === 'i16' ? 32767 : 2147483647));
      if (comps > 2) d[e + 2] = Math.round(clamp01(b) * (storage === 'i8' ? 127 : storage === 'i16' ? 32767 : 2147483647));
      if (comps > 3) d[e + 3] = Math.round(clamp01(a) * (storage === 'i8' ? 127 : storage === 'i16' ? 32767 : 2147483647));
    } else if (storage === 'u32') {
      d[e] = Math.round(clamp01(r) * 4294967295);
      if (comps > 1) d[e + 1] = Math.round(clamp01(g) * 4294967295);
      if (comps > 2) d[e + 2] = Math.round(clamp01(b) * 4294967295);
      if (comps > 3) d[e + 3] = Math.round(clamp01(a) * 4294967295);
    } else if (storage === 'u16') {
      d[e] = Math.round(clamp01(r) * 65535);
      if (comps > 1) d[e + 1] = Math.round(clamp01(g) * 65535);
      if (comps > 2) d[e + 2] = Math.round(clamp01(b) * 65535);
      if (comps > 3) d[e + 3] = Math.round(clamp01(a) * 65535);
    } else {
      d[e] = roundClamp(r);
      if (comps > 1) d[e + 1] = roundClamp(g);
      if (comps > 2) d[e + 2] = roundClamp(b);
      if (comps > 3) d[e + 3] = roundClamp(a);
    }
  };
}

/** Generic integer pack (raw truncation). */
function intPack(comps: number): FormatSpec['pack'] {
  return (data, off, r, g, b, a) => {
    const d = data as unknown as { [i: number]: number };
    const e = off / (data instanceof Uint16Array || data instanceof Int16Array ? 2 : data instanceof Uint32Array || data instanceof Int32Array ? 4 : 1);
    d[e] = r | 0;
    if (comps > 1) d[e + 1] = g | 0;
    if (comps > 2) d[e + 2] = b | 0;
    if (comps > 3) d[e + 3] = a | 0;
  };
}

/** Generic float pack (raw). */
function floatPack(comps: number): FormatSpec['pack'] {
  return (data, off, r, g, b, a) => {
    const e = off / 4;
    const d = data as unknown as { [i: number]: number };
    d[e] = r;
    if (comps > 1) d[e + 1] = g;
    if (comps > 2) d[e + 2] = b;
    if (comps > 3) d[e + 3] = a;
  };
}

/** u16 packed texels: 565 / 4444 / 5551. */
function packed565Unpack(data: ArrayBufferView, off: number, out: Float32Array): void {
  const v = (data as unknown as { [i: number]: number })[off / 2];
  out[0] = ((v >> 11) & 0x1f) / 31;
  out[1] = ((v >> 5) & 0x3f) / 63;
  out[2] = (v & 0x1f) / 31;
  out[3] = 1;
}
function packed565Pack(data: ArrayBufferView, off: number, r: number, g: number, b: number, _a: number): void {
  const v = (Math.round(clamp01(r) * 31) << 11) | (Math.round(clamp01(g) * 63) << 5) | Math.round(clamp01(b) * 31);
  (data as unknown as { [i: number]: number })[off / 2] = v;
}
function packed4444Unpack(data: ArrayBufferView, off: number, out: Float32Array): void {
  const v = (data as unknown as { [i: number]: number })[off / 2];
  out[0] = ((v >> 12) & 0xf) / 15;
  out[1] = ((v >> 8) & 0xf) / 15;
  out[2] = ((v >> 4) & 0xf) / 15;
  out[3] = (v & 0xf) / 15;
}
function packed4444Pack(data: ArrayBufferView, off: number, r: number, g: number, b: number, a: number): void {
  const v = (Math.round(clamp01(r) * 15) << 12) | (Math.round(clamp01(g) * 15) << 8) | (Math.round(clamp01(b) * 15) << 4) | Math.round(clamp01(a) * 15);
  (data as unknown as { [i: number]: number })[off / 2] = v;
}
function packed5551Unpack(data: ArrayBufferView, off: number, out: Float32Array): void {
  const v = (data as unknown as { [i: number]: number })[off / 2];
  out[0] = ((v >> 11) & 0x1f) / 31;
  out[1] = ((v >> 6) & 0x1f) / 31;
  out[2] = ((v >> 1) & 0x1f) / 31;
  out[3] = v & 1;
}
function packed5551Pack(data: ArrayBufferView, off: number, r: number, g: number, b: number, a: number): void {
  const v = (Math.round(clamp01(r) * 31) << 11) | (Math.round(clamp01(g) * 31) << 6) | (Math.round(clamp01(b) * 31) << 1) | (a > 0.5 ? 1 : 0);
  (data as unknown as { [i: number]: number })[off / 2] = v;
}

/** u32 packed RGB10_A2 (normalized). */
function packedRGB10A2Unpack(data: ArrayBufferView, off: number, out: Float32Array): void {
  const v = (data as unknown as { [i: number]: number })[off / 4];
  out[0] = ((v >>> 22) & 0x3ff) / 1023;
  out[1] = ((v >>> 12) & 0x3ff) / 1023;
  out[2] = ((v >>> 2) & 0x3ff) / 1023;
  out[3] = (v & 0x3) / 3;
}
function packedRGB10A2Pack(data: ArrayBufferView, off: number, r: number, g: number, b: number, a: number): void {
  const v = (Math.round(clamp01(r) * 1023) << 22) | (Math.round(clamp01(g) * 1023) << 12) | (Math.round(clamp01(b) * 1023) << 2) | Math.round(clamp01(a) * 3);
  (data as unknown as { [i: number]: number })[off / 4] = v >>> 0;
}
/** u32 packed RGB10_A2UI (integer). */
function packedRGB10A2UIUnpack(data: ArrayBufferView, off: number, out: Float32Array): void {
  const v = (data as unknown as { [i: number]: number })[off / 4];
  out[0] = (v >>> 22) & 0x3ff;
  out[1] = (v >>> 12) & 0x3ff;
  out[2] = (v >>> 2) & 0x3ff;
  out[3] = v & 0x3;
}
function packedRGB10A2UIPack(data: ArrayBufferView, off: number, r: number, g: number, b: number, a: number): void {
  const v = ((r | 0) << 22) | ((g | 0) << 12) | ((b | 0) << 2) | (a | 0);
  (data as unknown as { [i: number]: number })[off / 4] = v >>> 0;
}

/** Depth formats: Float32Array of 0..1, decoded as (d,d,d,1). */
function depthUnpack(data: ArrayBufferView, off: number, out: Float32Array): void {
  const d = (data as unknown as { [i: number]: number })[off / 4];
  out[0] = d; out[1] = d; out[2] = d; out[3] = 1;
}
function depthPack(data: ArrayBufferView, off: number, r: number, _g: number, _b: number, _a: number): void {
  (data as unknown as { [i: number]: number })[off / 4] = clamp01(r);
}

interface SpecOpts {
  format: GLenum;
  components: number;
  bytesPerPixel: number;
  storage: StorageKind;
  ctor: new (length: number) => ArrayBufferView;
  isColor?: boolean;
  isDepth?: boolean;
  isStencil?: boolean;
  isFloat?: boolean;
  isSigned?: boolean;
  isInteger?: boolean;
  isSRGB?: boolean;
  normalized?: boolean;
  mode?: ChannelMode;
  unpack?: FormatSpec['unpack'];
  pack?: FormatSpec['pack'];
}

function buildSpec(o: SpecOpts): FormatSpec {
  const mode = o.mode ?? (o.components === 4 ? 'rgba' : o.components === 3 ? 'rgb' : o.components === 2 ? 'rg' : 'red');
  const norm = o.normalized ?? false;
  const integer = o.isInteger ?? false;
  const flt = o.isFloat ?? false;
  const unpack =
    o.unpack ??
    (flt ? floatUnpack(o.components, mode)
      : integer ? intUnpack(o.storage, o.components, mode)
      : normUnpack(o.storage, o.components, mode));
  const pack =
    o.pack ??
    (flt ? floatPack(o.components)
      : integer ? intPack(o.components)
      : normPack(o.storage, o.components));
  const bytesPerElement = o.storage === 'u8' || o.storage === 'i8' ? 1 : o.storage === 'u16' || o.storage === 'i16' ? 2 : o.storage === 'f16' ? 2 : 4;
  return {
    format: o.format,
    components: o.components,
    bytesPerPixel: o.bytesPerPixel,
    storage: o.storage,
    bytesPerElement,
    ctor: o.ctor,
    isColor: o.isColor ?? false,
    isDepth: o.isDepth ?? false,
    isStencil: o.isStencil ?? false,
    isFloat: flt,
    isSigned: o.isSigned ?? false,
    isInteger: integer,
    isSRGB: o.isSRGB ?? false,
    normalized: norm,
    unpack,
    pack,
  };
}

const U8 = Uint8Array;
const I8 = Int8Array;
const U16 = Uint16Array;
const I16 = Int16Array;
const U32 = Uint32Array;
const I32 = Int32Array;
const F32 = Float32Array;

/** Storage registry keyed by the sized storage GLenum. */
const SPECS = new Map<GLenum, FormatSpec>();
function reg(spec: FormatSpec): void {
  SPECS.set(spec.format, spec);
}

// Normalized 8/16-bit color
reg(buildSpec({ format: C.RGBA8, components: 4, bytesPerPixel: 4, storage: 'u8', ctor: U8, isColor: true, normalized: true, mode: 'rgba' }));
reg(buildSpec({ format: C.RGB8, components: 3, bytesPerPixel: 3, storage: 'u8', ctor: U8, isColor: true, normalized: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGBA4, components: 4, bytesPerPixel: 2, storage: 'u16', ctor: U16, isColor: true, normalized: true, mode: 'packed', unpack: packed4444Unpack, pack: packed4444Pack }));
reg(buildSpec({ format: C.RGB5_A1, components: 4, bytesPerPixel: 2, storage: 'u16', ctor: U16, isColor: true, normalized: true, mode: 'packed', unpack: packed5551Unpack, pack: packed5551Pack }));
reg(buildSpec({ format: C.RGB565, components: 3, bytesPerPixel: 2, storage: 'u16', ctor: U16, isColor: true, normalized: true, mode: 'packed', unpack: packed565Unpack, pack: packed565Pack }));
reg(buildSpec({ format: C.R8, components: 1, bytesPerPixel: 1, storage: 'u8', ctor: U8, isColor: true, normalized: true, mode: 'red' }));
reg(buildSpec({ format: C.RG8, components: 2, bytesPerPixel: 2, storage: 'u8', ctor: U8, isColor: true, normalized: true, mode: 'rg' }));
reg(buildSpec({ format: C.R8_SNORM, components: 1, bytesPerPixel: 1, storage: 'i8', ctor: I8, isColor: true, isSigned: true, normalized: true, mode: 'red' }));
reg(buildSpec({ format: C.RG8_SNORM, components: 2, bytesPerPixel: 2, storage: 'i8', ctor: I8, isColor: true, isSigned: true, normalized: true, mode: 'rg' }));
reg(buildSpec({ format: C.RGB8_SNORM, components: 3, bytesPerPixel: 3, storage: 'i8', ctor: I8, isColor: true, isSigned: true, normalized: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGBA8_SNORM, components: 4, bytesPerPixel: 4, storage: 'i8', ctor: I8, isColor: true, isSigned: true, normalized: true, mode: 'rgba' }));
reg(buildSpec({ format: CExt.R16_EXT, components: 1, bytesPerPixel: 2, storage: 'u16', ctor: U16, isColor: true, normalized: true, mode: 'red' }));
reg(buildSpec({ format: CExt.RG16_EXT, components: 2, bytesPerPixel: 4, storage: 'u16', ctor: U16, isColor: true, normalized: true, mode: 'rg' }));
reg(buildSpec({ format: CExt.RGB16_EXT, components: 3, bytesPerPixel: 6, storage: 'u16', ctor: U16, isColor: true, normalized: true, mode: 'rgb' }));
reg(buildSpec({ format: CExt.RGBA16_EXT, components: 4, bytesPerPixel: 8, storage: 'u16', ctor: U16, isColor: true, normalized: true, mode: 'rgba' }));
reg(buildSpec({ format: CExt.R16_SNORM_EXT, components: 1, bytesPerPixel: 2, storage: 'i16', ctor: I16, isColor: true, isSigned: true, normalized: true, mode: 'red' }));
reg(buildSpec({ format: CExt.RG16_SNORM_EXT, components: 2, bytesPerPixel: 4, storage: 'i16', ctor: I16, isColor: true, isSigned: true, normalized: true, mode: 'rg' }));
reg(buildSpec({ format: CExt.RGB16_SNORM_EXT, components: 3, bytesPerPixel: 6, storage: 'i16', ctor: I16, isColor: true, isSigned: true, normalized: true, mode: 'rgb' }));
reg(buildSpec({ format: CExt.RGBA16_SNORM_EXT, components: 4, bytesPerPixel: 8, storage: 'i16', ctor: I16, isColor: true, isSigned: true, normalized: true, mode: 'rgba' }));

// Unsized WebGL1 formats (own storage entries)
reg(buildSpec({ format: C.LUMINANCE, components: 1, bytesPerPixel: 1, storage: 'u8', ctor: U8, isColor: true, normalized: true, mode: 'luminance' }));
reg(buildSpec({ format: C.LUMINANCE_ALPHA, components: 2, bytesPerPixel: 2, storage: 'u8', ctor: U8, isColor: true, normalized: true, mode: 'lumalpha' }));
reg(buildSpec({ format: C.ALPHA, components: 1, bytesPerPixel: 1, storage: 'u8', ctor: U8, isColor: true, normalized: true, mode: 'alpha' }));

// Float formats → Float32Array storage
reg(buildSpec({ format: C.R16F, components: 1, bytesPerPixel: 4, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'red' }));
reg(buildSpec({ format: C.RG16F, components: 2, bytesPerPixel: 8, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rg' }));
reg(buildSpec({ format: C.RGB16F, components: 3, bytesPerPixel: 12, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGBA16F, components: 4, bytesPerPixel: 16, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rgba' }));
reg(buildSpec({ format: C.R32F, components: 1, bytesPerPixel: 4, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'red' }));
reg(buildSpec({ format: C.RG32F, components: 2, bytesPerPixel: 8, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rg' }));
reg(buildSpec({ format: C.RGB32F, components: 3, bytesPerPixel: 12, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGBA32F, components: 4, bytesPerPixel: 16, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rgba' }));
reg(buildSpec({ format: C.R11F_G11F_B10F, components: 3, bytesPerPixel: 12, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGB9_E5, components: 3, bytesPerPixel: 12, storage: 'f32', ctor: F32, isColor: true, isFloat: true, mode: 'rgb' }));

// sRGB (stored like the matching linear format; isSRGB flag)
reg(buildSpec({ format: C.SRGB8, components: 3, bytesPerPixel: 3, storage: 'u8', ctor: U8, isColor: true, normalized: true, isSRGB: true, mode: 'rgb' }));
reg(buildSpec({ format: C.SRGB8_ALPHA8, components: 4, bytesPerPixel: 4, storage: 'u8', ctor: U8, isColor: true, normalized: true, isSRGB: true, mode: 'rgba' }));

// Integer formats → natural typed arrays (raw values)
reg(buildSpec({ format: C.R8UI, components: 1, bytesPerPixel: 1, storage: 'u8', ctor: U8, isColor: true, isInteger: true, mode: 'red' }));
reg(buildSpec({ format: C.R8I, components: 1, bytesPerPixel: 1, storage: 'i8', ctor: I8, isColor: true, isInteger: true, isSigned: true, mode: 'red' }));
reg(buildSpec({ format: C.RG8UI, components: 2, bytesPerPixel: 2, storage: 'u8', ctor: U8, isColor: true, isInteger: true, mode: 'rg' }));
reg(buildSpec({ format: C.RG8I, components: 2, bytesPerPixel: 2, storage: 'i8', ctor: I8, isColor: true, isInteger: true, isSigned: true, mode: 'rg' }));
reg(buildSpec({ format: C.RGB8UI, components: 3, bytesPerPixel: 3, storage: 'u8', ctor: U8, isColor: true, isInteger: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGB8I, components: 3, bytesPerPixel: 3, storage: 'i8', ctor: I8, isColor: true, isInteger: true, isSigned: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGBA8UI, components: 4, bytesPerPixel: 4, storage: 'u8', ctor: U8, isColor: true, isInteger: true, mode: 'rgba' }));
reg(buildSpec({ format: C.RGBA8I, components: 4, bytesPerPixel: 4, storage: 'i8', ctor: I8, isColor: true, isInteger: true, isSigned: true, mode: 'rgba' }));
reg(buildSpec({ format: C.R16UI, components: 1, bytesPerPixel: 2, storage: 'u16', ctor: U16, isColor: true, isInteger: true, mode: 'red' }));
reg(buildSpec({ format: C.R16I, components: 1, bytesPerPixel: 2, storage: 'i16', ctor: I16, isColor: true, isInteger: true, isSigned: true, mode: 'red' }));
reg(buildSpec({ format: C.RG16UI, components: 2, bytesPerPixel: 4, storage: 'u16', ctor: U16, isColor: true, isInteger: true, mode: 'rg' }));
reg(buildSpec({ format: C.RG16I, components: 2, bytesPerPixel: 4, storage: 'i16', ctor: I16, isColor: true, isInteger: true, isSigned: true, mode: 'rg' }));
reg(buildSpec({ format: C.RGB16UI, components: 3, bytesPerPixel: 6, storage: 'u16', ctor: U16, isColor: true, isInteger: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGB16I, components: 3, bytesPerPixel: 6, storage: 'i16', ctor: I16, isColor: true, isInteger: true, isSigned: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGBA16UI, components: 4, bytesPerPixel: 8, storage: 'u16', ctor: U16, isColor: true, isInteger: true, mode: 'rgba' }));
reg(buildSpec({ format: C.RGBA16I, components: 4, bytesPerPixel: 8, storage: 'i16', ctor: I16, isColor: true, isInteger: true, isSigned: true, mode: 'rgba' }));
reg(buildSpec({ format: C.R32UI, components: 1, bytesPerPixel: 4, storage: 'u32', ctor: U32, isColor: true, isInteger: true, mode: 'red' }));
reg(buildSpec({ format: C.R32I, components: 1, bytesPerPixel: 4, storage: 'i32', ctor: I32, isColor: true, isInteger: true, isSigned: true, mode: 'red' }));
reg(buildSpec({ format: C.RG32UI, components: 2, bytesPerPixel: 8, storage: 'u32', ctor: U32, isColor: true, isInteger: true, mode: 'rg' }));
reg(buildSpec({ format: C.RG32I, components: 2, bytesPerPixel: 8, storage: 'i32', ctor: I32, isColor: true, isInteger: true, isSigned: true, mode: 'rg' }));
reg(buildSpec({ format: C.RGB32UI, components: 3, bytesPerPixel: 12, storage: 'u32', ctor: U32, isColor: true, isInteger: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGB32I, components: 3, bytesPerPixel: 12, storage: 'i32', ctor: I32, isColor: true, isInteger: true, isSigned: true, mode: 'rgb' }));
reg(buildSpec({ format: C.RGBA32UI, components: 4, bytesPerPixel: 16, storage: 'u32', ctor: U32, isColor: true, isInteger: true, mode: 'rgba' }));
reg(buildSpec({ format: C.RGBA32I, components: 4, bytesPerPixel: 16, storage: 'i32', ctor: I32, isColor: true, isInteger: true, isSigned: true, mode: 'rgba' }));

// Packed 10:10:10:2
reg(buildSpec({ format: C.RGB10_A2, components: 4, bytesPerPixel: 4, storage: 'u32', ctor: U32, isColor: true, normalized: true, mode: 'packed', unpack: packedRGB10A2Unpack, pack: packedRGB10A2Pack }));
reg(buildSpec({ format: C.RGB10_A2UI, components: 4, bytesPerPixel: 4, storage: 'u32', ctor: U32, isColor: true, isInteger: true, mode: 'packed', unpack: packedRGB10A2UIUnpack, pack: packedRGB10A2UIPack }));

// Depth / depth-stencil (split stencil plane)
reg(buildSpec({ format: C.DEPTH_COMPONENT16, components: 1, bytesPerPixel: 4, storage: 'f32', ctor: F32, isDepth: true, normalized: true, mode: 'red', unpack: depthUnpack, pack: depthPack }));
reg(buildSpec({ format: C.DEPTH_COMPONENT24, components: 1, bytesPerPixel: 4, storage: 'f32', ctor: F32, isDepth: true, normalized: true, mode: 'red', unpack: depthUnpack, pack: depthPack }));
reg(buildSpec({ format: C.DEPTH_COMPONENT32F, components: 1, bytesPerPixel: 4, storage: 'f32', ctor: F32, isDepth: true, isFloat: true, mode: 'red', unpack: depthUnpack, pack: depthPack }));
reg(buildSpec({ format: C.DEPTH24_STENCIL8, components: 1, bytesPerPixel: 4, storage: 'f32', ctor: F32, isDepth: true, isStencil: true, normalized: true, mode: 'red', unpack: depthUnpack, pack: depthPack }));
reg(buildSpec({ format: C.DEPTH32F_STENCIL8, components: 1, bytesPerPixel: 4, storage: 'f32', ctor: F32, isDepth: true, isStencil: true, isFloat: true, mode: 'red', unpack: depthUnpack, pack: depthPack }));
reg(buildSpec({ format: C.STENCIL_INDEX8, components: 1, bytesPerPixel: 1, storage: 'u8', ctor: U8, isStencil: true, normalized: true, mode: 'red' }));

/** Unsized internal format → sized storage GLenum. */
function unsizedStorage(internalformat: GLenum): GLenum {
  switch (internalformat) {
    case C.RGBA: return C.RGBA8;
    case C.RGB: return C.RGB8;
    case C.RED: return C.R8;
    case C.RG: return C.RG8;
    case C.DEPTH_COMPONENT: return C.DEPTH_COMPONENT16;
    case C.DEPTH_STENCIL: return C.DEPTH24_STENCIL8;
    case CExt.SRGB_EXT: return C.SRGB8;
    case CExt.SRGB_ALPHA_EXT: return C.SRGB8_ALPHA8;
    default: return internalformat;
  }
}

/** Resolve a storage spec for an internal format (raster registry first, local fallback). */
function resolveStorageSpec(internalformat: GLenum): FormatSpec | null {
  const storage = unsizedStorage(internalformat);
  const local = SPECS.get(storage);
  if (!local) return null;
  // raster/formats registry wins when it has the format (it is the single
  // source of truth once implemented); local descriptors are the fallback.
  const rasterInfo = getFormat(storage);
  if (rasterInfo) {
    return {
      ...local,
      unpack: (data, off, out) => rasterInfo.decode(data, off, out),
      pack: (data, off, r, g, b, a) => rasterInfo.encode(data, off, r, g, b, a),
      ctor: local.ctor,
    };
  }
  return local;
}

/** Build the PixelFormatInfo stored on `_image.info` (raster-compatible). */
function toPixelFormatInfo(spec: FormatSpec): PixelFormatInfo {
  return {
    format: spec.format,
    components: spec.components,
    bytesPerPixel: spec.bytesPerPixel,
    storage: spec.storage,
    isColor: spec.isColor,
    isDepth: spec.isDepth,
    isStencil: spec.isStencil,
    isFloat: spec.isFloat,
    isSigned: spec.isSigned,
    isInteger: spec.isInteger,
    isSRGB: spec.isSRGB,
    normalized: spec.normalized,
    decode: (data, byteOffset, out) => {
      const o = out ?? new Float32Array(4);
      spec.unpack(data, byteOffset, o);
      return o;
    },
    encode: (data, byteOffset, r, g, b, a) => spec.pack(data, byteOffset, r, g, b, a),
  };
}

// ---------------------------------------------------------------------------
// Level allocation
// ---------------------------------------------------------------------------

/** Allocate a zero-filled level record (6 views for cube, else 1). */
function allocLevel(spec: FormatSpec, w: number, h: number, d: number, isCube: boolean): TextureLevel {
  const perFace = w * h * (isCube ? 1 : d);
  const count = perFace * spec.bytesPerPixel / spec.bytesPerElement;
  const views: ArrayBufferView[] = [];
  const n = isCube ? 6 : 1;
  for (let i = 0; i < n; i++) views.push(new spec.ctor(count));
  const level: TextureLevel = { width: w, height: h, depth: d, data: views };
  if (spec.isStencil) level.stencilData = new Uint8Array(perFace * (isCube ? 1 : 1));
  return level;
}

const isCubeTarget = (t: GLenum): boolean => t === C.TEXTURE_CUBE_MAP || (t >= C.TEXTURE_CUBE_MAP_POSITIVE_X && t <= C.TEXTURE_CUBE_MAP_NEGATIVE_Z);
const cubeFaceIndex = (t: GLenum): number => (t >= C.TEXTURE_CUBE_MAP_POSITIVE_X && t <= C.TEXTURE_CUBE_MAP_NEGATIVE_Z ? t - C.TEXTURE_CUBE_MAP_POSITIVE_X : -1);

/** Canonical target of an upload (cube faces → TEXTURE_CUBE_MAP). */
const canonTarget = (t: GLenum): GLenum => (cubeFaceIndex(t) >= 0 ? C.TEXTURE_CUBE_MAP : t);

// ---------------------------------------------------------------------------
// Source texel readers (upload conversion, (format, type) → [r,g,b,a])
// ---------------------------------------------------------------------------

/** 11-bit float (R11F_G11F_B10F components) → float. */
function float11ToFloat(v: number): number {
  const e = (v >> 6) & 0x1f;
  const m = v & 0x3f;
  if (e === 0) return m === 0 ? 0 : m * 2 ** -20;
  if (e === 31) return m === 0 ? Infinity : NaN;
  return (1 + m / 64) * 2 ** (e - 15);
}
/** 10-bit float (R11F_G11F_B10F R component) → float. */
function float10ToFloat(v: number): number {
  const e = (v >> 5) & 0x1f;
  const m = v & 0x1f;
  if (e === 0) return m === 0 ? 0 : m * 2 ** -19;
  if (e === 31) return m === 0 ? Infinity : NaN;
  return (1 + m / 32) * 2 ** (e - 15);
}
/** RGB9_E5 shared-exponent texel → [r,g,b]. */
function rgb9e5ToFloat(v: number): [number, number, number] {
  const e = (v >>> 27) & 0x1f;
  const scale = e === 0 ? 2 ** -24 : 2 ** (e - 15);
  const r = ((v >>> 18) & 0x1ff) / 512 * scale * 512;
  const g = ((v >>> 9) & 0x1ff) / 512 * scale * 512;
  const b = (v & 0x1ff) / 512 * scale * 512;
  return [r, g, b];
}

/** Source component count for a (client-side) format. */
function srcComponents(format: GLenum): number {
  switch (format) {
    case C.RGBA: case C.RGBA_INTEGER: case 0x80e1: return 4; // BGRA (EXT_texture_format_BGRA8888, null status)
    case C.RGB: case C.RGB_INTEGER: return 3;
    case C.LUMINANCE_ALPHA: case C.RG: case C.RG_INTEGER: return 2;
    case C.DEPTH_STENCIL: return 2; // depth + stencil (packed 32-bit texels)
    default: return 1; // LUMINANCE, ALPHA, RED, RED_INTEGER, DEPTH_COMPONENT
  }
}

/** Bytes per texel of the source data for (format, type). */
function srcBytesPerTexel(format: GLenum, type: GLenum): number {
  const comps = srcComponents(format);
  switch (type) {
    case C.UNSIGNED_BYTE: case C.BYTE: return comps;
    case C.UNSIGNED_SHORT: case C.SHORT: case C.HALF_FLOAT: case CExt.HALF_FLOAT_OES:
    case C.UNSIGNED_SHORT_5_6_5: case C.UNSIGNED_SHORT_4_4_4_4: case C.UNSIGNED_SHORT_5_5_5_1:
      return comps * 2;
    default: // UNSIGNED_INT, INT, UNSIGNED_INT_24_8, 2_10_10_10_REV, 10F_11F_11F_REV, 5_9_9_9_REV, FLOAT, FLOAT_32_UNSIGNED_INT_24_8_REV
      return 4;
  }
}

/**
 * Read one source texel as [r,g,b,a] in the DESTINATION domain:
 * domain 0 = normalized (0..1), 1 = float (raw), 2 = integer (raw).
 */
function readSourceTexel(dv: DataView, byteOff: number, format: GLenum, type: GLenum, domain: 0 | 1 | 2, out: Float32Array): void {
  const comps = srcComponents(format);
  const norm = domain === 0;
  const raw = domain !== 0;
  switch (type) {
    case C.UNSIGNED_BYTE:
      for (let i = 0; i < comps; i++) out[i] = norm ? dv.getUint8(byteOff + i) / 255 : dv.getUint8(byteOff + i);
      break;
    case C.BYTE:
      for (let i = 0; i < comps; i++) out[i] = norm ? dv.getInt8(byteOff + i) / 127 : dv.getInt8(byteOff + i);
      break;
    case C.UNSIGNED_SHORT:
      for (let i = 0; i < comps; i++) out[i] = norm ? dv.getUint16(byteOff + i * 2, true) / 65535 : dv.getUint16(byteOff + i * 2, true);
      break;
    case C.SHORT:
      for (let i = 0; i < comps; i++) out[i] = norm ? dv.getInt16(byteOff + i * 2, true) / 32767 : dv.getInt16(byteOff + i * 2, true);
      break;
    case C.UNSIGNED_INT:
      for (let i = 0; i < comps; i++) out[i] = norm ? dv.getUint32(byteOff + i * 4, true) / 4294967295 : dv.getUint32(byteOff + i * 4, true);
      break;
    case C.INT:
      for (let i = 0; i < comps; i++) out[i] = norm ? dv.getInt32(byteOff + i * 4, true) / 2147483647 : dv.getInt32(byteOff + i * 4, true);
      break;
    case C.FLOAT:
      for (let i = 0; i < comps; i++) out[i] = dv.getFloat32(byteOff + i * 4, true);
      break;
    case C.HALF_FLOAT: case CExt.HALF_FLOAT_OES:
      for (let i = 0; i < comps; i++) out[i] = halfToFloat(dv.getUint16(byteOff + i * 2, true));
      break;
    case C.UNSIGNED_SHORT_5_6_5: {
      const v = dv.getUint16(byteOff, true);
      out[0] = ((v >> 11) & 0x1f) / 31; out[1] = ((v >> 5) & 0x3f) / 63; out[2] = (v & 0x1f) / 31; out[3] = 1;
      break;
    }
    case C.UNSIGNED_SHORT_4_4_4_4: {
      const v = dv.getUint16(byteOff, true);
      out[0] = ((v >> 12) & 0xf) / 15; out[1] = ((v >> 8) & 0xf) / 15; out[2] = ((v >> 4) & 0xf) / 15; out[3] = (v & 0xf) / 15;
      break;
    }
    case C.UNSIGNED_SHORT_5_5_5_1: {
      const v = dv.getUint16(byteOff, true);
      out[0] = ((v >> 11) & 0x1f) / 31; out[1] = ((v >> 6) & 0x1f) / 31; out[2] = ((v >> 1) & 0x1f) / 31; out[3] = v & 1;
      break;
    }
    case C.UNSIGNED_INT_2_10_10_10_REV: {
      const v = dv.getUint32(byteOff, true);
      const r = (v >>> 22) & 0x3ff, g = (v >>> 12) & 0x3ff, b = (v >>> 2) & 0x3ff, a = v & 0x3;
      if (norm) { out[0] = r / 1023; out[1] = g / 1023; out[2] = b / 1023; out[3] = a / 3; }
      else { out[0] = r; out[1] = g; out[2] = b; out[3] = a; }
      break;
    }
    case C.UNSIGNED_INT_10F_11F_11F_REV: {
      const v = dv.getUint32(byteOff, true);
      out[0] = float10ToFloat(v >>> 22); out[1] = float11ToFloat((v >>> 11) & 0x7ff); out[2] = float11ToFloat(v & 0x7ff); out[3] = 1;
      break;
    }
    case C.UNSIGNED_INT_5_9_9_9_REV: {
      const [r, g, b] = rgb9e5ToFloat(dv.getUint32(byteOff, true));
      out[0] = r; out[1] = g; out[2] = b; out[3] = 1;
      break;
    }
    case C.UNSIGNED_INT_24_8: { // 0x84fa (same value as UNSIGNED_INT_24_8_WEBGL)
      const v = dv.getUint32(byteOff, true);
      out[0] = norm ? ((v >>> 8) & 0xffffff) / 0xffffff : (v >>> 8) & 0xffffff;
      out[1] = v & 0xff; // stencil (used only by depth-stencil destinations)
      out[2] = 0; out[3] = 1;
      break;
    }
    case C.FLOAT_32_UNSIGNED_INT_24_8_REV: {
      out[0] = dv.getFloat32(byteOff, true);
      out[1] = dv.getUint8(byteOff + 4); // stencil
      out[2] = 0; out[3] = 1;
      break;
    }
    default:
      out[0] = out[1] = out[2] = 0; out[3] = 1;
  }
  // Component fill for formats with fewer than 4 components.
  switch (format) {
    case C.RGB: case C.RGB_INTEGER: out[3] = 1; break;
    case C.RG: case C.RG_INTEGER: out[2] = 0; out[3] = 1; break;
    case C.LUMINANCE: out[1] = out[0]; out[2] = out[0]; out[3] = 1; break;
    case C.LUMINANCE_ALPHA: out[1] = out[0]; out[2] = out[0]; break;
    case C.ALPHA: out[0] = 0; out[1] = 0; out[2] = 0; break;
    case C.RED: case C.RED_INTEGER: case C.DEPTH_COMPONENT: out[1] = 0; out[2] = 0; out[3] = 1; break;
    default: break;
  }
  void raw;
}

// ---------------------------------------------------------------------------
// Row/slice copy
// ---------------------------------------------------------------------------

const align4 = (v: number): number => (v + 3) & ~3;

interface CopyParams {
  src: DataView;
  srcRowBytes: number; // padded row stride (bytes)
  srcSkipPixels: number;
  srcBpp: number; // source bytes per texel
  srcFormat: GLenum;
  srcType: GLenum;
  domain: 0 | 1 | 2;
  flipY: boolean;
  premultiply: boolean;
  write: (dst: ArrayBufferView, off: number, r: number, g: number, b: number, a: number) => void;
  dstBpp: number;
  dstStencil?: Uint8Array; // when destination is depth-stencil
}

/** Copy `height` rows of `width` texels into one destination view at (xoff, yoff). */
function copyRows(
  p: CopyParams,
  dst: ArrayBufferView,
  dstW: number,
  xoff: number,
  yoff: number,
  width: number,
  height: number,
  srcRow0: number,
  dstZOffset: number,
): void {
  const out = new Float32Array(4);
  for (let r = 0; r < height; r++) {
    const srcRow = srcRow0 + (p.flipY ? height - 1 - r : r);
    const dstY = yoff + r;
    const srcBase = srcRow * p.srcRowBytes + p.srcSkipPixels * p.srcBpp;
    const dstBase = (dstZOffset + dstY * dstW + xoff) * p.dstBpp;
    for (let x = 0; x < width; x++) {
      readSourceTexel(p.src, srcBase + x * p.srcBpp, p.srcFormat, p.srcType, p.domain, out);
      if (p.premultiply) { out[0] *= out[3]; out[1] *= out[3]; out[2] *= out[3]; }
      if (p.dstStencil !== undefined && p.srcFormat === C.DEPTH_STENCIL) {
        p.write(dst, dstBase + x * p.dstBpp, out[0], 0, 0, 0);
        p.dstStencil[dstY * dstW + xoff + x] = out[1] & 0xff;
      } else {
        p.write(dst, dstBase + x * p.dstBpp, out[0], out[1], out[2], out[3]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Image metadata + completeness
// ---------------------------------------------------------------------------

/** Placeholder info for a not-yet-allocated _image (RGBA8 descriptor). */
const PLACEHOLDER_INFO: PixelFormatInfo = toPixelFormatInfo(SPECS.get(C.RGBA8) as FormatSpec);

function ensureImage(texture: WebGLTexture, target: GLenum): NonNullable<WebGLTexture['_image']> {
  if (!texture._image) {
    texture._image = {
      target: canonTarget(target),
      internalFormat: 0,
      info: PLACEHOLDER_INFO,
      width: 0,
      height: 0,
      depth: 1,
      levels: [],
      baseLevel: 0,
      maxLevel: 0,
      immutable: false,
      complete: false,
    };
  }
  return texture._image;
}

const isPow2 = (v: number): boolean => v > 0 && (v & (v - 1)) === 0;

/** Recompute _image completeness + base/max level after any mutation. */
function updateCompleteness(texture: WebGLTexture, version: 1 | 2): void {
  const img = texture._image;
  if (!img) return;
  const base = Math.max(0, texture._params[0x813c] | 0);
  const maxParam = Math.max(0, texture._params[0x813d] | 0);
  img.maxLevel = Math.min(maxParam, img.levels.length - 1);
  if (base > img.maxLevel) {
    // BASE_LEVEL beyond the clamped MAX_LEVEL → incomplete per spec.
    img.baseLevel = base;
    img.complete = false;
    return;
  }
  img.baseLevel = base;
  const baseLevel = img.levels[base];
  if (!baseLevel || baseLevel.width < 1 || baseLevel.height < 1) {
    img.complete = false;
    return;
  }
  const isCube = img.target === C.TEXTURE_CUBE_MAP;
  const isArray = img.target === C.TEXTURE_2D_ARRAY;
  const is3D = img.target === C.TEXTURE_3D;
  const minFilter = texture._params[0x2801];
  const needsMips = minFilter !== C.NEAREST && minFilter !== C.LINEAR;
  // Level range that must be defined when mipmaps are required.
  const maxDim = Math.max(baseLevel.width, baseLevel.height, is3D ? baseLevel.depth : 1);
  const chainMax = base + Math.floor(Math.log2(maxDim));
  const requiredMax = needsMips ? Math.min(maxParam, chainMax) : base;
  let ok = true;
  for (let l = base; l <= requiredMax; l++) {
    const lv = img.levels[l];
    if (!lv) { ok = false; break; }
    const shift = l - base;
    const expW = Math.max(1, baseLevel.width >> shift);
    const expH = Math.max(1, baseLevel.height >> shift);
    const expD = isArray ? baseLevel.depth : is3D ? Math.max(1, baseLevel.depth >> shift) : 1;
    if (lv.width !== expW || lv.height !== expH || lv.depth !== expD) { ok = false; break; }
    if (isCube && lv.data.length < 6) { ok = false; break; }
    if (isCube) {
      for (let f = 0; f < 6; f++) {
        if (!lv.data[f]) { ok = false; break; }
      }
      if (!ok) break;
    }
  }
  if (ok && version === 1 && (!isPow2(baseLevel.width) || !isPow2(baseLevel.height))) {
    // WebGL1 NPOT: only NEAREST/LINEAR minification with CLAMP_TO_EDGE wrap.
    if (needsMips || texture._params[0x2802] !== C.CLAMP_TO_EDGE || texture._params[0x2803] !== C.CLAMP_TO_EDGE) {
      ok = false;
    }
  }
  img.complete = ok;
}

/** Per-level origin (format, type) for WebGL1 texSubImage match checks. */
const levelOrigins = new WeakMap<WebGLTexture, Map<number, { format: number; type: number }>>();

function recordLevelOrigin(texture: WebGLTexture, level: number, format: GLenum, type: GLenum): void {
  let m = levelOrigins.get(texture);
  if (!m) { m = new Map(); levelOrigins.set(texture, m); }
  m.set(level, { format, type });
}

/** Origin (format, type) of a level — used by api/teximage.ts (WebGL1 texSubImage). */
export function getLevelOrigin(texture: WebGLTexture, level: number): { format: number; type: number } | undefined {
  return levelOrigins.get(texture)?.get(level);
}

/** True when a level record exists for (target face, level). */
export function hasTextureLevel(texture: WebGLTexture, target: GLenum, level: number): boolean {
  const img = texture._image;
  if (!img) return false;
  const lv = img.levels[level];
  if (!lv) return false;
  const fi = cubeFaceIndex(target);
  if (fi >= 0) return lv.data[fi] !== undefined;
  return true;
}

// ---------------------------------------------------------------------------
// Engine entry points (called by api/teximage.ts after validation)
// ---------------------------------------------------------------------------

/** Shared upload path: convert source pixels into a level. */
function copyPixelsIntoLevel(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: number,
  spec: FormatSpec,
  format: GLenum,
  type: GLenum,
  pixels: TexImageSourceArg,
  source: unknown,
  width: number,
  height: number,
  depth: number,
  xoffset: number,
  yoffset: number,
  zoffset: number,
): void {
  const img = texture._image as NonNullable<WebGLTexture['_image']>;
  const levelData = img.levels[level];
  if (!levelData) return;
  const isCube = img.target === C.TEXTURE_CUBE_MAP;
  const face = cubeFaceIndex(target);
  const views: ArrayBufferView[] = face >= 0 ? [levelData.data[face]] : levelData.data;
  const dstBpp = spec.bytesPerPixel;

  // DOM source: decode via present/ (stub → zero-fill fallback, documented gap).
  if (pixels === null || pixels === undefined) return; // zero-filled allocation
  if (typeof pixels !== 'number' && !ArrayBuffer.isView(pixels) && source !== undefined) {
    try {
      const res = decodeImageSource(source as never) as { ok: boolean; image?: { width: number; height: number; data: Uint8ClampedArray } };
      if (res && res.ok && res.image) {
        const im = res.image;
        const dv = new DataView(im.data.buffer, im.data.byteOffset, im.data.byteLength);
        const p: CopyParams = {
          src: dv,
          srcRowBytes: im.width * 4,
          srcSkipPixels: 0,
          srcBpp: 4,
          srcFormat: C.RGBA,
          srcType: C.UNSIGNED_BYTE,
          domain: spec.isInteger ? 2 : spec.isFloat && !spec.isDepth ? 1 : 0,
          flipY: ctx._state.pixelStore.unpack.flipY,
          premultiply: ctx._state.pixelStore.unpack.premultiplyAlpha,
          write: spec.pack,
          dstBpp,
          dstStencil: levelData.stencilData,
        };
        // DOM sources ignore ROW_LENGTH/SKIP_* (WebGL2 spec §3.7); the decoded
        // image is tightly packed RGBA8. depth is always 1 here (the API layer
        // rejects DOM sources for 3D/2D_ARRAY targets).
        copyRows(p, views[0], levelData.width, xoffset, yoffset, width, height, 0, 0);
        updateCompleteness(texture, ctx._version);
        return;
      }
    } catch {
      // present/ decode is a stub → zero-filled allocation (documented gap).
    }
    return;
  }

  // Client data or PBO offset.
  let srcView: ArrayBufferView;
  let baseOffset = 0;
  if (typeof pixels === 'number') {
    const buf = ctx._state.pixelUnpackBuffer;
    if (!buf || !buf._data) return;
    srcView = new Uint8Array(buf._data);
    baseOffset = pixels;
  } else {
    srcView = pixels as ArrayBufferView;
  }
  if (baseOffset > srcView.byteLength) return; // defensive (API validates first)
  const s = ctx._state.pixelStore.unpack;
  const srcBpp = srcBytesPerTexel(format, type);
  const srcRowLength = s.rowLength > 0 ? s.rowLength : width;
  const srcRowBytes = align4(srcRowLength * srcBpp);
  const domain: 0 | 1 | 2 = spec.isInteger ? 2 : spec.isFloat && !spec.isDepth ? 1 : 0;
  const dv = new DataView(srcView.buffer, srcView.byteOffset + baseOffset, srcView.byteLength - baseOffset);
  const p: CopyParams = {
    src: dv,
    srcRowBytes,
    srcSkipPixels: s.skipPixels,
    srcBpp,
    srcFormat: format,
    srcType: type,
    domain,
    flipY: s.flipY,
    premultiply: false, // client arrays are not premultiplied by the API
    write: spec.pack,
    dstBpp,
    dstStencil: levelData.stencilData,
  };
  const srcImageHeight = s.imageHeight > 0 ? s.imageHeight : height;
  const srcSkipImages = s.skipImages;
  for (let z = 0; z < depth; z++) {
    const view = views[0];
    const srcRow0 = s.skipRows + (srcSkipImages + z) * srcImageHeight;
    copyRows(p, view, levelData.width, xoffset, yoffset, width, height, srcRow0, z * levelData.width * levelData.height);
  }
}

/** texImage2D/texImage3D storage mutation (target validated by caller). */
export function uploadTexImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  border: GLint,
  format: GLenum,
  type: GLenum,
  pixels: TexImageSourceArg,
  source?: unknown,
): void {
  void border;
  if (texture._immutable) return;
  const spec = resolveStorageSpec(internalformat);
  if (!spec) return;
  const img = ensureImage(texture, target);
  const isCube = img.target === C.TEXTURE_CUBE_MAP;
  const levelData = allocLevel(spec, width, height, depth, isCube);
  if (isCube) {
    // Only the uploaded face is defined (cube completeness needs all 6).
    const face = cubeFaceIndex(target);
    for (let f = 0; f < 6; f++) if (f !== face) levelData.data[f] = undefined as unknown as ArrayBufferView;
  }
  img.levels[level] = levelData;
  texture._internalFormat = internalformat;
  texture._compressed = false;
  img.internalFormat = internalformat;
  img.info = toPixelFormatInfo(spec);
  img.target = canonTarget(target);
  img.immutable = texture._immutable;
  if (level === 0) {
    img.width = width;
    img.height = height;
    img.depth = isCube ? 6 : depth;
  }
  recordLevelOrigin(texture, level, format, type);
  copyPixelsIntoLevel(ctx, texture, target, level, spec, format, type, pixels, source, width, height, depth, 0, 0, 0);
  updateCompleteness(texture, ctx._version);
}

/** texSubImage2D/3D partial update. */
export function uploadTexSubImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  xoffset: GLint, yoffset: GLint, zoffset: GLint,
  width: GLsizei, height: GLsizei, depth: GLsizei,
  format: GLenum, type: GLenum,
  pixels: TexImageSourceArg,
  source?: unknown,
): void {
  const img = texture._image;
  if (!img) return;
  const levelData = img.levels[level];
  if (!levelData) return;
  const spec = resolveStorageSpec(img.internalFormat);
  if (!spec) return;
  copyPixelsIntoLevel(ctx, texture, target, level, spec, format, type, pixels, source, width, height, depth, xoffset, yoffset, zoffset);
  updateCompleteness(texture, ctx._version);
}

/** texStorage2D/3D: allocate immutable mip chain. */
export function allocateImmutableStorage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  levels: GLsizei,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
): void {
  void ctx;
  if (texture._immutable) return;
  const spec = resolveStorageSpec(internalformat);
  if (!spec) return;
  const img = ensureImage(texture, target);
  const isCube = img.target === C.TEXTURE_CUBE_MAP;
  img.levels = [];
  let w = width;
  let h = height;
  let d = depth;
  for (let l = 0; l < levels; l++) {
    img.levels[l] = allocLevel(spec, w, h, d, isCube);
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
    if (!isCube) d = Math.max(1, d >> 1);
  }
  texture._immutable = true;
  texture._internalFormat = internalformat;
  texture._compressed = false;
  img.immutable = true;
  img.internalFormat = internalformat;
  img.info = toPixelFormatInfo(spec);
  img.target = canonTarget(target);
  img.width = width;
  img.height = height;
  img.depth = isCube ? 6 : depth;
  updateCompleteness(texture, ctx._version);
}

/** Copy the read framebuffer rect (x, y, w, h) into a level (GL coords, y-up). */
function copyFromReadSurface(
  ctx: WebGLRenderingContext,
  levelData: TextureLevel,
  face: number,
  spec: FormatSpec,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  let surface: Surface | null = null;
  try {
    surface = resolveReadSurface(ctx);
  } catch {
    surface = null;
  }
  if (!surface) return; // framebuffer-util stub / incomplete FBO → API reports the error
  const view = levelData.data[face >= 0 ? face : 0];
  const sW = surface.width;
  const sH = surface.height;
  const dstBpp = spec.bytesPerPixel;
  const out = new Float32Array(4);
  try {
    const srcIsRGBA8 = surface.format === C.RGBA8 || surface.format === C.RGBA;
    const srcData = surface.data as Uint8Array;
    for (let dy = 0; dy < height; dy++) {
      const sy = y + dy;
      if (sy < 0 || sy >= sH) continue;
      for (let dx = 0; dx < width; dx++) {
        const sx = x + dx;
        if (sx < 0 || sx >= sW) continue;
        if (srcIsRGBA8) {
          const o = (sy * sW + sx) * 4;
          out[0] = srcData[o] / 255; out[1] = srcData[o + 1] / 255; out[2] = srcData[o + 2] / 255; out[3] = srcData[o + 3] / 255;
        } else {
          surface.info.decode(srcData, (sy * sW + sx) * surface.info.bytesPerPixel, out);
        }
        spec.pack(view, (dy * levelData.width + dx) * dstBpp, out[0], out[1], out[2], out[3]);
      }
    }
  } catch {
    // Surface decode unavailable (raster formats stub) → leave zero-filled.
  }
}

/** copyTexImage2D / copyTexSubImage2D / copyTexSubImage3D. */
export function copyTexImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  x: GLint, y: GLint, width: GLsizei, height: GLsizei, border: GLint,
): void {
  void border;
  if (texture._immutable) return;
  const spec = resolveStorageSpec(internalformat);
  if (!spec) return;
  const img = ensureImage(texture, target);
  const isCube = img.target === C.TEXTURE_CUBE_MAP;
  const levelData = allocLevel(spec, width, height, 1, isCube);
  if (isCube) {
    const face = cubeFaceIndex(target);
    for (let f = 0; f < 6; f++) if (f !== face) levelData.data[f] = undefined as unknown as ArrayBufferView;
  }
  img.levels[level] = levelData;
  texture._internalFormat = internalformat;
  texture._compressed = false;
  img.internalFormat = internalformat;
  img.info = toPixelFormatInfo(spec);
  img.target = canonTarget(target);
  if (level === 0) {
    img.width = width;
    img.height = height;
    img.depth = isCube ? 6 : 1;
  }
  copyFromReadSurface(ctx, img.levels[level], cubeFaceIndex(target), spec, x, y, width, height);
  updateCompleteness(texture, ctx._version);
}

export function copyTexSubImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  xoffset: GLint, yoffset: GLint, zoffset: GLint,
  x: GLint, y: GLint, width: GLsizei, height: GLsizei,
): void {
  const img = texture._image;
  if (!img) return;
  const levelData = img.levels[level];
  if (!levelData) return;
  const spec = resolveStorageSpec(img.internalFormat);
  if (!spec) return;
  // Copy into the (xoffset, yoffset) region of a temporary of the sub size,
  // then blit — simpler: read directly with a dst offset via a temp level.
  const tmp: TextureLevel = { width, height, depth: 1, data: [new (spec.ctor as new (n: number) => ArrayBufferView)((width * height * spec.bytesPerPixel) / spec.bytesPerElement)] };
  if (spec.isStencil) tmp.stencilData = new Uint8Array(width * height);
  copyFromReadSurface(ctx, tmp, -1, spec, x, y, width, height);
  const face = cubeFaceIndex(target);
  const view = levelData.data[face >= 0 ? face : 0];
  const srcView = tmp.data[0];
  const elemsPerTexel = spec.bytesPerPixel / spec.bytesPerElement;
  for (let dy = 0; dy < height; dy++) {
    for (let dx = 0; dx < width; dx++) {
      const srcElem = ((dy * width + dx) * spec.bytesPerPixel) / spec.bytesPerElement;
      const dstElem = (zoffset * levelData.width * levelData.height + (yoffset + dy) * levelData.width + xoffset + dx) * spec.bytesPerPixel / spec.bytesPerElement;
      for (let e = 0; e < elemsPerTexel; e++) {
        (view as unknown as { [i: number]: number })[dstElem + e] = (srcView as unknown as { [i: number]: number })[srcElem + e];
      }
    }
  }
  if (spec.isStencil && levelData.stencilData && tmp.stencilData) {
    for (let dy = 0; dy < height; dy++) {
      for (let dx = 0; dx < width; dx++) {
        levelData.stencilData[(zoffset * levelData.height + yoffset + dy) * levelData.width + xoffset + dx] = tmp.stencilData[dy * width + dx];
      }
    }
  }
  updateCompleteness(texture, ctx._version);
}

/** compressedTexImage2D/3D + compressedTexSubImage2D/3D — no compressed
 *  format is implemented: the API layer always generates INVALID_ENUM, so the
 *  engine is a defensive no-op. */
export function compressedTexImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  width: GLsizei, height: GLsizei, depth: GLsizei,
  border: GLint,
  data: ArrayBufferView,
  sub: boolean,
  xoffset: GLint, yoffset: GLint, zoffset: GLint,
): void {
  void ctx; void texture; void target; void level; void internalformat;
  void width; void height; void depth; void border; void data; void sub;
  void xoffset; void yoffset; void zoffset;
}

/** generateMipmap: build the full mip chain from the base level (2×2 box filter). */
export function generateMipmap(ctx: WebGLRenderingContext, texture: WebGLTexture, target: GLenum): void {
  const img = texture._image;
  if (!img) return;
  const base = img.levels[0];
  if (!base) return;
  const spec = resolveStorageSpec(img.internalFormat);
  if (!spec) return;
  const isCube = img.target === C.TEXTURE_CUBE_MAP;
  const isArray = img.target === C.TEXTURE_2D_ARRAY;
  const maxDim = Math.max(img.width, img.height, !isCube && !isArray ? img.depth : 1);
  const levels = Math.floor(Math.log2(maxDim)) + 1;
  const out = new Float32Array(4);
  let w = img.width;
  let h = img.height;
  let d = isCube ? 6 : img.depth;
  for (let l = 1; l < levels; l++) {
    const nw = Math.max(1, w >> 1);
    const nh = Math.max(1, h >> 1);
    const nd = isArray ? d : Math.max(1, d >> 1);
    const levelData = allocLevel(spec, nw, nh, isCube ? 1 : nd, isCube);
    img.levels[l] = levelData;
    const prev = img.levels[l - 1];
    const faceCount = isCube ? 6 : 1;
    for (let f = 0; f < faceCount; f++) {
      const srcView = prev.data[f];
      const dstView = levelData.data[f];
      for (let z = 0; z < (isCube ? 1 : nd); z++) {
        for (let y = 0; y < nh; y++) {
          for (let x = 0; x < nw; x++) {
            let accR = 0, accG = 0, accB = 0, accA = 0, n = 0;
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                const sx = Math.min(w - 1, x * 2 + dx);
                const sy = Math.min(h - 1, y * 2 + dy);
                const srcOff = ((z * h + sy) * w + sx) * spec.bytesPerPixel;
                spec.unpack(srcView, srcOff, out);
                accR += out[0]; accG += out[1]; accB += out[2]; accA += out[3]; n++;
              }
            }
            const dstOff = ((z * nh + y) * nw + x) * spec.bytesPerPixel;
            spec.pack(dstView, dstOff, accR / n, accG / n, accB / n, accA / n);
          }
        }
      }
    }
    w = nw;
    h = nh;
    d = nd;
  }
  void ctx;
  updateCompleteness(texture, ctx._version);
}
