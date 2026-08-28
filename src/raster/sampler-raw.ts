/**
 * sampler-raw.ts — raw texel-read layer of the texture sampler (contract §3).
 *
 * Bit-preserving texel reads (readTexel / texelFetch) plus the shared decode
 * helpers and the int-transport scratch they use. Split out of sampler.ts so
 * the sampling core stays under the line limit. This module imports nothing
 * from sampler.ts / sampler-env.ts (the import graph stays acyclic).
 *
 * Semantics (see the sampler.ts header for the full picture):
 *  - readTexel: raw texel with sRGB linearization (sampling path);
 *    out-of-range / missing level → (0,0,0,1) (all-zero for integer formats).
 *  - texelFetch: raw texel, no filtering/wrap/LOD/sRGB linearization;
 *    out-of-range or incomplete → (0,0,0,0).
 *  - Integer formats: raw int bits travel in the float slot (read via
 *    outInt/outUint reinterprets); missing components 0.
 *
 * No per-texel allocation: results land in caller-provided `out` scratch.
 */

import type { GLenum } from './gl-enums';
import {
  TEXTURE_CUBE_MAP, TEXTURE_3D, TEXTURE_2D_ARRAY,
  ALPHA, LUMINANCE, LUMINANCE_ALPHA,
} from './gl-enums';
import { halfToFloat, sRGBToLinear } from './formats';
import type { PixelFormatInfo, StorageKind } from './formats';
import type { TextureImage } from './types';

/* ================================================================== */
/* Module-level scratch (shared by the generic core; sampling is        */
/* strictly sequential so no reentrancy hazard exists).                 */
/* ================================================================== */

/** Bit-preserving int transport: set _i32/_u32, copy _f32[0] into out. */
const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);
const _u32 = new Uint32Array(_f32.buffer);

/* ================================================================== */
/* Raw texel reads                                                     */
/* ================================================================== */

/** Writes the incomplete/null result: (0,0,0,1), or all-zero for integer formats. */
export function writeDefault(out: Float32Array, isInteger: boolean): void {
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;
  out[3] = isInteger ? 0 : 1;
}

/** Bytes per component for the uniform-width storage kinds. */
function compSize(storage: StorageKind): number {
  switch (storage) {
    case 'u8': case 'i8': return 1;
    case 'u16': case 'i16': case 'f16': return 2;
    default: return 4; // u32, i32, f32
  }
}

/** True when every component occupies compSize bytes (i.e. not packed 565/4444/5551/RGB10_A2/R11F/RGB9_E5). */
function isUniformStorage(info: PixelFormatInfo): boolean {
  return info.bytesPerPixel === info.components * compSize(info.storage);
}

/** Normalization divisor for a uniform fixed-point storage kind (GLES 3.0 §3.8.17). */
function normDivisor(storage: StorageKind): number {
  switch (storage) {
    case 'u8': return 255;
    case 'i8': return 127;
    case 'u16': return 65535;
    case 'i16': return 32767;
    default: return 1; // u32/i32 normalized storage does not occur (packed formats use decode)
  }
}

/** Reads one component at byte offset `o` (view-relative indexing). */
function readComp(entry: ArrayBufferView, storage: StorageKind, o: number): number {
  switch (storage) {
    case 'u8': return (entry as Uint8Array)[o];
    case 'i8': return (entry as Int8Array)[o];
    case 'u16': return (entry as Uint16Array)[o >> 1];
    case 'i16': return (entry as Int16Array)[o >> 1];
    case 'u32': return (entry as Uint32Array)[o >> 2];
    case 'i32': return (entry as Int32Array)[o >> 2];
    case 'f32': return (entry as Float32Array)[o >> 2];
    case 'f16': return halfToFloat((entry as Uint16Array)[o >> 1]);
  }
}

/** Reads an INTEGER-format texel bit-preserving: the raw int bits travel in the float slot (env.outInt/outUint reveal them); missing components → 0. */
function readInts(entry: ArrayBufferView, storage: StorageKind, byteOffset: number, n: number, out: Float32Array): void {
  switch (storage) {
    case 'u8': {
      const d = entry as Uint8Array;
      for (let c = 0; c < n; c++) { _u32[0] = d[byteOffset + c]; out[c] = _f32[0]; }
      break;
    }
    case 'i8': {
      const d = entry as Int8Array;
      for (let c = 0; c < n; c++) { _i32[0] = d[byteOffset + c]; out[c] = _f32[0]; }
      break;
    }
    case 'u16': {
      const d = entry as Uint16Array;
      const o = byteOffset >> 1;
      for (let c = 0; c < n; c++) { _u32[0] = d[o + c]; out[c] = _f32[0]; }
      break;
    }
    case 'i16': {
      const d = entry as Int16Array;
      const o = byteOffset >> 1;
      for (let c = 0; c < n; c++) { _i32[0] = d[o + c]; out[c] = _f32[0]; }
      break;
    }
    case 'u32': {
      const d = entry as Uint32Array;
      const o = byteOffset >> 2;
      for (let c = 0; c < n; c++) { _u32[0] = d[o + c]; out[c] = _f32[0]; }
      break;
    }
    case 'i32': {
      const d = entry as Int32Array;
      const o = byteOffset >> 2;
      for (let c = 0; c < n; c++) { _i32[0] = d[o + c]; out[c] = _f32[0]; }
      break;
    }
    default: // f32/f16 storage with an integer format: decode fallback (defensive)
      for (let c = 0; c < n; c++) out[c] = readComp(entry, storage, byteOffset + c * compSize(storage));
  }
  for (let c = n; c < 4; c++) out[c] = 0;
}

/** Decodes one texel from a level entry. `linearize` applies sRGB→linear (sampling; texelFetch must not). */
function readFromEntry(info: PixelFormatInfo, entry: ArrayBufferView, byteOffset: number, linearize: boolean, out: Float32Array): void {
  if (info.isInteger) {
    readInts(entry, info.storage, byteOffset, info.components, out);
    return;
  }
  if (info.isDepth) {
    let d: number;
    if (info.storage === 'f32') {
      d = (entry as Float32Array)[byteOffset >> 2];
    } else {
      info.decode(entry, byteOffset, out);
      d = out[0];
    }
    out[0] = d;
    out[1] = d;
    out[2] = d;
    out[3] = 1;
    return;
  }
  if (isUniformStorage(info)) {
    // WebGL1 unsized luminance/alpha formats have non-identity channel maps
    // (ALPHA → (0,0,0,a); LUMINANCE → (l,l,l,1); LUMINANCE_ALPHA → (l,l,l,a)).
    // The uniform-storage fast path below only handles identity RGBA layouts
    // (missing components 0/1), which would mis-sample these; route them
    // through the format's decode instead (same /div normalization, writes
    // the full 4-channel expansion — no allocation).
    if (info.format === ALPHA || info.format === LUMINANCE || info.format === LUMINANCE_ALPHA) {
      info.decode(entry, byteOffset, out);
    } else {
      const n = info.components;
      const cs = compSize(info.storage);
      if (info.normalized) {
        // Fixed-point normalized: decode to 0..1 (snorm −1..1, clamped at −1).
        const div = normDivisor(info.storage);
        const isSigned = info.isSigned;
        for (let c = 0; c < n; c++) {
          const raw = readComp(entry, info.storage, byteOffset + c * cs);
          const x = raw / div;
          out[c] = isSigned ? (x < -1 ? -1 : x) : x;
        }
      } else {
        // Float storage (or raw fixed-point defensive path): passthrough.
        for (let c = 0; c < n; c++) out[c] = readComp(entry, info.storage, byteOffset + c * cs);
      }
      for (let c = n; c < 4; c++) out[c] = c === 3 ? 1 : 0;
    }
  } else {
    info.decode(entry, byteOffset, out);
  }
  if (linearize && info.isSRGB) {
    out[0] = sRGBToLinear(out[0]);
    out[1] = sRGBToLinear(out[1]);
    out[2] = sRGBToLinear(out[2]);
  }
}

/**
 * Raw texel read. Out-of-range (level/face/x/y/z) → (0,0,0,1) (all-zero for
 * integer formats). Cube: `face` indexes data[face]; 3D/2D_ARRAY: `z`
 * indexes data[z]. sRGB is linearized (sampling path).
 */
export function readTexel(
  img: TextureImage, level: number, face: number, x: number, y: number, z: number,
  out: Float32Array,
): void {
  if (level < 0 || level >= img.levels.length) {
    writeDefault(out, img.info.isInteger);
    return;
  }
  const lv = img.levels[level];
  if (x < 0 || x >= lv.width || y < 0 || y >= lv.height) {
    writeDefault(out, img.info.isInteger);
    return;
  }
  let entry: ArrayBufferView | undefined;
  if (img.target === TEXTURE_CUBE_MAP) {
    if (face < 0 || face >= lv.data.length) { writeDefault(out, img.info.isInteger); return; }
    entry = lv.data[face];
  } else if (img.target === TEXTURE_3D || img.target === TEXTURE_2D_ARRAY) {
    if (z < 0 || z >= lv.data.length) { writeDefault(out, img.info.isInteger); return; }
    entry = lv.data[z];
  } else {
    entry = lv.data[0];
    if (!entry) { writeDefault(out, img.info.isInteger); return; }
  }
  readFromEntry(img.info, entry, (y * lv.width + x) * img.info.bytesPerPixel, true, out);
}

/**
 * texelFetch: raw texel at integer (x,y,z) + explicit level; no filtering,
 * wrap, LOD or sRGB linearization. `z` = 3D slice / array layer / 0.
 * Cube not fetchable. Out-of-range or incomplete → (0,0,0,0).
 */
export function texelFetch(
  img: TextureImage, x: number, y: number, z: number,
  level: number, out: Float32Array,
): void {
  if (!img.complete || level < 0 || level >= img.levels.length) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    return;
  }
  const lv = img.levels[level];
  if (x < 0 || x >= lv.width || y < 0 || y >= lv.height) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    return;
  }
  let entry: ArrayBufferView | undefined;
  if (img.target === TEXTURE_3D || img.target === TEXTURE_2D_ARRAY) {
    if (z < 0 || z >= lv.data.length) {
      out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
      return;
    }
    entry = lv.data[z];
  } else {
    entry = lv.data[0];
  }
  if (!entry) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    return;
  }
  readFromEntry(img.info, entry, (y * lv.width + x) * img.info.bytesPerPixel, false, out);
}
