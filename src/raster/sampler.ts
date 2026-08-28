/**
 * sampler.ts — texture sampling core (contract §3).
 *
 * Two entry levels:
 *  1. Generic functions (sampleTexture / sampleTextureLod / sampleTextureShadow
 *     / texelFetch) — explicit TextureImage + effective SamplerState; used by
 *     gl/ and unit tests.
 *  2. TextureEnv (see types.ts) — per-draw unit table + codegen-facing
 *     methods (glsl/ compiles `ctx.tex.*` against them). Each method looks up
 *     its unit in a table pre-resolved ONCE per draw (binding → img/state +
 *     sampling plan; null = unbound/incomplete → default result), calls the
 *     core, and writes the result into env.out (float domain); the raw bits
 *     are also visible via env.outInt / env.outUint for integer textures
 *     (same ArrayBuffer). NO allocation: all results land in preallocated
 *     scratch.
 *
 * Fast paths (all mathematically identical to the generic path):
 *  - Single-level images (hi <= base): the level is fixed, so only the sign
 *    of λ matters (mag vs min filter) — computeLod's log2/clamps are skipped
 *    (lodGtZero; full λ fallback at the λ = 0 boundary / under anisotropy).
 *  - 1×1 levels: every filter/wrap maps all taps to the one texel — read it
 *    directly (no LOD, no filter selection at all).
 *  - Normalized u8 RGBA bilinear: direct Uint8Array taps + LUT decode
 *    (bit-identical numerics, no per-tap format dispatch/scratch).
 *  - Per-binding SamplePlan: filter-enum semantics are precomputed per draw
 *    (filter dispatch hoist).
 *
 * Semantics (GLES 3.0 §3.8.14 / WebGL, verified against deqp):
 *  - Index-space wrap (table 3.22): NEAREST i = wrap(⌊u·w⌋); LINEAR
 *    u′ = u·w − 1/2, i0 = wrap(⌊u′⌋), i1 = wrap(⌊u′⌋+1), α = frac(u′).
 *    REPEAT = mod; CLAMP_TO_EDGE = clamp to [0, size−1]; MIRRORED_REPEAT =
 *    mirror of (c mod 2·size).
 *  - Filters: NEAREST/LINEAR + all four mipmap combos. Implicit LOD:
 *    ρ = max(ρx,ρy), ρx = max over axes of |∂coord_i/∂x|·size_i;
 *    λ = clamp(log2(ρ) + bias, minLod, maxLod), bias clamped to ±2 (ES 3.0).
 *    Zero/absent derivatives → λ = minLod. textureLod passes the level
 *    directly. λ ≤ 0 → mag filter at base level; λ > 0 → min filter.
 *    NEAREST_MIPMAP_NEAREST: level = round(λ); NEAREST_MIPMAP_LINEAR:
 *    floor(λ) + frac(λ) blend between d and d+1.
 *  - EXT_texture_filter_anisotropic (2D + mipmap min filter only):
 *    ρ = max(ρx,ρy) / min(maxAnisotropy, max(ρx,ρy)/min(ρx,ρy)), guarded when
 *    min(ρx,ρy) == 0 (→ max(ρx,ρy)/maxAnisotropy).
 *  - Shadow samplers: compare the (quantized for fixed-point depth) reference
 *    against the depth texel with compareFunc → 0/1 per tap; LINEAR averages
 *    the per-tap results.
 *  - Cube maps: dominant-axis face selection (spec mapping, deqp tie-break).
 *    NEAREST clamps within the face. LINEAR is seamless: taps leaving the
 *    face are re-projected onto the adjacent face; a doubly-OOB (corner) tap
 *    is approximated by the mean of the other three taps.
 *  - sRGB (SRGB8/SRGB8_ALPHA8): decoded to linear on sample (readTexel);
 *    texelFetch does NOT linearize.
 *  - Integer formats (R8I..RGBA32UI): NEAREST taps only; results written
 *    bit-preserving (raw int bits travel in the float slot, read via
 *    env.outInt/outUint); missing components 0.
 *  - Incomplete texture / null unit → (0,0,0,1) (float) or (0,0,0,0)
 *    (integer). texelFetch: raw texel, no filtering/wrap; out of range or
 *    incomplete → (0,0,0,0).
 *
 * No per-texel allocation: all scratch lives in module-level buffers (generic
 * core) or per-env buffers (TextureEnv coordinate scratch).
 */

import type { GLenum } from './gl-enums';
import {
  NEAREST, LINEAR, NEAREST_MIPMAP_NEAREST, LINEAR_MIPMAP_NEAREST,
  NEAREST_MIPMAP_LINEAR, LINEAR_MIPMAP_LINEAR,
  REPEAT, CLAMP_TO_EDGE, MIRRORED_REPEAT,
  TEXTURE_2D, TEXTURE_3D, TEXTURE_CUBE_MAP, TEXTURE_2D_ARRAY,
  NEVER, LESS, EQUAL, LEQUAL, GREATER, NOTEQUAL, GEQUAL, ALWAYS,
  DEPTH_COMPONENT16, DEPTH_COMPONENT24, DEPTH24_STENCIL8,
  ALPHA, LUMINANCE, LUMINANCE_ALPHA,
} from './gl-enums';
import { halfToFloat, sRGBToLinear } from './formats';
import type { PixelFormatInfo, StorageKind } from './formats';
import type { SampleCoord, SamplerState, TextureEnv, TextureImage, TextureLevel, TextureUnitBinding } from './types';

/* ================================================================== */
/* Module-level scratch (shared by the generic core; sampling is        */
/* strictly sequential so no reentrancy hazard exists).                 */
/* ================================================================== */

/** Bit-preserving int transport: set _i32/_u32, copy _f32[0] into out. */
const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);
const _u32 = new Uint32Array(_f32.buffer);

/** One texel read (bilinear taps accumulate immediately). */
const _tap = new Float32Array(4);
/** filterCoordRaw results: [i0, frac, i1] per axis. */
const _c0 = new Float32Array(3);
const _c1 = new Float32Array(3);
const _c2 = new Float32Array(3);
/** Mipmap-linear two-level results. */
const _mipA = new Float32Array(4);
const _mipB = new Float32Array(4);
/** Cube seamless: reconstructed direction + face projection. */
const _dir = new Float32Array(3);
const _uv = new Float32Array(2);
/** Cube corner tap = average of the other three taps. */
const _cornerSum = new Float32Array(4);

/* ================================================================== */
/* Small helpers                                                       */
/* ================================================================== */

/** Writes the incomplete/null result: (0,0,0,1), or all-zero for integer formats. */
function writeDefault(out: Float32Array, isInteger: boolean): void {
  out[0] = 0;
  out[1] = 0;
  out[2] = 0;
  out[3] = isInteger ? 0 : 1;
}

/** Wraps one integer texel index (GLES 3.0 table 3.22; index-space wrap). */
function wrapIndex(c: number, wrap: GLenum, size: number): number {
  switch (wrap) {
    case REPEAT: {
      const m = c % size;
      return m < 0 ? m + size : m;
    }
    case MIRRORED_REPEAT: {
      const m2 = c % (2 * size);
      const r = m2 < 0 ? m2 + 2 * size : m2;
      return r < size ? r : 2 * size - 1 - r;
    }
    case CLAMP_TO_EDGE:
    default:
      return c < 0 ? 0 : (c >= size ? size - 1 : c);
  }
}

/**
 * Raw (unwrapped) bilinear footprint for one axis: u′ = u·size − 1/2;
 * writes [i0, frac, i1] into dst. i0/i1 may be out of [0, size−1]
 * (CLAMP_TO_EDGE callers wrap afterwards; cube seamless handles the OOB
 * taps by re-projection).
 */
function filterCoordRaw(u: number, size: number, dst: Float32Array): void {
  const up = u * size - 0.5;
  const i0 = Math.floor(up);
  dst[0] = i0;
  dst[1] = up - i0;
  dst[2] = i0 + 1;
}

/** Wrapped bilinear footprint (2D/3D filtering). */
function filterCoord(u: number, size: number, wrap: GLenum, dst: Float32Array): void {
  filterCoordRaw(u, size, dst);
  if (wrap !== CLAMP_TO_EDGE) {
    dst[0] = wrapIndex(dst[0], wrap, size);
    dst[2] = wrapIndex(dst[2], wrap, size);
  } else {
    let i = dst[0];
    dst[0] = i < 0 ? 0 : (i >= size ? size - 1 : i);
    i = dst[2];
    dst[2] = i < 0 ? 0 : (i >= size ? size - 1 : i);
  }
}

/** NEAREST texel index: i = wrap(⌊u·size⌋). */
function nearestIndex(u: number, size: number, wrap: GLenum): number {
  return wrapIndex(Math.floor(u * size), wrap, size);
}

/** Clamps a selected mip level into [base, hi] (hi ≥ base assumed; defensive otherwise). */
function clampLevel(v: number, base: number, hi: number): number {
  return v < base ? base : (v > hi ? hi : v);
}

/** 2D_ARRAY layer: round-to-nearest, clamped (GLES: l = clamp(⌊r+0.5⌋, 0, dt−1)). */
function clampLayer(v: number, depth: number): number {
  const l = Math.round(v);
  return l < 0 ? 0 : (l >= depth ? depth - 1 : l);
}

/** Shadow depth test. */
function depthFuncPass(func: GLenum, ref: number, d: number): boolean {
  switch (func) {
    case NEVER: return false;
    case LESS: return ref < d;
    case EQUAL: return ref === d;
    case LEQUAL: return ref <= d;
    case GREATER: return ref > d;
    case NOTEQUAL: return ref !== d;
    case GEQUAL: return ref >= d;
    case ALWAYS:
    default: return true;
  }
}

/** Quantizes the shadow reference to the fixed-point depth grid (float depth formats compare in float). */
function quantizeShadowRef(ref: number, internalFormat: GLenum): number {
  if (internalFormat === DEPTH_COMPONENT16) {
    const r = ref < 0 ? 0 : (ref > 1 ? 1 : ref);
    return Math.round(r * 65535) / 65535;
  }
  if (internalFormat === DEPTH_COMPONENT24 || internalFormat === DEPTH24_STENCIL8) {
    const r = ref < 0 ? 0 : (ref > 1 ? 1 : ref);
    return Math.round(r * 16777215) / 16777215;
  }
  return ref;
}

/** True when minFilter selects among mip levels (anisotropy requires this). */
function isMipmapMinFilter(f: GLenum): boolean {
  return f === NEAREST_MIPMAP_NEAREST || f === LINEAR_MIPMAP_NEAREST ||
    f === NEAREST_MIPMAP_LINEAR || f === LINEAR_MIPMAP_LINEAR;
}

/* ================================================================== */
/* Per-(image, state) sampling plan (loop-invariant hoist)             */
/* ================================================================== */

/**
 * Sampling facts that are constant for a (TextureImage, SamplerState) pair.
 * The TextureEnv precomputes one per bound unit per draw; the generic entry
 * points fill a module-level scratch plan per call. This is the "filter
 * dispatch hoist": the per-fragment path reads booleans/derived values
 * instead of re-deriving filter semantics from the GL enums, and skips the
 * LOD machinery entirely for single-level images.
 */
interface SamplePlan {
  /** Effective LOD range (baseLevel/maxLevel clamped to the available levels). */
  base: number;
  hi: number;
  /** Only one level is addressable (hi <= base) → λ's magnitude is irrelevant. */
  single: boolean;
  /**
   * single && level(base) is 1×1 in every filtered dimension (2D / 3D / 2D_ARRAY;
   * NOT cube — the face still selects) → every filter/wrap maps all taps to
   * the one texel (2D_ARRAY still selects its layer).
   */
  oneTexel: boolean;
  /** Per-level filter when λ > 0 (minFilter-derived: NEAREST or LINEAR). */
  posFilter: GLenum;
  /** minFilter is one of the four mipmap filters. */
  mipMin: boolean;
  /** Normalized u8 4-component identity layout → bilinear fast tap eligible. */
  u8Fast: boolean;
}

/** Fills `p` from (img, state) — no allocation (writes into the caller's object). */
function fillPlan(img: TextureImage, state: SamplerState, p: SamplePlan): SamplePlan {
  let hi = img.maxLevel;
  const maxAvail = img.levels.length - 1;
  if (hi > maxAvail) hi = maxAvail;
  let base = img.baseLevel;
  if (hi < 0) hi = 0;
  if (base > hi) base = hi;
  const min = state.minFilter;
  const info = img.info;
  p.base = base;
  p.hi = hi;
  p.single = hi <= base;
  p.posFilter = (min === LINEAR || min === LINEAR_MIPMAP_NEAREST || min === LINEAR_MIPMAP_LINEAR)
    ? LINEAR : NEAREST;
  p.mipMin = min === NEAREST_MIPMAP_NEAREST || min === LINEAR_MIPMAP_NEAREST ||
    min === NEAREST_MIPMAP_LINEAR || min === LINEAR_MIPMAP_LINEAR;
  p.u8Fast = info.storage === 'u8' && info.normalized && !info.isInteger &&
    !info.isDepth && !info.isSRGB && info.components === 4;
  const target = img.target;
  p.oneTexel = p.single && target !== TEXTURE_CUBE_MAP && (() => {
    const lv = img.levels[base];
    if (lv.width !== 1 || lv.height !== 1) return false;
    return target !== TEXTURE_3D || lv.depth === 1;
  })();
  return p;
}

/** Module-level scratch plan for the generic entry points (sampling is sequential). */
const _plan: SamplePlan = { base: 0, hi: 0, single: false, oneTexel: false, posFilter: 0, mipMin: false, u8Fast: false };

/**
 * λ > 0 test for single-level textures: the level is fixed, so only the
 * mag-vs-min filter choice depends on the sign of λ. Avoids Math.log2 and the
 * LOD clamps when the decision is robust. Returns null (→ caller computes the
 * full λ via computeLod) when anisotropy is active or the footprint sits
 * within 1e-9 relative of the λ = 0 boundary, where log2 rounding could flip
 * the sign — the fallback is bit-identical to the pre-optimization path.
 */
function lodGtZero(img: TextureImage, state: SamplerState, coord: SampleCoord, bias: number): boolean | null {
  if (state.minLod > 0) return true;
  if (state.maxLod <= 0) return false;
  if (state.maxAnisotropy > 1 && img.target === TEXTURE_2D && isMipmapMinFilter(state.minFilter)) return null;
  const dx = coord.dx;
  const dy = coord.dy;
  let rhoX = 0;
  let rhoY = 0;
  if (dx && dy) {
    const target = img.target;
    let sx: number, sy: number, sz: number;
    if (target === TEXTURE_CUBE_MAP) { sx = img.width; sy = img.height; sz = img.width; }
    else if (target === TEXTURE_3D) { sx = img.width; sy = img.height; sz = img.depth; }
    else { sx = img.width; sy = img.height; sz = 0; } // 2D / 2D_ARRAY: layer not filtered
    rhoX = Math.abs(dx[0]) * sx;
    const dyx = Math.abs(dx[1]) * sy;
    if (dyx > rhoX) rhoX = dyx;
    if (sz > 0 && dx.length > 2) {
      const dzx = Math.abs(dx[2]) * sz;
      if (dzx > rhoX) rhoX = dzx;
    }
    rhoY = Math.abs(dy[0]) * sx;
    const dyy = Math.abs(dy[1]) * sy;
    if (dyy > rhoY) rhoY = dyy;
    if (sz > 0 && dy.length > 2) {
      const dzy = Math.abs(dy[2]) * sz;
      if (dzy > rhoY) rhoY = dzy;
    }
  }
  if (rhoX === 0 && rhoY === 0) return false; // λ = minLod ≤ 0 (minLod > 0 returned above)
  const b = bias < -2 ? -2 : (bias > 2 ? 2 : bias);
  const rel = (rhoX > rhoY ? rhoX : rhoY) / (2 ** -b);
  if (rel > 1 + 1e-9) return true;
  if (rel < 1 - 1e-9) return false;
  return null;
}

/* ================================================================== */
/* Raw texel reads                                                     */
/* ================================================================== */

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

/* ================================================================== */
/* LOD computation                                                     */
/* ================================================================== */

/**
 * Implicit LOD: ρx = max over axes of |∂coord_i/∂x|·size_i (ρy likewise),
 * ρ = max(ρx, ρy), λ = clamp(log2(ρ) + bias, minLod, maxLod). Bias is
 * clamped to ±2 before adding (ES 3.0). Zero/absent derivatives → ρ = 0 →
 * λ = minLod. Anisotropy applies to 2D with a mipmap min filter.
 */
function computeLod(img: TextureImage, state: SamplerState, coord: SampleCoord, bias: number): number {
  const dx = coord.dx;
  const dy = coord.dy;
  let rhoX = 0;
  let rhoY = 0;
  if (dx && dy) {
    const target = img.target;
    let sx: number, sy: number, sz: number;
    if (target === TEXTURE_CUBE_MAP) { sx = img.width; sy = img.height; sz = img.width; }
    else if (target === TEXTURE_3D) { sx = img.width; sy = img.height; sz = img.depth; }
    else { sx = img.width; sy = img.height; sz = 0; } // 2D / 2D_ARRAY: layer not filtered
    rhoX = Math.abs(dx[0]) * sx;
    const dyx = Math.abs(dx[1]) * sy;
    if (dyx > rhoX) rhoX = dyx;
    if (sz > 0 && dx.length > 2) {
      const dzx = Math.abs(dx[2]) * sz;
      if (dzx > rhoX) rhoX = dzx;
    }
    rhoY = Math.abs(dy[0]) * sx;
    const dyy = Math.abs(dy[1]) * sy;
    if (dyy > rhoY) rhoY = dyy;
    if (sz > 0 && dy.length > 2) {
      const dzy = Math.abs(dy[2]) * sz;
      if (dzy > rhoY) rhoY = dzy;
    }
  }
  let rho = rhoX > rhoY ? rhoX : rhoY;
  if (state.maxAnisotropy > 1 && img.target === TEXTURE_2D && isMipmapMinFilter(state.minFilter)) {
    if (rhoX === 0 || rhoY === 0) {
      rho = rho / state.maxAnisotropy;
    } else {
      const mn = rhoX < rhoY ? rhoX : rhoY;
      rho = rho / Math.min(state.maxAnisotropy, rho / mn);
    }
  }
  const b = bias < -2 ? -2 : (bias > 2 ? 2 : bias);
  let lambda = Math.log2(rho) + b;
  if (lambda < state.minLod) lambda = state.minLod;
  else if (lambda > state.maxLod) lambda = state.maxLod;
  return lambda;
}

/* ================================================================== */
/* Per-level filtering                                                 */
/* ================================================================== */

/**
 * Reads one tap and, for shadow sampling, converts it to a comparison
 * result (0/1) written into out[0] (0 elsewhere).
 */
function readTap(
  img: TextureImage, level: number, face: number, x: number, y: number, z: number,
  shadow: boolean, refQ: number, compareFunc: GLenum, out: Float32Array,
): void {
  readTexel(img, level, face, x, y, z, out);
  if (shadow) {
    out[0] = depthFuncPass(compareFunc, refQ, out[0]) ? 1 : 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
  }
}

/** out += w · t (bilinear/trilinear accumulation). */
function accum(out: Float32Array, t: Float32Array, w: number): void {
  out[0] += w * t[0];
  out[1] += w * t[1];
  out[2] += w * t[2];
  out[3] += w * t[3];
}

/** f32(i/255) for i in 0..255 — the exact u8 normalized decode (bit-identical to the generic readFromEntry path: raw/255 then Float32Array store). */
const U8_DIV = new Float64Array(256);
for (let i = 0; i < 256; i++) U8_DIV[i] = Math.fround(i / 255);

/**
 * Bilinear fast tap for the normalized u8 RGBA class (RGBA8/RGBA): direct
 * Uint8Array index arithmetic + LUT decode — no per-tap format dispatch, no
 * Float32Array scratch round-trips. Numerics replicate the generic path
 * EXACTLY: each tap t = f32(raw/255) (U8_DIV) and out accumulates
 * fl32(fl64(out) + fl64(w·t)) in the same tap order and with the same weight
 * expressions as filter2D's accum() sequence.
 */
function filter2DU8Tap(
  lv: TextureLevel, z: number, a: number, b: number,
  i0: number, i1: number, j0: number, j1: number, out: Float32Array,
): void {
  const data = lv.data[z] as Uint8Array;
  const w = lv.width;
  const o00 = (j0 * w + i0) * 4;
  const o10 = (j0 * w + i1) * 4;
  const o01 = (j1 * w + i0) * 4;
  const o11 = (j1 * w + i1) * 4;
  const w00 = (1 - a) * (1 - b);
  const w10 = a * (1 - b);
  const w01 = (1 - a) * b;
  const w11 = a * b;
  const div = U8_DIV;
  for (let c = 0; c < 4; c++) {
    out[c] = w00 * div[data[o00 + c]];
    out[c] += w10 * div[data[o10 + c]];
    out[c] += w01 * div[data[o01 + c]];
    out[c] += w11 * div[data[o11 + c]];
  }
}

/** 2D / 2D_ARRAY filtering at one level. `layer` is clamped to the level's depth, NOT filtered/wrapped. */
function filter2D(
  img: TextureImage, state: SamplerState, plan: SamplePlan, filter: GLenum, level: number,
  u: number, v: number, layer: number, shadow: boolean, refQ: number, out: Float32Array,
): void {
  const lv = img.levels[level];
  const z = img.target === TEXTURE_2D_ARRAY ? clampLayer(layer, lv.depth) : 0;
  if (filter === NEAREST || img.info.isInteger) {
    readTap(img, level, 0,
      nearestIndex(u, lv.width, state.wrapS), nearestIndex(v, lv.height, state.wrapT),
      z, shadow, refQ, state.compareFunc, out);
    return;
  }
  filterCoord(u, lv.width, state.wrapS, _c0);
  filterCoord(v, lv.height, state.wrapT, _c1);
  // Bilinear: 4 taps accumulated into out.
  const a = _c0[1];
  const b = _c1[1];
  const i0 = _c0[0];
  const i1 = _c0[2];
  const j0 = _c1[0];
  const j1 = _c1[2];
  if (plan.u8Fast && !shadow) {
    filter2DU8Tap(lv, z, a, b, i0, i1, j0, j1, out);
    return;
  }
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
  readTap(img, level, 0, i0, j0, z, shadow, refQ, state.compareFunc, _tap);
  accum(out, _tap, (1 - a) * (1 - b));
  readTap(img, level, 0, i1, j0, z, shadow, refQ, state.compareFunc, _tap);
  accum(out, _tap, a * (1 - b));
  readTap(img, level, 0, i0, j1, z, shadow, refQ, state.compareFunc, _tap);
  accum(out, _tap, (1 - a) * b);
  readTap(img, level, 0, i1, j1, z, shadow, refQ, state.compareFunc, _tap);
  accum(out, _tap, a * b);
}

/** 3D filtering at one level (trilinear for LINEAR; no shadow samplers in 3D). */
function filter3D(
  img: TextureImage, state: SamplerState, filter: GLenum, level: number,
  u: number, v: number, w: number, out: Float32Array,
): void {
  const lv = img.levels[level];
  if (filter === NEAREST) {
    readTexel(img, level, 0,
      nearestIndex(u, lv.width, state.wrapS), nearestIndex(v, lv.height, state.wrapT),
      nearestIndex(w, lv.depth, state.wrapR), out);
    return;
  }
  filterCoord(u, lv.width, state.wrapS, _c0);
  filterCoord(v, lv.height, state.wrapT, _c1);
  filterCoord(w, lv.depth, state.wrapR, _c2);
  const a = _c0[1];
  const b = _c1[1];
  const c = _c2[1];
  const i0 = _c0[0], i1 = _c0[2];
  const j0 = _c1[0], j1 = _c1[2];
  const k0 = _c2[0], k1 = _c2[2];
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
  const w000 = (1 - a) * (1 - b) * (1 - c);
  const w100 = a * (1 - b) * (1 - c);
  const w010 = (1 - a) * b * (1 - c);
  const w110 = a * b * (1 - c);
  const w001 = (1 - a) * (1 - b) * c;
  const w101 = a * (1 - b) * c;
  const w011 = (1 - a) * b * c;
  const w111 = a * b * c;
  readTexel(img, level, 0, i0, j0, k0, _tap); accum(out, _tap, w000);
  readTexel(img, level, 0, i1, j0, k0, _tap); accum(out, _tap, w100);
  readTexel(img, level, 0, i0, j1, k0, _tap); accum(out, _tap, w010);
  readTexel(img, level, 0, i1, j1, k0, _tap); accum(out, _tap, w110);
  readTexel(img, level, 0, i0, j0, k1, _tap); accum(out, _tap, w001);
  readTexel(img, level, 0, i1, j0, k1, _tap); accum(out, _tap, w101);
  readTexel(img, level, 0, i0, j1, k1, _tap); accum(out, _tap, w011);
  readTexel(img, level, 0, i1, j1, k1, _tap); accum(out, _tap, w111);
}

/* ------------------------------------------------------------------ */
/* Cube maps                                                           */
/* ------------------------------------------------------------------ */

/**
 * Face selection by dominant axis of the direction (GLES 3.0 table 3.22,
 * with deqp's tie-breaking). Returns face index 0..5
 * (+X, −X, +Y, −Y, +Z, −Z — CUBE_FACE_TO_INDEX order).
 */
function selectCubeFace(rx: number, ry: number, rz: number): number {
  const ax = Math.abs(rx);
  const ay = Math.abs(ry);
  const az = Math.abs(rz);
  if (ay < ax && az < ax) return rx >= 0 ? 0 : 1;
  if (ax < ay && az < ay) return ry >= 0 ? 2 : 3;
  if (ax < az && ay < az) return rz >= 0 ? 4 : 5;
  // Ties.
  if (ax === ay) {
    if (ax < az) return rz >= 0 ? 4 : 5;
    return rx >= 0 ? 0 : 1;
  }
  if (ax === az) {
    if (az < ay) return ry >= 0 ? 2 : 3;
    return rz >= 0 ? 4 : 5;
  }
  if (ay === az) {
    if (ay < ax) return rx >= 0 ? 0 : 1;
    return ry >= 0 ? 2 : 3;
  }
  return rx >= 0 ? 0 : 1;
}

/** Projects a direction onto `face`; writes [s, t] ∈ [0,1]² into `dst`. */
function projectToFace(face: number, rx: number, ry: number, rz: number, dst: Float32Array): void {
  let sc: number, tc: number, ma: number;
  switch (face) {
    case 0: sc = -rz; tc = -ry; ma = rx; break;   // +X
    case 1: sc = rz; tc = -ry; ma = -rx; break;   // −X
    case 2: sc = rx; tc = rz; ma = ry; break;     // +Y
    case 3: sc = rx; tc = -rz; ma = -ry; break;   // −Y
    case 4: sc = rx; tc = -ry; ma = rz; break;    // +Z
    default: sc = -rx; tc = -ry; ma = -rz; break; // −Z
  }
  dst[0] = (sc / ma + 1) / 2;
  dst[1] = (tc / ma + 1) / 2;
}

/** Inverse of projectToFace with ma = 1: face-local (sc, tc) → direction. */
function inverseFace(face: number, sc: number, tc: number, dst: Float32Array): void {
  switch (face) {
    case 0: dst[0] = 1; dst[1] = -tc; dst[2] = -sc; break;   // +X
    case 1: dst[0] = -1; dst[1] = -tc; dst[2] = sc; break;   // −X
    case 2: dst[0] = sc; dst[1] = 1; dst[2] = tc; break;     // +Y
    case 3: dst[0] = sc; dst[1] = -1; dst[2] = -tc; break;   // −Y
    case 4: dst[0] = sc; dst[1] = -tc; dst[2] = 1; break;    // +Z
    default: dst[0] = -sc; dst[1] = -tc; dst[2] = -1; break; // −Z
  }
}

/**
 * One cube tap at (x, y) on `face`. In-range taps read directly; taps
 * crossing exactly one face edge are re-projected onto the adjacent face
 * (equivalent to the spec's edge-remap table). Returns false when the tap
 * is out of bounds on BOTH axes (corner — caller averages the other taps).
 */
function cubeTap(
  img: TextureImage, level: number, face: number, x: number, y: number,
  shadow: boolean, refQ: number, compareFunc: GLenum, out: Float32Array,
): boolean {
  const size = img.levels[level].width;
  const xIn = x >= 0 && x < size;
  const yIn = y >= 0 && y < size;
  if (xIn && yIn) {
    readTap(img, level, face, x, y, 0, shadow, refQ, compareFunc, out);
    return true;
  }
  if (!xIn && !yIn) return false;
  // Single-edge crossing: re-project the tap's texel center onto the cube.
  const uu = (x + 0.5) / size;
  const vv = (y + 0.5) / size;
  inverseFace(face, 2 * uu - 1, 2 * vv - 1, _dir);
  const f2 = selectCubeFace(_dir[0], _dir[1], _dir[2]);
  projectToFace(f2, _dir[0], _dir[1], _dir[2], _uv);
  const tx = Math.floor(_uv[0] * size);
  const ty = Math.floor(_uv[1] * size);
  readTap(img, level, f2, tx < 0 ? 0 : (tx >= size ? size - 1 : tx), ty < 0 ? 0 : (ty >= size ? size - 1 : ty), 0, shadow, refQ, compareFunc, out);
  return true;
}

/**
 * Cube filtering at one level. NEAREST: single tap, clamped within the
 * face. LINEAR: seamless 4-tap bilinear; taps leaving the face are
 * re-projected onto the adjacent face; a doubly-out-of-bounds (corner) tap
 * is approximated by the average of the other three taps (deqp-recommended
 * approximation, allowed by the spec).
 */
function filterCube(
  img: TextureImage, state: SamplerState, filter: GLenum, level: number,
  v: Float32Array, shadow: boolean, refQ: number, out: Float32Array,
): void {
  const face = selectCubeFace(v[0], v[1], v[2]);
  projectToFace(face, v[0], v[1], v[2], _uv);
  const size = img.levels[level].width;
  if (filter === NEAREST || img.info.isInteger) {
    readTap(img, level, face, nearestIndex(_uv[0], size, CLAMP_TO_EDGE), nearestIndex(_uv[1], size, CLAMP_TO_EDGE), 0, shadow, refQ, state.compareFunc, out);
    return;
  }
  // Seamless bilinear.
  filterCoordRaw(_uv[0], size, _c0);
  filterCoordRaw(_uv[1], size, _c1);
  const a = _c0[1];
  const b = _c1[1];
  const i0 = _c0[0], i1 = _c0[2];
  const j0 = _c1[0], j1 = _c1[2];
  const w = [
    (1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b,
  ];
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
  _cornerSum[0] = 0; _cornerSum[1] = 0; _cornerSum[2] = 0; _cornerSum[3] = 0;
  let cornerW = -1;
  const xs = [i0, i1, i0, i1];
  const ys = [j0, j0, j1, j1];
  for (let t = 0; t < 4; t++) {
    if (cubeTap(img, level, face, xs[t], ys[t], shadow, refQ, state.compareFunc, _tap)) {
      accum(out, _tap, w[t]);
      accum(_cornerSum, _tap, 1);
    } else {
      cornerW = w[t];
    }
  }
  if (cornerW >= 0) {
    // Corner tap ≈ mean of the other three taps.
    out[0] += cornerW * (_cornerSum[0] / 3);
    out[1] += cornerW * (_cornerSum[1] / 3);
    out[2] += cornerW * (_cornerSum[2] / 3);
    out[3] += cornerW * (_cornerSum[3] / 3);
  }
}

/* ================================================================== */
/* Level selection + dispatch                                          */
/* ================================================================== */

/**
 * Chooses the level(s) + per-level filter from λ and the filter state and
 * dispatches to the target-specific filter. `shadow`/`refQ` drive the
 * depth-compare path (2D, 2D_ARRAY, cube). Only multi-level images reach here
 * (plan.single is short-circuited by the sample* entry points).
 */
function sampleLevels(
  img: TextureImage, state: SamplerState, plan: SamplePlan, lambda: number,
  coord: SampleCoord, shadow: boolean, refQ: number, out: Float32Array,
): void {
  const isInteger = img.info.isInteger;
  let filter: GLenum;
  let level0: number;
  let level1 = -1;
  let f = 0;

  if (isInteger) {
    // Integer textures: NEAREST taps only (also keeps filter3D's single-tap
    // path — it has no isInteger guard). Mipmap-linear blends are invalid for
    // integer formats (incomplete per spec) — degrade to a single level.
    filter = NEAREST;
    level0 = (lambda <= 0 || !plan.mipMin)
      ? plan.base : clampLevel(Math.round(lambda), plan.base, plan.hi);
  } else if (lambda <= 0) {
    filter = state.magFilter;
    level0 = plan.base;
  } else if (!plan.mipMin) {
    filter = plan.posFilter; // == minFilter (NEAREST or LINEAR)
    level0 = plan.base;
  } else if (state.minFilter === NEAREST_MIPMAP_NEAREST || state.minFilter === LINEAR_MIPMAP_NEAREST) {
    filter = plan.posFilter;
    level0 = clampLevel(Math.round(lambda), plan.base, plan.hi);
  } else {
    // NEAREST_MIPMAP_LINEAR / LINEAR_MIPMAP_LINEAR.
    filter = plan.posFilter;
    if (plan.hi <= plan.base) {
      level0 = plan.base;
    } else {
      level0 = clampLevel(Math.floor(lambda), plan.base, plan.hi - 1);
      level1 = level0 + 1;
      f = lambda - Math.floor(lambda);
    }
  }

  const target = img.target;
  // Single-texel fast path: when only level0 is used and it is 1×1, every
  // filter/wrap maps all taps to the one texel — read it directly (identical
  // result; covers non-mip min filters with λ > 0 and _MIPMAP_NEAREST levels).
  // NOT for cube (the face is still selected by the direction, and LINEAR is
  // seamless across faces — readTexel would only read face 0) or shadow
  // (depth compare must still run).
  if (!shadow && level1 < 0 && target !== TEXTURE_CUBE_MAP) {
    const l0 = img.levels[level0];
    if (l0.width === 1 && l0.height === 1 && (target !== TEXTURE_3D || l0.depth === 1)) {
      const z = target === TEXTURE_2D_ARRAY ? clampLayer(coord.v[2], l0.depth) : 0;
      readTexel(img, level0, 0, 0, 0, z, out);
      return;
    }
  }

  if (target === TEXTURE_CUBE_MAP) {
    if (level1 >= 0) {
      filterCube(img, state, filter, level0, coord.v, shadow, refQ, _mipA);
      filterCube(img, state, filter, level1, coord.v, shadow, refQ, _mipB);
      out[0] = _mipA[0] * (1 - f) + _mipB[0] * f;
      out[1] = _mipA[1] * (1 - f) + _mipB[1] * f;
      out[2] = _mipA[2] * (1 - f) + _mipB[2] * f;
      out[3] = _mipA[3] * (1 - f) + _mipB[3] * f;
    } else {
      filterCube(img, state, filter, level0, coord.v, shadow, refQ, out);
    }
  } else if (target === TEXTURE_3D) {
    if (level1 >= 0) {
      filter3D(img, state, filter, level0, coord.v[0], coord.v[1], coord.v[2], _mipA);
      filter3D(img, state, filter, level1, coord.v[0], coord.v[1], coord.v[2], _mipB);
      out[0] = _mipA[0] * (1 - f) + _mipB[0] * f;
      out[1] = _mipA[1] * (1 - f) + _mipB[1] * f;
      out[2] = _mipA[2] * (1 - f) + _mipB[2] * f;
      out[3] = _mipA[3] * (1 - f) + _mipB[3] * f;
    } else {
      filter3D(img, state, filter, level0, coord.v[0], coord.v[1], coord.v[2], out);
    }
  } else {
    // 2D / 2D_ARRAY.
    const layer = target === TEXTURE_2D_ARRAY ? coord.v[2] : 0;
    if (level1 >= 0) {
      filter2D(img, state, plan, filter, level0, coord.v[0], coord.v[1], layer, shadow, refQ, _mipA);
      filter2D(img, state, plan, filter, level1, coord.v[0], coord.v[1], layer, shadow, refQ, _mipB);
      out[0] = _mipA[0] * (1 - f) + _mipB[0] * f;
      out[1] = _mipA[1] * (1 - f) + _mipB[1] * f;
      out[2] = _mipA[2] * (1 - f) + _mipB[2] * f;
      out[3] = _mipA[3] * (1 - f) + _mipB[3] * f;
    } else {
      filter2D(img, state, plan, filter, level0, coord.v[0], coord.v[1], layer, shadow, refQ, out);
    }
  }
}

/** Reads the single texel of a 1×1-per-filtered-dims level (2D_ARRAY still selects its layer). */
function readSingleTexel(img: TextureImage, plan: SamplePlan, coord: SampleCoord, out: Float32Array): void {
  const lv = img.levels[plan.base];
  const z = img.target === TEXTURE_2D_ARRAY ? clampLayer(coord.v[2], lv.depth) : 0;
  readTexel(img, plan.base, 0, 0, 0, z, out);
}

/**
 * Single-level sampling (plan.single): the level is always plan.base — only
 * the mag-vs-min filter choice depends on λ (its sign, precomputed by the
 * caller). A 1×1 base level short-circuits entirely (every filter/wrap maps
 * all taps to the one texel).
 */
function sampleSingleLevel(
  img: TextureImage, state: SamplerState, plan: SamplePlan, lambdaGT0: boolean,
  coord: SampleCoord, shadow: boolean, refQ: number, out: Float32Array,
): void {
  if (!shadow && plan.oneTexel) {
    readSingleTexel(img, plan, coord, out);
    return;
  }
  const target = img.target;
  // Integer formats only allow NEAREST taps (spec; also keeps filter3D's
  // single-tap path — it has no isInteger guard, matching old sampleLevels
  // which forced NEAREST for integers in every branch).
  const filter = img.info.isInteger ? NEAREST : (lambdaGT0 ? plan.posFilter : state.magFilter);
  if (target === TEXTURE_CUBE_MAP) {
    filterCube(img, state, filter, plan.base, coord.v, shadow, refQ, out);
  } else if (target === TEXTURE_3D) {
    filter3D(img, state, filter, plan.base, coord.v[0], coord.v[1], coord.v[2], out);
  } else {
    const layer = target === TEXTURE_2D_ARRAY ? coord.v[2] : 0;
    filter2D(img, state, plan, filter, plan.base, coord.v[0], coord.v[1], layer, shadow, refQ, out);
  }
}

/** Implicit-LOD sample with a precomputed plan (TextureEnv path). */
function sampleTextureP(
  img: TextureImage, state: SamplerState, plan: SamplePlan,
  coord: SampleCoord, bias: number, out: Float32Array,
): void {
  if (plan.single) {
    if (plan.oneTexel) {
      readSingleTexel(img, plan, coord, out);
      return;
    }
    const gt0 = lodGtZero(img, state, coord, bias);
    sampleSingleLevel(img, state, plan,
      gt0 === null ? computeLod(img, state, coord, bias) > 0 : gt0,
      coord, false, 0, out);
    return;
  }
  const lambda = computeLod(img, state, coord, bias);
  sampleLevels(img, state, plan, lambda, coord, false, 0, out);
}

/** Explicit-LOD sample with a precomputed plan (TextureEnv path). */
function sampleTextureLodP(
  img: TextureImage, state: SamplerState, plan: SamplePlan,
  coord: SampleCoord, lod: number, out: Float32Array,
): void {
  let lambda = lod;
  if (lambda < state.minLod) lambda = state.minLod;
  else if (lambda > state.maxLod) lambda = state.maxLod;
  if (plan.single) {
    if (plan.oneTexel) {
      readSingleTexel(img, plan, coord, out);
      return;
    }
    sampleSingleLevel(img, state, plan, lambda > 0, coord, false, 0, out);
    return;
  }
  sampleLevels(img, state, plan, lambda, coord, false, 0, out);
}

/** Shadow sample with a precomputed plan (TextureEnv path). */
function sampleTextureShadowP(
  img: TextureImage, state: SamplerState, plan: SamplePlan,
  coord: SampleCoord, ref: number, bias: number, out: Float32Array,
): void {
  const refQ = quantizeShadowRef(ref, img.internalFormat);
  if (plan.single) {
    const gt0 = lodGtZero(img, state, coord, bias);
    sampleSingleLevel(img, state, plan,
      gt0 === null ? computeLod(img, state, coord, bias) > 0 : gt0,
      coord, true, refQ, out);
    return;
  }
  const lambda = computeLod(img, state, coord, bias);
  sampleLevels(img, state, plan, lambda, coord, true, refQ, out);
}

/**
 * Generic implicit-LOD sample. coord.v = [u,v] (2D/2D_ARRAY, v[2]=layer) or
 * [u,v,w] (3D, cube); dx/dy are per-component derivatives. Writes [r,g,b,a]
 * into `out` (length ≥ 4); integer formats write raw bit patterns.
 */
export function sampleTexture(
  img: TextureImage, state: SamplerState, coord: SampleCoord,
  bias: number, out: Float32Array,
): void {
  if (!img.complete) {
    writeDefault(out, img.info.isInteger);
    return;
  }
  sampleTextureP(img, state, fillPlan(img, state, _plan), coord, bias, out);
}

/** Explicit-LOD sample (textureLod; also used by vertex shaders). */
export function sampleTextureLod(
  img: TextureImage, state: SamplerState, coord: SampleCoord,
  lod: number, out: Float32Array,
): void {
  if (!img.complete) {
    writeDefault(out, img.info.isInteger);
    return;
  }
  sampleTextureLodP(img, state, fillPlan(img, state, _plan), coord, lod, out);
}

/** Shadow sample: compares `ref` against depth texels via compareFunc; writes 0/1 (fractional under LINEAR) into out[0], 0 elsewhere. */
export function sampleTextureShadow(
  img: TextureImage, state: SamplerState, coord: SampleCoord,
  ref: number, bias: number, out: Float32Array,
): void {
  if (!img.complete) {
    writeDefault(out, false);
    return;
  }
  sampleTextureShadowP(img, state, fillPlan(img, state, _plan), coord, ref, bias, out);
}

/* ================================================================== */
/* TextureEnv (codegen-facing)                                         */
/* ================================================================== */

/** One resolved texture unit (precomputed once per draw). `plan` null → image incomplete (default result, integer alpha rule). */
type ResolvedUnit =
  | { img: TextureImage; state: SamplerState; plan: SamplePlan }
  | { img: TextureImage; state: SamplerState; plan: null };

/**
 * Allocates the scratch (out/outInt/outUint share one ArrayBuffer, plus the
 * per-env coordinate scratch) and binds the per-draw unit table. Called once
 * per draw by the rasterizer. All methods are allocation-free.
 *
 * Per-draw resolve hoist: the unit table is fixed for the whole draw, so every
 * unit is resolved ONCE here (binding → img/state + sampling plan; null =
 * unbound, plan null = incomplete). Per-fragment sample* calls then do a
 * single array lookup + two null checks — no per-fragment resolve() object,
 * no completeness check, no plan computation, no filter-enum re-derivation.
 */
export function createTextureEnv(units: readonly (TextureUnitBinding | null)[]): TextureEnv {
  const out = new Float32Array(4);
  const outInt = new Int32Array(out.buffer);
  const outUint = new Uint32Array(out.buffer);
  const v = new Float32Array(4);
  const dx = new Float32Array(4);
  const dy = new Float32Array(4);
  const coord: SampleCoord = { v, dx, dy };

  const resolved: (ResolvedUnit | null)[] = new Array(units.length);
  for (let i = 0; i < units.length; i++) {
    const b = units[i];
    if (!b) {
      resolved[i] = null;
    } else if (!b.img.complete) {
      resolved[i] = { img: b.img, state: b.state, plan: null };
    } else {
      resolved[i] = {
        img: b.img, state: b.state,
        plan: fillPlan(b.img, b.state, { base: 0, hi: 0, single: false, oneTexel: false, posFilter: 0, mipMin: false, u8Fast: false }),
      };
    }
  }

  /** Resolves the unit; writes the null/incomplete result into env.out and returns null. */
  function resolve(unit: number): { img: TextureImage; state: SamplerState; plan: SamplePlan } | null {
    const b = resolved[unit];
    if (!b) {
      out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
      return null;
    }
    if (!b.plan) {
      out[0] = 0; out[1] = 0; out[2] = 0;
      out[3] = b.img.info.isInteger ? 0 : 1;
      return null;
    }
    return b;
  }

  return {
    units,
    out,
    outInt,
    outUint,

    sample2D(unit: number, u: number, vv: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = 0;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sample2DLod(unit: number, u: number, vv: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = 0;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sample3D(unit: number, u: number, vv: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      dx[0] = dux; dx[1] = dvx; dx[2] = dwx;
      dy[0] = duy; dy[1] = dvy; dy[2] = dwy;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sample3DLod(unit: number, u: number, vv: number, w: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sampleCube(unit: number, u: number, vv: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      dx[0] = dux; dx[1] = dvx; dx[2] = dwx;
      dy[0] = duy; dy[1] = dvy; dy[2] = dwy;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sampleCubeLod(unit: number, u: number, vv: number, w: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sample2DArray(unit: number, u: number, vv: number, layer: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = layer;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureP(b.img, b.state, b.plan, coord, bias, out);
    },

    sample2DArrayLod(unit: number, u: number, vv: number, layer: number, lod: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = layer;
      sampleTextureLodP(b.img, b.state, b.plan, coord, lod, out);
    },

    sample2DShadow(unit: number, u: number, vv: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = 0;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureShadowP(b.img, b.state, b.plan, coord, ref, bias, out);
    },

    sampleCubeShadow(unit: number, u: number, vv: number, w: number, ref: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = w;
      dx[0] = dux; dx[1] = dvx; dx[2] = dwx;
      dy[0] = duy; dy[1] = dvy; dy[2] = dwy;
      sampleTextureShadowP(b.img, b.state, b.plan, coord, ref, bias, out);
    },

    sample2DArrayShadow(unit: number, u: number, vv: number, layer: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void {
      const b = resolve(unit);
      if (!b) return;
      v[0] = u; v[1] = vv; v[2] = layer;
      dx[0] = dux; dx[1] = dvx; dx[2] = 0;
      dy[0] = duy; dy[1] = dvy; dy[2] = 0;
      sampleTextureShadowP(b.img, b.state, b.plan, coord, ref, bias, out);
    },

    texelFetch2D(unit: number, x: number, y: number, level: number): void {
      const b = resolve(unit);
      if (!b) return;
      texelFetch(b.img, x, y, 0, level, out);
    },

    texelFetch3D(unit: number, x: number, y: number, z: number, level: number): void {
      const b = resolve(unit);
      if (!b) return;
      texelFetch(b.img, x, y, z, level, out);
    },

    texelFetch2DArray(unit: number, x: number, y: number, layer: number, level: number): void {
      const b = resolve(unit);
      if (!b) return;
      texelFetch(b.img, x, y, layer, level, out);
    },
  };
}

/** Helper: resolves the (img, state) for a unit, with incomplete→null. */
export function resolveUnit(env: TextureEnv, unit: number): { img: TextureImage; state: SamplerState } | null {
  const b = env.units[unit];
  if (!b || !b.img.complete) return null;
  return { img: b.img, state: b.state };
}

/** Number of mip levels for a 2D image of the given size (floor(log2(max))+1). */
export function mipLevelCount(w: number, h: number, d: number): number {
  let m = Math.max(w, h, d);
  let n = 1;
  while (m > 1) { m >>= 1; n++; }
  return n;
}

/** Map of cube face GLenum → face index 0..5 (order +X,-X,+Y,-Y,+Z,-Z). */
export const CUBE_FACE_TO_INDEX: Readonly<Record<GLenum, number>> = {
  0x8515: 0, 0x8516: 1, 0x8517: 2, 0x8518: 3, 0x8519: 4, 0x851a: 5,
};
