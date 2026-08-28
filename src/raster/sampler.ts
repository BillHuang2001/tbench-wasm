/**
 * sampler.ts — texture sampling core (contract §3).
 *
 * Two entry levels:
 *  1. Generic functions (sampleTexture / sampleTextureLod / sampleTextureShadow
 *     / texelFetch) — explicit TextureImage + effective SamplerState; used by
 *     gl/ and unit tests. texelFetch and the raw texel reads live in
 *     sampler-raw.ts (re-exported here); the raw layer is imported by nothing
 *     above it (acyclic).
 *  2. TextureEnv (see types.ts; implemented in sampler-env.ts) — per-draw unit
 *     table + codegen-facing methods (glsl/ compiles `ctx.tex.*` against
 *     them). Each method looks up its unit in a table pre-resolved ONCE per
 *     draw (binding → img/state + sampling plan; null = unbound/incomplete →
 *     default result), calls the core, and writes the result into env.out
 *     (float domain); the raw bits are also visible via env.outInt /
 *     env.outUint for integer textures (same ArrayBuffer). NO allocation: all
 *     results land in preallocated scratch.
 *
 * Fast paths (all mathematically identical to the generic path):
 *  - Single-level images (hi <= base): the level is fixed, so only the sign
 *    of λ matters (mag vs min filter) — computeLod's log2/clamps are skipped
 *    (lodGtZero; full λ fallback at the λ = 0 boundary / under anisotropy).
 *  - 1×1 levels: every filter/wrap maps all taps to the one texel — read it
 *    directly (no LOD, no filter selection at all). NOT for cube (the face is
 *    still selected by the direction, and LINEAR is seamless across faces) or
 *    shadow (the depth compare must still run).
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
 *    λ = clamp(log2(ρ) + bias, minLod, maxLod) (bias added unclamped, GLES
 *    semantics — no GL_MAX_TEXTURE_LOD_BIAS in ES). Zero/absent derivatives
 *    → λ = minLod. Cube maps: ρx/ρy come from the derivatives of the
 *    face-mapped coordinates (s = (sc/|ma|+1)/2 chain rule, see cubeRho) —
 *    magnitude-invariant, matching deqp's computeCubeLodBoundsFromDerivates.
 *    textureLod passes the level directly. λ ≤ 0 → mag filter
 *    at base level; λ > 0 → min filter.
 *    NEAREST_MIPMAP_NEAREST: level = round(λ); NEAREST_MIPMAP_LINEAR:
 *    floor(λ) + frac(λ) blend between d and d+1.
 *  - EXT_texture_filter_anisotropic (2D + mipmap min filter only):
 *    ρ = max(ρx,ρy) / min(maxAnisotropy, max(ρx,ρy)/min(ρx,ρy)), guarded when
 *    min(ρx,ρy) == 0 (→ max(ρx,ρy)/maxAnisotropy). The reduced ρ selects the
 *    mip level(s) EXACTLY as an isotropic sample would (λ′ = log2(ρ/N) —
 *    A/B-verified optimal), then N-tap anisotropic filtering replaces the
 *    single bilinear tap with N equally-weighted bilinear taps along the
 *    major footprint axis (N = pow2ceil(min(ceil(ρmax/ρmin), maxAnisotropy));
 *    tap offsets ((i+0.5)/N − 0.5)·2^λ / size₀; same level pair + same f for
 *    every tap under LINEAR_MIPMAP_LINEAR; explicit-LOD textureLod bypasses
 *    AF; shadow samplers compare per tap). See sampler-aniso.ts for the full
 *    conventions and spec citations.
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
} from './gl-enums';
import type { SampleCoord, SamplerState, TextureImage, TextureLevel } from './types';
import { readTexel, writeDefault } from './sampler-raw';
import { anisoGate, anisoTapParams, pow2ceilN, sample2DAnisoTaps } from './sampler-aniso';
export { readTexel, texelFetch } from './sampler-raw';

/* ================================================================== */
/* Module-level scratch (shared by the generic core; sampling is        */
/* strictly sequential so no reentrancy hazard exists).                 */
/* ================================================================== */

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
/** Cube face mapping scratch: {sc, tc, ma} (faceMap). */
const _face: { sc: number; tc: number; ma: number } = { sc: 0, tc: 0, ma: 0 };
/** Cube LOD footprint scratch: ρx/ρy (cubeRho). */
const _rho: { x: number; y: number } = { x: 0, y: 0 };

/* ================================================================== */
/* Small helpers                                                       */
/* ================================================================== */

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
export interface SamplePlan {
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
export function fillPlan(img: TextureImage, state: SamplerState, p: SamplePlan): SamplePlan {
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
  if (anisoGate(img, state)) return null;
  const dx = coord.dx;
  const dy = coord.dy;
  let rhoX = 0;
  let rhoY = 0;
  if (dx && dy) {
    const target = img.target;
    if (target === TEXTURE_CUBE_MAP) {
      if (cubeRho(img, coord.v, dx, dy, _rho)) {
        rhoX = _rho.x;
        rhoY = _rho.y;
      }
    } else {
      let sx: number, sy: number, sz: number;
      if (target === TEXTURE_3D) { sx = img.width; sy = img.height; sz = img.depth; }
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
  }
  if (rhoX === 0 && rhoY === 0) return false; // λ = minLod ≤ 0 (minLod > 0 returned above)
  const b = bias;
  const rel = (rhoX > rhoY ? rhoX : rhoY) / (2 ** -b);
  if (rel > 1 + 1e-9) return true;
  if (rel < 1 - 1e-9) return false;
  return null;
}

/* ================================================================== */
/* LOD computation                                                     */
/* ================================================================== */

/**
 * Implicit LOD: ρx = max over axes of |∂coord_i/∂x|·size_i (ρy likewise),
 * ρ = max(ρx, ρy), λ = clamp(log2(ρ) + bias, minLod, maxLod). Bias is added
 * unclamped (GLES has no GL_MAX_TEXTURE_LOD_BIAS). Zero/absent derivatives →
 * ρ = 0 → λ = minLod. Cube maps use the face-mapped (s, t) chain rule instead
 * (see cubeRho) — magnitude-invariant, matching deqp. Anisotropy applies to
 * 2D with a mipmap min filter: ρ is reduced by min(maxAniso, ρ/min(ρx,ρy))
 * (λ′ = log2(ρ/N) per the EXT spec — A/B-verified optimal, do NOT change),
 * and the N-tap parameters for the filter stage are recorded in
 * anisoTapParams (see sampler-aniso.ts for the N convention and tap formula).
 */
function computeLod(img: TextureImage, state: SamplerState, coord: SampleCoord, bias: number): number {
  const dx = coord.dx;
  const dy = coord.dy;
  let rhoX = 0;
  let rhoY = 0;
  if (dx && dy) {
    const target = img.target;
    if (target === TEXTURE_CUBE_MAP) {
      if (cubeRho(img, coord.v, dx, dy, _rho)) {
        rhoX = _rho.x;
        rhoY = _rho.y;
      }
    } else {
      let sx: number, sy: number, sz: number;
      if (target === TEXTURE_3D) { sx = img.width; sy = img.height; sz = img.depth; }
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
  }
  let rho = rhoX > rhoY ? rhoX : rhoY;
  if (anisoGate(img, state)) {
    anisoTapParams.majorU = rhoX >= rhoY;
    if (rhoX === 0 || rhoY === 0) {
      rho = rho / state.maxAnisotropy;
      anisoTapParams.n = rho === 0 ? 0 : pow2ceilN(state.maxAnisotropy);
    } else {
      const mn = rhoX < rhoY ? rhoX : rhoY;
      const ratio = rho / mn;
      anisoTapParams.n = pow2ceilN(Math.min(Math.ceil(ratio), state.maxAnisotropy));
      rho = rho / Math.min(state.maxAnisotropy, ratio);
    }
  } else {
    anisoTapParams.n = 0;
  }
  const b = bias;
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

/**
 * Cube face mapping (GLES 3.0 table 3.22 / deqp projectToFace): writes
 * [sc, tc, ma] for `face` into `m` and returns ma (0 → degenerate direction,
 * no dominant axis). Shared by the VALUE path (projectToFace) and the cube
 * LOD chain rule (cubeRho), so the two paths use the SAME mapping and cannot
 * drift.
 */
function faceMap(face: number, rx: number, ry: number, rz: number, m: { sc: number; tc: number; ma: number }): number {
  let sc: number, tc: number, ma: number;
  switch (face) {
    case 0: sc = -rz; tc = -ry; ma = rx; break;   // +X
    case 1: sc = rz; tc = -ry; ma = -rx; break;   // −X
    case 2: sc = rx; tc = rz; ma = ry; break;     // +Y
    case 3: sc = rx; tc = -rz; ma = -ry; break;   // −Y
    case 4: sc = rx; tc = -ry; ma = rz; break;    // +Z
    default: sc = -rx; tc = -ry; ma = -rz; break; // −Z
  }
  m.sc = sc;
  m.tc = tc;
  m.ma = ma;
  return ma;
}

/** Projects a direction onto `face`; writes [s, t] ∈ [0,1]² into `dst`. */
function projectToFace(face: number, rx: number, ry: number, rz: number, dst: Float32Array): void {
  const ma = faceMap(face, rx, ry, rz, _face);
  if (ma === 0) {
    // Degenerate direction (e.g. vec3(0,0,0)): no dominant axis — fall back to
    // the face-center texel (0/0 would produce NaN, poisoning every compare).
    dst[0] = 0.5;
    dst[1] = 0.5;
    return;
  }
  dst[0] = (_face.sc / ma + 1) / 2;
  dst[1] = (_face.tc / ma + 1) / 2;
}

/**
 * Cube-map footprint (ρx, ρy) — GLES 3.0 §3.8.9/3.8.10, deqp
 * computeCubeLodBoundsFromDerivates. On the selected face
 * s = (sc/|ma| + 1)/2, t = (tc/|ma| + 1)/2, so the chain rule gives
 *   ∂s/∂x = ½·(∂sc/∂x·|ma| − sc·∂|ma|/∂x)/|ma|²,  ∂|ma|/∂x = sign(ma)·∂ma/∂x
 * (and the same for t and y). deqp's verifier evaluates this with the
 * MAGNITUDE of the major-axis derivative (|∂r_m/∂x| — the sign only flips
 * the correction term, and only magnitudes enter ρ), and with the in-plane
 * component indices below: the face-rotation sign flips of (sc, tc) cancel
 * in the outer |·|, so faceMap's rotated components with raw derivative
 * indices give IDENTICAL ρ to deqp's raw components.
 * ρx = max(|∂s/∂x|, |∂t/∂x|)·faceSize, ρy likewise — the same structure as
 * the 2D convention. Magnitude-invariant: scaling the direction by k scales
 * every component and its derivatives by k, which cancels in the ratio.
 * Writes ρx/ρy into `rho` and returns true; returns false for a degenerate
 * direction (ma == 0 — the caller then uses ρ = 0, λ = minLod).
 */
function cubeRho(
  img: TextureImage, v: Float32Array, dx: Float32Array, dy: Float32Array,
  rho: { x: number; y: number },
): boolean {
  if (dx.length < 3 || dy.length < 3) return false;
  const face = selectCubeFace(v[0], v[1], v[2]);
  const ma = faceMap(face, v[0], v[1], v[2], _face); // same mapping as the VALUE path
  if (ma === 0) return false;
  const sc = _face.sc;
  const tc = _face.tc;
  const M = ma < 0 ? -ma : ma;
  // Raw component indices (deqp): ±X → s = r_z, t = r_y; ±Y → s = r_x,
  // t = r_z; ±Z → s = r_x, t = r_y; major axis 0/1 → x, 2/3 → y, 4/5 → z.
  const sNdx = face === 0 || face === 1 ? 2 : 0;
  const tNdx = face === 2 || face === 3 ? 2 : 1;
  const maj = face >> 1;
  const inv = 0.5 / (M * M);
  const dmadx = dx[maj] < 0 ? -dx[maj] : dx[maj];
  const dmady = dy[maj] < 0 ? -dy[maj] : dy[maj];
  const dsdx = (dx[sNdx] * M - sc * dmadx) * inv;
  const dtdx = (dx[tNdx] * M - tc * dmadx) * inv;
  const dsdy = (dy[sNdx] * M - sc * dmady) * inv;
  const dtdy = (dy[tNdx] * M - tc * dmady) * inv;
  const size = img.width;
  rho.x = Math.max(Math.abs(dsdx), Math.abs(dtdx)) * size;
  rho.y = Math.max(Math.abs(dsdy), Math.abs(dtdy)) * size;
  return true;
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
 * (plan.single is short-circuited by the sample* entry points). `aniso` is
 * the implicit-LOD anisotropy gate (computeLod filled anisoTapParams right
 * before); explicit-LOD callers pass false.
 */
function sampleLevels(
  img: TextureImage, state: SamplerState, plan: SamplePlan, lambda: number,
  coord: SampleCoord, shadow: boolean, refQ: number, out: Float32Array,
  aniso: boolean,
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
      // Per spec: level0 = clamp(⌊λ⌋, base, hi), level1 = min(level0+1, hi),
      // f = frac(λ). floor(λ) may fall below base (e.g. 0 < λ < 1) and clamps
      // UP to base; a floor(λ) ≥ hi reaches the top level (no hi−1 cap), with
      // level1 == level0 == hi (self-blend at weight f is the level itself).
      level0 = clampLevel(Math.floor(lambda), plan.base, plan.hi);
      level1 = level0 + 1;
      if (level1 > plan.hi) level1 = plan.hi;
      f = lambda - Math.floor(lambda);
    }
  }

  const target = img.target;
  // N-tap anisotropy: minification (λ > 0) with an active gate and a
  // non-isotropic footprint. All taps use the SAME level pair selected above.
  const anisoTaps = aniso && anisoTapParams.n >= 2 && lambda > 0;
  // Single-texel fast path: when only level0 is used and it is 1×1, every
  // filter/wrap maps all taps to the one texel — read it directly (identical
  // result; covers non-mip min filters with λ > 0 and _MIPMAP_NEAREST levels).
  // NOT for cube (the face is still selected by the direction, and LINEAR is
  // seamless across faces — readTexel would only read face 0) or shadow
  // (depth compare must still run), or the aniso tap loop (N taps at shifted
  // UVs — all identical for a 1×1 level, but bypassing keeps the average
  // exact instead of relying on N·x/N round-trips).
  if (!shadow && !anisoTaps && level1 < 0 && target !== TEXTURE_CUBE_MAP) {
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
    if (anisoTaps) {
      sample2DAnisoTaps(filter2D, img, state, plan, filter, level0, level1, f,
        coord.v[0], coord.v[1], layer, shadow, refQ, lambda, out);
      return;
    }
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
 * the mag-vs-min filter choice depends on λ (its sign). A 1×1 base level
 * short-circuits entirely (every filter/wrap maps all taps to the one texel).
 * `lambda` is the full λ on the anisotropy path (taps need the level-λ texel
 * scale) and any value with the correct sign otherwise; `aniso` is the
 * implicit-LOD gate (false for explicit LOD).
 */
function sampleSingleLevel(
  img: TextureImage, state: SamplerState, plan: SamplePlan, lambda: number,
  coord: SampleCoord, shadow: boolean, refQ: number, out: Float32Array,
  aniso: boolean,
): void {
  if (!shadow && plan.oneTexel) {
    readSingleTexel(img, plan, coord, out);
    return;
  }
  const target = img.target;
  // Integer textures: NEAREST taps only (parity with sampleLevels; filter3D
  // has no isInteger guard).
  const filter = img.info.isInteger ? NEAREST : (lambda > 0 ? plan.posFilter : state.magFilter);
  if (target === TEXTURE_CUBE_MAP) {
    filterCube(img, state, filter, plan.base, coord.v, shadow, refQ, out);
  } else if (target === TEXTURE_3D) {
    filter3D(img, state, filter, plan.base, coord.v[0], coord.v[1], coord.v[2], out);
  } else {
    const layer = target === TEXTURE_2D_ARRAY ? coord.v[2] : 0;
    if (aniso && anisoTapParams.n >= 2 && lambda > 0) {
      // Minification at the single level: N taps along the major axis at
      // plan.base (level selection is trivial — no mip blend, level1 = −1).
      sample2DAnisoTaps(filter2D, img, state, plan, filter, plan.base, -1, 0,
        coord.v[0], coord.v[1], layer, shadow, refQ, lambda, out);
      return;
    }
    filter2D(img, state, plan, filter, plan.base, coord.v[0], coord.v[1], layer, shadow, refQ, out);
  }
}

/** Implicit-LOD sample with a precomputed plan (TextureEnv path). */
export function sampleTextureP(
  img: TextureImage, state: SamplerState, plan: SamplePlan,
  coord: SampleCoord, bias: number, out: Float32Array,
): void {
  const aniso = anisoGate(img, state);
  if (plan.single) {
    if (plan.oneTexel) {
      readSingleTexel(img, plan, coord, out);
      return;
    }
    let lambda: number;
    if (aniso) {
      // The gate makes lodGtZero's early returns (minLod > 0 / maxLod <= 0)
      // unsound for the tap path: the full λ AND anisoTapParams are needed.
      lambda = computeLod(img, state, coord, bias);
    } else {
      const gt0 = lodGtZero(img, state, coord, bias);
      lambda = gt0 === null ? computeLod(img, state, coord, bias) : (gt0 ? 1 : -1);
    }
    sampleSingleLevel(img, state, plan, lambda, coord, false, 0, out, aniso);
    return;
  }
  const lambda = computeLod(img, state, coord, bias);
  sampleLevels(img, state, plan, lambda, coord, false, 0, out, aniso);
}

/** Explicit-LOD sample with a precomputed plan (TextureEnv path). AF does NOT apply: the EXT spec defines N from the pixel-footprint derivatives, which an explicit LOD has none of (ANGLE/hardware likewise apply AF only on implicit-LOD lookups). */
export function sampleTextureLodP(
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
    sampleSingleLevel(img, state, plan, lambda, coord, false, 0, out, false);
    return;
  }
  sampleLevels(img, state, plan, lambda, coord, false, 0, out, false);
}

/** Shadow sample with a precomputed plan (TextureEnv path). */
export function sampleTextureShadowP(
  img: TextureImage, state: SamplerState, plan: SamplePlan,
  coord: SampleCoord, ref: number, bias: number, out: Float32Array,
): void {
  const refQ = quantizeShadowRef(ref, img.internalFormat);
  const aniso = anisoGate(img, state);
  if (plan.single) {
    if (plan.oneTexel) {
      readSingleTexel(img, plan, coord, out);
      return;
    }
    let lambda: number;
    if (aniso) {
      // Full λ + fresh anisoTapParams (see sampleTextureP).
      lambda = computeLod(img, state, coord, bias);
    } else {
      const gt0 = lodGtZero(img, state, coord, bias);
      lambda = gt0 === null ? computeLod(img, state, coord, bias) : (gt0 ? 1 : -1);
    }
    sampleSingleLevel(img, state, plan, lambda, coord, true, refQ, out, aniso);
    return;
  }
  const lambda = computeLod(img, state, coord, bias);
  sampleLevels(img, state, plan, lambda, coord, true, refQ, out, aniso);
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
