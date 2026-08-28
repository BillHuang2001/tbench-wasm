/**
 * sampler-aniso.ts — N-tap anisotropic filtering (EXT_texture_filter_anisotropic).
 *
 * Split out of sampler.ts so the sampling core stays under the line limit.
 * Imports only types.ts + gl-enums (like sampler-raw.ts) — sampler.ts passes
 * its filter2D into sample2DAnisoTaps, so no import cycle exists.
 *
 * Conventions (EXT_texture_filter_anisotropic §3.8.5, verified against the
 * Khronos GL registry text and ANGLE/hardware practice):
 *
 *  - Gate: 2D textures, TEXTURE_MAX_ANISOTROPY_EXT > 1, and a mipmap
 *    minification filter. Magnification has no anisotropy; the EXT scheme is
 *    defined for the 2D minification path only (3D/cube/2D_ARRAY not covered).
 *    Explicit LOD (textureLod) bypasses AF — the spec defines N from the
 *    pixel-footprint derivatives, which an explicit LOD has no notion of, and
 *    ANGLE/hardware apply AF only on implicit-LOD lookups.
 *
 *  - Tap count: N = min(ceil(ρmax/ρmin), maxAniso), then rounded UP to the
 *    nearest supported sampling rate (power of two) — the spec's literal
 *    sequence ("N = min(ceil(Pmax/Pmin), maxAniso)"; "It is acceptable for
 *    implementation to round 'N' up to the nearest supported sampling rate").
 *    Power-of-two tap counts are what GPU hardware uses (2x/4x/8x/16x AF
 *    levels); because the rounding happens AFTER the maxAniso min, N may
 *    exceed maxAniso when it is not a power of two (e.g. maxAniso = 3,
 *    ratio ≥ 3 → N = 4) — hardware with power-of-two-only rates behaves the
 *    same. When maxAniso is a power of two (Babylon default 4, three.js
 *    default 1) both readings coincide, so the choice does not affect the
 *    target suites. A degenerate footprint (one axis zero) is infinitely
 *    anisotropic → N = pow2ceil(maxAniso); both axes zero (no footprint
 *    direction) → no taps (n = 0).
 *
 *  - Tap placement: N equally weighted taps along the MAJOR footprint axis
 *    (u when ρx ≥ ρy, else v), centered on the sample UV. Tap i sits at
 *    (i+0.5)/N − 0.5 level-λ texels from the center — UV offset
 *    ((i+0.5)/N − 0.5)·2^λ / size₀. This mirrors the spec's "equally spaced
 *    samples in texture space at LOD Lamda" form with the symmetric
 *    (i+0.5)/N convention hardware uses (the spec's literal i/(N+1) − 1/2
 *    spacing is an errata-quality artifact of the GL 1.2-era text; the
 *    (i+0.5)/N form spans the full corrected footprint texel and is what
 *    ANGLE/hardware implementations of the scheme do). λ is the CLAMPED
 *    aniso-corrected LOD, so taps step in units of the level actually
 *    sampled; the level selection itself is untouched (λ′ = log2(ρ/N) —
 *    A/B-verified optimal).
 *
 *  - Each tap is a full per-level filter (bilinear — incl. the u8 fast path —
 *    or NEAREST), and LINEAR_MIPMAP_LINEAR blends the SAME level pair with
 *    the SAME f for every tap (trilinear-per-tap, as GPUs do). Shadow
 *    samplers run the depth compare per tap (inside filter2D/readTap).
 *
 * No per-tap allocation: tap accumulation and the mip-blend scratch are
 * module-level buffers (sampling is strictly sequential — no reentrancy).
 */

import type { GLenum } from './gl-enums';
import {
  TEXTURE_2D,
  NEAREST_MIPMAP_NEAREST, LINEAR_MIPMAP_NEAREST,
  NEAREST_MIPMAP_LINEAR, LINEAR_MIPMAP_LINEAR,
} from './gl-enums';
import type { SamplerState, TextureImage } from './types';

/**
 * EXT_texture_filter_anisotropic gate: 2D textures with a mipmap minification
 * filter and TEXTURE_MAX_ANISOTROPY_EXT > 1 (magnification has no anisotropy;
 * 3D/cube/2D_ARRAY are not covered by the EXT spec's 2D scheme). The same
 * condition gates computeLod's λ correction, lodGtZero's fast path, and the
 * N-tap filter stage.
 */
export function anisoGate(img: TextureImage, state: SamplerState): boolean {
  return state.maxAnisotropy > 1 && img.target === TEXTURE_2D && isMipmapMinFilter(state.minFilter);
}

/** True when minFilter selects among mip levels (anisotropy requires this). */
function isMipmapMinFilter(f: GLenum): boolean {
  return f === NEAREST_MIPMAP_NEAREST || f === LINEAR_MIPMAP_NEAREST ||
    f === NEAREST_MIPMAP_LINEAR || f === LINEAR_MIPMAP_LINEAR;
}

/** Smallest power of two ≥ x (x ≥ 1): the spec-permitted rounding of N. */
export function pow2ceilN(x: number): number {
  let p = 1;
  while (p < x) p *= 2;
  return p;
}

/**
 * Anisotropic-tap parameters, filled by computeLod immediately before the
 * filter stage on every implicit-LOD path (sampling is strictly sequential —
 * no reentrancy hazard). `n` = tap count (0 = inactive; 1 = isotropic
 * footprint → single tap); `majorU` = step along u (S) when ρx ≥ ρy, else
 * along v (T).
 */
export const anisoTapParams = { n: 0, majorU: true };

/**
 * Structural mirror of sampler.ts's SamplePlan (the per-tap filter reads
 * plan.u8Fast; the full shape is mirrored so sampler.ts's filter2D — whose
 * plan parameter is the full SamplePlan — stays assignable to AnisoFilter2D
 * under strict contravariance). Duplicated instead of a type-only import from
 * sampler.ts to keep this module's import set to types.ts + gl-enums only;
 * keep in sync with SamplePlan.
 */
interface SamplePlanLike {
  base: number;
  hi: number;
  single: boolean;
  oneTexel: boolean;
  posFilter: GLenum;
  mipMin: boolean;
  u8Fast: boolean;
}

/** Per-tap filter signature: sampler.ts passes its filter2D. */
export type AnisoFilter2D = (
  img: TextureImage, state: SamplerState, plan: SamplePlanLike, filter: GLenum,
  level: number, u: number, v: number, layer: number, shadow: boolean,
  refQ: number, out: Float32Array,
) => void;

/** Mip-blend scratch for the per-tap trilinear (module-level; sequential). */
const _mipA = new Float32Array(4);
const _mipB = new Float32Array(4);

/**
 * N-tap anisotropic filtering (2D only; see the module header). N full
 * per-level filters at equally spaced positions along the major footprint
 * axis, averaged with equal weights; trilinear taps all blend the same level
 * pair with the same f. `lambda` is the CLAMPED aniso-corrected λ (level
 * selection happened upstream); tap i's UV offset is
 * ((i+0.5)/N − 0.5)·2^λ / size₀. No allocation.
 */
export function sample2DAnisoTaps(
  filter2D: AnisoFilter2D,
  img: TextureImage, state: SamplerState, plan: SamplePlanLike, filter: GLenum,
  level0: number, level1: number, f: number,
  u: number, v: number, layer: number, shadow: boolean, refQ: number,
  lambda: number, out: Float32Array,
): void {
  const n = anisoTapParams.n;
  const majorU = anisoTapParams.majorU;
  const scale = Math.pow(2, lambda) / (majorU ? img.width : img.height);
  const invN = 1 / n;
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
  for (let i = 0; i < n; i++) {
    const off = ((i + 0.5) / n - 0.5) * scale;
    const uu = majorU ? u + off : u;
    const vv = majorU ? v : v + off;
    if (level1 >= 0) {
      filter2D(img, state, plan, filter, level0, uu, vv, layer, shadow, refQ, _mipA);
      filter2D(img, state, plan, filter, level1, uu, vv, layer, shadow, refQ, _mipB);
      out[0] += (_mipA[0] * (1 - f) + _mipB[0] * f) * invN;
      out[1] += (_mipA[1] * (1 - f) + _mipB[1] * f) * invN;
      out[2] += (_mipA[2] * (1 - f) + _mipB[2] * f) * invN;
      out[3] += (_mipA[3] * (1 - f) + _mipB[3] * f) * invN;
    } else {
      filter2D(img, state, plan, filter, level0, uu, vv, layer, shadow, refQ, _mipA);
      out[0] += _mipA[0] * invN;
      out[1] += _mipA[1] * invN;
      out[2] += _mipA[2] * invN;
      out[3] += _mipA[3] * invN;
    }
  }
}
