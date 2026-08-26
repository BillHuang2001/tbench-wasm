/**
 * sampler.ts — texture sampling core (contract §3).
 *
 * Two entry levels:
 *  1. Generic functions (sampleTexture / sampleTextureLod / sampleTextureShadow
 *     / texelFetch) — operate on an explicit TextureImage + effective
 *     SamplerState; used by gl/ and by unit tests.
 *  2. TextureEnv (see types.ts) — per-draw binding table + codegen-facing
 *     methods that glsl/ fragment code compiles against (`ctx.tex.*`). Each
 *     method resolves its unit → (img, state), calls the generic core, and
 *     writes the result into env.out (float domain) with the raw bit
 *     patterns also visible via env.outInt / env.outUint for integer
 *     textures. NO allocation: all results land in preallocated scratch.
 *
 * Semantics (per GLES 3.0 / WebGL):
 *  - Filtering: NEAREST/LINEAR + all four mipmap combos; implicit LOD from
 *    ∂(coord)/∂x, ∂(coord)/∂y (ρ = max over axes of the projected texel
 *    footprint; λ = log₂ρ, clamped to [minLod, maxLod], + bias; level
 *    selection per filter). Zero derivatives → λ clamps to minLod.
 *  - Wrap: REPEAT, CLAMP_TO_EDGE, MIRRORED_REPEAT (per axis).
 *  - EXT_texture_filter_anisotropic: ρ_aniso = max(ρx, ρy) /
 *    min(maxAnisotropy, max(ρx,ρy)/min(ρx,ρy)) when maxAnisotropy > 1.
 *  - Shadow samplers: compare the reference against the depth texel with
 *    compareFunc; result 0 or 1 (fractional under LINEAR filtering).
 *  - Cube maps: face selection by dominant axis; seamless filtering.
 *  - sRGB formats (SRGB8, SRGB8_ALPHA8): decode to linear on sample.
 *  - Integer formats (R8I..RGBA32UI): raw texels, no filtering; results
 *    written bit-preserving (read via env.outInt/outUint).
 *  - Incomplete textures (img.complete === false) or null bindings →
 *    (0,0,0,1).
 *  - texelFetch: integer coords, explicit level, no filtering/wrap; out of
 *    range → 0.
 */

import type { GLenum } from './gl-enums';
import type { SampleCoord, SamplerState, TextureImage } from './types';

/**
 * Generic implicit-LOD sample. `coord.v` holds [u,v] (2D, 2D_ARRAY —
 * v[2]=layer) or [u,v,w] (3D, cube). `coord.dx`/`coord.dy` are per-component
 * derivatives (zero-filled scratch when unavailable). `bias` is added to the
 * computed LOD. Writes [r,g,b,a] into `out` (length ≥ 4); integer formats
 * write raw bit patterns (see module docs).
 */
export function sampleTexture(
  img: TextureImage, state: SamplerState, coord: SampleCoord,
  bias: number, out: Float32Array,
): void {
  throw new Error('not implemented: sampleTexture');
}

/** Explicit-LOD sample (textureLod; also used by vertex shaders). */
export function sampleTextureLod(
  img: TextureImage, state: SamplerState, coord: SampleCoord,
  lod: number, out: Float32Array,
): void {
  throw new Error('not implemented: sampleTextureLod');
}

/**
 * Shadow sample: compares `ref` against depth texels using
 * state.compareFunc. coord.v = [u,v] (2D), [u,v,layer] (2D_ARRAY) or
 * [u,v,w] (cube). Writes the comparison result (0..1) into out[0], 0 elsewhere.
 */
export function sampleTextureShadow(
  img: TextureImage, state: SamplerState, coord: SampleCoord,
  ref: number, bias: number, out: Float32Array,
): void {
  throw new Error('not implemented: sampleTextureShadow');
}

/**
 * texelFetch: raw texel at integer (x, y, z) and explicit level, no
 * filtering/wrap/LOD. `z` = slice for 3D, layer for 2D_ARRAY, 0 for 2D.
 * Out-of-range coords or level → 0. Cube maps are not fetchable in GLSL
 * ES 3.00 and are not supported here.
 */
export function texelFetch(
  img: TextureImage, x: number, y: number, z: number,
  level: number, out: Float32Array,
): void {
  throw new Error('not implemented: texelFetch');
}

/** Raw texel read used by the filtering core (also handy for tests). */
export function readTexel(
  img: TextureImage, level: number, face: number, x: number, y: number, z: number,
  out: Float32Array,
): void {
  throw new Error('not implemented: readTexel');
}

/**
 * Allocates the scratch (out/outInt/outUint share one ArrayBuffer) and binds
 * the per-draw unit table. Called once per draw by rasterizer.
 */
export function createTextureEnv(units: readonly (import('./types').TextureUnitBinding | null)[]): import('./types').TextureEnv {
  throw new Error('not implemented: createTextureEnv');
}

/** Helper: resolves the (img, state) for a unit, with incomplete→null. */
export function resolveUnit(env: import('./types').TextureEnv, unit: number): { img: TextureImage; state: SamplerState } | null {
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
