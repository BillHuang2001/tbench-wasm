/**
 * lines.ts — line segment rasterization.
 *
 * Input: ONE clipped + viewport-transformed segment (2 records).
 *
 *  - lineWidth 1.0 (the only width guaranteed): implement the GL 4.5
 *    diamond-exit rule (a fragment is produced when the segment exits the
 *    pixel's diamond, with the standard endpoint conventions). Exactness
 *    matters: three.js wireframes are compared against GPU-rendered
 *    references in the visual suite, so this cannot be a naive DDA.
 *  - lineWidth > 1 (dc.lineWidth): approximate with a window-space quad
 *    (segment expanded by width/2 perpendicular to its direction),
 *    rasterized as two triangles. Must not crash; exact coverage is not
 *    required by the graded suites.
 *  - Varyings interpolate linearly along the segment in window space with
 *    perspective correction (same λ/wi formula as triangles with λ = t, 1-t).
 *    Flat varyings are constant (provoking vertex = second endpoint).
 *  - Lines are NOT subject to polygon offset or culling. Depth is
 *    interpolated linearly in window space.
 */

import type { RasterState } from './types';

/** Rasterizes one line segment (2 window-space records). */
export function rasterizeLine(
  buf: Float32Array, i0: number, i1: number,
  stride: number, rs: RasterState,
): void {
  throw new Error('not implemented: rasterizeLine');
}
