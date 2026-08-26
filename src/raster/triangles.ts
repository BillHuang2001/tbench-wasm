/**
 * triangles.ts — triangle rasterization.
 *
 * Input: ONE clipped + viewport-transformed triangle (3 records at i0/i1/i2,
 * window coords, w = clip w preserved). Output: fragments via runQuad().
 *
 * Algorithm requirements (see CONTEXT.md for rationale):
 *  - Edge functions with the GL TOP-LEFT fill rule (pixel centers at
 *    x+0.5, y+0.5); bounding box clamped to viewport ∪ surface bounds.
 *  - Backface culling is done by the DRAW DRIVER (needs winding parity for
 *    strips) — this module assumes the triangle is not culled.
 *  - Perspective-correct varying interpolation: window-space barycentrics λi,
 *    value = (Σ λi·vi/wi) / (Σ λi/wi); window depth z = Σ λi·zi (linear in
 *    window space, NOT perspective-corrected); gl_FragCoord.w = 1/(Σ λi/wi).
 *  - Flat varyings are constant (provoking-vertex values were copied to all
 *    records pre-clip): the interpolation still works since all three values
 *    are identical.
 *  - Polygon offset: per-fragment depth offset = m·factor + r·units with
 *    m = max(|∂z/∂x|, |∂z/∂y|) (constant per triangle, computed from the
 *    window-space depth gradient) and r = 2^-24; applied when
 *    dc.polygonOffset.enabled (triangles only, per GL).
 *  - Fragment evaluation: iterate 2×2 quads; compute the 4 per-pixel
 *    (varyings, depth, w) sets into rs.quadV/quadDepth/quadW, the inside
 *    mask, then runQuad(). When !usesDerivatives a per-pixel fast path
 *    (runFragment) may be used instead.
 */

import type { RasterState } from './types';

/** Rasterizes one triangle (3 window-space records). */
export function rasterizeTriangle(
  buf: Float32Array, i0: number, i1: number, i2: number,
  stride: number, rs: RasterState,
): void {
  throw new Error('not implemented: rasterizeTriangle');
}

/** Computes the polygon-offset depth slope m = max(|dz/dx|, |dz/dy|). */
export function depthSlope(
  buf: Float32Array, i0: number, i1: number, i2: number, stride: number,
): number {
  throw new Error('not implemented: depthSlope');
}

/** Window-space signed area (×2) of a triangle; sign gives facing. */
export function signedArea2(
  buf: Float32Array, i0: number, i1: number, i2: number, stride: number,
): number {
  throw new Error('not implemented: signedArea2');
}
