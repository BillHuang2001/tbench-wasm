/**
 * points.ts — point rasterization.
 *
 * Input: ONE clipped point record (window coords, pointSize in the record).
 *
 *  - Point size is clamped to ALIASED_POINT_SIZE_RANGE ([1, 1024] — the
 *    constants live in types.ts; gl/ reports the same values). Clamping
 *    happens at rasterization (gl_PointSize written by the VS is clamped
 *    here, per GLES).
 *  - Points rasterize as SQUARES: a fragment with center (xf+0.5, yf+0.5)
 *    is produced when its center lies inside
 *    [xc − s/2, xc + s/2) × [yc − s/2, yc + s/2). A size-1 point whose
 *    center falls exactly on a pixel center must still produce that pixel
 *    (implement the half-open rule so the containing pixel is covered).
 *  - gl_PointCoord: s = (xf + 0.5 − (xc − s/2)) / s, t likewise (0..1
 *    within the point; values at the far edges may reach exactly 1.0).
 *  - Fragments outside the viewport (and outside scissor, handled by
 *    FragmentOps) are clipped. Varyings are constant (provoking vertex =
 *    the point itself). Depth = window z of the point; gl_FragCoord.w =
 *    1/w_clip.
 *  - Points are NOT subject to polygon offset or culling.
 */

import type { RasterState } from './types';

/** Rasterizes one point (single window-space record). */
export function rasterizePoint(
  buf: Float32Array, i: number, stride: number, rs: RasterState,
): void {
  throw new Error('not implemented: rasterizePoint');
}
