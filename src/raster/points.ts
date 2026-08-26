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
import {
  ALIASED_POINT_SIZE_RANGE, RECORD_OFFSET_POINT_SIZE, RECORD_OFFSET_W,
  RECORD_OFFSET_X, RECORD_OFFSET_Y, RECORD_OFFSET_Z, VARYINGS_OFFSET,
} from './types';
import { runFragment, runQuad } from './fragment-ops';

/** Rasterizes one point (single window-space record). */
export function rasterizePoint(
  buf: Float32Array, i: number, stride: number, rs: RasterState,
): void {
  const xc = buf[i + RECORD_OFFSET_X];
  const yc = buf[i + RECORD_OFFSET_Y];
  const z = buf[i + RECORD_OFFSET_Z];
  const clipW = buf[i + RECORD_OFFSET_W];

  // Clamp gl_PointSize to ALIASED_POINT_SIZE_RANGE at rasterization
  // (also guards NaN/negative values from the vertex shader).
  let size = buf[i + RECORD_OFFSET_POINT_SIZE];
  if (!(size >= ALIASED_POINT_SIZE_RANGE[0])) size = ALIASED_POINT_SIZE_RANGE[0];
  else if (size > ALIASED_POINT_SIZE_RANGE[1]) size = ALIASED_POINT_SIZE_RANGE[1];

  const half = size * 0.5;
  const left = xc - half, right = xc + half;
  const bottom = yc - half, top = yc + half;
  const invSize = 1 / size;
  // gl_FragCoord.w = 1/w_clip.
  const w = clipW !== 0 ? 1 / clipW : 1;

  // Bounding box of pixel centers that can satisfy the half-open coverage
  // test, clamped to the viewport rect and the framebuffer bounds.
  const vp = rs.dc.viewport;
  const xMinB = Math.max(0, vp.x);
  const xMaxB = Math.min(rs.dc.fb.width - 1, vp.x + vp.w - 1);
  const yMinB = Math.max(0, vp.y);
  const yMaxB = Math.min(rs.dc.fb.height - 1, vp.y + vp.h - 1);
  const minX = Math.max(xMinB, Math.floor(left));
  const maxX = Math.min(xMaxB, Math.ceil(right) - 1);
  const minY = Math.max(yMinB, Math.floor(bottom));
  const maxY = Math.min(yMaxB, Math.ceil(top) - 1);
  if (minX > maxX || minY > maxY) return;

  const n = rs.totalVaryComponents;
  const quadV = rs.quadV;
  const quadDepth = rs.quadDepth;
  const quadW = rs.quadW;
  const quadPC = rs.quadPointCoord;
  const usesDeriv = rs.dc.program.fragment.usesDerivatives;

  const vary = i + VARYINGS_OFFSET;

  /**
   * Computes one pixel: half-open square coverage test plus the (constant)
   * attribute set into quad slot `slot`: varyings copied from the point
   * record, depth = point z, w = 1/w_clip, pointCoord per pixel. Returns
   * whether the pixel is covered.
   */
  const computePixel = (px: number, py: number, slot: number, fill: boolean): boolean => {
    const cx = px + 0.5, cy = py + 0.5;
    const inside = px <= maxX && py <= maxY
      && cx >= left && cx < right
      && cy >= bottom && cy < top;
    if (!fill) return inside;
    const base = slot * n;
    for (let c = 0; c < n; c++) quadV[base + c] = buf[vary + c];
    quadDepth[slot] = z;
    quadW[slot] = w;
    quadPC[slot * 2] = (cx - left) * invSize;
    quadPC[slot * 2 + 1] = (cy - bottom) * invSize;
    return inside;
  };

  if (usesDeriv) {
    // 2×2 quads: helpers (outside pixels) still need valid values — rows are
    // identical (constant varyings), pointCoord varies per pixel.
    for (let qy = minY; qy <= maxY; qy += 2) {
      for (let qx = minX; qx <= maxX; qx += 2) {
        let mask = 0;
        if (computePixel(qx, qy, 0, true)) mask |= 1;
        if (computePixel(qx + 1, qy, 1, true)) mask |= 2;
        if (computePixel(qx, qy + 1, 2, true)) mask |= 4;
        if (computePixel(qx + 1, qy + 1, 3, true)) mask |= 8;
        if (mask !== 0) runQuad(rs, qx, qy, mask);
      }
    }
  } else {
    for (let qy = minY; qy <= maxY; qy += 2) {
      for (let qx = minX; qx <= maxX; qx += 2) {
        if (computePixel(qx, qy, 0, true)) {
          runFragment(rs, qx, qy, quadDepth[0], quadW[0], 0);
        }
        if (computePixel(qx + 1, qy, 0, true)) {
          runFragment(rs, qx + 1, qy, quadDepth[0], quadW[0], 0);
        }
        if (computePixel(qx, qy + 1, 0, true)) {
          runFragment(rs, qx, qy + 1, quadDepth[0], quadW[0], 0);
        }
        if (computePixel(qx + 1, qy + 1, 0, true)) {
          runFragment(rs, qx + 1, qy + 1, quadDepth[0], quadW[0], 0);
        }
      }
    }
  }
}
