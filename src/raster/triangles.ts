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
 *    dc.polygonOffset.enabled (triangles only, per GL). Per spec it is NOT
 *    applied when the fragment shader writes gl_FragDepth.
 *  - Fragment evaluation: iterate 2×2 quads; compute the 4 per-pixel
 *    (varyings, depth, w) sets into rs.quadV/quadDepth/quadW, the inside
 *    mask, then runQuad(). When !usesDerivatives a per-pixel fast path
 *    (runFragment) may be used instead.
 *
 * Fill rule details:
 *  - Winding is normalized to CCW first (swap i1/i2 when the signed area is
 *    negative) so one rule applies uniformly.
 *  - A pixel center (x+0.5, y+0.5) is inside iff for every edge e:
 *    e > 0, or e == 0 and the edge is "top-or-left": B.y < A.y, or
 *    B.y == A.y and B.x < A.x (edge goes downward, or horizontal leftward).
 *    This makes the shared edge of two adjacent triangles covered by exactly
 *    one of them (crack-free).
 *  - Edge values are computed directly at each pixel center (no drift), with
 *    per-row y terms hoisted out of the inner loop.
 */

import type { RasterState } from './types';
import {
  RECORD_OFFSET_W, RECORD_OFFSET_X, RECORD_OFFSET_Y, RECORD_OFFSET_Z,
  VARYINGS_OFFSET,
} from './types';
import { runFragment, runQuad } from './fragment-ops';

/** Window-space signed area (×2) of a triangle; sign gives facing. */
export function signedArea2(
  buf: Float32Array, i0: number, i1: number, i2: number, stride: number,
): number {
  const x0 = buf[i0 + RECORD_OFFSET_X], y0 = buf[i0 + RECORD_OFFSET_Y];
  const x1 = buf[i1 + RECORD_OFFSET_X], y1 = buf[i1 + RECORD_OFFSET_Y];
  const x2 = buf[i2 + RECORD_OFFSET_X], y2 = buf[i2 + RECORD_OFFSET_Y];
  return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
}

/**
 * Computes the polygon-offset depth slope m = max(|dz/dx|, |dz/dy|) by
 * solving the window-space depth gradient system:
 *   z1−z0 = dzdx·(x1−x0) + dzdy·(y1−y0)
 *   z2−z0 = dzdx·(x2−x0) + dzdy·(y2−y0)
 * A singular system (degenerate triangle) yields 0.
 */
export function depthSlope(
  buf: Float32Array, i0: number, i1: number, i2: number, stride: number,
): number {
  const x0 = buf[i0 + RECORD_OFFSET_X], y0 = buf[i0 + RECORD_OFFSET_Y], z0 = buf[i0 + RECORD_OFFSET_Z];
  const x1 = buf[i1 + RECORD_OFFSET_X], y1 = buf[i1 + RECORD_OFFSET_Y], z1 = buf[i1 + RECORD_OFFSET_Z];
  const x2 = buf[i2 + RECORD_OFFSET_X], y2 = buf[i2 + RECORD_OFFSET_Y], z2 = buf[i2 + RECORD_OFFSET_Z];
  const dx1 = x1 - x0, dy1 = y1 - y0, dz1 = z1 - z0;
  const dx2 = x2 - x0, dy2 = y2 - y0, dz2 = z2 - z0;
  const det = dx1 * dy2 - dy1 * dx2;
  if (det === 0) return 0;
  const dzdx = (dz1 * dy2 - dy1 * dz2) / det;
  const dzdy = (dx1 * dz2 - dz1 * dx2) / det;
  return Math.max(Math.abs(dzdx), Math.abs(dzdy));
}

/** The edge function on plain [x, y] arrays:
 *  (b[0]−a[0])·(c[1]−a[1]) − (b[1]−a[1])·(c[0]−a[0]). Positive = inside for
 *  a CCW edge a→b. */
export function edgeFunction(a: number[], b: number[], c: number[]): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/**
 * Perspective-correct interpolation helper:
 *   (s·va/w[0] + t·vb/w[1] + (1−s−t)·vc/w[2]) / (s/w[0] + t/w[1] + (1−s−t)/w[2])
 * with a denominator-≈0 guard that falls back to a plain barycentric mix.
 */
export function perspectiveCorrect(
  s: number, t: number, w: number[], va: number, vb: number, vc: number,
): number {
  const denom = s / w[0] + t / w[1] + (1 - s - t) / w[2];
  if (!isFinite(denom) || Math.abs(denom) < 1e-15) {
    return s * va + t * vb + (1 - s - t) * vc;
  }
  return (s * va / w[0] + t * vb / w[1] + (1 - s - t) * vc / w[2]) / denom;
}

/** Rasterizes one triangle (3 window-space records). */
export function rasterizeTriangle(
  buf: Float32Array, i0: number, i1: number, i2: number,
  stride: number, rs: RasterState,
): void {
  let area2 = signedArea2(buf, i0, i1, i2, stride);
  if (area2 === 0) return; // degenerate (collinear / zero-area)
  // Normalize winding to CCW so the top-left rule applies uniformly.
  if (area2 < 0) {
    const t = i1; i1 = i2; i2 = t;
    area2 = -area2;
  }

  const x0 = buf[i0 + RECORD_OFFSET_X], y0 = buf[i0 + RECORD_OFFSET_Y];
  const z0 = buf[i0 + RECORD_OFFSET_Z], w0 = buf[i0 + RECORD_OFFSET_W];
  const x1 = buf[i1 + RECORD_OFFSET_X], y1 = buf[i1 + RECORD_OFFSET_Y];
  const z1 = buf[i1 + RECORD_OFFSET_Z], w1 = buf[i1 + RECORD_OFFSET_W];
  const x2 = buf[i2 + RECORD_OFFSET_X], y2 = buf[i2 + RECORD_OFFSET_Y];
  const z2 = buf[i2 + RECORD_OFFSET_Z], w2 = buf[i2 + RECORD_OFFSET_W];

  // Bounding box: pixel indices whose centers may fall inside, clamped to the
  // viewport rect and the framebuffer bounds.
  const vp = rs.dc.viewport;
  const xMinB = Math.max(0, vp.x);
  const xMaxB = Math.min(rs.dc.fb.width - 1, vp.x + vp.w - 1);
  const yMinB = Math.max(0, vp.y);
  const yMaxB = Math.min(rs.dc.fb.height - 1, vp.y + vp.h - 1);
  const minX = Math.max(xMinB, Math.floor(Math.min(x0, x1, x2)));
  const maxX = Math.min(xMaxB, Math.ceil(Math.max(x0, x1, x2)) - 1);
  const minY = Math.max(yMinB, Math.floor(Math.min(y0, y1, y2)));
  const maxY = Math.min(yMaxB, Math.ceil(Math.max(y0, y1, y2)) - 1);
  if (minX > maxX || minY > maxY) return;

  // Edge deltas in CCW order (v0→v1→v2→v0). e0(x,y) = dx01·(y−y0) − dy01·(x−x0).
  const dx01 = x1 - x0, dy01 = y1 - y0;
  const dx12 = x2 - x1, dy12 = y2 - y1;
  const dx20 = x0 - x2, dy20 = y0 - y2;

  // TOP-LEFT rule flags (applied after CCW normalization): an edge is
  // top-or-left iff it goes downward, or is horizontal and goes leftward.
  const tl0 = y1 < y0 || (y1 === y0 && x1 < x0);
  const tl1 = y2 < y1 || (y2 === y1 && x2 < x1);
  const tl2 = y0 < y2 || (y0 === y2 && x0 < x2);

  // Barycentric weights: the edge functions sum to area2 (= 2× signed area)
  // at every point. By cyclicity of the signed area, λ0 (weight of vertex 0)
  // = area(p,v1,v2)/area(v0,v1,v2) = e1/area2, λ1 = e2/area2, λ2 = e0/area2 —
  // i.e. the weights are rotated: l0=λ2, l1=λ0, l2=λ1 (see computePixel).
  const invArea = 1 / area2;
  const invW0 = 1 / w0, invW1 = 1 / w1, invW2 = 1 / w2;

  // Polygon offset (triangles only, per GL). Not applied when the shader
  // writes gl_FragDepth. r = 2^-24.
  let polyOffset = 0;
  const po = rs.dc.polygonOffset;
  if (po.enabled && !rs.dc.program.fragment.usesFragDepth) {
    polyOffset = depthSlope(buf, i0, i1, i2, stride) * po.factor
      + po.units * 5.960464477539063e-8;
  }

  const n = rs.totalVaryComponents;
  const quadV = rs.quadV;
  const quadDepth = rs.quadDepth;
  const quadW = rs.quadW;
  const quadPC = rs.quadPointCoord;
  const usesDeriv = rs.dc.program.fragment.usesDerivatives;

  const vary0 = i0 + VARYINGS_OFFSET;
  const vary1 = i1 + VARYINGS_OFFSET;
  const vary2 = i2 + VARYINGS_OFFSET;

  /**
   * Computes the edge values and interpolated attributes for one pixel at
   * (px, py), storing the attribute set into quad slot `slot`
   * (quadV[slot·n .. slot·n+n), quadDepth[slot], quadW[slot],
   * quadPointCoord[2·slot .. 2·slot+2] = 0 for triangles).
   * Returns whether the pixel center is inside the triangle (top-left rule,
   * restricted to the rasterization bbox so quads straddling the bbox edge do
   * not mark out-of-bounds pixels as inside).
   */
  const computePixel = (px: number, py: number, slot: number, fill: boolean): boolean => {
    const cx = px + 0.5, cy = py + 0.5;
    const e0 = dx01 * (cy - y0) - dy01 * (cx - x0);
    const e1 = dx12 * (cy - y1) - dy12 * (cx - x1);
    const e2 = dx20 * (cy - y2) - dy20 * (cx - x2);
    const inside = px <= maxX && py <= maxY
      && (e0 > 0 || (e0 === 0 && tl0))
      && (e1 > 0 || (e1 === 0 && tl1))
      && (e2 > 0 || (e2 === 0 && tl2));
    if (!fill) return inside;
    const l0 = e0 * invArea, l1 = e1 * invArea, l2 = e2 * invArea;
    // Weight rotation: e0 = E(v0→v1) pairs with vertex 2, e1 = E(v1→v2) with
    // vertex 0, e2 = E(v2→v0) with vertex 1 (see the barycentric comment).
    const wDenom = l1 * invW0 + l2 * invW1 + l0 * invW2;
    const base = slot * n;
    if (isFinite(wDenom) && Math.abs(wDenom) >= 1e-15) {
      const w = 1 / wDenom;
      for (let c = 0; c < n; c++) {
        quadV[base + c] = (l1 * buf[vary0 + c] * invW0
          + l2 * buf[vary1 + c] * invW1
          + l0 * buf[vary2 + c] * invW2) * w;
      }
      quadW[slot] = wDenom;
    } else {
      // Degenerate perspective denominator: plain barycentric mix.
      for (let c = 0; c < n; c++) {
        quadV[base + c] = l1 * buf[vary0 + c] + l2 * buf[vary1 + c] + l0 * buf[vary2 + c];
      }
      quadW[slot] = wDenom;
    }
    quadDepth[slot] = l1 * z0 + l2 * z1 + l0 * z2 + polyOffset;
    quadPC[slot * 2] = 0;
    quadPC[slot * 2 + 1] = 0;
    return inside;
  };

  if (usesDeriv) {
    // 2×2 quads: helpers (outside pixels) still need valid values, so every
    // slot is filled; the inside mask selects the real fragments.
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
    // Fast path: per-pixel fragment execution (no derivative support needed).
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
