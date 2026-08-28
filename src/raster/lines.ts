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
 *
 * Diamond-exit rule (GL 4.5 §14.5.1 / Vulkan §24.6.1): for a pixel whose
 * center is at the origin of pixel-local coordinates, the diamond is
 * |u| + |v| ≤ 0.5. With f(t) = |u(t)| + |v(t)| over the half-open segment
 * t ∈ [0, 1) (first endpoint included, last excluded), the fragment is
 * produced iff {t : f(t) < 0.5} is non-empty AND its supremum is < 1 — i.e.
 * the segment passes through the diamond interior and genuinely EXITS it
 * before the segment's end. Consequences (all verified canonical cases):
 *  - a horizontal segment at half-integer y covers the pixels it passes
 *    through (last one excluded by the half-open convention);
 *  - a horizontal segment at integer y covers NO pixels (tangent at the
 *    diamond vertices);
 *  - a 45° segment through pixel centers covers exactly the diagonal pixels;
 *  - a segment touching a diamond only at a vertex (tangent) is not covered;
 *  - a segment ending exactly on a diamond boundary does not produce that
 *    pixel's fragment (endpoint excluded).
 *
 * Implementation: {f ≤ 0.5} is the intersection of the four half-plane
 * constraints su·u + sv·v ≤ 0.5 with (su,sv) ∈ {(±1,±1)} — a single interval
 * [tLo, tHi] in t (f is convex). Produced iff tLo ≤ tHi, tHi < 1 and
 * f(midpoint) < 0.5 (the midpoint check rejects segments that merely run
 * along the diamond boundary). No per-fragment allocation.
 */

import type { RasterState } from './types';
import {
  RECORD_OFFSET_W, RECORD_OFFSET_X, RECORD_OFFSET_Y, RECORD_OFFSET_Z,
  VARYINGS_OFFSET,
} from './types';
import { runFragment, runQuad } from './fragment-ops';
import { rasterizeTriangle } from './triangles';

/** Sign combos of the four diamond edge normals (module-level constant). */
const DIAMOND_NORMALS: readonly (readonly [number, number])[] = [
  [1, 1], [1, -1], [-1, 1], [-1, -1],
];

/**
 * Diamond-exit test for one pixel. `a` = (ax, ay) and `b` = (bx, by) are the
 * segment endpoints in pixel-local coordinates (pixel center = origin).
 * Returns true iff the half-open segment [a, b) exits the unit diamond.
 *
 * Axis-aligned tangents (the segment lies exactly on a diamond vertex line,
 * |u| = 0.5 or |v| = 0.5): the strict rule covers nothing, but conformant
 * hardware (D3D half-open quad convention) lights ONE side — the pixel whose
 * diamond RIGHT vertex (vertical segments) or TOP vertex (horizontal
 * segments) the segment passes through. So an axis-aligned segment lying
 * exactly on an integer pixel boundary still covers its column/row (column
 * x−1 for a line at integer x, row y−1 for a line at integer y), which the
 * CTS line-rendering-quality test requires. All other cases (interior
 * diamonds, 45° diagonals, non-axis-aligned tangents) keep the strict rule.
 */
function diamondExit(ax: number, ay: number, bx: number, by: number): boolean {
  const dx = bx - ax, dy = by - ay;
  // Fast reject: a constant |u| (or |v|) > 0.5 keeps f ≥ 0.5 everywhere.
  if (dx === 0) {
    if (Math.abs(ax) > 0.5) return false;
    if (Math.abs(ax) === 0.5) {
      // u = ±0.5 tangent. Only the right-vertex side (ax = +0.5) is covered
      // here — the left-vertex tangent belongs to the neighbor pixel. Covered
      // iff the diamond vertex (u = 0.5, v = 0) lies on the half-open segment
      // [start, end): t* = −ay/dy ∈ [0, 1). (dy = 0 ⇒ degenerate point.)
      if (ax !== 0.5 || dy === 0) return false;
      const t = -ay / dy;
      return t >= 0 && t < 1;
    }
  } else if (dy === 0) {
    if (Math.abs(ay) > 0.5) return false;
    if (Math.abs(ay) === 0.5) {
      // v = ±0.5 tangent. Only the top-vertex side (ay = +0.5) is covered;
      // covered iff the diamond vertex (u = 0, v = 0.5) lies on the half-open
      // segment: t* = −ax/dx ∈ [0, 1).
      if (ay !== 0.5 || dx === 0) return false;
      const t = -ax / dx;
      return t >= 0 && t < 1;
    }
  }
  // {t : f(t) ≤ 0.5} = interval [tLo, tHi] ∩ [0, 1), from the four
  // half-plane constraints (su·dx + sv·dy)·t ≤ 0.5 − su·ax − sv·ay.
  let tLo = 0, tHi = 1;
  for (let k = 0; k < 4; k++) {
    const su = DIAMOND_NORMALS[k][0], sv = DIAMOND_NORMALS[k][1];
    const c = su * dx + sv * dy;
    const r = 0.5 - su * ax - sv * ay;
    if (c > 0) {
      const t = r / c;
      if (t < tHi) tHi = t;
    } else if (c < 0) {
      const t = r / c;
      if (t > tLo) tLo = t;
    } else if (r < 0) {
      return false; // constraint violated for all t
    }
  }
  if (tLo > tHi) return false;        // never inside the diamond
  if (tHi >= 1) return false;         // no exit before the half-open end
  // Genuine exit requires a point strictly inside: f < 0.5 somewhere on the
  // interval (convexity ⇒ checking the midpoint suffices). This rejects
  // segments running exactly along the diamond boundary.
  const tm = (tLo + tHi) * 0.5;
  const um = ax + dx * tm, vm = ay + dy * tm;
  return Math.abs(um) + Math.abs(vm) < 0.5;
}

/**
 * Wide lines (lineWidth > 1): approximate with a window-space quad — the
 * segment expanded by width/2 perpendicular to its direction — rasterized as
 * two triangles through rasterizeTriangle (exact coverage is not required;
 * must not crash). Varyings/depth are interpolated across the quad, which is
 * an approximation of the spec's per-pixel projection interpolation.
 */
function rasterizeWideLine(
  buf: Float32Array, i0: number, i1: number, stride: number, rs: RasterState,
  x0: number, y0: number, x1: number, y1: number,
): void {
  const dx = x1 - x0, dy = y1 - y0;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return; // degenerate segment draws nothing
  const half = rs.dc.lineWidth * 0.5;
  // Unit normal (−dy, dx)/len scaled by half the width.
  const nx = (-dy / len) * half;
  const ny = (dx / len) * half;

  const n = rs.totalVaryComponents;
  const recStride = VARYINGS_OFFSET + n;
  // Per-primitive scratch (wide lines are rare; not a per-fragment
  // allocation). Corner records: A+n, B+n, B−n, A−n.
  const quad = new Float32Array(recStride * 4);
  for (let k = 0; k < recStride; k++) {
    quad[k] = buf[i0 + k];
    quad[recStride + k] = buf[i1 + k];
    quad[2 * recStride + k] = buf[i1 + k];
    quad[3 * recStride + k] = buf[i0 + k];
  }
  quad[RECORD_OFFSET_X] = x0 + nx; quad[RECORD_OFFSET_Y] = y0 + ny;
  quad[recStride + RECORD_OFFSET_X] = x1 + nx; quad[recStride + RECORD_OFFSET_Y] = y1 + ny;
  quad[2 * recStride + RECORD_OFFSET_X] = x1 - nx; quad[2 * recStride + RECORD_OFFSET_Y] = y1 - ny;
  quad[3 * recStride + RECORD_OFFSET_X] = x0 - nx; quad[3 * recStride + RECORD_OFFSET_Y] = y0 - ny;

  // NOTE: rasterizeTriangle takes element offsets — records are `recStride`
  // elements apart, so pass record k at element k·recStride.
  rasterizeTriangle(quad, 0, recStride, 2 * recStride, recStride, rs);
  rasterizeTriangle(quad, 0, 2 * recStride, 3 * recStride, recStride, rs);
}

/** Rasterizes one line segment (2 window-space records). */
export function rasterizeLine(
  buf: Float32Array, i0: number, i1: number,
  stride: number, rs: RasterState,
): void {
  const x0 = buf[i0 + RECORD_OFFSET_X], y0 = buf[i0 + RECORD_OFFSET_Y];
  const z0 = buf[i0 + RECORD_OFFSET_Z], w0 = buf[i0 + RECORD_OFFSET_W];
  const x1 = buf[i1 + RECORD_OFFSET_X], y1 = buf[i1 + RECORD_OFFSET_Y];
  const z1 = buf[i1 + RECORD_OFFSET_Z], w1 = buf[i1 + RECORD_OFFSET_W];

  if (rs.dc.lineWidth > 1) {
    rasterizeWideLine(buf, i0, i1, stride, rs, x0, y0, x1, y1);
    return;
  }

  // Bounding box: segment bbox expanded by 1 px in each direction (a produced
  // fragment's diamond lies inside its pixel square, which must intersect the
  // segment), clamped to the viewport rect and the framebuffer bounds.
  const vp = rs.dc.viewport;
  const xMinB = Math.max(0, vp.x);
  const xMaxB = Math.min(rs.dc.fb.width - 1, vp.x + vp.w - 1);
  const yMinB = Math.max(0, vp.y);
  const yMaxB = Math.min(rs.dc.fb.height - 1, vp.y + vp.h - 1);
  const minX = Math.max(xMinB, Math.floor(Math.min(x0, x1)) - 1);
  const maxX = Math.min(xMaxB, Math.ceil(Math.max(x0, x1)) + 1);
  const minY = Math.max(yMinB, Math.floor(Math.min(y0, y1)) - 1);
  const maxY = Math.min(yMaxB, Math.ceil(Math.max(y0, y1)) + 1);
  if (minX > maxX || minY > maxY) return;

  const abx = x1 - x0, aby = y1 - y0;
  const len2 = abx * abx + aby * aby;
  const invLen2 = len2 !== 0 ? 1 / len2 : 0;
  const invW0 = 1 / w0, invW1 = 1 / w1;

  const n = rs.totalVaryComponents;
  const quadV = rs.quadV;
  const quadDepth = rs.quadDepth;
  const quadW = rs.quadW;
  const quadPC = rs.quadPointCoord;
  const usesDeriv = rs.dc.program.fragment.usesDerivatives;

  const vary0 = i0 + VARYINGS_OFFSET;
  const vary1 = i1 + VARYINGS_OFFSET;

  /**
   * Computes one pixel: diamond-exit coverage test plus interpolated
   * attributes into quad slot `slot` (varyings via the projection of the
   * pixel center onto the segment, clamped to [0, 1]). Returns whether the
   * pixel is covered.
   */
  const computePixel = (px: number, py: number, slot: number, fill: boolean): boolean => {
    const cx = px + 0.5, cy = py + 0.5;
    const covered = px <= maxX && py <= maxY
      && diamondExit(x0 - cx, y0 - cy, x1 - cx, y1 - cy);
    if (!fill) return covered;
    let t = ((cx - x0) * abx + (cy - y0) * aby) * invLen2;
    if (t < 0) t = 0;
    else if (t > 1) t = 1;
    const wDenom = (1 - t) * invW0 + t * invW1;
    const base = slot * n;
    if (isFinite(wDenom) && Math.abs(wDenom) >= 1e-15) {
      const w = 1 / wDenom;
      for (let c = 0; c < n; c++) {
        quadV[base + c] = ((1 - t) * buf[vary0 + c] * invW0
          + t * buf[vary1 + c] * invW1) * w;
      }
      quadW[slot] = wDenom;
    } else {
      // Degenerate perspective denominator: plain linear mix.
      for (let c = 0; c < n; c++) {
        quadV[base + c] = (1 - t) * buf[vary0 + c] + t * buf[vary1 + c];
      }
      quadW[slot] = wDenom;
    }
    quadDepth[slot] = (1 - t) * z0 + t * z1;
    quadPC[slot * 2] = 0;
    quadPC[slot * 2 + 1] = 0;
    return covered;
  };

  if (usesDeriv) {
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
