/**
 * clip.ts — homogeneous clipping against all 6 clip planes (-w ≤ x,y,z ≤ w)
 * plus the viewport transform.
 *
 * Clipping interpolates EVERY record field (x, y, z, w, pointSize, varyings)
 * linearly in clip space, preserving perspective correctness after the
 * divide. Flat varyings are made constant BEFORE clipping (the rasterizer
 * copies provoking-vertex values into every vertex). Triangle results are
 * written in polygon order (fan-able from vertex 0); line results are 0 or
 * 2 vertices; no allocation in the draw path.
 */
import type { DepthRange, Viewport } from './types';

/** Record header offsets (must match types.ts). */
const X = 0;
const Y = 1;
const Z = 2;
const W = 3;

/** The 6 clip planes as affine f = a·x + b·y + c·z + d·w ≥ 0 (inside). Fixed order. */
const CLIP_PLANES: readonly (readonly [number, number, number, number])[] = [
  [1, 0, 0, 1],   // x ≥ -w → x + w ≥ 0
  [-1, 0, 0, 1],  // x ≤  w → w - x ≥ 0
  [0, 1, 0, 1],   // y ≥ -w → y + w ≥ 0
  [0, -1, 0, 1],  // y ≤  w → w - y ≥ 0
  [0, 0, 1, 1],   // z ≥ -w → z + w ≥ 0
  [0, 0, -1, 1],  // z ≤  w → w - z ≥ 0
];

/** Plane function value at a record (double precision). */
function planeAt(
  plane: readonly [number, number, number, number],
  buf: Float32Array, idx: number,
): number {
  return plane[0] * buf[idx + X] + plane[1] * buf[idx + Y] +
    plane[2] * buf[idx + Z] + plane[3] * buf[idx + W];
}

/** Interpolates every field of record b - a at parameter t into dst. */
function interpRecord(
  src: Float32Array, aIdx: number, bIdx: number, t: number,
  stride: number, dst: Float32Array, dstIdx: number,
): void {
  for (let k = 0; k < stride; k++) {
    dst[dstIdx + k] = src[aIdx + k] + t * (src[bIdx + k] - src[aIdx + k]);
  }
}

/** Copies one full record. */
function copyRecord(
  src: Float32Array, srcIdx: number,
  dst: Float32Array, dstIdx: number, stride: number,
): void {
  for (let k = 0; k < stride; k++) dst[dstIdx + k] = src[srcIdx + k];
}

/**
 * One S-H pass: clips the n-vertex polygon at src[srcBase] against `plane`,
 * writing the result to dst[dstBase]; returns the vertex count (0 = fully
 * clipped), clamped to maxRecords. t = f(prev)/(f(prev) - f(curr)) is
 * computed only when endpoints are strictly on opposite sides (no NaN);
 * crossings with t ∈ {0,1} coincide with a vertex touching the plane and are
 * skipped, lest they duplicate a vertex and break "≤ input+1".
 */
function clipPlanePass(
  plane: readonly [number, number, number, number],
  src: Float32Array, srcBase: number, n: number, stride: number,
  dst: Float32Array, dstBase: number, maxRecords: number,
): number {
  let outCount = 0;
  let prevIdx = srcBase + (n - 1) * stride;
  let prevF = planeAt(plane, src, prevIdx);
  for (let i = 0; i < n; i++) {
    const currIdx = srcBase + i * stride;
    const currF = planeAt(plane, src, currIdx);
    if (currF >= 0) {
      if (prevF < 0) {
        const t = prevF / (prevF - currF);
        if (t > 0 && t < 1) {
          if (outCount >= maxRecords) break; // destination full: truncate
          interpRecord(src, prevIdx, currIdx, t, stride, dst, dstBase + outCount * stride);
          outCount++;
        }
      }
      if (outCount >= maxRecords) break; // destination full: truncate
      copyRecord(src, currIdx, dst, dstBase + outCount * stride, stride);
      outCount++;
    } else if (prevF >= 0) {
      const t = prevF / (prevF - currF);
      if (t > 0 && t < 1) {
        if (outCount >= maxRecords) break; // destination full: truncate
        interpRecord(src, prevIdx, currIdx, t, stride, dst, dstBase + outCount * stride);
        outCount++;
      }
    }
    prevIdx = currIdx;
    prevF = currF;
  }
  return outCount;
}

/**
 * Clips one primitive (count = 3 triangle or 2 line records) against the 6
 * clip planes; writes the polygon into `out` at outBase (record-aligned) and
 * returns the vertex count (0 = fully clipped).
 *
 * BUFFER SIZES: each S-H pass emits ≤ input+1 vertices, so a triangle can
 * reach 9 after 6 passes (triangle∩clip-box can be a nonagon; "7" is the 2D
 * triangle-vs-rectangle bound) — `scratch` ≥ 8 and `out` ≥ 9 records are
 * required for full correctness; per-pass output is clamped to the
 * destination's capacity (undersized callers get a truncated polygon, not
 * corruption). `buf`/`scratch`/`out` must be distinct; triangle passes
 * ping-pong buf → scratch → out[0..] → scratch → out[outBase..], the final
 * pass reads scratch, so there is never read/write aliasing (even outBase 0).
 *
 * The count == 2 (line) path uses a segment-parameter clip instead of S-H:
 * the visible portion is tracked as a per-plane parameter interval [t0, t1]
 * along the segment (required for segments whose endpoints are both outside
 * while the interior crosses the volume). A zero-length segment (identical
 * positions) or a collapsed interval (t0 >= t1) yields 0 vertices.
 */
export function clipPrimitive(
  buf: Float32Array, base: number, stride: number, count: number,
  scratch: Float32Array, out: Float32Array, outBase: number,
): number {
  if (count === 2) {
    // Segment clip: the intersection of a segment with the convex clip
    // region is a segment, tracked as a parameter interval [t0, t1] along
    // it. (Polygon S-H cannot represent a segment whose endpoints are both
    // outside while the middle is visible.)
    const outCap = (out.length - outBase) / stride;
    if (outCap < 2) return 0;
    const aIdx = base;
    const bIdx = base + stride;
    // Zero-length segment (identical positions): nothing to rasterize.
    if (buf[aIdx + X] === buf[bIdx + X] &&
        buf[aIdx + Y] === buf[bIdx + Y] &&
        buf[aIdx + Z] === buf[bIdx + Z] &&
        buf[aIdx + W] === buf[bIdx + W]) return 0;
    let t0 = 0;
    let t1 = 1;
    for (let p = 0; p < CLIP_PLANES.length; p++) {
      const plane = CLIP_PLANES[p];
      const fa = planeAt(plane, buf, aIdx);
      const fb = planeAt(plane, buf, bIdx);
      if (fa < 0 && fb < 0) return 0;
      if (fa < 0) {
        const t = fa / (fa - fb);
        if (t > t0) t0 = t;
      } else if (fb < 0) {
        const t = fa / (fa - fb);
        if (t < t1) t1 = t;
      }
      if (t0 >= t1) return 0; // collapsed (tangent/point-touch): zero-length result
    }
    interpRecord(buf, aIdx, bIdx, t0, stride, out, outBase);
    interpRecord(buf, aIdx, bIdx, t1, stride, out, outBase + stride);
    return 2;
  }

  // Polygon (triangle) clip: one pass per plane. Intermediate passes
  // ping-pong between `scratch` and the head of `out` (after pass k a
  // triangle has ≤ 3+k vertices); the final pass reads scratch and writes
  // the result at outBase, so it never aliases its own source. Per-pass
  // output is clamped to the destination's actual capacity (safety net for
  // undersized callers).
  const scratchCap = Math.floor(scratch.length / stride);
  const outCap = Math.floor(out.length / stride);
  const outBaseCap = Math.floor((out.length - outBase) / stride);
  let src: Float32Array = buf;
  let srcBase = base;
  let n = count;
  for (let p = 0; p < CLIP_PLANES.length; p++) {
    let dst: Float32Array;
    let dstBase: number;
    let cap: number;
    if (p === CLIP_PLANES.length - 1) {
      dst = out; dstBase = outBase; cap = outBaseCap;
    } else if (p % 2 === 0) {
      dst = scratch; dstBase = 0; cap = scratchCap;
    } else {
      dst = out; dstBase = 0; cap = outCap;
    }
    n = clipPlanePass(CLIP_PLANES[p], src, srcBase, n, stride, dst, dstBase, cap);
    if (n === 0) return 0;
    src = dst;
    srcBase = dstBase;
  }
  return n;
}

/**
 * Point visibility: true when w > 0 and the point satisfies -w ≤ z ≤ w.
 * GLES 2.0 §2.13: points are clipped ONLY against the near/far planes — an
 * x/y-outside center passes and the point square is clipped to the viewport
 * by the rasterizer's bbox clamp (points are not polygon-clipped).
 */
export function pointIsVisible(buf: Float32Array, base: number, stride: number): boolean {
  const z = buf[base + Z];
  const w = buf[base + W];
  return w > 0 && z >= -w && z <= w;
}

/**
 * In-place viewport transform of `count` records at `base` (post-clip):
 * divides by w, maps to window coordinates, keeps clip w in slot 3
 * (perspective interpolation, gl_FragCoord.w). Depth maps through
 * depthRange, clamped to [0,1] (polygon offset is applied later per
 * fragment, in the rasterizers).
 */
export function applyViewportTransform(
  buf: Float32Array, base: number, stride: number, count: number,
  viewport: Viewport, depthRange: DepthRange,
): void {
  for (let i = 0; i < count; i++) {
    const idx = base + i * stride;
    const x = buf[idx + X];
    const y = buf[idx + Y];
    const z = buf[idx + Z];
    const w = buf[idx + W];
    let winX: number;
    let winY: number;
    let winZ: number;
    if (w === 0) {
      // Degenerate clip-space origin (point at infinity) surviving the clip:
      // guard against NaN by placing it at the viewport center.
      winX = viewport.x + viewport.w * 0.5;
      winY = viewport.y + viewport.h * 0.5;
      winZ = (depthRange.near + depthRange.far) * 0.5;
    } else {
      const invW = 1 / w;
      winX = viewport.x + (x * invW * 0.5 + 0.5) * viewport.w;
      winY = viewport.y + (y * invW * 0.5 + 0.5) * viewport.h;
      winZ = depthRange.near + (depthRange.far - depthRange.near) * (z * invW * 0.5 + 0.5);
    }
    if (winZ < 0) winZ = 0;
    else if (winZ > 1) winZ = 1;
    buf[idx + X] = winX;
    buf[idx + Y] = winY;
    buf[idx + Z] = winZ;
    // Slot W keeps the ORIGINAL clip w.
  }
}

/**
 * Pinned contract value: a TRIANGLE's true worst case is 9 vertices
 * (nonagon — see clipPrimitive), so size `out` ≥ 9 and `scratch` ≥ 8
 * records; retained as the minimum safe to READ as a fan.
 */
export const MAX_CLIPPED_VERTICES = 7;

/**
 * Linear interpolation of two attribute records at parameter t: a + t·(b-a).
 * Convenience/test helper — ALLOCATES (hot paths use clipPrimitive).
 */
export function clipInterpolate(a: number[], b: number[], t: number): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + t * (b[i] - a[i]);
  return out;
}
