/**
 * clip.ts — homogeneous clipping against all 6 clip planes
 * (-w ≤ x,y,z ≤ w) plus the viewport transform.
 *
 * Clipping interpolates EVERY record field (x, y, z, w, pointSize, varyings)
 * linearly in clip space, which is exactly what preserves perspective
 * correctness after the divide. Flat varyings are made constant BEFORE
 * clipping (rasterizer copies the provoking-vertex values into every vertex
 * of the primitive), so interpolating them here is harmless.
 *
 * Output polygons are fan-able: triangle clip results are written in
 * polygon order (rasterize as a fan from vertex 0); line clip results are
 * 0 or 2 vertices. Worst-case triangle → 7 vertices (convex clip of a
 * triangle against 6 planes).
 *
 * All buffers are caller-owned scratch — no allocation in the draw path.
 */

import type { DepthRange, Viewport } from './types';

/**
 * Clips one primitive (count = 3 triangle or 2 line records) against the 6
 * clip planes. Writes the resulting polygon into `out` starting at outBase
 * (record-aligned). Returns the number of vertices written (0 = fully
 * clipped). `scratch` must hold ≥ 2×count records; `out` must hold ≥ 7
 * records.
 */
export function clipPrimitive(
  buf: Float32Array, base: number, stride: number, count: number,
  scratch: Float32Array, out: Float32Array, outBase: number,
): number {
  throw new Error('not implemented: clipPrimitive');
}

/**
 * Point visibility: true when the point's clip coords satisfy
 * -w ≤ x,y,z ≤ w (points are not polygon-clipped).
 */
export function pointIsVisible(buf: Float32Array, base: number, stride: number): boolean {
  throw new Error('not implemented: pointIsVisible');
}

/**
 * In-place viewport transform of `count` records starting at `base`
 * (post-clip, clip-space): divides by w and maps to window coordinates.
 * After this call each record is [winX, winY, winZ, clipW, pointSize, ...]
 * — clip w is PRESERVED in slot 3 (needed for perspective interpolation and
 * gl_FragCoord.w). Depth maps through depthRange: winZ = near + (far-near) *
 * (z/w*0.5+0.5), clamped to [0,1] after offsetting (polygon offset is
 * applied later, per fragment, in the rasterizers).
 */
export function applyViewportTransform(
  buf: Float32Array, base: number, stride: number, count: number,
  viewport: Viewport, depthRange: DepthRange,
): void {
  throw new Error('not implemented: applyViewportTransform');
}

/** Worst-case output vertices for clipPrimitive (triangle → hexagon). */
export const MAX_CLIPPED_VERTICES = 7;
