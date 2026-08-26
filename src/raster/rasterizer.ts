/**
 * rasterizer.ts — the draw driver (contract §2 entry point).
 *
 * draw(dc) flow:
 *  1. If dc.rasterizerDiscard → return (no fragments; the fragment shader is
 *     not executed at all).
 *  2. Build the per-draw RasterState (fragment ctx, FragmentOps, scratch,
 *     texture env) — allocated once per draw, never per fragment.
 *  3. Primitive assembly by dc.mode over [first, first+count):
 *       TRIANGLES        (v, v+1, v+2)   — 3i
 *       TRIANGLE_STRIP   (v+i, v+i+1, v+i+2), winding flips on odd i
 *       TRIANGLE_FAN     (v, v+i+1, v+i+2)
 *       LINES / LINE_STRIP / LINE_LOOP / POINTS
 *     Assembled records are copied into a scratch primitive buffer so that
 *     flat provoking-vertex fixup and clipping can mutate them freely.
 *  4. For each instance i: vertex base = first + i*count (records).
 *  5. Per primitive: copy provoking-vertex (LAST vertex per GLES) flat
 *     varying values into every vertex; clip (all 6 planes, clip.ts);
 *     apply the viewport transform; cull (window-space signed area vs
 *     cull.face/frontFace — strips need the winding parity); dispatch to
 *     rasterizeTriangle / rasterizeLine / rasterizePoint.
 *
 * Culling happens AFTER clipping per spec. `first` semantics and instancing
 * addressing are documented on DrawCall in types.ts.
 */

import type { DrawCall, RasterState } from './types';

/** Rasterizes one draw call (contract §2 `rasterizer.draw`). */
export function draw(dc: DrawCall): void {
  throw new Error('not implemented: draw');
}

/**
 * Builds the per-draw execution state: FragmentExecCtx (varyings arrays,
 * texture env + scratch), FragmentOps, and the quad scratch buffers.
 * Allocates once per draw call.
 */
export function createRasterState(dc: DrawCall): RasterState {
  throw new Error('not implemented: createRasterState');
}

/** Copies the provoking vertex's flat-varying values into all vertices of a
 *  primitive (called before clipping). Provoking vertex = LAST vertex. */
export function applyFlatFixup(
  buf: Float32Array, base: number, count: number, stride: number,
  varyingsOffset: number, flatRanges: readonly (readonly [number, number])[],
): void {
  throw new Error('not implemented: applyFlatFixup');
}
