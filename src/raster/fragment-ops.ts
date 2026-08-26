/**
 * fragment-ops.ts — per-fragment operations (scissor, sample coverage,
 * stencil, depth, blend, dither, sRGB, colorMask) + the quad fragment driver
 * + clear/blit helpers for gl/.
 *
 * Order of operations per GLES 3.0 §4.1:
 *   scissor test → sample coverage → stencil test → depth test → blending →
 *   dithering → (sRGB re-encode) → color write.
 * The two-phase protocol (FragmentOps.test / FragmentOps.finalize, see
 * types.ts) defers stencil-zpass/depth-write until AFTER the fragment shader
 * runs, so `discard` suppresses every write and gl_FragDepth shaders get a
 * post-shader depth test.
 *
 * Blending happens in LINEAR space when the target is sRGB (destination
 * decoded before the blend, result re-encoded before the write).
 */

import type { ColorMask, DrawCall, FragmentExecCtx, FragmentOps, RasterState, ScissorState, Surface } from './types';

/**
 * Per-draw fragment ops engine. Constructed once per draw call by
 * rasterizer.createRasterState; methods are called per fragment.
 */
export class FragmentOpsImpl implements FragmentOps {
  constructor(dc: DrawCall) {
    throw new Error('not implemented: FragmentOpsImpl constructor');
  }

  /** Scissor → coverage → stencil test → depth read. See FragmentOps. */
  test(x: number, y: number, frontFacing: boolean, depth: number): boolean {
    throw new Error('not implemented: FragmentOps.test');
  }

  /** Depth write + stencil zpass + blend + dither + sRGB + colorMask + write. */
  finalize(
    x: number, y: number, frontFacing: boolean, depth: number,
    colors: readonly Float32Array[],
  ): void {
    throw new Error('not implemented: FragmentOps.finalize');
  }
}

/**
 * Quad fragment driver. Runs the fragment shader for a 2×2 quad of pixels
 * with origins at (qx, qy) (pixels (qx,qy),(qx+1,qy),(qx,qy+1),(qx+1,qy+1)).
 * `rs.quadV/quadDepth/quadW/quadPointCoord` hold the 4 precomputed per-pixel
 * values (see RasterState) and `rs.frontFacing` the primitive facing.
 * `inside` is a 4-bit mask (bit p = pixel p inside the primitive).
 *
 *  - Inside pixels: ops.test → shader → (if not discarded) ops.finalize.
 *  - Outside pixels are HELPER INVOCATIONS (run whenever
 *    program.fragment.usesDerivatives so inside pixels get correct
 *    derivatives); their outputs are always discarded.
 *  - Per-invocation ctx setup: fragCoord, varyings (+ddx/ddy from the quad),
 *    pointCoord, discarded=false; helper pixels may read anything but must
 *    not affect the target.
 */
export function runQuad(rs: RasterState, qx: number, qy: number, inside: number): void {
  throw new Error('not implemented: runQuad');
}

/** Runs the fragment shader for a single pixel (non-quad fast path — only
 *  valid when !program.fragment.usesDerivatives). */
export function runFragment(
  rs: RasterState, x: number, y: number, depth: number, w: number,
  varyBase: number,
): void {
  throw new Error('not implemented: runFragment');
}

/* ================================================================== */
/* Clear (glClear / glClearBuffer*) — respects scissor + masks         */
/* ================================================================== */

export function clearColorSurface(
  s: Surface, r: number, g: number, b: number, a: number,
  scissor: ScissorState | null, mask: ColorMask,
): void {
  throw new Error('not implemented: clearColorSurface');
}

export function clearDepthSurface(
  s: Surface, depth: number, scissor: ScissorState | null, depthMask: boolean,
): void {
  throw new Error('not implemented: clearDepthSurface');
}

export function clearStencilSurface(
  s: Surface, value: number, scissor: ScissorState | null, writeMask: number,
): void {
  throw new Error('not implemented: clearStencilSurface');
}

/* ================================================================== */
/* Blit (blitFramebuffer) — color with optional filtering,             */
/* depth/stencil plain copies.                                         */
/* ================================================================== */

export function blitColorSurface(
  src: Surface, dst: Surface,
  srcX: number, srcY: number, srcW: number, srcH: number,
  dstX: number, dstY: number, dstW: number, dstH: number,
  filter: 'nearest' | 'linear',
): void {
  throw new Error('not implemented: blitColorSurface');
}

export function blitDepthStencilSurface(
  src: Surface, dst: Surface,
  srcX: number, srcY: number, srcW: number, srcH: number,
  dstX: number, dstY: number, dstW: number, dstH: number,
): void {
  throw new Error('not implemented: blitDepthStencilSurface');
}

/* ================================================================== */
/* Blend internals (exported for unit tests)                           */
/* ================================================================== */

/** Computes one blended RGBA result (linear space) from src + dst. */
export function blendColor(
  src: Float32Array, dst: Float32Array, out: Float32Array,
  srcRGB: number, dstRGB: number, srcAlpha: number, dstAlpha: number,
  eqRGB: number, eqAlpha: number, constColor: [number, number, number, number],
): void {
  throw new Error('not implemented: blendColor');
}

/** Convenience for gl/: builds the FragmentOps instance for a draw call. */
export function createFragmentOps(dc: DrawCall): FragmentOps {
  return new FragmentOpsImpl(dc);
}

/** Internal helper used by runQuad/runFragment: fills ctx for one pixel. */
export function setupFragmentCtx(
  ctx: FragmentExecCtx, x: number, y: number, depth: number, w: number,
  quadV: Float32Array, quadStride: number, pixel: number,
): void {
  throw new Error('not implemented: setupFragmentCtx');
}
