/**
 * surface.ts — surface creation and view resolution.
 *
 * Surfaces are owned by gl/ (drawing buffer, renderbuffers, texture levels
 * attached to FBOs). This module is the single place that knows how a format
 * maps to storage planes (see the Surface representation rules in types.ts),
 * so gl/ calls createSurface() instead of allocating raw arrays.
 */

import type { Surface } from './types';
import { getFormat } from './formats';
import type { GLenum } from './gl-enums';

/**
 * Allocates a surface for `format` with the correct plane types:
 *  - color formats → single `data` plane
 *  - DEPTH_COMPONENT_* → Float32Array `data`
 *  - DEPTH*_STENCIL* → Float32Array `data` (depth) + Uint8Array `stencilData`
 *  - STENCIL_INDEX8 → Uint8Array `data`
 * Row 0 = BOTTOM (GL window coordinates). Tightly packed (no row padding).
 */
export function createSurface(format: GLenum, width: number, height: number): Surface {
  throw new Error('not implemented: createSurface');
}

/** Resolves the depth plane of a surface (depth or depth-stencil formats).
 *  Depth always lives in `data` (Float32Array); `stencilData` holds the
 *  separate stencil plane for DEPTH*_STENCIL* formats. */
export function getDepthData(s: Surface): Float32Array {
  return s.data as Float32Array;
}

/** Resolves the stencil plane of a surface (stencil or depth-stencil formats). */
export function getStencilData(s: Surface): Uint8Array {
  return s.stencilData ?? (s.data as Uint8Array);
}

/** True when the surface stores depth (depth-only or depth-stencil). */
export function isDepthSurface(s: Surface): boolean {
  return s.info.isDepth;
}

/** True when the surface stores stencil (stencil-only or depth-stencil). */
export function isStencilSurface(s: Surface): boolean {
  return s.info.isStencil;
}

/** Byte size of a tightly packed surface for a known format (w*h*bpp). */
export function surfaceBytes(format: GLenum, width: number, height: number): number {
  const info = getFormat(format);
  if (!info) throw new Error(`unknown format 0x${format.toString(16)}`);
  return width * height * info.bytesPerPixel;
}
