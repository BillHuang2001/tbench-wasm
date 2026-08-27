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
import type { StorageKind } from './formats';
import type { GLenum } from './gl-enums';

/** Bytes per element for each storage class. */
function storageBPE(storage: StorageKind): number {
  switch (storage) {
    case 'u8': case 'i8': return 1;
    case 'u16': case 'i16': case 'f16': return 2;
    default: return 4; // u32, i32, f32
  }
}

/**
 * Allocates a surface for `format` with the correct plane types:
 *  - color formats → single `data` plane, one typed array per storage class
 *    (u8→Uint8Array, i8→Int8Array, u16→Uint16Array, i16→Int16Array,
 *    u32→Uint32Array, i32→Int32Array, f32→Float32Array), tightly packed
 *    (width*height*bpp / bytesPerElement elements)
 *  - DEPTH_COMPONENT_* → Float32Array `data` (w*h)
 *  - DEPTH*_STENCIL* → Float32Array `data` (depth, w*h) + Uint8Array
 *    `stencilData` (stencil, w*h)
 *  - STENCIL_INDEX8 → Uint8Array `data` (w*h)
 * Row 0 = BOTTOM (GL window coordinates). Tightly packed (no row padding).
 * Throws for unknown formats.
 */
export function createSurface(format: GLenum, width: number, height: number): Surface {
  const info = getFormat(format);
  if (!info) throw new Error(`createSurface: unknown format 0x${format.toString(16)}`);
  const n = width * height;
  let data: ArrayBufferView;
  if (info.isDepth || info.isStencil) {
    // Depth plane is always f32; stencil-only is u8.
    data = info.isDepth ? new Float32Array(n) : new Uint8Array(n);
  } else {
    const elems = (n * info.bytesPerPixel) / storageBPE(info.storage);
    switch (info.storage) {
      case 'u8': data = new Uint8Array(elems); break;
      case 'i8': data = new Int8Array(elems); break;
      case 'u16': data = new Uint16Array(elems); break;
      case 'i16': data = new Int16Array(elems); break;
      case 'u32': data = new Uint32Array(elems); break;
      case 'i32': data = new Int32Array(elems); break;
      case 'f32': data = new Float32Array(elems); break;
      case 'f16': data = new Uint16Array(elems); break;
    }
  }
  const surface: Surface = { width, height, format, info, data };
  if (info.isDepth && info.isStencil) surface.stencilData = new Uint8Array(n);
  return surface;
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
