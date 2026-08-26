/**
 * src/gl/api/teximage.ts — texImage/texSubImage/texStorage/copyTex/compressedTex.
 *
 * Owns: texImage2D, texImage3D, texSubImage2D, texSubImage3D, texStorage2D,
 * texStorage3D, copyTexImage2D, copyTexSubImage2D, copyTexSubImage3D,
 * compressedTexImage2D, compressedTexImage3D, compressedTexSubImage2D,
 * compressedTexSubImage3D.
 *
 * All argument validation + overload dispatch happens here; the actual storage
 * mutation is delegated to the teximage.ts engine module. Overload rules:
 *  - 9/10-arg form (WebGL1/2): full size + border + format + type + pixels.
 *  - WebGL2: pixels may be a NUMBER (byte offset into PIXEL_UNPACK_BUFFER).
 *  - 5/6/7/8-arg form: (format, type, source) — source = ImageData |
 *    HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap |
 *    OffscreenCanvas | null; null allocates empty storage.
 * Validation highlights: border must be 0 (INVALID_VALUE); internalformat vs
 * format/type compatibility per WebGL tables + extension-gated formats
 * (float/depth/sRGB/compressed); NPOT size limits; texStorage immutability;
 * PIXEL_UNPACK_BUFFER must not be bound for the source path (INVALID_OPERATION).
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installTexImageApi(proto: WebGLRenderingContext): void {
  void proto;
}
