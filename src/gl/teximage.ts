/**
 * src/gl/teximage.ts — texture upload engine (internal; api/teximage.ts delegates).
 *
 * Handles ALL texImage* / texSubImage* / texStorage* / copyTex* / compressedTex* / generateMipmap
 * storage mutations of WebGLTexture:
 *  - Overload dispatch: (w,h,border,format,type,pixels) vs (format,type,source)
 *    decided by argument count + type (ArrayBufferView → data path; DOM source →
 *    present/ decode path; null/absent → allocate).
 *  - Source decode via present/ ImageSource (contract §4): ImageData/
 *    HTMLImageElement/HTMLCanvasElement/HTMLVideoElement/ImageBitmap/OffscreenCanvas
 *    → RGBA8 Uint8Array (or raw when pixelStore says so), applying
 *    UNPACK_FLIP_Y_WEBGL, UNPACK_PREMULTIPLY_ALPHA_WEBGL,
 *    UNPACK_COLORSPACE_CONVERSION_WEBGL.
 *  - Typed-array upload: UNPACK_ALIGNMENT row padding, WebGL2 UNPACK_ROW_LENGTH/
 *    SKIP_ROWS/SKIP_PIXELS/IMAGE_HEIGHT/SKIP_IMAGES, source↔internal format
 *    conversion tables via raster/formats (contract §3), PIXEL_UNPACK_BUFFER
 *    offset path (WebGL2: pixels is a byte offset into the bound buffer).
 *  - Storage rules: NPOT restrictions (WebGL1: only with CLAMP_TO_EDGE +
 *    NEAREST/LINEAR, no mips), texStorage immutability, level limits
 *    (MAX_TEXTURE_SIZE etc.), cube face validation, border must be 0,
 *    format/type compatibility (incl. extension-gated float/depth formats),
 *    generateMipmap base-format support + completeness of level chain.
 *  - Allocations go through raster/formats so FBO attachment + sampling agree.
 */

import type { WebGLRenderingContext } from './webgl1';
import type { WebGLTexture } from './objects';
import type { GLenum, GLint, GLsizei } from './types';

export type TexImageSourceArg = ArrayBufferView | number | null;

/** texImage2D/texImage3D storage mutation (target validated by caller). */
export function uploadTexImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  border: GLint,
  format: GLenum,
  type: GLenum,
  pixels: TexImageSourceArg,
  source?: unknown,
): void {
  void ctx; void texture; void target; void level; void internalformat;
  void width; void height; void depth; void border; void format; void type;
  void pixels; void source;
  throw new Error('GL stub: texImage (Phase 2 — see src/gl/teximage.ts)');
}

/** texSubImage2D/3D partial update. */
export function uploadTexSubImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  xoffset: GLint, yoffset: GLint, zoffset: GLint,
  width: GLsizei, height: GLsizei, depth: GLsizei,
  format: GLenum, type: GLenum,
  pixels: TexImageSourceArg,
  source?: unknown,
): void {
  void ctx; void texture; void target; void level;
  void xoffset; void yoffset; void zoffset; void width; void height; void depth;
  void format; void type; void pixels; void source;
  throw new Error('GL stub: texSubImage (Phase 2)');
}

/** texStorage2D/3D: allocate immutable mip chain. */
export function allocateImmutableStorage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  levels: GLsizei,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
): void {
  void ctx; void texture; void target; void levels; void internalformat;
  void width; void height; void depth;
  throw new Error('GL stub: texStorage (Phase 2)');
}

/** copyTexImage2D / copyTexSubImage2D / copyTexSubImage3D: read from read framebuffer. */
export function copyTexImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  x: GLint, y: GLint, width: GLsizei, height: GLsizei, border: GLint,
): void {
  void ctx; void texture; void target; void level; void internalformat;
  void x; void y; void width; void height; void border;
  throw new Error('GL stub: copyTexImage (Phase 2)');
}

export function copyTexSubImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  xoffset: GLint, yoffset: GLint, zoffset: GLint,
  x: GLint, y: GLint, width: GLsizei, height: GLsizei,
): void {
  void ctx; void texture; void target; void level;
  void xoffset; void yoffset; void zoffset; void x; void y; void width; void height;
  throw new Error('GL stub: copyTexSubImage (Phase 2)');
}

/** compressedTexImage2D/3D + compressedTexSubImage2D/3D (decompression or raw block store). */
export function compressedTexImage(
  ctx: WebGLRenderingContext,
  texture: WebGLTexture,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  width: GLsizei, height: GLsizei, depth: GLsizei,
  border: GLint,
  data: ArrayBufferView,
  sub: boolean,
  xoffset: GLint, yoffset: GLint, zoffset: GLint,
): void {
  void ctx; void texture; void target; void level; void internalformat;
  void width; void height; void depth; void border; void data; void sub;
  void xoffset; void yoffset; void zoffset;
  throw new Error('GL stub: compressedTexImage (Phase 2)');
}

/** generateMipmap: build the full mip chain from the base level (format-aware). */
export function generateMipmap(ctx: WebGLRenderingContext, texture: WebGLTexture, target: GLenum): void {
  void ctx; void texture; void target;
  throw new Error('GL stub: generateMipmap (Phase 2)');
}
