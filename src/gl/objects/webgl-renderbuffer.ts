/**
 * src/gl/objects/webgl-renderbuffer.ts — WebGLRenderbuffer.
 *
 * Storage is allocated by the framebuffers module via raster's
 * `surface.createSurface` (contract §3) at renderbufferStorage time and kept
 * in `_surface` — the raster Surface consumed by FBO attachments, the draw
 * pipeline, readPixels, and getRenderbufferParameter.
 */

import { WebGLObject } from './webgl-object';
import type { GLsizei } from '../types';
import type { Surface, GLenum } from '../../raster';

export class WebGLRenderbuffer extends WebGLObject {
  /** INTERNALFORMAT (0 = unallocated). */
  _internalformat: GLenum = 0;
  _width: GLsizei = 0;
  _height: GLsizei = 0;
  /** Sample count (renderbufferStorageMultisample). Storage is single-sampled; blit resolves. */
  _samples = 0;
  /** Backing raster surface (typed-array store per raster/formats registry). */
  _surface: Surface | null = null;
}
