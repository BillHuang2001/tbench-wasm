/**
 * src/gl/framebuffer-util.ts — shared framebuffer resolution (contract pin).
 *
 * THE single place that resolves WebGLFramebuffer attachments (and the default
 * framebuffer) into raster `FramebufferTarget`/`Surface` objects. Owned and
 * implemented by the framebuffers API module (api/framebuffers.ts); consumed by
 * the draw engine (draw.ts), getters (getters.ts), and teximage's copyTex*.
 *
 * Surface resolution rules:
 *  - Renderbuffer attachment → `rb._surface` (allocated at renderbufferStorage).
 *  - Texture attachment → the level's view: 2D → `_image.levels[level].data[0]`;
 *    cube → `data[CUBE_FACE_TO_INDEX[face]]`; depth-stencil textures expose the
 *    split stencil plane via `levels[level].stencilData`.
 *  - The default framebuffer (drawFramebuffer === null) → `ctx._defaultFB`.
 *
 * Completeness (`checkFramebufferStatus`) implements every rule of the WebGL
 * spec (attachment presence, format/attachment-point compatibility, dimension
 * consistency, cube completeness, sample-count consistency, draw-buffer
 * coverage) and is consulted both by the API and by the draw pipeline
 * (INVALID_FRAMEBUFFER_OPERATION).
 */

import type { WebGLRenderingContext } from './webgl1';
import type { WebGLFramebuffer } from './objects';
import type { FramebufferTarget, Surface } from '../raster';
import type { GLenum } from './types';

/**
 * Resolve the current DRAW target: the bound draw framebuffer's attachments,
 * or the default framebuffer when none is bound. Returns null when the bound
 * FBO is incomplete (caller pushes INVALID_FRAMEBUFFER_OPERATION).
 */
export function resolveFramebufferTarget(ctx: WebGLRenderingContext): FramebufferTarget | null {
  void ctx;
  throw new Error('GL stub: framebuffer-util resolveFramebufferTarget (framebuffers agent)');
}

/** Resolve the READ surface for readPixels (read framebuffer + readBuffer). */
export function resolveReadSurface(ctx: WebGLRenderingContext): Surface | null {
  void ctx;
  throw new Error('GL stub: framebuffer-util resolveReadSurface (framebuffers agent)');
}

/** Full WebGL framebuffer completeness check (spec rules; see header). */
export function checkFramebufferStatus(ctx: WebGLRenderingContext, fbo: WebGLFramebuffer): GLenum {
  void ctx;
  void fbo;
  throw new Error('GL stub: framebuffer-util checkFramebufferStatus (framebuffers agent)');
}

/** Resolve one attachment point of a framebuffer to its surface (null when unset/invalid). */
export function getAttachmentSurface(fbo: WebGLFramebuffer, attachment: GLenum): Surface | null {
  void fbo;
  void attachment;
  throw new Error('GL stub: framebuffer-util getAttachmentSurface (framebuffers agent)');
}
