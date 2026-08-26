/**
 * src/gl/objects/webgl-framebuffer.ts — WebGLFramebuffer + attachment model.
 *
 * Attachments reference textures (with level/face/layer) or renderbuffers.
 * The null framebuffer object is the default framebuffer (drawing buffer).
 */

import { WebGLObject } from './webgl-object';
import type { GLenum, GLint, GLsizei } from '../types';
import type { WebGLTexture } from './webgl-texture';
import type { WebGLRenderbuffer } from './webgl-renderbuffer';

export type FramebufferAttachment =
  | { type: 'texture'; texture: WebGLTexture; level: GLint; face: GLenum; layer: GLint }
  | { type: 'renderbuffer'; renderbuffer: WebGLRenderbuffer };

export class WebGLFramebuffer extends WebGLObject {
  /** attachment point (COLOR_ATTACHMENTn | DEPTH_ATTACHMENT | STENCIL_ATTACHMENT | DEPTH_STENCIL_ATTACHMENT) → attachment. */
  _attachments: Map<GLenum, FramebufferAttachment> = new Map();
  /** Cached completeness (recomputed on check/draw). */
  _status: GLenum = 0x8cd5; // FRAMEBUFFER_COMPLETE (default for empty? no — recomputed; kept for draw-path fast check)
  /** True when any attachment is multisampled (sample-count consistency rule). */
  _multisampled = false;
  /** True while bound to the draw framebuffer — attachment mutation guard. */
  _isBound = false;
}

export interface FramebufferSurface {
  width: GLsizei;
  height: GLsizei;
  format: GLenum;
  samples: number;
}
