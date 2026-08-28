/**
 * src/gl/objects/webgl-transform-feedback.ts — WebGLTransformFeedback (WebGL2).
 *
 * Transform feedback captures the program's transformFeedbackVaryings outputs
 * into bound TRANSFORM_FEEDBACK_BUFFER ranges during draw calls. When active,
 * the rasterizer's fragment stage is skipped (rasterizerDiscard-like behavior);
 * the draw pipeline writes per-vertex varying outputs into the bound buffers
 * instead (or in addition when paused? — paused TF does NOT capture but still
 * renders). Primitive mode consistency (POINTS/LINES/TRIANGLES) is validated
 * at beginTransformFeedback.
 */

import { WebGLObject } from './webgl-object';
import type { GLenum } from '../types';
import type { WebGLBuffer } from './webgl-buffer';
import type { WebGLProgram } from './webgl-program';

export class WebGLTransformFeedback extends WebGLObject {
  /** Program captured at beginTransformFeedback. */
  _program: WebGLProgram | null = null;
  /** Indexed TRANSFORM_FEEDBACK_BUFFER bindings (0..MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS-1). */
  _buffers: (WebGLBuffer | null)[] = [];
  /**
   * Per-index binding ranges. `base: true` = whole-buffer bindBufferBase binding
   * (capture capacity tracks the CURRENT data store — bufferData growth between
   * bind and draw extends it); `base: false` = explicit bindBufferRange of FIXED
   * `size` (capacity = min(size, store - offset); a range whose size is smaller
   * than the capture makes the draw INVALID_OPERATION — CTS too-small-buffers).
   */
  _bufferRanges: { offset: number; size: number; base: boolean }[] = [];
  /** TRANSFORM_FEEDBACK_ACTIVE. */
  _active = false;
  /** TRANSFORM_FEEDBACK_PAUSED. */
  _paused = false;
  /** Primitive mode passed to beginTransformFeedback. */
  _primitiveMode: GLenum = 0;
  /** Primitives written counter (getParameter TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN). */
  _primitivesWritten = 0;
}
