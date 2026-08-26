/**
 * src/gl/objects/webgl-buffer.ts — WebGLBuffer.
 *
 * Buffer objects hold GPU-side (here: JS ArrayBuffer) data uploaded via
 * bufferData/bufferSubData. Per spec a buffer is bound to exactly one target
 * for its lifetime (first bind wins); that target is validated on subsequent
 * binds/uses. Deletion of a bound buffer is deferred until unbind.
 */

import { WebGLObject } from './webgl-object';
import type { GLenum, GLsizeiptr, GLuint } from '../types';

export class WebGLBuffer extends WebGLObject {
  /** Target the buffer was first bound to (0 = unbound yet). */
  _target: GLenum = 0;
  /** Uploaded data (null until bufferData). */
  _data: ArrayBuffer | null = null;
  /** Data size in bytes (bufferData size arg). */
  _size: GLsizeiptr = 0;
  /** STREAM/STATIC/DYNAMIC_DRAW usage hint. */
  _usage: GLenum = 0x88e4; // STATIC_DRAW
  /** True when deleteBuffer() was called while still bound — real delete deferred. */
  _deletePending = false;
  /** Indexed binding ranges (bindBufferRange) for TRANSFORM_FEEDBACK_BUFFER. */
  _tfRangeBindings: { index: GLuint; offset: number; size: number }[] = [];
}
