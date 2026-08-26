/**
 * src/gl/objects/webgl-renderbuffer.ts — WebGLRenderbuffer.
 */

import { WebGLObject } from './webgl-object';
import type { GLenum, GLsizei } from '../types';

export class WebGLRenderbuffer extends WebGLObject {
  /** INTERNALFORMAT (0 = unallocated). */
  _internalformat: GLenum = 0;
  _width: GLsizei = 0;
  _height: GLsizei = 0;
  /** Sample count (renderbufferStorageMultisample). */
  _samples = 0;
  /** Backing surface (typed-array store per raster/formats registry). */
  _data: ArrayBufferView | null = null;
  _depthData: ArrayBufferView | null = null;
  _stencilData: ArrayBufferView | null = null;
}
