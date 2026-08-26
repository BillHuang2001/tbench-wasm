/**
 * src/gl/objects/webgl-vao.ts — WebGLVertexArrayObject.
 *
 * VAO contents (attrib array state + ELEMENT_ARRAY_BUFFER binding) live in a
 * VAOState (state.ts). The context's default VAO is a bare VAOState with
 * `vaoBinding === null`; user VAOs carry their own VAOState.
 */

import { WebGLObject } from './webgl-object';
import type { VAOState } from '../state';

export class WebGLVertexArrayObject extends WebGLObject {
  /** Assigned at creation (createObject / constructor in Phase 2). */
  _vao!: VAOState;
}
