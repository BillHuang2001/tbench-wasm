/**
 * src/gl/extensions/vao.ts — OES_vertex_array_object (WebGL1).
 *
 * Vertex array objects hold per-attribute array state + the ELEMENT_ARRAY_BUFFER
 * binding (VAOState, state.ts). Semantics mirror the WebGL2 bindVertexArray
 * engine (api/webgl2.ts — parallel agent; same rules implemented inline here):
 *  - bindVertexArrayOES(null) binds the DEFAULT VAO (a bare VAOState, never a
 *    WebGLVertexArrayObject instance — isVertexArrayOES(null) is false).
 *  - The default VAO's contents are the initial state.vao (migrated into
 *    ctx._defaultVAO at the first user-VAO bind, so pre-VAO attrib setup is
 *    preserved when the default is re-bound).
 *  - Binding a deleted VAO → INVALID_OPERATION; deleting the bound VAO
 *    re-binds the default; deleting an unbound VAO is a silent no-op.
 *  - Cross-context / fake objects follow the standard WebIDL rules (TypeError
 *    for non-WebGLVertexArrayObject; INVALID_OPERATION + no-op for
 *    cross-context or deleted).
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1 } from '../constants';
import { WebGLVertexArrayObject, createObject } from '../objects';
import { defaultVAOState, type VAOState } from '../state';
import { buildExtension, ctorOf, isLost } from './util';

const VAOCtor = ctorOf(WebGLVertexArrayObject);

/** The default VAO contents for a context (lazily created / migrated). */
function defaultVAO(ctx: WebGLRenderingContext): VAOState {
  if (!ctx._defaultVAO) {
    if (ctx._state.vaoBinding === null) {
      // Nothing bound yet: the current state.vao IS the default VAO's contents
      // (includes any attrib setup done before the first VAO bind).
      ctx._defaultVAO = ctx._state.vao;
    } else {
      ctx._defaultVAO = defaultVAOState(ctx._state.limits.MAX_VERTEX_ATTRIBS);
    }
  }
  return ctx._defaultVAO;
}

/** Bind the default VAO (state.vaoBinding = null per spec). */
function bindDefault(ctx: WebGLRenderingContext): void {
  ctx._state.vaoBinding = null;
  ctx._state.vao = defaultVAO(ctx);
}

/** OES_vertex_array_object factory. */
export function createOESVertexArrayObject(ctx: WebGLRenderingContext): object {
  return buildExtension(
    { VERTEX_ARRAY_BINDING_OES: 0x85b5 },
    {
      createVertexArrayOES: (): WebGLVertexArrayObject | null => {
        const gl = ctx;
        if (isLost(gl)) return null;
        const vao = createObject(gl, VAOCtor);
        vao._vao = defaultVAOState(gl._state.limits.MAX_VERTEX_ATTRIBS);
        return vao;
      },

      deleteVertexArrayOES: (vertexArray: WebGLVertexArrayObject | null): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        if (vertexArray === null || vertexArray === undefined) return;
        if (!(vertexArray instanceof WebGLVertexArrayObject)) {
          throw new TypeError(`Argument is not of type 'WebGLVertexArrayObject'`);
        }
        if (vertexArray._context !== gl) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        if (vertexArray._deleted) return; // already deleted: silent no-op
        if (gl._state.vaoBinding === vertexArray) bindDefault(gl); // deleting the bound VAO re-binds default
        vertexArray._deleted = true;
        gl._resources.untrack(vertexArray);
      },

      isVertexArrayOES: (vertexArray: WebGLVertexArrayObject | null): boolean => {
        const gl = ctx;
        if (isLost(gl)) return false;
        if (vertexArray === null || vertexArray === undefined) return false;
        if (!(vertexArray instanceof WebGLVertexArrayObject)) {
          throw new TypeError(`Argument is not of type 'WebGLVertexArrayObject'`);
        }
        if (vertexArray._context !== gl) {
          gl._errors.push(C1.INVALID_OPERATION);
          return false;
        }
        return !vertexArray._deleted;
      },

      bindVertexArrayOES: (vertexArray: WebGLVertexArrayObject | null): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        if (vertexArray === null || vertexArray === undefined) {
          bindDefault(gl);
          return;
        }
        if (!(vertexArray instanceof WebGLVertexArrayObject)) {
          throw new TypeError(`Argument is not of type 'WebGLVertexArrayObject'`);
        }
        if (vertexArray._context !== gl) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        if (vertexArray._deleted) {
          gl._errors.push(C1.INVALID_OPERATION);
          return;
        }
        // Migrate the current (default) VAO contents before overwriting state.vao.
        if (!gl._defaultVAO && gl._state.vaoBinding === null) {
          gl._defaultVAO = gl._state.vao;
        }
        gl._state.vaoBinding = vertexArray;
        gl._state.vao = vertexArray._vao;
      },
    },
  );
}
