/**
 * src/gl/getters.ts — getParameter implementation (api/context.ts delegates here).
 *
 * Maps every spec pname to state/limits/drawing-buffer data. Behavior rules:
 *  - Return types per spec: booleans for caps/masks, Int32Array for vectors
 *    (COLOR_CLEAR_VALUE, DEPTH_RANGE, SCISSOR_BOX, VIEWPORT, ALIASED_*_RANGE,
 *    MAX_VIEWPORT_DIMS, COLOR_WRITEMASK as boolean[]), plain numbers otherwise.
 *  - WebGL1-only pnames (MAX_VERTEX_UNIFORM_VECTORS etc.) and WebGL2-only
 *    pnames (MAX_VERTEX_UNIFORM_COMPONENTS etc.) are rejected with
 *    INVALID_ENUM on the wrong context version.
 *  - FBO-dependent pnames (SAMPLE_BUFFERS, SAMPLES, RED_BITS, ...) read the
 *    bound draw framebuffer's attachments (null → default framebuffer).
 *  - Extension pnames (MAX_TEXTURE_MAX_ANISOTROPY_EXT, UNMASKED_*) work only
 *    when the extension is enabled.
 *  - getParameter(EXTENSIONS) is illegal in WebGL (use getSupportedExtensions).
 *  - WebGL2: getIndexedParameter handles the indexed pnames (UNIFORM_BUFFER_*,
 *    TRANSFORM_FEEDBACK_BUFFER_*).
 */

import type { WebGLRenderingContext } from './webgl1';
import type { GLenum } from './types';

/** Full getParameter dispatch. */
export function getParameter(ctx: WebGLRenderingContext, pname: GLenum): unknown {
  void ctx;
  void pname;
  throw new Error('GL stub: getParameter (Phase 2 — table in src/gl/getters.ts)');
}
