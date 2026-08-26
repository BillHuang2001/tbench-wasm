/**
 * src/gl/api/draw.ts — draw calls, clear, readPixels, flush/finish.
 *
 * Owns: drawArrays, drawElements, drawArraysInstanced (W2), drawElementsInstanced
 * (W2), drawRangeElements (W2), multiDrawArraysWEBGL/multiDrawElementsWEBGL/
 * multiDrawArraysInstancedWEBGL/multiDrawElementsInstancedWEBGL (WEBGL_multi_draw),
 * clear, clearBuffer{fv,iv,uiv,fi} (W2), flush, finish, readPixels.
 *
 * Validation before delegating to the draw.ts engine:
 *  - Program linked + in use (INVALID_OPERATION); FBO complete
 *    (INVALID_FRAMEBUFFER_OPERATION); attribs valid; count/first bounds
 *    (INVALID_OPERATION for negative — INVALID_VALUE vs OPERATION per spec);
 *    indexed: ELEMENT_ARRAY_BUFFER bound + type valid (UNSIGNED_BYTE W2 only
 *    via VAO; UNSIGNED_SHORT; UNSIGNED_INT with OES_element_index_uint/W2) +
 *    offset valid; instanced: instanceCount ≥ 0.
 *  - Transform feedback active: mode must match beginTransformFeedback mode
 *    (INVALID_OPERATION); TF with default framebuffer bound or with
 *    non-TF-varying program → INVALID_OPERATION; queries on
 *    ANY_SAMPLES_PASSED while TF active → INVALID_OPERATION.
 *  - Feedback loops (texture attached to FBO + bound to a sampler unit used by
 *    the program) → INVALID_OPERATION.
 *  - clear: mask bits ∈ {COLOR, DEPTH, STENCIL}_BUFFER_BIT; scissor-respecting
 *    clear per state.clearColor/Depth/Stencil + colorMask/depthMask/stencilMask.
 *  - readPixels: format/type validation (RGBA/UNSIGNED_BYTE is always legal —
 *    IMPLEMENTATION_COLOR_READ_FORMAT/TYPE; others per attachment format),
 *    pixel pack alignment/skip, buffer size check (INVALID_OPERATION when the
 *    ArrayBufferView is too small), PIXEL_PACK_BUFFER offset path (W2), and
 *    integer-format readPixels (W2) returning raw integers.
 *  - flush/finish: no-ops (synchronous renderer) but must still validate context.
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installDrawApi(proto: WebGLRenderingContext): void {
  void proto;
}
