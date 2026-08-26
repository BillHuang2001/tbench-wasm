/**
 * src/gl/api/buffers.ts — buffer objects and data.
 *
 * Owns: createBuffer, deleteBuffer, isBuffer, bindBuffer, bufferData,
 * bufferSubData, getBufferParameter (+ WebGL2: bindBufferBase, bindBufferRange,
 * getIndexedParameter for UNIFORM_BUFFER/TRANSFORM_FEEDBACK_BUFFER, getBufferSubData).
 *
 * Behavior notes:
 *  - bindBuffer: target ∈ {ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER} (WebGL1) +
 *    {COPY_READ/WRITE_BUFFER, PIXEL_PACK/UNPACK_BUFFER, TRANSFORM_FEEDBACK_BUFFER,
 *    UNIFORM_BUFFER} (WebGL2). First bind fixes the buffer's target; rebinding to
 *    another target → INVALID_OPERATION. null unbinds (ELEMENT_ARRAY_BUFFER
 *    binding lives in the VAO state).
 *  - bufferData: size number → allocate; ArrayBuffer/ArrayBufferView → copy;
 *    usage ∈ {STREAM,STATIC,DYNAMIC}_DRAW. WebGL2: PIXEL_UNPACK/PACK buffer
 *    targets cannot be bufferData'd → INVALID_OPERATION.
 *  - bufferSubData: offset+size bounds vs buffer size (INVALID_VALUE); same
 *    target restrictions; PIXEL_PACK_BUFFER → INVALID_OPERATION.
 *  - deleteBuffer: deferred when bound (deletePending until unbound); deleting
 *    a buffer bound to a VAO/UBO/TF binding point unbinds it there (per spec).
 *  - bindBufferBase/Range: UNIFORM_BUFFER index < MAX_UNIFORM_BUFFER_BINDINGS,
 *    TRANSFORM_FEEDBACK_BUFFER index < MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS,
 *    offset aligned to UNIFORM_BUFFER_OFFSET_ALIGNMENT (UBO) and
 *    offset+size ≤ buffer size; binding an active-TF buffer → INVALID_OPERATION.
 *  - getIndexedParameter: UNIFORM_BUFFER_BINDING/START/SIZE, TRANSFORM_FEEDBACK_BUFFER_*.
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installBuffersApi(proto: WebGLRenderingContext): void {
  void proto;
}
