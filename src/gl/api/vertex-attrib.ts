/**
 * src/gl/api/vertex-attrib.ts — vertex attrib array state.
 *
 * Owns: vertexAttribPointer, vertexAttribIPointer (WebGL2),
 * enableVertexAttribArray, disableVertexAttribArray,
 * vertexAttrib{1,2,3,4}{f,fv}, vertexAttribI4{i,iv,ui,uiv} (WebGL2),
 * vertexAttribDivisor (WebGL2/ANGLE_instanced_arrays), getVertexAttrib,
 * getVertexAttribOffset.
 *
 * Behavior notes:
 *  - vertexAttribPointer: index < MAX_VERTEX_ATTRIBS; size 1..4; type ∈
 *    {BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, FLOAT} (+WebGL2
 *    {INT, UNSIGNED_INT, HALF_FLOAT, INT_2_10_10_10_REV, UNSIGNED_INT_2_10_10_10_REV});
 *    stride ≤ MAX_VERTEX_ATTRIB_STRIDE; offset < 2^31; a buffer MUST be bound to
 *    ARRAY_BUFFER (else INVALID_OPERATION — the buffer binding is captured);
 *    normalized only valid for non-integer types.
 *  - vertexAttribIPointer: type ∈ {BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT,
 *    INT, UNSIGNED_INT}; sets attrib.integer = true (getVertexAttrib
 *    VERTEX_ATTRIB_ARRAY_INTEGER).
 *  - vertexAttrib*f: sets the constant value and (per spec) disables the array
 *    for that attrib in the current VAO when the array was enabled? NO — spec:
 *    setting the generic value does NOT touch enable state; the array is used
 *    when enabled, constant otherwise.
 *  - vertexAttribDivisor: divisor value is stored; draw instancing per divisor.
 *  - getVertexAttrib: CURRENT_VERTEX_ATTRIB returns the constant values
 *    (Float32Array(4)); VERTEX_ATTRIB_ARRAY_BUFFER_BINDING returns the bound
 *    buffer; VERTEX_ATTRIB_ARRAY_POINTER (WebGL1) returns the offset as a
 *    number; WebGL2 adds VERTEX_ATTRIB_ARRAY_INTEGER/DIVISOR.
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installVertexAttribApi(proto: WebGLRenderingContext): void {
  void proto;
}
