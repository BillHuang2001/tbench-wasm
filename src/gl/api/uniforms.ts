/**
 * src/gl/api/uniforms.ts — uniform setters (write into the per-program store).
 *
 * Owns: uniform{1,2,3,4}{f,i,ui}{,v} and uniformMatrix{2,3,4}{,x2,x3,x4}fv
 * (WebGL1: f/i + 2x2/3x3/4x4; WebGL2 adds ui + non-square matrices).
 *
 * Store contract (with glsl/): each Program.uniforms[i] has a `location`
 * (index into the store) and type; gl/ writes via DataView at offsets computed
 * from the glsl/ layout (aligned per GL std140 for blocks, packed for the
 * default block). uniform* validation:
 *  - location null → no-op (legal). Location from another program →
 *    INVALID_OPERATION (validateUniformLocation in objects/aux.ts).
 *  - TYPE MISMATCH: uniform1f on an int uniform (or uniform1i on a float
 *    uniform, etc.) → INVALID_OPERATION and the store is NOT written.
 *  - Matrix transpose must be false (INVALID_VALUE otherwise — WebGL spec).
 *  - Array uniforms: value length must be ≥ size*components (INVALID_VALUE);
 *    uniform*{v} on arrays writes the whole array starting at the location.
 *  - Sampler uniforms: uniform1i/1iv set the sampler's texture unit; the value
 *    must be < MAX_COMBINED_TEXTURE_IMAGE_UNITS (INVALID_VALUE otherwise).
 *    Sampler arrays set consecutive units.
 *  - getUniform (api/programs.ts) reads back the same store (bool uniforms
 *    stored as ints, returned as booleans).
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installUniformsApi(proto: WebGLRenderingContext): void {
  void proto;
}
