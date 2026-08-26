/**
 * src/gl/api/textures.ts — texture objects, parameters, pixelStore, mipmaps.
 *
 * Owns: createTexture, deleteTexture, isTexture, bindTexture, activeTexture,
 * texParameterf, texParameteri, getTexParameter, pixelStorei, generateMipmap.
 * (Storage uploads live in api/teximage.ts → engine teximage.ts.)
 *
 * Behavior notes:
 *  - bindTexture: target ∈ {TEXTURE_2D, TEXTURE_CUBE_MAP} (WebGL1) +
 *    {TEXTURE_3D, TEXTURE_2D_ARRAY} (WebGL2). First bind fixes the texture's
 *    target (INVALID_OPERATION on mismatch). Binding updates the active unit's
 *    slot (state.textureUnits[activeTexture]).
 *  - texParameter: valid (target, pname, value) table per version; WebGL1 NPOT
 *    enforcement happens at DRAW time (sampler), not at texParameter time.
 *    TEXTURE_MAX_ANISOTROPY_EXT requires the extension. WebGL2: BASE_LEVEL/
 *    MAX_LEVEL/MIN_LOD/MAX_LOD/COMPARE_MODE/COMPARE_FUNC/WRAP_R.
 *  - generateMipmap: complete base level required (INVALID_OPERATION if not
 *    texture-complete or format not mipmap-generatable — float formats need
 *    OES_texture_float_linear); builds levels via raster formats.
 *  - deleteTexture: unbinds from all units + FBO attachments (per spec).
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installTexturesApi(proto: WebGLRenderingContext): void {
  void proto;
}
