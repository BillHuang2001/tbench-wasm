/**
 * src/gl/objects/webgl-sampler.ts — WebGLSampler (WebGL2).
 *
 * Sampler objects override the texture's filter/wrap/LOD/compare parameters at
 * sampling time. Defaults mirror the texture parameter defaults (WebGL enum
 * convention: NEAREST_MIPMAP_LINEAR = 0x2702 — see constants.ts).
 */

import { WebGLObject } from './webgl-object';

export class WebGLSampler extends WebGLObject {
  _params: Record<string, number> = {
    [0x2801]: 0x2702, // MIN_FILTER = NEAREST_MIPMAP_LINEAR
    [0x2800]: 0x2601, // MAG_FILTER = LINEAR
    [0x2802]: 0x2901, // WRAP_S = REPEAT
    [0x2803]: 0x2901, // WRAP_T = REPEAT
    [0x8072]: 0x2901, // WRAP_R = REPEAT
    [0x813c]: 0, // BASE_LEVEL
    [0x813d]: 1000, // MAX_LEVEL
    [0x813a]: -1000, // MIN_LOD
    [0x813b]: 1000, // MAX_LOD
    [0x884c]: 0, // COMPARE_MODE = NONE
    [0x884d]: 0x0203, // COMPARE_FUNC = LEQUAL
    [0x84fe]: 1, // TEXTURE_MAX_ANISOTROPY_EXT
  };
}
