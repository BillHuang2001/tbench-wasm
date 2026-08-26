/**
 * src/gl/objects/webgl-texture.ts — WebGLTexture.
 *
 * Texture storage is allocated by gl/ via raster/formats and exposed to the
 * sampler as a `TextureImage` (contract §3). Cube maps hold 6 faces; 3D/2D_ARRAY
 * hold per-level boxes. Texture parameter defaults follow the GL spec
 * (MIN_FILTER=NEAREST_MIPMAP_LINEAR, MAG_FILTER=LINEAR, WRAP_S/T=REPEAT,
 * WRAP_R=REPEAT, BASE_LEVEL=0, MAX_LEVEL=1000, MIN/MAX_LOD=-1000/1000,
 * COMPARE_MODE=NONE, COMPARE_FUNC=LEQUAL).
 */

import { WebGLObject } from './webgl-object';
import type { GLenum, GLint, GLsizei } from '../types';

/** One mip level of one face/layer-slice of a texture. */
export interface TextureLevel {
  width: GLsizei;
  height: GLsizei;
  depth: GLsizei; // 1 for 2D/cube faces; layers for 2D_ARRAY; z-size for 3D
  /** Backing store (typed array) — format/encoding per raster/formats registry. */
  data: ArrayBufferView;
}

export class WebGLTexture extends WebGLObject {
  /** Target the texture was first bound to (0 = unbound yet). */
  _target: GLenum = 0;
  /** texStorage* set → immutable (further storage calls are INVALID_OPERATION). */
  _immutable = false;
  /** True when the texture is attached to a currently-bound FBO (feedback-loop checks). */
  _attachedToFramebuffer = false;
  /**
   * Mip storage: for TEXTURE_2D/3D/2D_ARRAY → levels[l]; for cube → faces[faceIndex][l]
   * where faceIndex = 0..5 (POSITIVE_X..NEGATIVE_Z). Contents are TextureImage
   * compatible (contract §3) once allocated.
   */
  _levels: (TextureLevel | null)[][] = [];
  /** glTexParameter state (per-texture; overridden by WebGLSampler when bound). */
  _params: Record<string, number> = {
    [0x2801]: 0x2700, // MIN_FILTER = NEAREST_MIPMAP_LINEAR
    [0x2800]: 0x2601, // MAG_FILTER = LINEAR
    [0x2802]: 0x2901, // WRAP_S = REPEAT
    [0x2803]: 0x2901, // WRAP_T = REPEAT
    [0x8072]: 0x2901, // WRAP_R = REPEAT (WebGL2)
    [0x813c]: 0, // BASE_LEVEL
    [0x813d]: 1000, // MAX_LEVEL
    [0x813a]: -1000, // MIN_LOD
    [0x813b]: 1000, // MAX_LOD
    [0x884c]: 0, // COMPARE_MODE = NONE
    [0x884d]: 0x0203, // COMPARE_FUNC = LEQUAL
    [0x84fe]: 1, // TEXTURE_MAX_ANISOTROPY_EXT
  };
  /** Base internal format (GLenum) of level 0 — for completeness rules. */
  _internalFormat: GLenum = 0;
  /** Internal format of the first level actually allocated (drives decode). */
  _allocatedFormat: GLenum = 0;
  /** True when base level is a compressed format. */
  _compressed = false;
}
