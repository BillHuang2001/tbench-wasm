/**
 * src/gl/objects/webgl-texture.ts — WebGLTexture.
 *
 * Texture storage is a raster-compatible `TextureImage` (contract §3), owned
 * HERE as `_image`: teximage.ts (engine) allocates/updates it on every storage
 * mutation; the sampler, FBO attachments, and draw-pipeline consume it. Cube
 * maps hold 6 face views per level (order +X, -X, +Y, -Y, +Z, -Z — see
 * raster CUBE_FACE_TO_INDEX); 3D/2D_ARRAY hold per-level boxes.
 *
 * `_image.levels[level].data` is `ArrayBufferView[]`: exactly one view for
 * 2D/3D/2D_ARRAY levels, six views for cube levels. Depth-stencil textures
 * keep the split stencil plane in `levels[level].stencilData` (mirroring
 * raster's Surface representation so FBO attachments can expose it).
 *
 * Texture parameter defaults follow the GL spec (MIN_FILTER=NEAREST_MIPMAP_LINEAR,
 * MAG_FILTER=LINEAR, WRAP_S/T/R=REPEAT, BASE_LEVEL=0, MAX_LEVEL=1000,
 * MIN/MAX_LOD=-1000/1000, COMPARE_MODE=NONE, COMPARE_FUNC=LEQUAL).
 */

import { WebGLObject } from './webgl-object';
import type { GLenum } from '../types';
import type { TextureImage } from '../../raster';

/** One mip level of one texture (raster-compatible; `data` = per-face views). */
export interface TextureLevel {
  width: number;
  height: number;
  depth: number; // 1 for 2D/cube; layers for 2D_ARRAY; z-size for 3D
  /** One view for 2D/3D/2D_ARRAY; six views for cube maps (face order +X..-Z). */
  data: ArrayBufferView[];
  /** Split stencil plane for DEPTH*_STENCIL* textures (else undefined). */
  stencilData?: Uint8Array;
}

/** TextureImage with gl's split-stencil-plane addition (assignable to raster's). */
export type GLTextureImage = Omit<TextureImage, 'levels'> & { levels: TextureLevel[] };

export class WebGLTexture extends WebGLObject {
  /** Target the texture was first bound to (0 = unbound yet). */
  _target: GLenum = 0;
  /** texStorage* set → immutable (further storage calls are INVALID_OPERATION). */
  _immutable = false;
  /** True when the texture is attached to a currently-bound FBO (feedback-loop checks). */
  _attachedToFramebuffer = false;
  /** Sample count set by WEBGL_multisampled_render_to_texture (0 = single-sampled). */
  _msaaSamples = 0;
  /**
   * The texture's storage + sampling metadata, maintained by teximage.ts:
   * levels (raster TextureLevel[]), internalFormat, dimensions, base/max level,
   * immutable flag, and the computed completeness (sampling + FBO rules).
   * null until the first storage allocation.
   */
  _image: GLTextureImage | null = null;
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
  /** True when the base level is a compressed format. */
  _compressed = false;
}
