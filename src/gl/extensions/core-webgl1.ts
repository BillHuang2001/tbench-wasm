/**
 * src/gl/extensions/core-webgl1.ts — factories for the WebGL1 capability /
 * constant-only extensions.
 *
 * These extensions have no methods: they exist so getExtension() returns a real
 * object and the feature gates (texture formats, framebuffer formats, blend
 * equations, shader extensions, hint enums) open up in the parallel agents'
 * validation code (each gates on `ctx._extensions.has(name)` — the cache is
 * populated by getExtensionObject before the object is returned).
 *
 * Constants per CExt (constants.ts — values verified against the extension
 * specs) and per the pinned interfaces in types.ts.
 */

import { CExt } from '../constants';
import { buildExtension } from './util';
import type { WebGLRenderingContext } from '../webgl1';

/** OES_texture_float — float texture formats (no members). */
export function createOESTextureFloat(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** OES_texture_half_float — HALF_FLOAT_OES texture format. */
export function createOESTextureHalfFloat(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({ HALF_FLOAT_OES: CExt.HALF_FLOAT_OES });
}

/** OES_texture_float_linear — linear filtering of float textures (no members). */
export function createOESTextureFloatLinear(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** OES_texture_half_float_linear — linear filtering of half-float textures. */
export function createOESTextureHalfFloatLinear(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** OES_element_index_uint — UNSIGNED_INT index buffers (no members). */
export function createOESElementIndexUint(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** OES_standard_derivatives — FRAGMENT_SHADER_DERIVATIVE_HINT_OES hint enum. */
export function createOESStandardDerivatives(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({ FRAGMENT_SHADER_DERIVATIVE_HINT_OES: CExt.FRAGMENT_SHADER_DERIVATIVE_HINT_OES });
}

/** OES_fbo_render_mipmap — render to texture mip levels (no members). */
export function createOESFboRenderMipmap(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** EXT_frag_depth — gl_FragDepthEXT in fragment shaders (no members). */
export function createEXTFragDepth(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** EXT_shader_texture_lod — textureLod in fragment shaders (no members). */
export function createEXTShaderTextureLod(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** EXT_sRGB — sRGB texture/renderbuffer formats + color-encoding query. */
export function createEXTSRGB(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    SRGB_EXT: CExt.SRGB_EXT,
    SRGB_ALPHA_EXT: CExt.SRGB_ALPHA_EXT,
    SRGB8_ALPHA8_EXT: CExt.SRGB8_ALPHA8_EXT,
    FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT: CExt.FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT,
  });
}

/** EXT_blend_minmax — MIN_EXT/MAX_EXT blend equations (WebGL1). */
export function createEXTBlendMinmax(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({ MIN_EXT: CExt.MIN_EXT, MAX_EXT: CExt.MAX_EXT });
}

/** EXT_texture_filter_anisotropic — anisotropic filtering params. */
export function createEXTTextureFilterAnisotropic(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    TEXTURE_MAX_ANISOTROPY_EXT: CExt.TEXTURE_MAX_ANISOTROPY_EXT,
    MAX_TEXTURE_MAX_ANISOTROPY_EXT: CExt.MAX_TEXTURE_MAX_ANISOTROPY_EXT,
  });
}

/** WEBGL_depth_texture — depth texture formats (UNSIGNED_INT_24_8_WEBGL etc.). */
export function createWEBGLDepthTexture(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    UNSIGNED_INT_24_8_WEBGL: CExt.UNSIGNED_INT_24_8_WEBGL,
    DEPTH_COMPONENT16: CExt.DEPTH_COMPONENT16,
    DEPTH_COMPONENT24: CExt.DEPTH_COMPONENT24,
    DEPTH_COMPONENT32F: CExt.DEPTH_COMPONENT32F,
    DEPTH24_STENCIL8: CExt.DEPTH24_STENCIL8,
    DEPTH_STENCIL: CExt.DEPTH_STENCIL,
  });
}

/** WEBGL_blend_func_extended — dual-source blending factors (no methods). */
export function createWEBGLBlendFuncExtended(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    SRC1_COLOR_WEBGL: CExt.SRC1_COLOR_WEBGL,
    SRC1_ALPHA_WEBGL: CExt.SRC1_ALPHA_WEBGL,
    ONE_MINUS_SRC1_COLOR_WEBGL: CExt.ONE_MINUS_SRC1_COLOR_WEBGL,
    ONE_MINUS_SRC1_ALPHA_WEBGL: CExt.ONE_MINUS_SRC1_ALPHA_WEBGL,
    MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL: CExt.MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL,
  });
}

/** WEBGL_debug_renderer_info — unmasked vendor/renderer pnames (no methods). */
export function createWEBGLDebugRendererInfo(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    UNMASKED_VENDOR_WEBGL: CExt.UNMASKED_VENDOR_WEBGL,
    UNMASKED_RENDERER_WEBGL: CExt.UNMASKED_RENDERER_WEBGL,
  });
}
