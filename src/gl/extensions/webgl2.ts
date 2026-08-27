/**
 * src/gl/extensions/webgl2.ts — WebGL2-side capability / constant-only factories.
 *
 * These extensions have no methods (per the pinned interfaces in types.ts and
 * the extension IDL): they exist so getExtension() returns a real object and the
 * feature gates (renderbuffer/texture formats, blend-float gating, COMPLETION_STATUS_KHR
 * shader query, RGB9_E5 renderability) open up in the parallel agents' validation
 * code — each gates on `ctx._extensions.has(name)` (the cache is populated by
 * getExtensionObject before the object is returned).
 *
 * Version visibility comes from the registry (extensions/index.ts): e.g.
 * EXT_color_buffer_float / EXT_texture_norm16 / WEBGL_render_shared_exponent are
 * WebGL2-only; EXT_color_buffer_half_float / EXT_float_blend also appear on
 * WebGL1 (where they promote the corresponding WebGL1 float formats).
 */

import { CExt } from '../constants';
import { buildExtension } from './util';
import type { WebGLRenderingContext } from '../webgl1';

/** EXT_color_buffer_float — float renderbuffer/attachment formats (WebGL2; no members). */
export function createEXTColorBufferFloat(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** EXT_color_buffer_half_float — half-float renderbuffer formats (W1+W2). */
export function createEXTColorBufferHalfFloat(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    RGB16F_EXT: CExt.RGB16F_EXT,
    RGBA16F_EXT: CExt.RGBA16F_EXT,
    FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT: CExt.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT,
    UNSIGNED_NORMALIZED_EXT: CExt.UNSIGNED_NORMALIZED_EXT,
  });
}

/** EXT_float_blend — blending of float framebuffers (no members). */
export function createEXTFloatBlend(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}

/** EXT_texture_norm16 — 16-bit normalized texture formats (WebGL2). */
export function createEXTTextureNorm16(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    R16_EXT: CExt.R16_EXT,
    RG16_EXT: CExt.RG16_EXT,
    RGB16_EXT: CExt.RGB16_EXT,
    RGBA16_EXT: CExt.RGBA16_EXT,
    R16_SNORM_EXT: CExt.R16_SNORM_EXT,
    RG16_SNORM_EXT: CExt.RG16_SNORM_EXT,
    RGB16_SNORM_EXT: CExt.RGB16_SNORM_EXT,
    RGBA16_SNORM_EXT: CExt.RGBA16_SNORM_EXT,
  });
}

/** KHR_parallel_shader_compile — COMPLETION_STATUS_KHR shader/program pname. */
export function createKHRParallelShaderCompile(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({ COMPLETION_STATUS_KHR: CExt.COMPLETION_STATUS_KHR });
}

/** WEBGL_render_shared_exponent — RGB9_E5 renderable (no members, verified from extension.xml). */
export function createWEBGLRenderSharedExponent(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({});
}
