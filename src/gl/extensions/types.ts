/**
 * src/gl/extensions/types.ts — extension object shapes (public API of each extension).
 *
 * These interfaces are the contract for what getExtension(name) returns.
 * Constants come from CExt (constants.ts). Extensions marked "null" in the
 * registry are NOT implemented: getExtension returns null and getSupportedExtensions
 * does not list them (the CTS invariant: listing ⟺ getExtension succeeds).
 *
 * Only the extensions the objective mandates are typed here; compressed formats
 * and timer queries are status 'null' (documented in index.ts) and share loose
 * shapes below for future implementation.
 */

import type { GLenum, GLint, GLsizei, GLuint } from '../types';

// ---------------------------------------------------------------------------
// WebGL1 extensions
// ---------------------------------------------------------------------------
export interface OES_texture_float { } // adds float texture formats (no members)
export interface OES_texture_half_float { readonly HALF_FLOAT_OES: GLenum }
export interface OES_texture_float_linear { }
export interface OES_texture_half_float_linear { }
export interface OES_element_index_uint { }
export interface OES_standard_derivatives { readonly FRAGMENT_SHADER_DERIVATIVE_HINT_OES: GLenum }
export interface OES_vertex_array_object {
  readonly VERTEX_ARRAY_BINDING_OES: GLenum;
  createVertexArrayOES(): WebGLVertexArrayObject | null;
  deleteVertexArrayOES(vertexArray: WebGLVertexArrayObject | null): void;
  isVertexArrayOES(vertexArray: WebGLVertexArrayObject | null): GLboolean;
  bindVertexArrayOES(vertexArray: WebGLVertexArrayObject | null): void;
}
export interface OES_fbo_render_mipmap { }
export interface ANGLE_instanced_arrays {
  readonly VERTEX_ATTRIB_ARRAY_DIVISOR_ANGLE: GLenum;
  drawArraysInstancedANGLE(mode: GLenum, first: GLint, count: GLsizei, primcount: GLsizei): void;
  drawElementsInstancedANGLE(mode: GLenum, count: GLsizei, type: GLenum, offset: GLintptr, primcount: GLsizei): void;
  vertexAttribDivisorANGLE(index: GLuint, divisor: GLuint): void;
}
export interface EXT_blend_minmax { readonly MIN_EXT: GLenum; readonly MAX_EXT: GLenum }
export interface EXT_frag_depth { }
export interface EXT_sRGB {
  readonly SRGB_EXT: GLenum;
  readonly SRGB_ALPHA_EXT: GLenum;
  readonly SRGB8_ALPHA8_EXT: GLenum;
  readonly FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING_EXT: GLenum;
}
export interface EXT_shader_texture_lod { }
export interface EXT_texture_filter_anisotropic {
  readonly TEXTURE_MAX_ANISOTROPY_EXT: GLenum;
  readonly MAX_TEXTURE_MAX_ANISOTROPY_EXT: GLenum;
}
export interface WEBGL_depth_texture {
  readonly UNSIGNED_INT_24_8_WEBGL: GLenum;
  readonly DEPTH_COMPONENT16: GLenum;
  readonly DEPTH_COMPONENT24: GLenum;
  readonly DEPTH_COMPONENT32F: GLenum;
  readonly DEPTH24_STENCIL8: GLenum;
  readonly DEPTH_STENCIL: GLenum;
}
export interface WEBGL_draw_buffers {
  readonly DRAW_BUFFER0_WEBGL: GLenum;
  readonly DRAW_BUFFER1_WEBGL: GLenum;
  readonly DRAW_BUFFER2_WEBGL: GLenum;
  readonly DRAW_BUFFER3_WEBGL: GLenum;
  readonly DRAW_BUFFER4_WEBGL: GLenum;
  readonly DRAW_BUFFER5_WEBGL: GLenum;
  readonly DRAW_BUFFER6_WEBGL: GLenum;
  readonly DRAW_BUFFER7_WEBGL: GLenum;
  readonly DRAW_BUFFER8_WEBGL: GLenum;
  readonly DRAW_BUFFER9_WEBGL: GLenum;
  readonly DRAW_BUFFER10_WEBGL: GLenum;
  readonly DRAW_BUFFER11_WEBGL: GLenum;
  readonly DRAW_BUFFER12_WEBGL: GLenum;
  readonly DRAW_BUFFER13_WEBGL: GLenum;
  readonly DRAW_BUFFER14_WEBGL: GLenum;
  readonly DRAW_BUFFER15_WEBGL: GLenum;
  readonly MAX_DRAW_BUFFERS_WEBGL: GLenum;
  readonly MAX_COLOR_ATTACHMENTS_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT0_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT1_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT2_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT3_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT4_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT5_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT6_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT7_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT8_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT9_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT10_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT11_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT12_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT13_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT14_WEBGL: GLenum;
  readonly COLOR_ATTACHMENT15_WEBGL: GLenum;
  drawBuffersWEBGL(buffers: GLenum[]): void;
}
export interface WEBGL_blend_func_extended {
  readonly SRC1_COLOR_WEBGL: GLenum;
  readonly SRC1_ALPHA_WEBGL: GLenum;
  readonly ONE_MINUS_SRC1_COLOR_WEBGL: GLenum;
  readonly ONE_MINUS_SRC1_ALPHA_WEBGL: GLenum;
  readonly MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL: GLenum;
}
export interface WEBGL_lose_context {
  loseContext(): void;
  restoreContext(): void;
}
export interface WEBGL_debug_renderer_info {
  readonly UNMASKED_VENDOR_WEBGL: GLenum;
  readonly UNMASKED_RENDERER_WEBGL: GLenum;
}
export interface WEBGL_debug_shaders {
  getTranslatedShaderSource(shader: WebGLShader): string;
}
// WebGL1 (also on WebGL2 for some)
export interface EXT_disjoint_timer_query {
  readonly QUERY_COUNTER_BITS_EXT: GLenum;
  readonly CURRENT_QUERY_EXT: GLenum;
  readonly QUERY_RESULT_EXT: GLenum;
  readonly QUERY_RESULT_AVAILABLE_EXT: GLenum;
  readonly TIME_ELAPSED_EXT: GLenum;
  readonly TIMESTAMP_EXT: GLenum;
  readonly GPU_DISJOINT_EXT: GLenum;
  createQueryEXT(): WebGLQuery | null;
  deleteQueryEXT(query: WebGLQuery | null): void;
  isQueryEXT(query: WebGLQuery | null): GLboolean;
  beginQueryEXT(target: GLenum, query: WebGLQuery): void;
  endQueryEXT(target: GLenum): void;
  queryCounterEXT(query: WebGLQuery, target: GLenum): void;
  getQueryEXT(target: GLenum, pname: GLenum): WebGLQuery | null;
  getQueryObjectEXT(query: WebGLQuery, pname: GLenum): unknown;
}

// ---------------------------------------------------------------------------
// WebGL2-only extensions
// ---------------------------------------------------------------------------
export interface EXT_color_buffer_float { }
export interface EXT_color_buffer_half_float {
  readonly RGB16F_EXT: GLenum;
  readonly RGBA16F_EXT: GLenum;
  readonly FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT: GLenum;
  readonly UNSIGNED_NORMALIZED_EXT: GLenum;
}
export interface EXT_float_blend { }
export interface EXT_texture_norm16 {
  readonly R16_EXT: GLenum;
  readonly RG16_EXT: GLenum;
  readonly RGB16_EXT: GLenum;
  readonly RGBA16_EXT: GLenum;
  readonly R16_SNORM_EXT: GLenum;
  readonly RG16_SNORM_EXT: GLenum;
  readonly RGB16_SNORM_EXT: GLenum;
  readonly RGBA16_SNORM_EXT: GLenum;
}
export interface EXT_clip_control {
  readonly CLIP_ORIGIN_EXT: GLenum;
  readonly CLIP_DEPTH_MODE_EXT: GLenum;
  readonly LOWER_LEFT_EXT: GLenum;
  readonly UPPER_LEFT_EXT: GLenum;
  readonly NEGATIVE_ONE_TO_ONE_EXT: GLenum;
  readonly ZERO_TO_ONE_EXT: GLenum;
  clipControlEXT(origin: GLenum, depth: GLenum): void;
}
export interface KHR_parallel_shader_compile { readonly COMPLETION_STATUS_KHR: GLenum }
export interface OES_draw_buffers_indexed {
  enableiOES(target: GLenum, index: GLuint): void;
  disableiOES(target: GLenum, index: GLuint): void;
  blendEquationiOES(buf: GLuint, mode: GLenum): void;
  blendEquationSeparateiOES(buf: GLuint, modeRGB: GLenum, modeAlpha: GLenum): void;
  blendFunciOES(buf: GLuint, src: GLenum, dst: GLenum): void;
  blendFuncSeparateiOES(buf: GLuint, srcRGB: GLenum, dstRGB: GLenum, srcAlpha: GLenum, dstAlpha: GLenum): void;
  colorMaskiOES(buf: GLuint, r: GLboolean, g: GLboolean, b: GLboolean, a: GLboolean): void;
}
export interface WEBGL_multi_draw {
  multiDrawArraysWEBGL(mode: GLenum, firsts: Int32List, counts: Int32List, drawcount: GLsizei): void;
  multiDrawElementsWEBGL(mode: GLenum, counts: Int32List, type: GLenum, offsets: Int32List, drawcount: GLsizei): void;
  multiDrawArraysInstancedWEBGL(mode: GLenum, firsts: Int32List, counts: Int32List, instanceCounts: Int32List, drawcount: GLsizei): void;
  multiDrawElementsInstancedWEBGL(mode: GLenum, counts: Int32List, type: GLenum, offsets: Int32List, instanceCounts: Int32List, drawcount: GLsizei): void;
}
export interface WEBGL_clip_cull_distance {
  readonly MAX_CLIP_DISTANCES_WEBGL: GLenum;
  readonly MAX_CULL_DISTANCES_WEBGL: GLenum;
  readonly MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL: GLenum;
  readonly CLIP_DISTANCE0_WEBGL: GLenum;
  readonly CLIP_DISTANCE1_WEBGL: GLenum;
  readonly CLIP_DISTANCE2_WEBGL: GLenum;
  readonly CLIP_DISTANCE3_WEBGL: GLenum;
  readonly CLIP_DISTANCE4_WEBGL: GLenum;
  readonly CLIP_DISTANCE5_WEBGL: GLenum;
  readonly CLIP_DISTANCE6_WEBGL: GLenum;
  readonly CLIP_DISTANCE7_WEBGL: GLenum;
}
export interface WEBGL_multisampled_render_to_texture {
  framebufferTexture2DMultisampleEXT(target: GLenum, attachment: GLenum, textarget: GLenum, texture: WebGLTexture | null, level: GLint, samples: GLsizei): void;
  renderbufferStorageMultisampleEXT(target: GLenum, samples: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei): void;
}
export interface WEBGL_render_shared_exponent { }
export interface EXT_disjoint_timer_query_webgl2 extends EXT_disjoint_timer_query { }
export interface WEBGL_polygon_mode { } // null (not implemented)
export interface EXT_depth_clamp { } // null
export interface EXT_polygon_offset_clamp { } // null
export interface EXT_texture_mirror_clamp_to_edge { } // null
export interface EXT_conservative_depth { } // null (WebGL2, not implemented)
export interface EXT_render_snorm { } // null (WebGL2)
export interface OVR_multiview2 { } // null (WebGL2)

// Compressed texture extensions (status 'null' — see index.ts)
export interface WEBGL_compressed_texture_s3tc {
  readonly COMPRESSED_RGB_S3TC_DXT1_EXT: GLenum;
  readonly COMPRESSED_RGBA_S3TC_DXT1_EXT: GLenum;
  readonly COMPRESSED_RGBA_S3TC_DXT3_EXT: GLenum;
  readonly COMPRESSED_RGBA_S3TC_DXT5_EXT: GLenum;
}
export interface WEBGL_compressed_texture_s3tc_srgb extends WEBGL_compressed_texture_s3tc { }
export interface WEBGL_compressed_texture_astc { }
export interface WEBGL_compressed_texture_etc { }
export interface WEBGL_compressed_texture_etc1 { }
export interface WEBGL_compressed_texture_pvrtc { }
export interface EXT_texture_compression_bptc { }
export interface EXT_texture_compression_rgtc { }

import type { GLintptr, GLboolean, Int32List } from '../types';
import type { WebGLShader } from '../objects';
import type { WebGLQuery } from '../objects';
import type { WebGLTexture } from '../objects';
import type { WebGLVertexArrayObject } from '../objects';
