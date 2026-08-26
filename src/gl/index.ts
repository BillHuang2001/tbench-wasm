/**
 * src/gl/index.ts — public barrel for the gl/ module.
 *
 * Contract surface for entry.ts, tests, and the rest of the renderer:
 *  - WebGLRenderingContext / WebGL2RenderingContext (full API surface, exact
 *    spec signatures; bodies installed by api/ mixins in Phase 2)
 *  - constants (C1/C2/C/CExt + installConstants)
 *  - engine entry points (draw.ts, teximage.ts, getters.ts, lifecycle.ts)
 *  - object classes, state model, errors, validation, extension registry
 *
 * gl/ is the SINGLE owner of GL state, objects, the error queue, validation,
 * attribute fetch + vertex evaluation, and draw-call assembly. It consumes
 * glsl/ (compileShader/linkProgram, contract §1), raster/ (DrawCall, formats,
 * sampler, occlusion hook — contracts §2/§3) and present/ (CanvasSurface,
 * ImageSource — contract §4). See ./CONTEXT.md for the full contract.
 */

// ---- Context classes (the core contract) ----
export { WebGLRenderingContext, CONTEXT_TOKEN } from './webgl1';
export { WebGL2RenderingContext } from './webgl2';

// ---- Constants ----
export { C1, C2, CExt, C, installConstants } from './constants';

// ---- Types ----
export type {
  GLenum,
  GLboolean,
  GLbitfield,
  GLbyte,
  GLshort,
  GLint,
  GLsizei,
  GLintptr,
  GLsizeiptr,
  GLuint,
  GLuint64,
  GLfloat,
  GLclampf,
  Float32List,
  Int32List,
  Uint32List,
  Uint8List,
  BufferDataSource,
  ContextType,
  CanvasLike,
  TexImageSource,
  WebGLPowerPreference,
  WebGLContextAttributes,
  WebGLContextAttributesInit,
} from './types';
export { DEFAULT_CONTEXT_ATTRIBUTES } from './types';

// ---- Object classes ----
export {
  WebGLObject,
  WebGLBuffer,
  WebGLTexture,
  type TextureLevel,
  WebGLShader,
  type ShaderCompileResult,
  WebGLProgram,
  type ProgramModel,
  WebGLFramebuffer,
  type FramebufferAttachment,
  WebGLRenderbuffer,
  WebGLVertexArrayObject,
  WebGLSampler,
  WebGLQuery,
  WebGLSync,
  WebGLTransformFeedback,
  WebGLUniformLocation,
  WebGLActiveInfo,
  WebGLShaderPrecisionFormat,
  validateUniformLocation,
  OBJECT_CLASSES,
  Resources,
  createObject,
} from './objects';

// ---- State model ----
export {
  defaultLimits,
  type Limits,
  defaultVertexAttrib,
  type VertexAttribState,
  defaultVAOState,
  type VAOState,
  type BlendState,
  type StencilState,
  type PixelStoreUnpack,
  type PixelStorePack,
  type TextureUnitState,
  type State,
  createDefaultState,
  KEEP,
} from './state';

// ---- Error queue & validation ----
export { ErrorQueue } from './errors';
export {
  validateObject,
  validateNullableObject,
  isNonNegativeInt,
  isFiniteNumber,
  isGLenum,
  requireBufferData,
  requireString,
} from './validation';

// ---- Engine entry points (deep pipeline logic) ----
export {
  type DrawRequest,
  executeDraw,
  executeClear,
  executeReadPixels,
  executeBlitFramebuffer,
  executeClearBuffer,
} from './draw';
export {
  type TexImageSourceArg,
  uploadTexImage,
  uploadTexSubImage,
  allocateImmutableStorage,
  copyTexImage,
  copyTexSubImage,
  compressedTexImage,
  generateMipmap,
} from './teximage';
export { getParameter } from './getters';
export {
  createContext,
  getContext,
  releaseContext,
  handleCanvasResize,
  loseContext,
  restoreContext,
} from './lifecycle';

// ---- Extensions ----
export {
  type ExtensionContextVersion,
  type ExtensionStatus,
  type ExtensionSpec,
  EXTENSION_SPECS,
  getSupportedExtensionNames,
  getExtensionObject,
} from './extensions';
export type {
  // WebGL1-visible (implemented)
  OES_texture_float,
  OES_texture_half_float,
  OES_texture_float_linear,
  OES_texture_half_float_linear,
  OES_element_index_uint,
  OES_standard_derivatives,
  OES_vertex_array_object,
  OES_fbo_render_mipmap,
  ANGLE_instanced_arrays,
  EXT_blend_minmax,
  EXT_frag_depth,
  EXT_sRGB,
  EXT_shader_texture_lod,
  EXT_texture_filter_anisotropic,
  WEBGL_depth_texture,
  WEBGL_draw_buffers,
  WEBGL_blend_func_extended,
  WEBGL_lose_context,
  WEBGL_debug_renderer_info,
  WEBGL_debug_shaders,
  EXT_clip_control,
  EXT_color_buffer_half_float,
  EXT_float_blend,
  // WebGL2-visible (implemented)
  EXT_color_buffer_float,
  EXT_texture_norm16,
  KHR_parallel_shader_compile,
  OES_draw_buffers_indexed,
  WEBGL_clip_cull_distance,
  WEBGL_multi_draw,
  WEBGL_multisampled_render_to_texture,
  WEBGL_render_shared_exponent,
  // status 'null' (advertised? no — returned null; declared for typing)
  WEBGL_compressed_texture_s3tc,
  WEBGL_compressed_texture_s3tc_srgb,
  WEBGL_compressed_texture_astc,
  WEBGL_compressed_texture_etc,
  WEBGL_compressed_texture_etc1,
  WEBGL_compressed_texture_pvrtc,
  EXT_texture_compression_bptc,
  EXT_texture_compression_rgtc,
  EXT_disjoint_timer_query,
  EXT_disjoint_timer_query_webgl2,
  EXT_depth_clamp,
  EXT_polygon_offset_clamp,
  WEBGL_polygon_mode,
  EXT_texture_mirror_clamp_to_edge,
} from './extensions/types';

// ---- Prototype-mixin aggregator (Phase 2 wiring point) ----
export { installAll } from './api';
