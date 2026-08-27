/**
 * src/gl/webgl1.ts — class WebGLRenderingContext: the FULL WebGL 1.0 API surface.
 *
 * THIS FILE IS THE CONTRACT: every WebGL1 method with the exact spec signature
 * (parameter order, arity — some CTS tests check function.length) and property
 * getters. Bodies are stubs for Phase 1; the api/ modules (see api/CONTEXT and
 * this file's Routing Table in CONTEXT.md) own the real implementations in
 * Phase 2. Constants (C1) are installed on the prototype at module load.
 *
 * WebGL2RenderingContext (webgl2.ts) extends this class; WebGL2 contexts inherit
 * every method here through the prototype chain.
 *
 * Overloads (texImage2D/texSubImage2D): declared as a single max-arity signature
 * with union types — WebIDL function length = max overload arity (9), and the
 * runtime dispatches on argument count/types (api/teximage.ts).
 */

import type {
  BufferDataSource,
  CanvasLike,
  ContextType,
  Float32List,
  GLbitfield,
  GLboolean,
  GLclampf,
  GLenum,
  GLfloat,
  GLint,
  GLintptr,
  GLsizei,
  GLsizeiptr,
  GLuint,
  Int32List,
  TexImageSource,
  WebGLContextAttributes,
  WebGLContextAttributesInit,
} from './types';
import { C1, installConstants } from './constants';
import { ErrorQueue } from './errors';
import { createDefaultState, type State } from './state';
import { installAll } from './api';
import { chainToNative } from './native-chain';
import { Resources } from './objects';
import type {
  WebGLActiveInfo,
  WebGLBuffer,
  WebGLFramebuffer,
  WebGLProgram,
  WebGLRenderbuffer,
  WebGLShader,
  WebGLShaderPrecisionFormat,
  WebGLTexture,
  WebGLUniformLocation,
} from './objects';
import type {
  OES_element_index_uint,
  OES_fbo_render_mipmap,
  OES_standard_derivatives,
  OES_texture_float,
  OES_texture_float_linear,
  OES_texture_half_float,
  OES_texture_half_float_linear,
  OES_vertex_array_object,
  ANGLE_instanced_arrays,
  EXT_blend_minmax,
  EXT_frag_depth,
  EXT_sRGB,
  EXT_shader_texture_lod,
  EXT_texture_filter_anisotropic,
  WEBGL_blend_func_extended,
  WEBGL_debug_renderer_info,
  WEBGL_debug_shaders,
  WEBGL_depth_texture,
  WEBGL_draw_buffers,
  WEBGL_lose_context,
  EXT_clip_control,
  EXT_color_buffer_half_float,
  EXT_float_blend,
  KHR_parallel_shader_compile,
  WEBGL_multi_draw,
} from './extensions/types';
import { DEFAULT_CONTEXT_ATTRIBUTES } from './types';
import type { CanvasSurface } from '../present';
import type { Surface, TextureImage } from '../raster';
import type { VAOState } from './state';

/**
 * The default framebuffer (drawing buffer) of a context. `color.data` is the
 * present surface's pixel buffer (zero-copy RGBA8; re-fetched after resize);
 * `depth`/`stencil` are raster surfaces allocated per context attributes
 * (DEPTH_COMPONENT16 on WebGL1 / DEPTH_COMPONENT24 on WebGL2 / STENCIL_INDEX8).
 * Lifecycle (lifecycle.ts) owns allocation; the draw pipeline resolves it to a
 * raster FramebufferTarget via framebuffer-util.ts.
 */
export interface DefaultFramebuffer {
  color: Surface;
  depth: Surface | null;
  stencil: Surface | null;
  width: number;
  height: number;
}

/** Token gate for construction — only lifecycle.ts may create contexts. */
export const CONTEXT_TOKEN: unique symbol = Symbol('software-webgl-context');

let nextContextId = 1;

/**
 * Drawing-buffer dimension: max(1, canvas.width/height), clamped to the max
 * viewport dims (spec: the drawing buffer is capped at MAX_VIEWPORT_DIMS — CTS
 * canvas/drawingbuffer-static-canvas-test.html asserts drawingBufferWidth <=
 * getParameter(MAX_VIEWPORT_DIMS)[0] for oversized canvases); 0 when absent (mocks).
 */
function drawingBufferDim(v: unknown, max: number): GLsizei {
  const d = typeof v === 'number' ? Math.max(1, v) : 0;
  return max > 0 && d > max ? max : d;
}

export class WebGLRenderingContext {
  // ---- internal engine state (underscore-prefixed; not part of the public API) ----
  /** 1 for WebGL1, 2 for WebGL2. */
  _version: 1 | 2 = 1;
  readonly _canvas: CanvasLike;
  readonly _attrs: WebGLContextAttributes;
  readonly _type: ContextType;
  readonly _contextId: number;
  _state: State;
  _errors: ErrorQueue;
  _resources: Resources;
  /** Extension singleton cache (canonical name → object). */
  _extensions: Map<string, object> = new Map();
  /** Present adapter (present/ contract §4) — set at construction; null before/after loss. */
  _presentSurface: CanvasSurface | null = null;
  /** The default framebuffer (drawing buffer) surfaces — owned by lifecycle.ts. */
  _defaultFB: DefaultFramebuffer | null = null;
  /** The persistent default VAO contents (WebGL2/OES_vertex_array_object rebind target). */
  _defaultVAO: VAOState | null = null;
  /** Default framebuffer surface (present/ CanvasSurface-compatible). */
  _drawingBuffer: unknown = null;
  _drawingBufferWidth: GLsizei = 0;
  _drawingBufferHeight: GLsizei = 0;
  /** WEBGL_lose_context state. */
  _isLost = false;
  /** Drawing-buffer clear/resize bookkeeping (preserveDrawingBuffer). */
  _needsPresent = false;

  /**
   * @internal — only lifecycle.ts may construct (via CONTEXT_TOKEN).
   */
  constructor(
    canvas: CanvasLike,
    attrs: WebGLContextAttributesInit | null,
    type: ContextType,
    token: symbol,
  ) {
    if (token !== CONTEXT_TOKEN) {
      throw new TypeError('Illegal constructor: use __createSoftwareWebGLContext()');
    }
    this._canvas = canvas;
    this._type = type;
    // antialias is normalized to false for WebGL1: the drawing buffer is
    // single-sampled, and CTS context-attributes-alpha-depth-stencil-antialias.html
    // REQUIRES real antialiased edges when antialias is reported true (WebGL1
    // tolerates reporting false when true was requested). WebGL2 must report the
    // requested value exactly (conformance2 context-attributes-depth-stencil-
    // antialias-obeyed.html) — webgl2.ts restores it after super().
    this._attrs = { ...DEFAULT_CONTEXT_ATTRIBUTES, ...(attrs ?? {}), antialias: false };
    this._contextId = nextContextId++;
    this._state = createDefaultState(1);
    this._errors = new ErrorQueue();
    this._resources = new Resources(this);
    // Drawing-buffer dimensions: max(1, canvas.width/height) — a 0-size canvas
    // yields a 1×1 buffer (CTS context/zero-sized-canvas.html); stored value is
    // the fallback for width-less mocks (the getters are live, see below).
    this._drawingBufferWidth = drawingBufferDim(canvas.width, this._state.limits.MAX_VIEWPORT_DIMS[0]);
    this._drawingBufferHeight = drawingBufferDim(canvas.height, this._state.limits.MAX_VIEWPORT_DIMS[1]);
    // Present adapter + default framebuffer + default VAO + initial viewport are
    // initialized by lifecycle.createContext via initContextResources() — the
    // ONLY construction site (CONTEXT_TOKEN gate), called AFTER the final
    // version is known (WebGL2 rebuilds state in its constructor).
  }

  // ---- Properties (spec: getters on the prototype) ----
  get canvas(): CanvasLike {
    return this._canvas;
  }
  /**
   * Live drawing-buffer size: tracks canvas.width/height immediately (CTS
   * canvas/drawingbuffer-test.html: drawingBufferWidth === canvas.width;
   * context/zero-sized-canvas.html: immediate update after width set, 0 → 1).
   */
  get drawingBufferWidth(): GLsizei {
    return drawingBufferDim(this._canvas.width, this._state.limits.MAX_VIEWPORT_DIMS[0]);
  }
  get drawingBufferHeight(): GLsizei {
    return drawingBufferDim(this._canvas.height, this._state.limits.MAX_VIEWPORT_DIMS[1]);
  }
  /**
   * Drawing-buffer color space ('srgb' default), unpack color space, and drawing
   * buffer format (RGBA8). These shadow the NATIVE WebIDL accessors: after
   * chainToNative() re-chains our prototypes under the native ones, the native
   * accessors would otherwise run on our non-native `this` and throw
   * "Illegal invocation" (WebIDL brand check). three.js/Babylon read AND assign
   * these unguarded, so the setters must never throw (the renderer stays sRGB
   * regardless of what is assigned — native enum validation is deliberately
   * skipped to avoid crashing engines).
   */
  get drawingBufferColorSpace(): string {
    return 'srgb';
  }
  set drawingBufferColorSpace(_value: string) {
    // no-op — software renderer always outputs sRGB
  }
  get unpackColorSpace(): string {
    return 'srgb';
  }
  set unpackColorSpace(_value: string) {
    // no-op — image uploads are interpreted as sRGB
  }
  /** Format of the drawing buffer: RGBA8 (0x8058). */
  get drawingBufferFormat(): GLenum {
    return C1.RGBA8;
  }

  // ---- Context attributes & lifecycle ----
  getContextAttributes(): WebGLContextAttributes | null { throw new Error('GL stub'); }
  isContextLost(): GLboolean { throw new Error('GL stub'); }
  getSupportedExtensions(): string[] | null { throw new Error('GL stub'); }
  getExtension(name: string): any { throw new Error('GL stub'); }
  getError(): GLenum { throw new Error('GL stub'); }
  getString(name: GLenum): string | null { throw new Error('GL stub'); }
  getParameter(pname: GLenum): any { throw new Error('GL stub'); }

  // ---- State setters ----
  activeTexture(texture: GLenum): void { throw new Error('GL stub'); }
  blendColor(red: GLclampf, green: GLclampf, blue: GLclampf, alpha: GLclampf): void { throw new Error('GL stub'); }
  blendEquation(mode: GLenum): void { throw new Error('GL stub'); }
  blendEquationSeparate(modeRGB: GLenum, modeAlpha: GLenum): void { throw new Error('GL stub'); }
  blendFunc(sfactor: GLenum, dfactor: GLenum): void { throw new Error('GL stub'); }
  blendFuncSeparate(srcRGB: GLenum, dstRGB: GLenum, srcAlpha: GLenum, dstAlpha: GLenum): void { throw new Error('GL stub'); }
  clearColor(red: GLclampf, green: GLclampf, blue: GLclampf, alpha: GLclampf): void { throw new Error('GL stub'); }
  clearDepth(depth: GLclampf): void { throw new Error('GL stub'); }
  clearStencil(s: GLint): void { throw new Error('GL stub'); }
  colorMask(red: GLboolean, green: GLboolean, blue: GLboolean, alpha: GLboolean): void { throw new Error('GL stub'); }
  cullFace(mode: GLenum): void { throw new Error('GL stub'); }
  depthFunc(func: GLenum): void { throw new Error('GL stub'); }
  depthMask(flag: GLboolean): void { throw new Error('GL stub'); }
  depthRange(zNear: GLclampf, zFar: GLclampf): void { throw new Error('GL stub'); }
  disable(cap: GLenum): void { throw new Error('GL stub'); }
  enable(cap: GLenum): void { throw new Error('GL stub'); }
  frontFace(mode: GLenum): void { throw new Error('GL stub'); }
  hint(target: GLenum, mode: GLenum): void { throw new Error('GL stub'); }
  isEnabled(cap: GLenum): GLboolean { throw new Error('GL stub'); }
  lineWidth(width: GLfloat): void { throw new Error('GL stub'); }
  pixelStorei(pname: GLenum, param: GLint): void { throw new Error('GL stub'); }
  polygonOffset(factor: GLfloat, units: GLfloat): void { throw new Error('GL stub'); }
  sampleCoverage(value: GLclampf, invert: GLboolean): void { throw new Error('GL stub'); }
  scissor(x: GLint, y: GLint, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }
  stencilFunc(func: GLenum, ref: GLint, mask: GLuint): void { throw new Error('GL stub'); }
  stencilFuncSeparate(face: GLenum, func: GLenum, ref: GLint, mask: GLuint): void { throw new Error('GL stub'); }
  stencilMask(mask: GLuint): void { throw new Error('GL stub'); }
  stencilMaskSeparate(face: GLenum, mask: GLuint): void { throw new Error('GL stub'); }
  stencilOp(fail: GLenum, zfail: GLenum, zpass: GLenum): void { throw new Error('GL stub'); }
  stencilOpSeparate(face: GLenum, fail: GLenum, zfail: GLenum, zpass: GLenum): void { throw new Error('GL stub'); }
  viewport(x: GLint, y: GLint, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }

  // ---- Buffer objects ----
  createBuffer(): WebGLBuffer | null { throw new Error('GL stub'); }
  deleteBuffer(buffer: WebGLBuffer | null): void { throw new Error('GL stub'); }
  isBuffer(buffer: WebGLBuffer | null): GLboolean { throw new Error('GL stub'); }
  bindBuffer(target: GLenum, buffer: WebGLBuffer | null): void { throw new Error('GL stub'); }
  bufferData(target: GLenum, size: GLsizeiptr | BufferDataSource, usage: GLenum): void { throw new Error('GL stub'); }
  bufferSubData(target: GLenum, offset: GLintptr, data: BufferDataSource): void { throw new Error('GL stub'); }
  getBufferParameter(target: GLenum, pname: GLenum): any { throw new Error('GL stub'); }

  // ---- Framebuffers ----
  createFramebuffer(): WebGLFramebuffer | null { throw new Error('GL stub'); }
  deleteFramebuffer(framebuffer: WebGLFramebuffer | null): void { throw new Error('GL stub'); }
  isFramebuffer(framebuffer: WebGLFramebuffer | null): GLboolean { throw new Error('GL stub'); }
  bindFramebuffer(target: GLenum, framebuffer: WebGLFramebuffer | null): void { throw new Error('GL stub'); }
  framebufferRenderbuffer(target: GLenum, attachment: GLenum, renderbuffertarget: GLenum, renderbuffer: WebGLRenderbuffer | null): void { throw new Error('GL stub'); }
  framebufferTexture2D(target: GLenum, attachment: GLenum, textarget: GLenum, texture: WebGLTexture | null, level: GLint): void { throw new Error('GL stub'); }
  checkFramebufferStatus(target: GLenum): GLenum { throw new Error('GL stub'); }
  getFramebufferAttachmentParameter(target: GLenum, attachment: GLenum, pname: GLenum): any { throw new Error('GL stub'); }

  // ---- Renderbuffers ----
  createRenderbuffer(): WebGLRenderbuffer | null { throw new Error('GL stub'); }
  deleteRenderbuffer(renderbuffer: WebGLRenderbuffer | null): void { throw new Error('GL stub'); }
  isRenderbuffer(renderbuffer: WebGLRenderbuffer | null): GLboolean { throw new Error('GL stub'); }
  bindRenderbuffer(target: GLenum, renderbuffer: WebGLRenderbuffer | null): void { throw new Error('GL stub'); }
  renderbufferStorage(target: GLenum, internalformat: GLenum, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }
  getRenderbufferParameter(target: GLenum, pname: GLenum): any { throw new Error('GL stub'); }

  // ---- Shaders & programs ----
  createProgram(): WebGLProgram | null { throw new Error('GL stub'); }
  deleteProgram(program: WebGLProgram | null): void { throw new Error('GL stub'); }
  isProgram(program: WebGLProgram | null): GLboolean { throw new Error('GL stub'); }
  useProgram(program: WebGLProgram | null): void { throw new Error('GL stub'); }
  createShader(type: GLenum): WebGLShader | null { throw new Error('GL stub'); }
  deleteShader(shader: WebGLShader | null): void { throw new Error('GL stub'); }
  isShader(shader: WebGLShader | null): GLboolean { throw new Error('GL stub'); }
  shaderSource(shader: WebGLShader, source: string): void { throw new Error('GL stub'); }
  compileShader(shader: WebGLShader): void { throw new Error('GL stub'); }
  attachShader(program: WebGLProgram, shader: WebGLShader): void { throw new Error('GL stub'); }
  detachShader(program: WebGLProgram, shader: WebGLShader): void { throw new Error('GL stub'); }
  linkProgram(program: WebGLProgram): void { throw new Error('GL stub'); }
  validateProgram(program: WebGLProgram): void { throw new Error('GL stub'); }
  getShaderParameter(shader: WebGLShader, pname: GLenum): any { throw new Error('GL stub'); }
  getProgramParameter(program: WebGLProgram, pname: GLenum): any { throw new Error('GL stub'); }
  getShaderInfoLog(shader: WebGLShader): string | null { throw new Error('GL stub'); }
  getProgramInfoLog(program: WebGLProgram): string | null { throw new Error('GL stub'); }
  getShaderSource(shader: WebGLShader): string | null { throw new Error('GL stub'); }
  getShaderPrecisionFormat(shadertype: GLenum, precisiontype: GLenum): WebGLShaderPrecisionFormat | null { throw new Error('GL stub'); }
  getAttribLocation(program: WebGLProgram, name: string): GLint { throw new Error('GL stub'); }
  getUniformLocation(program: WebGLProgram, name: string): WebGLUniformLocation | null { throw new Error('GL stub'); }
  bindAttribLocation(program: WebGLProgram, index: GLuint, name: string): void { throw new Error('GL stub'); }
  getActiveAttrib(program: WebGLProgram, index: GLuint): WebGLActiveInfo | null { throw new Error('GL stub'); }
  getActiveUniform(program: WebGLProgram, index: GLuint): WebGLActiveInfo | null { throw new Error('GL stub'); }
  getUniform(program: WebGLProgram, location: WebGLUniformLocation): any { throw new Error('GL stub'); }

  // ---- Uniform setters ----
  uniform1f(location: WebGLUniformLocation | null, x: GLfloat): void { throw new Error('GL stub'); }
  uniform1fv(location: WebGLUniformLocation | null, v: Float32List): void { throw new Error('GL stub'); }
  uniform1i(location: WebGLUniformLocation | null, x: GLint): void { throw new Error('GL stub'); }
  uniform1iv(location: WebGLUniformLocation | null, v: Int32List): void { throw new Error('GL stub'); }
  uniform2f(location: WebGLUniformLocation | null, x: GLfloat, y: GLfloat): void { throw new Error('GL stub'); }
  uniform2fv(location: WebGLUniformLocation | null, v: Float32List): void { throw new Error('GL stub'); }
  uniform2i(location: WebGLUniformLocation | null, x: GLint, y: GLint): void { throw new Error('GL stub'); }
  uniform2iv(location: WebGLUniformLocation | null, v: Int32List): void { throw new Error('GL stub'); }
  uniform3f(location: WebGLUniformLocation | null, x: GLfloat, y: GLfloat, z: GLfloat): void { throw new Error('GL stub'); }
  uniform3fv(location: WebGLUniformLocation | null, v: Float32List): void { throw new Error('GL stub'); }
  uniform3i(location: WebGLUniformLocation | null, x: GLint, y: GLint, z: GLint): void { throw new Error('GL stub'); }
  uniform3iv(location: WebGLUniformLocation | null, v: Int32List): void { throw new Error('GL stub'); }
  uniform4f(location: WebGLUniformLocation | null, x: GLfloat, y: GLfloat, z: GLfloat, w: GLfloat): void { throw new Error('GL stub'); }
  uniform4fv(location: WebGLUniformLocation | null, v: Float32List): void { throw new Error('GL stub'); }
  uniform4i(location: WebGLUniformLocation | null, x: GLint, y: GLint, z: GLint, w: GLint): void { throw new Error('GL stub'); }
  uniform4iv(location: WebGLUniformLocation | null, v: Int32List): void { throw new Error('GL stub'); }
  uniformMatrix2fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }
  uniformMatrix3fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }
  uniformMatrix4fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }

  // ---- Texture objects & parameters ----
  createTexture(): WebGLTexture | null { throw new Error('GL stub'); }
  deleteTexture(texture: WebGLTexture | null): void { throw new Error('GL stub'); }
  isTexture(texture: WebGLTexture | null): GLboolean { throw new Error('GL stub'); }
  bindTexture(target: GLenum, texture: WebGLTexture | null): void { throw new Error('GL stub'); }
  texParameterf(target: GLenum, pname: GLenum, param: GLfloat): void { throw new Error('GL stub'); }
  texParameteri(target: GLenum, pname: GLenum, param: GLint): void { throw new Error('GL stub'); }
  getTexParameter(target: GLenum, pname: GLenum): any { throw new Error('GL stub'); }

  /**
   * texImage2D — all WebGL1 overloads (max arity 9):
   *  (target, level, internalformat, width, height, border, format, type, pixels?)
   *  (target, level, internalformat, format, type, source?)   // ImageData/HTMLImageElement/HTMLCanvasElement/HTMLVideoElement/ImageBitmap/OffscreenCanvas
   */
  texImage2D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, border: GLint, format: GLenum, type: GLenum, pixels?: ArrayBufferView | TexImageSource | null): void { throw new Error('GL stub'); }
  /**
   * texSubImage2D — max arity 9:
   *  (target, level, xoffset, yoffset, width, height, format, type, pixels?)
   *  (target, level, xoffset, yoffset, format, type, source?)
   */
  texSubImage2D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, width: GLsizei, height: GLsizei, format: GLenum, type: GLenum, pixels?: ArrayBufferView | TexImageSource | null): void { throw new Error('GL stub'); }
  copyTexImage2D(target: GLenum, level: GLint, internalformat: GLenum, x: GLint, y: GLint, width: GLsizei, height: GLsizei, border: GLint): void { throw new Error('GL stub'); }
  copyTexSubImage2D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, x: GLint, y: GLint, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }
  compressedTexImage2D(target: GLenum, level: GLint, internalformat: GLenum, width: GLsizei, height: GLsizei, border: GLint, data: ArrayBufferView): void { throw new Error('GL stub'); }
  compressedTexSubImage2D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, width: GLsizei, height: GLsizei, format: GLenum, data: ArrayBufferView): void { throw new Error('GL stub'); }
  generateMipmap(target: GLenum): void { throw new Error('GL stub'); }

  // ---- Vertex attrib arrays ----
  vertexAttribPointer(index: GLuint, size: GLint, type: GLenum, normalized: GLboolean, stride: GLsizei, offset: GLintptr): void { throw new Error('GL stub'); }
  enableVertexAttribArray(index: GLuint): void { throw new Error('GL stub'); }
  disableVertexAttribArray(index: GLuint): void { throw new Error('GL stub'); }
  vertexAttrib1f(index: GLuint, x: GLfloat): void { throw new Error('GL stub'); }
  vertexAttrib1fv(index: GLuint, values: Float32List): void { throw new Error('GL stub'); }
  vertexAttrib2f(index: GLuint, x: GLfloat, y: GLfloat): void { throw new Error('GL stub'); }
  vertexAttrib2fv(index: GLuint, values: Float32List): void { throw new Error('GL stub'); }
  vertexAttrib3f(index: GLuint, x: GLfloat, y: GLfloat, z: GLfloat): void { throw new Error('GL stub'); }
  vertexAttrib3fv(index: GLuint, values: Float32List): void { throw new Error('GL stub'); }
  vertexAttrib4f(index: GLuint, x: GLfloat, y: GLfloat, z: GLfloat, w: GLfloat): void { throw new Error('GL stub'); }
  vertexAttrib4fv(index: GLuint, values: Float32List): void { throw new Error('GL stub'); }
  getVertexAttrib(index: GLuint, pname: GLenum): any { throw new Error('GL stub'); }
  getVertexAttribOffset(index: GLuint, pname: GLenum): GLintptr { throw new Error('GL stub'); }

  // ---- Drawing ----
  drawArrays(mode: GLenum, first: GLint, count: GLsizei): void { throw new Error('GL stub'); }
  drawElements(mode: GLenum, count: GLsizei, type: GLenum, offset: GLintptr): void { throw new Error('GL stub'); }
  clear(mask: GLbitfield): void { throw new Error('GL stub'); }
  flush(): void { throw new Error('GL stub'); }
  finish(): void { throw new Error('GL stub'); }
  readPixels(x: GLint, y: GLint, width: GLsizei, height: GLsizei, format: GLenum, type: GLenum, pixels: ArrayBufferView | null): void { throw new Error('GL stub'); }

  /**
   * @internal installed via installDrawApi — see api/draw.ts
   */
  _transferToImageBitmap(): { width: number; height: number; data: Uint8ClampedArray } | null {
    throw new Error('GL stub');
  }
}

// ---- Typed getExtension overloads for implemented extensions (interface merging) ----
export interface WebGLRenderingContext {
  getExtension(name: 'OES_texture_float'): OES_texture_float | null;
  getExtension(name: 'OES_texture_half_float'): OES_texture_half_float | null;
  getExtension(name: 'OES_texture_float_linear'): OES_texture_float_linear | null;
  getExtension(name: 'OES_texture_half_float_linear'): OES_texture_half_float_linear | null;
  getExtension(name: 'OES_element_index_uint'): OES_element_index_uint | null;
  getExtension(name: 'OES_standard_derivatives'): OES_standard_derivatives | null;
  getExtension(name: 'OES_vertex_array_object'): OES_vertex_array_object | null;
  getExtension(name: 'OES_fbo_render_mipmap'): OES_fbo_render_mipmap | null;
  getExtension(name: 'ANGLE_instanced_arrays'): ANGLE_instanced_arrays | null;
  getExtension(name: 'EXT_blend_minmax'): EXT_blend_minmax | null;
  getExtension(name: 'EXT_frag_depth'): EXT_frag_depth | null;
  getExtension(name: 'EXT_sRGB'): EXT_sRGB | null;
  getExtension(name: 'EXT_shader_texture_lod'): EXT_shader_texture_lod | null;
  getExtension(name: 'EXT_texture_filter_anisotropic'): EXT_texture_filter_anisotropic | null;
  getExtension(name: 'WEBGL_depth_texture'): WEBGL_depth_texture | null;
  getExtension(name: 'WEBGL_draw_buffers'): WEBGL_draw_buffers | null;
  getExtension(name: 'WEBGL_blend_func_extended'): WEBGL_blend_func_extended | null;
  getExtension(name: 'WEBGL_lose_context'): WEBGL_lose_context | null;
  getExtension(name: 'WEBGL_debug_renderer_info'): WEBGL_debug_renderer_info | null;
  getExtension(name: 'WEBGL_debug_shaders'): WEBGL_debug_shaders | null;
  getExtension(name: 'EXT_clip_control'): EXT_clip_control | null;
  getExtension(name: 'EXT_color_buffer_half_float'): EXT_color_buffer_half_float | null;
  getExtension(name: 'EXT_float_blend'): EXT_float_blend | null;
  getExtension(name: 'KHR_parallel_shader_compile'): KHR_parallel_shader_compile | null;
  getExtension(name: 'WEBGL_multi_draw'): WEBGL_multi_draw | null;
}

// Install WebGL1 constants on the prototype (gl.COLOR_BUFFER_BIT etc.).
installConstants(WebGLRenderingContext.prototype, C1);
// Wire the api/ prototype mixins (idempotent — entry.ts also calls installAll).
installAll(WebGLRenderingContext.prototype);

// Browser instanceof compatibility: re-chain this prototype under the NATIVE
// WebGLRenderingContext prototype so `gl instanceof WebGLRenderingContext` (the
// native global) is true — the CTS harness checks it (webgl-test-utils.js
// isWebGLContext). Prototype-only: the constructor chain (super resolution) is
// untouched; no-op in Node where the native class is absent.
chainToNative(WebGLRenderingContext.prototype, 'WebGLRenderingContext');
