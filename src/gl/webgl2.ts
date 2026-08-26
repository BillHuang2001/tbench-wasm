/**
 * src/gl/webgl2.ts — class WebGL2RenderingContext: the WebGL 2.0 API additions.
 *
 * Extends WebGLRenderingContext — WebGL2 contexts inherit the full WebGL1
 * surface through the prototype chain (instanceof WebGLRenderingContext is true).
 * This file declares ONLY the WebGL2 additions, with exact spec signatures
 * (arity matters — CTS checks function.length in places). Bodies are Phase-1
 * stubs; api/ modules implement them in Phase 2.
 *
 * Constructor sets `_version = 2` and rebuilds the state with WebGL2 limits
 * (state.ts createDefaultState(2)) — WebGL2-only state (UBO bindings, queries,
 * TF, pack/unpack buffers, drawBuffers, ...) must exist from the start.
 */

import type {
  BufferDataSource,
  Float32List,
  GLbitfield,
  GLboolean,
  GLenum,
  GLfloat,
  GLint,
  GLintptr,
  GLsizei,
  GLsizeiptr,
  GLuint,
  GLuint64,
  Int32List,
  TexImageSource,
  Uint32List,
} from './types';
import { C2, installConstants } from './constants';
import { createDefaultState } from './state';
import { WebGLRenderingContext, CONTEXT_TOKEN } from './webgl1';
import { installAll } from './api';
import type {
  WebGLActiveInfo,
  WebGLBuffer,
  WebGLProgram,
  WebGLQuery,
  WebGLSampler,
  WebGLSync,
  WebGLTexture,
  WebGLTransformFeedback,
  WebGLUniformLocation,
  WebGLVertexArrayObject,
} from './objects';
import type {
  EXT_color_buffer_float,
  EXT_texture_norm16,
  OES_draw_buffers_indexed,
  WEBGL_clip_cull_distance,
  WEBGL_multisampled_render_to_texture,
  WEBGL_render_shared_exponent,
} from './extensions/types';

export class WebGL2RenderingContext extends WebGLRenderingContext {
  /**
   * @internal — only lifecycle.ts may construct (via CONTEXT_TOKEN).
   */
  constructor(
    canvas: ConstructorParameters<typeof WebGLRenderingContext>[0],
    attrs: ConstructorParameters<typeof WebGLRenderingContext>[1],
    type: ConstructorParameters<typeof WebGLRenderingContext>[2],
    token: symbol,
  ) {
    super(canvas, attrs, type, token);
    if (token !== CONTEXT_TOKEN) return; // super already threw
    this._version = 2;
    this._state = createDefaultState(2);
    // WebGL1 constructor normalized antialias to false (single-sampled buffer);
    // WebGL2 must report the REQUESTED value exactly (CTS conformance2
    // context-attributes-depth-stencil-antialias-obeyed.html).
    if (attrs && typeof attrs.antialias === 'boolean') this._attrs.antialias = attrs.antialias;
    // Drawing-buffer resources are initialized by lifecycle.createContext via
    // initContextResources() AFTER this constructor returns, so the version-2
    // state (DEPTH_COMPONENT24 depth format, viewport sizing) is used.
  }

  // ---- Queries ----
  beginQuery(target: GLenum, query: WebGLQuery): void { throw new Error('GL stub'); }
  endQuery(target: GLenum): void { throw new Error('GL stub'); }
  createQuery(): WebGLQuery | null { throw new Error('GL stub'); }
  deleteQuery(query: WebGLQuery | null): void { throw new Error('GL stub'); }
  isQuery(query: WebGLQuery | null): GLboolean { throw new Error('GL stub'); }
  getQuery(target: GLenum, pname: GLenum): WebGLQuery | null { throw new Error('GL stub'); }
  getQueryParameter(query: WebGLQuery, pname: GLenum): any { throw new Error('GL stub'); }

  // ---- Transform feedback ----
  beginTransformFeedback(primitiveMode: GLenum): void { throw new Error('GL stub'); }
  endTransformFeedback(): void { throw new Error('GL stub'); }
  pauseTransformFeedback(): void { throw new Error('GL stub'); }
  resumeTransformFeedback(): void { throw new Error('GL stub'); }
  createTransformFeedback(): WebGLTransformFeedback | null { throw new Error('GL stub'); }
  deleteTransformFeedback(transformFeedback: WebGLTransformFeedback | null): void { throw new Error('GL stub'); }
  isTransformFeedback(transformFeedback: WebGLTransformFeedback | null): GLboolean { throw new Error('GL stub'); }
  bindTransformFeedback(target: GLenum, transformFeedback: WebGLTransformFeedback | null): void { throw new Error('GL stub'); }
  transformFeedbackVaryings(program: WebGLProgram, varyings: string[], bufferMode: GLenum): void { throw new Error('GL stub'); }
  getTransformFeedbackVarying(program: WebGLProgram, index: GLuint): WebGLActiveInfo | null { throw new Error('GL stub'); }

  // ---- Samplers ----
  createSampler(): WebGLSampler | null { throw new Error('GL stub'); }
  deleteSampler(sampler: WebGLSampler | null): void { throw new Error('GL stub'); }
  isSampler(sampler: WebGLSampler | null): GLboolean { throw new Error('GL stub'); }
  bindSampler(unit: GLuint, sampler: WebGLSampler | null): void { throw new Error('GL stub'); }
  samplerParameterf(sampler: WebGLSampler, pname: GLenum, param: GLfloat): void { throw new Error('GL stub'); }
  samplerParameteri(sampler: WebGLSampler, pname: GLenum, param: GLint): void { throw new Error('GL stub'); }
  getSamplerParameter(sampler: WebGLSampler, pname: GLenum): any { throw new Error('GL stub'); }

  // ---- VAOs ----
  createVertexArray(): WebGLVertexArrayObject | null { throw new Error('GL stub'); }
  deleteVertexArray(vertexArray: WebGLVertexArrayObject | null): void { throw new Error('GL stub'); }
  isVertexArray(vertexArray: WebGLVertexArrayObject | null): GLboolean { throw new Error('GL stub'); }
  bindVertexArray(array: WebGLVertexArrayObject | null): void { throw new Error('GL stub'); }

  // ---- Sync objects ----
  fenceSync(condition: GLenum, flags: GLbitfield): WebGLSync | null { throw new Error('GL stub'); }
  isSync(sync: WebGLSync | null): GLboolean { throw new Error('GL stub'); }
  deleteSync(sync: WebGLSync | null): void { throw new Error('GL stub'); }
  clientWaitSync(sync: WebGLSync, flags: GLbitfield, timeout: GLuint64): GLenum { throw new Error('GL stub'); }
  waitSync(sync: WebGLSync, flags: GLbitfield, timeout: GLuint64): void { throw new Error('GL stub'); }
  getSyncParameter(sync: WebGLSync, pname: GLenum): any { throw new Error('GL stub'); }

  // ---- Uniform buffer objects ----
  bindBufferBase(target: GLenum, index: GLuint, buffer: WebGLBuffer | null): void { throw new Error('GL stub'); }
  bindBufferRange(target: GLenum, index: GLuint, buffer: WebGLBuffer | null, offset: GLintptr, size: GLsizeiptr): void { throw new Error('GL stub'); }
  getIndexedParameter(target: GLenum, index: GLuint): any { throw new Error('GL stub'); }
  getUniformBlockIndex(program: WebGLProgram, uniformBlockName: string): GLuint { throw new Error('GL stub'); }
  getActiveUniformBlockParameter(program: WebGLProgram, uniformBlockIndex: GLuint, pname: GLenum): any { throw new Error('GL stub'); }
  getActiveUniformBlockName(program: WebGLProgram, uniformBlockIndex: GLuint): string | null { throw new Error('GL stub'); }
  uniformBlockBinding(program: WebGLProgram, uniformBlockIndex: GLuint, uniformBlockBinding: GLuint): void { throw new Error('GL stub'); }
  getUniformIndices(program: WebGLProgram, uniformNames: string[]): GLuint[] | null { throw new Error('GL stub'); }
  getActiveUniformsiv(program: WebGLProgram, uniformIndices: GLuint[], pname: GLenum): GLint[] | null { throw new Error('GL stub'); }

  // ---- Buffer data (WebGL2 additions) ----
  getBufferSubData(target: GLenum, srcByteOffset: GLintptr, dstBuffer: ArrayBufferView, dstOffset?: GLuint, length?: GLuint): void { throw new Error('GL stub'); }

  // ---- Framebuffer operations (WebGL2) ----
  blitFramebuffer(srcX0: GLint, srcY0: GLint, srcX1: GLint, srcY1: GLint, dstX0: GLint, dstY0: GLint, dstX1: GLint, dstY1: GLint, mask: GLbitfield, filter: GLenum): void { throw new Error('GL stub'); }
  framebufferTextureLayer(target: GLenum, attachment: GLenum, texture: WebGLTexture | null, level: GLint, layer: GLint): void { throw new Error('GL stub'); }
  invalidateFramebuffer(target: GLenum, attachments: GLenum[]): void { throw new Error('GL stub'); }
  invalidateSubFramebuffer(target: GLenum, attachments: GLenum[], x: GLint, y: GLint, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }
  readBuffer(src: GLenum): void { throw new Error('GL stub'); }
  drawBuffers(buffers: GLenum[]): void { throw new Error('GL stub'); }
  renderbufferStorageMultisample(target: GLenum, samples: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }
  getInternalformatParameter(target: GLenum, internalformat: GLenum, pname: GLenum): any { throw new Error('GL stub'); }

  // ---- Instanced / ranged draws ----
  drawArraysInstanced(mode: GLenum, first: GLint, count: GLsizei, instanceCount: GLsizei): void { throw new Error('GL stub'); }
  drawElementsInstanced(mode: GLenum, count: GLsizei, type: GLenum, offset: GLintptr, instanceCount: GLsizei): void { throw new Error('GL stub'); }
  drawRangeElements(mode: GLenum, start: GLuint, end: GLuint, count: GLsizei, type: GLenum, offset: GLintptr): void { throw new Error('GL stub'); }
  vertexAttribDivisor(index: GLuint, divisor: GLuint): void { throw new Error('GL stub'); }

  // ---- Integer vertex attribs (WebGL2) ----
  vertexAttribIPointer(index: GLuint, size: GLint, type: GLenum, stride: GLsizei, offset: GLintptr): void { throw new Error('GL stub'); }
  vertexAttribI4i(index: GLuint, x: GLint, y: GLint, z: GLint, w: GLint): void { throw new Error('GL stub'); }
  vertexAttribI4iv(index: GLuint, values: Int32List): void { throw new Error('GL stub'); }
  vertexAttribI4ui(index: GLuint, x: GLuint, y: GLuint, z: GLuint, w: GLuint): void { throw new Error('GL stub'); }
  vertexAttribI4uiv(index: GLuint, values: Uint32List): void { throw new Error('GL stub'); }

  // ---- Integer / unsigned uniform setters ----
  uniform1ui(location: WebGLUniformLocation | null, x: GLuint): void { throw new Error('GL stub'); }
  uniform1uiv(location: WebGLUniformLocation | null, v: Uint32List): void { throw new Error('GL stub'); }
  uniform2ui(location: WebGLUniformLocation | null, x: GLuint, y: GLuint): void { throw new Error('GL stub'); }
  uniform2uiv(location: WebGLUniformLocation | null, v: Uint32List): void { throw new Error('GL stub'); }
  uniform3ui(location: WebGLUniformLocation | null, x: GLuint, y: GLuint, z: GLuint): void { throw new Error('GL stub'); }
  uniform3uiv(location: WebGLUniformLocation | null, v: Uint32List): void { throw new Error('GL stub'); }
  uniform4ui(location: WebGLUniformLocation | null, x: GLuint, y: GLuint, z: GLuint, w: GLuint): void { throw new Error('GL stub'); }
  uniform4uiv(location: WebGLUniformLocation | null, v: Uint32List): void { throw new Error('GL stub'); }
  uniformMatrix2x3fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }
  uniformMatrix2x4fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }
  uniformMatrix3x2fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }
  uniformMatrix3x4fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }
  uniformMatrix4x2fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }
  uniformMatrix4x3fv(location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List): void { throw new Error('GL stub'); }

  // ---- 3D / array / multisample textures ----
  texImage3D(target: GLenum, level: GLint, internalformat: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, border: GLint, format: GLenum, type: GLenum, pixels?: ArrayBufferView | TexImageSource | null): void { throw new Error('GL stub'); }
  texSubImage3D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, zoffset: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, format: GLenum, type: GLenum, pixels?: ArrayBufferView | TexImageSource | null): void { throw new Error('GL stub'); }
  texStorage2D(target: GLenum, levels: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }
  texStorage3D(target: GLenum, levels: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei, depth: GLsizei): void { throw new Error('GL stub'); }
  compressedTexImage3D(target: GLenum, level: GLint, internalformat: GLenum, width: GLsizei, height: GLsizei, depth: GLsizei, border: GLint, data: ArrayBufferView): void { throw new Error('GL stub'); }
  compressedTexSubImage3D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, zoffset: GLint, width: GLsizei, height: GLsizei, depth: GLsizei, format: GLenum, data: ArrayBufferView): void { throw new Error('GL stub'); }
  copyTexSubImage3D(target: GLenum, level: GLint, xoffset: GLint, yoffset: GLint, zoffset: GLint, x: GLint, y: GLint, width: GLsizei, height: GLsizei): void { throw new Error('GL stub'); }

  // ---- Misc WebGL2 ----
  getFragDataLocation(program: WebGLProgram, name: string): GLint { throw new Error('GL stub'); }
  /** Clear individual buffers (clearBuffer* family). */
  clearBufferfv(buffer: GLenum, drawbuffer: GLint, values: Float32List): void { throw new Error('GL stub'); }
  clearBufferiv(buffer: GLenum, drawbuffer: GLint, values: Int32List): void { throw new Error('GL stub'); }
  clearBufferuiv(buffer: GLenum, drawbuffer: GLint, values: Uint32List): void { throw new Error('GL stub'); }
  clearBufferfi(buffer: GLenum, drawbuffer: GLint, depth: GLfloat, stencil: GLint): void { throw new Error('GL stub'); }
}

// ---- Typed getExtension overloads for implemented WebGL2-only extensions ----
// (The untyped `getExtension(name: string): any` overload from WebGLRenderingContext
// is redeclared here so the merged overload set stays a superset of the parent's —
// TS requires the subclass overload set to accept everything the base accepts.)
export interface WebGL2RenderingContext {
  getExtension(name: string): any;
  getExtension(name: 'EXT_color_buffer_float'): EXT_color_buffer_float | null;
  getExtension(name: 'EXT_texture_norm16'): EXT_texture_norm16 | null;
  getExtension(name: 'OES_draw_buffers_indexed'): OES_draw_buffers_indexed | null;
  getExtension(name: 'WEBGL_clip_cull_distance'): WEBGL_clip_cull_distance | null;
  getExtension(name: 'WEBGL_multisampled_render_to_texture'): WEBGL_multisampled_render_to_texture | null;
  getExtension(name: 'WEBGL_render_shared_exponent'): WEBGL_render_shared_exponent | null;
}

// Install WebGL2 constants on the WebGL2 prototype (gl2.X via prototype chain).
installConstants(WebGL2RenderingContext.prototype, C2);
// Wire the api/ prototype mixins for WebGL2 (idempotent — entry.ts also calls
// installAll; WebGL1 methods arrive via the prototype chain from webgl1.ts).
installAll(WebGL2RenderingContext.prototype);
