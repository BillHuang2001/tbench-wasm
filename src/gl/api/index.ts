/**
 * src/gl/api/index.ts — prototype-mixin aggregator.
 *
 * Phase 2 implementation plan: each api/<group>.ts exports `installXxx(proto)`
 * that REPLACES the Phase-1 stub methods on the class prototype with real
 * implementations (assignment preserves arity — define with the exact same
 * parameter lists as webgl1.ts/webgl2.ts). `installAll` is called once at
 * module load from webgl1.ts/webgl2.ts bottom (after class definition).
 *
 * The class declarations in webgl1.ts/webgl2.ts remain the type contract and
 * arity source; the api/ modules are the single homes of the logic, keeping
 * every file well under ~1000 lines. Engine support modules (draw.ts,
 * teximage.ts, getters.ts, lifecycle.ts) hold the deep pipeline logic that the
 * api methods validate-then-delegate to.
 *
 * Method ownership (routing table — delegate implementation work per module):
 *  - context.ts    getContextAttributes, isContextLost, getSupportedExtensions,
 *                  getExtension, getError, getString, getParameter
 *  - state.ts      capability + simple state setters (enable/disable/blendFunc/
 *                  depthFunc/viewport/scissor/colorMask/clear*...)
 *  - buffers.ts    create/delete/is/bindBuffer, bufferData/SubData,
 *                  getBufferParameter, bindBufferBase/Range, getIndexedParameter,
 *                  getBufferSubData (WebGL2)
 *  - vertex-attrib.ts vertexAttribPointer/IPointer, enable/disableVertexAttribArray,
 *                  vertexAttrib{1,2,3,4}{f,fv,I4i,I4iv,I4ui,I4uiv}, vertexAttribDivisor,
 *                  getVertexAttrib, getVertexAttribOffset
 *  - textures.ts   create/delete/is/bindTexture, activeTexture, texParameter[f|i],
 *                  getTexParameter, pixelStorei, generateMipmap
 *  - teximage.ts   texImage2D/3D, texSubImage2D/3D, texStorage2D/3D,
 *                  copyTexImage2D, copyTexSubImage2D/3D, compressedTexImage2D/3D,
 *                  compressedTexSubImage2D/3D
 *  - programs.ts   shaders+programs lifecycle, compile/link, getShader* / getProgram*,
 *                  getActiveAttrib/Uniform, bindAttribLocation, getAttribLocation,
 *                  getUniformLocation, getUniform, validateProgram, getShaderPrecisionFormat,
 *                  uniformBlockBinding, getUniformBlockIndex, getActiveUniformBlock*,
 *                  getUniformIndices, getActiveUniformsiv, getFragDataLocation,
 *                  transformFeedbackVaryings, getTransformFeedbackVarying
 *  - uniforms.ts   all uniform{1,2,3,4}{f,i,ui}{,v} + uniformMatrix* (incl. 2x3..4x3)
 *  - framebuffers.ts create/delete/is/bindFramebuffer, framebufferRenderbuffer,
 *                  framebufferTexture2D/Layer, checkFramebufferStatus,
 *                  getFramebufferAttachmentParameter, renderbuffers,
 *                  renderbufferStorage(+Multisample), getRenderbufferParameter,
 *                  drawBuffers, readBuffer, invalidateFramebuffer/SubFramebuffer,
 *                  blitFramebuffer
 *  - draw.ts       drawArrays/Elements(+Instanced, drawRangeElements),
 *                  multiDraw*WEBGL, clear, clearBuffer*, flush, finish, readPixels
 *  - webgl2.ts     queries (begin/end/get*), sync objects, samplers, VAOs,
 *                  transform feedback bind/begin/end/pause/resume
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';
import { installContextApi } from './context';
import { installStateApi } from './state';
import { installBuffersApi } from './buffers';
import { installVertexAttribApi } from './vertex-attrib';
import { installTexturesApi } from './textures';
import { installTexImageApi } from './teximage';
import { installProgramsApi } from './programs';
import { installUniformsApi } from './uniforms';
import { installFramebuffersApi } from './framebuffers';
import { installDrawApi } from './draw';
import { installWebGL2Api } from './webgl2';

export type ContextProto = WebGLRenderingContext;

/** Install every api group onto a context prototype. Called at module load. */
export function installAll(proto: ContextProto): void {
  installContextApi(proto);
  installStateApi(proto);
  installBuffersApi(proto);
  installVertexAttribApi(proto);
  installTexturesApi(proto);
  installTexImageApi(proto);
  installProgramsApi(proto);
  installUniformsApi(proto);
  installFramebuffersApi(proto);
  installDrawApi(proto);
  // installAll is invoked with each context prototype in turn (WebGL1 then
  // WebGL2); the WebGL2-only mixin is a no-op when the prototype lacks those
  // methods (it installs onto WebGL2RenderingContext.prototype only).
  installWebGL2Api(proto as unknown as WebGL2RenderingContext);
}
