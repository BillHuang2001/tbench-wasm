/**
 * src/gl/objects/index.ts — object module barrel + per-context resource tracking.
 *
 * `Resources` tracks every object created by a context so that:
 *  - context loss invalidates all resources (spec: resources are lost);
 *  - delete*() removes objects from tracking;
 *  - tests can enumerate leaked resources.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { chainToNative } from '../native-chain';
import { WebGLObject } from './webgl-object';
import { WebGLBuffer } from './webgl-buffer';
import { WebGLTexture } from './webgl-texture';
import { WebGLShader } from './webgl-shader';
import { WebGLProgram } from './webgl-program';
import { WebGLFramebuffer } from './webgl-framebuffer';
import { WebGLRenderbuffer } from './webgl-renderbuffer';
import { WebGLVertexArrayObject } from './webgl-vao';
import { WebGLSampler } from './webgl-sampler';
import { WebGLQuery } from './webgl-query';
import { WebGLSync } from './webgl-sync';
import { WebGLTransformFeedback } from './webgl-transform-feedback';

export { WebGLObject } from './webgl-object';
export { WebGLBuffer } from './webgl-buffer';
export { WebGLTexture, type TextureLevel } from './webgl-texture';
export { WebGLShader, type ShaderCompileResult } from './webgl-shader';
export { WebGLProgram, type ProgramModel } from './webgl-program';
export { WebGLFramebuffer, type FramebufferAttachment } from './webgl-framebuffer';
export { WebGLRenderbuffer } from './webgl-renderbuffer';
export { WebGLVertexArrayObject } from './webgl-vao';
export { WebGLSampler } from './webgl-sampler';
export { WebGLQuery } from './webgl-query';
export { WebGLSync } from './webgl-sync';
export { WebGLTransformFeedback } from './webgl-transform-feedback';
export {
  WebGLUniformLocation,
  WebGLActiveInfo,
  WebGLShaderPrecisionFormat,
  validateUniformLocation,
} from './aux';

/** All concrete object classes (for instanceof dispatch / resource tracking). */
export const OBJECT_CLASSES = [
  WebGLBuffer,
  WebGLTexture,
  WebGLShader,
  WebGLProgram,
  WebGLFramebuffer,
  WebGLRenderbuffer,
  WebGLVertexArrayObject,
  WebGLSampler,
  WebGLQuery,
  WebGLSync,
  WebGLTransformFeedback,
] as const;

/** Per-context resource tracking (created objects by class). */
export class Resources {
  readonly all: Set<WebGLObject> = new Set();

  constructor(private readonly ctx: WebGLRenderingContext) {}

  /** Register a freshly created object. */
  track<T extends WebGLObject>(obj: T): T {
    this.all.add(obj);
    return obj;
  }

  /** Forget a deleted object. */
  untrack(obj: WebGLObject): void {
    this.all.delete(obj);
  }

  /** Mark all resources deleted (context loss / restore). */
  invalidateAll(): void {
    for (const obj of this.all) obj._deleted = true;
    this.all.clear();
  }
}

/** Create a new object of class T bound to context ctx (used by create* API). */
export function createObject<T extends WebGLObject>(
  ctx: WebGLRenderingContext,
  ctor: new (context: WebGLRenderingContext) => T,
): T {
  return ctx._resources.track(new ctor(ctx));
}

// ---- Native instanceof chaining (browser only; no-op in Node) ----
// Re-chain every concrete object class prototype under the native browser class
// of the same name so e.g. `gl.createBuffer() instanceof WebGLBuffer` (the
// native global) holds — the CTS harness checks object identity via instanceof
// in places. PROTOTYPE-ONLY: never touch the constructor [[Prototype]] (native
// WebGL constructors are illegal to call — that would break super() in our
// constructors). Instances are still `new OurClass(...)`, so internal
// `instanceof OurClass` checks keep working (our prototype remains the
// instance's immediate prototype).
//
// The abstract WebGLObject base is deliberately NOT chained: it holds only
// instance fields (_context/_deleted) set in constructors, no prototype
// methods, and no engine code does `instanceof WebGLObject` at runtime.
chainToNative(WebGLBuffer.prototype, 'WebGLBuffer');
chainToNative(WebGLTexture.prototype, 'WebGLTexture');
chainToNative(WebGLProgram.prototype, 'WebGLProgram');
chainToNative(WebGLShader.prototype, 'WebGLShader');
chainToNative(WebGLFramebuffer.prototype, 'WebGLFramebuffer');
chainToNative(WebGLRenderbuffer.prototype, 'WebGLRenderbuffer');
chainToNative(WebGLVertexArrayObject.prototype, 'WebGLVertexArrayObject');
chainToNative(WebGLSampler.prototype, 'WebGLSampler');
chainToNative(WebGLQuery.prototype, 'WebGLQuery');
chainToNative(WebGLSync.prototype, 'WebGLSync');
chainToNative(WebGLTransformFeedback.prototype, 'WebGLTransformFeedback');
// aux classes (opaque handles / plain records — NOT WebGLObjects)
chainToNative(WebGLUniformLocation.prototype, 'WebGLUniformLocation');
chainToNative(WebGLActiveInfo.prototype, 'WebGLActiveInfo');
chainToNative(WebGLShaderPrecisionFormat.prototype, 'WebGLShaderPrecisionFormat');
