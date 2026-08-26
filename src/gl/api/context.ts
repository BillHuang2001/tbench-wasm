/**
 * src/gl/api/context.ts — context attributes, extensions, errors, strings, getParameter.
 *
 * Owns: getContextAttributes, isContextLost, getSupportedExtensions, getExtension,
 * getError, getString, getParameter. Installed onto the prototype via
 * `installContextApi(proto)` (prototype-mixin pattern — assignment preserves arity).
 *
 * Behavior notes:
 *  - getContextAttributes returns the RESOLVED attributes (requested merged with
 *    defaults), copied per call. `antialias` reports the honest value: the default
 *    framebuffer is NOT multisampled → antialias is reported as requested but the
 *    drawing buffer is single-sampled (documented decision — see CONTEXT.md
 *    Design Decisions; verify CTS context/context-attributes.html expectations).
 *  - getError drains ErrorQueue (errors.ts); when the context is lost it returns
 *    CONTEXT_LOST_WEBGL the first time, then NO_ERROR (spec [WebGLHandlesContextLoss]).
 *  - getString: VERSION 'WebGL 1.0 (Software Renderer)' / 'WebGL 2.0 (Software
 *    Renderer)', SHADING_LANGUAGE_VERSION 'WebGL GLSL ES 1.00 (Software)' /
 *    'WebGL GLSL ES 3.00 (Software)', VENDOR/RENDERER fixed strings, EXTENSIONS
 *    → space-separated getSupportedExtensions() names. Returns null while lost.
 *  - isContextLost: true only after loseContext() (lifecycle.ts).
 *  - getExtension/getSupportedExtensions delegate to extensions/index.ts
 *    (version-aware registry; singleton cache on ctx._extensions). Extension
 *    factory stubs (parallel wave) may throw — degraded to null until they land.
 *  - getParameter delegates to getters.ts (the full pname table).
 *  - Internal exceptions NEVER escape: getParameter is wrapped; unexpected
 *    engine failures report INVALID_OPERATION.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { getExtensionObject, getSupportedExtensionNames } from '../extensions';
import { getParameter } from '../getters';

/** 0x0500 INVALID_ENUM / 0x0502 INVALID_OPERATION (constants.ts values). */
const INVALID_ENUM = 0x0500;
const INVALID_OPERATION = 0x0502;
const NO_ERROR = 0x0000;
const CONTEXT_LOST_WEBGL = 0x9242;

/** Per-context getError bookkeeping for the "first call returns CONTEXT_LOST_WEBGL" rule. */
const lostErrorState = new WeakMap<WebGLRenderingContext, { lostEpoch: boolean; consumed: boolean }>();

export function installContextApi(proto: WebGLRenderingContext): void {
  proto.getContextAttributes = function (this: WebGLRenderingContext) {
    // Resolved attributes, copied (spec: returns a fresh dictionary each call).
    return { ...this._attrs };
  };

  proto.isContextLost = function (this: WebGLRenderingContext) {
    return this._isLost;
  };

  proto.getSupportedExtensions = function (this: WebGLRenderingContext) {
    // Spec: extension queries keep working while the context is lost.
    return getSupportedExtensionNames(this._version);
  };

  proto.getExtension = function (this: WebGLRenderingContext, name: string) {
    // WebIDL DOMString conversion.
    const n = String(name);
    try {
      return getExtensionObject(this, n);
    } catch {
      // Extension factories are stubs in the parallel implementation wave;
      // degrade to null instead of throwing to the page. Remove once the
      // extensions factories land (extensions/*.ts).
      return null;
    }
  } as WebGLRenderingContext['getExtension'];

  proto.getError = function (this: WebGLRenderingContext) {
    if (this._isLost) {
      // Spec: the first getError after context loss returns CONTEXT_LOST_WEBGL,
      // subsequent calls return NO_ERROR until the context is restored.
      let st = lostErrorState.get(this);
      if (!st || !st.lostEpoch) {
        st = { lostEpoch: true, consumed: false };
        lostErrorState.set(this, st);
      }
      if (!st.consumed) {
        st.consumed = true;
        return CONTEXT_LOST_WEBGL;
      }
      return NO_ERROR;
    }
    lostErrorState.delete(this);
    return this._errors.get();
  };

  proto.getString = function (this: WebGLRenderingContext, name: number) {
    if (this._isLost) return null; // spec: getters return null while lost
    switch (name) {
      case 0x1f02 /* VERSION */:
        return this._version === 2
          ? 'WebGL 2.0 (Software Renderer)'
          : 'WebGL 1.0 (Software Renderer)';
      case 0x8b8c /* SHADING_LANGUAGE_VERSION */:
        return this._version === 2
          ? 'WebGL GLSL ES 3.00 (Software)'
          : 'WebGL GLSL ES 1.00 (Software)';
      case 0x1f00 /* VENDOR */:
        return 'Software Renderer';
      case 0x1f01 /* RENDERER */:
        return 'Software Renderer (JS)';
      case 0x1f03 /* EXTENSIONS */:
        return getSupportedExtensionNames(this._version).join(' ');
      default:
        this._errors.push(INVALID_ENUM);
        return null;
    }
  };

  proto.getParameter = function (this: WebGLRenderingContext, pname: number) {
    try {
      return getParameter(this, pname);
    } catch {
      // Engine must never throw to the page — report INVALID_OPERATION.
      this._errors.push(INVALID_OPERATION);
      return null;
    }
  };
}
