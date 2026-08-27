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
 *    CONTEXT_LOST_WEBGL the first time, then drains the queue (errors generated
 *    while lost — e.g. INVALID_OPERATION from loseContext() on an already-lost
 *    context — must be observable; CTS context-lost.html asserts exactly that).
 *  - getString: VERSION 'WebGL 1.0 (Software Renderer)' / 'WebGL 2.0 (Software
 *    Renderer)', SHADING_LANGUAGE_VERSION 'WebGL GLSL ES 1.00 (Software)' /
 *    'WebGL GLSL ES 3.00 (Software)', VENDOR/RENDERER fixed strings, EXTENSIONS
 *    → space-separated getSupportedExtensions() names. Returns null while lost.
 *  - isContextLost: true only after loseContext() (lifecycle.ts).
 *  - getExtension/getSupportedExtensions return null while the context is lost
 *    (CTS context-lost.html nullTests); after restore they work again. The
 *    extension singleton cache is NOT touched by the lost path — doRestore
 *    (lost.ts) re-creates non-WEBGL_lose_context singletons on restore
 *    (context-lost-restored.html requires NEW objects without prior properties).
 *  - getParameter delegates to getters.ts (the full pname table).
 *  - Internal exceptions NEVER escape: getParameter is wrapped; unexpected
 *    engine failures report INVALID_OPERATION.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { getExtensionObject, getSupportedExtensionNames } from '../extensions';
import { getErrorWhileLost } from '../lost';
import { getParameter, stringValueForPname } from '../getters';

/** 0x0500 INVALID_ENUM / 0x0502 INVALID_OPERATION (constants.ts values). */
const INVALID_ENUM = 0x0500;
const INVALID_OPERATION = 0x0502;

export function installContextApi(proto: WebGLRenderingContext): void {
  proto.getContextAttributes = function (this: WebGLRenderingContext) {
    // Spec: getContextAttributes returns null while the context is lost (CTS
    // conformance/context/context-lost.html nullTests).
    if (this._isLost) return null;
    // Resolved attributes, copied (spec: returns a fresh dictionary each call).
    return { ...this._attrs };
  };

  proto.isContextLost = function (this: WebGLRenderingContext) {
    return this._isLost;
  };

  proto.getSupportedExtensions = function (this: WebGLRenderingContext) {
    // Spec: extension queries return null while the context is lost (CTS
    // context-lost.html nullTests); they work again after restore.
    if (this._isLost) return null;
    return getSupportedExtensionNames(this._version);
  };

  proto.getExtension = function (this: WebGLRenderingContext, name: string) {
    // WebIDL DOMString conversion.
    const n = String(name);
    // Spec: null while the context is lost (CTS context-lost.html nullTests).
    // Deliberately BEFORE getExtensionObject: the singleton cache is untouched
    // (doRestore re-creates non-WEBGL_lose_context singletons; the kept
    // WEBGL_lose_context object must keep its identity — context-lost-restored.html).
    if (this._isLost) return null;
    return getExtensionObject(this, n);
  } as WebGLRenderingContext['getExtension'];

  proto.getError = function (this: WebGLRenderingContext) {
    if (this._isLost) {
      // Spec: the first getError after context loss returns CONTEXT_LOST_WEBGL;
      // subsequent calls return errors generated while lost (e.g.
      // INVALID_OPERATION from loseContext() while already lost — CTS
      // context-lost.html) or NO_ERROR. Lost-epoch bookkeeping lives in lost.ts.
      return getErrorWhileLost(this);
    }
    return this._errors.get();
  };

  proto.getString = function (this: WebGLRenderingContext, name: number) {
    if (this._isLost) return null; // spec: getters return null while lost
    // VERSION / SHADING_LANGUAGE_VERSION / VENDOR / RENDERER share the strings
    // with getParameter (single source of truth in getters.ts).
    const str = stringValueForPname(this, name);
    if (str !== null) return str;
    switch (name) {
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
