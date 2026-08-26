/**
 * src/gl/api/context.ts — context attributes, extensions, errors, strings, getParameter.
 *
 * Owns: getContextAttributes, isContextLost, getSupportedExtensions, getExtension,
 * getError, getString, getParameter.
 *
 * Behavior notes:
 *  - getContextAttributes returns the RESOLVED attributes (requested merged with
 *    defaults). `antialias` reports the honest value: the default framebuffer is
 *    NOT multisampled → antialias is reported as requested but the drawing buffer
 *    is single-sampled (documented decision — see CONTEXT.md Design Decisions;
 *    verify CTS context/context-attributes.html expectations in Phase 2).
 *  - getError drains ErrorQueue (errors.ts). All API methods push errors via
 *    ctx._errors; internal exceptions NEVER propagate (catch at this boundary).
 *  - getString: VERSION 'WebGL 1.0 (Software Renderer)' / 'WebGL 2.0 (Software
 *    Renderer)', SHADING_LANGUAGE_VERSION 'WebGL GLSL ES 1.00 (Software)' /
 *    'WebGL GLSL ES 3.00 (Software)', VENDOR/RENDERER fixed strings, EXTENSIONS
 *    → space-separated getSupportedExtensions() names.
 *  - isContextLost: true only after loseContext() (lifecycle.ts).
 *  - getExtension/getSupportedExtensions delegate to extensions/index.ts
 *    (version-aware registry; singleton cache on ctx._extensions).
 */

import type { WebGLRenderingContext } from '../webgl1';

export function installContextApi(proto: WebGLRenderingContext): void {
  // Phase 2: assign implementations, e.g.:
  // proto.getContextAttributes = function (this: WebGLRenderingContext) { ... };
  void proto;
}
