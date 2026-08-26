/**
 * src/entry.ts — bundle entry for the software WebGL renderer.
 *
 * Builds (scripts/build.mjs, esbuild IIFE) to `renderer.js` which defines:
 *   window.__createSoftwareWebGLContext(canvas, attrs[, type])
 *     → WebGLRenderingContext | WebGL2RenderingContext | null
 *
 * The 3rd `type` argument ('webgl' | 'webgl2' | 'experimental-webgl', optional,
 * default 'webgl') is an extension used by src/context-intercept.ts so the
 * harness can route both context types; the 2-arg form from the project
 * objective is fully supported.
 *
 * This module has NO exports (esbuild IIFE format): it wires the gl/ API
 * mixins onto the context prototypes at load and assigns the factory to
 * globalThis (+ module.exports when running under Node).
 */
import { WebGLRenderingContext } from './gl/webgl1';
import { WebGL2RenderingContext } from './gl/webgl2';
import { installAll } from './gl/api';
import { createContext } from './gl/lifecycle';

// Wire the prototype-mixin implementations once, before any context exists.
installAll(WebGLRenderingContext.prototype);
installAll(WebGL2RenderingContext.prototype);

const SWGL_VERSION = '0.1.0';

/**
 * Create (or fetch) a software WebGL context for a canvas.
 * Spec semantics (via gl/lifecycle): one context per canvas per type-slot;
 * 'webgl' and 'experimental-webgl' share the WebGL1 slot; a conflicting
 * getContext returns null.
 */
export function __createSoftwareWebGLContext(
  canvas: unknown,
  attrs?: unknown,
  type?: unknown,
): WebGLRenderingContext | null {
  const t =
    typeof type === 'string' && (type === 'webgl2' || type === 'experimental-webgl')
      ? type
      : 'webgl';
  return createContext(canvas as Parameters<typeof createContext>[0], (attrs as never) ?? null, t);
}

// ---- Global assignment (browser + workers + Node) ----
const g = typeof globalThis !== 'undefined' ? globalThis : ({} as Record<string, unknown>);
(g as Record<string, unknown>).__createSoftwareWebGLContext = __createSoftwareWebGLContext;
(g as Record<string, unknown>).__swglVersion = SWGL_VERSION;

// ---- Node / CommonJS interop (unit tests, headless Node usage) ----
if (typeof module !== 'undefined' && (module as { exports?: unknown }).exports !== undefined) {
  (module as { exports: unknown }).exports = { __createSoftwareWebGLContext, SWGL_VERSION };
}
