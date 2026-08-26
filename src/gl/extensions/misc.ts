/**
 * src/gl/extensions/misc.ts — WEBGL_lose_context, WEBGL_debug_shaders,
 * EXT_clip_control, WEBGL_clip_cull_distance, WEBGL_multi_draw.
 *
 *  - WEBGL_lose_context: implements the spec semantics inline. NOTE: a static
 *    import of '../lifecycle' here creates a module cycle
 *    (webgl1 → api → extensions → misc → lifecycle → webgl2 → webgl1) that
 *    breaks evaluation (TDZ on WebGLRenderingContext); dynamic import would
 *    break the esbuild IIFE bundle. So loseContext/restoreContext set the
 *    context flag + invalidate resources here; when the lifecycle agent lands,
 *    it should either route its lose/restore engine through these helpers or
 *    switch this file to lifecycle.ts after the api-mixin wiring is reworked.
 *  - WEBGL_debug_shaders: getTranslatedShaderSource returns the shader's
 *    translated-source cache (set at compile time by the programs agent) — the
 *    CTS test requires '' before compilation, the source after, and the cached
 *    value to survive shaderSource() without recompiling. null/non-shader args
 *    throw TypeError (WebIDL non-nullable interface), cross-context/deleted →
 *    INVALID_OPERATION + null.
 *  - EXT_clip_control: clipControlEXT(origin, depth) validates its two GLenums
 *    and stores them in per-context state (clip-state.ts — consumed by getters
 *    and later by the rasterizer's NDC transform).
 *  - WEBGL_clip_cull_distance: constants only; the enable/disable/isEnabled/
 *    getParameter integration lives in api/state.ts + getters.ts (patched),
 *    reading/writing clip-state.ts.
 *  - WEBGL_multi_draw: the four multi-draw methods. The canonical
 *    implementations are installed on the context prototypes by the draw agent
 *    (api/draw.ts); this factory's copies delegate to them (try/catch → the
 *    Phase-1 stub throw degrades to INVALID_OPERATION instead of crashing).
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1, CExt } from '../constants';
import { WebGLShader } from '../objects';
import { setClipControl } from './clip-state';
import { buildExtension, isLost } from './util';

// ---------------------------------------------------------------------------
// WEBGL_lose_context
// ---------------------------------------------------------------------------

/** WEBGL_lose_context factory (versions [1,2] per registry). */
export function createWEBGLLoseContext(ctx: WebGLRenderingContext): object {
  return buildExtension({}, {
    loseContext: (): void => {
      const gl = ctx;
      // Spec: context lost → all resources invalidated, API calls no-ops.
      gl._isLost = true;
      gl._resources.invalidateAll();
    },

    restoreContext: (): void => {
      const gl = ctx;
      // Spec: restore → state reset, error queue cleared (minimal engine; the
      // lifecycle agent's restoreContext owns the full reset + events).
      gl._isLost = false;
      gl._errors.clear();
    },
  });
}

// ---------------------------------------------------------------------------
// WEBGL_debug_shaders
// ---------------------------------------------------------------------------

/** Strict shader argument check (null throws — WebIDL non-nullable interface). */
function requireShader(gl: WebGLRenderingContext, shader: unknown): WebGLShader | null {
  if (shader === null || shader === undefined) {
    throw new TypeError("Argument is not of type 'WebGLShader'");
  }
  if (!(shader instanceof WebGLShader)) {
    throw new TypeError("Argument is not of type 'WebGLShader'");
  }
  if (shader._context !== gl) {
    gl._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (shader._deleted) {
    gl._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  return shader;
}

/** WEBGL_debug_shaders factory. */
export function createWEBGLDebugShaders(ctx: WebGLRenderingContext): object {
  return buildExtension({}, {
    getTranslatedShaderSource: (shader: WebGLShader): string | null => {
      const gl = ctx;
      const s = requireShader(gl, shader);
      if (s === null) return null;
      // The translated source is the source at last compile (programs agent
      // fills _translatedSource at compileShader). '' before any compilation;
      // fall back to _source only once a compile actually succeeded.
      return s._translatedSource ?? (s._compileStatus ? s._source : '');
    },
  });
}

// ---------------------------------------------------------------------------
// EXT_clip_control
// ---------------------------------------------------------------------------

const LOWER_LEFT_EXT = 0x8ca1;
const UPPER_LEFT_EXT = 0x8ca2;
const NEGATIVE_ONE_TO_ONE_EXT = 0x935e;
const ZERO_TO_ONE_EXT = 0x935f;

/** EXT_clip_control factory (versions [1,2] per registry). */
export function createEXTClipControl(ctx: WebGLRenderingContext): object {
  return buildExtension(
    {
      CLIP_ORIGIN_EXT: CExt.CLIP_ORIGIN_EXT,
      CLIP_DEPTH_MODE_EXT: CExt.CLIP_DEPTH_MODE_EXT,
      LOWER_LEFT_EXT: CExt.LOWER_LEFT_EXT,
      UPPER_LEFT_EXT: CExt.UPPER_LEFT_EXT,
      NEGATIVE_ONE_TO_ONE_EXT: CExt.NEGATIVE_ONE_TO_ONE_EXT,
      ZERO_TO_ONE_EXT: CExt.ZERO_TO_ONE_EXT,
    },
    {
      clipControlEXT: (origin: number, depth: number): void => {
        const gl = ctx;
        if (isLost(gl)) return;
        if (
          (origin !== LOWER_LEFT_EXT && origin !== UPPER_LEFT_EXT) ||
          (depth !== NEGATIVE_ONE_TO_ONE_EXT && depth !== ZERO_TO_ONE_EXT)
        ) {
          gl._errors.push(C1.INVALID_ENUM);
          return;
        }
        setClipControl(gl, origin, depth);
      },
    },
  );
}

// ---------------------------------------------------------------------------
// WEBGL_clip_cull_distance
// ---------------------------------------------------------------------------

/** WEBGL_clip_cull_distance factory (WebGL2 — registry versions: [2]). */
export function createWEBGLClipCullDistance(ctx: WebGLRenderingContext): object {
  void ctx;
  return buildExtension({
    MAX_CLIP_DISTANCES_WEBGL: CExt.MAX_CLIP_DISTANCES_WEBGL,
    MAX_CULL_DISTANCES_WEBGL: CExt.MAX_CULL_DISTANCES_WEBGL,
    MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL: CExt.MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL,
    CLIP_DISTANCE0_WEBGL: CExt.CLIP_DISTANCE0_WEBGL,
    CLIP_DISTANCE1_WEBGL: CExt.CLIP_DISTANCE1_WEBGL,
    CLIP_DISTANCE2_WEBGL: CExt.CLIP_DISTANCE2_WEBGL,
    CLIP_DISTANCE3_WEBGL: CExt.CLIP_DISTANCE3_WEBGL,
    CLIP_DISTANCE4_WEBGL: CExt.CLIP_DISTANCE4_WEBGL,
    CLIP_DISTANCE5_WEBGL: CExt.CLIP_DISTANCE5_WEBGL,
    CLIP_DISTANCE6_WEBGL: CExt.CLIP_DISTANCE6_WEBGL,
    CLIP_DISTANCE7_WEBGL: CExt.CLIP_DISTANCE7_WEBGL,
  });
}

// ---------------------------------------------------------------------------
// WEBGL_multi_draw
// ---------------------------------------------------------------------------

/** Invoke an installed prototype multi-draw method (stub-safe). */
function callInstalled(ctx: WebGLRenderingContext, name: string, args: unknown[]): void {
  const fn = (ctx as unknown as Record<string, unknown>)[name];
  if (typeof fn === 'function') {
    try {
      (fn as (...a: unknown[]) => void).apply(ctx, args);
    } catch {
      // Phase-1 stub throw (api/draw.ts parallel agent) — spec-shaped error.
      ctx._errors.push(C1.INVALID_OPERATION);
    }
  } else {
    // Method not installed yet — same degraded behavior.
    ctx._errors.push(C1.INVALID_OPERATION);
  }
}

/** WEBGL_multi_draw factory — methods delegate to the prototype implementations. */
export function createWEBGLMultiDraw(ctx: WebGLRenderingContext): object {
  return buildExtension({}, {
    multiDrawArraysWEBGL: (mode: number, firsts: unknown, counts: unknown, drawcount: number): void => {
      if (isLost(ctx)) return;
      callInstalled(ctx, 'multiDrawArraysWEBGL', [mode, firsts, counts, drawcount]);
    },
    multiDrawElementsWEBGL: (mode: number, counts: unknown, type: number, offsets: unknown, drawcount: number): void => {
      if (isLost(ctx)) return;
      callInstalled(ctx, 'multiDrawElementsWEBGL', [mode, counts, type, offsets, drawcount]);
    },
    multiDrawArraysInstancedWEBGL: (mode: number, firsts: unknown, counts: unknown, instanceCounts: unknown, drawcount: number): void => {
      if (isLost(ctx)) return;
      callInstalled(ctx, 'multiDrawArraysInstancedWEBGL', [mode, firsts, counts, instanceCounts, drawcount]);
    },
    multiDrawElementsInstancedWEBGL: (mode: number, counts: unknown, type: number, offsets: unknown, instanceCounts: unknown, drawcount: number): void => {
      if (isLost(ctx)) return;
      callInstalled(ctx, 'multiDrawElementsInstancedWEBGL', [mode, counts, type, offsets, instanceCounts, drawcount]);
    },
  });
}
