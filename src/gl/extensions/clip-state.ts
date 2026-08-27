/**
 * src/gl/extensions/clip-state.ts — per-context state for EXT_clip_control and
 * WEBGL_clip_cull_distance.
 *
 * These two extensions add state that does not fit the fixed `State` shape:
 *  - EXT_clip_control: clipControlEXT(origin, depth) — two GLenums with spec
 *    defaults LOWER_LEFT_EXT / NEGATIVE_ONE_TO_ONE_EXT. The rasterizer consumes
 *    this when transforming clip coordinates to NDC (Phase 2, raster agent);
 *    getParameter (getters.ts) reads it here.
 *  - WEBGL_clip_cull_distance: enable/disable of CLIP_DISTANCE0..7 — 8
 *    booleans, default all false. api/state.ts (enable/disable/isEnabled) and
 *    getters.ts (getParameter) read/write through the helpers; the draw engine
 *    consumes them when evaluating gl_ClipDistance (Phase 2, draw agent).
 *
 * Both stores are WeakMaps keyed by context (no context field pollution; the
 * state resets naturally when the context object is replaced on restore).
 */

import type { WebGLRenderingContext } from '../webgl1';

// ---------------------------------------------------------------------------
// WEBGL_clip_cull_distance
// ---------------------------------------------------------------------------

const CLIP_DISTANCE_COUNT = 8;

const clipDistanceState = new WeakMap<WebGLRenderingContext, Uint8Array>();

/** The 8 CLIP_DISTANCEi enabled flags for a context (lazily created). */
export function getClipDistances(ctx: WebGLRenderingContext): Uint8Array {
  let flags = clipDistanceState.get(ctx);
  if (!flags) {
    flags = new Uint8Array(CLIP_DISTANCE_COUNT);
    clipDistanceState.set(ctx, flags);
  }
  return flags;
}

/** True when CLIP_DISTANCEi (0 ≤ i < 8) is enabled for the context. */
export function isClipDistanceEnabled(ctx: WebGLRenderingContext, index: number): boolean {
  if (index < 0 || index >= CLIP_DISTANCE_COUNT) return false;
  return clipDistanceState.get(ctx)?.[index] === 1;
}

/** Set the enabled flag of CLIP_DISTANCEi (0 ≤ i < 8). */
export function setClipDistanceEnabled(ctx: WebGLRenderingContext, index: number, enabled: boolean): void {
  if (index < 0 || index >= CLIP_DISTANCE_COUNT) return;
  getClipDistances(ctx)[index] = enabled ? 1 : 0;
}

// ---------------------------------------------------------------------------
// EXT_clip_control
// ---------------------------------------------------------------------------

/** Defaults per EXT_clip_control spec. */
const CLIP_ORIGIN_LOWER_LEFT = 0x8ca1; // LOWER_LEFT_EXT
const CLIP_DEPTH_NEGATIVE_ONE_TO_ONE = 0x935e; // NEGATIVE_ONE_TO_ONE_EXT

const clipControlState = new WeakMap<WebGLRenderingContext, { origin: number; depth: number }>();

/** Current clip origin/depth mode (spec defaults until clipControlEXT is called). */
export function getClipControl(ctx: WebGLRenderingContext): { origin: number; depth: number } {
  let state = clipControlState.get(ctx);
  if (!state) {
    state = { origin: CLIP_ORIGIN_LOWER_LEFT, depth: CLIP_DEPTH_NEGATIVE_ONE_TO_ONE };
    clipControlState.set(ctx, state);
  }
  return state;
}

/** Store clipControlEXT(origin, depth) — validated by the factory before this. */
export function setClipControl(ctx: WebGLRenderingContext, origin: number, depth: number): void {
  clipControlState.set(ctx, { origin, depth });
}
