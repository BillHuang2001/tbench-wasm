/**
 * src/gl/lifecycle.ts — context creation, canvas registry, drawing-buffer lifecycle,
 * WEBGL_lose_context semantics.
 *
 * OWNED BY: api/context.ts + entry.ts (entry.ts wires `window.__createSoftwareWebGLContext`
 * to createContext). This module is the ONLY place that constructs context instances.
 *
 * Registry semantics (WebGL spec §"Context Creation"):
 *  - One context per canvas per type-slot. 'webgl' and 'experimental-webgl' share
 *    the WebGL1 slot; 'webgl2' has its own slot. A second getContext on an
 *    occupied slot returns null.
 *  - Changing canvas.width/height resizes the drawing buffer and CLEARS it
 *    (contents become (0,0,0,0) + depth/stencil cleared); other GL state is NOT
 *    reset on resize (only context creation resets state). Exact expectations
 *    verified against CTS `canvas/` and `context/context-attribute-preserve-drawing-buffer.html`
 *    during Phase 2.
 *  - WEBGL_lose_context: loseContext() → isContextLost() true, all API calls
 *    become no-ops that push CONTEXT_LOST_WEBGL, and a `webglcontextlost` event
 *    fires on the canvas (not cancelable here → no auto-restore). restoreContext()
 *    re-initializes the drawing buffer + state and fires `webglcontextrestored`.
 */

import type { CanvasLike, ContextType, WebGLContextAttributesInit } from './types';
import { WebGLRenderingContext, CONTEXT_TOKEN } from './webgl1';
import { WebGL2RenderingContext } from './webgl2';

interface CanvasSlot {
  webgl: WebGLRenderingContext | null; // 'webgl' + 'experimental-webgl' share this slot
  webgl2: WebGL2RenderingContext | null;
}

// WeakMap so canvases can be GC'd (Node mocks included — any object works).
const registry = new WeakMap<object, CanvasSlot>();

function slotFor(canvas: CanvasLike): CanvasSlot {
  let slot = registry.get(canvas as object);
  if (!slot) {
    slot = { webgl: null, webgl2: null };
    registry.set(canvas as object, slot);
  }
  return slot;
}

/** Map a requested context type to its slot key. */
function slotKey(type: ContextType): 'webgl' | 'webgl2' {
  return type === 'webgl2' ? 'webgl2' : 'webgl';
}

/**
 * Create (or fetch) a context for a canvas. Mirrors `canvas.getContext(type, attrs)`:
 * returns the existing context when the slot matches, null on slot conflict.
 * attrs are only applied at creation; subsequent calls return the existing context.
 */
export function createContext(
  canvas: CanvasLike,
  attrs: WebGLContextAttributesInit | null,
  type: ContextType,
): WebGLRenderingContext | null {
  const slot = slotFor(canvas);
  const key = slotKey(type);
  if (key === 'webgl2') {
    if (slot.webgl2) return slot.webgl2;
    if (slot.webgl) return null; // WebGL1 context already on this canvas
    slot.webgl2 = new WebGL2RenderingContext(canvas, attrs, type, CONTEXT_TOKEN);
    return slot.webgl2;
  }
  if (slot.webgl) return slot.webgl;
  if (slot.webgl2) return null; // WebGL2 context already on this canvas
  slot.webgl = new WebGLRenderingContext(canvas, attrs, type, CONTEXT_TOKEN);
  return slot.webgl;
}

/** Fetch an existing context without creating (used by tests/intercept). */
export function getContext(canvas: CanvasLike, type: ContextType): WebGLRenderingContext | null {
  const slot = registry.get(canvas as object);
  if (!slot) return null;
  return slot[slotKey(type)];
}

/** Release a canvas's slot (test cleanup). */
export function releaseContext(canvas: CanvasLike, type: ContextType): void {
  const slot = registry.get(canvas as object);
  if (!slot) return;
  slot[slotKey(type)] = null;
}

/**
 * Resize the drawing buffer after canvas.width/height changed (called by the
 * present/ adapter or entry.ts on canvas resize observation). Reallocates the
 * default framebuffer surface and clears it. State is preserved.
 * @internal engine
 */
export function handleCanvasResize(ctx: WebGLRenderingContext): void {
  // Phase 2: realloc surface via raster/formats, clear, update _drawingBufferWidth/Height.
  void ctx;
}

/**
 * loseContext()/restoreContext() — WEBGL_lose_context extension implementation
 * (api/context.ts delegates here).
 * @internal engine
 */
export function loseContext(ctx: WebGLRenderingContext): void {
  ctx._isLost = true;
  // Phase 2: fire 'webglcontextlost' on canvas; mark resources invalidated.
}

export function restoreContext(ctx: WebGLRenderingContext): void {
  // Phase 2: re-create state + drawing buffer; fire 'webglcontextrestored'.
  void ctx;
}
