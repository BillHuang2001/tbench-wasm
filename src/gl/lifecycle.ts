/**
 * src/gl/lifecycle.ts — context creation, canvas registry, drawing-buffer lifecycle.
 *
 * OWNED BY: api/context.ts + entry.ts (entry.ts wires `window.__createSoftwareWebGLContext`
 * to createContext). This module is the ONLY place that constructs context instances
 * (CONTEXT_TOKEN gate). Construction runs `initContextResources` (from ./lost) right
 * after `new` returns — it allocates the drawing buffer, default VAO and initial
 * viewport/scissor, using the FINAL context version (WebGL2's constructor rebuilds
 * state before init runs).
 *
 * WEBGL_lose_context semantics (loseContext/restoreContext/isContextLost/
 * ensureNotLost) live in ./lost.ts — the SHARED loss engine used by both this
 * module and the WEBGL_lose_context extension factory. Lifecycle cannot own them
 * inline: the extension cannot import this module (webgl1 → api → extensions →
 * misc → lifecycle → webgl1 is an ESM cycle that breaks evaluation with a TDZ
 * ReferenceError), so the engine was extracted into the cycle-free ./lost.ts and
 * is re-exported here for backward compatibility.
 *
 * Registry semantics (WebGL spec §"Context Creation"):
 *  - One context per canvas per type-slot. 'webgl' and 'experimental-webgl' share
 *    the WebGL1 slot; 'webgl2' has its own slot. A second getContext on an
 *    occupied slot returns null; a same-slot getContext returns the EXISTING
 *    context (attrs are ignored).
 *  - Changing canvas.width/height resizes the drawing buffer and CLEARS it
 *    (color → (0,0,0,0), depth → 1.0, stencil → 0); other GL state (viewport,
 *    scissor, clear color, masks, bindings) is NOT reset on resize — verified
 *    against CTS `conformance/canvas/canvas-test.html` (viewport/clearColor/
 *    colorMask unchanged; content cleared) and `viewport-unchanged-upon-resize.html`.
 *    drawingBufferWidth/Height are LIVE (max(1, canvas.width)) — CTS
 *    `context/zero-sized-canvas.html` requires immediate tracking (and a 0-size
 *    canvas yields a 1×1 drawing buffer); `canvas/drawingbuffer-test.html`
 *    requires drawingBufferWidth === canvas.width.
 */

import type { CanvasLike, ContextType, WebGLContextAttributesInit } from './types';
import { WebGLRenderingContext, CONTEXT_TOKEN } from './webgl1';
import { WebGL2RenderingContext } from './webgl2';
import { initContextResources, installMockEventShim } from './lost';

// Re-export the shared loss/restore engine + drawing-buffer machinery (single
// home: ./lost.ts — this module only delegates). Keeps existing importers
// (`import { handleCanvasResize } from './lifecycle'`, gl/index.ts) working.
export {
  loseContext,
  restoreContext,
  isContextLost,
  ensureNotLost,
  initContextResources,
  handleCanvasResize,
} from './lost';

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
 * Re-define the context's own properties as non-enumerable. The engine's
 * internal state lives in underscore-prefixed instance fields, which class-field
 * initializers and constructor assignments create ENUMERABLE. The CTS pages
 * `conformance/context/constants-and-properties.html` and
 * `conformance2/context/constants-and-properties-2.html` enumerate the context
 * with `for (var i in gl)` and fail on ANY non-function property outside the
 * spec's constant list — internal fields must be hidden so the public constant
 * surface matches native Chromium exactly. All fields are pre-declared before
 * this runs (class body + constructor), so later reassignments (resize,
 * lose/restore) preserve the non-enumerable descriptor.
 */
function hideInternalFields(ctx: WebGLRenderingContext): void {
  for (const key of Object.keys(ctx)) {
    Object.defineProperty(ctx, key, {
      value: (ctx as unknown as Record<string, unknown>)[key],
      writable: true,
      enumerable: false,
      configurable: true,
    });
  }
}

/**
 * Create (or fetch) a context for a canvas. Mirrors `canvas.getContext(type, attrs)`:
 * returns the existing context when the slot matches, null on slot conflict.
 * attrs are only applied at creation; subsequent calls return the existing context.
 * Construction runs initContextResources() AFTER the constructor returns — the
 * final version (WebGL2 rebuilds state in its constructor) determines the depth
 * format and viewport sizing. (The constructors themselves cannot call it:
 * lifecycle → webgl2 → webgl1 → lifecycle would be an ESM cycle that breaks
 * webgl2's class definition.)
 */
export function createContext(
  canvas: CanvasLike,
  attrs: WebGLContextAttributesInit | null,
  type: ContextType,
): WebGLRenderingContext | null {
  const slot = slotFor(canvas);
  const key = slotKey(type);
  // Non-DOM canvases (Node mocks) without dispatchEvent get our event-listener
  // shim so webglcontextlost/restored can be delivered headlessly.
  installMockEventShim(canvas);
  if (key === 'webgl2') {
    if (slot.webgl2) return slot.webgl2;
    if (slot.webgl) return null; // WebGL1 context already on this canvas
    slot.webgl2 = new WebGL2RenderingContext(canvas, attrs, type, CONTEXT_TOKEN);
    initContextResources(slot.webgl2);
    hideInternalFields(slot.webgl2);
    return slot.webgl2;
  }
  if (slot.webgl) return slot.webgl;
  if (slot.webgl2) return null; // WebGL2 context already on this canvas
  slot.webgl = new WebGLRenderingContext(canvas, attrs, type, CONTEXT_TOKEN);
  initContextResources(slot.webgl);
  hideInternalFields(slot.webgl);
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
