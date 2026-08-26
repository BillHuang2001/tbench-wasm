/**
 * src/gl/lifecycle.ts — context creation, canvas registry, drawing-buffer lifecycle,
 * WEBGL_lose_context semantics.
 *
 * OWNED BY: api/context.ts + entry.ts (entry.ts wires `window.__createSoftwareWebGLContext`
 * to createContext). This module is the ONLY place that constructs context instances
 * (CONTEXT_TOKEN gate). Construction runs `initContextResources` (also here) right
 * after `new` returns — it allocates the drawing buffer, default VAO and initial
 * viewport/scissor, using the FINAL context version (WebGL2's constructor rebuilds
 * state before init runs).
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
 *  - WEBGL_lose_context (CTS `context/context-lost.html` +
 *    `context/context-lost-restored.html`): loseContext() → isContextLost() true
 *    immediately, all resources invalidated, and a `webglcontextlost` event is
 *    QUEUED (async — the event must not fire synchronously). restoreContext() only
 *    restores when the page called preventDefault() on the lost event; otherwise
 *    it generates INVALID_OPERATION and the context is never restored. Restoration
 *    is also async: isContextLost() stays true until `webglcontextrestored` fires.
 *    While lost, API calls are no-ops WITHOUT generating errors (the tests assert
 *    NO_ERROR after each call); getError returns CONTEXT_LOST_WEBGL once (api/context.ts).
 */

import type { CanvasLike, ContextType, WebGLContextAttributesInit } from './types';
import { WebGLRenderingContext, CONTEXT_TOKEN, type DefaultFramebuffer } from './webgl1';
import { WebGL2RenderingContext } from './webgl2';
import { createDefaultState, defaultVAOState } from './state';
import { ErrorQueue } from './errors';
import { createCanvasSurface, type CanvasSurface } from '../present';
import {
  createSurface,
  getFormat,
  type PixelFormatInfo,
  type Surface,
} from '../raster';

// Internal format GLenums used by the drawing buffer (raster/gl-enums values).
import { RGBA8, DEPTH_COMPONENT16, DEPTH_COMPONENT24, STENCIL_INDEX8 } from '../raster';

const INVALID_OPERATION = 0x0502;

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
    return slot.webgl2;
  }
  if (slot.webgl) return slot.webgl;
  if (slot.webgl2) return null; // WebGL2 context already on this canvas
  slot.webgl = new WebGLRenderingContext(canvas, attrs, type, CONTEXT_TOKEN);
  initContextResources(slot.webgl);
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
 * Resolve a canvas's drawing-buffer dimensions: max(1, canvas.width/height) —
 * a 0-size canvas yields a 1×1 buffer (CTS context/zero-sized-canvas.html).
 * Returns null when the canvas has no numeric width/height (degenerate mocks).
 */
function canvasSize(ctx: WebGLRenderingContext): { w: number; h: number } | null {
  const c = ctx._canvas as { width?: unknown; height?: unknown };
  if (typeof c.width !== 'number' || typeof c.height !== 'number') return null;
  return { w: Math.max(1, c.width), h: Math.max(1, c.height) };
}

/* ================================================================== */
/* Drawing-buffer (default framebuffer) allocation                     */
/* ================================================================== */

/**
 * Initialize the per-context resources the context owns: present adapter,
 * default VAO, default framebuffer (drawing buffer), and the initial
 * viewport/scissor (drawing buffer size per spec — CTS canvas/drawingbuffer-test.html
 * checks VIEWPORT === drawingBuffer size at creation). Every allocation is
 * stub-safe: present/raster surface factories may throw (parallel implementation
 * wave) → fields degrade to null / plain typed-array surfaces.
 *
 * Called by createContext() after construction (and by restoreContext()).
 */
export function initContextResources(ctx: WebGLRenderingContext): void {
  // Present adapter (browser 2D blit / Node buffer). Stub-safe: null on throw.
  try {
    ctx._presentSurface = createCanvasSurface(ctx._canvas);
  } catch {
    ctx._presentSurface = null;
  }
  // The persistent default VAO (pure state function — cannot throw).
  ctx._defaultVAO = defaultVAOState(ctx._state.limits.MAX_VERTEX_ATTRIBS);
  // Default framebuffer (drawing buffer) — null on stub throw (callers guard).
  const w = Math.max(1, ctx._drawingBufferWidth);
  const h = Math.max(1, ctx._drawingBufferHeight);
  ctx._defaultFB = allocateDefaultFramebuffer(ctx, w, h);
  // Initial viewport + scissor = drawing buffer size (WebGL spec initial state).
  ctx._state.viewport = { x: 0, y: 0, w, h };
  ctx._state.scissor = { x: 0, y: 0, w, h };
}

/**
 * Allocate the default framebuffer: color wraps the present surface's pixel
 * buffer ZERO-COPY (RGBA8); depth/stencil per context attributes (depth only,
 * stencil only, or both — independent per spec; CTS
 * context/context-attributes-alpha-depth-stencil-antialias.html exercises all
 * 16 combos). Depth format: DEPTH_COMPONENT16 on WebGL1, DEPTH_COMPONENT24 on
 * WebGL2 (reported via DEPTH_BITS). Returns null when surface factories throw.
 */
function allocateDefaultFramebuffer(
  ctx: WebGLRenderingContext,
  w: number,
  h: number,
): DefaultFramebuffer | null {
  const width = Math.max(1, w);
  const height = Math.max(1, h);
  try {
    const color = makeColorSurface(ctx, width, height);
    const depth = ctx._attrs.depth ? makeDepthSurface(ctx, width, height) : null;
    const stencil = ctx._attrs.stencil ? makeStencilSurface(width, height) : null;
    return { color, depth, stencil, width, height };
  } catch {
    return null; // surface stubs / unexpected failures — callers null-guard
  }
}

/** Color surface: zero-copy wrap of the present surface's RGBA8 buffer. */
function makeColorSurface(ctx: WebGLRenderingContext, w: number, h: number): Surface {
  let pixels: Uint8Array | null = null;
  if (ctx._presentSurface) {
    try {
      pixels = ctx._presentSurface.getPixels();
    } catch {
      pixels = null; // present stub — allocate locally
    }
  }
  if (!pixels || pixels.length < w * h * 4) pixels = new Uint8Array(w * h * 4);
  return makeSurface(RGBA8, w, h, pixels);
}

function makeDepthSurface(ctx: WebGLRenderingContext, w: number, h: number): Surface {
  const format = ctx._version === 2 ? DEPTH_COMPONENT24 : DEPTH_COMPONENT16;
  try {
    return createSurface(format, w, h);
  } catch {
    // Raster surface stub / unknown format: local fallback. Depth storage is a
    // Float32Array of 0..1 depths per the MANDATORY raster representation rules
    // (src/raster/types.ts) — DEPTH_COMPONENT16 vs 24 affects only reported
    // DEPTH_BITS and readPixels conversion, not the storage class.
    return makeSurface(format, w, h, new Float32Array(w * h));
  }
}

function makeStencilSurface(w: number, h: number): Surface {
  try {
    return createSurface(STENCIL_INDEX8, w, h);
  } catch {
    // Local fallback: Uint8Array (raster representation rules).
    return makeSurface(STENCIL_INDEX8, w, h, new Uint8Array(w * h));
  }
}

/** Build a Surface, resolving the format descriptor (fallback while formats.ts is a stub). */
function makeSurface(format: number, w: number, h: number, data: ArrayBufferView): Surface {
  const info = getFormat(format) ?? fallbackFormatInfo(format);
  return { width: w, height: h, format, info, data };
}

/**
 * Minimal format descriptors for the drawing-buffer formats, used only until the
 * raster formats registry lands (getFormat returns null). Correct for the common
 * clear/readPixels paths so the default framebuffer works during the transition.
 */
function fallbackFormatInfo(format: number): PixelFormatInfo {
  if (format === DEPTH_COMPONENT16 || format === DEPTH_COMPONENT24) {
    return {
      format,
      components: 1,
      bytesPerPixel: 4,
      storage: 'f32',
      isColor: false,
      isDepth: true,
      isStencil: false,
      isFloat: true,
      isSigned: false,
      isInteger: false,
      isSRGB: false,
      normalized: false,
      decode(src, byteOffset, out) {
        const d = (src as Float32Array)[byteOffset >> 2];
        out[0] = d;
        out[1] = d;
        out[2] = d;
        out[3] = 1;
      },
      encode(src, byteOffset, r) {
        (src as Float32Array)[byteOffset >> 2] = r;
      },
    };
  }
  if (format === STENCIL_INDEX8) {
    return {
      format,
      components: 1,
      bytesPerPixel: 1,
      storage: 'u8',
      isColor: false,
      isDepth: false,
      isStencil: true,
      isFloat: false,
      isSigned: false,
      isInteger: false,
      isSRGB: false,
      normalized: false,
      decode(src, byteOffset, out) {
        const s = (src as Uint8Array)[byteOffset];
        out[0] = s;
        out[1] = s;
        out[2] = s;
        out[3] = 1;
      },
      encode(src, byteOffset, r) {
        (src as Uint8Array)[byteOffset] = r & 0xff;
      },
    };
  }
  // RGBA8 (color)
  return {
    format,
    components: 4,
    bytesPerPixel: 4,
    storage: 'u8',
    isColor: true,
    isDepth: false,
    isStencil: false,
    isFloat: false,
    isSigned: false,
    isInteger: false,
    isSRGB: false,
    normalized: true,
    decode(src, byteOffset, out) {
      const u = src as Uint8Array;
      out[0] = u[byteOffset] / 255;
      out[1] = u[byteOffset + 1] / 255;
      out[2] = u[byteOffset + 2] / 255;
      out[3] = u[byteOffset + 3] / 255;
    },
    encode(src, byteOffset, r, g, b, a) {
      const u = src as Uint8Array;
      u[byteOffset] = clamp255(r);
      u[byteOffset + 1] = clamp255(g);
      u[byteOffset + 2] = clamp255(b);
      u[byteOffset + 3] = clamp255(a);
    },
  };
}

function clamp255(v: number): number {
  const x = Math.round(v * 255);
  return x < 0 ? 0 : x > 255 ? 255 : x;
}

/** Contents of a freshly resized drawing buffer are undefined → clear per spec. */
function clearDefaultFramebuffer(fb: DefaultFramebuffer): void {
  (fb.color.data as Uint8Array).fill(0);
  if (fb.depth) (fb.depth.data as Float32Array).fill(1.0);
  if (fb.stencil) (fb.stencil.data as Uint8Array).fill(0);
}

/**
 * Resize the drawing buffer after canvas.width/height changed (called by the
 * present/ adapter or entry.ts on canvas resize observation). Reallocates the
 * default framebuffer surfaces and CLEARS them (color → 0, depth → 1.0,
 * stencil → 0 — CTS canvas/canvas-test.html checks the canvas content is
 * (0,0,0,0) after resize). State (viewport, scissor, clear color, masks,
 * bindings) is PRESERVED — CTS canvas-test.html + viewport-unchanged-upon-resize.html.
 * No-op while the context is lost (restore re-sizes) or when size is unchanged.
 * @internal engine
 */
export function handleCanvasResize(ctx: WebGLRenderingContext): void {
  if (ctx._isLost) return; // loss freezes the drawing buffer; restore re-sizes
  const size = canvasSize(ctx);
  if (!size) return;
  const { w, h } = size;
  const surf = ctx._presentSurface;
  const curW = surf ? surf.width : ctx._defaultFB ? ctx._defaultFB.width : 0;
  const curH = surf ? surf.height : ctx._defaultFB ? ctx._defaultFB.height : 0;
  if (w === curW && h === curH) return; // unchanged (incl. same-value re-set)
  if (surf) {
    try {
      surf.resize(w, h);
    } catch {
      /* present stub — local reallocation below still works */
    }
  }
  const fb = allocateDefaultFramebuffer(ctx, w, h);
  if (fb) {
    clearDefaultFramebuffer(fb);
    ctx._defaultFB = fb;
    if (surf) {
      try {
        surf.present(); // show the cleared frame (canvas-test.html composites it)
      } catch {
        /* stub */
      }
    }
  }
  ctx._drawingBufferWidth = w;
  ctx._drawingBufferHeight = h;
}

/* ================================================================== */
/* WEBGL_lose_context: loseContext() / restoreContext()                */
/* ================================================================== */

interface LostState {
  /** Set when the page calls preventDefault() on the webglcontextlost event. */
  restoreAllowed: boolean;
  /** restoreContext() was called; a restore task is queued (idempotent). */
  restorePending: boolean;
}

/** Per-context loss bookkeeping (kept off the class — lifecycle-owned). */
const lostState = new WeakMap<WebGLRenderingContext, LostState>();

/** True while the context is lost (api/context.ts isContextLost reads _isLost). */
export function isContextLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

/**
 * Guard for API entry points: returns false while the context is lost — callers
 * no-op. IMPORTANT: no error is pushed — CTS context-lost.html asserts NO_ERROR
 * after every void call while lost (the spec's CONTEXT_LOST_WEBGL is delivered
 * once via getError, handled in api/context.ts).
 */
export function ensureNotLost(ctx: WebGLRenderingContext): boolean {
  return !ctx._isLost;
}

/**
 * loseContext() — WEBGL_lose_context implementation (the extension factory
 * delegates here; also usable by the harness for context-loss tests).
 * Spec + CTS context-lost.html / context-lost-restored.html:
 *  - already lost → INVALID_OPERATION, no-op;
 *  - isContextLost() is true IMMEDIATELY;
 *  - all resources are invalidated (objects become "deleted" — their methods
 *    generate INVALID_OPERATION; they stay invalid after restore);
 *  - the `webglcontextlost` event is queued asynchronously (must NOT fire
 *    synchronously — the tests assert `contextLostEventFired == false` right
 *    after loseContext());
 *  - restoration is only allowed if the page preventDefault()s the event.
 */
export function loseContext(ctx: WebGLRenderingContext): void {
  if (ctx._isLost) {
    ctx._errors.push(INVALID_OPERATION); // WEBGL_lose_context: already lost
    return;
  }
  ctx._isLost = true;
  ctx._resources.invalidateAll();
  lostState.set(ctx, { restoreAllowed: false, restorePending: false });
  const canvas = ctx._canvas;
  schedule(() => {
    let prevented = false;
    const event = {
      type: 'webglcontextlost',
      statusMessage: '',
      cancelable: true,
      preventDefault(): void {
        prevented = true;
      },
    };
    dispatchCanvasEvent(canvas, event);
    const st = lostState.get(ctx);
    if (st) st.restoreAllowed = prevented;
  });
}

/**
 * restoreContext() — WEBGL_lose_context implementation. Per CTS
 * context-lost-restored.html:
 *  - not lost → INVALID_OPERATION;
 *  - lost but the webglcontextlost event was NOT preventDefault()'d →
 *    INVALID_OPERATION (restore has no effect; the context is never restored);
 *  - restore is ASYNC: isContextLost() stays true until the queued restore
 *    task re-initializes the context and fires `webglcontextrestored`
 *    (the test asserts isContextLost() right after restoreContext());
 *  - a second restoreContext() while a restore is pending is a NO_ERROR no-op.
 */
export function restoreContext(ctx: WebGLRenderingContext): void {
  const st = lostState.get(ctx);
  if (!ctx._isLost || !st) {
    ctx._errors.push(INVALID_OPERATION); // nothing to restore
    return;
  }
  if (!st.restoreAllowed) {
    ctx._errors.push(INVALID_OPERATION); // event default not prevented
    return;
  }
  if (st.restorePending) return; // idempotent (NO_ERROR)
  st.restorePending = true;
  schedule(() => doRestore(ctx));
}

/** The queued restore task: full state reset + drawing buffer re-creation. */
function doRestore(ctx: WebGLRenderingContext): void {
  // Context state is fully reset to defaults (spec).
  ctx._state = createDefaultState(ctx._version);
  ctx._errors.clear();
  // Re-create the drawing buffer, default VAO, present adapter, initial viewport.
  initContextResources(ctx);
  ctx._isLost = false;
  // Extension singletons are re-created on restore — EXCEPT WEBGL_lose_context,
  // which must be the SAME object (CTS context-lost-restored.html: only
  // WEBGL_lose_context keeps its webglTestProperty).
  for (const key of Array.from(ctx._extensions.keys())) {
    if (!/lose_context$/i.test(key)) ctx._extensions.delete(key);
  }
  lostState.delete(ctx);
  dispatchCanvasEvent(ctx._canvas, { type: 'webglcontextrestored', statusMessage: '' });
}

/* ================================================================== */
/* Async event delivery (browser task semantics for lost/restored)     */
/* ================================================================== */

/** Queue a task; degrades to synchronous when no async primitives exist. */
function schedule(fn: () => void): void {
  const g = globalThis as {
    setTimeout?: (f: () => void, ms?: number) => unknown;
    queueMicrotask?: (f: () => void) => void;
  };
  if (typeof g.setTimeout === 'function') {
    g.setTimeout(fn, 0);
    return;
  }
  if (typeof g.queueMicrotask === 'function') {
    g.queueMicrotask(fn);
    return;
  }
  fn(); // pure sandbox without timers — degrade to synchronous
}

interface CanvasEvent {
  type: string;
  statusMessage?: string;
  cancelable?: boolean;
  preventDefault?: () => void;
}

/**
 * Deliver an event to a canvas: real DOM canvases dispatchEvent (listeners were
 * registered through the DOM); non-DOM mocks get the events delivered to
 * listeners registered via our addEventListener shim (see installMockEventShim).
 * Listener exceptions never propagate (GL errors are the only page channel).
 */
function dispatchCanvasEvent(canvas: CanvasLike, event: CanvasEvent): void {
  const c = canvas as { dispatchEvent?: (e: unknown) => unknown };
  if (typeof c.dispatchEvent === 'function') {
    try {
      c.dispatchEvent(event);
    } catch {
      /* listener errors must not reach GL */
    }
    return;
  }
  const listeners = mockListeners.get(canvas as object);
  const fns = listeners ? listeners.get(event.type) : undefined;
  if (fns) {
    for (const fn of Array.from(fns)) {
      try {
        fn(event);
      } catch {
        /* ignore listener errors */
      }
    }
  }
}

/** Listeners registered on non-DOM canvases (addEventListener shim). */
const mockListeners = new WeakMap<object, Map<string, Set<(e: unknown) => void>>>();

/**
 * For canvases without dispatchEvent (Node mocks), wrap addEventListener so
 * webglcontextlost/restored can be delivered. Installed once per canvas at
 * context creation; never touches real DOM canvases (they have dispatchEvent).
 */
function installMockEventShim(canvas: CanvasLike): void {
  const c = canvas as {
    dispatchEvent?: unknown;
    addEventListener?: unknown;
    removeEventListener?: unknown;
    __swglEventShim?: boolean;
  };
  if (typeof c.dispatchEvent === 'function' || typeof c.addEventListener !== 'function') return;
  if (c.__swglEventShim) return;
  c.__swglEventShim = true;
  const listeners = new Map<string, Set<(e: unknown) => void>>();
  mockListeners.set(canvas as object, listeners);
  const origAdd = c.addEventListener.bind(c);
  const origRemove =
    typeof c.removeEventListener === 'function' ? c.removeEventListener.bind(c) : null;
  c.addEventListener = ((type: string, fn: (e: unknown) => void) => {
    origAdd(type, fn);
    let set = listeners.get(type);
    if (!set) {
      set = new Set();
      listeners.set(type, set);
    }
    set.add(fn);
  }) as typeof c.addEventListener;
  if (origRemove) {
    c.removeEventListener = ((type: string, fn: (e: unknown) => void) => {
      origRemove(type, fn);
      listeners.get(type)?.delete(fn);
    }) as typeof c.removeEventListener;
  }
}
