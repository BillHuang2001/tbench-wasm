/**
 * src/gl/lost.ts — the SHARED WEBGL_lose_context engine (context loss/restore).
 *
 * Single home for the lose/restore semantics used by BOTH the
 * WEBGL_lose_context extension factory (extensions/misc.ts) and the lifecycle
 * module (lifecycle.ts — which re-exports these functions). Extracted into its
 * own module because the extension cannot import lifecycle.ts directly: a
 * static import there would create the module cycle
 * (webgl1 → api → extensions → misc → lifecycle → webgl1) that breaks
 * evaluation with a TDZ ReferenceError on WebGLRenderingContext.
 *
 * CYCLE-FREE CONSTRAINT: this module imports ONLY cycle-free modules
 * ('./state', './errors', './constants', '../present', '../raster' — none of
 * which import webgl1/webgl2/lifecycle/api) and operates on the context
 * STRUCTURALLY via `ContextLike` (no import of webgl1.ts/webgl2.ts).
 *
 * Semantics (CTS context-lost.html + context-lost-restored.html):
 *  - loseContext() → isContextLost() true IMMEDIATELY, all resources
 *    invalidated, `webglcontextlost` event QUEUED asynchronously (must NOT fire
 *    synchronously — the tests assert `contextLostEventFired == false` right
 *    after loseContext());
 *  - restoration is only allowed when the page preventDefault()s the lost event;
 *  - restoreContext() is ASYNC: isContextLost() stays true until the queued
 *    restore task re-initializes the context and fires `webglcontextrestored`;
 *  - a second restoreContext() while a restore is pending is a NO_ERROR no-op;
 *  - while lost, API calls are no-ops WITHOUT generating errors (the tests
 *    assert NO_ERROR after each call); getError returns CONTEXT_LOST_WEBGL
 *    once (handled in api/context.ts).
 *
 * The drawing-buffer re-allocation machinery (initContextResources,
 * handleCanvasResize) lives here too so the engine is self-contained;
 * lifecycle.ts re-exports it unchanged.
 */

import { createDefaultState, defaultVAOState, type State, type VAOState } from './state';
import type { ErrorQueue } from './errors';
import { C1 } from './constants';
import { createCanvasSurface, type CanvasSurface } from '../present';
import {
  createSurface,
  getFormat,
  type PixelFormatInfo,
  type Surface,
} from '../raster';
// Internal format GLenums used by the drawing buffer (raster/gl-enums values).
import { RGBA8, DEPTH_COMPONENT16, DEPTH_COMPONENT24, STENCIL_INDEX8 } from '../raster';

/**
 * Structural view of a WebGL context — everything the engine touches. Deliberately
 * NOT the WebGLRenderingContext class (importing it would re-enter the module
 * cycle); the real contexts satisfy this shape.
 */
interface ContextLike {
  _version: 1 | 2;
  _isLost: boolean;
  _canvas: unknown;
  _attrs: { depth: boolean; stencil: boolean; alpha?: boolean };
  _state: State;
  _errors: ErrorQueue;
  _resources: { invalidateAll(): void };
  /** Extension singleton cache (canonical name → object). */
  _extensions: Map<string, unknown>;
  _presentSurface: CanvasSurface | null;
  _defaultFB: DefaultFramebuffer | null;
  _defaultVAO: VAOState | null;
  _drawingBufferWidth: number;
  _drawingBufferHeight: number;
}

/** The default framebuffer (drawing buffer) — shape matches webgl1.ts's type. */
interface DefaultFramebuffer {
  color: Surface;
  depth: Surface | null;
  stencil: Surface | null;
  width: number;
  height: number;
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

/** Per-context loss bookkeeping (kept off the class — engine-owned). */
const lostState = new WeakMap<object, LostState>();

/** Per-context getError bookkeeping for the "first call returns CONTEXT_LOST_WEBGL" rule. */
const lostErrorState = new WeakMap<object, { consumed: boolean }>();

/** True while the context is lost (api/context.ts isContextLost reads _isLost). */
export function isContextLost(ctx: ContextLike): boolean {
  return ctx._isLost;
}

/**
 * getError() while the context is lost (api/context.ts delegates here): the
 * first call after a loss returns CONTEXT_LOST_WEBGL; subsequent calls drain
 * the error queue normally. Errors generated WHILE lost must be observable —
 * e.g. loseContext() on an already-lost context pushes INVALID_OPERATION and
 * CTS context-lost.html asserts it via getError. Void API calls while lost are
 * silent no-ops (the api/ guards push nothing), so the queue drains to
 * NO_ERROR for them.
 */
export function getErrorWhileLost(ctx: ContextLike): number {
  let st = lostErrorState.get(ctx);
  if (!st) {
    st = { consumed: false };
    lostErrorState.set(ctx, st);
  }
  if (!st.consumed) {
    st.consumed = true;
    return C1.CONTEXT_LOST_WEBGL;
  }
  return ctx._errors.get();
}

/** Forget the lost-epoch bookkeeping (fresh epoch per loss; cleared on restore). */
export function resetLostErrorState(ctx: ContextLike): void {
  lostErrorState.delete(ctx);
}

/**
 * Guard for API entry points: returns false while the context is lost — callers
 * no-op. IMPORTANT: no error is pushed — CTS context-lost.html asserts NO_ERROR
 * after every void call while lost (the spec's CONTEXT_LOST_WEBGL is delivered
 * once via getError, handled in api/context.ts).
 */
export function ensureNotLost(ctx: ContextLike): boolean {
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
export function loseContext(ctx: ContextLike): void {
  if (ctx._isLost) {
    ctx._errors.push(C1.INVALID_OPERATION); // WEBGL_lose_context: already lost
    return;
  }
  resetLostErrorState(ctx); // fresh epoch: first getError after THIS loss → CONTEXT_LOST_WEBGL
  ctx._isLost = true;
  // Set the spec's "invalidated flag" on every tracked object BEFORE
  // invalidateAll() (which sets _deleted and clears the tracking set). Query
  // validators reject invalidated objects with INVALID_OPERATION even when
  // they are still in use — CTS gl-get-attrib-location-errors.html checks
  // getAttribLocation on a pre-loss program after restore (expects -1 +
  // INVALID_OPERATION, "Program created before the context lost event").
  const tracked = (ctx._resources as unknown as { all?: Set<{ _invalidated?: boolean }> }).all;
  if (tracked) {
    for (const obj of tracked) obj._invalidated = true;
  }
  ctx._resources.invalidateAll();
  lostState.set(ctx, { restoreAllowed: false, restorePending: false });
  const canvas = ctx._canvas;
  schedule(() => {
    // A REAL Event in browsers — canvas.dispatchEvent throws TypeError on plain
    // objects, which would silently swallow the page's webglcontextlost
    // listener (CTS context-lost tests hang forever without it).
    const event = makeContextEvent('webglcontextlost', '', true);
    dispatchCanvasEvent(canvas, event);
    const st = lostState.get(ctx);
    if (st) st.restoreAllowed = event.defaultPrevented === true;
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
export function restoreContext(ctx: ContextLike): void {
  const st = lostState.get(ctx);
  if (!ctx._isLost || !st) {
    ctx._errors.push(C1.INVALID_OPERATION); // nothing to restore
    return;
  }
  if (!st.restoreAllowed) {
    ctx._errors.push(C1.INVALID_OPERATION); // event default not prevented
    return;
  }
  if (st.restorePending) return; // idempotent (NO_ERROR)
  st.restorePending = true;
  schedule(() => doRestore(ctx));
}

/** The queued restore task: full state reset + drawing buffer re-creation. */
function doRestore(ctx: ContextLike): void {
  // Context state is fully reset to defaults (spec).
  ctx._state = createDefaultState(ctx._version);
  ctx._errors.clear();
  // Re-create the drawing buffer, default VAO, present adapter, initial viewport.
  initContextResources(ctx);
  ctx._isLost = false;
  resetLostErrorState(ctx); // a later loss starts a fresh CONTEXT_LOST_WEBGL epoch
  // Extension singletons are re-created on restore — EXCEPT WEBGL_lose_context,
  // which must be the SAME object (CTS context-lost-restored.html: only
  // WEBGL_lose_context keeps its webglTestProperty).
  for (const key of Array.from(ctx._extensions.keys())) {
    if (!/lose_context$/i.test(key)) ctx._extensions.delete(key);
  }
  lostState.delete(ctx);
  dispatchCanvasEvent(ctx._canvas, makeContextEvent('webglcontextrestored', '', false));
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
 * Called by lifecycle.createContext() after construction (and by doRestore()).
 * @internal engine — re-exported by lifecycle.ts.
 */
export function initContextResources(ctx: ContextLike): void {
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
  // Resize the present surface BEFORE the color surface aliases its pixel
  // buffer: a freshly created surface is 0×0, and its first present() would
  // auto-resize to a NEW zeroed buffer, orphaning the zero-copy alias (canvas
  // screenshots/toDataURL would composite black). Stub-safe.
  if (ctx._presentSurface) {
    try {
      const ps = ctx._presentSurface as { resize?: (w: number, h: number) => void };
      if (typeof ps.resize === 'function') ps.resize(w, h);
    } catch {
      /* present stub — local surface fallback below still works */
    }
  }
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
  ctx: ContextLike,
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

/**
 * Color surface: zero-copy wrap of the present surface's RGBA8 buffer. For an
 * alpha:false drawing buffer the back buffer has NO alpha channel — the
 * surface's info is wrapped so every encode stores a=1.0 and every decode
 * reports a=1.0 (CTS context-attributes-alpha-depth-stencil-antialias.html,
 * context-hidden-alpha.html, context-no-alpha-fbo-with-alpha.html). FBO color
 * surfaces are allocated elsewhere and keep their real alpha.
 */
function makeColorSurface(ctx: ContextLike, w: number, h: number): Surface {
  let pixels: Uint8Array | null = null;
  if (ctx._presentSurface) {
    try {
      pixels = ctx._presentSurface.getPixels();
    } catch {
      pixels = null; // present stub — allocate locally
    }
  }
  if (!pixels || pixels.length < w * h * 4) pixels = new Uint8Array(w * h * 4);
  const surf = makeSurface(RGBA8, w, h, pixels);
  if (ctx._attrs.alpha === false) {
    const base = surf.info;
    surf.info = {
      ...base,
      encode(src, byteOffset, r, g, b, a) {
        base.encode(src, byteOffset, r, g, b, 1);
      },
      decode(src, byteOffset, out) {
        const o = base.decode(src, byteOffset, out);
        o[3] = 1;
        return o;
      },
    };
  }
  return surf;
}

function makeDepthSurface(ctx: ContextLike, w: number, h: number): Surface {
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
        out = out ?? new Float32Array(4);
        const d = (src as Float32Array)[byteOffset >> 2];
        out[0] = d;
        out[1] = d;
        out[2] = d;
        out[3] = 1;
        return out;
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
        out = out ?? new Float32Array(4);
        const s = (src as Uint8Array)[byteOffset];
        out[0] = s;
        out[1] = s;
        out[2] = s;
        out[3] = 1;
        return out;
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
      out = out ?? new Float32Array(4);
      const u = src as Uint8Array;
      out[0] = u[byteOffset] / 255;
      out[1] = u[byteOffset + 1] / 255;
      out[2] = u[byteOffset + 2] / 255;
      out[3] = u[byteOffset + 3] / 255;
      return out;
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
 * Resolve a canvas's drawing-buffer dimensions: max(1, canvas.width/height) —
 * a 0-size canvas yields a 1×1 buffer (CTS context/zero-sized-canvas.html).
 * Returns null when the canvas has no numeric width/height (degenerate mocks).
 */
function canvasSize(ctx: ContextLike): { w: number; h: number } | null {
  const c = ctx._canvas as { width?: unknown; height?: unknown };
  if (typeof c.width !== 'number' || typeof c.height !== 'number') return null;
  return { w: Math.max(1, c.width), h: Math.max(1, c.height) };
}

/**
 * Resize the drawing buffer after canvas.width/height changed (called by the
 * present/ adapter or entry.ts on canvas resize observation). Reallocates the
 * default framebuffer surfaces and CLEARS them (color → 0, depth → 1.0,
 * stencil → 0 — CTS canvas/canvas-test.html checks the canvas content is
 * (0,0,0,0) after resize). State (viewport, scissor, clear color, masks,
 * bindings) is PRESERVED — CTS canvas-test.html + viewport-unchanged-upon-resize.html.
 * No-op while the context is lost (restore re-sizes) or when size is unchanged.
 * @internal engine — re-exported by lifecycle.ts.
 */
export function handleCanvasResize(ctx: ContextLike): void {
  if (ctx._isLost) return; // loss freezes the drawing buffer; restore re-sizes
  const size = canvasSize(ctx);
  if (!size) return;
  const { w, h } = size;
  const surf = ctx._presentSurface;
  // Reference the DEFAULT FRAMEBUFFER size, not the present surface's: the
  // present adapter may have auto-resized its pixel buffer ahead of the
  // drawing buffer (its present() safety net), and comparing against the
  // surface would no-op here and leave the drawing buffer stale forever.
  const curW = ctx._defaultFB ? ctx._defaultFB.width : surf ? surf.width : 0;
  const curH = ctx._defaultFB ? ctx._defaultFB.height : surf ? surf.height : 0;
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
  /** True after preventDefault() ran (real DOM Events; our mocks mirror it). */
  defaultPrevented?: boolean;
}

/**
 * Build a context-lifecycle event (webglcontextlost / webglcontextrestored).
 * Environments with an Event constructor (browsers — and Node ≥15, where unit
 * tests run) get a REAL Event: canvas.dispatchEvent throws TypeError on plain
 * objects, so without this the page's listeners never run and CTS context-lost
 * tests hang. Environments without Event get a plain mock with the same
 * observable shape (type/statusMessage/cancelable/preventDefault/
 * defaultPrevented). statusMessage is attached as a property either way (CTS
 * context-lost tests read event.statusMessage); preventDefault() tracks via
 * defaultPrevented, which the caller reads after dispatch.
 */
function makeContextEvent(type: string, statusMessage: string, cancelable: boolean): CanvasEvent {
  const evtCtor = (globalThis as { Event?: new (type: string, init?: EventInit) => Event }).Event;
  if (typeof evtCtor === 'function') {
    const ev = new evtCtor(type, { cancelable });
    (ev as unknown as { statusMessage?: string }).statusMessage = statusMessage;
    return ev as unknown as CanvasEvent;
  }
  const ev: CanvasEvent & { defaultPrevented: boolean } = {
    type,
    statusMessage,
    cancelable,
    defaultPrevented: false,
    preventDefault(): void {
      if (cancelable) ev.defaultPrevented = true;
    },
  };
  return ev;
}

/**
 * Deliver an event to a canvas: real DOM canvases dispatchEvent (listeners were
 * registered through the DOM); non-DOM mocks get the events delivered to
 * listeners registered via our addEventListener shim (see installMockEventShim).
 * Listener exceptions never propagate (GL errors are the only page channel).
 */
function dispatchCanvasEvent(canvas: unknown, event: CanvasEvent): void {
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
 * @internal engine — called by lifecycle.createContext().
 */
export function installMockEventShim(canvas: unknown): void {
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
