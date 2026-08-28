/**
 * src/gl/api/webgl2.ts — WebGL2-only object methods (queries, sync, samplers,
 * VAOs, transform feedback bindings).
 *
 * Owns: beginQuery, endQuery, createQuery, deleteQuery, isQuery, getQuery,
 * getQueryParameter; fenceSync, isSync, deleteSync, clientWaitSync, waitSync,
 * getSyncParameter; createSampler, deleteSampler, isSampler, bindSampler,
 * samplerParameterf, samplerParameteri, getSamplerParameter; createVertexArray,
 * deleteVertexArray, isVertexArray, bindVertexArray; createTransformFeedback,
 * deleteTransformFeedback, isTransformFeedback, bindTransformFeedback,
 * beginTransformFeedback, endTransformFeedback, pauseTransformFeedback,
 * resumeTransformFeedback; getInternalformatParameter.
 *
 * NOT owned here (parallel agents): bindBufferBase/Range, getIndexedParameter,
 * getBufferSubData (buffers.ts); vertexAttribDivisor + I* attribs (vertex-attrib.ts);
 * instanced/ranged/multi draws, clearBuffer*, readPixels (draw.ts); texImage3D,
 * texStorage2D/3D, compressedTexImage3D, compressedTexSubImage3D, copyTexSubImage3D
 * (teximage.ts); uniformBlock*,
 * getUniformIndices, getActiveUniformsiv, getFragDataLocation,
 * transformFeedbackVaryings, getTransformFeedbackVarying (programs.ts);
 * framebufferTextureLayer, drawBuffers, readBuffer, renderbufferStorageMultisample,
 * invalidate*, blitFramebuffer (framebuffers.ts).
 *
 * Installer contract: `installWebGL2Api` is called with BOTH context prototypes
 * (installAll runs it on WebGL1 too). EVERY method installed here MUST be guarded
 * with `if ('<name>' in proto)` so the installer is a no-op on the WebGL1
 * prototype (WebGL1 contexts must NOT expose beginQuery etc. — calling them must
 * throw TypeError "not a function", which absence achieves).
 *
 * Behavior notes / decisions (implemented):
 *  - Query targets: ANY_SAMPLES_PASSED, ANY_SAMPLES_PASSED_CONSERVATIVE and
 *    TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN are legal (CTS query/query.html; the
 *    objective's "INVALID_ENUM otherwise" list is expanded accordingly).
 *    Active queries live in state.activeQueries (plain object; the two typed
 *    fields feed the draw engine's occlusion accumulation; the TF-primitives
 *    slot is an extra key). Deleting an ACTIVE query clears its slot (CTS).
 *    isQuery requires a prior beginQuery; endQuery sets _resultAvailable=true
 *    (synchronous renderer) and _result holds whatever the draw engine
 *    accumulated (0 until it lands — real sample counting is draw.ts's job).
 *  - Sync: fenceSync signals immediately; clientWaitSync(…, 0) →
 *    ALREADY_SIGNALED; timeout > MAX_CLIENT_WAIT_TIMEOUT_WEBGL (0) →
 *    INVALID_OPERATION (sync-webgl-specific.html); waitSync validates only.
 *    CTS spin-loops expecting frame-delayed availability are a known gap
 *    (synchronous design).
 *  - Samplers: pname/value table per CTS samplers.html (MIN/MAX_LOD any value;
 *    BASE_LEVEL/MAX_LEVEL are NOT sampler pnames → INVALID_ENUM; MAG rejects
 *    mipmap filters; CLAMP_TO_BORDER rejected; TEXTURE_MAX_ANISOTROPY_EXT gated
 *    on ctx.getExtension('EXT_texture_filter_anisotropic')). isSampler true as
 *    soon as the object exists.
 *  - VAOs: bindVertexArray(null) restores the default VAO (ctx._defaultVAO,
 *    lazily created — pinned field on the context, webgl1.ts untouched);
 *    deleting the BOUND VAO rebinds the default (vertex-array-object.html);
 *    isVertexArray requires a prior bind (WeakSet).
 *  - Transform feedback: begin/end/pause/resume operate on the BOUND TF (or the
 *    default TF when none bound — module-level WeakMap). The default object is
 *    kept OUT of state.transformFeedback so TRANSFORM_FEEDBACK_BINDING stays
 *    null per CTS, EXCEPT during an active default-TF session: begin puts it
 *    there (the draw engine's tfActive check reads state.transformFeedback),
 *    end restores null, and bind(null) keeps it while a default session is
 *    paused (capture resumes correctly).
 *    beginTransformFeedback → INVALID_OPERATION when the BOUND TF is active
 *    (switching-objects.html allows beginning a DIFFERENT TF while another is
 *    active-paused), no linked program in use, program has no TF varyings
 *    (runNoOutputsTest), or fewer TF buffers bound than needed (INTERLEAVED ≥1;
 *    SEPARATE one per varying — counted against the BEGUN object's own indexed
 *    bindings).
 *    bindTransformFeedback rejected only while BOUND TF active AND unpaused.
 *    Indexed TF bindings are PER-TF-OBJECT state (GLES 3.0 §6.24): while a TF
 *    object is bound, bindBufferBase/Range writes its _buffers/_bufferRanges
 *    and they persist across bindTransformFeedback switches
 *    (switching-objects.html); bindTransformFeedback only initializes a
 *    NEVER-bound object once from the buffers agent's global mirror. With no
 *    TF object bound the indexed bindings belong to the DEFAULT TF object
 *    (name 0) — the mirror — copied into the default object at begin. The
 *    generic TRANSFORM_FEEDBACK_BUFFER binding point is context-global
 *    (getters.ts reads it) and bindTransformFeedback never touches it. Delete
 *    of an active TF → INVALID_OPERATION (CTS).
 *  - getInternalformatParameter: RENDERBUFFER + SAMPLES only; internalformat
 *    must be an ES3 renderable format (local list — raster/formats.ts
 *    isValidRenderbufferFormat is a throwing stub); returns Int32Array([4]),
 *    consistent with MAX_SAMPLES=4 and multisample-with-full-sample-counts
 *    (samples 1..array[0] must work).
 */

import type { WebGL2RenderingContext } from '../webgl2';
import type { WebGLRenderingContext } from '../webgl1';
import { C1, C2, CExt } from '../constants';
import {
  WebGLBuffer,
  WebGLQuery,
  WebGLSampler,
  WebGLSync,
  WebGLTransformFeedback,
  WebGLVertexArrayObject,
  createObject,
} from '../objects';
import type { WebGLObject } from '../objects';
import { validateObject, validateNonNullableObject } from '../validation';
import { ensureProgramLinked } from './programs';
import { syncCurrentAttribs } from './vertex-attrib';
import { defaultVAOState } from '../state';
import type { VAOState } from '../state';
import { executeClearBuffer, toU64 } from '../draw';
import { refreshUnitSamplerBindings } from '../teximage';
import { createSurface, getFormat, linearToSRGB } from '../../raster';
import type { Surface } from '../../raster';
import type {
  GLbitfield, GLboolean, GLenum, GLfloat, GLint, GLsizei, GLuint, GLuint64,
  Int32List, Uint32List,
} from '../types';

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

/**
 * Constructor casts: WebGL object subclasses inherit WebGLObject's PROTECTED
 * constructor, so cast once per class (mirrors buffers.ts).
 */
function ctorPair<T extends WebGLObject>(C: { prototype: T }): {
  make: new (context: WebGLRenderingContext) => T;
  any: new (...args: never[]) => T;
} {
  const make = C as unknown as new (context: WebGLRenderingContext) => T;
  return { make, any: make as unknown as new (...args: never[]) => T };
}

const Query = ctorPair(WebGLQuery);
const Sync = ctorPair(WebGLSync);
const Sampler = ctorPair(WebGLSampler);
const Vao = ctorPair(WebGLVertexArrayObject);
const Tf = ctorPair(WebGLTransformFeedback);

// ---------------------------------------------------------------------------
// drawingBufferStorage + clearBufferiv/uiv helpers
// (installed in installWebGL2Api — see below)
// ---------------------------------------------------------------------------

/** WebIDL Int32List/Uint32List → typed array (TypeError on junk; mirrors api/draw.ts). */
function toListLocal<T extends Int32Array | Uint32Array>(
  values: Int32List | Uint32List,
  Ctor: new (src: ArrayLike<number>) => T,
  name: string,
): T {
  if (values instanceof Ctor) return values;
  if (Array.isArray(values) || ArrayBuffer.isView(values)) {
    return new Ctor(values as ArrayLike<number>);
  }
  throw new TypeError(`Argument is not of type '${name}'`);
}

/**
 * Format info of the color surface mapped to draw-buffer slot `idx` by the
 * CURRENT drawBuffers() state (the caller has already validated that the slot
 * maps to a real attachment). Returns null when the attachment is missing —
 * per the WebGL 2.0 spec §4.2.2, clearing a missing attachment of a complete
 * framebuffer clears nothing and generates no error.
 */
function clearColorAttachmentInfo(ctx: WebGLRenderingContext, idx: number): Surface['info'] | null {
  const fbo = ctx._state.drawFramebuffer;
  if (fbo === null) {
    const dfb = ctx._defaultFB;
    return idx === 0 && dfb ? dfb.color.info : null;
  }
  const att = fbo._attachments.get(C1.COLOR_ATTACHMENT0 + idx);
  if (!att) return null;
  if (att.type === 'renderbuffer') return att.renderbuffer._surface?.info ?? null;
  return att.texture._image?.info ?? null;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/** Legal beginQuery/endQuery/getQuery targets. */
const QUERY_TARGETS = new Set<number>([
  C2.ANY_SAMPLES_PASSED,
  C2.ANY_SAMPLES_PASSED_CONSERVATIVE,
  C2.TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN,
]);

/**
 * Active query for `target`. state.activeQueries is a plain object; the two
 * typed fields serve the draw engine's occlusion accumulation, and the
 * TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN slot lives on the same object under an
 * extra key (no state.ts edit needed).
 */
function activeQueryAt(ctx: WebGLRenderingContext, target: number): WebGLQuery | null {
  const slots = ctx._state.activeQueries as unknown as Record<number, WebGLQuery | null>;
  return slots[target] ?? null;
}

function setActiveQuery(ctx: WebGLRenderingContext, target: number, q: WebGLQuery | null): void {
  const slots = ctx._state.activeQueries as unknown as Record<number, WebGLQuery | null>;
  slots[target] = q;
}

/**
 * Event-loop-deferred query availability. The renderer is synchronous, but the
 * spec (and CTS occlusion-query.html / transform_feedback.html) require a
 * query's result to become AVAILABLE only after control returns to the event
 * loop — pages spin-loop with gl.finish() and assert QUERY_RESULT_AVAILABLE is
 * false throughout the current task. The result VALUE is computed synchronously
 * (the draw engine accumulates `_result`); only the availability flag is
 * deferred: endQuery records the query in a per-context pending set and ONE
 * setTimeout(0) flips `_resultAvailable` for the whole batch.
 *
 * The draw engine (draw.ts, read-only) sets `_resultAvailable = true`
 * synchronously DURING the draw (between beginQuery and endQuery) — the
 * pending-set gate in getQueryParameter keeps that unobservable until the
 * timer fires. Module-level WeakMap so no new enumerable fields ever appear on
 * the context object (CTS constants-and-properties* enumerate `for (var i in
 * gl)`).
 */
interface QueryAvailabilityState {
  pending: Set<WebGLQuery>;
  timer: ReturnType<typeof setTimeout> | null;
}

const queryAvailability = new WeakMap<WebGLRenderingContext, QueryAvailabilityState>();

function queryAvailabilityState(ctx: WebGLRenderingContext): QueryAvailabilityState {
  let st = queryAvailability.get(ctx);
  if (!st) {
    st = { pending: new Set(), timer: null };
    queryAvailability.set(ctx, st);
  }
  return st;
}

/** Arm the deferral for `q`: availability stays false until the next event-loop turn. */
function deferQueryAvailability(ctx: WebGLRenderingContext, q: WebGLQuery): void {
  const st = queryAvailabilityState(ctx);
  st.pending.add(q);
  if (st.timer === null) {
    st.timer = setTimeout(() => {
      st.timer = null;
      for (const qq of st.pending) qq._resultAvailable = true;
      st.pending.clear();
    }, 0);
  }
}

// ---------------------------------------------------------------------------
// Sync objects
// ---------------------------------------------------------------------------

let nextSyncId = 1;

// ---------------------------------------------------------------------------
// Samplers — pname/value validation tables (WebGL2 sampler pnames only).
// ---------------------------------------------------------------------------

const MIN_FILTER_VALUES = new Set<number>([
  C1.NEAREST,
  C1.LINEAR,
  C1.NEAREST_MIPMAP_NEAREST,
  C1.NEAREST_MIPMAP_LINEAR,
  C1.LINEAR_MIPMAP_NEAREST,
  C1.LINEAR_MIPMAP_LINEAR,
]);
const MAG_FILTER_VALUES = new Set<number>([C1.NEAREST, C1.LINEAR]);
const WRAP_VALUES = new Set<number>([C1.REPEAT, C1.CLAMP_TO_EDGE, C1.MIRRORED_REPEAT]);
const COMPARE_MODE_VALUES = new Set<number>([C1.NONE, C2.COMPARE_REF_TO_TEXTURE]);
const COMPARE_FUNC_VALUES = new Set<number>([
  C1.NEVER,
  C1.LESS,
  C1.EQUAL,
  C1.LEQUAL,
  C1.GREATER,
  C1.NOTEQUAL,
  C1.GEQUAL,
  C1.ALWAYS,
]);

/**
 * EXT_texture_filter_anisotropic enabled for this context. Checks the enabled
 * registry (`_extensions`), NOT ctx.getExtension — getExtension ENABLES the
 * extension, and CTS ext-texture-filter-anisotropic.html expects
 * TEXTURE_MAX_ANISOTROPY_EXT on samplers to be INVALID_ENUM before any
 * getExtension call (mirrors api/textures.ts).
 */
function anisotropyEnabled(ctx: WebGLRenderingContext): boolean {
  return ctx._extensions.has('EXT_texture_filter_anisotropic');
}

// ---------------------------------------------------------------------------
// VAOs
// ---------------------------------------------------------------------------

/** VAOs that have been bound at least once (isVertexArray — CTS expects false before first bind). */
const everBoundVAOs = new WeakSet<WebGLVertexArrayObject>();

/** The context's default VAO contents, created lazily (pinned field on the context). */
function defaultVAO(ctx: WebGLRenderingContext): VAOState {
  if (!ctx._defaultVAO) ctx._defaultVAO = defaultVAOState(ctx._state.limits.MAX_VERTEX_ATTRIBS);
  return ctx._defaultVAO;
}

// ---------------------------------------------------------------------------
// Transform feedback
// ---------------------------------------------------------------------------

/** TF objects that have been bound at least once (isTransformFeedback). */
const everBoundTFs = new WeakSet<WebGLTransformFeedback>();

/**
 * Named TF objects whose indexed TRANSFORM_FEEDBACK_BUFFER bindings have been
 * initialized (from the global mirror) at least once. After that, the object's
 * OWN bindings win: bindBufferBase/Range writes them directly while the object
 * is bound, and bindTransformFeedback must never clobber them (GLES 3.0 §2.15/
 * §6.24 — CTS switching-objects.html persists per-object bindings across
 * object switches). A fresh named object inherits the mirror once so bindings
 * made while no TF object was bound are visible on its first bind/begin.
 */
const tfInitialized = new WeakSet<WebGLTransformFeedback>();

/**
 * The default transform feedback object (the active TF when no TF object is
 * bound; name 0 per GLES 3.0 §2.15). Usually kept OUT of state.transformFeedback
 * so getParameter(TRANSFORM_FEEDBACK_BINDING) stays null (CTS
 * default_transform_feedback.html / transform_feedback.html bindings tests) —
 * EXCEPT during an active default-TF session, when beginTransformFeedback
 * exposes it there so the draw engine captures into it (see begin/end below).
 */
const defaultTFs = new WeakMap<WebGLRenderingContext, WebGLTransformFeedback>();

function getDefaultTF(ctx: WebGLRenderingContext): WebGLTransformFeedback {
  let tf = defaultTFs.get(ctx);
  if (!tf) {
    tf = new Tf.make(ctx);
    defaultTFs.set(ctx, tf);
  }
  return tf;
}

/**
 * Indexed TRANSFORM_FEEDBACK_BUFFER binding at `index` from the GLOBAL mirror
 * (last bind wins). The mirror is the DEFAULT TF object's (name 0) indexed
 * bindings — the bindings made via bindBufferBase/Range while NO TF object was
 * bound (api/buffers.ts records them in the buffers' _tfRangeBindings). Used
 * to initialize the default object and never-bound named objects. `base`
 * (whole-buffer bindBufferBase vs FIXED bindBufferRange) is carried through so
 * per-object capture-capacity semantics survive the sync.
 */
function tfBindingAtIndex(
  ctx: WebGLRenderingContext,
  index: number,
): { buffer: WebGLBuffer | null; offset: number; size: number; base: boolean } {
  for (const obj of ctx._resources.all) {
    if (obj instanceof WebGLBuffer) {
      for (const e of obj._tfRangeBindings) {
        if (e.index === index) return { buffer: obj, offset: e.offset, size: e.size, base: e.base };
      }
    }
  }
  return { buffer: null, offset: 0, size: 0, base: false };
}

/**
 * Initialize `tf`'s indexed TRANSFORM_FEEDBACK_BUFFER bindings. Called at
 * bindTransformFeedback and beginTransformFeedback — it must NOT clobber
 * existing per-object bindings (switching-objects.html rebinds tf1 after tf2
 * and expects tf1's own buffer):
 * - DEFAULT TF object (name 0): always rebuilt from the global mirror (the
 *   mirror IS the default object's state; bindBufferBase while no TF object is
 *   bound writes there — default_transform_feedback.html, too-small-buffers.html).
 * - Named object on FIRST bind/begin: initialized once from the mirror;
 *   afterwards the object's own bindings win (bindBufferBase/Range writes them
 *   directly while the object is bound).
 */
function syncTfBuffers(ctx: WebGLRenderingContext, tf: WebGLTransformFeedback): void {
  const n = ctx._state.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
  if (tf._buffers.length !== n) {
    tf._buffers = new Array<WebGLBuffer | null>(n).fill(null);
    tf._bufferRanges = Array.from({ length: n }, () => ({ offset: 0, size: 0, base: false }));
  }
  const isDefault = tf === defaultTFs.get(ctx);
  if (isDefault) {
    for (let i = 0; i < n; i++) {
      const b = tfBindingAtIndex(ctx, i);
      tf._buffers[i] = b.buffer;
      tf._bufferRanges[i] = { offset: b.offset, size: b.size, base: b.base };
    }
  } else if (!tfInitialized.has(tf)) {
    tfInitialized.add(tf);
    for (let i = 0; i < n; i++) {
      const b = tfBindingAtIndex(ctx, i);
      if (b.buffer !== null) {
        tf._buffers[i] = b.buffer;
        tf._bufferRanges[i] = { offset: b.offset, size: b.size, base: b.base };
      }
    }
  }
}

/**
 * The currently ACTIVE transform feedback object: the bound TF object when it
 * is active, else the default TF object when IT is active, else null. Per CTS
 * (switching-objects.html) begin/end/pause/resume all operate on the BOUND
 * object — no resource scanning needed.
 */
function activeTF(ctx: WebGLRenderingContext): WebGLTransformFeedback | null {
  const bound = ctx._state.transformFeedback;
  if (bound && bound._active) return bound;
  const def = defaultTFs.get(ctx);
  if (def && def._active) return def;
  return null;
}

// ---------------------------------------------------------------------------
// getInternalformatParameter — ES3 renderable renderbuffer formats.
// ---------------------------------------------------------------------------

/** Color- + depth/stencil-renderable internal formats (ES3 core + EXT_texture_norm16). */
const RENDERBUFFER_FORMATS = new Set<number>([
  C1.RGBA4,
  C1.RGB565,
  C1.RGB5_A1,
  C2.R8,
  C2.RG8,
  C2.RGB8,
  C2.RGBA8,
  C2.RGB10_A2,
  C2.SRGB8_ALPHA8,
  C2.R8I,
  C2.R8UI,
  C2.R16I,
  C2.R16UI,
  C2.R32I,
  C2.R32UI,
  C2.RG8I,
  C2.RG8UI,
  C2.RG16I,
  C2.RG16UI,
  C2.RG32I,
  C2.RG32UI,
  C2.RGBA8I,
  C2.RGBA8UI,
  C2.RGBA16I,
  C2.RGBA16UI,
  C2.RGBA32I,
  C2.RGBA32UI,
  C2.RGB10_A2UI,
  C1.DEPTH_COMPONENT16,
  C2.DEPTH_COMPONENT24,
  C2.DEPTH_COMPONENT32F,
  C2.DEPTH24_STENCIL8,
  C2.DEPTH32F_STENCIL8,
  C1.STENCIL_INDEX8,
]);

/** WebGL2 float renderbuffer formats (gated on EXT_color_buffer_float / _half_float). */
const RB_EXT_FLOAT_FORMATS = new Set<number>([
  C2.R16F, C2.RG16F, C2.RGBA16F, C2.R32F, C2.RG32F, C2.RGBA32F, C2.R11F_G11F_B10F,
]);

/** WebGL2 renderbuffer formats gated on EXT_texture_norm16. */
const RB_EXT_NORM16_FORMATS = new Set<number>([CExt.R16_EXT, CExt.RG16_EXT, CExt.RGBA16_EXT]);

/** True when an EXT-gated internal format is a legal getInternalformatParameter target. */
function extRenderbufferFormatOK(ctx: WebGLRenderingContext, internalformat: number): boolean {
  if (RB_EXT_FLOAT_FORMATS.has(internalformat)) {
    if (ctx._extensions.has('EXT_color_buffer_float')) return true;
    return (internalformat === C2.R16F || internalformat === C2.RG16F || internalformat === C2.RGBA16F) &&
      ctx._extensions.has('EXT_color_buffer_half_float');
  }
  if (RB_EXT_NORM16_FORMATS.has(internalformat)) return ctx._extensions.has('EXT_texture_norm16');
  return false;
}

export function installWebGL2Api(proto: WebGL2RenderingContext): void {
  // ---- Queries ----
  if ('beginQuery' in proto) {
    proto.beginQuery = function (this: WebGL2RenderingContext, target: GLenum, query: WebGLQuery): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (!QUERY_TARGETS.has(target)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const q = validateNonNullableObject<WebGLQuery>(ctx, query, Query.any);
      if (q === null) return; // cross-context/deleted → INVALID_OPERATION pushed
      if (activeQueryAt(ctx, target) !== null) {
        ctx._errors.push(C1.INVALID_OPERATION); // target already has an active query
        return;
      }
      if (q._active) {
        ctx._errors.push(C1.INVALID_OPERATION); // query object already active elsewhere
        return;
      }
      if (q._target !== 0 && q._target !== target) {
        ctx._errors.push(C1.INVALID_OPERATION); // can't re-use for incompatible target
        return;
      }
      q._target = target;
      q._active = true;
      q._result = 0;
      q._resultAvailable = false;
      // Availability is deferred to the event loop (the draw engine may set
      // _resultAvailable synchronously during the session — see header note).
      deferQueryAvailability(ctx, q);
      setActiveQuery(ctx, target, q);
    };
  }

  if ('endQuery' in proto) {
    proto.endQuery = function (this: WebGL2RenderingContext, target: GLenum): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (!QUERY_TARGETS.has(target)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const q = activeQueryAt(ctx, target);
      if (q === null) {
        ctx._errors.push(C1.INVALID_OPERATION); // no active query on target
        return;
      }
      q._active = false;
      // The result (accumulated by the draw engine into _result) is final NOW,
      // but QUERY_RESULT_AVAILABLE only becomes observable after control
      // returns to the event loop (CTS occlusion-query.html "became available
      // too early"; transform_feedback.html same). Re-arm in case the session
      // spanned an event-loop turn and the previous timer already fired.
      deferQueryAvailability(ctx, q);
      setActiveQuery(ctx, target, null);
    };
  }

  if ('createQuery' in proto) {
    proto.createQuery = function (this: WebGL2RenderingContext): WebGLQuery | null {
      const ctx = this;
      // No [WebGLHandlesContextLoss]: while lost it still creates an object
      // (CTS context-lost.html nonNullTests WebGL2 branch) with NO error.
      return createObject(ctx, Query.make);
    };
  }

  if ('deleteQuery' in proto) {
    proto.deleteQuery = function (this: WebGL2RenderingContext, query: WebGLQuery | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (query === null || query === undefined) return;
      if (!(query instanceof WebGLQuery)) {
        throw new TypeError(`Argument is not of type 'WebGLQuery'`);
      }
      if (query._context !== ctx) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (query._deleted) return; // already deleted: silent no-op (spec)
      // Deleting an ACTIVE query deactivates it and clears its target slot
      // (CTS query/query.html expects NO_ERROR and CURRENT_QUERY → null).
      for (const t of QUERY_TARGETS) {
        if (activeQueryAt(ctx, t) === query) setActiveQuery(ctx, t, null);
      }
      query._active = false;
      // Drop a pending availability flip (deleted queries must stay invisible).
      queryAvailabilityState(ctx).pending.delete(query);
      query._deleted = true;
      ctx._resources.untrack(query);
    };
  }

  if ('isQuery' in proto) {
    proto.isQuery = function (this: WebGL2RenderingContext, query: WebGLQuery | null): GLboolean {
      const ctx = this;
      if (isLost(ctx)) return false;
      if (query === null || query === undefined) return false;
      if (!(query instanceof WebGLQuery)) {
        throw new TypeError(`Argument is not of type 'WebGLQuery'`);
      }
      if (query._context !== ctx) return false; // cross-context: false, NO error
      // CTS query/query.html: false until first beginQuery (_target set), true
      // afterwards, false after delete.
      return !query._deleted && query._target !== 0;
    };
  }

  if ('getQuery' in proto) {
    proto.getQuery = function (this: WebGL2RenderingContext, target: GLenum, pname: GLenum): WebGLQuery | null {
      const ctx = this;
      if (isLost(ctx)) return null;
      if (!QUERY_TARGETS.has(target)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      if (pname !== C2.CURRENT_QUERY) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      return activeQueryAt(ctx, target);
    };
  }

  if ('getQueryParameter' in proto) {
    proto.getQueryParameter = function (this: WebGL2RenderingContext, query: WebGLQuery, pname: GLenum): any {
      const ctx = this;
      if (isLost(ctx)) return null;
      const q = validateNonNullableObject<WebGLQuery>(ctx, query, Query.any);
      if (q === null) return null; // cross-context/deleted → INVALID_OPERATION pushed
      switch (pname) {
        case C2.QUERY_RESULT:
          return q._result; // synchronous: no blocking (never-begun → 0)
        case C2.QUERY_RESULT_AVAILABLE:
          // Gated on the pending set: even though the draw engine may have set
          // _resultAvailable synchronously, the flag is only observable after
          // the deferred flip (see deferQueryAvailability).
          return !queryAvailabilityState(ctx).pending.has(q) && q._resultAvailable;
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
      }
    };
  }

  // ---- Sync objects ----
  if ('fenceSync' in proto) {
    proto.fenceSync = function (this: WebGL2RenderingContext, condition: GLenum, flags: GLbitfield): WebGLSync | null {
      const ctx = this;
      if (isLost(ctx)) return null;
      if (condition !== C2.SYNC_GPU_COMMANDS_COMPLETE) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      if (flags !== 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return null;
      }
      const sync = createObject(ctx, Sync.make);
      sync._condition = condition;
      sync._flags = flags;
      // All GL work completes synchronously, but the spec (CTS
      // sync-webgl-specific.html) requires the sync to become SIGNALED only
      // after control returns to the event loop — pages spin-loop with
      // readPixels and assert UNSIGNALED throughout the current task. A
      // one-shot setTimeout(0) flips it (the timer cannot fire mid-task).
      sync._signaled = false;
      sync._id = nextSyncId++;
      setTimeout(() => {
        sync._signaled = true;
      }, 0);
      return sync;
    };
  }

  if ('isSync' in proto) {
    proto.isSync = function (this: WebGL2RenderingContext, sync: WebGLSync | null): GLboolean {
      const ctx = this;
      if (isLost(ctx)) return false;
      if (sync === null || sync === undefined) return false;
      if (!(sync instanceof WebGLSync)) {
        throw new TypeError(`Argument is not of type 'WebGLSync'`);
      }
      if (sync._context !== ctx) return false; // cross-context: false, NO error
      return !sync._deleted;
    };
  }

  if ('deleteSync' in proto) {
    proto.deleteSync = function (this: WebGL2RenderingContext, sync: WebGLSync | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (sync === null || sync === undefined) return;
      if (!(sync instanceof WebGLSync)) {
        throw new TypeError(`Argument is not of type 'WebGLSync'`);
      }
      if (sync._context !== ctx) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (sync._deleted) return;
      sync._deleted = true;
      ctx._resources.untrack(sync);
    };
  }

  if ('clientWaitSync' in proto) {
    proto.clientWaitSync = function (this: WebGL2RenderingContext, sync: WebGLSync, flags: GLbitfield, timeout: GLuint64): GLenum {
      const ctx = this;
      if (isLost(ctx)) return C2.WAIT_FAILED;
      const s = validateNonNullableObject<WebGLSync>(ctx, sync, Sync.any);
      if (s === null) return C2.WAIT_FAILED; // INVALID_OPERATION pushed
      if (flags !== 0 && flags !== C2.SYNC_FLUSH_COMMANDS_BIT) {
        ctx._errors.push(C1.INVALID_VALUE);
        return C2.WAIT_FAILED;
      }
      if (timeout > ctx._state.limits.MAX_CLIENT_WAIT_TIMEOUT_WEBGL) {
        ctx._errors.push(C1.INVALID_OPERATION); // spec: timeout > MAX_CLIENT_WAIT_TIMEOUT_WEBGL
        return C2.WAIT_FAILED;
      }
      // The sync flips to SIGNALED on the event loop after fenceSync; before
      // that a zero-timeout wait must report TIMEOUT_EXPIRED (CTS
      // sync-webgl-specific.html spin-loop). Once signaled, ALREADY_SIGNALED
      // (SYNC_FLUSH_COMMANDS_BIT is only a flush hint — the result is the
      // same for a synchronous renderer).
      return s._signaled ? C2.ALREADY_SIGNALED : C2.TIMEOUT_EXPIRED;
    };
  }

  if ('waitSync' in proto) {
    proto.waitSync = function (this: WebGL2RenderingContext, sync: WebGLSync, flags: GLbitfield, timeout: GLuint64): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const s = validateNonNullableObject<WebGLSync>(ctx, sync, Sync.any);
      if (s === null) return; // INVALID_OPERATION pushed
      if (flags !== 0 && flags !== C2.SYNC_FLUSH_COMMANDS_BIT) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (timeout > ctx._state.limits.MAX_CLIENT_WAIT_TIMEOUT_WEBGL) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // No-op: synchronous renderer has nothing to wait for.
    };
  }

  if ('getSyncParameter' in proto) {
    proto.getSyncParameter = function (this: WebGL2RenderingContext, sync: WebGLSync, pname: GLenum): any {
      const ctx = this;
      if (isLost(ctx)) return null;
      const s = validateNonNullableObject<WebGLSync>(ctx, sync, Sync.any);
      if (s === null) return null; // INVALID_OPERATION pushed
      switch (pname) {
        case C2.OBJECT_TYPE:
          return C2.SYNC_FENCE; // fenceSync objects always have type SYNC_FENCE
        case C2.SYNC_STATUS:
          return s._signaled ? C2.SIGNALED : C2.UNSIGNALED;
        case C2.SYNC_CONDITION:
          return s._condition;
        case C2.SYNC_FLAGS:
          return s._flags;
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
      }
    };
  }

  // ---- Samplers ----
  if ('createSampler' in proto) {
    proto.createSampler = function (this: WebGL2RenderingContext): WebGLSampler | null {
      const ctx = this;
      // No [WebGLHandlesContextLoss]: while lost it still creates an object
      // (CTS context-lost.html nonNullTests WebGL2 branch) with NO error.
      return createObject(ctx, Sampler.make); // _params defaults set in the class
    };
  }

  if ('deleteSampler' in proto) {
    proto.deleteSampler = function (this: WebGL2RenderingContext, sampler: WebGLSampler | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (sampler === null || sampler === undefined) return;
      if (!(sampler instanceof WebGLSampler)) {
        throw new TypeError(`Argument is not of type 'WebGLSampler'`);
      }
      if (sampler._context !== ctx) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (sampler._deleted) return;
      // Unbind from every texture unit (spec: delete while bound → unbound).
      const units = ctx._state.textureUnits;
      for (let i = 0; i < units.length; i++) {
        if (units[i].sampler === sampler) {
          units[i].sampler = null;
          refreshUnitSamplerBindings(ctx._state, i);
        }
      }
      sampler._deleted = true;
      ctx._resources.untrack(sampler);
    };
  }

  if ('isSampler' in proto) {
    proto.isSampler = function (this: WebGL2RenderingContext, sampler: WebGLSampler | null): GLboolean {
      const ctx = this;
      if (isLost(ctx)) return false;
      if (sampler === null || sampler === undefined) return false;
      if (!(sampler instanceof WebGLSampler)) {
        throw new TypeError(`Argument is not of type 'WebGLSampler'`);
      }
      if (sampler._context !== ctx) return false; // cross-context: false, NO error
      // CTS samplers.html: true even if never bound.
      return !sampler._deleted;
    };
  }

  if ('bindSampler' in proto) {
    proto.bindSampler = function (this: WebGL2RenderingContext, unit: GLuint, sampler: WebGLSampler | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const s = ctx._state;
      if (unit < 0 || unit >= s.limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (sampler === null || sampler === undefined) {
        s.textureUnits[unit].sampler = null;
        refreshUnitSamplerBindings(s, unit);
        return;
      }
      const smp = validateObject<WebGLSampler>(ctx, sampler, Sampler.any);
      if (smp === null) return; // cross-context/deleted → INVALID_OPERATION pushed
      s.textureUnits[unit].sampler = smp;
      // Keep the completeness engine's texture→sampler association current
      // (sampler params replace the texture's for the sampling-completeness
      // decision — conformance2/samplers/sampler-drawing-test.html).
      refreshUnitSamplerBindings(s, unit);
    };
  }

  if ('samplerParameterf' in proto) {
    proto.samplerParameterf = function (this: WebGL2RenderingContext, sampler: WebGLSampler, pname: GLenum, param: GLfloat): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const smp = validateNonNullableObject<WebGLSampler>(ctx, sampler, Sampler.any);
      if (smp === null) return;
      setSamplerParam(ctx, smp, pname, param);
    };
  }

  if ('samplerParameteri' in proto) {
    proto.samplerParameteri = function (this: WebGL2RenderingContext, sampler: WebGLSampler, pname: GLenum, param: GLint): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const smp = validateNonNullableObject<WebGLSampler>(ctx, sampler, Sampler.any);
      if (smp === null) return;
      setSamplerParam(ctx, smp, pname, param);
    };
  }

  if ('getSamplerParameter' in proto) {
    proto.getSamplerParameter = function (this: WebGL2RenderingContext, sampler: WebGLSampler, pname: GLenum): any {
      const ctx = this;
      if (isLost(ctx)) return null;
      const smp = validateNonNullableObject<WebGLSampler>(ctx, sampler, Sampler.any);
      if (smp === null) return null; // INVALID_OPERATION pushed
      switch (pname) {
        case C1.TEXTURE_MIN_FILTER:
        case C1.TEXTURE_MAG_FILTER:
        case C1.TEXTURE_WRAP_S:
        case C1.TEXTURE_WRAP_T:
        case C2.TEXTURE_WRAP_R:
        case C2.TEXTURE_COMPARE_MODE:
        case C2.TEXTURE_COMPARE_FUNC:
          return smp._params[pname]; // int
        case C2.TEXTURE_MIN_LOD:
        case C2.TEXTURE_MAX_LOD:
          return smp._params[pname]; // float
        case CExt.TEXTURE_MAX_ANISOTROPY_EXT:
          if (!anisotropyEnabled(ctx)) {
            ctx._errors.push(C1.INVALID_ENUM);
            return null;
          }
          return smp._params[pname]; // float
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
      }
    };
  }

  // ---- VAOs ----
  if ('createVertexArray' in proto) {
    proto.createVertexArray = function (this: WebGL2RenderingContext): WebGLVertexArrayObject | null {
      const ctx = this;
      // No [WebGLHandlesContextLoss]: while lost it still creates an object
      // (CTS context-lost.html nonNullTests WebGL2 branch) with NO error;
      // isVertexArray on it → false while lost (isLost guard).
      const vao = createObject(ctx, Vao.make);
      vao._vao = defaultVAOState(ctx._state.limits.MAX_VERTEX_ATTRIBS);
      return vao;
    };
  }

  if ('deleteVertexArray' in proto) {
    proto.deleteVertexArray = function (this: WebGL2RenderingContext, vertexArray: WebGLVertexArrayObject | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (vertexArray === null || vertexArray === undefined) return;
      if (!(vertexArray instanceof WebGLVertexArrayObject)) {
        throw new TypeError(`Argument is not of type 'WebGLVertexArrayObject'`);
      }
      if (vertexArray._context !== ctx) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (vertexArray._deleted) return;
      if (ctx._state.vaoBinding === vertexArray) {
        // Deleting the bound VAO unbinds it — the default VAO becomes bound
        // (CTS vertex-array-object.html).
        ctx._state.vaoBinding = null;
        ctx._state.vao = defaultVAO(ctx);
        // CURRENT_VERTEX_ATTRIB is context-global: the fresh default VAO's
        // attrib mirrors must carry the global generic values so draws keep
        // using them ("should be green" after deleteVertexArray).
        syncCurrentAttribs(ctx, ctx._state.vao);
      }
      vertexArray._deleted = true;
      ctx._resources.untrack(vertexArray);
    };
  }

  if ('isVertexArray' in proto) {
    proto.isVertexArray = function (this: WebGL2RenderingContext, vertexArray: WebGLVertexArrayObject | null): GLboolean {
      const ctx = this;
      if (isLost(ctx)) return false;
      if (vertexArray === null || vertexArray === undefined) return false;
      if (!(vertexArray instanceof WebGLVertexArrayObject)) {
        throw new TypeError(`Argument is not of type 'WebGLVertexArrayObject'`);
      }
      if (vertexArray._context !== ctx) return false; // cross-context: false, NO error
      // CTS vertex-array-object.html: false until first bind.
      return !vertexArray._deleted && everBoundVAOs.has(vertexArray);
    };
  }

  if ('bindVertexArray' in proto) {
    proto.bindVertexArray = function (this: WebGL2RenderingContext, array: WebGLVertexArrayObject | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const s = ctx._state;
      if (array === null || array === undefined) {
        s.vaoBinding = null;
        s.vao = defaultVAO(ctx);
        // CURRENT_VERTEX_ATTRIB is context-global, not VAO state: the newly
        // bound VAO's attrib mirrors must carry the global generic values
        // (CTS vertex-array-object.html runAttributeValueTests).
        syncCurrentAttribs(ctx, s.vao);
        return;
      }
      const vao = validateObject<WebGLVertexArrayObject>(ctx, array, Vao.any);
      if (vao === null) return; // cross-context/deleted → INVALID_OPERATION pushed
      s.vaoBinding = vao;
      s.vao = vao._vao;
      syncCurrentAttribs(ctx, s.vao);
      everBoundVAOs.add(vao);
    };
  }

  // ---- Transform feedback ----
  if ('createTransformFeedback' in proto) {
    proto.createTransformFeedback = function (this: WebGL2RenderingContext): WebGLTransformFeedback | null {
      const ctx = this;
      // No [WebGLHandlesContextLoss]: while lost it still creates an object
      // (CTS context-lost.html nonNullTests WebGL2 branch) with NO error.
      const tf = createObject(ctx, Tf.make);
      const n = ctx._state.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
      tf._buffers = new Array<WebGLBuffer | null>(n).fill(null);
      tf._bufferRanges = Array.from({ length: n }, () => ({ offset: 0, size: 0, base: false }));
      return tf;
    };
  }

  if ('deleteTransformFeedback' in proto) {
    proto.deleteTransformFeedback = function (this: WebGL2RenderingContext, transformFeedback: WebGLTransformFeedback | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (transformFeedback === null || transformFeedback === undefined) return;
      if (!(transformFeedback instanceof WebGLTransformFeedback)) {
        throw new TypeError(`Argument is not of type 'WebGLTransformFeedback'`);
      }
      if (transformFeedback._context !== ctx) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (transformFeedback._deleted) return;
      if (transformFeedback._active) {
        // CTS transform_feedback.html: deleting an active TF fails, no effect.
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (ctx._state.transformFeedback === transformFeedback) {
        ctx._state.transformFeedback = null;
      }
      transformFeedback._deleted = true;
      ctx._resources.untrack(transformFeedback);
    };
  }

  if ('isTransformFeedback' in proto) {
    proto.isTransformFeedback = function (this: WebGL2RenderingContext, transformFeedback: WebGLTransformFeedback | null): GLboolean {
      const ctx = this;
      if (isLost(ctx)) return false;
      if (transformFeedback === null || transformFeedback === undefined) return false;
      if (!(transformFeedback instanceof WebGLTransformFeedback)) {
        throw new TypeError(`Argument is not of type 'WebGLTransformFeedback'`);
      }
      if (transformFeedback._context !== ctx) return false; // cross-context: false, NO error
      // CTS transform_feedback.html: false until first bind.
      return !transformFeedback._deleted && everBoundTFs.has(transformFeedback);
    };
  }

  if ('bindTransformFeedback' in proto) {
    proto.bindTransformFeedback = function (this: WebGL2RenderingContext, target: GLenum, transformFeedback: WebGLTransformFeedback | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (target !== C2.TRANSFORM_FEEDBACK) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const s = ctx._state;
      // Cannot rebind while the BOUND TF is active and not paused (CTS
      // switching-objects.html: binding while active+paused is legal).
      if (s.transformFeedback && s.transformFeedback._active && !s.transformFeedback._paused) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (transformFeedback === null || transformFeedback === undefined) {
        // Bind the default transform feedback object. It stays OUT of
        // state.transformFeedback so getParameter(TRANSFORM_FEEDBACK_BINDING)
        // reports null (CTS transform_feedback.html / gl-get-calls.html) —
        // EXCEPT while a default-TF session is active-paused: keep the default
        // object exposed so the draw engine keeps capturing after
        // resumeTransformFeedback (binding the default during a paused default
        // session is a no-op bind per GLES 3.0 §2.15).
        const def = defaultTFs.get(ctx);
        s.transformFeedback = def && def._active ? def : null;
        return;
      }
      const tf = validateObject<WebGLTransformFeedback>(ctx, transformFeedback, Tf.any);
      if (tf === null) return; // cross-context/deleted → INVALID_OPERATION pushed
      s.transformFeedback = tf;
      everBoundTFs.add(tf);
      // Initialize only a NEVER-bound object's indexed bindings from the
      // global mirror (once); a previously-bound object keeps its own
      // per-object bindings (GLES 3.0 §2.15 / §6.24 — CTS switching-objects.html).
      syncTfBuffers(ctx, tf);
    };
  }

  if ('beginTransformFeedback' in proto) {
    proto.beginTransformFeedback = function (this: WebGL2RenderingContext, primitiveMode: GLenum): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (primitiveMode !== C1.POINTS && primitiveMode !== C1.LINES && primitiveMode !== C1.TRIANGLES) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const s = ctx._state;
      const bound = s.transformFeedback;
      if (bound && bound._active) {
        // Only the BOUND TF blocks a new begin (CTS switching-objects.html
        // begins a different TF while another is active-paused).
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const program = s.currentProgram;
      if (program !== null) ensureProgramLinked(ctx, program); // KHR: finish any deferred link
      if (program === null || !program._linkStatus) {
        ctx._errors.push(C1.INVALID_OPERATION); // no linked program in use
        return;
      }
      const varyings = program._transformFeedbackVaryings;
      if (!varyings || varyings.length === 0) {
        ctx._errors.push(C1.INVALID_OPERATION); // CTS runNoOutputsTest
        return;
      }
      const tf = bound !== null ? bound : getDefaultTF(ctx);
      // Initialize the begun object's indexed bindings (first bind of a named
      // object from the mirror; the default object always from the mirror),
      // then count what THIS object has bound (GLES 3.0 §2.15: the buffers of
      // the transform feedback object being begun — per-object state, not the
      // global mirror; the default object's bindings ARE the mirror).
      syncTfBuffers(ctx, tf);
      // GLES 3.0 §2.15.2 begin-time binding validation (CTS
      // same-buffer-two-binding-points.html asserts the error IMMEDIATELY after
      // beginTransformFeedback): the same buffer bound to two or more of THIS
      // object's indexed binding points generates INVALID_OPERATION in EVERY
      // mode (bindBufferBase case, bindBufferRange overlapping AND
      // non-overlapping ranges); a bound point with no corresponding captured
      // varying is an error in SEPARATE_ATTRIBS (varying i ↔ binding point i).
      // INTERLEAVED_ATTRIBS ignores binding points ≥ 1 — leftover bindings are
      // legal (CTS too-small-buffers.html leaves index 1 bound from its
      // separate sections across interleaved sections and expects NO_ERROR).
      // Mirrors the draw-side tfBindingConflict check in draw.ts.
      const n = s.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
      for (let i = 0; i < n; i++) {
        const b = tf._buffers[i];
        if (!b) continue;
        for (let j = i + 1; j < n; j++) {
          if (tf._buffers[j] === b) {
            ctx._errors.push(C1.INVALID_OPERATION);
            return;
          }
        }
      }
      if (program._tfBufferMode === C2.SEPARATE_ATTRIBS) {
        for (let i = varyings.length; i < n; i++) {
          if (tf._buffers[i]) {
            ctx._errors.push(C1.INVALID_OPERATION);
            return;
          }
        }
      }
      const needed = program._tfBufferMode === C2.SEPARATE_ATTRIBS ? varyings.length : 1;
      let boundCount = 0;
      for (let i = 0; i < s.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS; i++) {
        if (tf._buffers[i] !== null) boundCount++;
      }
      if (boundCount < needed) {
        ctx._errors.push(C1.INVALID_OPERATION); // insufficient TF buffers bound
        return;
      }
      if (bound === null) {
        // Default-TF session: expose the default object through
        // state.transformFeedback so the draw engine's tfActive check
        // (`!!s.transformFeedback && tf._active && !tf._paused`, draw.ts)
        // captures into it. getParameter(TRANSFORM_FEEDBACK_BINDING) reads the
        // same field and reports the object mid-session — no CTS page queries
        // it during a default session (all null expectations in
        // transform_feedback.html / gl-get-calls.html are outside sessions;
        // endTransformFeedback restores null below).
        s.transformFeedback = tf;
      }
      tf._program = program;
      tf._active = true;
      tf._paused = false;
      tf._primitiveMode = primitiveMode;
      tf._primitivesWritten = 0; // counter starts at begin (draw engine accumulates)
    };
  }

  if ('endTransformFeedback' in proto) {
    proto.endTransformFeedback = function (this: WebGL2RenderingContext): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const tf = activeTF(ctx);
      if (tf === null) {
        ctx._errors.push(C1.INVALID_OPERATION); // not active
        return;
      }
      const isDefaultSession = tf === defaultTFs.get(ctx);
      tf._active = false;
      tf._paused = false;
      tf._program = null; // ES3: EndTransformFeedback resets the TF program binding
      tf._primitivesWritten = 0; // ES3: primitives-written counter resets on deactivation
      if (isDefaultSession && ctx._state.transformFeedback === tf) {
        // Restore the unbound state: TRANSFORM_FEEDBACK_BINDING → null (CTS
        // transform_feedback.html expects null after binding the default).
        ctx._state.transformFeedback = null;
      }
    };
  }

  if ('pauseTransformFeedback' in proto) {
    proto.pauseTransformFeedback = function (this: WebGL2RenderingContext): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const tf = activeTF(ctx);
      if (tf === null) {
        ctx._errors.push(C1.INVALID_OPERATION); // not active
        return;
      }
      if (tf._paused) {
        ctx._errors.push(C1.INVALID_OPERATION); // double-pause (CTS)
        return;
      }
      tf._paused = true;
    };
  }

  if ('resumeTransformFeedback' in proto) {
    proto.resumeTransformFeedback = function (this: WebGL2RenderingContext): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const tf = activeTF(ctx);
      if (tf === null) {
        ctx._errors.push(C1.INVALID_OPERATION); // not active
        return;
      }
      if (!tf._paused) {
        ctx._errors.push(C1.INVALID_OPERATION); // not paused (CTS)
        return;
      }
      tf._paused = false;
    };
  }

  // ---- getInternalformatParameter ----
  if ('getInternalformatParameter' in proto) {
    proto.getInternalformatParameter = function (this: WebGL2RenderingContext, target: GLenum, internalformat: GLenum, pname: GLenum): any {
      const ctx = this;
      if (isLost(ctx)) return null;
      if (target !== C1.RENDERBUFFER) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      if (pname !== C1.SAMPLES) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      if (!RENDERBUFFER_FORMATS.has(internalformat) && !extRenderbufferFormatOK(ctx, internalformat)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      // Supported sample counts — [4] is consistent with MAX_SAMPLES=4 and the
      // CTS multisample-with-full-sample-counts test (samples 1..array[0] must
      // be accepted by renderbufferStorageMultisample).
      return new Int32Array([4]);
    };
  }

  // ---- drawingBufferStorage (WebGL 1.0 IDL base method — WebGL2 prototype) ----
  // The class-body declaration is added in parallel (webgl2.ts); install via a
  // cast so this module is self-contained. Semantics (WebGL 1.0 spec
  // "drawingBufferStorage"): respecify the drawing buffer size + format;
  // alpha:false → INVALID_OPERATION; unsupported sizedFormat → INVALID_ENUM;
  // width/height > MAX_RENDERBUFFER_SIZE → INVALID_VALUE; allocation failure →
  // OUT_OF_MEMORY. On success drawingBufferFormat/Width/Height reflect the
  // request and the buffer is cleared (spec: clearing behavior equivalent to
  // setting HTMLCanvasElement.width/height).
  if ('beginQuery' in proto) {
    (proto as unknown as {
      drawingBufferStorage(sizedFormat: GLenum, width: GLsizei, height: GLsizei): void;
    }).drawingBufferStorage = function (this: WebGL2RenderingContext, sizedFormat: GLenum, width: GLsizei, height: GLsizei): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const s = ctx._state;
      if (width > s.limits.MAX_RENDERBUFFER_SIZE || height > s.limits.MAX_RENDERBUFFER_SIZE) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (ctx._attrs.alpha === false) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // Supported sized formats (spec): RGBA8 always; SRGB8_ALPHA8 (WebGL2
      // core, WebGL1 needs EXT_sRGB); RGBA16F (WebGL2 needs
      // EXT_color_buffer_float, WebGL1 EXT_color_buffer_half_float).
      if (sizedFormat !== C2.RGBA8 && sizedFormat !== C2.SRGB8_ALPHA8 && sizedFormat !== C2.RGBA16F) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      if (sizedFormat === C2.RGBA16F) {
        // RGBA16F must be "currently valid for renderbufferStorage", i.e. the
        // enabling extension must already be ENABLED (a prior getExtension
        // call) — WebGL1: EXT_color_buffer_half_float; WebGL2:
        // EXT_color_buffer_float. Deliberately NOT ctx.getExtension(...):
        // calling it here would ENABLE the extension and break the CTS
        // expectation (drawingbuffer-storage-test testMissingExtension) that
        // RGBA16F without a prior enable is INVALID_ENUM.
        const ext = ctx._version === 2 ? 'EXT_color_buffer_float' : 'EXT_color_buffer_half_float';
        if (!ctx._extensions.has(ext)) {
          ctx._errors.push(C1.INVALID_ENUM);
          return;
        }
      }
      const w = Math.max(1, width);
      const h = Math.max(1, height);
      try {
        // Canvas attribute mirror: the spec's clearing behavior is "equivalent
        // to setting HTMLCanvasElement.width/height", and the draw engine's
        // ensureCanvasSize keeps the drawing buffer in sync with the canvas
        // dims (it reallocates + clears on mismatch) — mirror the storage size
        // there so the next draw/clear does not reallocate the buffer back to
        // the old canvas size. (No CTS page observes canvas.width/height
        // across a drawingBufferStorage call.)
        const canvas = ctx._canvas as { width: number; height: number };
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        // Present surface: resize BEFORE the color surface aliases its pixel
        // buffer (a fresh surface is 0×0 — its first present() would
        // auto-resize to the canvas dims, orphaning the alias).
        let pixels: Uint8Array | null = null;
        if (ctx._presentSurface) {
          try {
            (ctx._presentSurface as { resize?: (w: number, h: number) => void }).resize?.(w, h);
            pixels = ctx._presentSurface.getPixels();
          } catch {
            pixels = null; // present stub — local surface fallback below
          }
        }
        const base = getFormat(sizedFormat);
        let color: Surface;
        if (sizedFormat === C2.RGBA16F) {
          // Float storage cannot alias the RGBA8 present buffer — allocate a
          // real f32 surface (readPixels RGBA/FLOAT from the default buffer and
          // canvas presentation of float buffers are draw.ts/present concerns).
          color = createSurface(sizedFormat, w, h);
        } else if (base && pixels && pixels.length >= w * h * 4) {
          color = { width: w, height: h, format: sizedFormat, info: base, data: pixels };
          if (sizedFormat === C2.SRGB8_ALPHA8) {
            // sRGB store conversion: the raster fragment path converts
            // linear→sRGB before info.encode, but the CLEAR path
            // (clearColorSurface) encodes raw. Wrap encode so glClear stores
            // sRGB-encoded bytes (CTS expects clear(linear values) →
            // readPixels/canvas show sRGB bytes), and drop isSRGB so the
            // fragment path's own conversion does not double-apply. (Blending
            // onto this buffer then treats stored values as linear — no CTS
            // coverage for blending onto the sRGB drawing buffer.)
            const enc = base.encode;
            color.info = {
              ...base,
              isSRGB: false,
              encode(src, byteOffset, r, g, b, a) {
                enc(src, byteOffset, linearToSRGB(r), linearToSRGB(g), linearToSRGB(b), a);
              },
            };
          }
        } else {
          color = createSurface(sizedFormat, w, h);
        }
        let depth: Surface | null = null;
        if (ctx._attrs.depth) {
          try {
            depth = createSurface(ctx._version === 2 ? C2.DEPTH_COMPONENT24 : C1.DEPTH_COMPONENT16, w, h);
            (depth.data as Float32Array).fill(1.0); // cleared state
          } catch {
            depth = null;
          }
        }
        let stencil: Surface | null = null;
        if (ctx._attrs.stencil) {
          try {
            stencil = createSurface(C1.STENCIL_INDEX8, w, h);
          } catch {
            stencil = null;
          }
        }
        ctx._defaultFB = { color, depth, stencil, width: w, height: h };
        ctx._drawingBufferWidth = w;
        ctx._drawingBufferHeight = h;
        (ctx as unknown as { _drawingBufferFormat: number })._drawingBufferFormat = sizedFormat;
        try {
          ctx._presentSurface?.present(); // show the cleared frame (canvas tests)
        } catch {
          /* present stub */
        }
      } catch {
        ctx._errors.push(C1.OUT_OF_MEMORY);
      }
    };

    // drawingBufferFormat/Width/Height: the base-class getters (webgl1.ts) are
    // LIVE canvas-dimension getters and hardcode RGBA8. After a
    // drawingBufferStorage call the buffer's size/format no longer matches the
    // canvas attributes — shadow them with the storage-tracked values. Before
    // any storage call: live canvas semantics, except drawingBufferFormat
    // reports RGB8 for alpha:false buffers (spec; CTS drawingbuffer-storage-test).
    Object.defineProperty(proto, 'drawingBufferFormat', {
      configurable: true,
      get(this: WebGL2RenderingContext): GLenum {
        const c = this as unknown as { _drawingBufferFormat?: number; _attrs: { alpha?: boolean } };
        return c._drawingBufferFormat ?? (c._attrs.alpha === false ? C2.RGB8 : C2.RGBA8);
      },
    });
    Object.defineProperty(proto, 'drawingBufferWidth', {
      configurable: true,
      get(this: WebGL2RenderingContext): GLsizei {
        const c = this as unknown as { _drawingBufferFormat?: number; _drawingBufferWidth: number };
        if (c._drawingBufferFormat !== undefined) return c._drawingBufferWidth;
        const max = this._state.limits.MAX_VIEWPORT_DIMS[0];
        const d = typeof this._canvas.width === 'number' ? Math.max(1, this._canvas.width) : 0;
        return max > 0 && d > max ? max : d;
      },
    });
    Object.defineProperty(proto, 'drawingBufferHeight', {
      configurable: true,
      get(this: WebGL2RenderingContext): GLsizei {
        const c = this as unknown as { _drawingBufferFormat?: number; _drawingBufferHeight: number };
        if (c._drawingBufferFormat !== undefined) return c._drawingBufferHeight;
        const max = this._state.limits.MAX_VIEWPORT_DIMS[1];
        const d = typeof this._canvas.height === 'number' ? Math.max(1, this._canvas.height) : 0;
        return max > 0 && d > max ? max : d;
      },
    });
  }

  // ---- clearBufferiv / clearBufferuiv ----
  // Overrides of the api/draw.ts installs: installWebGL2Api runs AFTER
  // installDrawApi in installAll, so the assignments below win on the WebGL2
  // prototype. Per the WebGL 2.0 spec (§4.2.2) the buffer argument is COLOR
  // for every color variant — the FUNCTION selects the interpretation
  // (signed integer → clearBufferiv, unsigned integer → clearBufferuiv,
  // float/fixed → clearBufferfv). The api/draw.ts versions only accepted the
  // GLES COLOR_INT/COLOR_UINT enums (0x8b8f/0x8b90), so
  // gl.clearBufferiv(gl.COLOR, …) — the CTS usage — raised INVALID_OPERATION
  // and never cleared. Also fixed here: the drawbuffer index is mapped through
  // the current drawBuffers() state (slot → attachment; NONE → no-op, no
  // error; missing attachment → no-op), and a type mismatch between the
  // function and the mapped attachment raises INVALID_OPERATION.
  if ('clearBufferiv' in proto) {
    proto.clearBufferiv = function (this: WebGL2RenderingContext, buffer: GLenum, drawbuffer: GLint, values: Int32List): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const v = toListLocal(values, Int32Array, 'Int32List');
      if (buffer !== C2.COLOR && buffer !== C2.DEPTH && buffer !== C2.STENCIL) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const s = ctx._state;
      if (drawbuffer < 0 || drawbuffer >= s.limits.MAX_DRAW_BUFFERS) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (buffer !== C2.COLOR && drawbuffer !== 0) {
        ctx._errors.push(C1.INVALID_OPERATION); // depth/stencil only clear drawbuffer 0
        return;
      }
      // Optional 4th arg srcOffset (u64; see api/draw.ts clearBufferfv).
      const srcOffset = toU64(arguments[3]);
      if (srcOffset + (buffer === C2.COLOR ? 4 : 1) > v.length) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (buffer === C2.COLOR) {
        const db = s.drawBuffers[drawbuffer] ?? C1.NONE;
        if (db === C1.NONE) return; // DRAW_BUFFERi = NONE → nothing cleared, no error
        const info = clearColorAttachmentInfo(ctx, db - C1.COLOR_ATTACHMENT0);
        if (info && (!info.isInteger || !info.isSigned)) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
      }
      try {
        executeClearBuffer(ctx, buffer, drawbuffer, v, srcOffset);
      } catch {
        ctx._errors.push(C1.INVALID_OPERATION);
      }
    };
  }

  if ('clearBufferuiv' in proto) {
    proto.clearBufferuiv = function (this: WebGL2RenderingContext, buffer: GLenum, drawbuffer: GLint, values: Uint32List): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const v = toListLocal(values, Uint32Array, 'Uint32List');
      if (buffer !== C2.COLOR) {
        ctx._errors.push(C1.INVALID_ENUM); // unsigned color only
        return;
      }
      const s = ctx._state;
      if (drawbuffer < 0 || drawbuffer >= s.limits.MAX_DRAW_BUFFERS) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      // Optional 4th arg srcOffset (u64; see api/draw.ts clearBufferfv).
      const srcOffset = toU64(arguments[3]);
      if (srcOffset + 4 > v.length) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      const db = s.drawBuffers[drawbuffer] ?? C1.NONE;
      if (db === C1.NONE) return; // DRAW_BUFFERi = NONE → nothing cleared, no error
      const info = clearColorAttachmentInfo(ctx, db - C1.COLOR_ATTACHMENT0);
      if (info && (!info.isInteger || info.isSigned)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      try {
        executeClearBuffer(ctx, C2.COLOR, drawbuffer, v, srcOffset);
      } catch {
        ctx._errors.push(C1.INVALID_OPERATION);
      }
    };
  }
}

/** Shared samplerParameterf/i validation + store (errors pushed by the caller's ctx). */
function setSamplerParam(ctx: WebGLRenderingContext, sampler: WebGLSampler, pname: number, param: number): void {
  switch (pname) {
    case C1.TEXTURE_MIN_FILTER:
      if (!MIN_FILTER_VALUES.has(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      break;
    case C1.TEXTURE_MAG_FILTER:
      if (!MAG_FILTER_VALUES.has(param)) {
        ctx._errors.push(C1.INVALID_ENUM); // mipmap filters illegal for MAG (CTS)
        return;
      }
      break;
    case C1.TEXTURE_WRAP_S:
    case C1.TEXTURE_WRAP_T:
    case C2.TEXTURE_WRAP_R:
      if (!WRAP_VALUES.has(param)) {
        ctx._errors.push(C1.INVALID_ENUM); // e.g. CLAMP_TO_BORDER (CTS)
        return;
      }
      break;
    case C2.TEXTURE_MIN_LOD:
    case C2.TEXTURE_MAX_LOD:
      break; // any value accepted
    case C2.TEXTURE_COMPARE_MODE:
      if (!COMPARE_MODE_VALUES.has(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      break;
    case C2.TEXTURE_COMPARE_FUNC:
      if (!COMPARE_FUNC_VALUES.has(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      break;
    case CExt.TEXTURE_MAX_ANISOTROPY_EXT:
      if (!anisotropyEnabled(ctx)) {
        ctx._errors.push(C1.INVALID_ENUM); // extension not enabled
        return;
      }
      if (param < 1) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      break;
    default:
      ctx._errors.push(C1.INVALID_ENUM); // e.g. BASE_LEVEL/MAX_LEVEL (CTS samplers.html)
      return;
  }
  sampler._params[pname] = param;
}
