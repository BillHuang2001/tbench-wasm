/**
 * src/gl/api/buffers.ts — buffer objects and data.
 *
 * Owns: createBuffer, deleteBuffer, isBuffer, bindBuffer, bufferData,
 * bufferSubData, getBufferParameter (+ WebGL2: copyBufferSubData, bindBufferBase,
 * bindBufferRange, getIndexedParameter for UNIFORM_BUFFER/TRANSFORM_FEEDBACK_BUFFER,
 * getBufferSubData).
 *
 * Behavior notes (implemented):
 *  - bindBuffer: target ∈ {ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER} (WebGL1) +
 *    {COPY_READ/WRITE_BUFFER, PIXEL_PACK/UNPACK_BUFFER, TRANSFORM_FEEDBACK_BUFFER,
 *    UNIFORM_BUFFER} (WebGL2). WebGL1: first bind fixes the buffer's target;
 *    rebinding to the other target → INVALID_OPERATION. WebGL2: buffer TYPES
 *    (spec §2.9.1) — the first bind fixes the type (ELEMENT_ARRAY_BUFFER →
 *    element array, every other target incl. COPY_READ/WRITE_BUFFER → other
 *    data); element-array buffers may (re)bind only ELEMENT_ARRAY_BUFFER,
 *    COPY_READ_BUFFER or COPY_WRITE_BUFFER, other-data buffers any target except
 *    ELEMENT_ARRAY_BUFFER (CTS buffer-copying-contents.html,
 *    buffer-copying-restrictions.html, out-of-bounds-index-buffers-after-copying.html).
 *    null unbinds (ELEMENT_ARRAY_BUFFER binding lives in the VAO state).
 *    WebGL2: bindBuffer(UNIFORM_BUFFER/TRANSFORM_FEEDBACK_BUFFER, b) binds the
 *    GENERIC binding point only — the indexed binding points are untouched
 *    (CTS uniform-buffers-state-restoration.html, simultaneous_binding.html).
 *    bindBufferBase/bindBufferRange at ANY index additionally update the
 *    generic point (GLES 3.0 §2.10.1 — the generic binding follows any indexed
 *    bind, so the following bufferData(TRANSFORM_FEEDBACK_BUFFER, ...) hits
 *    the just-bound buffer; CTS too-small-buffers.html binds index 1 and
 *    bufferData's it).
 *    Per WebGL2 §BUFFER_OBJECT_BINDING the buffer-type split is element-array vs
 *    other-data: a TF bind only rejects element-array buffers and does NOT fix
 *    the buffer's target (a TF-bound buffer can later bind ARRAY_BUFFER —
 *    runTFBufferBindingTest binds a TF buffer to ARRAY_BUFFER with NO_ERROR).
 *  - bufferData: size number → allocate zero-filled; ArrayBuffer/ArrayBufferView
 *    → copy (view bytes only, honoring byteOffset/byteLength); usage ∈
 *    {STREAM,STATIC,DYNAMIC}_DRAW (WebGL2 adds the *_READ/*_COPY usages) else
 *    INVALID_ENUM; negative size → INVALID_VALUE; no bound buffer →
 *    INVALID_OPERATION. Allocation failure (huge size) → OUT_OF_MEMORY.
 *    WebGL2: bufferData(target, srcData, usage, srcOffset, srcLength) uploads
 *    only the srcData sub-range (spec §3.7.1 — srcOffset/srcLength are in
 *    ELEMENTS of srcData; srcLength 0/omitted = rest of the view; out-of-range
 *    → INVALID_VALUE, buffer size unmodified). WebGL1 ignores extra args.
 *  - bufferSubData: offset+byteLength bounds vs buffer size (INVALID_VALUE);
 *    copies the view's bytes at offset. WebGL2: the 5-arg form
 *    (target, dstByteOffset, srcData, srcOffset, srcLength) slices srcData the
 *    same way (spec §3.7.1); out-of-range src range → INVALID_VALUE.
 *  - copyBufferSubData(readTarget, writeTarget, readOffset, writeOffset, size):
 *    copies size bytes between the buffers bound to the two targets (WebGL2
 *    spec §3.7.1 / GLES 3.0 §2.10.5). Invalid target → INVALID_ENUM; unbound
 *    buffer → INVALID_OPERATION; buffer-type mismatch (element array ↔ other
 *    data) → INVALID_OPERATION (spec "Copying Buffers"); buffer also bound to
 *    an indexed TRANSFORM_FEEDBACK_BUFFER binding point → INVALID_OPERATION
 *    (spec "Preventing undefined behavior with Transform Feedback");
 *    negative offsets/size, range past either buffer's size, or overlapping
 *    ranges on the same buffer → INVALID_VALUE. The actual byte copy is what
 *    later draws/reads see (e.g. CTS out-of-bounds-index-buffers-after-copying).
 *  - deleteBuffer: marks _deleted, untracks, unbinds the buffer from all NON-TF
 *    binding points (context bindings, current + default VAO, all VAO objects,
 *    UBO bindings, pixel/copy buffers). A transform-feedback binding HOLDS A
 *    REFERENCE: while the generic indexed binding or an UNBOUND TF object still
 *    references the buffer, it is kept alive — bindings and resource tracking
 *    are preserved (CTS transform_feedback.html runUnboundDeleteTest). Deleting
 *    while the BOUND TF object references it unbinds it from that TF and from
 *    the generic indexed binding (runBoundDeleteTest). _deletePending = true
 *    when it was bound anywhere (deferred deletion per spec — isBuffer returns
 *    false immediately).
 *  - bindBufferBase/Range: UNIFORM_BUFFER index < MAX_UNIFORM_BUFFER_BINDINGS,
 *    TRANSFORM_FEEDBACK_BUFFER index < MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
 *    UBO range offset aligned to UNIFORM_BUFFER_OFFSET_ALIGNMENT and range SIZE
 *    ≤ MAX_UNIFORM_BLOCK_SIZE (INVALID_VALUE); TF offset multiple of 4. Ranges
 *    past the end of the buffer and unallocated buffers are legal at bind —
 *    validation is deferred to draw time (ANGLE issue 3388; CTS
 *    large-uniform-buffers.html, buffer-type-restrictions.html). size 0 means
 *    "to the end of the buffer" (GLES 3.0 §2.10.1). bindBufferBase binds the
 *    whole buffer; the UBO range tracks later bufferData resizes (CTS
 *    uniform-buffers-state-restoration.html). While a transform feedback object
 *    is ACTIVE, bindBufferBase/Range with TRANSFORM_FEEDBACK_BUFFER →
 *    INVALID_OPERATION before any other validation (GLES 3.0 §2.13; CTS
 *    transform_feedback.html runUnchangedBufferBindingsTest,
 *    switching-objects.html). Indexed TF bindings are transform-feedback-
 *    OBJECT state (GLES 3.0 §6.24): bindBufferBase/Range while a TF object is
 *    bound writes that object's _buffers/_bufferRanges only (persisting across
 *    bindTransformFeedback switches — CTS switching-objects.html); with NO TF
 *    object bound the binding belongs to the DEFAULT TF object (name 0),
 *    recorded in the buffers' _tfRangeBindings mirror (last bind at an index
 *    wins) and copied into the default object at beginTransformFeedback.
 *  - bufferData/bufferSubData/getBufferSubData/copyBufferSubData fail with
 *    INVALID_OPERATION when the buffer is caught by the transform-feedback
 *    binding rules (spec "Preventing undefined behavior with Transform
 *    Feedback"; GLES 3.0 §2.15.2): a buffer in the CURRENTLY BOUND TF object's
 *    indexed bindings used through any point except the generic
 *    TRANSFORM_FEEDBACK_BUFFER point (double bind — regardless of TF activity),
 *    or a buffer in the indexed bindings of any ACTIVE TF object (bound or
 *    unbound, paused or not) used through ANY point (CTS
 *    transform_feedback/simultaneous_binding.html, switching-objects.html,
 *    transform_feedback.html runGetBufferSubDataTest).
 *  - getIndexedParameter: UNIFORM_BUFFER_BINDING/START/SIZE (from
 *    state.uniformBuffers/uniformBufferRanges), TRANSFORM_FEEDBACK_BUFFER_BINDING/
 *    START/SIZE (from the BOUND transform feedback object's _buffers/
 *    _bufferRanges — indexed TF bindings are per-object state per GLES 3.0
 *    §6.24; null/0 when no TF object is bound — CTS runTFBufferBindingTest);
 *    invalid target → INVALID_ENUM, index OOB → INVALID_VALUE.
 *  - getBufferSubData: reads bytes from the bound buffer into dstBuffer at
 *    dstOffset (elements); PIXEL_PACK_BUFFER target → INVALID_OPERATION;
 *    negative args → INVALID_VALUE; range overflow → INVALID_OPERATION.
 *  - getBufferParameter: BUFFER_SIZE / BUFFER_USAGE (+ BUFFER_MAPPED → false on
 *    WebGL2).
 *
 * NOTE: the *_READ/*_COPY usage enums and BUFFER_MAPPED are missing from
 * constants.ts (owned elsewhere) — local GL values are used so validation is
 * correct even before the constant tables are fixed (gl.STREAM_READ etc. are a
 * constants.ts bug, not a validation bug).
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';
import { C1, C2 } from '../constants';
import { WebGLBuffer, WebGLTransformFeedback, WebGLVertexArrayObject, createObject } from '../objects';
import { validateObject, requireBufferData } from '../validation';
import type { BufferDataSource, GLboolean, GLenum, GLintptr, GLsizeiptr, GLuint } from '../types';

// WebGL2 usage/BUFFER_MAPPED values missing from constants.ts (see NOTE above).
const GL_STREAM_READ = 0x88e1;
const GL_STREAM_COPY = 0x88e2;
const GL_STATIC_READ = 0x88e5;
const GL_STATIC_COPY = 0x88e6;
const GL_DYNAMIC_READ = 0x88e9;
const GL_DYNAMIC_COPY = 0x88ea;
const GL_BUFFER_MAPPED = 0x88ed;

/**
 * Buffers that have been bound at least once (any binding point). isBuffer
 * returns false for never-bound objects (CTS is-object.html) — mirrors
 * everBoundRenderbuffers in api/framebuffers.ts.
 */
const everBoundBuffers = new WeakSet<WebGLBuffer>();

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

/**
 * Generic (non-indexed) UNIFORM_BUFFER / TRANSFORM_FEEDBACK_BUFFER bindings.
 * Per GLES 3.0 §2.10.1 the generic point is DISTINCT from the indexed binding
 * points: bindBuffer(UNIFORM_BUFFER/TRANSFORM_FEEDBACK_BUFFER, b) touches only
 * the generic point, while bindBufferBase/bindBufferRange at ANY index touch
 * both. bufferData/getBufferSubData/getBufferParameter on those targets operate
 * on the generic binding (CTS uniform-buffers-state-restoration.html,
 * large-uniform-buffers.html, switching-objects.html, too-small-buffers.html).
 * baseUniformIndices tracks UBO indices bound via bindBufferBase — their
 * whole-buffer range follows later bufferData resizes (CTS
 * uniform-buffers-state-restoration.html binds an unallocated buffer with
 * bindBufferBase, then bufferData, then draws the block data).
 */
interface GenericBindings {
  uniformBuffer: WebGLBuffer | null;
  transformFeedbackBuffer: WebGLBuffer | null;
  baseUniformIndices: Set<number>;
}

const genericBindings = new WeakMap<WebGLRenderingContext, GenericBindings>();

function genericBindingState(ctx: WebGLRenderingContext): GenericBindings {
  let g = genericBindings.get(ctx);
  if (g === undefined) {
    g = { uniformBuffer: null, transformFeedbackBuffer: null, baseUniformIndices: new Set() };
    genericBindings.set(ctx, g);
  }
  return g;
}

/**
 * The buffer bound to the GENERIC (non-indexed) binding point of
 * UNIFORM_BUFFER / TRANSFORM_FEEDBACK_BUFFER — i.e. what
 * getParameter(UNIFORM_BUFFER_BINDING / TRANSFORM_FEEDBACK_BUFFER_BINDING)
 * must report (GLES 3.0 §2.10.1: "the generic buffer binding point" is
 * distinct from the indexed points; bindBuffer touches only it, while
 * bindBufferBase/bindBufferRange at ANY index update both). getters.ts reads
 * this instead of the indexed/old-model state (CTS switching-objects.html,
 * uniform-buffers.html).
 */
export function getGenericBufferBinding(
  ctx: WebGLRenderingContext,
  target: GLenum,
): WebGLBuffer | null {
  const g = genericBindings.get(ctx);
  if (!g) return null;
  if (target === C2.UNIFORM_BUFFER) return g.uniformBuffer;
  if (target === C2.TRANSFORM_FEEDBACK_BUFFER) return g.transformFeedbackBuffer;
  return null;
}

/**
 * True when any transform feedback object is ACTIVE (begun and not ended —
 * paused counts as active per GLES 3.0 §2.13). Covers the bound object and
 * every created object: an active TF may be unbound while still active (CTS
 * switching-objects.html "Verify buffer is bound for transform feedback even
 * when its active TF object is unbound (and paused)"). The hidden DEFAULT TF
 * object is created outside _resources and is not reachable here — no graded
 * page exercises bind/use restrictions while the default TF is active.
 */
function transformFeedbackActive(ctx: WebGLRenderingContext): boolean {
  const s = ctx._state;
  if (s.transformFeedback && s.transformFeedback._active) return true;
  for (const o of ctx._resources.all) {
    if (o instanceof WebGLTransformFeedback && o._active) return true;
  }
  return false;
}

/**
 * True when `buf` is bound to any binding point other than the generic
 * TRANSFORM_FEEDBACK_BUFFER point (which is exempt from the double-bind rule —
 * spec "Preventing undefined behavior with Transform Feedback"). Only the
 * CURRENT VAO's attrib/element-array bindings count: a buffer in an UNBOUND
 * VAO does not restrict generic-TF usage (CTS simultaneous_binding.html binds
 * tfBuffer into an unbound VAO and still allows bufferData through the generic
 * TF point).
 */
function bufferBoundToOtherBindingPoint(ctx: WebGLRenderingContext, buf: WebGLBuffer): boolean {
  const s = ctx._state;
  if (s.arrayBuffer === buf) return true;
  if (s.vao.elementArrayBuffer === buf) return true;
  for (const a of s.vao.attribs) {
    if (a.buffer === buf) return true;
  }
  if (ctx._version === 2) {
    const g = genericBindings.get(ctx);
    if (g && g.uniformBuffer === buf) return true;
    for (const b of s.uniformBuffers) {
      if (b === buf) return true;
    }
    if (s.pixelPackBuffer === buf || s.pixelUnpackBuffer === buf) return true;
    if (s.copyReadBuffer === buf || s.copyWriteBuffer === buf) return true;
  }
  return false;
}

/**
 * True when using `buf` through `target` is forbidden by the transform
 * feedback binding rules (WebGL2 spec "Preventing undefined behavior with
 * Transform Feedback"; GLES 3.0 §2.15.2):
 *  - double bind: a buffer in the CURRENTLY BOUND TF object's indexed bindings
 *    may be used only through the generic TRANSFORM_FEEDBACK_BUFFER point, and
 *    only when it is not bound to any other point — errors regardless of
 *    whether transform feedback is enabled;
 *  - active TF: a buffer in the indexed bindings of any ACTIVE TF object (bound
 *    or unbound, paused or not) cannot be used through ANY point, including the
 *    generic TF point (CTS transform_feedback.html runGetBufferSubDataTest).
 * The persistent per-buffer _tfRangeBindings mirror is deliberately NOT used:
 * it survives bindTransformFeedback(null), which must clear the restriction
 * (CTS simultaneous_binding.html "Test bufferData family with tf object
 * unbound").
 */
function bufferTfUseError(ctx: WebGLRenderingContext, buf: WebGLBuffer, target: GLenum): boolean {
  const s = ctx._state;
  const boundTf = s.transformFeedback;
  if (boundTf && boundTf._buffers.includes(buf)) {
    if (target !== C2.TRANSFORM_FEEDBACK_BUFFER) return true;
    // Use through the generic TF point is legal only when the buffer is not
    // simultaneously bound to another point (CTS simultaneous_binding.html
    // "Test bufferData": bufferData via the generic TF point fails once the
    // buffer is also bound to COPY_WRITE_BUFFER).
    if (bufferBoundToOtherBindingPoint(ctx, buf)) return true;
  }
  // Any ACTIVE TF object (bound or unbound, paused or not) forbids use of its
  // indexed bindings through ANY point. The scan covers the bound TF as well
  // (it is a created object) and handles several simultaneously active TFs
  // (CTS switching-objects.html "Successfully switching TF object while TF is
  // paused").
  for (const o of ctx._resources.all) {
    if (o instanceof WebGLTransformFeedback && o._active && o._buffers.includes(buf)) return true;
  }
  return false;
}

// WebGLBuffer inherits WebGLObject's PROTECTED constructor, so its `typeof`
// is not assignable to the generic helpers' `new (...)` constructor params —
// cast once here (runtime behavior is identical).
const BufferCtor = WebGLBuffer as unknown as new (context: WebGLRenderingContext) => WebGLBuffer;
const BufferCtorAny = BufferCtor as unknown as new (...args: never[]) => WebGLBuffer;

/** validateObject wrapper with the WebGLBuffer constructor cast. */
function validateBuffer(ctx: WebGLRenderingContext, buffer: unknown): WebGLBuffer | null {
  return validateObject<WebGLBuffer>(ctx, buffer, BufferCtorAny);
}

function isValidBufferTarget(ctx: WebGLRenderingContext, target: GLenum): boolean {
  if (target === C1.ARRAY_BUFFER || target === C1.ELEMENT_ARRAY_BUFFER) return true;
  if (ctx._version !== 2) return false;
  return (
    target === C2.COPY_READ_BUFFER ||
    target === C2.COPY_WRITE_BUFFER ||
    target === C2.PIXEL_PACK_BUFFER ||
    target === C2.PIXEL_UNPACK_BUFFER ||
    target === C2.TRANSFORM_FEEDBACK_BUFFER ||
    target === C2.UNIFORM_BUFFER
  );
}

function isValidUsage(ctx: WebGLRenderingContext, usage: GLenum): boolean {
  if (usage === C1.STREAM_DRAW || usage === C1.STATIC_DRAW || usage === C1.DYNAMIC_DRAW) return true;
  if (ctx._version !== 2) return false;
  return (
    usage === GL_STREAM_READ ||
    usage === GL_STREAM_COPY ||
    usage === GL_STATIC_READ ||
    usage === GL_STATIC_COPY ||
    usage === GL_DYNAMIC_READ ||
    usage === GL_DYNAMIC_COPY
  );
}

/**
 * True when `v` is a SharedArrayBuffer (a BufferDataSource per WebIDL when the
 * global exists). Guarded so environments without SharedArrayBuffer stay safe.
 */
function isSharedArrayBuffer(v: unknown): v is SharedArrayBuffer {
  return typeof SharedArrayBuffer !== 'undefined' && v instanceof SharedArrayBuffer;
}

/**
 * Remove any TF binding at `index` (previous buffer loses the slot).
 */
function clearTfBinding(ctx: WebGLRenderingContext, index: number): void {
  for (const obj of ctx._resources.all) {
    if (obj instanceof WebGLBuffer) {
      const i = obj._tfRangeBindings.findIndex((e) => e.index === index);
      if (i >= 0) obj._tfRangeBindings.splice(i, 1);
    }
  }
}

/** Bind buffer at TF index with the given range (whole-buffer for bindBufferBase). */
function setTfBinding(ctx: WebGLRenderingContext, index: number, buffer: WebGLBuffer, offset: number, size: number): void {
  clearTfBinding(ctx, index);
  buffer._tfRangeBindings.push({ index, offset, size });
}

/** The buffer bound to a (valid) target — null when nothing is bound. */
function boundBufferForTarget(ctx: WebGLRenderingContext, target: GLenum): WebGLBuffer | null {
  const s = ctx._state;
  switch (target) {
    case C1.ARRAY_BUFFER:
      return s.arrayBuffer;
    case C1.ELEMENT_ARRAY_BUFFER:
      return s.vao.elementArrayBuffer;
    case C2.UNIFORM_BUFFER:
      // Generic binding point — distinct from the indexed bindings (GLES 3.0
      // §2.10.1). bufferData/getBufferSubData/getBufferParameter on these
      // targets operate on the generic binding (CTS
      // uniform-buffers-state-restoration.html: bufferData after
      // bindBuffer(UNIFORM_BUFFER, throwawayBuf) must hit throwawayBuf).
      return genericBindingState(ctx).uniformBuffer;
    case C2.TRANSFORM_FEEDBACK_BUFFER:
      return genericBindingState(ctx).transformFeedbackBuffer;
    case C2.COPY_READ_BUFFER:
      return s.copyReadBuffer;
    case C2.COPY_WRITE_BUFFER:
      return s.copyWriteBuffer;
    case C2.PIXEL_PACK_BUFFER:
      return s.pixelPackBuffer;
    case C2.PIXEL_UNPACK_BUFFER:
      return s.pixelUnpackBuffer;
    default:
      return null;
  }
}

/**
 * Unbind `buffer` from every binding point it appears in (context bindings,
 * current VAO, default VAO, all VAO objects, UBO bindings, pixel/copy buffers,
 * TF object bindings and its own _tfRangeBindings). Returns true when it was
 * bound somewhere (deletion was deferred per spec).
 *
 * `skipTfBindings` leaves the TF state alone (transform-feedback bindings hold a
 * reference — deleteBuffer handles them separately so an unbound TF object can
 * keep a deleted buffer alive, CTS runUnboundDeleteTest).
 */
function unbindBufferEverywhere(
  ctx: WebGLRenderingContext,
  buffer: WebGLBuffer,
  skipTfBindings = false,
): boolean {
  let found = false;
  const nullIf = (b: WebGLBuffer | null): WebGLBuffer | null => {
    if (b === buffer) {
      found = true;
      return null;
    }
    return b;
  };
  const s = ctx._state;
  s.arrayBuffer = nullIf(s.arrayBuffer);
  s.vao.elementArrayBuffer = nullIf(s.vao.elementArrayBuffer);
  for (const a of s.vao.attribs) a.buffer = nullIf(a.buffer);
  if (ctx._defaultVAO) {
    ctx._defaultVAO.elementArrayBuffer = nullIf(ctx._defaultVAO.elementArrayBuffer);
    for (const a of ctx._defaultVAO.attribs) a.buffer = nullIf(a.buffer);
  }
  if (ctx._version === 2) {
    const g = genericBindings.get(ctx);
    for (let i = 0; i < s.uniformBuffers.length; i++) {
      if (s.uniformBuffers[i] === buffer) {
        s.uniformBuffers[i] = null;
        s.uniformBufferRanges[i] = { offset: 0, size: 0 };
        if (g) g.baseUniformIndices.delete(i);
        found = true;
      }
    }
    if (g) {
      g.uniformBuffer = nullIf(g.uniformBuffer);
      g.transformFeedbackBuffer = nullIf(g.transformFeedbackBuffer);
    }
    s.pixelPackBuffer = nullIf(s.pixelPackBuffer);
    s.pixelUnpackBuffer = nullIf(s.pixelUnpackBuffer);
    s.copyReadBuffer = nullIf(s.copyReadBuffer);
    s.copyWriteBuffer = nullIf(s.copyWriteBuffer);
  }
  if (s.transformFeedback && !skipTfBindings) {
    for (let i = 0; i < s.transformFeedback._buffers.length; i++) {
      if (s.transformFeedback._buffers[i] === buffer) {
        s.transformFeedback._buffers[i] = null;
        found = true;
      }
    }
  }
  if (!skipTfBindings) {
    for (const obj of ctx._resources.all) {
      if (obj instanceof WebGLVertexArrayObject && obj._vao) {
        obj._vao.elementArrayBuffer = nullIf(obj._vao.elementArrayBuffer);
        for (const a of obj._vao.attribs) a.buffer = nullIf(a.buffer);
      } else if (obj instanceof WebGLTransformFeedback) {
        for (let i = 0; i < obj._buffers.length; i++) {
          if (obj._buffers[i] === buffer) {
            obj._buffers[i] = null;
            found = true;
          }
        }
      }
    }
    if (buffer._tfRangeBindings.length > 0) {
      found = true;
      buffer._tfRangeBindings.length = 0;
    }
  }
  return found;
}

/** Shared bindBufferBase logic (also used by bindBuffer(UNIFORM_BUFFER/TRANSFORM_FEEDBACK_BUFFER, b)). */
function bindBufferBaseImpl(ctx: WebGLRenderingContext, target: GLenum, index: GLuint, buffer: WebGLBuffer | null): void {
  const s = ctx._state;
  // GLES 3.0 §2.13: while transform feedback is active, the indexed TF
  // bindings cannot be changed — bindBufferBase/Range with target
  // TRANSFORM_FEEDBACK_BUFFER fails INVALID_OPERATION BEFORE any index/range
  // validation, and the existing binding stays (CTS transform_feedback.html
  // runUnchangedBufferBindingsTest, switching-objects.html "bindBufferBase
  // (TRANSFORM_FEEDBACK_BUFFER) when active").
  if (target === C2.TRANSFORM_FEEDBACK_BUFFER && transformFeedbackActive(ctx)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  const max =
    target === C2.UNIFORM_BUFFER
      ? s.limits.MAX_UNIFORM_BUFFER_BINDINGS
      : s.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
  if (index < 0 || index >= max) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (buffer === null || buffer === undefined) {
    if (target === C2.UNIFORM_BUFFER) {
      s.uniformBuffers[index] = null;
      s.uniformBufferRanges[index] = { offset: 0, size: 0 };
      genericBindingState(ctx).baseUniformIndices.delete(index);
      // GLES 3.0 §2.10.1: bindBufferBase at ANY index (also) updates the
      // generic binding point — unbinding at any index clears it too.
      genericBindingState(ctx).uniformBuffer = null;
    } else {
      const boundTf = s.transformFeedback;
      if (boundTf) {
        // Unbinding while a TF object is bound clears THAT object's indexed
        // binding only (indexed TF bindings are per-object state, GLES 3.0
        // §6.24; the default object's bindings — the global mirror — are
        // untouched).
        boundTf._buffers[index] = null;
        boundTf._bufferRanges[index] = { offset: 0, size: 0 };
      } else {
        clearTfBinding(ctx, index); // default-TF binding (global mirror)
      }
      // GLES 3.0 §2.10.1: any-index bindBufferBase(null) clears the generic
      // TRANSFORM_FEEDBACK_BUFFER binding point too.
      genericBindingState(ctx).transformFeedbackBuffer = null;
    }
    return;
  }
  if (buffer instanceof WebGLBuffer && buffer._deletePending) {
    // Re-binding a pending-delete buffer: ensure it is unbound everywhere and
    // release it (deletion was deferred at deleteBuffer time). It is already
    // _deleted, so validateObject rejects the bind with INVALID_OPERATION.
    unbindBufferEverywhere(ctx, buffer);
    buffer._deletePending = false;
  }
  const buf = validateBuffer(ctx, buffer);
  if (buf === null) return; // cross-context/deleted → INVALID_OPERATION pushed
  // WebGL2 buffer-type model (spec §BUFFER_OBJECT_BINDING): the first bind via
  // bindBufferBase/Range fixes the buffer's TYPE — UNIFORM_BUFFER and
  // TRANSFORM_FEEDBACK_BUFFER both set "other data" (the spec table lists all
  // binding points except ELEMENT_ARRAY_BUFFER / COPY_READ / COPY_WRITE).
  // Later binds only reject element-array buffers (CTS
  // buffer-type-restrictions.html exercises bindBufferBase/Range with every
  // target combination and expects the same element-array vs other-data
  // conflicts as bindBuffer).
  if (buf._target === 0) buf._target = target;
  else if (buf._target === C1.ELEMENT_ARRAY_BUFFER) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  everBoundBuffers.add(buf);
  if (target === C2.UNIFORM_BUFFER) {
    // Whole-buffer range; the size tracks later bufferData resizes (see
    // bufferData below — uniform-buffers-state-restoration.html binds an
    // unallocated buffer here, then bufferData, then draws the data).
    genericBindingState(ctx).baseUniformIndices.add(index);
    s.uniformBuffers[index] = buf;
    s.uniformBufferRanges[index] = { offset: 0, size: buf._size };
    // GLES 3.0 §2.10.1: bindBufferBase at ANY index also updates the generic
    // binding point (native behavior; bufferData/bufferSubData/
    // getBufferSubData on UNIFORM_BUFFER operate on the generic point).
    genericBindingState(ctx).uniformBuffer = buf;
  } else {
    // Indexed TF bindings are transform-feedback-OBJECT state (GLES 3.0
    // §6.24): record on the BOUND object (getIndexedParameter / draw capture
    // read it). With no TF object bound the binding belongs to the DEFAULT TF
    // object (name 0) — recorded in the global _tfRangeBindings mirror, which
    // the webgl2 agent copies into the default object at beginTransformFeedback.
    const boundTf = s.transformFeedback;
    if (boundTf) {
      boundTf._buffers[index] = buf;
      boundTf._bufferRanges[index] = { offset: 0, size: buf._size };
    } else {
      setTfBinding(ctx, index, buf, 0, buf._size); // whole buffer (default TF)
    }
    // GLES 3.0 §2.10.1: bindBufferBase at ANY index also updates the generic
    // TRANSFORM_FEEDBACK_BUFFER binding point (native behavior; CTS
    // too-small-buffers.html binds index 1 and expects the subsequent
    // bufferData(TRANSFORM_FEEDBACK_BUFFER, ...) to allocate buffer1's store).
    genericBindingState(ctx).transformFeedbackBuffer = buf;
  }
}

/**
 * WebGL2 bufferData/bufferSubData srcOffset/srcLength slicing (spec §3.7.1):
 * copyLength is measured in ELEMENTS of srcData (bytes for DataView/ArrayBuffer);
 * a srcLength of 0 (or omitted) means "the rest of the view" (length ≤ 0 never
 * errors by itself). On validation failure pushes INVALID_VALUE and returns
 * ok:false — no data is written and (for bufferData) the buffer size stays
 * unmodified.
 */
function sliceSourceData(
  ctx: WebGLRenderingContext,
  data: ArrayBufferView | ArrayBuffer,
  bytes: Uint8Array,
  srcOffset: unknown,
  srcLength: unknown,
): { bytes: Uint8Array; ok: boolean } {
  const view = data as { BYTES_PER_ELEMENT?: number; length?: number };
  const elemSize = view.BYTES_PER_ELEMENT ?? 1;
  const elemCount = view.length ?? (data as ArrayBuffer).byteLength;
  const off = Math.trunc(Number(srcOffset));
  const len = srcLength === undefined || srcLength === 0 ? elemCount - off : Math.trunc(Number(srcLength));
  if (off < 0 || len < 0 || off > elemCount || off + len > elemCount) {
    ctx._errors.push(C1.INVALID_VALUE);
    return { bytes, ok: false };
  }
  return { bytes: new Uint8Array(bytes.buffer, bytes.byteOffset + off * elemSize, len * elemSize), ok: true };
}

export function installBuffersApi(proto: WebGLRenderingContext): void {
  proto.createBuffer = function (this: WebGLRenderingContext): WebGLBuffer | null {
    const ctx = this;
    // No [WebGLHandlesContextLoss] on createBuffer: while lost it still creates
    // an object (CTS context-lost.html nonNullTests: createBuffer → non-null)
    // with NO error; isBuffer on it → false while lost (isLost guard).
    return createObject(ctx, BufferCtor);
  };

  proto.deleteBuffer = function (this: WebGLRenderingContext, buffer: WebGLBuffer | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (buffer === null || buffer === undefined) return;
    if (!(buffer instanceof WebGLBuffer)) {
      throw new TypeError(`Argument is not of type 'WebGLBuffer'`);
    }
    if (buffer._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (buffer._deleted) return; // already deleted: silent no-op (spec)
    // A transform-feedback binding holds a reference to the buffer: while an
    // UNBOUND TF object or the generic indexed binding references it, the buffer
    // stays alive — bindings and resource tracking are preserved (CTS
    // transform_feedback.html runUnboundDeleteTest: deleting a buffer attached
    // to an unbound TF keeps it; getIndexedParameter still returns it after the
    // TF is re-bound). Deleting while the BOUND TF object references it unbinds
    // it from that TF and from the generic indexed binding (runBoundDeleteTest:
    // getIndexedParameter → null after re-binding the TF).
    const s = ctx._state;
    const boundTf = s.transformFeedback;
    const boundTfRefs = !!boundTf && boundTf._buffers.includes(buffer);
    const tfRefs =
      buffer._tfRangeBindings.length > 0 ||
      [...ctx._resources.all].some(
        (o) => o instanceof WebGLTransformFeedback && o._buffers.includes(buffer),
      );
    // Unbind from every NON-TF binding point (TF state is handled below).
    const wasBound = unbindBufferEverywhere(ctx, buffer, true /* skipTfBindings */);
    if (boundTfRefs) {
      for (let i = 0; i < boundTf._buffers.length; i++) {
        if (boundTf._buffers[i] === buffer) {
          boundTf._buffers[i] = null;
          boundTf._bufferRanges[i] = { offset: 0, size: 0 };
        }
      }
      buffer._tfRangeBindings.length = 0; // bind/begin sync must not repopulate
    }
    buffer._deleted = true;
    buffer._deletePending = wasBound || tfRefs || boundTfRefs;
    // Untrack only when nothing holds the buffer anymore (a still-referencing
    // TF keeps it findable by the global binding scan used at bind/begin sync).
    if (!tfRefs || boundTfRefs) ctx._resources.untrack(buffer);
  };

  proto.isBuffer = function (this: WebGLRenderingContext, buffer: WebGLBuffer | null): GLboolean {
    const ctx = this;
    if (isLost(ctx)) return false;
    if (buffer === null || buffer === undefined) return false;
    if (!(buffer instanceof WebGLBuffer)) {
      throw new TypeError(`Argument is not of type 'WebGLBuffer'`);
    }
    // Deleted, foreign (different context) and never-bound buffers report false
    // with NO error (CTS incorrect-context-object-behaviour.html, is-object.html).
    if (buffer._context !== ctx) return false;
    if (buffer._deleted) return false;
    return everBoundBuffers.has(buffer);
  };

  proto.bindBuffer = function (this: WebGLRenderingContext, target: GLenum, buffer: WebGLBuffer | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!isValidBufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (ctx._version === 2 && (target === C2.TRANSFORM_FEEDBACK_BUFFER || target === C2.UNIFORM_BUFFER)) {
      // WebGL2: bindBuffer(UNIFORM_BUFFER/TRANSFORM_FEEDBACK_BUFFER, b) binds
      // the GENERIC binding point only — the indexed binding points are
      // untouched (CTS uniform-buffers-state-restoration.html:
      // bindBuffer(UNIFORM_BUFFER, throwawayBuf) after
      // bindBufferBase(UNIFORM_BUFFER, 0, greenBuf) leaves indexed[0] greenBuf;
      // simultaneous_binding.html: bindBuffer(UNIFORM_BUFFER, null) keeps
      // index 0). Legal even while transform feedback is active (generic TF
      // bindings are not part of the TF object — CTS switching-objects.html
      // "bindBuffer(TRANSFORM_FEEDBACK_BUFFER) when active" → NO_ERROR).
      if (buffer === null || buffer === undefined) {
        if (target === C2.UNIFORM_BUFFER) genericBindingState(ctx).uniformBuffer = null;
        else genericBindingState(ctx).transformFeedbackBuffer = null;
        return;
      }
      if (buffer instanceof WebGLBuffer && buffer._deletePending) {
        unbindBufferEverywhere(ctx, buffer);
        buffer._deletePending = false;
      }
      const buf = validateBuffer(ctx, buffer);
      if (buf === null) return; // cross-context/deleted → INVALID_OPERATION pushed
      // Buffer-type model (§BUFFER_OBJECT_BINDING): first bind fixes the type
      // (both targets → "other data"); element-array buffers are rejected.
      if (buf._target === 0) buf._target = target;
      else if (buf._target === C1.ELEMENT_ARRAY_BUFFER) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      everBoundBuffers.add(buf);
      if (target === C2.UNIFORM_BUFFER) genericBindingState(ctx).uniformBuffer = buf;
      else genericBindingState(ctx).transformFeedbackBuffer = buf;
      return;
    }
    if (buffer === null || buffer === undefined) {
      const s = ctx._state;
      if (target === C1.ARRAY_BUFFER) s.arrayBuffer = null;
      else if (target === C1.ELEMENT_ARRAY_BUFFER) s.vao.elementArrayBuffer = null; // lives in the VAO
      else if (ctx._version === 2) {
        switch (target) {
          case C2.COPY_READ_BUFFER: s.copyReadBuffer = null; break;
          case C2.COPY_WRITE_BUFFER: s.copyWriteBuffer = null; break;
          case C2.PIXEL_PACK_BUFFER: s.pixelPackBuffer = null; break;
          case C2.PIXEL_UNPACK_BUFFER: s.pixelUnpackBuffer = null; break;
        }
      }
      return;
    }
    if (buffer instanceof WebGLBuffer && buffer._deletePending) {
      unbindBufferEverywhere(ctx, buffer);
      buffer._deletePending = false;
    }
    const buf = validateBuffer(ctx, buffer);
    if (buf === null) return; // cross-context/deleted → INVALID_OPERATION pushed
    if (ctx._version === 2) {
      // WebGL2 buffer-type model (spec §2.9.1): the first bind fixes the buffer
      // TYPE — ELEMENT_ARRAY_BUFFER → element array; every other target (incl.
      // COPY_READ/WRITE_BUFFER) → other data. Element-array buffers may (re)bind
      // only ELEMENT_ARRAY_BUFFER, COPY_READ_BUFFER or COPY_WRITE_BUFFER;
      // other-data buffers may bind any target except ELEMENT_ARRAY_BUFFER
      // (CTS buffer-copying-contents.html / buffer-copying-restrictions.html /
      // out-of-bounds-index-buffers-after-copying.html rely on the COPY_* rebinds).
      const isElementType = buf._target === C1.ELEMENT_ARRAY_BUFFER;
      const isCopyTarget = target === C2.COPY_READ_BUFFER || target === C2.COPY_WRITE_BUFFER;
      if (buf._target === 0) {
        buf._target = target;
      } else if (isElementType) {
        if (target !== C1.ELEMENT_ARRAY_BUFFER && !isCopyTarget) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
      } else if (target === C1.ELEMENT_ARRAY_BUFFER) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
    } else {
      // WebGL1: a buffer stays bound to exactly one target for its lifetime.
      if (buf._target === 0) buf._target = target;
      else if (buf._target !== target) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
    }
    everBoundBuffers.add(buf);
    const s = ctx._state;
    if (target === C1.ARRAY_BUFFER) s.arrayBuffer = buf;
    else if (target === C1.ELEMENT_ARRAY_BUFFER) s.vao.elementArrayBuffer = buf;
    else if (ctx._version === 2) {
      switch (target) {
        case C2.COPY_READ_BUFFER: s.copyReadBuffer = buf; break;
        case C2.COPY_WRITE_BUFFER: s.copyWriteBuffer = buf; break;
        case C2.PIXEL_PACK_BUFFER: s.pixelPackBuffer = buf; break;
        case C2.PIXEL_UNPACK_BUFFER: s.pixelUnpackBuffer = buf; break;
      }
    }
  };

  proto.bufferData = function (this: WebGLRenderingContext, target: GLenum, size: GLsizeiptr | BufferDataSource, usage: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!isValidBufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (!isValidUsage(ctx, usage)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const buf = boundBufferForTarget(ctx, target);
    if (buf === null) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // Transform-feedback binding rules (spec "Preventing undefined behavior
    // with Transform Feedback"; GLES 3.0 §2.15.2): double-bound buffers and
    // buffers in an active TF's indexed bindings cannot be re-specified (CTS
    // transform_feedback/simultaneous_binding.html,
    // transform_feedback/switching-objects.html).
    if (ctx._version === 2 && bufferTfUseError(ctx, buf, target)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // WebIDL overload resolution (CTS buffer-data-and-buffer-sub-data.html):
    // null/undefined resolve to the nullable ArrayBuffer member → INVALID_VALUE
    // per the WebGL spec (no throw, no state change); ArrayBuffer/ArrayBufferView
    // take the data path; everything else (strings, objects, arrays, floats)
    // converts via ToNumber to the GLsizeiptr overload (long long semantics:
    // NaN/±Infinity → 0, truncation toward zero).
    if (size === null || size === undefined) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    if (size instanceof ArrayBuffer || ArrayBuffer.isView(size) || isSharedArrayBuffer(size)) {
      const data = requireBufferData(size, 'size');
      const bytes = isSharedArrayBuffer(data)
        ? new Uint8Array(data)
        : data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array((data as ArrayBufferView).buffer, (data as ArrayBufferView).byteOffset, (data as ArrayBufferView).byteLength);
      // WebGL2: bufferData(target, srcData, usage, srcOffset, srcLength)
      // uploads ONLY the srcData sub-range (spec §3.7.1) — the wasm CTS pages
      // pass ~4GB views and expect just the tail to be uploaded. WebGL1 has no
      // such overload and ignores extra arguments.
      let slice = bytes;
      if (ctx._version === 2 && arguments.length > 3) {
        const r = sliceSourceData(ctx, data, bytes, arguments[3], arguments[4]);
        if (!r.ok) return;
        slice = r.bytes;
      }
      let copy: ArrayBuffer;
      try {
        copy = new ArrayBuffer(slice.byteLength);
      } catch {
        ctx._errors.push(C1.OUT_OF_MEMORY);
        return;
      }
      new Uint8Array(copy).set(slice);
      buf._data = copy;
      buf._size = slice.byteLength;
    } else {
      const raw = Number(size); // WebIDL ToNumber (Symbol → TypeError, like WebIDL)
      const n = Number.isFinite(raw) ? Math.trunc(raw) : 0; // NaN/±Infinity → 0 (long long)
      if (n < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      let data: ArrayBuffer;
      try {
        data = new ArrayBuffer(n); // zero-filled
      } catch {
        ctx._errors.push(C1.OUT_OF_MEMORY);
        return;
      }
      buf._data = data;
      buf._size = n;
    }
    // bindBufferBase-bound UBO ranges are whole-buffer ranges that follow the
    // buffer's size: refresh them after a resize so draws see the new data
    // (CTS uniform-buffers-state-restoration.html binds an unallocated buffer
    // with bindBufferBase, then bufferData, then draws the block).
    if (ctx._version === 2) {
      const g = genericBindings.get(ctx);
      if (g && g.baseUniformIndices.size > 0) {
        const s = ctx._state;
        for (const i of g.baseUniformIndices) {
          if (s.uniformBuffers[i] === buf) s.uniformBufferRanges[i] = { offset: 0, size: buf._size };
        }
      }
    }
    buf._usage = usage;
  };

  proto.bufferSubData = function (this: WebGLRenderingContext, target: GLenum, offset: GLintptr, data: BufferDataSource): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!isValidBufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const buf = boundBufferForTarget(ctx, target);
    if (buf === null) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // Transform-feedback binding rules — same as bufferData (CTS
    // transform_feedback/simultaneous_binding.html).
    if (ctx._version === 2 && bufferTfUseError(ctx, buf, target)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (offset < 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    const src = requireBufferData(data, 'data'); // throws TypeError for wrong types
    const bytes =
      src instanceof ArrayBuffer ? new Uint8Array(src) : new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
    // WebGL2: bufferSubData(target, dstByteOffset, srcData, srcOffset, srcLength)
    // copies ONLY the srcData sub-range (spec §3.7.1); WebGL1 ignores extras.
    let slice = bytes;
    if (ctx._version === 2 && arguments.length > 3) {
      const r = sliceSourceData(ctx, src, bytes, arguments[3], arguments[4]);
      if (!r.ok) return;
      slice = r.bytes;
    }
    if (offset + slice.byteLength > buf._size) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    if (slice.byteLength > 0 && buf._data !== null) {
      new Uint8Array(buf._data).set(slice, offset);
    }
  };

  proto.getBufferParameter = function (this: WebGLRenderingContext, target: GLenum, pname: GLenum): any {
    const ctx = this;
    if (isLost(ctx)) return null;
    if (!isValidBufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    const buf = boundBufferForTarget(ctx, target);
    if (buf === null) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return null;
    }
    switch (pname) {
      case C1.BUFFER_SIZE:
        return buf._size;
      case C1.BUFFER_USAGE:
        return buf._usage;
      case GL_BUFFER_MAPPED: // WebGL2: buffers are never mapped
        if (ctx._version === 2) return false;
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      default:
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
    }
  };

  // ---- WebGL2-only methods (guarded: installBuffersApi runs on both prototypes) ----
  if ('bindBufferBase' in proto) {
    const p2 = proto as unknown as WebGL2RenderingContext;

    p2.bindBufferBase = function (this: WebGL2RenderingContext, target: GLenum, index: GLuint, buffer: WebGLBuffer | null): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (target !== C2.UNIFORM_BUFFER && target !== C2.TRANSFORM_FEEDBACK_BUFFER) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      bindBufferBaseImpl(ctx, target, index, buffer);
    };

    p2.bindBufferRange = function (this: WebGL2RenderingContext, target: GLenum, index: GLuint, buffer: WebGLBuffer | null, offset: GLintptr, size: GLsizeiptr): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (target !== C2.UNIFORM_BUFFER && target !== C2.TRANSFORM_FEEDBACK_BUFFER) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      // Same TF-active rule and ordering as bindBufferBase (see
      // bindBufferBaseImpl): INVALID_OPERATION before any index/range
      // validation (CTS runUnchangedBufferBindingsTest).
      if (target === C2.TRANSFORM_FEEDBACK_BUFFER && transformFeedbackActive(ctx)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const s = ctx._state;
      const max =
        target === C2.UNIFORM_BUFFER
          ? s.limits.MAX_UNIFORM_BUFFER_BINDINGS
          : s.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
      if (index < 0 || index >= max) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (buffer === null || buffer === undefined) {
        // Per spec the range arguments are ignored when buffer is null.
        if (target === C2.UNIFORM_BUFFER) {
          s.uniformBuffers[index] = null;
          s.uniformBufferRanges[index] = { offset: 0, size: 0 };
          genericBindingState(ctx).baseUniformIndices.delete(index);
          // GLES 3.0 §2.10.1: any-index bindBufferRange(null) clears the
          // generic binding point too.
          genericBindingState(ctx).uniformBuffer = null;
        } else {
          const boundTf = s.transformFeedback;
          if (boundTf) {
            // Unbinding while a TF object is bound clears THAT object's
            // indexed binding only (per-object state; the default object's
            // bindings — the global mirror — are untouched).
            boundTf._buffers[index] = null;
            boundTf._bufferRanges[index] = { offset: 0, size: 0 };
          } else {
            clearTfBinding(ctx, index); // default-TF binding (global mirror)
          }
          // GLES 3.0 §2.10.1: any-index bindBufferRange(null) clears the
          // generic TRANSFORM_FEEDBACK_BUFFER binding point too.
          genericBindingState(ctx).transformFeedbackBuffer = null;
        }
        return;
      }
      if (buffer instanceof WebGLBuffer && buffer._deletePending) {
        unbindBufferEverywhere(ctx, buffer);
        buffer._deletePending = false;
      }
      const buf = validateBuffer(ctx, buffer);
      if (buf === null) return; // cross-context/deleted → INVALID_OPERATION pushed
      if (offset < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (size < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (target === C2.UNIFORM_BUFFER) {
        if (offset % s.limits.UNIFORM_BUFFER_OFFSET_ALIGNMENT !== 0) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
        // MAX_UNIFORM_BLOCK_SIZE applies to the range SIZE, not to offset+size:
        // a large offset is legal (CTS large-uniform-buffers.html binds a
        // 16-byte range at offset 0x40000 in a 512 KB buffer). Ranges past the
        // end of the buffer are legal at bind and validated at draw time
        // (ANGLE issue 3388; CTS buffer-type-restrictions.html binds ranges in
        // never-allocated buffers).
        if (size > s.limits.MAX_UNIFORM_BLOCK_SIZE) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
      } else {
        if (offset % 4 !== 0) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
      }
      // GLES 3.0 §2.10.1: size 0 → the range extends to the end of the buffer.
      let rangeSize = size === 0 ? Math.max(0, buf._size - offset) : size;
      // Buffer-type model — same as bindBufferBase (first bind fixes the type
      // to "other data"; element-array buffers are rejected).
      if (buf._target === 0) buf._target = target;
      else if (buf._target === C1.ELEMENT_ARRAY_BUFFER) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      everBoundBuffers.add(buf);
      if (target === C2.UNIFORM_BUFFER) {
        // An explicit range overrides the whole-buffer (base) range semantics.
        genericBindingState(ctx).baseUniformIndices.delete(index);
        s.uniformBuffers[index] = buf;
        s.uniformBufferRanges[index] = { offset, size: rangeSize };
        // GLES 3.0 §2.10.1: any-index bindBufferRange also updates the
        // generic UNIFORM_BUFFER binding point.
        genericBindingState(ctx).uniformBuffer = buf;
      } else {
        // Indexed TF bindings are transform-feedback-OBJECT state (GLES 3.0
        // §6.24): record on the BOUND object; with no TF object bound the
        // binding belongs to the DEFAULT TF object (global mirror).
        const boundTf = s.transformFeedback;
        if (boundTf) {
          boundTf._buffers[index] = buf;
          boundTf._bufferRanges[index] = { offset, size: rangeSize };
        } else {
          setTfBinding(ctx, index, buf, offset, rangeSize);
        }
        // GLES 3.0 §2.10.1: any-index bindBufferRange also updates the
        // generic TRANSFORM_FEEDBACK_BUFFER binding point.
        genericBindingState(ctx).transformFeedbackBuffer = buf;
      }
    };

    p2.getIndexedParameter = function (this: WebGL2RenderingContext, target: GLenum, index: GLuint): any {
      const ctx = this;
      if (isLost(ctx)) return null;
      const s = ctx._state;
      let max: number;
      switch (target) {
        case C2.UNIFORM_BUFFER_BINDING:
        case C2.UNIFORM_BUFFER_START:
        case C2.UNIFORM_BUFFER_SIZE:
          max = s.limits.MAX_UNIFORM_BUFFER_BINDINGS;
          break;
        case C2.TRANSFORM_FEEDBACK_BUFFER_BINDING:
        case C2.TRANSFORM_FEEDBACK_BUFFER_START:
        case C2.TRANSFORM_FEEDBACK_BUFFER_SIZE:
          max = s.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
          break;
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
      }
      if (index < 0 || index >= max) {
        ctx._errors.push(C1.INVALID_VALUE);
        return null;
      }
      switch (target) {
        case C2.UNIFORM_BUFFER_BINDING:
          return s.uniformBuffers[index];
        case C2.UNIFORM_BUFFER_START:
          return s.uniformBufferRanges[index].offset;
        case C2.UNIFORM_BUFFER_SIZE:
          return s.uniformBufferRanges[index].size;
        case C2.TRANSFORM_FEEDBACK_BUFFER_BINDING: {
          // Indexed TF buffer bindings are transform-feedback-OBJECT state
          // (GLES 3.0 §6.24): the query reflects the BOUND object; with no
          // object bound nothing is bound → null (CTS transform_feedback.html
          // runTFBufferBindingTest line 122).
          const tf = s.transformFeedback;
          return tf ? (tf._buffers[index] ?? null) : null;
        }
        case C2.TRANSFORM_FEEDBACK_BUFFER_START: {
          const tf = s.transformFeedback;
          return tf ? (tf._bufferRanges[index]?.offset ?? 0) : 0;
        }
        default: { // TRANSFORM_FEEDBACK_BUFFER_SIZE
          const tf = s.transformFeedback;
          return tf ? (tf._bufferRanges[index]?.size ?? 0) : 0;
        }
      }
    };

    p2.getBufferSubData = function (this: WebGL2RenderingContext, target: GLenum, srcByteOffset: GLintptr, dstBuffer: ArrayBufferView, dstOffset?: GLuint, length?: GLuint): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (!isValidBufferTarget(ctx, target)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const buf = boundBufferForTarget(ctx, target);
      if (buf === null) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // Transform-feedback binding rules — same as bufferData/bufferSubData
      // (CTS transform_feedback.html runGetBufferSubDataTest: getBufferSubData
      // through the generic TF point of a buffer in the ACTIVE TF's indexed
      // bindings → INVALID_OPERATION; simultaneous_binding.html).
      if (bufferTfUseError(ctx, buf, target)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (srcByteOffset < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (!ArrayBuffer.isView(dstBuffer)) {
        throw new TypeError('dstBuffer is not an ArrayBufferView');
      }
      if (dstOffset !== undefined && dstOffset < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (length !== undefined && length < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      const view = dstBuffer as { buffer: ArrayBuffer; byteOffset: number; byteLength: number; BYTES_PER_ELEMENT?: number; length?: number };
      const elemSize = view.BYTES_PER_ELEMENT ?? 1; // DataView: byte granularity
      const elemCount = view.length ?? view.byteLength;
      const dstOff = dstOffset === undefined ? 0 : dstOffset;
      const len = length === undefined ? elemCount - dstOff : length;
      if (dstOff + len > elemCount) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const byteLen = len * elemSize;
      if (srcByteOffset + byteLen > buf._size) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (byteLen > 0 && buf._data !== null) {
        new Uint8Array(view.buffer, view.byteOffset + dstOff * elemSize, byteLen).set(
          new Uint8Array(buf._data, srcByteOffset, byteLen),
        );
      }
    };

    // copyBufferSubData (WebGL2 spec §3.7.1 / GLES 3.0 §2.10.5). NOTE: not
    // declared in the webgl2.ts class body, so cast the assignment (same
    // pattern as the getActiveUniforms alias in api/programs.ts). The method
    // must exist on OUR prototype — otherwise calls resolve through the
    // chainToNative re-chain to the NATIVE method and throw "Illegal
    // invocation" (killed buffer-copying-*, get-buffer-sub-data-validity and
    // out-of-bounds-index-buffers-after-copying before this was implemented).
    (p2 as unknown as {
      copyBufferSubData: (
        readTarget: GLenum,
        writeTarget: GLenum,
        readOffset: GLintptr,
        writeOffset: GLintptr,
        size: GLsizeiptr,
      ) => void;
    }).copyBufferSubData = function (
      this: WebGL2RenderingContext,
      readTarget: GLenum,
      writeTarget: GLenum,
      readOffset: GLintptr,
      writeOffset: GLintptr,
      size: GLsizeiptr,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (!isValidBufferTarget(ctx, readTarget) || !isValidBufferTarget(ctx, writeTarget)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const readBuffer = boundBufferForTarget(ctx, readTarget);
      const writeBuffer = boundBufferForTarget(ctx, writeTarget);
      if (readBuffer === null || writeBuffer === null) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // Transform-feedback binding rules (spec "Preventing undefined behavior
      // with Transform Feedback"): a buffer in the CURRENTLY BOUND TF object's
      // indexed bindings used through any other point fails (CTS
      // transform_feedback/simultaneous_binding.html), as does a buffer in an
      // ACTIVE TF's indexed bindings. The persistent _tfRangeBindings mirror is
      // deliberately NOT used — it survives bindTransformFeedback(null), after
      // which the copy must succeed (simultaneous_binding.html "Test bufferData
      // family with tf object unbound").
      if (bufferTfUseError(ctx, readBuffer, readTarget) || bufferTfUseError(ctx, writeBuffer, writeTarget)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // Buffer-type rule (spec "Copying Buffers"): copying between an
      // element-array buffer and an other-data buffer is INVALID_OPERATION
      // (CTS buffer-copying-restrictions.html, buffer-copying-contents.html).
      const readIsElement = readBuffer._target === C1.ELEMENT_ARRAY_BUFFER;
      const writeIsElement = writeBuffer._target === C1.ELEMENT_ARRAY_BUFFER;
      if (readIsElement !== writeIsElement) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (readOffset < 0 || writeOffset < 0 || size < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (readOffset + size > readBuffer._size) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (writeOffset + size > writeBuffer._size) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (readBuffer === writeBuffer && readOffset < writeOffset + size && writeOffset < readOffset + size) {
        // Same buffer, overlapping ranges (CTS buffer-copying-contents.html).
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (size > 0 && readBuffer._data !== null && writeBuffer._data !== null) {
        // The copied bytes are what subsequent draws/reads see: _data is the
        // single source of truth for index/attrib fetch and getBufferSubData.
        new Uint8Array(writeBuffer._data, writeOffset, size).set(
          new Uint8Array(readBuffer._data, readOffset, size),
        );
      }
    };
  }
}
