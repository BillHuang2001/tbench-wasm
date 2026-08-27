/**
 * src/gl/api/buffers.ts — buffer objects and data.
 *
 * Owns: createBuffer, deleteBuffer, isBuffer, bindBuffer, bufferData,
 * bufferSubData, getBufferParameter (+ WebGL2: bindBufferBase, bindBufferRange,
 * getIndexedParameter for UNIFORM_BUFFER/TRANSFORM_FEEDBACK_BUFFER, getBufferSubData).
 *
 * Behavior notes (implemented):
 *  - bindBuffer: target ∈ {ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER} (WebGL1) +
 *    {COPY_READ/WRITE_BUFFER, PIXEL_PACK/UNPACK_BUFFER, TRANSFORM_FEEDBACK_BUFFER,
 *    UNIFORM_BUFFER} (WebGL2). First bind fixes the buffer's target; rebinding to
 *    another target → INVALID_OPERATION (except TRANSFORM_FEEDBACK_BUFFER — see
 *    below). null unbinds (ELEMENT_ARRAY_BUFFER binding lives in the VAO state).
 *    WebGL2: bindBuffer(UNIFORM_BUFFER, b) is equivalent to
 *    bindBufferBase(UNIFORM_BUFFER, 0, b); bindBuffer(TRANSFORM_FEEDBACK_BUFFER, b)
 *    is equivalent to bindBufferBase(TRANSFORM_FEEDBACK_BUFFER, 0, b) (CTS
 *    transform_feedback.html runTFBufferBindingTest + switching-objects.html).
 *    Per WebGL2 §BUFFER_OBJECT_BINDING the buffer-type split is element-array vs
 *    other-data: a TF bind only rejects element-array buffers and does NOT fix
 *    the buffer's target (a TF-bound buffer can later bind ARRAY_BUFFER —
 *    runTFBufferBindingTest binds a TF buffer to ARRAY_BUFFER with NO_ERROR).
 *  - bufferData: size number → allocate zero-filled; ArrayBuffer/ArrayBufferView
 *    → copy (view bytes only, honoring byteOffset/byteLength); usage ∈
 *    {STREAM,STATIC,DYNAMIC}_DRAW (WebGL2 adds the *_READ/*_COPY usages) else
 *    INVALID_ENUM; negative size → INVALID_VALUE; no bound buffer →
 *    INVALID_OPERATION. Allocation failure (huge size) → OUT_OF_MEMORY.
 *  - bufferSubData: offset+byteLength bounds vs buffer size (INVALID_VALUE);
 *    copies the view's bytes at offset.
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
 *    UBO range offset aligned to UNIFORM_BUFFER_OFFSET_ALIGNMENT,
 *    offset+size ≤ buffer size and ≤ MAX_UNIFORM_BLOCK_SIZE (INVALID_VALUE);
 *    TF offset multiple of 4, offset+size ≤ buffer size. bindBufferBase sets the
 *    range to the whole buffer ({0, _size}). TF indexed bindings are recorded
 *    on the buffer object (_tfRangeBindings — last bind at an index wins).
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
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
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
 * Indexed TRANSFORM_FEEDBACK_BUFFER binding at `index` (last bind wins).
 * The source of truth is the buffer objects' _tfRangeBindings entries, which
 * persist regardless of which (if any) transform feedback object is bound.
 */
function tfBindingAtIndex(
  ctx: WebGLRenderingContext,
  index: number,
): { buffer: WebGLBuffer | null; offset: number; size: number } {
  for (const obj of ctx._resources.all) {
    if (obj instanceof WebGLBuffer) {
      for (const e of obj._tfRangeBindings) {
        if (e.index === index) return { buffer: obj, offset: e.offset, size: e.size };
      }
    }
  }
  return { buffer: null, offset: 0, size: 0 };
}

/** Remove any TF binding at `index` (previous buffer loses the slot). */
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
      return s.uniformBuffers[0];
    case C2.TRANSFORM_FEEDBACK_BUFFER:
      return tfBindingAtIndex(ctx, 0).buffer;
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
    for (let i = 0; i < s.uniformBuffers.length; i++) {
      if (s.uniformBuffers[i] === buffer) {
        s.uniformBuffers[i] = null;
        s.uniformBufferRanges[i] = { offset: 0, size: 0 };
        found = true;
      }
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
    } else {
      clearTfBinding(ctx, index);
      // Unbinding while a TF object is bound clears that object's indexed
      // binding too (indexed TF bindings are per-object state).
      const boundTf = s.transformFeedback;
      if (boundTf) {
        boundTf._buffers[index] = null;
        boundTf._bufferRanges[index] = { offset: 0, size: 0 };
      }
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
  if (target === C2.UNIFORM_BUFFER) {
    if (buf._target === 0) buf._target = target;
    else if (buf._target !== target) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    s.uniformBuffers[index] = buf;
    s.uniformBufferRanges[index] = { offset: 0, size: buf._size }; // whole buffer
  } else {
    // TRANSFORM_FEEDBACK_BUFFER: per WebGL2 §BUFFER_OBJECT_BINDING the buffer
    // type split is element-array vs other-data — only element-array buffers are
    // rejected, and a TF bind does NOT fix the buffer's target (a TF-bound
    // buffer can later bind ARRAY_BUFFER — CTS runTFBufferBindingTest).
    if (buf._target === C1.ELEMENT_ARRAY_BUFFER) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // Indexed TF bindings are transform-feedback-OBJECT state (GLES 3.0 §6.24):
    // record on the bound object (getIndexedParameter reads it) and mirror into
    // the global _tfRangeBindings (source for the webgl2 agent's bind/begin
    // sync, begin's buffer-count validation, and generic bufferData access).
    const boundTf = s.transformFeedback;
    if (boundTf) {
      boundTf._buffers[index] = buf;
      boundTf._bufferRanges[index] = { offset: 0, size: buf._size };
    }
    setTfBinding(ctx, index, buf, 0, buf._size); // whole buffer
  }
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
    if (buffer._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return false;
    }
    // Deleted (incl. pending-delete) buffers report false immediately, no error.
    return !buffer._deleted;
  };

  proto.bindBuffer = function (this: WebGLRenderingContext, target: GLenum, buffer: WebGLBuffer | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!isValidBufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (ctx._version === 2 && target === C2.TRANSFORM_FEEDBACK_BUFFER) {
      // WebGL2: bindBuffer(TRANSFORM_FEEDBACK_BUFFER, b) binds the generic TF
      // binding point — equivalent to bindBufferBase(TRANSFORM_FEEDBACK_BUFFER,
      // 0, b) (CTS transform_feedback.html runTFBufferBindingTest line 108,
      // switching-objects.html line 101).
      bindBufferBaseImpl(ctx, target, 0, buffer);
      return;
    }
    if (ctx._version === 2 && target === C2.UNIFORM_BUFFER) {
      bindBufferBaseImpl(ctx, target, 0, buffer);
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
    if (buf._target === 0) buf._target = target; // first bind fixes the target
    else if (buf._target !== target) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
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
    if (typeof size === 'number') {
      if (size < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      const n = Math.trunc(size); // WebIDL long long conversion
      let data: ArrayBuffer;
      try {
        data = new ArrayBuffer(n); // zero-filled
      } catch {
        ctx._errors.push(C1.OUT_OF_MEMORY);
        return;
      }
      buf._data = data;
      buf._size = n;
    } else {
      const data = requireBufferData(size, 'size'); // throws TypeError for wrong types
      const bytes =
        data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      let copy: ArrayBuffer;
      try {
        copy = new ArrayBuffer(bytes.byteLength);
      } catch {
        ctx._errors.push(C1.OUT_OF_MEMORY);
        return;
      }
      new Uint8Array(copy).set(bytes);
      buf._data = copy;
      buf._size = bytes.byteLength;
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
    if (offset < 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    const src = requireBufferData(data, 'data'); // throws TypeError for wrong types
    const bytes =
      src instanceof ArrayBuffer ? new Uint8Array(src) : new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
    if (offset + bytes.byteLength > buf._size) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    if (bytes.byteLength > 0 && buf._data !== null) {
      new Uint8Array(buf._data).set(bytes, offset);
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
      const s = ctx._state;
      if (target !== C2.UNIFORM_BUFFER && target !== C2.TRANSFORM_FEEDBACK_BUFFER) {
        ctx._errors.push(C1.INVALID_ENUM);
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
        // Per spec the range arguments are ignored when buffer is null.
        if (target === C2.UNIFORM_BUFFER) {
          s.uniformBuffers[index] = null;
          s.uniformBufferRanges[index] = { offset: 0, size: 0 };
        } else {
          clearTfBinding(ctx, index);
          // Unbinding while a TF object is bound clears that object's indexed
          // binding too (indexed TF bindings are per-object state).
          const boundTf = s.transformFeedback;
          if (boundTf) {
            boundTf._buffers[index] = null;
            boundTf._bufferRanges[index] = { offset: 0, size: 0 };
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
        if (offset + size > buf._size) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
        if (offset + size > s.limits.MAX_UNIFORM_BLOCK_SIZE) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
      } else {
        if (offset % 4 !== 0) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
        if (offset + size > buf._size) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
      }
      if (target === C2.UNIFORM_BUFFER) {
        if (buf._target === 0) buf._target = target;
        else if (buf._target !== target) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
        s.uniformBuffers[index] = buf;
        s.uniformBufferRanges[index] = { offset, size };
      } else {
        // TRANSFORM_FEEDBACK_BUFFER: same buffer-type rule as bindBufferBase —
        // only element-array buffers are rejected; a TF bind does NOT fix the
        // buffer's target (see bindBufferBaseImpl).
        if (buf._target === C1.ELEMENT_ARRAY_BUFFER) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
        const boundTf = s.transformFeedback;
        if (boundTf) {
          boundTf._buffers[index] = buf;
          boundTf._bufferRanges[index] = { offset, size };
        }
        setTfBinding(ctx, index, buf, offset, size);
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
      if (target === C2.PIXEL_PACK_BUFFER) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const buf = boundBufferForTarget(ctx, target);
      if (buf === null) {
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
  }
}
