/**
 * src/gl/api/vertex-attrib.ts — vertex attrib array state.
 *
 * Owns: vertexAttribPointer, vertexAttribIPointer (WebGL2),
 * enableVertexAttribArray, disableVertexAttribArray,
 * vertexAttrib{1,2,3,4}{f,fv}, vertexAttribI4{i,iv,ui,uiv} (WebGL2),
 * vertexAttribDivisor (WebGL2/ANGLE_instanced_arrays), getVertexAttrib,
 * getVertexAttribOffset.
 *
 * Behavior notes (implemented):
 *  - index args: 0 ≤ index < MAX_VERTEX_ATTRIBS (WebIDL unsigned-long
 *    converted), else INVALID_VALUE.
 *  - vertexAttribPointer: a bound ARRAY_BUFFER is CAPTURED into attrib.buffer;
 *    with NO buffer bound, offset 0 succeeds (attrib.buffer = null) and
 *    offset ≠ 0 → INVALID_OPERATION; size 1..4 (INVALID_VALUE); type ∈ {BYTE,
 *    UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, FLOAT} (+ WebGL2 {INT,
 *    UNSIGNED_INT, HALF_FLOAT, INT_2_10_10_10_REV, UNSIGNED_INT_2_10_10_10_REV},
 *    INVALID_ENUM otherwise); for the 2_10_10_10_REV types size must be 4
 *    (INVALID_OPERATION); stride 0..255 (INVALID_VALUE); offset ≥ 0 and < 2^31
 *    (INVALID_VALUE); stride and offset must be multiples of the type's
 *    component size (INVALID_OPERATION). Sets attrib.integer = false.
 *  - vertexAttribIPointer (WebGL2): type ∈ {BYTE, UNSIGNED_BYTE, SHORT,
 *    UNSIGNED_SHORT, INT, UNSIGNED_INT} (INVALID_ENUM); same stride/offset
 *    and null-buffer rules as vertexAttribPointer; sets attrib.integer = true
 *    (getVertexAttrib VERTEX_ATTRIB_ARRAY_INTEGER).
 *  - vertexAttrib*f(v): sets the constant generic value (constantF, default
 *    (v...,0,0,1) fill) — does NOT touch the enabled state (the array is used
 *    when enabled, the constant otherwise). fv: values shorter than needed →
 *    INVALID_OPERATION; wrong value type → TypeError (WebIDL Float32List).
 *  - vertexAttribI4i/iv/I4ui/uiv (WebGL2): set constantI/constantUI
 *    (WebIDL long / unsigned long converted).
 *  - vertexAttribDivisor (WebGL2): stores state.vao.attribs[index].divisor.
 *  - getVertexAttrib: CURRENT_VERTEX_ATTRIB → Float32Array copy of constantF;
 *    VERTEX_ATTRIB_ARRAY_ENABLED/SIZE/STRIDE/TYPE/NORMALIZED/POINTER (0) /
 *    BUFFER_BINDING from the attrib; WebGL2 adds VERTEX_ATTRIB_ARRAY_INTEGER /
 *    VERTEX_ATTRIB_ARRAY_DIVISOR (INVALID_ENUM on WebGL1). Other pnames →
 *    INVALID_ENUM.
 *  - getVertexAttribOffset: pname must be VERTEX_ATTRIB_ARRAY_POINTER
 *    (INVALID_ENUM otherwise); returns attrib.offset.
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';
import { C1, C2 } from '../constants';
import type { VertexAttribState } from '../state';
import type { Float32List, GLboolean, GLenum, GLint, GLintptr, GLsizei, GLuint, Int32List, Uint32List } from '../types';

/** Context-loss guard: no-op + one CONTEXT_LOST_WEBGL per call. */
function isLost(ctx: WebGLRenderingContext): boolean {
  if (ctx._isLost) ctx._errors.push(C1.CONTEXT_LOST_WEBGL);
  return ctx._isLost;
}

/** Validate an attrib index (WebIDL unsigned-long converted). null → INVALID_VALUE pushed. */
function attribIndex(ctx: WebGLRenderingContext, index: GLuint): number | null {
  const i = index >>> 0;
  if (i >= ctx._state.limits.MAX_VERTEX_ATTRIBS) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  return i;
}

/** WebGL1 pointer types (+ WebGL2 additions). */
function isPointerType(ctx: WebGLRenderingContext, type: GLenum): boolean {
  switch (type) {
    case C1.BYTE:
    case C1.UNSIGNED_BYTE:
    case C1.SHORT:
    case C1.UNSIGNED_SHORT:
    case C1.FLOAT:
      return true;
    case C1.INT:
    case C1.UNSIGNED_INT:
    case C2.HALF_FLOAT:
    case C2.INT_2_10_10_10_REV:
    case C2.UNSIGNED_INT_2_10_10_10_REV:
      return ctx._version === 2;
    default:
      return false;
  }
}

/**
 * Size in bytes of one component for each vertexAttribPointer/IPointer type.
 * The 2_10_10_10_REV types are 4-byte packed formats.
 */
const TYPE_BYTE_SIZE: Record<number, number> = {
  [C1.BYTE]: 1,
  [C1.UNSIGNED_BYTE]: 1,
  [C1.SHORT]: 2,
  [C1.UNSIGNED_SHORT]: 2,
  [C1.FLOAT]: 4,
  [C1.INT]: 4,
  [C1.UNSIGNED_INT]: 4,
  [C2.HALF_FLOAT]: 2,
  [C2.INT_2_10_10_10_REV]: 4,
  [C2.UNSIGNED_INT_2_10_10_10_REV]: 4,
};

/**
 * WebGL supports vertex attribute data strides up to 255 bytes (spec
 * "Vertex Attribute Data Stride": a call with stride > 255 generates
 * INVALID_VALUE). Applies to BOTH WebGL1 and WebGL2, for all types.
 */
const MAX_VERTEX_ATTRIB_STRIDE_BYTES = 255;

/**
 * Shared stride/offset validation for vertexAttribPointer & vertexAttribIPointer.
 * Enforces, per spec ("Buffer Offset and Stride Requirements" / "Vertex
 * Attribute Data Stride"):
 *  - 0 ≤ stride ≤ 255, else INVALID_VALUE
 *  - 0 ≤ offset < 2^31, else INVALID_VALUE
 *  - stride and offset must be multiples of the size of the data type,
 *    else INVALID_OPERATION.
 * Returns the converted [stride, offset] or null (INVALID_VALUE or
 * INVALID_OPERATION already pushed).
 */
function validateStrideOffset(ctx: WebGLRenderingContext, stride: GLsizei, offset: GLintptr, typeSize: number): [number, number] | null {
  const st = stride | 0;
  if (st < 0 || st > MAX_VERTEX_ATTRIB_STRIDE_BYTES) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  // Raw offset check: valid iff 0 ≤ offset < 2^31 (WebGL spec). NaN slips
  // through the comparison and becomes 0 below (WebIDL NaN → 0).
  if (offset < 0 || offset >= 0x80000000) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (st % typeSize !== 0 || offset % typeSize !== 0) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  return [st, offset | 0];
}

/** WebIDL Float32List conversion — TypeError for non-sequence values. */
function toFloat32List(v: Float32List): Float32Array {
  if (v instanceof DataView) throw new TypeError('Argument is not a Float32List');
  if (v instanceof Float32Array) return v;
  if (Array.isArray(v)) return new Float32Array(v as number[]);
  if (ArrayBuffer.isView(v)) {
    // Other typed arrays are iterable → accepted as a sequence<float> per WebIDL.
    return new Float32Array(v as ArrayLike<number>);
  }
  throw new TypeError('Argument is not a Float32List');
}

/** WebIDL Int32List conversion. */
function toInt32List(v: Int32List): Int32Array {
  if (v instanceof DataView) throw new TypeError('Argument is not an Int32List');
  if (v instanceof Int32Array) return v;
  if (Array.isArray(v)) return Int32Array.from(v as number[]);
  if (ArrayBuffer.isView(v)) return Int32Array.from(v as ArrayLike<number>);
  throw new TypeError('Argument is not an Int32List');
}

/** WebIDL Uint32List conversion. */
function toUint32List(v: Uint32List): Uint32Array {
  if (v instanceof DataView) throw new TypeError('Argument is not a Uint32List');
  if (v instanceof Uint32Array) return v;
  if (Array.isArray(v)) return Uint32Array.from(v as number[]);
  if (ArrayBuffer.isView(v)) return Uint32Array.from(v as ArrayLike<number>);
  throw new TypeError('Argument is not a Uint32List');
}

/** Shared vertexAttrib{f,fv} constant-setter. */
function setConstantF(attrib: VertexAttribState, vals: Float32Array, needed: number): void {
  const c = attrib.constantF;
  c[0] = vals[0];
  c[1] = needed > 1 ? vals[1] : 0;
  c[2] = needed > 2 ? vals[2] : 0;
  c[3] = needed > 3 ? vals[3] : 1;
}

export function installVertexAttribApi(proto: WebGLRenderingContext): void {
  proto.enableVertexAttribArray = function (this: WebGLRenderingContext, index: GLuint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    ctx._state.vao.attribs[i].enabled = true;
  };

  proto.disableVertexAttribArray = function (this: WebGLRenderingContext, index: GLuint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    ctx._state.vao.attribs[i].enabled = false;
  };

  proto.vertexAttribPointer = function (this: WebGLRenderingContext, index: GLuint, size: GLint, type: GLenum, normalized: GLboolean, stride: GLsizei, offset: GLintptr): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const sz = size | 0;
    if (sz < 1 || sz > 4) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    if (!isPointerType(ctx, type)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if ((type === C2.INT_2_10_10_10_REV || type === C2.UNSIGNED_INT_2_10_10_10_REV) && sz !== 4) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    const so = validateStrideOffset(ctx, stride, offset, TYPE_BYTE_SIZE[type]);
    if (so === null) return;
    // No buffer bound: only legal when offset is 0 — the attrib's buffer
    // binding is cleared to null (spec); offset ≠ 0 → INVALID_OPERATION.
    const bound = ctx._state.arrayBuffer;
    if (bound === null && so[1] !== 0) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    const attrib = ctx._state.vao.attribs[i];
    attrib.size = sz;
    attrib.type = type;
    attrib.normalized = !!normalized;
    attrib.integer = false;
    attrib.stride = so[0];
    attrib.offset = so[1];
    attrib.buffer = bound; // captured at pointer time (null when no buffer bound + offset 0)
  };

  proto.vertexAttrib1f = function (this: WebGLRenderingContext, index: GLuint, x: GLfloat): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const c = ctx._state.vao.attribs[i].constantF;
    c[0] = x;
    c[1] = 0;
    c[2] = 0;
    c[3] = 1;
  };

  proto.vertexAttrib1fv = function (this: WebGLRenderingContext, index: GLuint, values: Float32List): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const a = toFloat32List(values);
    if (a.length < 1) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    setConstantF(ctx._state.vao.attribs[i], a, 1);
  };

  proto.vertexAttrib2f = function (this: WebGLRenderingContext, index: GLuint, x: GLfloat, y: GLfloat): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const c = ctx._state.vao.attribs[i].constantF;
    c[0] = x;
    c[1] = y;
    c[2] = 0;
    c[3] = 1;
  };

  proto.vertexAttrib2fv = function (this: WebGLRenderingContext, index: GLuint, values: Float32List): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const a = toFloat32List(values);
    if (a.length < 2) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    setConstantF(ctx._state.vao.attribs[i], a, 2);
  };

  proto.vertexAttrib3f = function (this: WebGLRenderingContext, index: GLuint, x: GLfloat, y: GLfloat, z: GLfloat): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const c = ctx._state.vao.attribs[i].constantF;
    c[0] = x;
    c[1] = y;
    c[2] = z;
    c[3] = 1;
  };

  proto.vertexAttrib3fv = function (this: WebGLRenderingContext, index: GLuint, values: Float32List): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const a = toFloat32List(values);
    if (a.length < 3) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    setConstantF(ctx._state.vao.attribs[i], a, 3);
  };

  proto.vertexAttrib4f = function (this: WebGLRenderingContext, index: GLuint, x: GLfloat, y: GLfloat, z: GLfloat, w: GLfloat): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const c = ctx._state.vao.attribs[i].constantF;
    c[0] = x;
    c[1] = y;
    c[2] = z;
    c[3] = w;
  };

  proto.vertexAttrib4fv = function (this: WebGLRenderingContext, index: GLuint, values: Float32List): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const i = attribIndex(ctx, index);
    if (i === null) return;
    const a = toFloat32List(values);
    if (a.length < 4) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    setConstantF(ctx._state.vao.attribs[i], a, 4);
  };

  proto.getVertexAttrib = function (this: WebGLRenderingContext, index: GLuint, pname: GLenum): any {
    const ctx = this;
    if (isLost(ctx)) return null;
    const i = attribIndex(ctx, index);
    if (i === null) return null;
    const attrib = ctx._state.vao.attribs[i];
    switch (pname) {
      case C1.CURRENT_VERTEX_ATTRIB:
        return new Float32Array(attrib.constantF); // copy
      case C1.VERTEX_ATTRIB_ARRAY_ENABLED:
        return attrib.enabled;
      case C1.VERTEX_ATTRIB_ARRAY_SIZE:
        return attrib.size;
      case C1.VERTEX_ATTRIB_ARRAY_STRIDE:
        return attrib.stride;
      case C1.VERTEX_ATTRIB_ARRAY_TYPE:
        return attrib.type;
      case C1.VERTEX_ATTRIB_ARRAY_NORMALIZED:
        return attrib.normalized;
      case C1.VERTEX_ATTRIB_ARRAY_BUFFER_BINDING:
        return attrib.buffer;
      case C1.VERTEX_ATTRIB_ARRAY_POINTER:
        return 0; // WebGL has no client-side pointers
      case C2.VERTEX_ATTRIB_ARRAY_INTEGER:
      case C2.VERTEX_ATTRIB_ARRAY_DIVISOR:
        if (ctx._version !== 2) {
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
        }
        return pname === C2.VERTEX_ATTRIB_ARRAY_INTEGER ? attrib.integer : attrib.divisor;
      default:
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
    }
  };

  proto.getVertexAttribOffset = function (this: WebGLRenderingContext, index: GLuint, pname: GLenum): GLintptr {
    const ctx = this;
    if (isLost(ctx)) return 0;
    const i = attribIndex(ctx, index);
    if (i === null) return 0;
    if (pname !== C1.VERTEX_ATTRIB_ARRAY_POINTER) {
      ctx._errors.push(C1.INVALID_ENUM);
      return 0;
    }
    return ctx._state.vao.attribs[i].offset;
  };

  // ---- WebGL2-only methods (guarded: installVertexAttribApi runs on both prototypes) ----
  if ('vertexAttribIPointer' in proto) {
    const p2 = proto as unknown as WebGL2RenderingContext;

    p2.vertexAttribIPointer = function (this: WebGL2RenderingContext, index: GLuint, size: GLint, type: GLenum, stride: GLsizei, offset: GLintptr): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const i = attribIndex(ctx, index);
      if (i === null) return;
      const sz = size | 0;
      if (sz < 1 || sz > 4) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      switch (type) {
        case C1.BYTE:
        case C1.UNSIGNED_BYTE:
        case C1.SHORT:
        case C1.UNSIGNED_SHORT:
        case C1.INT:
        case C1.UNSIGNED_INT:
          break;
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return;
      }
      const so = validateStrideOffset(ctx, stride, offset, TYPE_BYTE_SIZE[type]);
      if (so === null) return;
      // No buffer bound: only legal when offset is 0 (attrib.buffer = null);
      // offset ≠ 0 → INVALID_OPERATION (same rule as vertexAttribPointer).
      const bound = ctx._state.arrayBuffer;
      if (bound === null && so[1] !== 0) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const attrib = ctx._state.vao.attribs[i];
      attrib.size = sz;
      attrib.type = type;
      attrib.normalized = false;
      attrib.integer = true;
      attrib.stride = so[0];
      attrib.offset = so[1];
      attrib.buffer = bound;
    };

    p2.vertexAttribI4i = function (this: WebGL2RenderingContext, index: GLuint, x: GLint, y: GLint, z: GLint, w: GLint): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const i = attribIndex(ctx, index);
      if (i === null) return;
      const c = ctx._state.vao.attribs[i].constantI;
      c[0] = x | 0;
      c[1] = y | 0;
      c[2] = z | 0;
      c[3] = w | 0;
    };

    p2.vertexAttribI4iv = function (this: WebGL2RenderingContext, index: GLuint, values: Int32List): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const i = attribIndex(ctx, index);
      if (i === null) return;
      const a = toInt32List(values);
      if (a.length < 4) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const c = ctx._state.vao.attribs[i].constantI;
      c[0] = a[0];
      c[1] = a[1];
      c[2] = a[2];
      c[3] = a[3];
    };

    p2.vertexAttribI4ui = function (this: WebGL2RenderingContext, index: GLuint, x: GLuint, y: GLuint, z: GLuint, w: GLuint): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const i = attribIndex(ctx, index);
      if (i === null) return;
      const c = ctx._state.vao.attribs[i].constantUI;
      c[0] = x >>> 0;
      c[1] = y >>> 0;
      c[2] = z >>> 0;
      c[3] = w >>> 0;
    };

    p2.vertexAttribI4uiv = function (this: WebGL2RenderingContext, index: GLuint, values: Uint32List): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const i = attribIndex(ctx, index);
      if (i === null) return;
      const a = toUint32List(values);
      if (a.length < 4) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const c = ctx._state.vao.attribs[i].constantUI;
      c[0] = a[0];
      c[1] = a[1];
      c[2] = a[2];
      c[3] = a[3];
    };

    p2.vertexAttribDivisor = function (this: WebGL2RenderingContext, index: GLuint, divisor: GLuint): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const i = attribIndex(ctx, index);
      if (i === null) return;
      ctx._state.vao.attribs[i].divisor = divisor >>> 0;
    };
  }
}
