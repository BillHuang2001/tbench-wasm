/**
 * src/gl/api/uniforms.ts — uniform setters (write into the per-program store).
 *
 * Owns: uniform{1,2,3,4}{f,i,ui}{,v} and uniformMatrix{2,3,4}{,x2,x3,x4}fv
 * (WebGL1: f/i + 2x2/3x3/4x4; WebGL2 adds ui + non-square matrices).
 *
 * Store contract (with glsl/, see api/programs.ts header "UNIFORM STORE LAYOUT"):
 * each glsl Program.uniforms[i] has a `location` — a FLOAT index into the
 * store (NOT a vec4 slot). Float/matrix writes go through
 * WebGLProgram._uniformStore (a DataView over the glsl floatStore's
 * ArrayBuffer): byte offset = floatIndex*4, little-endian, where floatIndex =
 * location + element*stride + comp (stride = elementSlots(): scalar/sampler 1,
 * vector 4, matrix cols*4; matrix comp = col*4 + row). Int/uint/bool/sampler
 * writes go to the glsl Program.intStore (Int32Array) at the SAME float index
 * (uint stored as raw int32 bits; the generated code reinterprets via >>> 0).
 * The draw engine passes the same floatStore/intStore as ctx.uniforms /
 * ctx.intUniforms, so these writes are what the shaders read.
 *
 * Validation (WebGL 1.0 spec §5.14.10 + CTS):
 *  - location null → silent no-op. Location from a non-current program, or
 *    stale (program relinked since getUniformLocation) → INVALID_OPERATION;
 *    no current program → INVALID_OPERATION.
 *  - TYPE MISMATCH: uniform{1..4}f on a non-float-family uniform (or wrong
 *    component count) → INVALID_OPERATION, store NOT written. Int setters
 *    accept int/uint-excluded int family + bool + samplers; uint setters only
 *    uint types. BOOL uniforms are additionally settable via the float setters
 *    (gl-uniform-bool.html) — stored as 0/1.
 *  - uniform*v length: < k components or not a multiple of k → INVALID_VALUE;
 *    longer than the array needs → the extra values are ignored
 *    (min(len, size*k) written, no error). WebGL2 adds srcOffset/srcLength
 *    (WebGL 2.0 §3.7.6): elements [srcOffset, srcOffset+srcLength) of the
 *    converted array are used (srcLength 0/omitted → to the end); INVALID_VALUE
 *    if srcOffset > data length, srcOffset+srcLength out of bounds, 0 elements,
 *    or a non-multiple count — nothing is written on failure. WebGL1 ignores
 *    the extra arguments (not part of its IDL).
 *  - Sampler uniforms (uniform1i/1iv): EVERY value must be <
 *    MAX_COMBINED_TEXTURE_IMAGE_UNITS (all validated before any write) else
 *    INVALID_VALUE; values land in the int store.
 *  - uniformMatrix*fv: transpose must be false → INVALID_VALUE otherwise;
 *    length < cols*rows or not a multiple → INVALID_VALUE; column-major copy.
 *  - getUniform (api/programs.ts) reads the same store back.
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';
import { C1, C2 } from '../constants';
import { WebGLUniformLocation } from '../objects';
import { programModels, linkGen, locGen, elementSlots, getUniformLocationInfo } from './programs';
import type { Float32List, GLboolean, GLfloat, GLint, GLuint, Int32List, Uint32List } from '../types';

// FLOAT_MAT2x3..FLOAT_MAT4x3 missing from constants.ts (owned elsewhere) —
// local GL values so non-square matrix validation is correct.
const FLOAT_MAT2x3 = 0x8b65;
const FLOAT_MAT2x4 = 0x8b66;
const FLOAT_MAT3x2 = 0x8b67;
const FLOAT_MAT3x4 = 0x8b68;
const FLOAT_MAT4x2 = 0x8b69;
const FLOAT_MAT4x3 = 0x8b6a;

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

// ---- WebIDL *List conversions (pattern from api/vertex-attrib.ts) ----

function toFloat32List(v: Float32List): Float32Array {
  if (v instanceof DataView) throw new TypeError('Argument is not a Float32List');
  if (v instanceof Float32Array) return v;
  if (Array.isArray(v)) return new Float32Array(v as number[]);
  if (ArrayBuffer.isView(v)) return new Float32Array(v as ArrayLike<number>);
  throw new TypeError('Argument is not a Float32List');
}

function toInt32List(v: Int32List): Int32Array {
  if (v instanceof DataView) throw new TypeError('Argument is not an Int32List');
  if (v instanceof Int32Array) return v;
  if (Array.isArray(v)) return Int32Array.from(v as number[]);
  if (ArrayBuffer.isView(v)) return Int32Array.from(v as ArrayLike<number>);
  throw new TypeError('Argument is not an Int32List');
}

function toUint32List(v: Uint32List): Uint32Array {
  if (v instanceof DataView) throw new TypeError('Argument is not a Uint32List');
  if (v instanceof Uint32Array) return v;
  if (Array.isArray(v)) return Uint32Array.from(v as number[]);
  if (ArrayBuffer.isView(v)) return Uint32Array.from(v as ArrayLike<number>);
  throw new TypeError('Argument is not a Uint32List');
}

// ---- Type families ----

function isSamplerType(type: number): boolean {
  switch (type) {
    case C1.SAMPLER_2D:
    case C1.SAMPLER_CUBE:
    case C2.SAMPLER_3D:
    case C2.SAMPLER_2D_ARRAY:
    case C2.SAMPLER_2D_SHADOW:
    case C2.SAMPLER_CUBE_SHADOW:
    case C2.SAMPLER_2D_ARRAY_SHADOW:
    case C2.INT_SAMPLER_2D:
    case C2.INT_SAMPLER_3D:
    case C2.INT_SAMPLER_CUBE:
    case C2.INT_SAMPLER_2D_ARRAY:
    case C2.UNSIGNED_INT_SAMPLER_2D:
    case C2.UNSIGNED_INT_SAMPLER_3D:
    case C2.UNSIGNED_INT_SAMPLER_CUBE:
    case C2.UNSIGNED_INT_SAMPLER_2D_ARRAY:
      return true;
    default:
      return false;
  }
}

/** float setters accept float scalars/vectors + bool family (gl-uniform-bool.html). */
function typeIsFloatFamily(type: number, k: number): boolean {
  switch (type) {
    case C1.FLOAT:
      return k === 1;
    case C1.FLOAT_VEC2:
      return k === 2;
    case C1.FLOAT_VEC3:
      return k === 3;
    case C1.FLOAT_VEC4:
      return k === 4;
    case C1.BOOL:
      return k === 1;
    case C1.BOOL_VEC2:
      return k === 2;
    case C1.BOOL_VEC3:
      return k === 3;
    case C1.BOOL_VEC4:
      return k === 4;
    default:
      return false;
  }
}

/** int setters accept int vectors + bool family + samplers (k=1). */
function typeIsIntFamily(type: number, k: number): boolean {
  switch (type) {
    case C1.INT:
      return k === 1;
    case C1.INT_VEC2:
      return k === 2;
    case C1.INT_VEC3:
      return k === 3;
    case C1.INT_VEC4:
      return k === 4;
    case C1.BOOL:
      return k === 1;
    case C1.BOOL_VEC2:
      return k === 2;
    case C1.BOOL_VEC3:
      return k === 3;
    case C1.BOOL_VEC4:
      return k === 4;
    default:
      return isSamplerType(type) && k === 1;
  }
}

/** uint setters accept uint scalars/vectors only. */
function typeIsUintFamily(type: number, k: number): boolean {
  switch (type) {
    case C1.UNSIGNED_INT:
      return k === 1;
    case C2.UNSIGNED_INT_VEC2:
      return k === 2;
    case C2.UNSIGNED_INT_VEC3:
      return k === 3;
    case C2.UNSIGNED_INT_VEC4:
      return k === 4;
    default:
      return false;
  }
}

// ---- Write target + store writes ----

interface WriteTarget {
  program: import('../objects').WebGLProgram;
  pm: import('../../glsl').Program;
  uniform: import('../../glsl').Program['uniforms'][number];
  slots: number;
  /** Array element the location points at (0 for a bare array name / scalar). */
  elem: number;
}

/**
 * Validate location against the CURRENT program (uniform* writes only ever
 * target the program in use). Returns null for a legal no-op (null location)
 * or after pushing the appropriate error.
 */
function prepareUniform(ctx: WebGLRenderingContext, loc: unknown): WriteTarget | null {
  if (loc === null || loc === undefined) return null; // spec: silently ignored
  if (!(loc instanceof WebGLUniformLocation)) throw new TypeError(`Argument is not of type 'WebGLUniformLocation'`);
  const program = ctx._state.currentProgram;
  if (program === null || program._deleted) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (loc._program !== program) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if ((locGen.get(loc) ?? -1) !== (linkGen.get(program) ?? 0)) {
    ctx._errors.push(C1.INVALID_OPERATION); // location from a previous link
    return null;
  }
  const pm = programModels.get(program);
  if (pm === undefined || program._program === null || !program._linkStatus) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  const uniform = pm.uniforms[loc._index];
  if (uniform === undefined || uniform.blockIndex >= 0) {
    ctx._errors.push(C1.INVALID_OPERATION); // block members are not settable here
    return null;
  }
  // The location's array element (same info getUniform reads back) — writes
  // MUST target that element, not element 0.
  const info = getUniformLocationInfo(loc);
  return { program, pm, uniform, slots: elementSlots(uniform), elem: info.elem };
}

/**
 * Float write at float-index offset: store index = location + slotOffset + comp
 * (matrices pass slotOffset = element*stride + col*4, comp = row). Wrapped so
 * an internal addressing bug can never throw to the page.
 */
function writeFloatAt(t: WriteTarget, slotOffset: number, comp: number, v: number): void {
  const dv = t.program._uniformStore;
  if (dv === null) return; // linked program always has a store; defensive
  try {
    dv.setFloat32((t.uniform.location + slotOffset + comp) * 4, v, true);
  } catch {
    /* internal store write must never escape to the page */
  }
}

/** Int-family write: 'int' ToInt32, 'uint' ToUint32, 'bool' ToInt32≠0, 'boolf' float≠0. */
function writeIntAt(t: WriteTarget, slotOffset: number, comp: number, v: number, mode: 'int' | 'uint' | 'bool' | 'boolf'): void {
  const store = t.pm.intStore;
  const idx = t.uniform.location + slotOffset + comp;
  try {
    switch (mode) {
      case 'int':
        store[idx] = v | 0;
        break;
      case 'uint':
        store[idx] = v >>> 0;
        break;
      case 'bool':
        store[idx] = (v | 0) !== 0 ? 1 : 0;
        break;
      case 'boolf':
        store[idx] = v !== 0 ? 1 : 0;
        break;
    }
  } catch {
    /* internal store write must never escape to the page */
  }
}

function isBoolType(type: number): boolean {
  return type === C1.BOOL || type === C1.BOOL_VEC2 || type === C1.BOOL_VEC3 || type === C1.BOOL_VEC4;
}

// ---- Shared setter bodies (arity preserved by the explicit wrappers below) ----

/** Scalar/component setter (uniform1f..uniform4ui). */
function uniformScalar(ctx: WebGLRenderingContext, location: WebGLUniformLocation | null, k: number, family: 'f' | 'i' | 'ui', vals: number[]): void {
  if (isLost(ctx)) return;
  const t = prepareUniform(ctx, location);
  if (t === null) return;
  const ok =
    family === 'f' ? typeIsFloatFamily(t.uniform.type, k) : family === 'i' ? typeIsIntFamily(t.uniform.type, k) : typeIsUintFamily(t.uniform.type, k);
  if (!ok) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (family === 'i' && isSamplerType(t.uniform.type)) {
    for (const v of vals) {
      if ((v | 0) >= ctx._state.limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
    }
  }
  const isBool = isBoolType(t.uniform.type);
  // Element base in float units (scalar/sampler 1, vector 4); the location may
  // point at an array element ≠ 0.
  const base = t.elem * t.slots;
  for (let c = 0; c < k; c++) {
    const v = vals[c];
    if (family === 'f') {
      if (isBool) writeIntAt(t, base, c, v, 'boolf');
      else writeFloatAt(t, base, c, v);
    } else if (family === 'i') writeIntAt(t, base, c, v, isBool ? 'bool' : 'int');
    else writeIntAt(t, base, c, v, 'uint');
  }
}

/**
 * {v} setter (uniform1fv..uniform4uiv) with WebGL2 srcOffset/srcLength
 * (WebGL 2.0 §3.7.6; CTS gl-uniform-arrays-sub-source.html).
 *
 * WebGL2: `values` is the WebIDL-converted array (JS arrays converted up
 * front); the elements used are [srcOffset, srcOffset + srcLength) when
 * srcLength > 0, else [srcOffset, values.length). INVALID_VALUE if
 * srcOffset > values.length, srcOffset + srcLength > values.length (when
 * srcLength > 0), the element count is 0, or it is not a multiple of the
 * component count k. Nothing is written on failure.
 * WebGL1: srcOffset/srcLength are NOT part of the API — ignored entirely
 * (WebIDL drops the extra arguments; no WebGL1 test passes them).
 */
function uniformVector(
  ctx: WebGLRenderingContext,
  location: WebGLUniformLocation | null,
  k: number,
  family: 'f' | 'i' | 'ui',
  values: Float32Array | Int32Array | Uint32Array,
  srcOffset: GLuint,
  srcLength: GLuint,
): void {
  if (isLost(ctx)) return;
  const t = prepareUniform(ctx, location);
  if (t === null) return;
  const ok =
    family === 'f' ? typeIsFloatFamily(t.uniform.type, k) : family === 'i' ? typeIsIntFamily(t.uniform.type, k) : typeIsUintFamily(t.uniform.type, k);
  if (!ok) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  let off = 0;
  let count = values.length;
  if (ctx._version === 2) {
    off = srcOffset >>> 0; // WebIDL GLuint (ToUint32) conversion
    const len = srcLength >>> 0;
    if (off > values.length || (len > 0 && off + len > values.length)) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    count = len === 0 ? values.length - off : len;
    if (count === 0 || count % k !== 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
  } else if (values.length < k || values.length % k !== 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (family === 'i' && isSamplerType(t.uniform.type)) {
    for (let i = 0; i < count; i++) {
      if ((values[off + i] | 0) >= ctx._state.limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
    }
  }
  const isBool = isBoolType(t.uniform.type);
  // The value run starts at the location's array element (elem), never before
  // the array start; extra values beyond the array end are ignored (WebGL spec
  // §5.14.10 — and must NOT spill into the next uniform's store).
  const total = Math.min(count, (t.uniform.size - t.elem) * k);
  for (let i = 0; i < total; i++) {
    const element = Math.floor(i / k) + t.elem;
    const comp = i % k;
    const v = values[off + i];
    // element offset = element * elementSlots() (float stride: scalar 1,
    // vector 4, matrix cols*4 — see api/programs.ts elementSlots).
    const slotOff = element * t.slots;
    if (family === 'f') {
      if (isBool) writeIntAt(t, slotOff, comp, v, 'boolf');
      else writeFloatAt(t, slotOff, comp, v);
    } else if (family === 'i') writeIntAt(t, slotOff, comp, v, isBool ? 'bool' : 'int');
    else writeIntAt(t, slotOff, comp, v, 'uint');
  }
}

/**
 * Matrix setter (uniformMatrix*fv, incl. non-square) with WebGL2
 * srcOffset/srcLength — same semantics as uniformVector with n = cols*rows
 * components per matrix (WebGL 2.0 §3.7.6; gl-uniform-arrays-sub-source.html).
 */
function uniformMatrix(
  ctx: WebGLRenderingContext,
  location: WebGLUniformLocation | null,
  transpose: GLboolean,
  value: Float32List,
  cols: number,
  rows: number,
  typeConst: number,
  srcOffset: GLuint,
  srcLength: GLuint,
): void {
  if (isLost(ctx)) return;
  const t = prepareUniform(ctx, location);
  if (t === null) return;
  if (t.uniform.type !== typeConst) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (transpose) {
    ctx._errors.push(C1.INVALID_VALUE); // transpose must be FALSE
    return;
  }
  const values = toFloat32List(value);
  const n = cols * rows;
  let off = 0;
  let count = values.length;
  if (ctx._version === 2) {
    off = srcOffset >>> 0; // WebIDL GLuint (ToUint32) conversion
    const len = srcLength >>> 0;
    if (off > values.length || (len > 0 && off + len > values.length)) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    count = len === 0 ? values.length - off : len;
    if (count === 0 || count % n !== 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
  } else if (values.length < n || values.length % n !== 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  // The value run starts at the location's array element (elem); extra
  // matrices beyond the array end are ignored (must not spill into the next
  // uniform's store).
  const total = Math.min(count, (t.uniform.size - t.elem) * n);
  for (let i = 0; i < total; i++) {
    const element = Math.floor(i / n) + t.elem;
    const col = Math.floor((i % n) / rows);
    const row = i % rows;
    // Column-major: value index i = column col, row row; float offset =
    // element*stride + col*4 (stride = cols*4 floats per element).
    writeFloatAt(t, element * t.slots + col * 4, row, values[off + i]);
  }
}

// ---------------------------------------------------------------------------
// installUniformsApi
// ---------------------------------------------------------------------------

export function installUniformsApi(proto: WebGLRenderingContext): void {
  // ---- WebGL1 scalar setters (exact signatures/arity from webgl1.ts) ----
  proto.uniform1f = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLfloat): void {
    uniformScalar(this, location, 1, 'f', [x]);
  };
  proto.uniform2f = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLfloat, y: GLfloat): void {
    uniformScalar(this, location, 2, 'f', [x, y]);
  };
  proto.uniform3f = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLfloat, y: GLfloat, z: GLfloat): void {
    uniformScalar(this, location, 3, 'f', [x, y, z]);
  };
  proto.uniform4f = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLfloat, y: GLfloat, z: GLfloat, w: GLfloat): void {
    uniformScalar(this, location, 4, 'f', [x, y, z, w]);
  };
  proto.uniform1i = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLint): void {
    uniformScalar(this, location, 1, 'i', [x]);
  };
  proto.uniform2i = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLint, y: GLint): void {
    uniformScalar(this, location, 2, 'i', [x, y]);
  };
  proto.uniform3i = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLint, y: GLint, z: GLint): void {
    uniformScalar(this, location, 3, 'i', [x, y, z]);
  };
  proto.uniform4i = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, x: GLint, y: GLint, z: GLint, w: GLint): void {
    uniformScalar(this, location, 4, 'i', [x, y, z, w]);
  };

  // ---- WebGL1 {v} setters (WebGL2 adds srcOffset/srcLength — optional
  // defaults keep the declared arity: WebGL1 IDL (2 args) and WebGL2 IDL
  // (2 required + 2 optional → function.length 2) both hold) ----
  proto.uniform1fv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 1, 'f', toFloat32List(v), srcOffset, srcLength);
  };
  proto.uniform2fv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 2, 'f', toFloat32List(v), srcOffset, srcLength);
  };
  proto.uniform3fv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 3, 'f', toFloat32List(v), srcOffset, srcLength);
  };
  proto.uniform4fv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 4, 'f', toFloat32List(v), srcOffset, srcLength);
  };
  proto.uniform1iv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Int32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 1, 'i', toInt32List(v), srcOffset, srcLength);
  };
  proto.uniform2iv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Int32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 2, 'i', toInt32List(v), srcOffset, srcLength);
  };
  proto.uniform3iv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Int32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 3, 'i', toInt32List(v), srcOffset, srcLength);
  };
  proto.uniform4iv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, v: Int32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformVector(this, location, 4, 'i', toInt32List(v), srcOffset, srcLength);
  };

  // ---- WebGL1 matrix setters ----
  proto.uniformMatrix2fv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformMatrix(this, location, transpose, value, 2, 2, C1.FLOAT_MAT2, srcOffset, srcLength);
  };
  proto.uniformMatrix3fv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformMatrix(this, location, transpose, value, 3, 3, C1.FLOAT_MAT3, srcOffset, srcLength);
  };
  proto.uniformMatrix4fv = function (this: WebGLRenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
    uniformMatrix(this, location, transpose, value, 4, 4, C1.FLOAT_MAT4, srcOffset, srcLength);
  };

  // ---- WebGL2 additions (gated: only present on the WebGL2 prototype) ----
  if ('uniform1ui' in proto) {
    const p2 = proto as unknown as WebGL2RenderingContext;

    p2.uniform1ui = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, x: GLuint): void {
      uniformScalar(this, location, 1, 'ui', [x]);
    };
    p2.uniform2ui = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, x: GLuint, y: GLuint): void {
      uniformScalar(this, location, 2, 'ui', [x, y]);
    };
    p2.uniform3ui = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, x: GLuint, y: GLuint, z: GLuint): void {
      uniformScalar(this, location, 3, 'ui', [x, y, z]);
    };
    p2.uniform4ui = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, x: GLuint, y: GLuint, z: GLuint, w: GLuint): void {
      uniformScalar(this, location, 4, 'ui', [x, y, z, w]);
    };
    p2.uniform1uiv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, v: Uint32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformVector(this, location, 1, 'ui', toUint32List(v), srcOffset, srcLength);
    };
    p2.uniform2uiv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, v: Uint32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformVector(this, location, 2, 'ui', toUint32List(v), srcOffset, srcLength);
    };
    p2.uniform3uiv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, v: Uint32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformVector(this, location, 3, 'ui', toUint32List(v), srcOffset, srcLength);
    };
    p2.uniform4uiv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, v: Uint32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformVector(this, location, 4, 'ui', toUint32List(v), srcOffset, srcLength);
    };

    p2.uniformMatrix2x3fv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformMatrix(this, location, transpose, value, 2, 3, FLOAT_MAT2x3, srcOffset, srcLength);
    };
    p2.uniformMatrix2x4fv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformMatrix(this, location, transpose, value, 2, 4, FLOAT_MAT2x4, srcOffset, srcLength);
    };
    p2.uniformMatrix3x2fv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformMatrix(this, location, transpose, value, 3, 2, FLOAT_MAT3x2, srcOffset, srcLength);
    };
    p2.uniformMatrix3x4fv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformMatrix(this, location, transpose, value, 3, 4, FLOAT_MAT3x4, srcOffset, srcLength);
    };
    p2.uniformMatrix4x2fv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformMatrix(this, location, transpose, value, 4, 2, FLOAT_MAT4x2, srcOffset, srcLength);
    };
    p2.uniformMatrix4x3fv = function (this: WebGL2RenderingContext, location: WebGLUniformLocation | null, transpose: GLboolean, value: Float32List, srcOffset: GLuint = 0, srcLength: GLuint = 0): void {
      uniformMatrix(this, location, transpose, value, 4, 3, FLOAT_MAT4x3, srcOffset, srcLength);
    };
  }
}
