/**
 * typedarray.ts — typed-array plumbing shared by every module.
 *
 * Provides: a canonical kind registry (string ↔ constructor) used by gl/
 * (vertex attrib types, readPixels pack types, texture upload), view
 * construction over shared ArrayBuffers (texture levels, FBO surfaces,
 * uniform stores), element-wise copies across different typed-array kinds
 * (format conversion plumbing), and byte-level reinterpretation (upload to
 * GPU-less surfaces, present blit).
 *
 * Pure functions; no state. No DOM, no GL enums — kinds are strings, the GL
 * layer maps its GLenum types onto `TypedArrayKind`.
 */

/** Canonical string names for all supported typed-array kinds. */
export type TypedArrayKind =
  | 'int8'
  | 'uint8'
  | 'uint8clamped'
  | 'int16'
  | 'uint16'
  | 'int32'
  | 'uint32'
  | 'float32'
  | 'float64';

/** Union of all supported typed-array instances. */
export type TypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

/** Constructor signature used by the registry (length-constructed). */
export type TypedArrayCtor = new (length: number) => TypedArray;

/** Signature of the native (buffer, byteOffset?, length?) constructors. */
type ViewCtor = new (
  buffer: ArrayBuffer,
  byteOffset?: number,
  length?: number
) => TypedArray;

/** Inclusive value ranges of the integer kinds (for clamp conversion). */
const INT_RANGES: Record<string, [number, number]> = {
  int8: [-128, 127],
  uint8: [0, 255],
  uint8clamped: [0, 255],
  int16: [-32768, 32767],
  uint16: [0, 65535],
  int32: [-2147483648, 2147483647],
  uint32: [0, 4294967295],
};

/**
 * Map of `TypedArrayKind` → constructor. Shared table; do not mutate.
 */
export const TYPED_ARRAY_CTORS: Record<TypedArrayKind, TypedArrayCtor> = {
  int8: Int8Array,
  uint8: Uint8Array,
  uint8clamped: Uint8ClampedArray,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  float32: Float32Array,
  float64: Float64Array,
};

/**
 * Returns the constructor for `kind` (same as TYPED_ARRAY_CTORS lookup).
 */
export function typedArrayCtor(kind: TypedArrayKind): TypedArrayCtor {
  return TYPED_ARRAY_CTORS[kind];
}

/**
 * Allocates a new typed array of `kind` with `length` elements, zero-filled.
 */
export function makeTypedArray(kind: TypedArrayKind, length: number): TypedArray {
  return new TYPED_ARRAY_CTORS[kind](length);
}

/**
 * Creates a typed-array view of `kind` over `buffer` starting at `byteOffset`
 * (default 0) with `length` elements (default: the remainder of the buffer,
 * aligned to the element size). No copy — the view aliases `buffer`.
 * Throws RangeError on misalignment / out-of-bounds, like native ctors.
 */
export function viewOf(
  buffer: ArrayBuffer,
  kind: TypedArrayKind,
  byteOffset = 0,
  length?: number
): TypedArray {
  const Ctor = TYPED_ARRAY_CTORS[kind] as unknown as ViewCtor;
  return length === undefined ? new Ctor(buffer, byteOffset) : new Ctor(buffer, byteOffset, length);
}

/**
 * Element-wise copy: dst[dstOffset + i] = src[srcOffset + i] for i in
 * [0, length). Works across different typed-array kinds; values convert per
 * normal JS typed-array assignment semantics (float → int truncates toward
 * zero, out-of-range clamps). `length` defaults to src.length - srcOffset.
 * No-op when length <= 0.
 */
export function copyTypedArray(
  dst: TypedArray,
  dstOffset: number,
  src: TypedArray,
  srcOffset = 0,
  length?: number
): void {
  const n = length === undefined ? src.length - srcOffset : length;
  if (n <= 0) return;
  const kind = typedArrayKindOf(dst);
  if (kind === 'float32' || kind === 'float64') {
    // Float destinations: plain assignment (float64 → float32 rounding).
    for (let i = 0; i < n; i++) {
      dst[dstOffset + i] = src[srcOffset + i];
    }
    return;
  }
  // Integer destinations per the JSDoc contract: NaN → 0, values truncate
  // toward zero and out-of-range values clamp to the destination range
  // (deliberately NOT native wrap-around, which would turn -1 into 255 for
  // Uint8Array; GL format conversion needs clamp semantics).
  const [min, max] = INT_RANGES[kind];
  for (let i = 0; i < n; i++) {
    const v = src[srcOffset + i];
    dst[dstOffset + i] = v !== v ? 0 : v < min ? min : v > max ? max : Math.trunc(v);
  }
}

/**
 * Raw byte copy: copies `byteLength` bytes from src (ArrayBuffer or a
 * Uint8Array view) at `srcByteOffset` into dst at `dstByteOffset`. The
 * buffers may overlap; copy is performed as if via memmove.
 */
export function copyBytes(
  dst: ArrayBuffer | Uint8Array,
  dstByteOffset: number,
  src: ArrayBuffer | Uint8Array,
  srcByteOffset: number,
  byteLength: number
): void {
  if (byteLength <= 0) return;
  const dstIsView = dst instanceof Uint8Array;
  const srcIsView = src instanceof Uint8Array;
  const dstBuf = dstIsView ? dst.buffer : dst;
  const dstOff = dstIsView ? dst.byteOffset + dstByteOffset : dstByteOffset;
  const srcBuf = srcIsView ? src.buffer : src;
  const srcOff = srcIsView ? src.byteOffset + srcByteOffset : srcByteOffset;
  // Uint8Array.prototype.set has memmove semantics for overlapping regions
  // (the copy behaves as if routed through a temporary).
  new Uint8Array(dstBuf, dstOff, byteLength).set(new Uint8Array(srcBuf, srcOff, byteLength));
}

/**
 * Sets every element of `arr` to `value`.
 */
export function fillTypedArray(arr: TypedArray, value: number): void {
  arr.fill(value);
}

/**
 * Zero-fills `arr` (optimized path: fill(0)).
 */
export function zeroTypedArray(arr: TypedArray): void {
  arr.fill(0);
}

/**
 * Bytes per element of `arr` (arr.BYTES_PER_ELEMENT).
 */
export function bytesPerElement(arr: TypedArray): number {
  return arr.BYTES_PER_ELEMENT;
}

/**
 * Total byte length of the viewed region (arr.byteLength).
 */
export function byteLengthOf(arr: TypedArray): number {
  return arr.byteLength;
}

/**
 * Returns a Uint8Array view over the SAME memory as `arr` (no copy),
 * honoring arr's byteOffset and byteLength. Useful for texture uploads,
 * surface blits, and presenting pixels.
 */
export function typedArrayToBytes(arr: TypedArray): Uint8Array {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

/**
 * Deep-copies `arr` into a new typed array of the same kind and length.
 */
export function cloneTypedArray<T extends TypedArray>(arr: T): T {
  return arr.slice() as T;
}

/**
 * Concatenates same-kind typed arrays into one new typed array of that kind
 * (total length = sum of lengths). All arrays must share the same concrete
 * kind; throws TypeError otherwise.
 */
export function concatTypedArrays<T extends TypedArray>(...arrays: T[]): T {
  if (arrays.length === 0) {
    throw new TypeError('concatTypedArrays: at least one array is required');
  }
  const Ctor = arrays[0].constructor as new (length: number) => T;
  let total = 0;
  for (const arr of arrays) {
    if (arr.constructor !== Ctor) {
      throw new TypeError('concatTypedArrays: all arrays must be of the same kind');
    }
    total += arr.length;
  }
  const out = new Ctor(total);
  let offset = 0;
  for (const arr of arrays) {
    out.set(arr, offset);
    offset += arr.length;
  }
  return out;
}

/**
 * Type guard: true iff `v` is a typed array (ArrayBuffer view that is not a
 * DataView). Used by gl/ object validation (e.g. bufferData/bufferSubData
 * argument checks) and by the harness.
 */
export function isTypedArray(v: unknown): v is TypedArray {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

/**
 * Reverse lookup: the `TypedArrayKind` of `arr` (its concrete ctor), or
 * throws TypeError for unknown kinds (should be impossible for TypedArray).
 */
export function typedArrayKindOf(arr: TypedArray): TypedArrayKind {
  const Ctor = arr.constructor;
  for (const kind of Object.keys(TYPED_ARRAY_CTORS) as TypedArrayKind[]) {
    if (TYPED_ARRAY_CTORS[kind] === Ctor) return kind;
  }
  throw new TypeError('typedArrayKindOf: unknown typed-array kind');
}
