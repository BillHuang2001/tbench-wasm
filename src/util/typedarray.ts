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
  throw new Error('not implemented');
}

/**
 * Allocates a new typed array of `kind` with `length` elements, zero-filled.
 */
export function makeTypedArray(kind: TypedArrayKind, length: number): TypedArray {
  throw new Error('not implemented');
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
  throw new Error('not implemented');
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
  throw new Error('not implemented');
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
  throw new Error('not implemented');
}

/**
 * Sets every element of `arr` to `value`.
 */
export function fillTypedArray(arr: TypedArray, value: number): void {
  throw new Error('not implemented');
}

/**
 * Zero-fills `arr` (optimized path: fill(0)).
 */
export function zeroTypedArray(arr: TypedArray): void {
  throw new Error('not implemented');
}

/**
 * Bytes per element of `arr` (arr.BYTES_PER_ELEMENT).
 */
export function bytesPerElement(arr: TypedArray): number {
  throw new Error('not implemented');
}

/**
 * Total byte length of the viewed region (arr.byteLength).
 */
export function byteLengthOf(arr: TypedArray): number {
  throw new Error('not implemented');
}

/**
 * Returns a Uint8Array view over the SAME memory as `arr` (no copy),
 * honoring arr's byteOffset and byteLength. Useful for texture uploads,
 * surface blits, and presenting pixels.
 */
export function typedArrayToBytes(arr: TypedArray): Uint8Array {
  throw new Error('not implemented');
}

/**
 * Deep-copies `arr` into a new typed array of the same kind and length.
 */
export function cloneTypedArray<T extends TypedArray>(arr: T): T {
  throw new Error('not implemented');
}

/**
 * Concatenates same-kind typed arrays into one new typed array of that kind
 * (total length = sum of lengths). All arrays must share the same concrete
 * kind; throws TypeError otherwise.
 */
export function concatTypedArrays<T extends TypedArray>(...arrays: T[]): T {
  throw new Error('not implemented');
}

/**
 * Type guard: true iff `v` is a typed array (ArrayBuffer view that is not a
 * DataView). Used by gl/ object validation (e.g. bufferData/bufferSubData
 * argument checks) and by the harness.
 */
export function isTypedArray(v: unknown): v is TypedArray {
  throw new Error('not implemented');
}

/**
 * Reverse lookup: the `TypedArrayKind` of `arr` (its concrete ctor), or
 * throws TypeError for unknown kinds (should be impossible for TypedArray).
 */
export function typedArrayKindOf(arr: TypedArray): TypedArrayKind {
  throw new Error('not implemented');
}
