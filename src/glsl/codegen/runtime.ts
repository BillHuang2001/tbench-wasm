/**
 * codegen/runtime.ts — the shared runtime object `R`.
 *
 * Generated shader bodies are compiled via `new Function('ctx', 'R', body)`;
 * `R` is the 2nd argument and holds the builtin helpers that are NOT trivially
 * inlined by codegen/expressions.ts (matrix inverse, pack/unpack, bit
 * conversions, bitfield ops, extended multiply, texture-image resolution).
 *
 * CONVENTIONS
 * - Every helper is a plain function with ZERO per-call allocation (module
 *   level scratch buffers only) — the per-invocation path of generated shaders
 *   must never allocate.
 * - Helpers with a SINGLE scalar result return the number. Helpers with
 *   MULTIPLE results write into a caller-provided output array `(out, off,
 *   ...)`; generated code passes `ctx.scratch` / `ctx.intScratch` offsets.
 *   `out` may be any writable numeric array (Float32Array / Int32Array /
 *   Uint32Array / number[]); values are stored as plain JS numbers and the
 *   array type applies its usual conversion (an int store wraps uint32 values
 *   to int32 — generated code reinterprets with `>>> 0`).
 * - Matrix helpers take their scalar elements in COLUMN-MAJOR storage order
 *   (the order a GLSL matrix occupies in memory: m[0][0], m[1][0], m[2][0],
 *   m[3][0], m[0][1], …) — parameter `a{col}{row}` = element at column
 *   `col`, row `row`. Results are written to `dst` in the same column-major
 *   order, so codegen can copy `dst[off + i]` straight back into its flattened
 *   scalar locals.
 *
 * ROUNDING DECISIONS (pack functions, GLSL ES 3.00 §8.4)
 * - The spec defines packSnorm2x16(c) as `round(clamp(c, -1, +1) * 32767.0)`
 *   (×65535.0 / ×127.0 / ×255.0 for the other packers), and GLSL ES 3.00 §8.3
 *   states that `round()`'s tie behavior (fraction exactly 0.5) is
 *   implementation-defined. This implementation uses round-half-to-even
 *   (banker's rounding), matching the common implementation choice (Mesa's
 *   `_mesa_roundevenf` for snorm/unorm packing) and the later GLSL
 *   clarifications. The dEQP pack tests tolerate ±1 packed unit (highp:
 *   "Rounding only"), so any consistent tie rule passes CTS.
 * - packHalf2x16: IEEE fp32→fp16 with round-to-nearest-even, denormals
 *   flushed per rounding, ±Inf → ±Inf, NaN → quiet NaN (0x7E00, sign
 *   preserved), overflow (|x| ≥ 65520) → ±Inf. unpackHalf2x16 is exact
 *   (fp16→fp32 is lossless).
 * - bitfieldExtract: `bits == 0` → 0; `offset + bits > 32` is UNDEFINED per
 *   spec — treated permissively here (bits clamped to `32 - offset`;
 *   `offset >= 32` → 0). Negative offset/bits are likewise undefined; treated
 *   as offset clamped to 0 / bits ≤ 0 → 0.
 * - Matrix inverse of a singular matrix is UNDEFINED per GLSL — helpers write
 *   zeros and return (documented; callers must not rely on it).
 */
import type { TextureImage } from '../../raster/types.js';

/** A writable numeric array — generated code passes ctx.scratch / ctx.intScratch. */
export interface NumOut {
  [index: number]: number;
}

/**
 * Structural subset of the fragment/vertex exec ctx needed by resolveImage.
 * The exact ctx contract between gl/ and raster/ is being reconciled by the
 * parent manager; this interface is deliberately minimal and defensive.
 */
export interface TextureResolveCtx {
  /** Direct per-unit texture images (reconciled contract). */
  textures?: ArrayLike<TextureImage | null | undefined> | null;
  /** Raster-style environment: `tex.units[unit].img`. */
  tex?: {
    units?: ArrayLike<{ img?: TextureImage | null } | null | undefined> | null;
  } | null;
}

/* ------------------------------------------------------------------ */
/* Module-level scratch: fp32 ↔ int bit conversions + packHalf.        */
/* ------------------------------------------------------------------ */

const bitBuf = new ArrayBuffer(4);
const f32v = new Float32Array(bitBuf);
const i32v = new Int32Array(bitBuf);
const u32v = new Uint32Array(bitBuf);

/** Clamp x into [lo, hi]. */
function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Round half to even (banker's rounding): exact ties (fraction 0.5) round to
 * the nearest EVEN integer. Used for all pack() fixed-point conversions.
 */
function roundEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/* ------------------------------------------------------------------ */
/* Matrix inverse (GLSL ES 3.00 §8.5 inverse())                        */
/* ------------------------------------------------------------------ */

/**
 * Inverse of a 2×2 matrix. Parameters are in column-major storage order
 * (GLSL memory layout): a{col}{row} — a00, a01 (column 0), a10, a11
 * (column 1). Writes the 4 inverse elements to dst[off..off+3] in the same
 * column-major order. Singular matrix (det === 0) → writes zeros
 * (GLSL: undefined behavior).
 */
export function inv2(
  dst: NumOut,
  off: number,
  a00: number,
  a01: number,
  a10: number,
  a11: number,
): void {
  const det = a00 * a11 - a01 * a10;
  if (det === 0) {
    dst[off] = 0;
    dst[off + 1] = 0;
    dst[off + 2] = 0;
    dst[off + 3] = 0;
    return;
  }
  const idet = 1 / det;
  dst[off] = a11 * idet;
  dst[off + 1] = -a01 * idet;
  dst[off + 2] = -a10 * idet;
  dst[off + 3] = a00 * idet;
}

/** Determinant of a 3×3 matrix given in row-major order (9 scalars). */
function det3(
  x00: number,
  x01: number,
  x02: number,
  x10: number,
  x11: number,
  x12: number,
  x20: number,
  x21: number,
  x22: number,
): number {
  return (
    x00 * (x11 * x22 - x12 * x21) -
    x01 * (x10 * x22 - x12 * x20) +
    x02 * (x10 * x21 - x11 * x20)
  );
}

/**
 * Inverse of a 3×3 matrix. Parameters are column-major: a{col}{row}
 * (m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1],
 * m[2][2]).
 * Writes the 9 inverse elements to dst[off..off+8] in column-major order.
 * Singular matrix (det === 0) → writes zeros (GLSL: undefined behavior).
 */
export function inv3(
  dst: NumOut,
  off: number,
  a00: number,
  a01: number,
  a02: number,
  a10: number,
  a11: number,
  a12: number,
  a20: number,
  a21: number,
  a22: number,
): void {
  // Cofactors (undivided): inv[col][row] = (-1)^(row+col) * minor(row, col).
  const c00 = a11 * a22 - a12 * a21; // inv[0][0]
  const c10 = -(a01 * a22 - a02 * a21); // inv[1][0]
  const c20 = a01 * a12 - a02 * a11; // inv[2][0]
  const c01 = -(a10 * a22 - a12 * a20); // inv[0][1]
  const c11 = a00 * a22 - a02 * a20; // inv[1][1]
  const c21 = -(a00 * a12 - a02 * a10); // inv[2][1]
  const c02 = a10 * a21 - a11 * a20; // inv[0][2]
  const c12 = -(a00 * a21 - a01 * a20); // inv[1][2]
  const c22 = a00 * a11 - a01 * a10; // inv[2][2]
  const det = a00 * c00 + a10 * c10 + a20 * c20;
  if (det === 0) {
    for (let i = 0; i < 9; i++) dst[off + i] = 0;
    return;
  }
  const idet = 1 / det;
  dst[off] = c00 * idet;
  dst[off + 1] = c10 * idet;
  dst[off + 2] = c20 * idet;
  dst[off + 3] = c01 * idet;
  dst[off + 4] = c11 * idet;
  dst[off + 5] = c21 * idet;
  dst[off + 6] = c02 * idet;
  dst[off + 7] = c12 * idet;
  dst[off + 8] = c22 * idet;
}

/**
 * Inverse of a 4×4 matrix. Parameters are column-major: a{col}{row}
 * (m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3],
 * m[2][0], …, m[3][3]). Writes the 16 inverse elements to dst[off..off+15]
 * in column-major order. Computed via the adjugate (16 cofactor 3×3
 * determinants, standard cofactor expansion). Singular matrix (det === 0)
 * → writes zeros (GLSL: undefined behavior).
 */
export function inv4(
  dst: NumOut,
  off: number,
  a00: number,
  a01: number,
  a02: number,
  a03: number,
  a10: number,
  a11: number,
  a12: number,
  a13: number,
  a20: number,
  a21: number,
  a22: number,
  a23: number,
  a30: number,
  a31: number,
  a32: number,
  a33: number,
): void {
  // Cofactors (undivided) — each = ±det of the 3×3 minor obtained by
  // removing GLSL row r and column c; inv[col][row] = cof(row, col).
  const c00 = det3(a11, a21, a31, a12, a22, a32, a13, a23, a33); // inv[0][0]
  const c10 = -det3(a01, a21, a31, a02, a22, a32, a03, a23, a33); // inv[1][0]
  const c20 = det3(a01, a11, a31, a02, a12, a32, a03, a13, a33); // inv[2][0]
  const c30 = -det3(a01, a11, a21, a02, a12, a22, a03, a13, a23); // inv[3][0]
  const c01 = -det3(a10, a20, a30, a12, a22, a32, a13, a23, a33); // inv[0][1]
  const c11 = det3(a00, a20, a30, a02, a22, a32, a03, a23, a33); // inv[1][1]
  const c21 = -det3(a00, a10, a30, a02, a12, a32, a03, a13, a33); // inv[2][1]
  const c31 = det3(a00, a10, a20, a02, a12, a22, a03, a13, a23); // inv[3][1]
  const c02 = det3(a10, a20, a30, a11, a21, a31, a13, a23, a33); // inv[0][2]
  const c12 = -det3(a00, a20, a30, a01, a21, a31, a03, a23, a33); // inv[1][2]
  const c22 = det3(a00, a10, a30, a01, a11, a31, a03, a13, a33); // inv[2][2]
  const c32 = -det3(a00, a10, a20, a01, a11, a21, a03, a13, a23); // inv[3][2]
  const c03 = -det3(a10, a20, a30, a11, a21, a31, a12, a22, a32); // inv[0][3]
  const c13 = det3(a00, a20, a30, a01, a21, a31, a02, a22, a32); // inv[1][3]
  const c23 = -det3(a00, a10, a30, a01, a11, a31, a02, a12, a32); // inv[2][3]
  const c33 = det3(a00, a10, a20, a01, a11, a21, a02, a12, a22); // inv[3][3]
  const det = a00 * c00 + a10 * c10 + a20 * c20 + a30 * c30;
  if (det === 0) {
    for (let i = 0; i < 16; i++) dst[off + i] = 0;
    return;
  }
  const idet = 1 / det;
  dst[off] = c00 * idet;
  dst[off + 1] = c10 * idet;
  dst[off + 2] = c20 * idet;
  dst[off + 3] = c30 * idet;
  dst[off + 4] = c01 * idet;
  dst[off + 5] = c11 * idet;
  dst[off + 6] = c21 * idet;
  dst[off + 7] = c31 * idet;
  dst[off + 8] = c02 * idet;
  dst[off + 9] = c12 * idet;
  dst[off + 10] = c22 * idet;
  dst[off + 11] = c32 * idet;
  dst[off + 12] = c03 * idet;
  dst[off + 13] = c13 * idet;
  dst[off + 14] = c23 * idet;
  dst[off + 15] = c33 * idet;
}

/* ------------------------------------------------------------------ */
/* Pack / unpack (GLSL ES 3.00 §8.4 — EXACT semantics)                 */
/* ------------------------------------------------------------------ */

/**
 * packSnorm2x16(x, y) → uint32: clamp each component to [-1, 1], ×32767.0,
 * round-half-to-even (see header). Low 16 bits = x, high 16 bits = y
 * (two's-complement bit patterns). Result is a JS number in [0, 0xFFFFFFFF].
 */
export function packSnorm2x16(x: number, y: number): number {
  return (((packSnorm16(y) << 16) | packSnorm16(x)) >>> 0);
}

/** One snorm16 component: roundEven(clamp(c, -1, 1) * 32767.0) as uint16 bits. */
function packSnorm16(c: number): number {
  return roundEven(clamp(c, -1, 1) * 32767.0) & 0xffff;
}

/**
 * unpackSnorm2x16(v, out, off): writes the two float components (each
 * clamp(int16/32767.0, -1, 1)) to out[off] and out[off + 1].
 */
export function unpackSnorm2x16(v: number, out: NumOut, off: number): void {
  const x = ((v & 0xffff) << 16) >> 16; // sign-extend low 16 bits
  const y = (((v >>> 16) & 0xffff) << 16) >> 16; // sign-extend high 16 bits
  out[off] = clamp(x / 32767.0, -1, 1);
  out[off + 1] = clamp(y / 32767.0, -1, 1);
}

/**
 * packUnorm2x16(x, y) → uint32: clamp each component to [0, 1], ×65535.0,
 * round-half-to-even. Low 16 bits = x, high 16 bits = y.
 */
export function packUnorm2x16(x: number, y: number): number {
  return (((packUnorm16(y) << 16) | packUnorm16(x)) >>> 0);
}

/** One unorm16 component: roundEven(clamp(c, 0, 1) * 65535.0) as uint16 bits. */
function packUnorm16(c: number): number {
  return roundEven(clamp(c, 0, 1) * 65535.0) & 0xffff;
}

/**
 * unpackUnorm2x16(v, out, off): writes the two float components
 * (uint16/65535.0) to out[off] and out[off + 1].
 */
export function unpackUnorm2x16(v: number, out: NumOut, off: number): void {
  out[off] = (v & 0xffff) / 65535.0;
  out[off + 1] = ((v >>> 16) & 0xffff) / 65535.0;
}

/**
 * packHalf2x16(x, y) → uint32: converts each fp32 component to IEEE fp16
 * (round-to-nearest-even, denormals handled, ±Inf → ±Inf, NaN → 0x7E00 with
 * sign preserved, |x| ≥ 65520 → ±Inf). Low 16 bits = x, high 16 bits = y.
 */
export function packHalf2x16(x: number, y: number): number {
  return (((packHalf1(y) << 16) | packHalf1(x)) >>> 0);
}

/** One fp32 → fp16 component (round-to-nearest-even). */
function packHalf1(x: number): number {
  f32v[0] = x;
  const u = u32v[0];
  const sign = (u >>> 16) & 0x8000;
  const abs = u & 0x7fffffff;
  if (abs >= 0x7f800000) {
    // ±Inf or NaN.
    return sign | (abs > 0x7f800000 ? 0x7e00 : 0x7c00);
  }
  if (abs >= 0x477ff000) return sign | 0x7c00; // rounds to ±Inf (≥ 65520)
  if (abs >= 0x38800000) {
    // Normal fp16: keep top 10 mantissa bits with round-half-to-even at bit 13.
    const e = ((abs >>> 23) - 127 + 15) << 10;
    const m = abs & 0x7fffff;
    return sign | e | ((m + 0x0fff + ((m >>> 13) & 1)) >>> 13);
  }
  if (abs < 0x33000000) return sign; // rounds to ±0 (< 2^-25)
  // fp16 denormal: m16 = round-half-even(value * 2^24).
  const shift = 126 - (abs >>> 23); // 14..24
  const mant = 0x800000 | (abs & 0x7fffff);
  return sign | ((mant + ((1 << (shift - 1)) - 1) + ((mant >>> shift) & 1)) >>> shift);
}

/**
 * unpackHalf2x16(v, out, off): writes the two fp32 components (exact fp16→fp32
 * conversion; NaN → NaN, ±Inf → ±Inf, ±0 preserved) to out[off] / out[off+1].
 */
export function unpackHalf2x16(v: number, out: NumOut, off: number): void {
  out[off] = unpackHalf1(v & 0xffff);
  out[off + 1] = unpackHalf1((v >>> 16) & 0xffff);
}

/** One fp16 → fp32 component (exact — fp16 values are losslessly representable). */
function unpackHalf1(h: number): number {
  const sign = h & 0x8000;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) {
    // Denormal (incl. ±0): value = m * 2^-24 — exact in fp32/fp64.
    const val = m * 5.960464477539063e-8; // 2^-24
    return sign ? -val : val;
  }
  if (e === 31) return m === 0 ? (sign ? -Infinity : Infinity) : NaN;
  // Normal: (1 + m/1024) * 2^(e-15) — exact (11 significant bits).
  const val = (1 + m / 1024) * Math.pow(2, e - 15);
  return sign ? -val : val;
}

/**
 * packUnorm4x8(a, b, c, d) → uint32: clamp each to [0, 1], ×255.0,
 * round-half-to-even. a → bits 0-7, b → 8-15, c → 16-23, d → 24-31.
 */
export function packUnorm4x8(a: number, b: number, c: number, d: number): number {
  return (
    (packUnorm8(d) << 24) |
    (packUnorm8(c) << 16) |
    (packUnorm8(b) << 8) |
    packUnorm8(a)
  ) >>> 0;
}

/** One unorm8 component: roundEven(clamp(c, 0, 1) * 255.0) as uint8 bits. */
function packUnorm8(c: number): number {
  return roundEven(clamp(c, 0, 1) * 255.0) & 0xff;
}

/**
 * packSnorm4x8(a, b, c, d) → uint32: clamp each to [-1, 1], ×127.0,
 * round-half-to-even. a → bits 0-7, b → 8-15, c → 16-23, d → 24-31.
 */
export function packSnorm4x8(a: number, b: number, c: number, d: number): number {
  return (
    (packSnorm8(d) << 24) |
    (packSnorm8(c) << 16) |
    (packSnorm8(b) << 8) |
    packSnorm8(a)
  ) >>> 0;
}

/** One snorm8 component: roundEven(clamp(c, -1, 1) * 127.0) as uint8 bits. */
function packSnorm8(c: number): number {
  return roundEven(clamp(c, -1, 1) * 127.0) & 0xff;
}

/** Sign-extend an 8-bit two's-complement value to a full int32. */
function signExtend8(x: number): number {
  return ((x & 0xff) << 24) >> 24;
}

/**
 * unpackUnorm4x8(v, out, off): writes the four float components (uint8/255.0)
 * to out[off .. off+3].
 */
export function unpackUnorm4x8(v: number, out: NumOut, off: number): void {
  out[off] = (v & 0xff) / 255.0;
  out[off + 1] = ((v >>> 8) & 0xff) / 255.0;
  out[off + 2] = ((v >>> 16) & 0xff) / 255.0;
  out[off + 3] = ((v >>> 24) & 0xff) / 255.0;
}

/**
 * unpackSnorm4x8(v, out, off): writes the four float components
 * (clamp(int8/127.0, -1, 1)) to out[off .. off+3].
 */
export function unpackSnorm4x8(v: number, out: NumOut, off: number): void {
  out[off] = clamp(signExtend8(v) / 127.0, -1, 1);
  out[off + 1] = clamp(signExtend8(v >>> 8) / 127.0, -1, 1);
  out[off + 2] = clamp(signExtend8(v >>> 16) / 127.0, -1, 1);
  out[off + 3] = clamp(signExtend8(v >>> 24) / 127.0, -1, 1);
}

/* ------------------------------------------------------------------ */
/* Float / int bit conversions (floatBitsToInt etc.)                   */
/* ------------------------------------------------------------------ */

/**
 * floatBitsToInt: the fp32 bit pattern of x as an int32 (NaN → canonical
 * quiet-NaN pattern 0x7FC00000; input rounded to fp32 first).
 */
export function f2i(x: number): number {
  f32v[0] = x;
  return i32v[0];
}

/** floatBitsToUint: the fp32 bit pattern of x as a uint32 JS number. */
export function f2u(x: number): number {
  f32v[0] = x;
  return u32v[0];
}

/** intBitsToFloat: int32 bit pattern → the corresponding fp32 value. */
export function i2f(x: number): number {
  i32v[0] = x;
  return Math.fround(f32v[0]);
}

/** uintBitsToFloat: uint32 bit pattern → the corresponding fp32 value. */
export function u2f(x: number): number {
  u32v[0] = x;
  return Math.fround(f32v[0]);
}

/* ------------------------------------------------------------------ */
/* Bitfield ops (GLSL ES 3.00 §8.10 — uint32/int32 semantics)          */
/* ------------------------------------------------------------------ */

/**
 * bitfieldExtract (uint version): `(v >> offset) & mask` over the low 32
 * bits, zero-filling. bits == 0 → 0; offset+bits > 32 is undefined per spec,
 * handled permissively (see header).
 */
export function bitfieldExtractU(v: number, offset: number, bits: number): number {
  if (bits <= 0) return 0;
  if (offset >= 32 || offset < 0) return 0;
  const b = Math.min(bits, 32 - offset);
  const mask = b >= 32 ? 0xffffffff : ((1 << b) - 1) >>> 0;
  return ((v >>> offset) & mask) >>> 0;
}

/**
 * bitfieldExtract (int version): extracts `bits` bits at `offset` and
 * sign-extends from bit `bits - 1`. bits == 0 → 0; offset+bits > 32 is
 * undefined per spec, handled permissively.
 */
export function bitfieldExtractI(v: number, offset: number, bits: number): number {
  if (bits <= 0) return 0;
  if (offset >= 32 || offset < 0) return 0;
  const b = Math.min(bits, 32 - offset);
  if (b >= 32) return (v >>> offset) | 0;
  const mask = ((1 << b) - 1) >>> 0;
  const val = ((v >>> offset) & mask) >>> 0;
  const signBit = 1 << (b - 1);
  return (val ^ signBit) - signBit; // sign-extend from bit b-1
}

/**
 * bitfieldInsert(base, insert, offset, bits): clears bits [offset, offset+bits)
 * of `base` and ORs the low `bits` bits of `insert` shifted in. Result as
 * uint32. offset+bits > 32 is undefined per spec, handled permissively.
 */
export function bitfieldInsert(base: number, insert: number, offset: number, bits: number): number {
  if (bits <= 0) return base >>> 0;
  if (offset >= 32 || offset < 0) return base >>> 0;
  const b = Math.min(bits, 32 - offset);
  const m = (b >= 32 ? -1 : (1 << b) - 1) << offset; // int32 mask (bit pattern)
  return (((base | 0) & ~m) | (((insert | 0) << offset) & m)) >>> 0;
}

/** bitfieldReverse: reverses the low 32 bits of v (result as uint32). */
export function bitfieldReverse(v: number): number {
  let x = v >>> 0;
  x = ((x >>> 1) & 0x55555555) | ((x & 0x55555555) << 1);
  x = ((x >>> 2) & 0x33333333) | ((x & 0x33333333) << 2);
  x = ((x >>> 4) & 0x0f0f0f0f) | ((x & 0x0f0f0f0f) << 4);
  x = ((x >>> 8) & 0x00ff00ff) | ((x & 0x00ff00ff) << 8);
  return ((x >>> 16) | (x << 16)) >>> 0;
}

/** bitCount: population count of the low 32 bits of v. */
export function bitCount(v: number): number {
  let x = v >>> 0;
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >>> 24;
}

/**
 * findLSB (uint semantics): index of the least significant 1 bit of the low
 * 32 bits of v; -1 when v == 0.
 */
export function findLSB(v: number): number {
  const x = v >>> 0;
  if (x === 0) return -1;
  return 31 - Math.clz32((x & -x) >>> 0);
}

/**
 * findMSB (uint semantics): index of the most significant 1 bit of the low
 * 32 bits of v; -1 when v == 0.
 */
export function findMSB(v: number): number {
  const x = v >>> 0;
  return x === 0 ? -1 : 31 - Math.clz32(x);
}

/**
 * findMSB (int semantics): for x > 0 the most significant 1 bit; for x < 0
 * the position of the most significant 0 bit (per GLSL ES 3.00 §8.10); 0 →
 * -1; -1 (all bits set, no 0 bit) → -1.
 */
export function findMSBI(v: number): number {
  const x = v | 0;
  if (x > 0) return 31 - Math.clz32(x);
  if (x < 0) {
    const inv = (~x) >>> 0;
    return inv === 0 ? -1 : 31 - Math.clz32(inv);
  }
  return -1;
}

/**
 * uaddCarry(x, y, out, off): writes the low 32 bits of x + y to out[off] and
 * the carry (0 or 1) to out[off + 1] (uint32 semantics).
 */
export function uaddCarry(x: number, y: number, out: NumOut, off: number): void {
  const s = x + y;
  out[off] = s % 0x100000000;
  out[off + 1] = s >= 0x100000000 ? 1 : 0;
}

/**
 * usubBorrow(x, y, out, off): writes (x - y) mod 2^32 to out[off] and the
 * borrow (0 or 1) to out[off + 1] (uint32 semantics).
 */
export function usubBorrow(x: number, y: number, out: NumOut, off: number): void {
  const d = x - y;
  if (d < 0) {
    out[off] = d + 0x100000000;
    out[off + 1] = 1;
  } else {
    out[off] = d;
    out[off + 1] = 0;
  }
}

/**
 * umulExtended(x, y, out, off): 64-bit product of two uint32 values; writes
 * the high 32 bits to out[off] and the low 32 bits to out[off + 1].
 * 16-bit-limb math keeps every intermediate exact in fp64.
 */
export function umulExtended(x: number, y: number, out: NumOut, off: number): void {
  const xl = x & 0xffff;
  const xh = (x >>> 16) & 0xffff;
  const yl = y & 0xffff;
  const yh = (y >>> 16) & 0xffff;
  let lo = xl * yl;
  let mid = xl * yh + xh * yl;
  let hi = xh * yh;
  const lo32 = lo & 0xffff;
  mid += lo >>> 16;
  const mid32 = mid & 0xffff;
  hi += (mid / 65536) | 0; // exact carry: floor(mid / 2^16); mid < 2^33
  out[off] = hi;
  out[off + 1] = ((mid32 << 16) | lo32) >>> 0;
}

/**
 * imulExtended(x, y, out, off): signed 64-bit product of two int32 values;
 * writes the high 32 bits (int32) to out[off] and the low 32 bits (uint32
 * pattern) to out[off + 1]. Computed as the unsigned product of the
 * magnitudes with two's-complement negation applied when signs differ.
 */
export function imulExtended(x: number, y: number, out: NumOut, off: number): void {
  const neg = (x < 0) !== (y < 0);
  const ax = x < 0 ? -x : x;
  const ay = y < 0 ? -y : y;
  const xl = ax & 0xffff;
  const xh = (ax / 65536) | 0;
  const yl = ay & 0xffff;
  const yh = (ay / 65536) | 0;
  let lo = xl * yl;
  let mid = xl * yh + xh * yl;
  let hi = xh * yh;
  const lo32 = lo & 0xffff;
  mid += lo >>> 16;
  const mid32 = mid & 0xffff;
  hi += (mid / 65536) | 0;
  let hi32 = hi;
  let loBits = ((mid32 << 16) | lo32) >>> 0;
  if (neg) {
    // Two's-complement negate the 64-bit (hi32:loBits) pair.
    const carry = loBits === 0 ? 1 : 0;
    loBits = ((~loBits + 1) & 0xffffffff) >>> 0;
    hi32 = ((~hi32 + carry) & 0xffffffff) >>> 0;
  }
  out[off] = hi32 | 0; // int32
  out[off + 1] = loBits;
}

/* ------------------------------------------------------------------ */
/* Texture image resolution (internal; used by the texture helpers)    */
/* ------------------------------------------------------------------ */

/**
 * resolveImage(ctx, unit) → TextureImage | null: defensive lookup of the
 * texture image bound to a sampler unit, supporting both the direct
 * `ctx.textures[unit]` contract and the raster-style `ctx.tex.units[unit].img`
 * contract (the gl/raster ctx contract is being reconciled by the parent
 * manager). Unbound/invalid units → null.
 */
export function resolveImage(ctx: TextureResolveCtx, unit: number): TextureImage | null {
  const t = ctx.textures && ctx.textures[unit];
  if (t) return t;
  const units = ctx.tex && ctx.tex.units;
  const b = units && units[unit];
  const img = b && b.img;
  return img || null;
}

/* ------------------------------------------------------------------ */
/* The runtime object handed to generated code                         */
/* ------------------------------------------------------------------ */

/** The shared runtime object passed as the 2nd arg of new Function('ctx','R', body). */
export const R: Readonly<Record<string, Function>> = {
  inv2,
  inv3,
  inv4,
  packSnorm2x16,
  unpackSnorm2x16,
  packUnorm2x16,
  unpackUnorm2x16,
  packHalf2x16,
  unpackHalf2x16,
  packUnorm4x8,
  unpackUnorm4x8,
  packSnorm4x8,
  unpackSnorm4x8,
  f2i,
  f2u,
  i2f,
  u2f,
  bitfieldExtractU,
  bitfieldExtractI,
  bitfieldInsert,
  bitfieldReverse,
  bitCount,
  findLSB,
  findMSB,
  findMSBI,
  uaddCarry,
  usubBorrow,
  umulExtended,
  imulExtended,
  resolveImage,
};