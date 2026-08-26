/**
 * misc.ts — scalar utilities: exact GLSL ES scalar built-ins, power-of-two
 * helpers, IEEE-754 bit reinterpretation (float32 ↔ int32/uint32), binary16
 * (half-float) pack/unpack, and GLSL ES 3.00 bitfield built-ins.
 *
 * The GLSL-named functions here are the single source of truth for scalar
 * semantics — glsl codegen may emit them directly (or inline their bodies,
 * but must match these definitions exactly). Everything is a pure function.
 *
 * Bit-reinterpretation helpers use one shared module-level DataView scratch.
 * This is the only (immutable) hidden state: results are pure; JS is
 * single-threaded so no re-entrancy is possible.
 */

/** Shared scratch for IEEE-754 bit reinterpretation (see file header). */
const scratch = new DataView(new ArrayBuffer(4));

/**
 * GLSL `clamp(x, lo, hi)` = min(max(x, lo), hi). Undefined in GLSL when
 * lo > hi; result follows the min(max) evaluation.
 */
export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}

/**
 * clamp(x, 0, 1).
 */
export function clamp01(x: number): number {
  return Math.min(Math.max(x, 0), 1);
}

/**
 * True iff `n` is a power of two (n >= 1). isPow2(0) = false.
 */
export function isPow2(n: number): boolean {
  return n >= 1 && (n & (n - 1)) === 0;
}

/**
 * Smallest power of two >= `n`. For n <= 0 returns 1. Used for NPOT
 * mipmap/level sizing and texture dimension validation.
 */
export function nextPow2(n: number): number {
  if (n <= 0) return 1;
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/**
 * floor(log2(n)) for n >= 1 (integer). Used for mipmap level math and
 * anisotropy LOD computation.
 */
export function log2Int(n: number): number {
  return 31 - Math.clz32(n);
}

/**
 * Rounds `n` up to the next multiple of `alignment` (power-of-two aligned
 * buffers: UBO std140, pixel pack/unpack, row strides).
 */
export function alignUp(n: number, alignment: number): number {
  return Math.ceil(n / alignment) * alignment;
}

/**
 * GLSL `fract(x)` = x - floor(x).
 */
export function fract(x: number): number {
  return x - Math.floor(x);
}

/**
 * GLSL `mod(x, y)` = x - y * floor(x / y). NOT the JS % operator; y must
 * not be 0 (undefined in GLSL).
 */
export function mod(x: number, y: number): number {
  return x - y * Math.floor(x / y);
}

/**
 * GLSL `mix(x, y, t)` = x * (1 - t) + y * t.
 */
export function mix(x: number, y: number, t: number): number {
  return x * (1 - t) + y * t;
}

/**
 * GLSL `step(edge, x)` = x < edge ? 0 : 1.
 */
export function step(edge: number, x: number): number {
  return x < edge ? 0 : 1;
}

/**
 * GLSL `smoothstep(e0, e1, x)`:
 *   t = clamp((x - e0) / (e1 - e0), 0, 1); result = t*t*(3 - 2*t).
 * e0 == e1 is undefined in GLSL; this implementation returns 0 (no NaN).
 */
export function smoothstep(e0: number, e1: number, x: number): number {
  if (e0 === e1) return 0;
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * GLSL `roundEven(x)`: round to nearest integer, ties to even.
 */
export function roundEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  let r: number;
  if (d < 0.5) {
    r = f;
  } else if (d > 0.5) {
    r = f + 1;
  } else {
    // Exact tie: round to the even neighbor (f % 2 keeps the sign, so -0
    // compares equal to 0 for even f).
    r = f % 2 === 0 ? f : f + 1;
  }
  // IEEE ties-to-even: -0.5 → -0 (and any negative x rounding to zero gives
  // negative zero, matching roundToIntegralTiesToEven).
  if (r === 0 && x < 0) return -0;
  return r;
}

/**
 * GLSL `sign(x)`: 1 if x > 0, -1 if x < 0, 0 if x == 0 (Math.sign).
 */
export function sign(x: number): number {
  return Math.sign(x);
}

/**
 * Converts a float32 value to its IEEE 754 binary16 (half-float) bit pattern
 * (uint16). Round-to-nearest-even; handles subnormals, ±Inf, NaN per spec.
 * Used by float16 formats and GLSL `packHalf2x16`.
 */
export function toHalfFloat(f: number): number {
  scratch.setFloat32(0, f, false);
  const x = scratch.getUint32(0, false);
  const signBits = (x >>> 16) & 0x8000;
  const e = (x >>> 23) & 0xff;
  const m = x & 0x7fffff;
  if (e === 0xff) {
    // Inf or NaN: half exponent all ones; keep the top 10 mantissa bits
    // (any NaN pattern is acceptable, sign preserved).
    return signBits | 0x7c00 | (m >>> 13);
  }
  const exp = e - 112; // half exponent (e - 127 + 15)
  if (exp >= 31) return signBits | 0x7c00; // overflow → ±Inf
  if (exp > 0) {
    // Normal half: round the 23-bit mantissa to 10 bits, ties to even.
    let keep = m >>> 13;
    const dropped = m & 0x1fff;
    if (dropped > 0x1000 || (dropped === 0x1000 && (keep & 1) === 1)) keep++;
    if (keep === 0x400) return signBits | ((exp + 1) << 10); // mantissa overflow → next exponent
    return signBits | (exp << 10) | keep;
  }
  // exp <= 0: subnormal half or zero. (e === 0 gives exp = -112 < -10 → zero.)
  if (exp < -10) return signBits; // value < 2^-25 → rounds to ±0
  const mant24 = 0x800000 | m; // implicit leading 1 → 24-bit significand
  const shift = 14 - exp; // 14..24
  let half = mant24 >>> shift;
  const dropped = mant24 & ((1 << shift) - 1);
  const threshold = 1 << (shift - 1);
  if (dropped > threshold || (dropped === threshold && (half & 1) === 1)) half++;
  if (half === 0x400) return signBits | 0x400; // rounded up to smallest normal half (2^-14)
  return signBits | half;
}

/**
 * Converts an IEEE 754 binary16 bit pattern (uint16) to a float32 value.
 * Handles subnormals, ±Inf, NaN per spec. Used by float16 formats and GLSL
 * `unpackHalf2x16`.
 */
export function fromHalfFloat(h: number): number {
  const hh = h & 0xffff;
  const signBits = (hh & 0x8000) << 16;
  const exp = (hh >>> 10) & 0x1f;
  const mant = hh & 0x3ff;
  if (exp === 0x1f) {
    if (mant === 0) return signBits !== 0 ? -Infinity : Infinity;
    // NaN: float32 exponent all ones, nonzero mantissa (any pattern, sign kept).
    scratch.setUint32(0, signBits | 0x7f800000 | (mant << 13), false);
    return scratch.getFloat32(0, false);
  }
  if (exp === 0) {
    if (mant === 0) return signBits !== 0 ? -0 : 0;
    // Subnormal: value = mant * 2^-24; exactly representable in float32
    // (mant < 2^10 significant bits) and in float64.
    const v = mant * Math.pow(2, -24);
    return signBits !== 0 ? -v : v;
  }
  // Normal: value = (1 + mant/2^10) * 2^(exp-15) → float32 exponent exp + 112.
  scratch.setUint32(0, signBits | ((exp + 112) << 23) | (mant << 13), false);
  return scratch.getFloat32(0, false);
}

/**
 * Reinterprets the float32 bit pattern of `f` as a signed int32 (GLSL ES
 * `floatBitsToInt`). Uses a shared DataView scratch — not reentrant but JS
 * is single-threaded.
 */
export function floatToIntBits(f: number): number {
  scratch.setFloat32(0, f, false);
  return scratch.getInt32(0, false);
}

/**
 * Reinterprets the int32 bit pattern of `i` as a float32 (GLSL ES
 * `intBitsToFloat`).
 */
export function intBitsToFloat(i: number): number {
  scratch.setInt32(0, i | 0, false);
  return scratch.getFloat32(0, false);
}

/**
 * Reinterprets the float32 bit pattern of `f` as an unsigned int32 (GLSL ES
 * `floatBitsToUint`).
 */
export function floatToUintBits(f: number): number {
  scratch.setFloat32(0, f, false);
  return scratch.getUint32(0, false);
}

/**
 * Reinterprets the uint32 bit pattern of `u` as a float32 (GLSL ES
 * `uintBitsToFloat`).
 */
export function uintBitsToFloat(u: number): number {
  scratch.setUint32(0, u >>> 0, false);
  return scratch.getFloat32(0, false);
}

/**
 * GLSL ES 3.00 `bitfieldExtract(value, offset, bits)`: extracts `bits` bits
 * of `value` starting at `offset`; when `signed` is true the result is
 * sign-extended. All arithmetic is 32-bit (values coerced via |0 / >>>0).
 */
export function bitfieldExtract(value: number, offset: number, bits: number, signed: boolean): number {
  const v = value | 0;
  // bits == 32 must not shift by 32 (JS shift counts are mod 32), so clamp
  // the mask to -1 in that case.
  const mask = bits >= 32 ? -1 : (1 << bits) - 1;
  let r = (v >>> offset) & mask;
  if (signed) {
    // Sign-extend: signBit = 1 << (bits - 1); (x ^ signBit) - signBit is
    // exact for bits == 32 too (signBit is then 1 << 31, an exact int32).
    const signBit = bits >= 32 ? 1 << 31 : 1 << (bits - 1);
    r = (r ^ signBit) - signBit;
  }
  return signed ? r | 0 : r >>> 0;
}

/**
 * GLSL ES 3.00 `bitfieldInsert(base, insert, offset, bits)`: inserts the low
 * `bits` bits of `insert` into `base` at `offset`. 32-bit semantics.
 */
export function bitfieldInsert(base: number, insert: number, offset: number, bits: number): number {
  const b = base | 0;
  const i = insert | 0;
  if (bits >= 32) return i | 0;
  const mask = (1 << bits) - 1;
  return ((b & ~(mask << offset)) | ((i & mask) << offset)) | 0;
}

/**
 * GLSL ES 3.00 `bitCount(value)`: population count of the 32-bit pattern of
 * `value` (unsigned interpretation).
 */
export function bitCount(value: number): number {
  let v = value >>> 0;
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

/**
 * GLSL ES 3.00 `findLSB(value)`: index of the least significant set bit of
 * the 32-bit pattern, or -1 when no bit is set.
 */
export function findLSB(value: number): number {
  const v = value >>> 0;
  if (v === 0) return -1;
  // v ^ (v - 1) sets bits 0..k where k is the lowest set bit of v.
  return 31 - Math.clz32((v ^ (v - 1)) >>> 0);
}

/**
 * GLSL ES 3.00 `findMSB(value)`: index of the most significant set bit, or
 * -1 when no bit is set. For signed interpretation: negative values report
 * the MSB of the bitwise complement (per GLSL spec, findMSB(-1) = -1).
 *
 * Signed/unsigned overload discrimination: the sign of the JS number selects
 * the overload — a negative number is a signed int (bit 31 set) and reports
 * the most significant ZERO bit (i.e. MSB of ~value); a non-negative number
 * is the uint32 pattern and reports the most significant ONE bit. So
 * findMSB(-1) = -1, findMSB(-2) = 0 (since ~(-2) = 1), and
 * findMSB(0x80000000) = 31 (uint) while findMSB(-0x80000000) = 30 (signed).
 */
export function findMSB(value: number): number {
  const u = value >>> 0; // uint32 pattern
  if (u === 0) return -1;
  if (value < 0) {
    const c = ~u >>> 0; // complement: most significant zero bit of the signed pattern
    if (c === 0) return -1; // value === -1
    return 31 - Math.clz32(c);
  }
  return 31 - Math.clz32(u);
}

/**
 * Development assertion: throws an Error with `msg` (default 'assertion
 * failed') when `cond` is falsy. Compiled out or kept per build flags —
 * NEVER used for user-facing validation (GL errors handle that); only for
 * internal invariants. When assertions are disabled this becomes a no-op.
 */
export function assert(cond: unknown, msg?: string): asserts cond {
  if (!cond) throw new Error(msg ?? 'assertion failed');
}
