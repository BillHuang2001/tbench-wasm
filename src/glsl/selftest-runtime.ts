/**
 * selftest-runtime.ts — direct unit checks for codegen/runtime.ts (the shared
 * runtime object `R` passed as the 2nd arg of new Function('ctx','R', body)).
 *
 * Run: npx tsx src/glsl/selftest-runtime.ts
 *
 * Verifies: matrix inverse (inv2/inv3/inv4) against an independent generic
 * adjugate reference; pack/unpack functions (exact bit patterns for the
 * spec's canonical values, round-trip error bounds over a deterministic LCG,
 * and fp16 edge cases: ±0, 65504, ±Inf, NaN, denormals); float↔int bit
 * conversions; all bitfield ops (incl. the int/uint findMSB split and
 * extended 64-bit multiplies vs BigInt references); resolveImage ctx paths;
 * and that codegen/index.ts re-exports the SAME R object. Prints "OK" and
 * exits 0 on success.
 */
import {
  R,
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
} from './codegen/runtime.js';
import { R as RFromIndex } from './codegen/index.js';
import type { TextureImage } from '../raster/types.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function near(a: number, b: number, eps: number): boolean {
  return Math.abs(a - b) <= eps;
}

/* Deterministic LCG (Numerical Recipes constants). */
let seed = 0x12345678;
function lcg(): number {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}
function lcgInt(): number {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed;
}

const out: number[] = [0, 0, 0, 0, 0, 0, 0, 0];

/* ------------------------------------------------------------------ */
/* pack / unpack — exact values                                        */
/* ------------------------------------------------------------------ */

check(packUnorm2x16(0.5, 1.0) === 0xffff8000, `packUnorm2x16(0.5, 1.0): got 0x${packUnorm2x16(0.5, 1.0).toString(16)}`);
check(packUnorm2x16(0, 0) === 0, 'packUnorm2x16(0, 0) === 0');
check(packUnorm2x16(1, 1) === 0xffffffff, 'packUnorm2x16(1, 1) === 0xFFFFFFFF');
check(packUnorm2x16(-1, 2) === 0xffff0000, 'packUnorm2x16(-1, 2) clamps to (0, 1)');

// -1.0 * 32767.0 = -32767 → 16-bit two's complement 0x8001 (NOT 0x8000).
check(packSnorm2x16(1.0, -1.0) === 0x80017fff, `packSnorm2x16(1.0, -1.0): got 0x${packSnorm2x16(1.0, -1.0).toString(16)}`);
// 0.5*32767 = 16383.5 → tie → even → 16384 (0x4000); -0.5 → -16384 (0xC000).
check(packSnorm2x16(0.5, -0.5) === 0xc0004000, `packSnorm2x16(0.5, -0.5): got 0x${packSnorm2x16(0.5, -0.5).toString(16)}`);
check(packSnorm2x16(0, 0) === 0, 'packSnorm2x16(0, 0) === 0');

check(packHalf2x16(1.0, -2.0) === 0xc0003c00, `packHalf2x16(1.0, -2.0): got 0x${packHalf2x16(1.0, -2.0).toString(16)}`);
check(packHalf2x16(0.0, 0.0) === 0, 'packHalf2x16(0, 0) === 0');
check(packHalf2x16(-0.0, 0.0) === 0x00008000, 'packHalf2x16(-0.0, 0.0) keeps the sign in the low half');
check(packHalf2x16(0.0, -0.0) === 0x80000000, 'packHalf2x16(0.0, -0.0) keeps the sign in the high half');
check(packHalf2x16(65504, 65504) === 0x7bff7bff, `packHalf2x16(65504): got 0x${packHalf2x16(65504, 65504).toString(16)}`);
check(packHalf2x16(Infinity, -Infinity) === 0xfc007c00, 'packHalf2x16(Inf, -Inf) === 0xFC007C00');
check((packHalf2x16(NaN, 0) & 0xffff) === 0x7e00, 'packHalf2x16(NaN) low half === 0x7E00 (quiet NaN)');
check(packHalf2x16(0.5, 2.0) === 0x40003800, 'packHalf2x16(0.5, 2.0) === 0x40003800');
check(packHalf2x16(-1.5, 0.25) === 0x3400be00, 'packHalf2x16(-1.5, 0.25) === 0x3400BE00');
check(packHalf2x16(1 / 1048576, 0) === 0x00000010, 'packHalf2x16(2^-20) low half === 0x0010 (denormal: 16 * 2^-24)');
check(packHalf2x16(1 / 16777216, 0) === 0x00000001, 'packHalf2x16(2^-24) low half === 0x0001 (min denormal)');
check(packHalf2x16(1 / 33554432, 0) === 0, 'packHalf2x16(2^-25) rounds to 0 (half-even)');

unpackSnorm2x16(0x80017fff, out, 0);
check(out[0] === 1.0 && out[1] === -1.0, `unpackSnorm2x16(0x80017FFF): got [${out[0]}, ${out[1]}]`);
unpackSnorm2x16(0, out, 0);
check(out[0] === 0 && out[1] === 0, 'unpackSnorm2x16(0) === [0, 0]');
unpackSnorm2x16(0x80000000, out, 0);
check(out[0] === 0 && out[1] === -1, `unpackSnorm2x16(0x80000000): got [${out[0]}, ${out[1]}] (y = -32768/32767 clamps to -1)`);

unpackUnorm2x16(0xffff8000, out, 0);
check(near(out[0], 0.5, 1e-4) && out[1] === 1.0, `unpackUnorm2x16(0xFFFF8000): got [${out[0]}, ${out[1]}]`);

unpackHalf2x16(0x3c00, out, 0);
check(out[0] === 1.0 && out[1] === 0, 'unpackHalf2x16(0x3C00) === [1, 0]');
unpackHalf2x16(0xc000, out, 0);
check(out[0] === -2.0, 'unpackHalf2x16(0xC000) === -2.0');
unpackHalf2x16(0x7bff, out, 0);
check(out[0] === 65504, 'unpackHalf2x16(0x7BFF) === 65504 (max finite half)');
unpackHalf2x16(0x0001, out, 0);
check(out[0] === 1 / 16777216, 'unpackHalf2x16(0x0001) === 2^-24 (min denormal)');
unpackHalf2x16(0x7c00, out, 0);
check(out[0] === Infinity, 'unpackHalf2x16(0x7C00) === +Inf');
unpackHalf2x16(0xfc00, out, 0);
check(out[0] === -Infinity, 'unpackHalf2x16(0xFC00) === -Inf');
unpackHalf2x16(0x7e00, out, 0);
check(Number.isNaN(out[0]), 'unpackHalf2x16(0x7E00) === NaN');

check(packUnorm4x8(0.5, 1.0, 0.0, 1.0) === 0xff00ff80, `packUnorm4x8: got 0x${packUnorm4x8(0.5, 1.0, 0.0, 1.0).toString(16)}`);
check(packSnorm4x8(1.0, -1.0, 0.5, -0.5) === 0xc040817f, `packSnorm4x8: got 0x${packSnorm4x8(1.0, -1.0, 0.5, -0.5).toString(16)}`);
unpackUnorm4x8(0xff00ff80, out, 0);
check(
  out[0] === 128 / 255 && out[1] === 1.0 && out[2] === 0 && out[3] === 1.0,
  `unpackUnorm4x8(0xFF00FF80): got [${out[0]}, ${out[1]}, ${out[2]}, ${out[3]}]`,
);
unpackSnorm4x8(0xc040817f, out, 0);
check(
  out[0] === 1.0 && out[1] === -1.0 && out[2] === 64 / 127 && out[3] === -64 / 127,
  `unpackSnorm4x8(0xC040817F): got [${out[0]}, ${out[1]}, ${out[2]}, ${out[3]}]`,
);

/* ------------------------------------------------------------------ */
/* pack / unpack — round trips over a deterministic LCG                */
/* ------------------------------------------------------------------ */

{
  let ok = true;
  for (let i = 0; i < 10; i++) {
    const x = lcg() * 2 - 1;
    const y = lcg() * 2 - 1;
    unpackSnorm2x16(packSnorm2x16(x, y), out, 0);
    if (!near(out[0], x, 1e-3) || !near(out[1], y, 1e-3)) ok = false;
  }
  check(ok, 'packSnorm2x16/unpackSnorm2x16 round trips (|err| < 1e-3)');
}
{
  let ok = true;
  for (let i = 0; i < 10; i++) {
    const x = lcg();
    const y = lcg();
    unpackUnorm2x16(packUnorm2x16(x, y), out, 0);
    if (!near(out[0], x, 1e-3) || !near(out[1], y, 1e-3)) ok = false;
  }
  check(ok, 'packUnorm2x16/unpackUnorm2x16 round trips (|err| < 1e-3)');
}
{
  let ok = true;
  for (let i = 0; i < 10; i++) {
    const x = lcg() * 2 - 1;
    const y = lcg() * 2 - 1;
    unpackHalf2x16(packHalf2x16(x, y), out, 0);
    if (!near(out[0], x, 1e-3) || !near(out[1], y, 1e-3)) ok = false;
  }
  check(ok, 'packHalf2x16/unpackHalf2x16 round trips (|err| < 1e-3)');
}
{
  let ok = true;
  for (let i = 0; i < 10; i++) {
    const a = lcg();
    const b = lcg();
    const c = lcg();
    const d = lcg();
    unpackUnorm4x8(packUnorm4x8(a, b, c, d), out, 0);
    // 8-bit quantization: max error = 0.5/255 ≈ 1.96e-3.
    if (!near(out[0], a, 2.5e-3) || !near(out[1], b, 2.5e-3) || !near(out[2], c, 2.5e-3) || !near(out[3], d, 2.5e-3)) ok = false;
  }
  check(ok, 'packUnorm4x8/unpackUnorm4x8 round trips (|err| < 2.5e-3)');
}

/* ------------------------------------------------------------------ */
/* float ↔ int bit conversions                                         */
/* ------------------------------------------------------------------ */

check(f2i(1.0) === 0x3f800000, `f2i(1.0): got 0x${(f2i(1.0) >>> 0).toString(16)}`);
check(f2u(-1.0) === 0xbf800000, `f2u(-1.0): got 0x${f2u(-1.0).toString(16)}`);
check(f2u(-0.0) === 0x80000000, 'f2u(-0.0) === 0x80000000');
check(f2u(Infinity) === 0x7f800000, 'f2u(Infinity) === 0x7F800000');
check(i2f(0x3f800000) === 1.0, 'i2f(0x3F800000) === 1.0');
check(u2f(0x3f800000) === 1.0, 'u2f(0x3F800000) === 1.0');
check(i2f(-1082130432) === -1.0, 'i2f(0xBF800000) === -1.0');
check(u2f(0xbf800000) === -1.0, 'u2f(0xBF800000) === -1.0');
check(i2f(f2i(3.5)) === 3.5, 'i2f(f2i(3.5)) round trip');
check(u2f(f2u(0.25)) === 0.25, 'u2f(f2u(0.25)) round trip');
check(f2i(i2f(0x7f800000)) === 0x7f800000, 'f2i(i2f(Inf bits)) round trip');

/* ------------------------------------------------------------------ */
/* bitfield ops                                                        */
/* ------------------------------------------------------------------ */

check(bitfieldExtractU(0xffffffff, 0, 4) === 15, 'bitfieldExtractU(0xFFFFFFFF, 0, 4) === 15');
check(bitfieldExtractU(0xffffffff, 28, 4) === 15, 'bitfieldExtractU(0xFFFFFFFF, 28, 4) === 15');
check(bitfieldExtractU(0xf0f0f0f0, 4, 8) === 0x0f, 'bitfieldExtractU(0xF0F0F0F0, 4, 8) === 0x0F');
check(bitfieldExtractU(0xffffffff, 0, 0) === 0, 'bitfieldExtractU(..., 0, 0) === 0 (bits == 0)');
check(bitfieldExtractU(0xffffffff, 8, 32) === 0xffffff, 'bitfieldExtractU(0xFFFFFFFF, 8, 32) permissive clamp');
check(bitfieldExtractU(0xffffffff, 40, 4) === 0, 'bitfieldExtractU(..., offset >= 32) === 0');
check(bitfieldExtractI(0xffffffff, 0, 4) === -1, 'bitfieldExtractI(0xFFFFFFFF, 0, 4) === -1 (sign-extended)');
check(bitfieldExtractI(0x80000000, 31, 1) === -1, 'bitfieldExtractI(0x80000000, 31, 1) === -1');
// Per GLSL ES 3.00 §8.10 the extracted bit IS the sign bit of the result:
// a 1-bit field of 1 sign-extends to -1; a 2-bit field of 0b10 → -2.
check(bitfieldExtractI(0x00000008, 3, 1) === -1, 'bitfieldExtractI(0x8, 3, 1) === -1 (1-bit field sign-extends)');
check(bitfieldExtractI(0x00000008, 2, 2) === -2, 'bitfieldExtractI(0x8, 2, 2) === -2 (0b10 sign-extends)');
check(bitfieldExtractI(0x00000008, 3, 0) === 0, 'bitfieldExtractI(..., 3, 0) === 0 (bits == 0)');
check(bitfieldExtractI(0xffffff00, 0, 8) === 0, 'bitfieldExtractI(0xFFFFFF00, 0, 8) === 0');
check(bitfieldInsert(0, 0xf, 4, 4) === 0xf0, 'bitfieldInsert(0, 0xF, 4, 4) === 0xF0');
check(bitfieldInsert(0xffff, 0, 0, 16) === 0, 'bitfieldInsert(0xFFFF, 0, 0, 16) === 0');
check(bitfieldInsert(0xff00ff00, 0x0f, 8, 8) === 0xff000f00, `bitfieldInsert(0xFF00FF00, 0x0F, 8, 8): got 0x${bitfieldInsert(0xff00ff00, 0x0f, 8, 8).toString(16)}`);
check(bitfieldReverse(0x80000000) === 1, 'bitfieldReverse(0x80000000) === 1');
check(bitfieldReverse(1) === 0x80000000, 'bitfieldReverse(1) === 0x80000000');
check(bitfieldReverse(0) === 0, 'bitfieldReverse(0) === 0');
check(bitfieldReverse(0xffff0000) === 0x0000ffff, 'bitfieldReverse(0xFFFF0000) === 0x0000FFFF');
check(bitCount(0) === 0, 'bitCount(0) === 0');
check(bitCount(0xf0f0f0f0) === 16, 'bitCount(0xF0F0F0F0) === 16');
check(bitCount(0xffffffff) === 32, 'bitCount(0xFFFFFFFF) === 32');
check(findLSB(8) === 3, 'findLSB(8) === 3');
check(findLSB(0) === -1, 'findLSB(0) === -1');
check(findLSB(1) === 0, 'findLSB(1) === 0');
check(findLSB(0x80000000) === 31, 'findLSB(0x80000000) === 31');
check(findMSB(0) === -1, 'findMSB(0) === -1');
check(findMSB(0x80000000) === 31, 'findMSB(0x80000000) === 31');
check(findMSB(1) === 0, 'findMSB(1) === 0');
check(findMSB(0x7fffffff) === 30, 'findMSB(0x7FFFFFFF) === 30');
check(findMSB(0x0000ffff) === 15, 'findMSB(0x0000FFFF) === 15');
check(findMSBI(-1) === -1, 'findMSBI(-1) === -1 (no 0 bit exists)');
check(findMSBI(-2) === 0, 'findMSBI(-2) === 0 (…11110 → highest 0 at bit 0)');
check(findMSBI(-3) === 1, 'findMSBI(-3) === 1 (…11101 → highest 0 at bit 1)');
check(findMSBI(0) === -1, 'findMSBI(0) === -1');
check(findMSBI(1) === 0, 'findMSBI(1) === 0');
check(findMSBI(0x7fffffff) === 30, 'findMSBI(0x7FFFFFFF) === 30');
check(findMSBI(-2147483648) === 30, 'findMSBI(-2147483648) === 30 (0x80000000 → ~ = 0x7FFFFFFF)');

/* ------------------------------------------------------------------ */
/* uaddCarry / usubBorrow / umulExtended / imulExtended                */
/* ------------------------------------------------------------------ */

uaddCarry(0xffffffff, 1, out, 0);
check(out[0] === 0 && out[1] === 1, `uaddCarry(0xFFFFFFFF, 1): got [${out[0]}, ${out[1]}]`);
uaddCarry(1, 2, out, 0);
check(out[0] === 3 && out[1] === 0, 'uaddCarry(1, 2) === [3, 0]');
usubBorrow(0, 1, out, 0);
check(out[0] === 0xffffffff && out[1] === 1, `usubBorrow(0, 1): got [${out[0]}, ${out[1]}]`);
usubBorrow(5, 3, out, 0);
check(out[0] === 2 && out[1] === 0, 'usubBorrow(5, 3) === [2, 0]');

umulExtended(0xffffffff, 0xffffffff, out, 0);
check(out[0] === 0xfffffffe && out[1] === 1, `umulExtended(0xFFFFFFFF, 0xFFFFFFFF): got [0x${(out[0] >>> 0).toString(16)}, 0x${(out[1] >>> 0).toString(16)}]`);
umulExtended(0x10000, 0x10000, out, 0);
check(out[0] === 1 && out[1] === 0, 'umulExtended(0x10000, 0x10000) === [1, 0]');
umulExtended(0, 0xffffffff, out, 0);
check(out[0] === 0 && out[1] === 0, 'umulExtended(0, x) === [0, 0]');
{
  let ok = true;
  for (let i = 0; i < 5; i++) {
    const a = lcgInt();
    const b = lcgInt();
    umulExtended(a, b, out, 0);
    const p = BigInt(a) * BigInt(b);
    if (out[0] !== Number((p >> 32n) & 0xffffffffn) || out[1] !== Number(p & 0xffffffffn)) ok = false;
  }
  check(ok, 'umulExtended matches BigInt 64-bit product on random inputs');
}

imulExtended(-1, -1, out, 0);
check(out[0] === 0 && out[1] === 1, `imulExtended(-1, -1): got [${out[0]}, ${out[1]}]`);
imulExtended(-1, 1, out, 0);
check(out[0] === -1 && out[1] === 0xffffffff, `imulExtended(-1, 1): got [${out[0]}, 0x${(out[1] >>> 0).toString(16)}]`);
imulExtended(1, 1, out, 0);
check(out[0] === 0 && out[1] === 1, 'imulExtended(1, 1) === [0, 1]');
imulExtended(-2147483648, -1, out, 0);
check(out[0] === 0 && out[1] === 0x80000000, 'imulExtended(-2^31, -1) === [0, 0x80000000]');
{
  let ok = true;
  for (let i = 0; i < 5; i++) {
    const a = lcgInt() | 0;
    const b = lcgInt() | 0;
    imulExtended(a, b, out, 0);
    const p = BigInt(a) * BigInt(b);
    const hi = p >> 32n;
    const lo = p & 0xffffffffn;
    const refHi = hi >= 0x80000000n ? Number(hi - 0x100000000n) : Number(hi);
    if (out[0] !== refHi || out[1] !== Number(lo)) ok = false;
  }
  check(ok, 'imulExtended matches BigInt signed 64-bit product on random inputs');
}

/* ------------------------------------------------------------------ */
/* Matrix inverse vs an independent generic adjugate reference         */
/* ------------------------------------------------------------------ */

/** Generic determinant (row-major matrix, cofactor expansion). */
function refDet(m: number[][]): number {
  const n = m.length;
  if (n === 1) return m[0][0];
  let d = 0;
  for (let c = 0; c < n; c++) {
    const sub = m.slice(1).map((row) => row.filter((_, j) => j !== c));
    d += (c % 2 === 0 ? 1 : -1) * m[0][c] * refDet(sub);
  }
  return d;
}

/** Generic adjugate inverse; returns inv[col][row] (GLSL column-major view). */
function refInv(m: number[][]): number[][] {
  const n = m.length;
  const inv: number[][] = [];
  for (let r = 0; r < n; r++) inv.push(new Array(n).fill(0));
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const sub = m.filter((_, i) => i !== r).map((row) => row.filter((_, j) => j !== c));
      const cof = ((r + c) % 2 === 0 ? 1 : -1) * refDet(sub);
      inv[c][r] = cof / refDet(m);
    }
  }
  return inv;
}

/** Call R.invN with the column-major flattening of row-major matrix m. */
function callInv(n: number, m: number[][]): number[] {
  const flat: number[] = [];
  for (let c = 0; c < n; c++) for (let r = 0; r < n; r++) flat.push(m[r][c]);
  const res: number[] = [];
  if (n === 2) inv2(res, 0, flat[0], flat[1], flat[2], flat[3]);
  else if (n === 3) inv3(res, 0, flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6], flat[7], flat[8]);
  else inv4(res, 0, flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], flat[6], flat[7], flat[8], flat[9], flat[10], flat[11], flat[12], flat[13], flat[14], flat[15]);
  return res;
}

function checkInv(n: number, m: number[][], msg: string): void {
  const got = callInv(n, m);
  const ref = refInv(m);
  // got[c*n+r] = GLSL inverse m⁻¹[c][r] = (Mᵀ)⁻¹[c][r] = M⁻¹[r][c] = ref[r][c].
  let ok = true;
  for (let c = 0; c < n; c++) {
    for (let r = 0; r < n; r++) {
      if (!near(got[c * n + r], ref[r][c], 1e-6)) ok = false;
    }
  }
  check(ok, `${msg}: R.inv${n} matches the adjugate reference`);
  // Sanity: M * inv(M) ≈ I.
  let prodOk = true;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += m[i][k] * got[j * n + k];
      if (!near(s, i === j ? 1 : 0, 1e-6)) prodOk = false;
    }
  }
  check(prodOk, `${msg}: M·inv ≈ I`);
}

checkInv(2, [[4, 7], [2, 6]], '2x2 well-conditioned');
checkInv(2, [[1, 0], [0, 1]], '2x2 identity');
checkInv(3, [[1, 2, 3], [0, 1, 4], [5, 6, 0]], '3x3 det=1');
checkInv(3, [[3, 0, 2], [2, 0, -2], [0, 1, 1]], '3x3 full');
checkInv(4, [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], '4x4 diagonal');
checkInv(4, [[1, 0, 2, 3], [0, 1, 4, 5], [0, 0, 1, 6], [0, 0, 0, 1]], '4x4 unit upper triangular');
{
  // Random entries in [-1, 1] plus 3 on the diagonal → diagonally dominant
  // (well-conditioned).
  const m: number[][] = [];
  for (let r = 0; r < 4; r++) {
    m.push([]);
    for (let c = 0; c < 4; c++) m[r].push(lcg() * 2 - 1 + (r === c ? 3 : 0));
  }
  checkInv(4, m, '4x4 random');
}
{
  const got2 = callInv(2, [[1, 2], [2, 4]]);
  check(got2.every((x) => x === 0), 'inv2 singular → zeros, no crash');
  const got3 = callInv(3, [[1, 2, 3], [2, 4, 6], [0, 0, 0]]);
  check(got3.every((x) => x === 0), 'inv3 singular → zeros, no crash');
  const got4 = callInv(4, [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]);
  check(got4.every((x) => x === 0), 'inv4 singular → zeros, no crash');
}

/* ------------------------------------------------------------------ */
/* resolveImage                                                        */
/* ------------------------------------------------------------------ */

{
  const img1 = { width: 4, height: 4 } as unknown as TextureImage;
  const img2 = { width: 8, height: 8 } as unknown as TextureImage;
  check(resolveImage({ textures: [null, img1] }, 1) === img1, 'resolveImage: ctx.textures path');
  check(resolveImage({ textures: [null, img1] }, 0) === null, 'resolveImage: empty unit → null');
  check(resolveImage({ tex: { units: [{ img: img2 }, null] } }, 0) === img2, 'resolveImage: ctx.tex.units path');
  check(resolveImage({ tex: { units: [{ img: img2 }, null] } }, 1) === null, 'resolveImage: null unit → null');
  check(resolveImage({}, 0) === null, 'resolveImage: empty ctx → null');
  check(resolveImage({ textures: [] }, 3) === null, 'resolveImage: out-of-range → null');
  check(resolveImage({ tex: {} }, 0) === null, 'resolveImage: tex without units → null');
}

/* ------------------------------------------------------------------ */
/* R object + codegen/index.ts seam                                    */
/* ------------------------------------------------------------------ */

check(R === RFromIndex, 'codegen/index.ts re-exports the SAME R object');
check(typeof R.packHalf2x16 === 'function' && typeof R.inv4 === 'function', 'R exposes packHalf2x16/inv4');
check(typeof R.bitfieldInsert === 'function' && typeof R.resolveImage === 'function', 'R exposes bitfieldInsert/resolveImage');
check(typeof R.findMSBI === 'function' && typeof R.imulExtended === 'function', 'R exposes findMSBI/imulExtended');
check(Object.keys(R).length >= 30, `R has >= 30 helpers (got ${Object.keys(R).length})`);

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`runtime selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
