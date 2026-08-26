/**
 * math.ts — GLSL-style vector & matrix math for the software renderer.
 *
 * Shared runtime consumed by:
 *  - `glsl/` codegen: GLSL built-in functions that operate on vectors/matrices
 *    (mat4 * vec4 vertex transforms, normalize, cross, inverse(), transpose(),
 *    outerProduct(), matrixCompMult(), pack*2x16, ...) are emitted as calls
 *    into this module.
 *  - `raster/`: homogeneous clipping (clip-space interpolation), varying
 *    interpolation (mix), point-sphere math.
 *
 * Conventions (MANDATORY — glsl codegen relies on these):
 *  - Vectors are plain `Float32Array`s of length 2/3/4 (vec2/vec3/vec4).
 *  - Matrices are column-major `Float32Array`s, exactly like GLSL uniform
 *    layout: mat2 = 4 elements, mat3 = 9, mat4 = 16. Column c occupies
 *    elements [c*size, (c+1)*size); element (col, row) = m[col*size + row].
 *  - Every operation that produces a vector/matrix takes an optional trailing
 *    `out` buffer (caller-preallocated, for zero-allocation hot paths such as
 *    per-fragment shader execution). When `out` is omitted a new Float32Array
 *    is allocated. `out` must have the exact expected length; no bounds
 *    checks are performed.
 *  - Functions are pure: inputs are only read, never mutated; results are
 *    written to `out` (or a fresh array).
 *  - GLSL ES semantics are authoritative for GLSL-named functions — this
 *    module is the single source of truth; do NOT reimplement these in
 *    glsl/ codegen or raster/.
 */

// ---------------------------------------------------------------------------
// Types & constructors
// ---------------------------------------------------------------------------

/** A 2-component float vector (Float32Array of length 2). */
export type Vec2 = Float32Array;
/** A 3-component float vector (Float32Array of length 3). */
export type Vec3 = Float32Array;
/** A 4-component float vector (Float32Array of length 4). */
export type Vec4 = Float32Array;
/** A 2x2 column-major matrix (Float32Array of length 4). */
export type Mat2 = Float32Array;
/** A 3x3 column-major matrix (Float32Array of length 9). */
export type Mat3 = Float32Array;
/** A 4x4 column-major matrix (Float32Array of length 16). */
export type Mat4 = Float32Array;

/**
 * Allocates a new vec2. GLSL `vec2(x, y)` constructor semantics.
 */
export function vec2(x = 0, y = 0): Vec2 {
  return new Float32Array([x, y]);
}

/**
 * Allocates a new vec3. GLSL `vec3(x, y, z)` constructor semantics.
 */
export function vec3(x = 0, y = 0, z = 0): Vec3 {
  return new Float32Array([x, y, z]);
}

/**
 * Allocates a new vec4. GLSL `vec4(x, y, z, w)` constructor semantics.
 */
export function vec4(x = 0, y = 0, z = 0, w = 0): Vec4 {
  return new Float32Array([x, y, z, w]);
}

/**
 * Allocates a new mat2 with diagonal `s` (GLSL `mat2()` = identity, `mat2(s)`
 * = diagonal s; off-diagonal zeros). Column-major.
 */
export function mat2(s = 1): Mat2 {
  return new Float32Array([s, 0, 0, s]);
}

/**
 * Allocates a new mat3 with diagonal `s` (GLSL `mat3()` = identity, `mat3(s)`
 * = diagonal s; off-diagonal zeros). Column-major.
 */
export function mat3(s = 1): Mat3 {
  return new Float32Array([s, 0, 0, 0, s, 0, 0, 0, s]);
}

/**
 * Allocates a new mat4 with diagonal `s` (GLSL `mat4()` = identity, `mat4(s)`
 * = diagonal s; off-diagonal zeros). Column-major.
 */
export function mat4(s = 1): Mat4 {
  return new Float32Array([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, s]);
}

/**
 * Fills `out` (length 4) with the identity matrix and returns it.
 */
export function mat2Identity(out: Mat2): Mat2 {
  out.fill(0);
  out[0] = 1;
  out[3] = 1;
  return out;
}

/**
 * Fills `out` (length 9) with the identity matrix and returns it.
 */
export function mat3Identity(out: Mat3): Mat3 {
  out.fill(0);
  out[0] = 1;
  out[4] = 1;
  out[8] = 1;
  return out;
}

/**
 * Fills `out` (length 16) with the identity matrix and returns it.
 */
export function mat4Identity(out: Mat4): Mat4 {
  out.fill(0);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
  return out;
}

// ---------------------------------------------------------------------------
// Componentwise arithmetic
// ---------------------------------------------------------------------------

/**
 * out[i] = a[i] + b[i] (componentwise add; `a` and `b` same length 2/3/4).
 */
export function vecAdd(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  return binary(a, b, addNum, out);
}

/**
 * out[i] = a[i] - b[i] (componentwise subtract; `a` and `b` same length 2/3/4).
 */
export function vecSub(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  return binary(a, b, subNum, out);
}

/**
 * out[i] = a[i] * b[i] (componentwise multiply, NOT dot/cross; `a` and `b`
 * same length 2/3/4).
 */
export function vecMul(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  return binary(a, b, mulNum, out);
}

/**
 * out[i] = a[i] / b[i] (componentwise divide; `a` and `b` same length 2/3/4).
 * Division by zero yields Infinity/NaN per IEEE 754, like GPUs.
 */
export function vecDiv(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  return binary(a, b, divNum, out);
}

/**
 * out[i] = a[i] * s (scalar multiply of a vector).
 */
export function vecScale(a: Float32Array, s: number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] * s;
  return o;
}

/**
 * out[i] = -a[i] (componentwise negation).
 */
export function vecNeg(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, negNum, out);
}

/**
 * out[i] = fn(a[i]) — generic componentwise unary map. Used by glsl codegen
 * for trig/exp/log family built-ins (sin, cos, tan, asin, acos, tanh, exp,
 * log2, ...). `fn` is called with one number and must return a number.
 */
export function vecMap(a: Float32Array, fn: (x: number) => number, out?: Float32Array): Float32Array {
  return unary(a, fn, out);
}

/**
 * out[i] = fn(a[i], b[i]) — generic componentwise binary map. `a` and `b`
 * must have the same length.
 */
export function vecMap2(
  a: Float32Array,
  b: Float32Array,
  fn: (x: number, y: number) => number,
  out?: Float32Array
): Float32Array {
  return binary(a, b, fn, out);
}

// ---------------------------------------------------------------------------
// Geometric functions
// ---------------------------------------------------------------------------

/**
 * GLSL `dot(a, b)` — sum of componentwise products (any length).
 */
export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * GLSL `cross(a, b)` — 3D cross product. Both inputs must be length 3.
 */
export function cross(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  const a0 = a[0], a1 = a[1], a2 = a[2];
  const b0 = b[0], b1 = b[1], b2 = b[2];
  const o = out ?? new Float32Array(3);
  o[0] = a1 * b2 - a2 * b1;
  o[1] = a2 * b0 - a0 * b2;
  o[2] = a0 * b1 - a1 * b0;
  return o;
}

/**
 * GLSL `length(a)` = sqrt(dot(a, a)).
 */
export function length(a: Float32Array): number {
  return Math.sqrt(dot(a, a));
}

/**
 * GLSL `length(a)` squared = dot(a, a) (avoids sqrt; for comparisons).
 */
export function lengthSq(a: Float32Array): number {
  return dot(a, a);
}

/**
 * GLSL `distance(a, b)` = length(a - b).
 */
export function distance(a: Float32Array, b: Float32Array): number {
  return Math.sqrt(distanceSq(a, b));
}

/**
 * GLSL `distance(a, b)` squared (avoids sqrt; for comparisons).
 */
export function distanceSq(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

/**
 * GLSL `normalize(a)` = a / length(a). Zero vector is undefined in GLSL;
 * this implementation returns the zero vector unchanged (no NaN).
 */
export function normalize(a: Float32Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(a.length);
  const len = Math.sqrt(dot(a, a));
  if (len === 0) {
    o.fill(0);
    return o;
  }
  for (let i = 0; i < a.length; i++) o[i] = a[i] / len;
  return o;
}

/**
 * GLSL `reflect(I, N)` = I - 2 * dot(N, I) * N. All inputs length 3.
 */
export function reflect(I: Float32Array, N: Float32Array, out?: Float32Array): Float32Array {
  const i0 = I[0], i1 = I[1], i2 = I[2];
  const n0 = N[0], n1 = N[1], n2 = N[2];
  const d = 2 * (n0 * i0 + n1 * i1 + n2 * i2);
  const o = out ?? new Float32Array(3);
  o[0] = i0 - d * n0;
  o[1] = i1 - d * n1;
  o[2] = i2 - d * n2;
  return o;
}

/**
 * GLSL `refract(I, N, eta)`:
 *   k = 1 - eta^2 * (1 - dot(N, I)^2)
 *   if k < 0: return zero vector, else eta*I - (eta*dot(N,I) + sqrt(k))*N.
 * All inputs length 3.
 */
export function refract(I: Float32Array, N: Float32Array, eta: number, out?: Float32Array): Float32Array {
  const i0 = I[0], i1 = I[1], i2 = I[2];
  const n0 = N[0], n1 = N[1], n2 = N[2];
  const d = n0 * i0 + n1 * i1 + n2 * i2;
  const k = 1 - eta * eta * (1 - d * d);
  const o = out ?? new Float32Array(3);
  if (k < 0) {
    o[0] = 0;
    o[1] = 0;
    o[2] = 0;
    return o;
  }
  const c = eta * d + Math.sqrt(k);
  o[0] = eta * i0 - c * n0;
  o[1] = eta * i1 - c * n1;
  o[2] = eta * i2 - c * n2;
  return o;
}

/**
 * GLSL `faceforward(N, I, Nref)` = dot(Nref, I) < 0 ? N : -N. All inputs length 3.
 */
export function faceforward(N: Float32Array, I: Float32Array, Nref: Float32Array, out?: Float32Array): Float32Array {
  const n0 = N[0], n1 = N[1], n2 = N[2];
  const d = Nref[0] * I[0] + Nref[1] * I[1] + Nref[2] * I[2];
  const o = out ?? new Float32Array(3);
  if (d < 0) {
    o[0] = n0;
    o[1] = n1;
    o[2] = n2;
  } else {
    o[0] = -n0;
    o[1] = -n1;
    o[2] = -n2;
  }
  return o;
}

// ---------------------------------------------------------------------------
// Componentwise GLSL functions (exact GLSL ES semantics — single source of truth)
// ---------------------------------------------------------------------------

/**
 * out[i] = abs(a[i]) (GLSL `abs`).
 */
export function vecAbs(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, absNum, out);
}

/**
 * out[i] = sign(a[i]) — 1.0 if > 0, -1.0 if < 0, 0.0 if == 0 (GLSL `sign`).
 */
export function vecSign(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, signNum, out);
}

/**
 * out[i] = floor(a[i]) (GLSL `floor`).
 */
export function vecFloor(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, floorNum, out);
}

/**
 * out[i] = ceil(a[i]) (GLSL `ceil`).
 */
export function vecCeil(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, ceilNum, out);
}

/**
 * out[i] = trunc(a[i]) (GLSL `trunc` — toward zero).
 */
export function vecTrunc(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, truncNum, out);
}

/**
 * out[i] = round(a[i]) (GLSL `round` — round half away from zero).
 */
export function vecRound(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, roundNum, out);
}

/**
 * out[i] = roundEven(a[i]) (GLSL `roundEven` — round half to even).
 */
export function vecRoundEven(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, roundEvenNum, out);
}

/**
 * out[i] = fract(a[i]) = a[i] - floor(a[i]) (GLSL `fract`).
 */
export function vecFract(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, fractNum, out);
}

/**
 * out[i] = sqrt(a[i]) (GLSL `sqrt`).
 */
export function vecSqrt(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, sqrtNum, out);
}

/**
 * out[i] = inversesqrt(a[i]) = 1/sqrt(a[i]) (GLSL `inversesqrt`).
 */
export function vecInversesqrt(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, inversesqrtNum, out);
}

/**
 * out[i] = exp(a[i]) (GLSL `exp`).
 */
export function vecExp(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, expNum, out);
}

/**
 * out[i] = log(a[i]) (GLSL `log`, natural log).
 */
export function vecLog(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, logNum, out);
}

/**
 * out[i] = exp2(a[i]) (GLSL `exp2`).
 */
export function vecExp2(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, exp2Num, out);
}

/**
 * out[i] = log2(a[i]) (GLSL `log2`).
 */
export function vecLog2(a: Float32Array, out?: Float32Array): Float32Array {
  return unary(a, log2Num, out);
}

/**
 * out[i] = pow(a[i], b[i]) (GLSL `pow`; undefined for a[i] < 0, like GLSL).
 */
export function vecPow(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  return binary(a, b, powNum, out);
}

/**
 * out[i] = atan(y[i], x[i]) = atan2(y[i], x[i]) (GLSL two-arg `atan`).
 */
export function vecAtan2(y: Float32Array, x: Float32Array, out?: Float32Array): Float32Array {
  return binary(y, x, atan2Num, out);
}

/**
 * out[i] = min(a[i], b[i]) (GLSL `min`).
 */
export function vecMin(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  return binary(a, b, minNum, out);
}

/**
 * out[i] = max(a[i], b[i]) (GLSL `max`).
 */
export function vecMax(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  return binary(a, b, maxNum, out);
}

/**
 * out[i] = clamp(a[i], lo, hi) = min(max(a[i], lo), hi). `lo`/`hi` may each be
 * a scalar (broadcast) or a vector of the same length (GLSL `clamp`).
 * Undefined when lo > hi (GLSL); result follows min(max) evaluation.
 */
export function vecClamp(a: Float32Array, lo: number | Float32Array, hi: number | Float32Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = clampNum(a[i], resolveSV(lo, i), resolveSV(hi, i));
  return o;
}

/**
 * out[i] = mix(x[i], y[i], t) = x[i]*(1-t) + y[i]*t. `t` may be a scalar
 * (broadcast) or a vector of the same length (GLSL `mix`).
 */
export function vecMix(x: Float32Array, y: Float32Array, t: number | Float32Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = mixNum(x[i], y[i], resolveSV(t, i));
  return o;
}

/**
 * out[i] = mod(x[i], y) = x[i] - y*floor(x[i]/y). `y` may be a scalar
 * (broadcast) or a vector of the same length (GLSL `mod`; NOT the JS % op).
 */
export function vecMod(x: Float32Array, y: number | Float32Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = modNum(x[i], resolveSV(y, i));
  return o;
}

/**
 * out[i] = step(edge, x[i]) = x[i] < edge ? 0 : 1. `edge` may be a scalar
 * (broadcast) or a vector of the same length (GLSL `step`).
 */
export function vecStep(edge: number | Float32Array, x: Float32Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = stepNum(resolveSV(edge, i), x[i]);
  return o;
}

/**
 * out[i] = smoothstep(e0, e1, x[i]):
 *   t = clamp((x - e0)/(e1 - e0), 0, 1); result = t*t*(3 - 2*t).
 * `e0`/`e1` may be scalars (broadcast) or vectors of the same length.
 * e0 == e1 is undefined in GLSL; this implementation returns 0 (no NaN).
 */
export function vecSmoothstep(e0: number | Float32Array, e1: number | Float32Array, x: Float32Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = smoothstepNum(resolveSV(e0, i), resolveSV(e1, i), x[i]);
  return o;
}

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

/**
 * out = a * b (GLSL matrix multiply). Column-major storage: element
 * (col c, row r) of the result = sum over k of a[k*4 + r] * b[c*4 + k].
 * All matrices length 16.
 */
export function mat4Mul(a: Mat4, b: Mat4, out?: Mat4): Mat4 {
  const o = out ?? new Float32Array(16);
  // All inputs are read into locals first: `out` may alias `a` or `b`.
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11], a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
  const b8 = b[8], b9 = b[9], b10 = b[10], b11 = b[11], b12 = b[12], b13 = b[13], b14 = b[14], b15 = b[15];
  o[0] = a0 * b0 + a4 * b1 + a8 * b2 + a12 * b3; o[1] = a1 * b0 + a5 * b1 + a9 * b2 + a13 * b3;
  o[2] = a2 * b0 + a6 * b1 + a10 * b2 + a14 * b3; o[3] = a3 * b0 + a7 * b1 + a11 * b2 + a15 * b3;
  o[4] = a0 * b4 + a4 * b5 + a8 * b6 + a12 * b7; o[5] = a1 * b4 + a5 * b5 + a9 * b6 + a13 * b7;
  o[6] = a2 * b4 + a6 * b5 + a10 * b6 + a14 * b7; o[7] = a3 * b4 + a7 * b5 + a11 * b6 + a15 * b7;
  o[8] = a0 * b8 + a4 * b9 + a8 * b10 + a12 * b11; o[9] = a1 * b8 + a5 * b9 + a9 * b10 + a13 * b11;
  o[10] = a2 * b8 + a6 * b9 + a10 * b10 + a14 * b11; o[11] = a3 * b8 + a7 * b9 + a11 * b10 + a15 * b11;
  o[12] = a0 * b12 + a4 * b13 + a8 * b14 + a12 * b15; o[13] = a1 * b12 + a5 * b13 + a9 * b14 + a13 * b15;
  o[14] = a2 * b12 + a6 * b13 + a10 * b14 + a14 * b15; o[15] = a3 * b12 + a7 * b13 + a11 * b14 + a15 * b15;
  return o;
}

/**
 * out = a * b (GLSL matrix multiply, 3x3). All matrices length 9.
 */
export function mat3Mul(a: Mat3, b: Mat3, out?: Mat3): Mat3 {
  const o = out ?? new Float32Array(9);
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7], b8 = b[8];
  o[0] = a0 * b0 + a3 * b1 + a6 * b2; o[1] = a1 * b0 + a4 * b1 + a7 * b2;
  o[2] = a2 * b0 + a5 * b1 + a8 * b2; o[3] = a0 * b3 + a3 * b4 + a6 * b5;
  o[4] = a1 * b3 + a4 * b4 + a7 * b5; o[5] = a2 * b3 + a5 * b4 + a8 * b5;
  o[6] = a0 * b6 + a3 * b7 + a6 * b8; o[7] = a1 * b6 + a4 * b7 + a7 * b8;
  o[8] = a2 * b6 + a5 * b7 + a8 * b8;
  return o;
}

/**
 * out = a * b (GLSL matrix multiply, 2x2). All matrices length 4.
 */
export function mat2Mul(a: Mat2, b: Mat2, out?: Mat2): Mat2 {
  const o = out ?? new Float32Array(4);
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  const b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
  o[0] = a0 * b0 + a2 * b1; o[1] = a1 * b0 + a3 * b1;
  o[2] = a0 * b2 + a2 * b3; o[3] = a1 * b2 + a3 * b3;
  return o;
}

/**
 * out = m * v (GLSL mat4 * vec4). `m` length 16, `v`/`out` length 4.
 * out[r] = sum over k of m[k*4 + r] * v[k].
 */
export function mat4MulVec4(m: Mat4, v: Vec4, out?: Vec4): Vec4 {
  const v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
  const o = out ?? new Float32Array(4);
  o[0] = m[0] * v0 + m[4] * v1 + m[8] * v2 + m[12] * v3;
  o[1] = m[1] * v0 + m[5] * v1 + m[9] * v2 + m[13] * v3;
  o[2] = m[2] * v0 + m[6] * v1 + m[10] * v2 + m[14] * v3;
  o[3] = m[3] * v0 + m[7] * v1 + m[11] * v2 + m[15] * v3;
  return o;
}

/**
 * out = m * v (GLSL mat3 * vec3). `m` length 9, `v`/`out` length 3.
 */
export function mat3MulVec3(m: Mat3, v: Vec3, out?: Vec3): Vec3 {
  const v0 = v[0], v1 = v[1], v2 = v[2];
  const o = out ?? new Float32Array(3);
  o[0] = m[0] * v0 + m[3] * v1 + m[6] * v2;
  o[1] = m[1] * v0 + m[4] * v1 + m[7] * v2;
  o[2] = m[2] * v0 + m[5] * v1 + m[8] * v2;
  return o;
}

/**
 * out = m * v (GLSL mat2 * vec2). `m` length 4, `v`/`out` length 2.
 */
export function mat2MulVec2(m: Mat2, v: Vec2, out?: Vec2): Vec2 {
  const v0 = v[0], v1 = v[1];
  const o = out ?? new Float32Array(2);
  o[0] = m[0] * v0 + m[2] * v1;
  o[1] = m[1] * v0 + m[3] * v1;
  return o;
}

/**
 * out = v * m (GLSL vec4 * mat4, row-vector multiply). `m` length 16,
 * `v`/`out` length 4. out[c] = sum over k of v[k] * m[c*4 + k].
 */
export function vec4MulMat4(v: Vec4, m: Mat4, out?: Vec4): Vec4 {
  const v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3];
  const o = out ?? new Float32Array(4);
  o[0] = v0 * m[0] + v1 * m[1] + v2 * m[2] + v3 * m[3];
  o[1] = v0 * m[4] + v1 * m[5] + v2 * m[6] + v3 * m[7];
  o[2] = v0 * m[8] + v1 * m[9] + v2 * m[10] + v3 * m[11];
  o[3] = v0 * m[12] + v1 * m[13] + v2 * m[14] + v3 * m[15];
  return o;
}

/**
 * out = v * m (GLSL vec3 * mat3, row-vector multiply). `m` length 9.
 */
export function vec3MulMat3(v: Vec3, m: Mat3, out?: Vec3): Vec3 {
  const v0 = v[0], v1 = v[1], v2 = v[2];
  const o = out ?? new Float32Array(3);
  o[0] = v0 * m[0] + v1 * m[1] + v2 * m[2];
  o[1] = v0 * m[3] + v1 * m[4] + v2 * m[5];
  o[2] = v0 * m[6] + v1 * m[7] + v2 * m[8];
  return o;
}

/**
 * out = v * m (GLSL vec2 * mat2, row-vector multiply). `m` length 4.
 */
export function vec2MulMat2(v: Vec2, m: Mat2, out?: Vec2): Vec2 {
  const v0 = v[0], v1 = v[1];
  const o = out ?? new Float32Array(2);
  o[0] = v0 * m[0] + v1 * m[1];
  o[1] = v0 * m[2] + v1 * m[3];
  return o;
}

/**
 * out = transpose(a) (GLSL `transpose`). Column-major storage: out[c*4 + r] =
 * a[r*4 + c].
 */
export function mat4Transpose(a: Mat4, out?: Mat4): Mat4 {
  const o = out ?? new Float32Array(16);
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
  const a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11], a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
  o[0] = a0; o[1] = a4; o[2] = a8; o[3] = a12;
  o[4] = a1; o[5] = a5; o[6] = a9; o[7] = a13;
  o[8] = a2; o[9] = a6; o[10] = a10; o[11] = a14;
  o[12] = a3; o[13] = a7; o[14] = a11; o[15] = a15;
  return o;
}

/**
 * out = transpose(a) (GLSL `transpose`, 3x3): out[c*3 + r] = a[r*3 + c].
 */
export function mat3Transpose(a: Mat3, out?: Mat3): Mat3 {
  const o = out ?? new Float32Array(9);
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8];
  o[0] = a0; o[1] = a3; o[2] = a6;
  o[3] = a1; o[4] = a4; o[5] = a7;
  o[6] = a2; o[7] = a5; o[8] = a8;
  return o;
}

/**
 * out = transpose(a) (GLSL `transpose`, 2x2): out[c*2 + r] = a[r*2 + c].
 */
export function mat2Transpose(a: Mat2, out?: Mat2): Mat2 {
  const o = out ?? new Float32Array(4);
  const a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  o[0] = a0; o[1] = a2;
  o[2] = a1; o[3] = a3;
  return o;
}

/**
 * GLSL `determinant(a)` for 4x4 column-major `a`.
 */
export function mat4Determinant(a: Mat4): number {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
}

/**
 * GLSL `determinant(a)` for 3x3 column-major `a`.
 */
export function mat3Determinant(a: Mat3): number {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[3] * (a[1] * a[8] - a[2] * a[7]) + a[6] * (a[1] * a[5] - a[2] * a[4]);
}

/**
 * GLSL `determinant(a)` for 2x2 column-major `a`.
 */
export function mat2Determinant(a: Mat2): number {
  return a[0] * a[3] - a[1] * a[2];
}

/**
 * out = inverse(a) (GLSL `inverse`, 4x4). Singular matrices are undefined in
 * GLSL; this implementation does not check and may produce NaN/Inf entries.
 */
export function mat4Inverse(a: Mat4, out?: Mat4): Mat4 {
  const o = out ?? new Float32Array(16);
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const invDet = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * invDet; o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * invDet;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * invDet; o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * invDet;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * invDet; o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * invDet;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * invDet; o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * invDet;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * invDet; o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * invDet;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * invDet; o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * invDet;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * invDet; o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * invDet;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * invDet; o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * invDet;
  return o;
}

/**
 * out = inverse(a) (GLSL `inverse`, 3x3). Singular matrices are undefined in
 * GLSL; this implementation does not check and may produce NaN/Inf entries.
 */
export function mat3Inverse(a: Mat3, out?: Mat3): Mat3 {
  const o = out ?? new Float32Array(9);
  const a00 = a[0], a01 = a[1], a02 = a[2];
  const a10 = a[3], a11 = a[4], a12 = a[5];
  const a20 = a[6], a21 = a[7], a22 = a[8];
  const b01 = a22 * a11 - a12 * a21;
  const b11 = -a22 * a10 + a12 * a20;
  const b21 = a21 * a10 - a11 * a20;
  const invDet = 1 / (a00 * b01 + a01 * b11 + a02 * b21);
  o[0] = b01 * invDet;
  o[1] = (-a22 * a01 + a02 * a21) * invDet;
  o[2] = (a12 * a01 - a02 * a11) * invDet;
  o[3] = b11 * invDet;
  o[4] = (a22 * a00 - a02 * a20) * invDet;
  o[5] = (-a12 * a00 + a02 * a10) * invDet;
  o[6] = b21 * invDet;
  o[7] = (-a21 * a00 + a01 * a20) * invDet;
  o[8] = (a11 * a00 - a01 * a10) * invDet;
  return o;
}

/**
 * out = inverse(a) (GLSL `inverse`, 2x2). Singular matrices are undefined in
 * GLSL; this implementation does not check and may produce NaN/Inf entries.
 */
export function mat2Inverse(a: Mat2, out?: Mat2): Mat2 {
  const o = out ?? new Float32Array(4);
  const a00 = a[0], a01 = a[1], a10 = a[2], a11 = a[3];
  const invDet = 1 / (a00 * a11 - a01 * a10);
  o[0] = a11 * invDet;
  o[1] = -a01 * invDet;
  o[2] = -a10 * invDet;
  o[3] = a00 * invDet;
  return o;
}

/**
 * out = matrixCompMult(a, b) (GLSL `matrixCompMult`): componentwise
 * multiplication of two matrices of the same size (4/9/16 elements).
 */
export function matrixCompMult(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] * b[i];
  return o;
}

/**
 * out = outerProduct(c, r) (GLSL `outerProduct`): result[col][row] =
 * c[col] * r[row]. `c` and `r` must have the same length n (2/3/4); the
 * result is an n×n column-major matrix (n^2 elements).
 */
export function outerProduct(c: Float32Array, r: Float32Array, out?: Float32Array): Float32Array {
  const n = c.length;
  const o = out ?? new Float32Array(n * n);
  // Read all inputs into locals first: `out` may alias `c` or `r`.
  const c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
  const r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3];
  for (let col = 0; col < n; col++) {
    const cv = col === 0 ? c0 : col === 1 ? c1 : col === 2 ? c2 : c3;
    for (let row = 0; row < n; row++) {
      const rv = row === 0 ? r0 : row === 1 ? r1 : row === 2 ? r2 : r3;
      o[col * n + row] = cv * rv;
    }
  }
  return o;
}

// ---------------------------------------------------------------------------
// WebGL2 packing built-ins (GLSL ES 3.00 pack*2x16 / unpack*2x16)
// ---------------------------------------------------------------------------

/**
 * GLSL `packUnorm2x16(v)`: packs v.x, v.y into a uint16 pair:
 * round(clamp(c, 0, 1) * 65535) per component, low 16 bits = v.x.
 */
export function packUnorm2x16(v: Vec2): number {
  const x = roundNum(clampNum(v[0], 0, 1) * 65535);
  const y = roundNum(clampNum(v[1], 0, 1) * 65535);
  return y * 65536 + (x & 0xFFFF);
}

/**
 * GLSL `packSnorm2x16(v)`: round(clamp(c, -1, 1) * 32767) per component,
 * low 16 bits = v.x.
 */
export function packSnorm2x16(v: Vec2): number {
  const x = roundNum(clampNum(v[0], -1, 1) * 32767);
  const y = roundNum(clampNum(v[1], -1, 1) * 32767);
  // `x & 0xFFFF` is required: a negative x would otherwise decrement the
  // high 16 bits (y*65536 + (-k) == (y-1)*65536 + (65536-k)).
  return y * 65536 + (x & 0xFFFF);
}

/**
 * GLSL `packHalf2x16(v)`: float16 (binary16) bit patterns per component,
 * low 16 bits = v.x (uses `toHalfFloat`).
 */
export function packHalf2x16(v: Vec2): number {
  const x = toHalfFloat(v[0]);
  const y = toHalfFloat(v[1]);
  return y * 65536 + (x & 0xFFFF);
}

/**
 * GLSL `unpackUnorm2x16(packed)`: out.x = (packed & 0xFFFF) / 65535,
 * out.y = (packed >>> 16) / 65535.
 */
export function unpackUnorm2x16(packed: number, out?: Vec2): Vec2 {
  const o = out ?? new Float32Array(2);
  o[0] = (packed & 0xFFFF) / 65535;
  o[1] = (packed >>> 16) / 65535;
  return o;
}

/**
 * GLSL `unpackSnorm2x16(packed)`: out.x = clamp(s16(packed) / 32767, -1, 1),
 * out.y = clamp(s16(packed >>> 16) / 32767, -1, 1) where s16 sign-extends.
 */
export function unpackSnorm2x16(packed: number, out?: Vec2): Vec2 {
  const o = out ?? new Float32Array(2);
  o[0] = clampNum(s16(packed & 0xFFFF) / 32767, -1, 1);
  o[1] = clampNum(s16(packed >>> 16) / 32767, -1, 1);
  return o;
}

/**
 * GLSL `unpackHalf2x16(packed)`: out.x = float16(packed & 0xFFFF),
 * out.y = float16(packed >>> 16) (uses `fromHalfFloat`).
 */
export function unpackHalf2x16(packed: number, out?: Vec2): Vec2 {
  const o = out ?? new Float32Array(2);
  o[0] = fromHalfFloat(packed & 0xFFFF);
  o[1] = fromHalfFloat(packed >>> 16);
  return o;
}

// ---------------------------------------------------------------------------
// Private helpers (not exported)
// ---------------------------------------------------------------------------

/** Componentwise unary map; elementwise read-before-write → alias-safe. */
function unary(a: Float32Array, fn: (x: number) => number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = fn(a[i]);
  return o;
}

/** Componentwise binary map; elementwise read-before-write → alias-safe. */
function binary(a: Float32Array, b: Float32Array, fn: (x: number, y: number) => number, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = fn(a[i], b[i]);
  return o;
}

/** Resolves a GLSL scalar-or-vector argument to the i-th component value. */
function resolveSV(v: number | Float32Array, i: number): number {
  return typeof v === 'number' ? v : v[i];
}

// Scalar GLSL semantics: private mirrors of the './misc' scalar built-ins
// (same contracts), kept local so this module does not depend on sibling
// implementation details (e.g. roundEven -0 handling). Module-level function
// references — no per-call closures, so hot paths stay allocation-free.
const addNum = (x: number, y: number): number => x + y;
const subNum = (x: number, y: number): number => x - y;
const mulNum = (x: number, y: number): number => x * y;
const divNum = (x: number, y: number): number => x / y;
const negNum = (x: number): number => -x;
const absNum = Math.abs;
const signNum = Math.sign;
const floorNum = Math.floor;
const ceilNum = Math.ceil;
const truncNum = Math.trunc;
const sqrtNum = Math.sqrt;
const expNum = Math.exp;
const logNum = Math.log;
const log2Num = Math.log2;
const exp2Num = (x: number): number => 2 ** x;
const inversesqrtNum = (x: number): number => 1 / Math.sqrt(x);
const powNum = (x: number, y: number): number => Math.pow(x, y);
const atan2Num = (y: number, x: number): number => Math.atan2(y, x);
const minNum = Math.min;
const maxNum = Math.max;

/** GLSL `clamp(x, lo, hi)` = min(max(x, lo), hi). */
function clampNum(x: number, lo: number, hi: number): number {
  return minNum(maxNum(x, lo), hi);
}

/** GLSL `fract(x)` = x - floor(x). */
function fractNum(x: number): number {
  return x - floorNum(x);
}

/** GLSL `mod(x, y)` = x - y * floor(x / y). NOT the JS % operator. */
function modNum(x: number, y: number): number {
  return x - y * floorNum(x / y);
}

/** GLSL `mix(x, y, t)` = x * (1 - t) + y * t (exact form). */
function mixNum(x: number, y: number, t: number): number {
  return x * (1 - t) + y * t;
}

/** GLSL `step(edge, x)` = x < edge ? 0 : 1. */
function stepNum(edge: number, x: number): number {
  return x < edge ? 0 : 1;
}

/**
 * GLSL `round(x)` — nearest integer, halves away from zero (NOT Math.round,
 * which rounds halves toward +Infinity).
 */
function roundNum(x: number): number {
  return signNum(x) * floorNum(absNum(x) + 0.5);
}

/**
 * GLSL `roundEven(x)` — nearest integer, ties to even. Preserves negative
 * zero when the correctly-rounded result is 0 and the input is negative
 * (spec: roundEven(-0.5) = -0.0). NaN/±Inf pass through.
 */
function roundEvenNum(x: number): number {
  const f = floorNum(x);
  const frac = x - f;
  let r: number;
  if (frac > 0.5) r = f + 1;
  else if (frac < 0.5) r = f;
  else if (frac === 0.5) r = f % 2 === 0 ? f : f + 1;
  else r = f; // frac is NaN (x = ±Inf or NaN): pass the floor through
  if (r === 0 && x < 0) return -0;
  return r;
}

/**
 * GLSL `smoothstep(e0, e1, x)`: t = clamp((x-e0)/(e1-e0), 0, 1);
 * result = t*t*(3-2*t). e0 == e1 is undefined in GLSL; returns 0 (no NaN).
 */
function smoothstepNum(e0: number, e1: number, x: number): number {
  if (e0 === e1) return 0;
  const t = clampNum((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** Sign-extends a 16-bit two's-complement value to a JS number. */
function s16(v: number): number {
  return (v << 16) >> 16;
}

// Private IEEE 754 binary16 (half-float) conversion — mirrors of the './misc'
// toHalfFloat/fromHalfFloat contracts (round-to-nearest-even, subnormals,
// ±Inf, NaN), kept local so packHalf2x16/unpackHalf2x16 work independently.

/** Converts a float32 value to its binary16 bit pattern (uint16). */
function toHalfFloat(f: number): number {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setFloat32(0, f);
  const bits = dv.getUint32(0);
  const sign = (bits >>> 16) & 0x8000;
  const exp = (bits >>> 23) & 0xff;
  const mant = bits & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 | (mant >>> 13) : 0); // ±Inf / NaN
  if (exp === 0) return sign; // ±0 or float32 subnormal (< 2^-126) → ±0
  const halfExp = exp - 112; // exp - 127 + 15
  if (halfExp >= 31) return sign | 0x7c00; // overflow → ±Inf (max finite 65504)
  if (halfExp <= 0) {
    // Half subnormal: value = m10 * 2^-24. q = f * 2^24 is exact (float32 is
    // exact in double; power-of-two scaling); round q to nearest even.
    const m10 = roundEvenNum(f * 16777216);
    if (m10 >= 1024) return sign | 0x0400; // rounded up to min normal 2^-14
    return m10 <= 0 ? sign : sign | m10;
  }
  // Normal half: 10 stored mantissa bits + 13 dropped, round-to-nearest-even.
  const m10 = mant >> 13;
  const dropped = mant & 0x1fff;
  const hm = dropped > 0x1000 || (dropped === 0x1000 && (m10 & 1) === 1) ? m10 + 1 : m10;
  return hm > 1023 ? sign | ((halfExp + 1) << 10) : sign | (halfExp << 10) | hm;
}

/** Converts a binary16 bit pattern (uint16) to a float32 value. Exact for finite values. */
function fromHalfFloat(h: number): number {
  const neg = (h & 0x8000) !== 0;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) {
    if (mant === 0) return neg ? -0 : 0; // ±0
    const v = mant * 2 ** -24; // subnormal
    return neg ? -v : v;
  }
  if (exp === 0x1f) {
    if (mant === 0) return neg ? -Infinity : Infinity;
    return NaN;
  }
  const v = (1 + mant / 1024) * 2 ** (exp - 15); // normal
  return neg ? -v : v;
}
