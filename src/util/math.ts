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
  throw new Error('not implemented');
}

/**
 * Allocates a new vec3. GLSL `vec3(x, y, z)` constructor semantics.
 */
export function vec3(x = 0, y = 0, z = 0): Vec3 {
  throw new Error('not implemented');
}

/**
 * Allocates a new vec4. GLSL `vec4(x, y, z, w)` constructor semantics.
 */
export function vec4(x = 0, y = 0, z = 0, w = 0): Vec4 {
  throw new Error('not implemented');
}

/**
 * Allocates a new mat2 with diagonal `s` (GLSL `mat2()` = identity, `mat2(s)`
 * = diagonal s; off-diagonal zeros). Column-major.
 */
export function mat2(s = 1): Mat2 {
  throw new Error('not implemented');
}

/**
 * Allocates a new mat3 with diagonal `s` (GLSL `mat3()` = identity, `mat3(s)`
 * = diagonal s; off-diagonal zeros). Column-major.
 */
export function mat3(s = 1): Mat3 {
  throw new Error('not implemented');
}

/**
 * Allocates a new mat4 with diagonal `s` (GLSL `mat4()` = identity, `mat4(s)`
 * = diagonal s; off-diagonal zeros). Column-major.
 */
export function mat4(s = 1): Mat4 {
  throw new Error('not implemented');
}

/**
 * Fills `out` (length 4) with the identity matrix and returns it.
 */
export function mat2Identity(out: Mat2): Mat2 {
  throw new Error('not implemented');
}

/**
 * Fills `out` (length 9) with the identity matrix and returns it.
 */
export function mat3Identity(out: Mat3): Mat3 {
  throw new Error('not implemented');
}

/**
 * Fills `out` (length 16) with the identity matrix and returns it.
 */
export function mat4Identity(out: Mat4): Mat4 {
  throw new Error('not implemented');
}

// ---------------------------------------------------------------------------
// Componentwise arithmetic
// ---------------------------------------------------------------------------

/**
 * out[i] = a[i] + b[i] (componentwise add; `a` and `b` same length 2/3/4).
 */
export function vecAdd(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = a[i] - b[i] (componentwise subtract; `a` and `b` same length 2/3/4).
 */
export function vecSub(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = a[i] * b[i] (componentwise multiply, NOT dot/cross; `a` and `b`
 * same length 2/3/4).
 */
export function vecMul(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = a[i] / b[i] (componentwise divide; `a` and `b` same length 2/3/4).
 * Division by zero yields Infinity/NaN per IEEE 754, like GPUs.
 */
export function vecDiv(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = a[i] * s (scalar multiply of a vector).
 */
export function vecScale(a: Float32Array, s: number, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = -a[i] (componentwise negation).
 */
export function vecNeg(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = fn(a[i]) — generic componentwise unary map. Used by glsl codegen
 * for trig/exp/log family built-ins (sin, cos, tan, asin, acos, tanh, exp,
 * log2, ...). `fn` is called with one number and must return a number.
 */
export function vecMap(a: Float32Array, fn: (x: number) => number, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
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
  throw new Error('not implemented');
}

// ---------------------------------------------------------------------------
// Geometric functions
// ---------------------------------------------------------------------------

/**
 * GLSL `dot(a, b)` — sum of componentwise products (any length).
 */
export function dot(a: Float32Array, b: Float32Array): number {
  throw new Error('not implemented');
}

/**
 * GLSL `cross(a, b)` — 3D cross product. Both inputs must be length 3.
 */
export function cross(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * GLSL `length(a)` = sqrt(dot(a, a)).
 */
export function length(a: Float32Array): number {
  throw new Error('not implemented');
}

/**
 * GLSL `length(a)` squared = dot(a, a) (avoids sqrt; for comparisons).
 */
export function lengthSq(a: Float32Array): number {
  throw new Error('not implemented');
}

/**
 * GLSL `distance(a, b)` = length(a - b).
 */
export function distance(a: Float32Array, b: Float32Array): number {
  throw new Error('not implemented');
}

/**
 * GLSL `distance(a, b)` squared (avoids sqrt; for comparisons).
 */
export function distanceSq(a: Float32Array, b: Float32Array): number {
  throw new Error('not implemented');
}

/**
 * GLSL `normalize(a)` = a / length(a). Zero vector is undefined in GLSL;
 * this implementation returns the zero vector unchanged (no NaN).
 */
export function normalize(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * GLSL `reflect(I, N)` = I - 2 * dot(N, I) * N. All inputs length 3.
 */
export function reflect(I: Float32Array, N: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * GLSL `refract(I, N, eta)`:
 *   k = 1 - eta^2 * (1 - dot(N, I)^2)
 *   if k < 0: return zero vector, else eta*I - (eta*dot(N,I) + sqrt(k))*N.
 * All inputs length 3.
 */
export function refract(I: Float32Array, N: Float32Array, eta: number, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * GLSL `faceforward(N, I, Nref)` = dot(Nref, I) < 0 ? N : -N. All inputs length 3.
 */
export function faceforward(N: Float32Array, I: Float32Array, Nref: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

// ---------------------------------------------------------------------------
// Componentwise GLSL functions (exact GLSL ES semantics — single source of truth)
// ---------------------------------------------------------------------------

/**
 * out[i] = abs(a[i]) (GLSL `abs`).
 */
export function vecAbs(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = sign(a[i]) — 1.0 if > 0, -1.0 if < 0, 0.0 if == 0 (GLSL `sign`).
 */
export function vecSign(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = floor(a[i]) (GLSL `floor`).
 */
export function vecFloor(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = ceil(a[i]) (GLSL `ceil`).
 */
export function vecCeil(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = trunc(a[i]) (GLSL `trunc` — toward zero).
 */
export function vecTrunc(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = round(a[i]) (GLSL `round` — round half away from zero).
 */
export function vecRound(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = roundEven(a[i]) (GLSL `roundEven` — round half to even).
 */
export function vecRoundEven(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = fract(a[i]) = a[i] - floor(a[i]) (GLSL `fract`).
 */
export function vecFract(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = sqrt(a[i]) (GLSL `sqrt`).
 */
export function vecSqrt(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = inversesqrt(a[i]) = 1/sqrt(a[i]) (GLSL `inversesqrt`).
 */
export function vecInversesqrt(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = exp(a[i]) (GLSL `exp`).
 */
export function vecExp(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = log(a[i]) (GLSL `log`, natural log).
 */
export function vecLog(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = exp2(a[i]) (GLSL `exp2`).
 */
export function vecExp2(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = log2(a[i]) (GLSL `log2`).
 */
export function vecLog2(a: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = pow(a[i], b[i]) (GLSL `pow`; undefined for a[i] < 0, like GLSL).
 */
export function vecPow(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = atan(y[i], x[i]) = atan2(y[i], x[i]) (GLSL two-arg `atan`).
 */
export function vecAtan2(y: Float32Array, x: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = min(a[i], b[i]) (GLSL `min`).
 */
export function vecMin(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = max(a[i], b[i]) (GLSL `max`).
 */
export function vecMax(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = clamp(a[i], lo, hi) = min(max(a[i], lo), hi). `lo`/`hi` may each be
 * a scalar (broadcast) or a vector of the same length (GLSL `clamp`).
 * Undefined when lo > hi (GLSL); result follows min(max) evaluation.
 */
export function vecClamp(
  a: Float32Array,
  lo: number | Float32Array,
  hi: number | Float32Array,
  out?: Float32Array
): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = mix(x[i], y[i], t) = x[i]*(1-t) + y[i]*t. `t` may be a scalar
 * (broadcast) or a vector of the same length (GLSL `mix`).
 */
export function vecMix(
  x: Float32Array,
  y: Float32Array,
  t: number | Float32Array,
  out?: Float32Array
): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = mod(x[i], y) = x[i] - y*floor(x[i]/y). `y` may be a scalar
 * (broadcast) or a vector of the same length (GLSL `mod`; NOT the JS % op).
 */
export function vecMod(x: Float32Array, y: number | Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = step(edge, x[i]) = x[i] < edge ? 0 : 1. `edge` may be a scalar
 * (broadcast) or a vector of the same length (GLSL `step`).
 */
export function vecStep(edge: number | Float32Array, x: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out[i] = smoothstep(e0, e1, x[i]):
 *   t = clamp((x - e0)/(e1 - e0), 0, 1); result = t*t*(3 - 2*t).
 * `e0`/`e1` may be scalars (broadcast) or vectors of the same length.
 * e0 == e1 is undefined in GLSL; this implementation returns 0 (no NaN).
 */
export function vecSmoothstep(
  e0: number | Float32Array,
  e1: number | Float32Array,
  x: Float32Array,
  out?: Float32Array
): Float32Array {
  throw new Error('not implemented');
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
  throw new Error('not implemented');
}

/**
 * out = a * b (GLSL matrix multiply, 3x3). All matrices length 9.
 */
export function mat3Mul(a: Mat3, b: Mat3, out?: Mat3): Mat3 {
  throw new Error('not implemented');
}

/**
 * out = a * b (GLSL matrix multiply, 2x2). All matrices length 4.
 */
export function mat2Mul(a: Mat2, b: Mat2, out?: Mat2): Mat2 {
  throw new Error('not implemented');
}

/**
 * out = m * v (GLSL mat4 * vec4). `m` length 16, `v`/`out` length 4.
 * out[r] = sum over k of m[k*4 + r] * v[k].
 */
export function mat4MulVec4(m: Mat4, v: Vec4, out?: Vec4): Vec4 {
  throw new Error('not implemented');
}

/**
 * out = m * v (GLSL mat3 * vec3). `m` length 9, `v`/`out` length 3.
 */
export function mat3MulVec3(m: Mat3, v: Vec3, out?: Vec3): Vec3 {
  throw new Error('not implemented');
}

/**
 * out = m * v (GLSL mat2 * vec2). `m` length 4, `v`/`out` length 2.
 */
export function mat2MulVec2(m: Mat2, v: Vec2, out?: Vec2): Vec2 {
  throw new Error('not implemented');
}

/**
 * out = v * m (GLSL vec4 * mat4, row-vector multiply). `m` length 16,
 * `v`/`out` length 4. out[c] = sum over k of v[k] * m[c*4 + k].
 */
export function vec4MulMat4(v: Vec4, m: Mat4, out?: Vec4): Vec4 {
  throw new Error('not implemented');
}

/**
 * out = v * m (GLSL vec3 * mat3, row-vector multiply). `m` length 9.
 */
export function vec3MulMat3(v: Vec3, m: Mat3, out?: Vec3): Vec3 {
  throw new Error('not implemented');
}

/**
 * out = v * m (GLSL vec2 * mat2, row-vector multiply). `m` length 4.
 */
export function vec2MulMat2(v: Vec2, m: Mat2, out?: Vec2): Vec2 {
  throw new Error('not implemented');
}

/**
 * out = transpose(a) (GLSL `transpose`). Column-major storage: out[c*4 + r] =
 * a[r*4 + c].
 */
export function mat4Transpose(a: Mat4, out?: Mat4): Mat4 {
  throw new Error('not implemented');
}

/**
 * out = transpose(a) (GLSL `transpose`, 3x3): out[c*3 + r] = a[r*3 + c].
 */
export function mat3Transpose(a: Mat3, out?: Mat3): Mat3 {
  throw new Error('not implemented');
}

/**
 * out = transpose(a) (GLSL `transpose`, 2x2): out[c*2 + r] = a[r*2 + c].
 */
export function mat2Transpose(a: Mat2, out?: Mat2): Mat2 {
  throw new Error('not implemented');
}

/**
 * GLSL `determinant(a)` for 4x4 column-major `a`.
 */
export function mat4Determinant(a: Mat4): number {
  throw new Error('not implemented');
}

/**
 * GLSL `determinant(a)` for 3x3 column-major `a`.
 */
export function mat3Determinant(a: Mat3): number {
  throw new Error('not implemented');
}

/**
 * GLSL `determinant(a)` for 2x2 column-major `a`.
 */
export function mat2Determinant(a: Mat2): number {
  throw new Error('not implemented');
}

/**
 * out = inverse(a) (GLSL `inverse`, 4x4). Singular matrices are undefined in
 * GLSL; this implementation does not check and may produce NaN/Inf entries.
 */
export function mat4Inverse(a: Mat4, out?: Mat4): Mat4 {
  throw new Error('not implemented');
}

/**
 * out = inverse(a) (GLSL `inverse`, 3x3). Singular matrices are undefined in
 * GLSL; this implementation does not check and may produce NaN/Inf entries.
 */
export function mat3Inverse(a: Mat3, out?: Mat3): Mat3 {
  throw new Error('not implemented');
}

/**
 * out = inverse(a) (GLSL `inverse`, 2x2). Singular matrices are undefined in
 * GLSL; this implementation does not check and may produce NaN/Inf entries.
 */
export function mat2Inverse(a: Mat2, out?: Mat2): Mat2 {
  throw new Error('not implemented');
}

/**
 * out = matrixCompMult(a, b) (GLSL `matrixCompMult`): componentwise
 * multiplication of two matrices of the same size (4/9/16 elements).
 */
export function matrixCompMult(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

/**
 * out = outerProduct(c, r) (GLSL `outerProduct`): result[col][row] =
 * c[col] * r[row]. `c` and `r` must have the same length n (2/3/4); the
 * result is an n×n column-major matrix (n^2 elements).
 */
export function outerProduct(c: Float32Array, r: Float32Array, out?: Float32Array): Float32Array {
  throw new Error('not implemented');
}

// ---------------------------------------------------------------------------
// WebGL2 packing built-ins (GLSL ES 3.00 pack*2x16 / unpack*2x16)
// ---------------------------------------------------------------------------

/**
 * GLSL `packUnorm2x16(v)`: packs v.x, v.y into a uint16 pair:
 * round(clamp(c, 0, 1) * 65535) per component, low 16 bits = v.x.
 */
export function packUnorm2x16(v: Vec2): number {
  throw new Error('not implemented');
}

/**
 * GLSL `packSnorm2x16(v)`: round(clamp(c, -1, 1) * 32767) per component,
 * low 16 bits = v.x.
 */
export function packSnorm2x16(v: Vec2): number {
  throw new Error('not implemented');
}

/**
 * GLSL `packHalf2x16(v)`: float16 (binary16) bit patterns per component,
 * low 16 bits = v.x (uses `toHalfFloat`).
 */
export function packHalf2x16(v: Vec2): number {
  throw new Error('not implemented');
}

/**
 * GLSL `unpackUnorm2x16(packed)`: out.x = (packed & 0xFFFF) / 65535,
 * out.y = (packed >>> 16) / 65535.
 */
export function unpackUnorm2x16(packed: number, out?: Vec2): Vec2 {
  throw new Error('not implemented');
}

/**
 * GLSL `unpackSnorm2x16(packed)`: out.x = clamp(s16(packed) / 32767, -1, 1),
 * out.y = clamp(s16(packed >>> 16) / 32767, -1, 1) where s16 sign-extends.
 */
export function unpackSnorm2x16(packed: number, out?: Vec2): Vec2 {
  throw new Error('not implemented');
}

/**
 * GLSL `unpackHalf2x16(packed)`: out.x = float16(packed & 0xFFFF),
 * out.y = float16(packed >>> 16) (uses `fromHalfFloat`).
 */
export function unpackHalf2x16(packed: number, out?: Vec2): Vec2 {
  throw new Error('not implemented');
}
