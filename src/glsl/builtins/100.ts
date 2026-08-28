/**
 * builtins/100.ts — GLSL ES 1.00 built-in FUNCTIONS, VARIABLES and gl_Max*
 * constants, exactly as WebGL 1.0 exposes them.
 *
 * Sources: GLSL ES 1.00 spec §8 (function tables) with the WebGL 1.0
 * restrictions applied:
 * - WebGL 1.0 deliberately removes the ES 1.00 shadow samplers
 *   (sampler2DShadow / samplerCubeShadow) from the language — they cannot even
 *   be declared, so NO shadow texture functions appear in the core 1.00 table
 *   (the only shadow function in this codebase is texture2DShadowLodEXT from
 *   the GL_EXT_shader_texture_lod extension — see extensions.ts).
 * - texture2DLod / texture2DProjLod / textureCubeLod exist only in VERTEX
 *   shaders in core ES 1.00 (stage: 'VERTEX'); the EXT_shader_texture_lod
 *   extension adds fragment-stage + Grad variants under EXT names.
 * - gl_FragData is a core ES 1.00 builtin (WebGL1: gl_MaxDrawBuffers = 1, so
 *   only index 0 is legal — CTS gl-fragdata-and-fragcolor.html relies on a
 *   core shader compiling a write to gl_FragData[0]). GL_EXT_draw_buffers
 *   raises the array size to 4 (see extensions.ts, which overrides this
 *   entry when the extension is enabled).
 *
 * Counts (verified by selftest-builtins.ts): 216 function signatures in
 * builtinFunctions100 + 24 ES 1.00 relational int/bool variants in
 * `relational100` (merged into the version-100 table by builtinSignatures(100)
 * — see below), 8 variables, 8 constants.
 */
import {
  arr,
  B,
  bvec,
  F,
  gen,
  gen2,
  gen3,
  genType,
  ivec,
  mat,
  sig,
  smp,
  V2,
  V3,
  V4,
  vec,
} from './types.js';
import type { BuiltinConstant, BuiltinSignature, BuiltinVariable } from './types.js';
import type { GLSLType } from '../types.js';

/**
 * min/max/clamp-style: (genType, genType[, genType]) plus a float-scalar
 * variant. The scalar variant for t = float would duplicate the (t, t[, t])
 * row (e.g. min(float, float) appears in both spec rows), so it is skipped
 * for the scalar case — every signature is unique.
 */
const scalarMix = (name: string, scalarCount: 2 | 3): BuiltinSignature[] => {
  const all: BuiltinSignature[] = [];
  for (const t of genType) {
    const params: GLSLType[] = [];
    for (let i = 0; i < scalarCount; i++) params.push(t);
    all.push(sig(name, params, t));
    const scalar: GLSLType[] = [t, F];
    for (let i = 2; i < scalarCount; i++) scalar.push(F);
    if (t !== F) all.push(sig(name, scalar, t));
  }
  return all;
};

/**
 * §8.6 relational float variants: (vecN, vecN) → bvecN. Deliberately FLOAT-ONLY
 * here: 300.ts builds its table as a superset of builtinFunctions100
 * (`common100`) and supplies the non-float variants itself via rel300/eq300 —
 * adding int/bool variants to this shared helper would DUPLICATE them in the
 * 3.00 table (overload resolution errors "ambiguous call"). The ES 1.00
 * int/bool variants live in `relational100` below, merged only into the
 * version-100 table (builtins/index.ts).
 */
const rel = (name: string): BuiltinSignature[] =>
  ([2, 3, 4] as const).map((n) => sig(name, [vec(n), vec(n)], ({ kind: 'vector', base: 'bool', size: n })));

const bvecN = (n: 2 | 3 | 4): GLSLType => ({ kind: 'vector', base: 'bool', size: n });

/**
 * ES 1.00 §8.6 additions to the vector relational functions (NOT part of
 * builtinFunctions100 — see `rel` for why; merged into the version-100 table
 * by builtinSignatures(100) in builtins/index.ts).
 *
 * Spec text (GLSL ES 1.00 rev 17 §8.6, placeholders "vec"/"ivec"/"bvec"):
 *   bvec lessThan(vec x, vec y)        bvec lessThan(ivec x, ivec y)
 *   bvec lessThanEqual(vec x, vec y)   bvec lessThanEqual(ivec x, ivec y)
 *   bvec greaterThan(vec x, vec y)     bvec greaterThan(ivec x, ivec y)
 *   bvec greaterThanEqual(vec x, vec y) bvec greaterThanEqual(ivec x, ivec y)
 *   bvec equal(vec x, vec y)  bvec equal(ivec x, ivec y)  bvec equal(bvec x, bvec y)
 *   bvec notEqual(vec x, vec y)  bvec notEqual(ivec x, ivec y)  bvec notEqual(bvec x, bvec y)
 * so the lessThan* family takes float AND int vectors (no bvec — no ordering),
 * and equal/notEqual additionally take bool vectors. Verified against the
 * OGLEs equal/notEqual/lessThan/lessThanEqual/greaterThan/greaterThanEqual
 * groups (ivec2/ivec3 and bvec2/bvec3 shader variants).
 */
export const relational100: BuiltinSignature[] = [
  ...([2, 3, 4] as const).flatMap((n) => [
    sig('lessThan', [ivec(n), ivec(n)], bvecN(n)),
    sig('lessThanEqual', [ivec(n), ivec(n)], bvecN(n)),
    sig('greaterThan', [ivec(n), ivec(n)], bvecN(n)),
    sig('greaterThanEqual', [ivec(n), ivec(n)], bvecN(n)),
    sig('equal', [ivec(n), ivec(n)], bvecN(n)),
    sig('equal', [bvec(n), bvec(n)], bvecN(n)),
    sig('notEqual', [ivec(n), ivec(n)], bvecN(n)),
    sig('notEqual', [bvec(n), bvec(n)], bvecN(n)),
  ]),
];

const boolFn = (name: string): BuiltinSignature[] =>
  ([2, 3, 4] as const).map((n) => sig(name, [{ kind: 'vector', base: 'bool', size: n }], ({ kind: 'vector', base: 'bool', size: n })));

const boolToScalar = (name: string): BuiltinSignature[] =>
  ([2, 3, 4] as const).map((n) => sig(name, [{ kind: 'vector', base: 'bool', size: n }], B));

export const builtinFunctions100: BuiltinSignature[] = [
  /* ------------------------------------------------------------------ */
  /* §8.1 Angle & Trigonometry Functions                                 */
  /* ------------------------------------------------------------------ */
  ...gen('radians', genType),
  ...gen('degrees', genType),
  ...gen('sin', genType),
  ...gen('cos', genType),
  ...gen('tan', genType),
  ...gen('asin', genType),
  ...gen('acos', genType),
  ...gen('atan', genType), // atan(y_over_x)
  ...gen2('atan', genType), // atan(y, x)

  /* ------------------------------------------------------------------ */
  /* §8.2 Exponential Functions                                          */
  /* ------------------------------------------------------------------ */
  ...gen2('pow', genType),
  ...gen('exp', genType),
  ...gen('log', genType),
  ...gen('exp2', genType),
  ...gen('log2', genType),
  ...gen('sqrt', genType),
  ...gen('inversesqrt', genType),

  /* ------------------------------------------------------------------ */
  /* §8.3 Common Functions                                               */
  /* ------------------------------------------------------------------ */
  ...gen('abs', genType),
  ...gen('sign', genType),
  ...gen('floor', genType),
  ...gen('ceil', genType),
  ...gen('fract', genType),
  ...genType.slice(1).map((t) => sig('mod', [t, F], t)), // mod(x, float)
  ...gen2('mod', genType), // mod(x, y)
  ...scalarMix('min', 2),
  ...scalarMix('max', 2),
  ...scalarMix('clamp', 3),
  ...gen3('mix', genType),
  ...genType.slice(1).map((t) => sig('mix', [t, t, F], t)), // mix(x, y, a) with float a
  ...gen2('step', genType), // step(genType edge, genType x)
  ...genType.slice(1).map((t) => sig('step', [F, t], t)), // step(float edge, genType x)
  ...gen3('smoothstep', genType), // smoothstep(genType, genType, genType)
  ...genType.slice(1).map((t) => sig('smoothstep', [F, F, t], t)), // smoothstep(float, float, genType)

  /* ------------------------------------------------------------------ */
  /* §8.4 Geometric Functions                                            */
  /* ------------------------------------------------------------------ */
  ...genType.map((t) => sig('length', [t], F)),
  ...genType.map((t) => sig('distance', [t, t], F)),
  ...genType.map((t) => sig('dot', [t, t], F)),
  sig('cross', [V3, V3], V3),
  ...gen('normalize', genType),
  ...gen3('faceforward', genType),
  ...gen2('reflect', genType),
  ...genType.map((t) => sig('refract', [t, t, F], t)),

  /* ------------------------------------------------------------------ */
  /* §8.5 Matrix Functions                                               */
  /* ------------------------------------------------------------------ */
  ...([2, 3, 4] as const).map((n) => sig('matrixCompMult', [mat(n), mat(n)], mat(n))),
  ...([2, 3, 4] as const).map((n) => sig('outerProduct', [vec(n), vec(n)], mat(n))),
  ...([2, 3, 4] as const).map((n) => sig('transpose', [mat(n)], mat(n))),
  ...([2, 3, 4] as const).map((n) => sig('determinant', [mat(n)], F)),
  ...([2, 3, 4] as const).map((n) => sig('inverse', [mat(n)], mat(n))),

  /* ------------------------------------------------------------------ */
  /* §8.6 Vector Relational Functions — FLOAT variants only here (the     */
  /* ES 1.00 int/bool variants live in `relational100`, merged by          */
  /* builtinSignatures(100); see the `rel` helper comment).                */
  /* ------------------------------------------------------------------ */
  ...rel('lessThan'),
  ...rel('lessThanEqual'),
  ...rel('greaterThan'),
  ...rel('greaterThanEqual'),
  ...rel('equal'),
  ...rel('notEqual'),
  ...boolToScalar('any'),
  ...boolToScalar('all'),
  ...boolFn('not'),

  /* ------------------------------------------------------------------ */
  /* §8.7 Texture Lookup Functions (WebGL 1.0 core: no shadow samplers)  */
  /* ------------------------------------------------------------------ */
  sig('texture2D', [smp('sampler2D'), V2], V4),
  sig('texture2D', [smp('sampler2D'), V2, F], V4), // bias
  sig('texture2DProj', [smp('sampler2D'), V3], V4),
  sig('texture2DProj', [smp('sampler2D'), V4], V4),
  sig('texture2DProj', [smp('sampler2D'), V3, F], V4), // bias
  sig('texture2DProj', [smp('sampler2D'), V4, F], V4), // bias
  sig('texture2DLod', [smp('sampler2D'), V2, F], V4, { stage: 'VERTEX' }),
  sig('texture2DProjLod', [smp('sampler2D'), V3, F], V4, { stage: 'VERTEX' }),
  sig('texture2DProjLod', [smp('sampler2D'), V4, F], V4, { stage: 'VERTEX' }),
  sig('textureCube', [smp('samplerCube'), V3], V4),
  sig('textureCube', [smp('samplerCube'), V3, F], V4), // bias
  sig('textureCubeLod', [smp('samplerCube'), V3, F], V4, { stage: 'VERTEX' }),
];

/**
 * GLSL ES 1.00 §7.6 built-in uniform state type:
 * `uniform gl_DepthRangeParameters { float near; float far; float diff; }`.
 */
const depthRangeParams: GLSLType = {
  kind: 'struct',
  name: 'gl_DepthRangeParameters',
  members: [
    { name: 'near', type: F },
    { name: 'far', type: F },
    { name: 'diff', type: F },
  ],
};

export const builtinVariables100: BuiltinVariable[] = [
  { name: 'gl_Position', type: V4, stage: 'VERTEX', writable: true },
  { name: 'gl_PointSize', type: F, stage: 'VERTEX', writable: true },
  { name: 'gl_FragCoord', type: V4, stage: 'FRAGMENT', writable: false },
  { name: 'gl_FrontFacing', type: B, stage: 'FRAGMENT', writable: false },
  { name: 'gl_PointCoord', type: V2, stage: 'FRAGMENT', writable: false },
  { name: 'gl_FragColor', type: V4, stage: 'FRAGMENT', writable: true },
  // Core ES 1.00: size = gl_MaxDrawBuffers (1 in WebGL 1.0). GL_EXT_draw_buffers
  // overrides this entry with a size-4 array (extensions.ts).
  { name: 'gl_FragData', type: arr(V4, 1), stage: 'FRAGMENT', writable: true },
  // Core ES 1.00 §7.6 built-in uniform state (usable in BOTH stages):
  // `uniform gl_DepthRangeParameters { float near; float far; float diff; }
  //  gl_DepthRange;`. Member reads (gl_DepthRange.near) resolve via the struct
  // type (semantics analyzeMember). NOTE: codegen has no gl_DepthRange
  // lowering yet (codegen/env.ts builtinRef + the member walker) — LINK of a
  // shader reading it is not supported; compile is.
  { name: 'gl_DepthRange', type: depthRangeParams, stage: 'BOTH', writable: false },
];

/**
 * gl_Max* values = the EXACT limits gl/ reports via getParameter
 * (src/gl/state.ts defaultLimits — MAX_VERTEX_UNIFORM_VECTORS 4096,
 * MAX_VARYING_VECTORS 64, MAX_COMBINED_TEXTURE_IMAGE_UNITS 32,
 * MAX_TEXTURE_IMAGE_UNITS 16, MAX_FRAGMENT_UNIFORM_VECTORS 4096). The CTS
 * (conformance/glsl/variables/glsl-built-ins.html) requires each builtin to
 * EQUAL gl.getParameter's value. gl_MaxDrawBuffers stays 1 (WebGL1 core,
 * no WEBGL_draw_buffers); gl_MaxVertexAttribs / gl_MaxVertexTextureImageUnits
 * are 16 in defaultLimits.
 */
export const builtinConstants100: BuiltinConstant[] = [
  { name: 'gl_MaxVertexAttribs', value: 16 },
  { name: 'gl_MaxVertexUniformVectors', value: 4096 },
  { name: 'gl_MaxVaryingVectors', value: 64 },
  { name: 'gl_MaxVertexTextureImageUnits', value: 16 },
  { name: 'gl_MaxCombinedTextureImageUnits', value: 32 },
  { name: 'gl_MaxTextureImageUnits', value: 16 },
  { name: 'gl_MaxFragmentUniformVectors', value: 4096 },
  { name: 'gl_MaxDrawBuffers', value: 1 },
];
