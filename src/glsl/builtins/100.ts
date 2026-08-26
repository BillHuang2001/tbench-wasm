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
 * Counts (verified by selftest-builtins.ts): 216 function signatures,
 * 7 variables, 8 constants.
 */
import {
  arr,
  B,
  F,
  gen,
  gen2,
  gen3,
  genType,
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

const rel = (name: string): BuiltinSignature[] =>
  ([2, 3, 4] as const).map((n) => sig(name, [vec(n), vec(n)], ({ kind: 'vector', base: 'bool', size: n })));

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
  /* §8.6 Vector Relational Functions (float vectors only in ES 1.00)    */
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
];

/**
 * gl_Max* values = the EXACT WebGL 1.0 minimums (gl/getParameter returns the
 * implementation's real capability; these must stay consistent with gl/).
 * gl_MaxVertexTextureImageUnits is 16 here because the software renderer
 * supports 16 vertex texture units.
 */
export const builtinConstants100: BuiltinConstant[] = [
  { name: 'gl_MaxVertexAttribs', value: 16 },
  { name: 'gl_MaxVertexUniformVectors', value: 128 },
  { name: 'gl_MaxVaryingVectors', value: 8 },
  { name: 'gl_MaxVertexTextureImageUnits', value: 16 },
  { name: 'gl_MaxCombinedTextureImageUnits', value: 8 },
  { name: 'gl_MaxTextureImageUnits', value: 8 },
  { name: 'gl_MaxFragmentUniformVectors', value: 16 },
  { name: 'gl_MaxDrawBuffers', value: 1 },
];
