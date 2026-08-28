/**
 * builtins/300.ts — GLSL ES 3.00 built-in FUNCTIONS, VARIABLES and gl_Max*
 * constants (WebGL 2.0). A `#version 300 es` shader uses ONLY this table.
 *
 * The 3.00 table is a superset of the 1.00 core MATH functions (same
 * signatures) PLUS the 3.00 additions. The one deliberate exception: the
 * ES 1.00 texture functions (texture2D/texture2DProj/texture2DLod/
 * texture2DProjLod/textureCube/textureCubeLod) are REMOVED in 3.00 and
 * replaced by the texture()/textureProj()/textureLod()/... family — see
 * REPLACED_100_NAMES below. Variables differ too (no gl_FragColor/
 * gl_FragData; gl_VertexID/gl_InstanceID/gl_FragDepth added; gl_DepthRange
 * is shared with 1.00 — see builtinVariables300).
 *
 * ES 3.00 specifics encoded here:
 * - Shadow texture functions return FLOAT (not vec4) — a 3.00 change.
 * - texture(sampler2DArrayShadow, vec4) has NO bias variant; all other
 *   shadow texture() forms do.
 * - textureLod has a sampler2DShadow variant but NO samplerCubeShadow /
 *   sampler2DArrayShadow variants (those arrived in later revisions).
 * - texelFetch/texelFetchOffset exist only for 2D/3D/2DArray samplers
 *   (no cube, no shadow).
 * - textureSize returns ivec2/ivec3 for EVERY sampler kind (3 components
 *   for 3D / 2DArray / 2DArrayShadow).
 * - abs/sign take genType + genIType; min/max/clamp take genType/genIType/
 *   genUType (each with its scalar overload); mix adds a genBType selector.
 * - isnan/isinf take genType (float) args only and return genBType.
 *   trunc/round/roundEven are genType only.
 * - bitCount/findLSB/findMSB return genIType even for genUType inputs.
 * - NOT included (deliberately): textureGather, textureQueryLod,
 *   textureQueryLevels (not in ES 3.00), fma (not in ES 3.00).
 * - gl_ClipDistance / gl_CullDistance (float[8]) and gl_MaxCullDistances /
 *   gl_MaxCombinedClipAndCullDistances ARE in the 3.00 table as
 *   GL_ANGLE_clip_cull_distance-gated entries (WebGL 2.0's CTS exercises
 *   them only through the WEBGL_clip_cull_distance extension) — visible
 *   only when the shader enables that extension, mirroring gl_DrawID's
 *   extension-gating shape.
 *
 * Counts (verified by selftest-builtins.ts): 625 function signatures,
 * 12 variables, 22 constants.
 */
import { builtinFunctions100 } from './100.js';
import {
  B,
  F,
  I,
  U,
  VOID,
  arr,
  bvec,
  gen,
  gen2,
  gen3,
  genBType,
  genIType,
  genType,
  genUType,
  ivec,
  mat,
  sig,
  smp,
  uvec,
  v,
  vec,
  V2,
  V3,
  V4,
} from './types.js';
import type { BuiltinConstant, BuiltinSignature, BuiltinVariable } from './types.js';
import type { GLSLType, SamplerKind } from '../types.js';

/* ------------------------------------------------------------------ */
/* 1.00 core functions that do NOT carry into 3.00                     */
/* ------------------------------------------------------------------ */

const REPLACED_100_NAMES: ReadonlySet<string> = new Set([
  'texture2D',
  'texture2DProj',
  'texture2DLod',
  'texture2DProjLod',
  'textureCube',
  'textureCubeLod',
]);

/** ES 1.00 math core, minus the replaced texture functions. */
const common100 = builtinFunctions100.filter((s) => !REPLACED_100_NAMES.has(s.name));

/* ------------------------------------------------------------------ */
/* Local construction helpers                                          */
/* ------------------------------------------------------------------ */

/** min/max-style: (t, t) → t and (t, scalar) → t for each t in `types`.
 * The scalar variant for t = scalar would duplicate (t, t) (e.g. min(int,int)
 * appears in both spec rows), so it is skipped — every signature is unique. */
const minmax = (name: string, types: GLSLType[], scalar: GLSLType): BuiltinSignature[] => {
  const out: BuiltinSignature[] = [];
  for (const t of types) {
    out.push(sig(name, [t, t], t));
    if (t !== scalar) out.push(sig(name, [t, scalar], t));
  }
  return out;
};

/** clamp-style: (t, t, t) → t and (t, scalar, scalar) → t for each t in `types` (scalar-t case skipped, see minmax). */
const clampFns = (name: string, types: GLSLType[], scalar: GLSLType): BuiltinSignature[] => {
  const out: BuiltinSignature[] = [];
  for (const t of types) {
    out.push(sig(name, [t, t, t], t));
    if (t !== scalar) out.push(sig(name, [t, scalar, scalar], t));
  }
  return out;
};

/** Size-matched unary conversion: f(genType[i]) → to[i]. */
const convert = (name: string, to: GLSLType[]): BuiltinSignature[] =>
  genType.map((t, i) => sig(name, [t], to[i]));

/** (genX[i], genX[i], out genX[i]) → genX[i] (uaddCarry / usubBorrow). */
const outParam = (name: string, types: GLSLType[]): BuiltinSignature[] =>
  types.map((t) => sig(name, [t, t, t], t));

/** (genX[i], genX[i], out genX[i], out genX[i]) → void (umul/imulExtended). */
const outParam2 = (name: string, types: GLSLType[]): BuiltinSignature[] =>
  types.map((t) => sig(name, [t, t, t, t], VOID));

/** bitCount/findLSB/findMSB: genIType→genIType and genUType→genIType (size-matched). */
const toGenIType = (name: string): BuiltinSignature[] => {
  const out: BuiltinSignature[] = [];
  for (let i = 0; i < 4; i++) {
    out.push(sig(name, [genIType[i]], genIType[i]));
    out.push(sig(name, [genUType[i]], genIType[i]));
  }
  return out;
};

/* ------------------------------------------------------------------ */
/* 3.00 texture helpers                                                */
/* ------------------------------------------------------------------ */

const G2D = ['sampler2D', 'isampler2D', 'usampler2D'] as const;
const G3D = ['sampler3D', 'isampler3D', 'usampler3D'] as const;
const GCUBE = ['samplerCube', 'isamplerCube', 'usamplerCube'] as const;
const G2DARRAY = ['sampler2DArray', 'isampler2DArray', 'usampler2DArray'] as const;

/** Return type for a (non-shadow) gsampler lookup: vec4/ivec4/uvec4 by prefix. */
const gvec4 = (k: SamplerKind): GLSLType => (k.startsWith('i') ? ivec(4) : k.startsWith('u') ? uvec(4) : vec(4));

const isShadow = (k: SamplerKind): boolean => k.endsWith('Shadow');

/** textureSize return: ivec3 for 3D/2DArray/2DArrayShadow, else ivec2. */
const sizeRet = (k: SamplerKind): GLSLType =>
  k.endsWith('3D') || k.endsWith('2DArray') || k === 'sampler2DArrayShadow' ? ivec(3) : ivec(2);

/**
 * Expand `name` over sampler `kinds`: one signature per kind with params
 * [sampler, ...P, ...extra]; when `bias` is set, an additional signature
 * appends a float bias. Non-shadow kinds return gvec4, shadow kinds float
 * (unless `ret` overrides).
 */
const tex = (
  name: string,
  kinds: readonly SamplerKind[],
  P: GLSLType[],
  extra: GLSLType[] = [],
  bias = false,
  ret?: (k: SamplerKind) => GLSLType,
): BuiltinSignature[] => {
  const out: BuiltinSignature[] = [];
  for (const k of kinds) {
    const r = ret ? ret(k) : isShadow(k) ? F : gvec4(k);
    out.push(sig(name, [smp(k), ...P, ...extra], r));
    if (bias) out.push(sig(name, [smp(k), ...P, ...extra, F], r));
  }
  return out;
};

/* ------------------------------------------------------------------ */
/* §8.3 Common Functions — ES 3.00 additions                           */
/* ------------------------------------------------------------------ */

const common300: BuiltinSignature[] = [
  ...gen('abs', genIType),
  ...gen('sign', genIType),
  ...gen('trunc', genType),
  ...gen('round', genType),
  ...gen('roundEven', genType),
  ...genType.map((t) => sig('modf', [t, t], t)), // modf(genType x, out genType i)
  // min/max/clamp genType (float) variants come from common100 — only the
  // int/uint families are new in 3.00.
  ...minmax('min', genIType, I),
  ...minmax('min', genUType, U),
  ...minmax('max', genIType, I),
  ...minmax('max', genUType, U),
  ...clampFns('clamp', genIType, I),
  ...clampFns('clamp', genUType, U),
  ...genType.map((t, i) => sig('mix', [t, t, genBType[i]], t)), // mix(x, y, bvec selector)
  // isnan/isinf: float-family args, genBType returns (GLSL ES 3.00 §8.3).
  ...genType.map((t, i) => sig('isnan', [t], genBType[i])),
  ...genType.map((t, i) => sig('isinf', [t], genBType[i])),
  ...convert('floatBitsToInt', genIType),
  ...convert('floatBitsToUint', genUType),
  ...genIType.map((t) => sig('intBitsToFloat', [t], sizeMatchedFloat(t))),
  ...genUType.map((t) => sig('uintBitsToFloat', [t], sizeMatchedFloat(t))),
  // pack/unpack (exact ES 3.00 set)
  sig('packSnorm2x16', [V2], U),
  sig('packUnorm2x16', [V2], U),
  sig('packHalf2x16', [V2], U),
  sig('packUnorm4x8', [V4], U),
  sig('packSnorm4x8', [V4], U),
  sig('unpackSnorm2x16', [U], V2),
  sig('unpackUnorm2x16', [U], V2),
  sig('unpackHalf2x16', [U], V2),
  sig('unpackUnorm4x8', [U], V4),
  sig('unpackSnorm4x8', [U], V4),
];

/** float type size-matched to t (scalar → float, vector → vecN). */
function sizeMatchedFloat(t: GLSLType): GLSLType {
  return t.kind === 'vector' ? vec(t.size) : F;
}

/* ------------------------------------------------------------------ */
/* §8.6 Vector Relational Functions — ES 3.00 adds int/uint/bool       */
/* (the float variants already come from common100)                    */
/* ------------------------------------------------------------------ */

const rel300 = (name: string): BuiltinSignature[] => {
  const out: BuiltinSignature[] = [];
  for (const base of ['int', 'uint'] as const) {
    for (const n of [2, 3, 4] as const) out.push(sig(name, [v(base, n), v(base, n)], bvec(n)));
  }
  return out;
};

const eq300 = (name: string): BuiltinSignature[] => {
  const out: BuiltinSignature[] = [];
  for (const base of ['int', 'uint', 'bool'] as const) {
    for (const n of [2, 3, 4] as const) out.push(sig(name, [v(base, n), v(base, n)], bvec(n)));
  }
  return out;
};

/* ------------------------------------------------------------------ */
/* §8.7 Integer Functions (bitfield + extended multiply)               */
/* ------------------------------------------------------------------ */

const integer300: BuiltinSignature[] = [
  ...genIType.map((t) => sig('bitfieldExtract', [t, I, I], t)),
  ...genUType.map((t) => sig('bitfieldExtract', [t, I, I], t)),
  ...genIType.map((t) => sig('bitfieldInsert', [t, t, I, I], t)),
  ...genUType.map((t) => sig('bitfieldInsert', [t, t, I, I], t)),
  ...gen('bitfieldReverse', genIType),
  ...gen('bitfieldReverse', genUType),
  ...toGenIType('bitCount'),
  ...toGenIType('findLSB'),
  ...toGenIType('findMSB'),
  ...outParam('uaddCarry', genUType),
  ...outParam('usubBorrow', genUType),
  ...outParam2('umulExtended', genUType),
  ...outParam2('imulExtended', genIType),
];

/* ------------------------------------------------------------------ */
/* §8.8 Texture Lookup Functions                                       */
/* ------------------------------------------------------------------ */

const texture300: BuiltinSignature[] = [
  // texture — implicit LOD, optional bias (NO bias for sampler2DArrayShadow)
  ...tex('texture', G2D, [V2], [], true),
  ...tex('texture', G3D, [V3], [], true),
  ...tex('texture', GCUBE, [V3], [], true),
  ...tex('texture', G2DARRAY, [V3], [], true),
  ...tex('texture', ['sampler2DShadow'], [V3], [], true),
  ...tex('texture', ['samplerCubeShadow'], [V4], [], true),
  ...tex('texture', ['sampler2DArrayShadow'], [V4], [], false),
  // textureProj
  ...tex('textureProj', G2D, [V3], [], true),
  ...tex('textureProj', G2D, [V4], [], true),
  ...tex('textureProj', G3D, [V4], [], true),
  ...tex('textureProj', ['sampler2DShadow'], [V4], [], true),
  // textureLod — explicit LOD; no cube-shadow / array-shadow variants in ES 3.00
  ...tex('textureLod', G2D, [V2], [F]),
  ...tex('textureLod', G3D, [V3], [F]),
  ...tex('textureLod', GCUBE, [V3], [F]),
  ...tex('textureLod', G2DARRAY, [V3], [F]),
  ...tex('textureLod', ['sampler2DShadow'], [V3], [F]),
  // textureOffset
  ...tex('textureOffset', G2D, [V2], [ivec(2)], true),
  ...tex('textureOffset', G3D, [V3], [ivec(3)], true),
  ...tex('textureOffset', G2DARRAY, [V3], [ivec(2)], true),
  ...tex('textureOffset', ['sampler2DShadow'], [V3], [ivec(2)], true),
  // texelFetch / texelFetchOffset — 2D, 3D, 2DArray only
  ...tex('texelFetch', G2D, [ivec(2), I]),
  ...tex('texelFetch', G3D, [ivec(3), I]),
  ...tex('texelFetch', G2DARRAY, [ivec(3), I]),
  ...tex('texelFetchOffset', G2D, [ivec(2), I, ivec(2)]),
  ...tex('texelFetchOffset', G3D, [ivec(3), I, ivec(3)]),
  ...tex('texelFetchOffset', G2DARRAY, [ivec(3), I, ivec(2)]),
  // textureProjOffset
  ...tex('textureProjOffset', G2D, [V3], [ivec(2)], true),
  ...tex('textureProjOffset', G2D, [V4], [ivec(2)], true),
  ...tex('textureProjOffset', G3D, [V4], [ivec(3)], true),
  ...tex('textureProjOffset', ['sampler2DShadow'], [V4], [ivec(2)], true),
  // textureLodOffset — no bias variant
  ...tex('textureLodOffset', G2D, [V2], [F, ivec(2)]),
  ...tex('textureLodOffset', G3D, [V3], [F, ivec(3)]),
  ...tex('textureLodOffset', G2DARRAY, [V3], [F, ivec(2)]),
  ...tex('textureLodOffset', ['sampler2DShadow'], [V3], [F, ivec(2)]),
  // textureProjLod
  ...tex('textureProjLod', G2D, [V3], [F]),
  ...tex('textureProjLod', G2D, [V4], [F]),
  ...tex('textureProjLod', G3D, [V4], [F]),
  ...tex('textureProjLod', ['sampler2DShadow'], [V4], [F]),
  // textureProjLodOffset
  ...tex('textureProjLodOffset', G2D, [V3], [F, ivec(2)]),
  ...tex('textureProjLodOffset', G2D, [V4], [F, ivec(2)]),
  ...tex('textureProjLodOffset', G3D, [V4], [F, ivec(3)]),
  ...tex('textureProjLodOffset', ['sampler2DShadow'], [V4], [F, ivec(2)]),
  // textureGrad — all three shadow kinds
  ...tex('textureGrad', G2D, [V2], [V2, V2]),
  ...tex('textureGrad', G3D, [V3], [V3, V3]),
  ...tex('textureGrad', GCUBE, [V3], [V3, V3]),
  ...tex('textureGrad', G2DARRAY, [V3], [V2, V2]),
  ...tex('textureGrad', ['sampler2DShadow'], [V3], [V2, V2]),
  ...tex('textureGrad', ['samplerCubeShadow'], [V4], [V3, V3]),
  ...tex('textureGrad', ['sampler2DArrayShadow'], [V4], [V2, V2]),
  // textureGradOffset
  ...tex('textureGradOffset', G2D, [V2], [V2, V2, ivec(2)]),
  ...tex('textureGradOffset', G3D, [V3], [V3, V3, ivec(3)]),
  ...tex('textureGradOffset', G2DARRAY, [V3], [V2, V2, ivec(2)]),
  ...tex('textureGradOffset', ['sampler2DShadow'], [V3], [V2, V2, ivec(2)]),
  ...tex('textureGradOffset', ['sampler2DArrayShadow'], [V4], [V2, V2, ivec(2)]),
  // textureProjGrad
  ...tex('textureProjGrad', G2D, [V3], [V2, V2]),
  ...tex('textureProjGrad', G2D, [V4], [V2, V2]),
  ...tex('textureProjGrad', G3D, [V4], [V3, V3]),
  ...tex('textureProjGrad', ['sampler2DShadow'], [V4], [V2, V2]),
  // textureProjGradOffset
  ...tex('textureProjGradOffset', G2D, [V3], [V2, V2, ivec(2)]),
  ...tex('textureProjGradOffset', G2D, [V4], [V2, V2, ivec(2)]),
  ...tex('textureProjGradOffset', G3D, [V4], [V3, V3, ivec(3)]),
  ...tex('textureProjGradOffset', ['sampler2DShadow'], [V4], [V2, V2, ivec(2)]),
  // textureSize — ivec2/ivec3 for every kind (incl. shadow and int/uint samplers)
  ...tex('textureSize', G2D, [I], [], false, sizeRet),
  ...tex('textureSize', G3D, [I], [], false, sizeRet),
  ...tex('textureSize', GCUBE, [I], [], false, sizeRet),
  ...tex('textureSize', G2DARRAY, [I], [], false, sizeRet),
  ...tex('textureSize', ['sampler2DShadow'], [I], [], false, sizeRet),
  ...tex('textureSize', ['samplerCubeShadow'], [I], [], false, sizeRet),
  ...tex('textureSize', ['sampler2DArrayShadow'], [I], [], false, sizeRet),
];

/* ------------------------------------------------------------------ */
/* Assembly                                                            */
/* ------------------------------------------------------------------ */

export const builtinFunctions300: BuiltinSignature[] = [
  ...common100,
  ...common300,
  ...rel300('lessThan'),
  ...rel300('lessThanEqual'),
  ...rel300('greaterThan'),
  ...rel300('greaterThanEqual'),
  ...eq300('equal'),
  ...eq300('notEqual'),
  ...integer300,
  ...texture300,
];

/**
 * GLSL ES 3.00 §7.7 built-in uniform state type (identical to ES 1.00 §7.6):
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

export const builtinVariables300: BuiltinVariable[] = [
  { name: 'gl_Position', type: V4, stage: 'VERTEX', writable: true },
  { name: 'gl_PointSize', type: F, stage: 'VERTEX', writable: true },
  { name: 'gl_FragCoord', type: V4, stage: 'FRAGMENT', writable: false },
  { name: 'gl_FrontFacing', type: B, stage: 'FRAGMENT', writable: false },
  { name: 'gl_PointCoord', type: V2, stage: 'FRAGMENT', writable: false },
  { name: 'gl_VertexID', type: I, stage: 'VERTEX', writable: false },
  { name: 'gl_InstanceID', type: I, stage: 'VERTEX', writable: false },
  {
    name: 'gl_DrawID',
    type: I,
    stage: 'VERTEX',
    writable: false,
    extension: 'GL_ANGLE_multi_draw',
  },
  {
    name: 'gl_ClipDistance',
    type: arr(F, 8),
    stage: 'BOTH',
    writable: true,
    extension: 'GL_ANGLE_clip_cull_distance',
  },
  {
    name: 'gl_CullDistance',
    type: arr(F, 8),
    stage: 'BOTH',
    writable: true,
    extension: 'GL_ANGLE_clip_cull_distance',
  },
  { name: 'gl_FragDepth', type: F, stage: 'FRAGMENT', writable: true },
  // GLSL ES 3.00 §7.7 built-in uniform state (usable in BOTH stages):
  // `uniform gl_DepthRangeParameters { float near; float far; float diff; }
  //  gl_DepthRange;`. Member reads (gl_DepthRange.near) resolve via the struct
  // type (semantics analyzeMember). NOTE: codegen has no gl_DepthRange
  // lowering yet (codegen/env.ts builtinRef + the member walker) — LINK of a
  // shader reading it is not supported; compile is.
  { name: 'gl_DepthRange', type: depthRangeParams, stage: 'BOTH', writable: false },
];

/**
 * gl_Max* values = the limits gl/ reports via getParameter on a WebGL2 context
 * (src/gl/state.ts defaultLimits — the same Limits object serves WebGL1 and
 * WebGL2, so every constant with a getParameter-backed limit equals it:
 * MAX_VERTEX_UNIFORM_VECTORS 4096, MAX_FRAGMENT_UNIFORM_VECTORS 4096,
 * MAX_VERTEX_OUTPUT_COMPONENTS/4 = 32, MAX_FRAGMENT_INPUT_COMPONENTS/4 = 32,
 * MAX_DRAW_BUFFERS 8, MAX_UNIFORM_BUFFER_BINDINGS 72, MAX_UNIFORM_BLOCK_SIZE
 * 65536, MAX_COMBINED_UNIFORM_BLOCKS 36). CTS checks these equalities
 * (conformance2/rendering/draw-buffers.html: gl_MaxDrawBuffers ==
 * getParameter(MAX_DRAW_BUFFERS)). gl_MaxImageUnits and
 * gl_MaxCombinedShaderOutputResources have NO getParameter in gl/ and stay at
 * the GLES 3.00 minimums; gl_MaxClipDistances matches the
 * WEBGL_clip_cull_distance MAX_CLIP_DISTANCES_WEBGL value (8). The two
 * extension-gated constants (gl_MaxCullDistances 8, gl_MaxCombinedClipAndCullDistances 16)
 * are visible only when GL_ANGLE_clip_cull_distance is enabled and match the
 * extension's MAX_CULL_DISTANCES_WEBGL / MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL
 * getParameter values (src/gl/state.ts).
 */
export const builtinConstants300: BuiltinConstant[] = [
  { name: 'gl_MaxVertexAttribs', value: 16 },
  { name: 'gl_MaxVertexUniformVectors', value: 4096 },
  { name: 'gl_MaxVertexOutputVectors', value: 32 },
  { name: 'gl_MaxVertexTextureImageUnits', value: 16 },
  { name: 'gl_MaxFragmentUniformVectors', value: 4096 },
  { name: 'gl_MaxFragmentInputVectors', value: 32 },
  { name: 'gl_MaxTextureImageUnits', value: 16 },
  { name: 'gl_MaxCombinedTextureImageUnits', value: 32 },
  { name: 'gl_MaxDrawBuffers', value: 8 },
  { name: 'gl_MaxClipDistances', value: 8 },
  { name: 'gl_MaxCullDistances', value: 8, extension: 'GL_ANGLE_clip_cull_distance' },
  { name: 'gl_MaxCombinedClipAndCullDistances', value: 16, extension: 'GL_ANGLE_clip_cull_distance' },
  { name: 'gl_MaxTransformFeedbackSeparateAttribs', value: 4 },
  { name: 'gl_MaxTransformFeedbackInterleavedComponents', value: 64 },
  { name: 'gl_MaxTransformFeedbackSeparateComponents', value: 4 },
  { name: 'gl_MaxUniformBufferBindings', value: 72 },
  { name: 'gl_MaxUniformBlockSize', value: 65536 },
  { name: 'gl_MaxVertexUniformBlocks', value: 12 },
  { name: 'gl_MaxFragmentUniformBlocks', value: 12 },
  { name: 'gl_MaxCombinedUniformBlocks', value: 36 },
  { name: 'gl_MaxImageUnits', value: 4 },
  { name: 'gl_MaxCombinedShaderOutputResources', value: 4 },
];
