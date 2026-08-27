/**
 * selftest-builtins.ts — sanity checks for the GLSL ES builtin tables.
 *
 * Run: npx tsx src/glsl/builtins/selftest-builtins.ts
 *
 * Verifies: structural validity of every entry (no void params, no unsized
 * arrays, concrete variable types, gl_Max* > 0), exact spot-check counts,
 * extension tagging, absence of duplicates, and that the 3.00 table is a
 * superset of the 1.00 core math table (the ES 1.00 texture2D* family is the
 * only deliberate exception). Prints "OK" and exits 0 on success.
 */
import {
  builtinConstants100,
  builtinConstants300,
  builtinFunctions100,
  builtinFunctions300,
  builtinVariables100,
  builtinVariables300,
  extensionConstants,
  extensionFunctions,
  extensionVariables,
  matches,
} from './index.js';
import type { BuiltinConstant, BuiltinSignature, BuiltinVariable } from './types.js';
import type { GLSLType, SamplerKind } from '../types.js';
import { compileShader } from '../compiler.js';

let failures = 0;
function check(cond: boolean, msg: string): void {
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

const SAMPLER_KINDS: ReadonlySet<string> = new Set<SamplerKind>([
  'sampler2D', 'samplerCube', 'sampler3D', 'sampler2DArray',
  'sampler2DShadow', 'samplerCubeShadow', 'sampler2DArrayShadow',
  'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray',
  'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray',
]);

/** Structural canonical form of a type (used for equality + keys). */
function canon(t: GLSLType): string {
  switch (t.kind) {
    case 'void': return 'void';
    case 'scalar': return t.base;
    case 'vector': return (t.base === 'float' ? 'vec' : t.base === 'int' ? 'ivec' : t.base === 'uint' ? 'uvec' : 'bvec') + t.size;
    case 'matrix': return 'mat' + t.cols + 'x' + t.rows;
    case 'sampler': return t.sampler;
    case 'array': return canon(t.element) + '[' + (t.size ?? '?') + ']';
    case 'struct': return 'struct:' + t.name;
  }
}

/**
 * A type is structurally valid; `allowVoid` permits void (return types only).
 * `allowStruct` permits struct types — legal for builtin VARIABLES
 * (gl_DepthRange is a struct per GLSL ES 1.00 §7.6 / 3.00 §7.7) but never for
 * function signature params/returns, so the signature path keeps
 * allowStruct=false.
 */
function typeValid(t: GLSLType, allowVoid: boolean, where: string, allowStruct: boolean = false): boolean {
  switch (t.kind) {
    case 'void':
      return allowVoid;
    case 'scalar':
      return t.base === 'float' || t.base === 'int' || t.base === 'uint' || t.base === 'bool';
    case 'vector':
      return (t.base === 'float' || t.base === 'int' || t.base === 'uint' || t.base === 'bool') &&
        (t.size === 2 || t.size === 3 || t.size === 4);
    case 'matrix':
      return (t.cols === 2 || t.cols === 3 || t.cols === 4) && (t.rows === 2 || t.rows === 3 || t.rows === 4);
    case 'sampler':
      return SAMPLER_KINDS.has(t.sampler);
    case 'struct':
      if (allowStruct) {
        if (t.members.length === 0) {
          check(false, `${where}: struct ${t.name} has no members`);
          return false;
        }
        let ok = true;
        for (const m of t.members) {
          if (m.name.length === 0) {
            check(false, `${where}: struct ${t.name} has an unnamed member`);
            ok = false;
          }
          if (!typeValid(m.type, false, `${where}: member ${t.name}.${m.name}`, true)) ok = false;
        }
        return ok;
      }
      check(false, `${where}: builtin types must never be structs`);
      return false;
    case 'array':
      if (t.size === null || t.size <= 0) {
        check(false, `${where}: unsized/empty array`);
        return false;
      }
      return typeValid(t.element, false, where, allowStruct);
  }
}

const sigKey = (s: BuiltinSignature): string => s.name + '(' + s.params.map(canon).join(',') + ')';

function validateSignatures(table: BuiltinSignature[], label: string): void {
  const keys = new Set<string>();
  for (const s of table) {
    check(s.name.length > 0, `${label}: empty function name`);
    check(s.params.length <= 8, `${label}: ${sigKey(s)} has >8 params`);
    for (const p of s.params) typeValid(p, false, `${label}: param of ${sigKey(s)}`);
    typeValid(s.ret, true, `${label}: return of ${sigKey(s)}`);
    if (s.stage !== undefined) check(s.stage === 'VERTEX' || s.stage === 'FRAGMENT', `${label}: ${sigKey(s)} bad stage`);
    const k = sigKey(s);
    check(!keys.has(k), `${label}: duplicate signature ${k}`);
    keys.add(k);
  }
}

function validateVariables(table: BuiltinVariable[], label: string): void {
  const names = new Set<string>();
  for (const v of table) {
    check(v.name.startsWith('gl_'), `${label}: ${v.name} not a gl_ builtin`);
    typeValid(v.type, false, `${label}: type of ${v.name}`, true);
    check(v.stage === 'VERTEX' || v.stage === 'FRAGMENT' || v.stage === 'BOTH', `${label}: ${v.name} bad stage`);
    check(typeof v.writable === 'boolean', `${label}: ${v.name} missing writable`);
    check(!names.has(v.name), `${label}: duplicate variable ${v.name}`);
    names.add(v.name);
  }
}

function validateConstants(table: BuiltinConstant[], label: string): void {
  const names = new Set<string>();
  for (const c of table) {
    check(c.name.startsWith('gl_'), `${label}: ${c.name} not a gl_ builtin`);
    check(c.value > 0, `${label}: ${c.name} value ${c.value} must be > 0`);
    check(!names.has(c.name), `${label}: duplicate constant ${c.name}`);
    names.add(c.name);
  }
}

/**
 * gl_DepthRange must be the built-in uniform-state struct (GLSL ES 1.00 §7.6 /
 * 3.00 §7.7): `uniform gl_DepthRangeParameters { float near; float far;
 * float diff; } gl_DepthRange;`, usable in both stages, read-only.
 */
function checkDepthRangeVariable(table: BuiltinVariable[], label: string): void {
  const v = table.find((vv) => vv.name === 'gl_DepthRange');
  check(v !== undefined, `${label}: missing variable gl_DepthRange`);
  if (v === undefined) return;
  check(v.type.kind === 'struct', `${label}: gl_DepthRange must be a struct`);
  if (v.type.kind !== 'struct') return;
  check(v.type.name === 'gl_DepthRangeParameters', `${label}: gl_DepthRange struct must be gl_DepthRangeParameters`);
  const memberNames = v.type.members.map((m) => m.name);
  check(
    memberNames.length === 3 && memberNames[0] === 'near' && memberNames[1] === 'far' && memberNames[2] === 'diff',
    `${label}: gl_DepthRangeParameters must have exactly 3 members near/far/diff`,
  );
  check(v.type.members.every((m) => canon(m.type) === 'float'), `${label}: gl_DepthRangeParameters members must be float`);
  check(v.stage === 'BOTH', `${label}: gl_DepthRange must be BOTH-stage`);
  check(v.writable === false, `${label}: gl_DepthRange must be read-only`);
}

/* ------------------------------------------------------------------ */
/* 1. Structural validity + counts                                     */
/* ------------------------------------------------------------------ */

validateSignatures(builtinFunctions100, '100');
validateSignatures(builtinFunctions300, '300');
validateSignatures(extensionFunctions, 'ext');
validateVariables(builtinVariables100, '100');
validateVariables(builtinVariables300, '300');
validateVariables(extensionVariables, 'ext');
validateConstants(builtinConstants100, '100');
validateConstants(builtinConstants300, '300');
validateConstants(extensionConstants, 'ext');

/* ------------------------------------------------------------------ */
/* 2. Core tables must be pure (no extension tags)                     */
/* ------------------------------------------------------------------ */

for (const s of builtinFunctions100) check(s.extension === undefined, `100: core signature ${sigKey(s)} has extension tag`);
for (const s of builtinFunctions300) check(s.extension === undefined, `300: core signature ${sigKey(s)} has extension tag`);
for (const v of [...builtinVariables100, ...builtinVariables300]) {
  // gl_DrawID is the SOLE extension-tagged core-table entry — it must be
  // extension-gated in ES 3.00 too (GL_ANGLE_multi_draw), unlike the other
  // core variables (checked below).
  if (v.name === 'gl_DrawID') continue;
  check(v.extension === undefined, `core variable ${v.name} has extension tag`);
}
for (const c of [...builtinConstants100, ...builtinConstants300]) check(c.extension === undefined, `core constant ${c.name} has extension tag`);

/* Every extension-gated entry carries an `extension` tag. */
for (const s of extensionFunctions) check(s.extension !== undefined, `ext: ${sigKey(s)} missing extension tag`);
for (const v of extensionVariables) check(v.extension !== undefined, `ext: ${v.name} missing extension tag`);
for (const c of extensionConstants) check(c.extension !== undefined, `ext: ${c.name} missing extension tag`);

/* ------------------------------------------------------------------ */
/* 3. Exact spot-check counts (ES 1.00)                                */
/* ------------------------------------------------------------------ */

check(matches('sin', builtinFunctions100).length === 4, 'sin must have exactly 4 signatures');
check(matches('atan', builtinFunctions100).length === 8, 'atan must have exactly 8 signatures (atan(x) + atan(y,x))');
check(matches('texture2D', builtinFunctions100).length === 2, 'texture2D must have exactly 2 signatures (bias variant)');
check(matches('texture2DProj', builtinFunctions100).length === 4, 'texture2DProj must have exactly 4 signatures (vec3/vec4 × bias)');
check(matches('textureCube', builtinFunctions100).length === 2, 'textureCube must have exactly 2 signatures (bias variant)');
check(matches('texture2DLod', builtinFunctions100).length === 1, 'texture2DLod must have exactly 1 signature');
check(matches('texture2DLod', builtinFunctions100)[0]?.stage === 'VERTEX', 'texture2DLod must be vertex-stage in core 1.00');
check(matches('cross', builtinFunctions100).length === 1, 'cross must have exactly 1 signature (vec3)');
check(matches('length', builtinFunctions100).length === 4, 'length must have exactly 4 signatures');
check(matches('matrixCompMult', builtinFunctions100).length === 3, 'matrixCompMult must have exactly 3 signatures');
check(matches('outerProduct', builtinFunctions100).length === 3, 'outerProduct must have exactly 3 signatures');
check(matches('mix', builtinFunctions100).length === 7, 'mix must have exactly 7 signatures (genType×3 + genType,genType,float; float,float,float is shared)');
check(matches('min', builtinFunctions100).length === 7, 'min must have exactly 7 signatures (genType + float scalar; float,float is shared)');
check(matches('max', builtinFunctions100).length === 7, 'max must have exactly 7 signatures');
check(matches('clamp', builtinFunctions100).length === 7, 'clamp must have exactly 7 signatures');
check(matches('step', builtinFunctions100).length === 7, 'step must have exactly 7 signatures');
check(matches('smoothstep', builtinFunctions100).length === 7, 'smoothstep must have exactly 7 signatures');
check(matches('mod', builtinFunctions100).length === 7, 'mod must have exactly 7 signatures (4 (x,y) + 3 (x,float); float,float is shared)');
check(matches('mod', builtinFunctions100).some((s) => canon(s.params[0]) === 'float' && canon(s.params[1]) === 'float'), 'mod must have a mod(float, float) signature');

/* ES 1.00 has NO shadow texture functions in core. */
for (const s of builtinFunctions100) check(!s.name.includes('Shadow'), `100: shadow function ${sigKey(s)} must not be core`);

/* gl_FragData: core size 1 (= gl_MaxDrawBuffers), extension size 4. */
const fd100 = builtinVariables100.find((v) => v.name === 'gl_FragData');
const fdext = extensionVariables.find((v) => v.name === 'gl_FragData');
check(fd100 !== undefined && fd100.type.kind === 'array' && fd100.type.size === 1, 'gl_FragData must be vec4[1] in core 1.00');
check(fdext !== undefined && fdext.type.kind === 'array' && fdext.type.size === 4, 'gl_FragData must be vec4[4] with GL_EXT_draw_buffers');
check(builtinConstants100.find((c) => c.name === 'gl_MaxDrawBuffers')?.value === 1, 'core gl_MaxDrawBuffers must be 1');
check(extensionConstants.find((c) => c.name === 'gl_MaxDrawBuffers')?.value === 4, 'extension gl_MaxDrawBuffers must be 4');
checkDepthRangeVariable(builtinVariables100, '100');

/* ------------------------------------------------------------------ */
/* 4. Extension-gated function checks                                  */
/* ------------------------------------------------------------------ */

const dFdx = matches('dFdx', extensionFunctions);
check(dFdx.length === 4, 'dFdx must have exactly 4 signatures');
check(dFdx.every((s) => s.extension === 'GL_OES_standard_derivatives' && s.stage === 'FRAGMENT'), 'dFdx must be GL_OES_standard_derivatives fragment-stage');
const texLodExt = matches('texture2DLodEXT', extensionFunctions);
check(texLodExt.length === 1 && texLodExt[0]?.stage === 'FRAGMENT' && texLodExt[0]?.extension === 'GL_EXT_shader_texture_lod', 'texture2DLodEXT must be fragment-stage GL_EXT_shader_texture_lod');
check(matches('texture2DProjLodEXT', extensionFunctions).length === 2, 'texture2DProjLodEXT must have vec3 + vec4 variants');
check(matches('texture2DProjGradEXT', extensionFunctions).length === 2, 'texture2DProjGradEXT must have vec3 + vec4 variants');
check(matches('texture2DGradEXT', extensionFunctions).length === 1, 'texture2DGradEXT must have 1 signature');
check(matches('textureCubeGradEXT', extensionFunctions).length === 1, 'textureCubeGradEXT must have 1 signature');
check(matches('textureCubeLodEXT', extensionFunctions).length === 1, 'textureCubeLodEXT must have 1 signature');
check(matches('texture2DShadowLodEXT', extensionFunctions).length === 1, 'texture2DShadowLodEXT must have 1 signature');
check(matches('fwidth', extensionFunctions).length === 4, 'fwidth must have exactly 4 signatures');
check(extensionVariables.some((v) => v.name === 'gl_FragDepthEXT' && v.extension === 'GL_EXT_frag_depth' && v.writable), 'gl_FragDepthEXT must be writable GL_EXT_frag_depth');
/* gl_DrawID (GL_ANGLE_multi_draw / WEBGL_multi_draw): read-only VERTEX int,
 * extension-gated in BOTH ES 1.00 (extensionVariables) and ES 3.00 (the
 * sole extension-tagged core-table entry). */
check(extensionVariables.some((v) => v.name === 'gl_DrawID' && v.extension === 'GL_ANGLE_multi_draw' && v.stage === 'VERTEX' && !v.writable), 'ext: gl_DrawID must be read-only VERTEX GL_ANGLE_multi_draw');
check(builtinVariables300.some((v) => v.name === 'gl_DrawID' && v.extension === 'GL_ANGLE_multi_draw' && v.stage === 'VERTEX' && !v.writable), '300: gl_DrawID must be read-only VERTEX GL_ANGLE_multi_draw');

/* ------------------------------------------------------------------ */
/* 5. ES 3.00 checks                                                   */
/* ------------------------------------------------------------------ */

// 3.00 is a superset of 1.00 core EXCEPT the replaced texture2D* family.
const REPLACED = new Set(['texture2D', 'texture2DProj', 'texture2DLod', 'texture2DProjLod', 'textureCube', 'textureCubeLod']);
const keys300 = new Set(builtinFunctions300.map(sigKey));
for (const s of builtinFunctions100) {
  if (REPLACED.has(s.name)) continue;
  check(keys300.has(sigKey(s)), `300: missing 1.00 core function ${sigKey(s)}`);
}
for (const s of builtinFunctions300) check(!REPLACED.has(s.name), `300: replaced 1.00 function ${s.name} must not exist`);

// 3.00 additions present.
check(matches('texture', builtinFunctions300).length === 29, 'texture must have 29 signatures in 3.00');
check(matches('textureLod', builtinFunctions300).length === 13, 'textureLod must have 13 signatures in 3.00');
check(matches('textureProj', builtinFunctions300).length === 20, 'textureProj must have 20 signatures in 3.00');
check(matches('texelFetch', builtinFunctions300).length === 9, 'texelFetch must have 9 signatures in 3.00');
check(matches('texelFetchOffset', builtinFunctions300).length === 9, 'texelFetchOffset must have 9 signatures in 3.00');
check(matches('textureSize', builtinFunctions300).length === 15, 'textureSize must have 15 signatures in 3.00');
check(matches('textureGrad', builtinFunctions300).length === 15, 'textureGrad must have 15 signatures in 3.00');
check(matches('textureOffset', builtinFunctions300).length === 20, 'textureOffset must have 20 signatures in 3.00');
check(matches('textureGather', builtinFunctions300).length === 0, 'textureGather must NOT exist in 3.00');
check(matches('textureQueryLod', builtinFunctions300).length === 0, 'textureQueryLod must NOT exist in 3.00');
check(matches('fma', builtinFunctions300).length === 0, 'fma must NOT exist in 3.00');

// Shadow texture functions return float in 3.00; non-shadow return vec4/ivec4/uvec4.
const tex2DShadow = builtinFunctions300.find((s) => s.name === 'texture' && canon(s.params[0]) === 'sampler2DShadow' && s.params.length === 2);
check(tex2DShadow !== undefined && canon(tex2DShadow.ret) === 'float', 'texture(sampler2DShadow, vec3) must return float');
const tex2D = builtinFunctions300.find((s) => s.name === 'texture' && canon(s.params[0]) === 'sampler2D' && s.params.length === 2);
check(tex2D !== undefined && canon(tex2D.ret) === 'vec4', 'texture(sampler2D, vec2) must return vec4');
const texI2D = builtinFunctions300.find((s) => s.name === 'texture' && canon(s.params[0]) === 'isampler2D' && s.params.length === 2);
check(texI2D !== undefined && canon(texI2D.ret) === 'ivec4', 'texture(isampler2D, vec2) must return ivec4');
const size2DArrShadow = builtinFunctions300.find((s) => s.name === 'textureSize' && canon(s.params[0]) === 'sampler2DArrayShadow');
check(size2DArrShadow !== undefined && canon(size2DArrShadow.ret) === 'ivec3', 'textureSize(sampler2DArrayShadow) must return ivec3');
const size3D = builtinFunctions300.find((s) => s.name === 'textureSize' && canon(s.params[0]) === 'isampler3D');
check(size3D !== undefined && canon(size3D.ret) === 'ivec3', 'textureSize(isampler3D) must return ivec3');
// texture(sampler2DArrayShadow, vec4) has NO bias variant.
const tex2DArrShadowBias = builtinFunctions300.find((s) => s.name === 'texture' && canon(s.params[0]) === 'sampler2DArrayShadow' && s.params.length === 3);
check(tex2DArrShadowBias === undefined, 'texture(sampler2DArrayShadow) must have no bias variant');
// textureLod has sampler2DShadow but no cube-shadow / array-shadow.
check(matches('textureLod', builtinFunctions300).some((s) => canon(s.params[0]) === 'sampler2DShadow'), 'textureLod must have sampler2DShadow variant');
check(!matches('textureLod', builtinFunctions300).some((s) => canon(s.params[0]) === 'samplerCubeShadow'), 'textureLod must NOT have samplerCubeShadow variant');

// Integer/common function spot checks.
check(matches('abs', builtinFunctions300).length === 8, 'abs must have 8 signatures in 3.00 (genType + genIType)');
check(matches('min', builtinFunctions300).length === 21, 'min must have 21 signatures in 3.00 (3 bases × 7 unique)');
check(matches('max', builtinFunctions300).length === 21, 'max must have 21 signatures in 3.00');
check(matches('clamp', builtinFunctions300).length === 21, 'clamp must have 21 signatures in 3.00');
check(matches('mix', builtinFunctions300).length === 11, 'mix must have 11 signatures in 3.00 (7 from 1.00 + 4 genBType selector)');
check(matches('isnan', builtinFunctions300).length === 4, 'isnan must have 4 signatures (genType only)');
check(matches('trunc', builtinFunctions300).length === 4, 'trunc must have 4 signatures (genType only)');
check(matches('modf', builtinFunctions300).length === 4, 'modf must have 4 signatures');
check(matches('bitfieldExtract', builtinFunctions300).length === 8, 'bitfieldExtract must have 8 signatures');
check(matches('bitCount', builtinFunctions300).length === 8, 'bitCount must have 8 signatures');
check(matches('uaddCarry', builtinFunctions300).length === 4, 'uaddCarry must have 4 signatures');
check(matches('umulExtended', builtinFunctions300).length === 4, 'umulExtended must have 4 signatures');
check(matches('imulExtended', builtinFunctions300).every((s) => canon(s.ret) === 'void'), 'imulExtended must return void');
check(matches('packUnorm4x8', builtinFunctions300).length === 1, 'packUnorm4x8 must have 1 signature');
check(matches('unpackSnorm4x8', builtinFunctions300).length === 1, 'unpackSnorm4x8 must have 1 signature');
check(matches('lessThan', builtinFunctions300).length === 9, 'lessThan must have 9 signatures in 3.00 (vec/ivec/uvec)');
check(matches('equal', builtinFunctions300).length === 12, 'equal must have 12 signatures in 3.00 (vec/ivec/uvec/bvec)');
const bitCountU = builtinFunctions300.find((s) => s.name === 'bitCount' && canon(s.params[0]) === 'uvec2');
check(bitCountU !== undefined && canon(bitCountU.ret) === 'ivec2', 'bitCount(uvec2) must return ivec2');

// 3.00 variables.
for (const name of ['gl_Position', 'gl_PointSize', 'gl_FragCoord', 'gl_FrontFacing', 'gl_PointCoord', 'gl_VertexID', 'gl_InstanceID', 'gl_FragDepth']) {
  check(builtinVariables300.some((v) => v.name === name), `300: missing variable ${name}`);
}
check(!builtinVariables300.some((v) => v.name === 'gl_FragColor'), '300: gl_FragColor must not exist');
check(!builtinVariables300.some((v) => v.name === 'gl_FragData'), '300: gl_FragData must not exist');
check(builtinVariables300.find((v) => v.name === 'gl_FragDepth')?.writable === true, 'gl_FragDepth must be writable');
check(builtinVariables300.find((v) => v.name === 'gl_VertexID')?.writable === false, 'gl_VertexID must be read-only');
checkDepthRangeVariable(builtinVariables300, '300');

// 3.00 constants: exact WebGL minimums.
const EXPECTED_300: Record<string, number> = {
  gl_MaxVertexAttribs: 16, gl_MaxVertexUniformVectors: 256, gl_MaxVertexOutputVectors: 16,
  gl_MaxVertexTextureImageUnits: 16, gl_MaxFragmentUniformVectors: 224, gl_MaxFragmentInputVectors: 15,
  gl_MaxTextureImageUnits: 16, gl_MaxCombinedTextureImageUnits: 32, gl_MaxDrawBuffers: 4,
  gl_MaxClipDistances: 8, gl_MaxTransformFeedbackSeparateAttribs: 4,
  gl_MaxTransformFeedbackInterleavedComponents: 64, gl_MaxTransformFeedbackSeparateComponents: 4,
  gl_MaxUniformBufferBindings: 24, gl_MaxUniformBlockSize: 16384, gl_MaxVertexUniformBlocks: 12,
  gl_MaxFragmentUniformBlocks: 12, gl_MaxCombinedUniformBlocks: 24, gl_MaxImageUnits: 4,
  gl_MaxCombinedShaderOutputResources: 4,
};
check(builtinConstants300.length === Object.keys(EXPECTED_300).length, `300: expected ${Object.keys(EXPECTED_300).length} constants, got ${builtinConstants300.length}`);
for (const [name, value] of Object.entries(EXPECTED_300)) {
  const c = builtinConstants300.find((cc) => cc.name === name);
  check(c !== undefined && c.value === value, `300: ${name} must be ${value}`);
}

const EXPECTED_100: Record<string, number> = {
  gl_MaxVertexAttribs: 16, gl_MaxVertexUniformVectors: 128, gl_MaxVaryingVectors: 8,
  gl_MaxVertexTextureImageUnits: 16, gl_MaxCombinedTextureImageUnits: 8,
  gl_MaxTextureImageUnits: 8, gl_MaxFragmentUniformVectors: 16, gl_MaxDrawBuffers: 1,
};
check(builtinConstants100.length === Object.keys(EXPECTED_100).length, `100: expected ${Object.keys(EXPECTED_100).length} constants, got ${builtinConstants100.length}`);
for (const [name, value] of Object.entries(EXPECTED_100)) {
  const c = builtinConstants100.find((cc) => cc.name === name);
  check(c !== undefined && c.value === value, `100: ${name} must be ${value}`);
}

/* ------------------------------------------------------------------ */
/* 6. gl_DepthRange member reads compile in both versions              */
/* ------------------------------------------------------------------ */

const DEPTH_RANGE_SRC_BODY = [
  'precision highp float;',
  'void main() {',
  '  float n = gl_DepthRange.near;',
  '  float f = gl_DepthRange.far;',
  '  float d = gl_DepthRange.diff;',
  '  gl_Position = vec4(n + f + d, 0.0, 0.0, 1.0);',
  '}',
].join('\n');

const dr100 = compileShader(DEPTH_RANGE_SRC_BODY, { type: 'VERTEX', version: 100 });
check(dr100.ok, '100: shader reading gl_DepthRange.near/far/diff must compile');
if (!dr100.ok) console.error('  100 errors: ' + JSON.stringify(dr100.errors));

const dr300 = compileShader('#version 300 es\n' + DEPTH_RANGE_SRC_BODY, { type: 'VERTEX', version: 300 });
check(dr300.ok, '300: shader reading gl_DepthRange.near/far/diff must compile');
if (!dr300.ok) console.error('  300 errors: ' + JSON.stringify(dr300.errors));

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(
  `builtins selftest: 100 functions=${builtinFunctions100.length} variables=${builtinVariables100.length} constants=${builtinConstants100.length}; ` +
  `ext functions=${extensionFunctions.length} variables=${extensionVariables.length} constants=${extensionConstants.length}; ` +
  `300 functions=${builtinFunctions300.length} variables=${builtinVariables300.length} constants=${builtinConstants300.length}`,
);

if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
