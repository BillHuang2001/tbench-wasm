/**
 * selftest-semantics.ts — comprehensive battery for the FULL semantics stage:
 * declaration/qualifier/stage/precision rules, ShaderInfo assembly, ShaderUses
 * and the `analyze()` entry, driven end-to-end through `compileShader` (the
 * public contract) plus direct `analyze` where the AST matters.
 *
 * Run: npx tsx src/glsl/selftest-semantics.ts   (prints "OK", exit 0)
 *
 * Coverage (asserting EXACT ShaderInfo contents, error lines and uses flags):
 *  - ShaderInfo: attributes (location / arrays / element type), varyings
 *    (order, flat/centroid/noperspective/invariant), uniforms (plain / struct
 *    / array-of-struct full type, precision, sampler binding), uniform blocks
 *    (name/instance/arraySize/binding/members), fragment outputs (gl_FragColor,
 *    gl_FragData[N]+EXT, 3.00 out with/without layout(location=));
 *  - ShaderUses: pointSize (write), fragCoord/frontFacing/pointCoord (reads),
 *    fragDepth (3.00 gl_FragDepth / 1.00 gl_FragDepthEXT), vertexId/instanceId,
 *    derivatives via dFdx/dFdy/fwidth and via every implicit-LOD texture
 *    function name per version — NOT via textureLod/texelFetch/textureSize;
 *  - precision rules: fragment float declarations need a default (exact
 *    declaration-line errors; `precision` statements in order; vertex and
 *    int/sampler leniency; params and return types);
 *  - stage/qualifier rules: attribute in fragment, 1.00 integral
 *    attributes/varyings, 3.00 integral varyings must be flat, 3.00 fragment
 *    out types, discard in vertex;
 *  - extension gating (dFdx, texture2DLodEXT, EXT_draw_buffers/EXT_frag_depth
 *    paths) incl. the #extension-in-source + opts.extensions contract;
 *  - version rules (#version 300 es in a 100 context, uint in 100,
 *    attribute/varying keywords in 300);
 *  - standalone `invariant <name>;` rules (GLSL ES 1.00 §4.6.1: must FOLLOW
 *    the variable's declaration; only the invariant-capable builtins
 *    gl_Position/gl_PointSize/gl_FragCoord/gl_PointCoord/gl_FragColor/
 *    gl_FragData may be named — gl_FrontFacing and unknown names are errors);
 *  - compileShader result shape (error lines, infoLog '', extensions set,
 *    version/type) and the direct analyze() entry.
 */
import { compileShader } from './compiler.js';
import { analyze } from './semantics.js';
import { preprocess } from './preprocessor.js';
import { tokenize } from './lexer.js';
import { parse } from './parser.js';
import { typeEquals, typeName } from './types.js';
import type { GLSLType } from './types.js';
import type { CompileError } from './compiler.js';
import type { ShaderInfo } from './compiler.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function comp(
  src: string,
  version: 100 | 300,
  type: 'VERTEX' | 'FRAGMENT',
  exts?: string[],
) {
  return compileShader(src, { type, version, extensions: exts !== undefined ? new Set(exts) : undefined });
}

/** Assert success and return the ShaderInfo. */
function okInfo(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT', exts?: string[]): ShaderInfo {
  const r = comp(src, version, type, exts);
  if (!r.ok) {
    check(false, `expected success: ${JSON.stringify(src.slice(0, 60))} — got ${JSON.stringify(r.errors)}`);
    return null as unknown as ShaderInfo;
  }
  return r.shader.info;
}

/** Assert failure and return the errors. */
function errs(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT', exts?: string[]): CompileError[] {
  const r = comp(src, version, type, exts);
  if (r.ok) {
    check(false, `expected failure: ${JSON.stringify(src.slice(0, 60))}`);
    return [];
  }
  return r.errors;
}

/** Assert that an error with the exact message exists at the exact line. */
function hasErr(errors: CompileError[], line: number, msg: string): boolean {
  return errors.some((e) => e.line === line && e.message === msg);
}

/* Expected GLSLType builders. */
const F = { kind: 'scalar', base: 'float' } as const;
const I = { kind: 'scalar', base: 'int' } as const;
const V2 = { kind: 'vector', base: 'float', size: 2 } as const;
const V3 = { kind: 'vector', base: 'float', size: 3 } as const;
const V4 = { kind: 'vector', base: 'float', size: 4 } as const;
const IV3 = { kind: 'vector', base: 'int', size: 3 } as const;
const IV4 = { kind: 'vector', base: 'int', size: 4 } as const;
const UV4 = { kind: 'vector', base: 'uint', size: 4 } as const;
const M4 = { kind: 'matrix', cols: 4, rows: 4 } as const;

function checkType(t: GLSLType, expected: GLSLType, label: string): void {
  check(typeEquals(t, expected), `${label}: expected ${typeName(expected)}, got ${typeName(t)}`);
}

/* ------------------------------------------------------------------ */
/* 1. ShaderInfo — attributes                                          */
/* ------------------------------------------------------------------ */

{
  // 1.00 attributes must be scalar: `attribute vec2 q;` (attribute ARRAYS are
  // an error in ESSL 1.00 — GLSL ES 1.00 Appendix A §5, CTS
  // shader-with-attrib-array.vert.html + ogles build attribute2_vert; version
  // 300 is unaffected, see info2 below).
  const info = okInfo('attribute vec4 p; attribute vec2 q; uniform mat4 m; void main() { gl_Position = p; }', 100, 'VERTEX');
  check(info.attributes.length === 2, 'two attributes in declaration order');
  check(info.attributes[0].name === 'p' && info.attributes[0].arraySize === 1 && info.attributes[0].location === null, 'attr p: name/arraySize/location null');
  checkType(info.attributes[0].type, V4, 'attr p element type');
  check(info.attributes[1].name === 'q' && info.attributes[1].arraySize === 1 && info.attributes[1].location === null, 'attr q: bare name, arraySize 1');
  checkType(info.attributes[1].type, V2, 'attr q element type');
  check(info.uniforms.length === 1 && info.uniforms[0].name === 'm', 'uniform m collected');
  checkType(info.uniforms[0].type, M4, 'uniform m full type');
  check(info.uniforms[0].precision === 'highp', 'vertex float uniform defaults highp');
  check(info.uniforms[0].binding === null, 'uniform m binding null');
  check(info.varyings.length === 0 && info.outputs.length === 0 && info.uniformBlocks.length === 0, 'vertex: no varyings/outputs/blocks');

  const info2 = okInfo('#version 300 es\nlayout(location=2) in vec4 p;\nin vec3 n[2];\nvoid main() { gl_Position = p; }', 300, 'VERTEX');
  check(info2.attributes.length === 2, '3.00: two in-attributes');
  check(info2.attributes[0].name === 'p' && info2.attributes[0].location === 2, 'layout(location=2) recorded');
  check(info2.attributes[1].name === 'n' && info2.attributes[1].arraySize === 2 && info2.attributes[1].location === null, '3.00 array attribute: element+size, location null');
  checkType(info2.attributes[1].type, V3, '3.00 array attribute element type');

  okInfo('#version 300 es\nin ivec3 v;\nvoid main() {}', 300, 'VERTEX'); // integral vertex input OK in 3.00
  const eArr = errs('attribute vec2 q[3]; void main() { gl_Position = q[0].xyxy; }', 100, 'VERTEX');
  check(hasErr(eArr, 1, "'q' : attribute variables cannot be arrays in GLSL ES 1.00"), '1.00 attribute array → error');
  const e1 = errs('attribute vec4 p; void main() { gl_FragColor = p; }', 100, 'FRAGMENT');
  check(hasErr(e1, 1, "'attribute' : only valid in vertex shaders"), 'attribute in fragment → stage error line 1');

  const e2 = errs('attribute int i; void main() { gl_Position = vec4(float(i)); }', 100, 'VERTEX');
  check(hasErr(e2, 1, "'i' : attribute variables must have a float type in GLSL ES 1.00"), '1.00 int attribute → error');

  const e3 = errs('#version 300 es\nin bool b;\nvoid main() {}', 300, 'VERTEX');
  check(hasErr(e3, 2, "'b' : attribute variables cannot have a boolean type"), '3.00 bool attribute → error');
}

/* ------------------------------------------------------------------ */
/* 2. ShaderInfo — varyings (vertex outs / fragment ins)               */
/* ------------------------------------------------------------------ */

{
  const info = okInfo(
    'attribute vec4 p;\ninvariant varying vec2 uv;\nvarying vec3 v;\nvarying float w;\nvoid main() { gl_Position = p; uv = vec2(1.0); v = vec3(1.0); w = 1.0; }',
    100,
    'VERTEX',
  );
  check(info.varyings.length === 3, 'three varyings in declaration order');
  check(info.varyings[0].name === 'uv' && info.varyings[0].invariant === true, 'invariant varying recorded');
  check(info.varyings[1].name === 'v' && info.varyings[1].flat === false && info.varyings[1].centroid === false && info.varyings[1].noperspective === false, 'plain varying flags false');
  checkType(info.varyings[1].type, V3, 'varying v element type');
  check(info.varyings[2].name === 'w' && info.varyings[2].arraySize === 1, 'varying w scalar');

  const info2 = okInfo(
    '#version 300 es\ncentroid out vec2 cv;\nnoperspective out vec3 nv;\nflat out ivec3 fid;\ninvariant out vec4 iv;\nvoid main() { cv = vec2(1.0); nv = vec3(1.0); fid = ivec3(1); iv = vec4(1.0); }',
    300,
    'VERTEX',
  );
  check(info2.varyings.length === 4, '3.00: four out-varyings in order');
  check(info2.varyings[0].name === 'cv' && info2.varyings[0].centroid === true, 'centroid recorded');
  check(info2.varyings[1].name === 'nv' && info2.varyings[1].noperspective === true, 'noperspective recorded');
  check(info2.varyings[2].name === 'fid' && info2.varyings[2].flat === true, 'flat out ivec3 → flat true');
  checkType(info2.varyings[2].type, IV3, 'flat out ivec3 element type');
  check(info2.varyings[3].name === 'iv' && info2.varyings[3].invariant === true, 'invariant out recorded');

  // Fragment side: varying/in declarations are the fragment's interface inputs.
  const info3 = okInfo('precision mediump float;\nvarying vec2 uv;\nuniform sampler2D s;\nvoid main() { gl_FragColor = texture2D(s, uv); }', 100, 'FRAGMENT');
  check(info3.varyings.length === 1 && info3.varyings[0].name === 'uv', 'fragment varying recorded as input');
  checkType(info3.varyings[0].type, V2, 'fragment varying element type');
  check(info3.attributes.length === 0, 'fragment: no attributes');

  const info4 = okInfo('#version 300 es\nprecision mediump float;\nflat in ivec3 v;\nin vec2 uv;\nout vec4 c;\nvoid main() { c = vec4(v, 1.0) + vec4(uv, 0.0, 0.0); }', 300, 'FRAGMENT');
  check(info4.varyings.length === 2 && info4.varyings[0].name === 'v' && info4.varyings[0].flat === true, '3.00 fragment flat in recorded');
  check(info4.varyings[1].name === 'uv' && info4.varyings[1].flat === false, '3.00 fragment plain in recorded');

  const e1 = errs('varying int v; void main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
  check(hasErr(e1, 1, "'v' : varying variables must have a float type in GLSL ES 1.00"), '1.00 int varying → error');

  const e2 = errs('#version 300 es\nout ivec3 v;\nvoid main() { v = ivec3(1); }', 300, 'VERTEX');
  check(hasErr(e2, 2, "'v' : integral varying variables must be declared with 'flat'"), '3.00 out ivec3 without flat → error line 2');
  okInfo('#version 300 es\nflat out ivec3 v;\nvoid main() { v = ivec3(1); }', 300, 'VERTEX'); // with flat OK

  const e3 = errs('#version 300 es\nprecision mediump float;\nin ivec3 v;\nout vec4 c;\nvoid main() { c = vec4(v, 1.0); }', 300, 'FRAGMENT');
  check(hasErr(e3, 3, "'v' : integral varying variables must be declared with 'flat'"), '3.00 fragment in ivec3 without flat → error line 3');
}

/* ------------------------------------------------------------------ */
/* 3. ShaderInfo — uniforms (full types, precision, binding)           */
/* ------------------------------------------------------------------ */

{
  const info = okInfo(
    'precision mediump float;\nstruct Light { vec3 pos; float intensity; };\nuniform Light light;\nuniform Light lights[4];\nuniform sampler2D s;\nvarying vec2 uv;\nvoid main() { gl_FragColor = texture2D(s, uv) + vec4(light.intensity); }',
    100,
    'FRAGMENT',
  );
  check(info.uniforms.length === 3, 'three uniforms in order (struct, array-of-struct, sampler)');
  const light = info.uniforms[0];
  check(light.name === 'light' && light.type.kind === 'struct' && light.type.name === 'Light', 'struct uniform keeps struct type by name');
  check(light.type.kind === 'struct' && light.type.members.length === 2 && light.type.members[0].name === 'pos' && light.type.members[1].name === 'intensity', 'struct uniform members intact');
  checkType(light.type.kind === 'struct' ? light.type.members[0].type : F, V3, 'struct member pos type');
  check(light.precision === null, 'struct uniform precision null (structs have no single precision)');
  const lights = info.uniforms[1];
  check(lights.name === 'lights' && lights.type.kind === 'array' && lights.type.size === 4, 'array-of-struct uniform: array size 4');
  check(lights.type.kind === 'array' && lights.type.element.kind === 'struct' && lights.type.element.name === 'Light', 'array-of-struct uniform: element is the struct type');
  const s = info.uniforms[2];
  check(s.name === 's' && s.type.kind === 'sampler' && s.type.sampler === 'sampler2D', 'sampler uniform type');
  check(s.precision === null, 'fragment sampler uniform precision null (lenient)');
  check(s.binding === null, 'sampler without layout(binding=) → null');

  const info2 = okInfo(
    '#version 300 es\nlayout(binding=3) uniform samplerCube env;\nuniform sampler2D tex;\nvoid main() { gl_Position = vec4(1.0); }',
    300,
    'VERTEX',
  );
  check(info2.uniforms[0].name === 'env' && info2.uniforms[0].binding === 3, 'layout(binding=3) recorded');
  checkType(info2.uniforms[0].type, { kind: 'sampler', sampler: 'samplerCube' } as GLSLType, 'samplerCube type');
  check(info2.uniforms[1].name === 'tex' && info2.uniforms[1].binding === null, 'second sampler binding null');
  check(info2.uniforms[1].precision === 'highp', '3.00 vertex sampler defaults highp');
}

/* ------------------------------------------------------------------ */
/* 4. ShaderInfo — uniform blocks (3.00)                               */
/* ------------------------------------------------------------------ */

{
  const info = okInfo(
    '#version 300 es\nprecision mediump float;\nlayout(std140) uniform Lights { vec4 pos; mat4 m; vec3 colors[3]; } lights;\nuniform Lights b2[2];\nlayout(binding=5) uniform Mat { vec4 x; };\nout vec4 c;\nvoid main() { c = lights.pos; }',
    300,
    'FRAGMENT',
  );
  check(info.uniformBlocks.length === 2, 'two uniform blocks in order');
  const lights = info.uniformBlocks[0];
  check(lights.name === 'Lights' && lights.instanceName === 'lights' && lights.arraySize === 1, 'block Lights: name/instance/arraySize 1');
  check(lights.binding === null, 'block without layout(binding=) → null');
  check(lights.members.length === 3, 'block Lights has 3 members');
  check(lights.members[0].name === 'pos' && lights.members[1].name === 'm' && lights.members[2].name === 'colors', 'block member names in order');
  checkType(lights.members[0].type, V4, 'block member pos type');
  checkType(lights.members[1].type, M4, 'block member m type');
  // Member arrays keep their dims: vec3 colors[3] → array of 3 vec3 (GLSL ES
  // 3.00 §4.3.7 allows arrays of built-in types as block members).
  checkType(lights.members[2].type, { kind: 'array', element: V3, size: 3 }, 'block member colors type vec3[3]');
  check(lights.members[0].precision === 'mediump', 'block member precision from default');
  const mat = info.uniformBlocks[1];
  check(mat.name === 'Mat' && mat.instanceName === null && mat.binding === 5, 'block Mat: no instance, binding 5');
  checkType(mat.members[0].type, V4, 'block Mat member type');
  // The struct-typed default-block uniform b2 coexists with the block.
  check(info.uniforms.length === 1 && info.uniforms[0].name === 'b2', 'struct-typed default-block uniform collected separately');
  check(info.uniforms[0].type.kind === 'array' && info.uniforms[0].type.size === 2, 'b2 array size 2');

  const info2 = okInfo('#version 300 es\nprecision mediump float;\nlayout(std140) uniform Block { vec4 p; } inst[2];\nout vec4 c;\nvoid main() { c = inst[0].p; }', 300, 'FRAGMENT');
  check(info2.uniformBlocks[0].instanceName === 'inst' && info2.uniformBlocks[0].arraySize === 2, 'block array instance → arraySize 2');
}

/* ------------------------------------------------------------------ */
/* 5. ShaderInfo — fragment outputs                                    */
/* ------------------------------------------------------------------ */

{
  const info = okInfo('precision mediump float;\nvoid main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
  check(info.outputs.length === 1, 'gl_FragColor → one output');
  check(info.outputs[0].name === 'gl_FragColor' && info.outputs[0].index === null && info.outputs[0].location === null, 'gl_FragColor output shape');
  checkType(info.outputs[0].type, V4, 'gl_FragColor output type vec4');

  const info2 = okInfo('#extension GL_EXT_draw_buffers : enable\nprecision mediump float;\nvoid main() { gl_FragData[1] = vec4(1.0); }', 100, 'FRAGMENT', ['GL_EXT_draw_buffers']);
  check(info2.outputs.length === 1 && info2.outputs[0].name === 'gl_FragData[1]' && info2.outputs[0].index === 1, 'gl_FragData[1] + EXT → output index 1');

  const info3 = okInfo('#extension GL_EXT_draw_buffers : enable\nprecision mediump float;\nvoid main() { gl_FragData[0] = vec4(1.0); gl_FragData[3] = vec4(2.0); }', 100, 'FRAGMENT', ['GL_EXT_draw_buffers']);
  check(info3.outputs.length === 2 && info3.outputs[0].index === 0 && info3.outputs[1].index === 3, 'two gl_FragData outputs in write order');

  okInfo('precision mediump float;\nvoid main() { gl_FragData[0] = vec4(1.0); }', 100, 'FRAGMENT'); // core gl_FragData[0] legal

  const info4 = okInfo('#version 300 es\nprecision mediump float;\nlayout(location=3) out vec4 c;\nout vec4 d;\nvoid main() { c = vec4(1.0); d = vec4(2.0); }', 300, 'FRAGMENT');
  check(info4.outputs.length === 2, '3.00: two outputs');
  check(info4.outputs[0].name === 'c' && info4.outputs[0].location === 3, 'layout(location=3) output');
  check(info4.outputs[1].name === 'd' && info4.outputs[1].location === null, 'output without layout → location null');
  check(info4.outputs[0].index === null, '3.00 output index null');

  okInfo('#version 300 es\nprecision mediump float;\nout ivec4 i;\nout uvec4 u;\nvoid main() { i = ivec4(1); u = uvec4(1u); }', 300, 'FRAGMENT'); // ivec4/uvec4 outputs OK

  const e1 = errs('precision mediump float;\nvoid main() { gl_FragColor = vec4(1.0); gl_FragData[0] = vec4(1.0); }', 100, 'FRAGMENT');
  check(hasErr(e1, 2, "'gl_FragData' : cannot write both gl_FragColor and gl_FragData"), 'gl_FragColor + gl_FragData → error');

  const e2 = errs('precision mediump float;\nvoid main() { gl_FragData[1] = vec4(1.0); }', 100, 'FRAGMENT');
  check(hasErr(e2, 2, "'gl_FragData' : index 1 requires GL_EXT_draw_buffers"), 'gl_FragData[1] without EXT → error');

  const e3 = errs('#extension GL_EXT_draw_buffers : enable\nprecision mediump float;\nvoid main() { gl_FragData[4] = vec4(1.0); }', 100, 'FRAGMENT', ['GL_EXT_draw_buffers']);
  check(hasErr(e3, 3, "'gl_FragData' : index 4 out of range [0, 3]"), 'gl_FragData[4] with EXT → out of range');

  const e4 = errs('#version 300 es\nprecision mediump float;\nout vec3 c;\nvoid main() { c = vec3(1.0); }', 300, 'FRAGMENT');
  check(hasErr(e4, 3, "'c' : fragment shader outputs must be vec4, ivec4 or uvec4"), '3.00 out vec3 → error (strict vec4 contract)');

  const e5 = errs('#version 300 es\nprecision mediump float;\nout vec4 c[2];\nvoid main() { c[0] = vec4(1.0); }', 300, 'FRAGMENT');
  check(e5.length > 0 && e5[0].message.includes('fragment shader outputs must be vec4'), '3.00 out vec4[2] (array) → error');
}

/* ------------------------------------------------------------------ */
/* 6. ShaderUses                                                       */
/* ------------------------------------------------------------------ */

{
  const vs = okInfo('attribute vec4 p;\nvoid main() { gl_PointSize = 4.0; gl_Position = p; }', 100, 'VERTEX');
  check(vs.uses.pointSize === true, 'gl_PointSize write → pointSize');
  check(vs.uses.vertexId === false && vs.uses.instanceId === false && vs.uses.fragCoord === false, 'vertex: no fragment uses');

  const vs2 = okInfo('#version 300 es\nin vec4 p;\nvoid main() { gl_Position = p; gl_PointSize = float(gl_VertexID) + float(gl_InstanceID); }', 300, 'VERTEX');
  check(vs2.uses.vertexId === true && vs2.uses.instanceId === true, 'gl_VertexID/gl_InstanceID reads');

  // gl_DrawID (GL_ANGLE_multi_draw / WEBGL_multi_draw): extension-gated in
  // BOTH ES 1.00 and ES 3.00 (CTS webgl-multi-draw.html: vshaderIllegalDrawID
  // uses gl_DrawID with NO directive and must fail; vshaderDrawIDZero /
  // vshaderWithDrawID use `#extension GL_ANGLE_multi_draw : require`).
  const did1 = okInfo('#version 300 es\n#extension GL_ANGLE_multi_draw : require\nvoid main() { gl_Position = vec4(float(gl_DrawID)); }', 300, 'VERTEX', ['GL_ANGLE_multi_draw']);
  check(did1.uses.drawId === true && did1.uses.instanceId === false && did1.uses.vertexId === false, '3.00 gl_DrawID + extension → drawId (not instanceId/vertexId)');

  const errDid1 = errs('#version 300 es\nvoid main() { gl_Position = vec4(float(gl_DrawID)); }', 300, 'VERTEX');
  check(hasErr(errDid1, 2, "'gl_DrawID' : undeclared identifier"), '3.00 gl_DrawID without extension → undeclared');

  const did2 = okInfo('#extension GL_ANGLE_multi_draw : require\nattribute vec4 p;\nvoid main() { gl_Position = p + vec4(float(gl_DrawID)); }', 100, 'VERTEX', ['GL_ANGLE_multi_draw']);
  check(did2.uses.drawId === true && did2.uses.instanceId === false, '1.00 gl_DrawID + extension → drawId (vshaderDrawIDZero/vshaderWithDrawID)');

  const errDid2 = errs('attribute vec4 p;\nvoid main() { gl_Position = p + vec4(float(gl_DrawID)); }', 100, 'VERTEX');
  check(hasErr(errDid2, 2, "'gl_DrawID' : undeclared identifier"), '1.00 gl_DrawID without extension → undeclared (vshaderIllegalDrawID)');

  // `#extension GL_ANGLE_multi_draw : require` defines the macro to 1 (spec);
  // the #error fires (and compile fails) if the macro is missing or != 1.
  const did3 = okInfo('#extension GL_ANGLE_multi_draw : require\n#if GL_ANGLE_multi_draw != 1\n#error "GL_ANGLE_multi_draw macro != 1"\n#endif\nattribute vec4 p;\nvoid main() { gl_Position = p; }', 100, 'VERTEX', ['GL_ANGLE_multi_draw']);
  check(did3.uses.drawId === false, 'GL_ANGLE_multi_draw macro defined to 1 by #extension require');

  const fs = okInfo('#version 300 es\nprecision mediump float;\nuniform sampler2D s;\nin vec2 uv;\nout vec4 c;\nvoid main() { c = texture(s, uv) + vec4(gl_FragCoord.xy, gl_FrontFacing ? 1.0 : 0.0, gl_PointCoord.x); }', 300, 'FRAGMENT');
  check(fs.uses.fragCoord === true && fs.uses.frontFacing === true && fs.uses.pointCoord === true, 'fragCoord/frontFacing/pointCoord reads');
  check(fs.uses.derivatives === true, 'implicit texture() → derivatives');
  check(fs.uses.fragDepth === false, 'no fragDepth');

  const fs2 = okInfo('#version 300 es\nprecision mediump float;\nout vec4 c;\nvoid main() { gl_FragDepth = 0.5; c = vec4(1.0); }', 300, 'FRAGMENT');
  check(fs2.uses.fragDepth === true, '3.00 gl_FragDepth write → fragDepth');

  const fs3 = okInfo('#extension GL_EXT_frag_depth : enable\nprecision mediump float;\nvoid main() { gl_FragDepthEXT = 0.5; gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT', ['GL_EXT_frag_depth']);
  check(fs3.uses.fragDepth === true, '1.00 gl_FragDepthEXT write → fragDepth');

  const d1 = okInfo('#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nvoid main() { float d = dFdx(1.0); gl_FragColor = vec4(d); }', 100, 'FRAGMENT', ['GL_OES_standard_derivatives']);
  check(d1.uses.derivatives === true, 'dFdx → derivatives');
  const d2 = okInfo('#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nvoid main() { float d = fwidth(1.0); gl_FragColor = vec4(d); }', 100, 'FRAGMENT', ['GL_OES_standard_derivatives']);
  check(d2.uses.derivatives === true, 'fwidth → derivatives');

  // Every implicit-LOD name per version → derivatives; explicit-LOD families → not.
  const implicit100: [string, string][] = [
    ['texture2D', 'precision mediump float;\nuniform sampler2D s;\nvarying vec2 uv;\nvoid main() { gl_FragColor = texture2D(s, uv); }'],
    ['texture2DProj', 'precision mediump float;\nuniform sampler2D s;\nvarying vec4 uv;\nvoid main() { gl_FragColor = texture2DProj(s, uv); }'],
    ['textureCube', 'precision mediump float;\nuniform samplerCube sc;\nvarying vec3 r;\nvoid main() { gl_FragColor = textureCube(sc, r); }'],
  ];
  for (const [name, src] of implicit100) {
    const i = okInfo(src, 100, 'FRAGMENT');
    check(i.uses.derivatives === true, `1.00 ${name} → derivatives`);
  }
  const implicit300: [string, string][] = [
    ['texture', '#version 300 es\nprecision mediump float;\nuniform sampler2D s;\nin vec2 uv;\nout vec4 c;\nvoid main() { c = texture(s, uv); }'],
    ['textureProj', '#version 300 es\nprecision mediump float;\nuniform sampler2D s;\nin vec4 uv;\nout vec4 c;\nvoid main() { c = textureProj(s, uv); }'],
    ['textureOffset', '#version 300 es\nprecision mediump float;\nuniform sampler2D s;\nin vec2 uv;\nout vec4 c;\nvoid main() { c = textureOffset(s, uv, ivec2(1)); }'],
    ['textureProjOffset', '#version 300 es\nprecision mediump float;\nuniform sampler2D s;\nin vec4 uv;\nout vec4 c;\nvoid main() { c = textureProjOffset(s, uv, ivec2(1)); }'],
  ];
  for (const [name, src] of implicit300) {
    const i = okInfo(src, 300, 'FRAGMENT');
    check(i.uses.derivatives === true, `3.00 ${name} → derivatives`);
  }
  const lod300 = okInfo('#version 300 es\nprecision mediump float;\nuniform sampler2D s;\nin vec2 uv;\nout vec4 c;\nvoid main() { c = textureLod(s, uv, 0.0) + vec4(textureSize(s, 0).x); }', 300, 'FRAGMENT');
  check(lod300.uses.derivatives === false, 'textureLod + textureSize → NOT derivatives');
  const lod300b = okInfo('#version 300 es\nprecision mediump float;\nuniform sampler2D s;\nout vec4 c;\nvoid main() { c = texelFetch(s, ivec2(0), 0); }', 300, 'FRAGMENT');
  check(lod300b.uses.derivatives === false, 'texelFetch → NOT derivatives');
  const lod100 = okInfo('#extension GL_EXT_shader_texture_lod : enable\nprecision mediump float;\nuniform sampler2D s;\nvarying vec2 uv;\nvoid main() { gl_FragColor = texture2DLodEXT(s, uv, 0.0); }', 100, 'FRAGMENT', ['GL_EXT_shader_texture_lod']);
  check(lod100.uses.derivatives === false, 'texture2DLodEXT → NOT derivatives');
}

/* ------------------------------------------------------------------ */
/* 7. Precision rules                                                  */
/* ------------------------------------------------------------------ */

{
  const e1 = errs('void main() { float f = 1.0; gl_FragColor = vec4(f); }', 100, 'FRAGMENT');
  check(hasErr(e1, 1, "'f' : No precision specified for (float)"), '1.00 fragment float without default → error at declaration line 1');

  okInfo('precision mediump float;\nvoid main() { float f = 1.0; gl_FragColor = vec4(f); }', 100, 'FRAGMENT'); // with default OK
  okInfo('void main() { float f = 1.0; gl_Position = vec4(f); }', 100, 'VERTEX'); // vertex float defaults highp

  const e2 = errs('void main() { float f = 1.0; gl_FragColor = vec4(f); }\nprecision mediump float;', 100, 'FRAGMENT');
  check(hasErr(e2, 1, "'f' : No precision specified for (float)"), 'precision AFTER use → error at the use line, not the statement');

  okInfo('precision highp float;\nvoid main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT'); // precision statement alone OK

  const e3 = errs('uniform vec4 color;\nvoid main() { gl_FragColor = color; }', 100, 'FRAGMENT');
  check(hasErr(e3, 1, "'color' : No precision specified for (float)"), 'CTS shader-without-precision: uniform vec4 without default → error line 1');

  const e4 = errs('struct S { float x; };\nvoid main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
  check(hasErr(e4, 1, "'x' : No precision specified for (float)"), 'struct member float without default → error at member line');
  okInfo('precision mediump float;\nstruct S { float x; };\nvoid main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');

  const e5 = errs('#version 300 es\nout vec4 c;\nvoid main() { float f = 1.0; c = vec4(f); }', 300, 'FRAGMENT');
  check(hasErr(e5, 2, "'c' : No precision specified for (float)"), '3.00 fragment out vec4 without default → error line 2');
  check(hasErr(e5, 3, "'f' : No precision specified for (float)"), '3.00 fragment local float without default → error line 3');

  const e6 = errs('void f(float x);\nprecision mediump float;\nvoid main() { f(1.0); gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
  check(hasErr(e6, 1, "'x' : No precision specified for (float)"), 'fragment float param before default → error at param line');
  okInfo('precision mediump float;\nfloat f() { return 1.0; }\nvoid main() { gl_FragColor = vec4(f()); }', 100, 'FRAGMENT'); // return type with default OK
  const e7 = errs('float f() { return 1.0; }\nprecision mediump float;\nvoid main() { gl_FragColor = vec4(f()); }', 100, 'FRAGMENT');
  check(hasErr(e7, 1, "'f' : No precision specified for (float)"), 'fragment float return type before default → error');

  okInfo('uniform sampler2D s;\nvoid main() { gl_FragColor = texture2D(s, vec2(0.0)); }', 100, 'FRAGMENT'); // sampler lenient
  okInfo('#version 300 es\nprecision mediump float;\nuniform int i;\nuniform ivec2 v;\nout vec4 c;\nvoid main() { c = vec4(1.0); }', 300, 'FRAGMENT'); // int uniform fine; float default before the out

  const info = okInfo('precision highp float;\nuniform vec4 color;\nvoid main() { gl_FragColor = color; }', 100, 'FRAGMENT');
  check(info.uniforms[0].precision === 'highp', 'ShaderInfo precision = declared default (highp)');
}

/* ------------------------------------------------------------------ */
/* 8. Extension gating                                                 */
/* ------------------------------------------------------------------ */

{
  const e1 = errs('precision mediump float;\nvoid main() { float d = dFdx(1.0); gl_FragColor = vec4(d); }', 100, 'FRAGMENT');
  check(hasErr(e1, 2, "'dFdx' : requires extension 'GL_OES_standard_derivatives' which is not enabled"), 'dFdx without extension → error line 2');
  okInfo('#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nvoid main() { float d = dFdx(1.0); gl_FragColor = vec4(d); }', 100, 'FRAGMENT', ['GL_OES_standard_derivatives']);

  const e2 = errs('precision mediump float;\nvoid main() { float d = dFdx(1.0); gl_FragColor = vec4(d); }', 100, 'FRAGMENT', ['GL_OES_standard_derivatives']);
  check(hasErr(e2, 2, "'dFdx' : requires extension 'GL_OES_standard_derivatives' which is not enabled"), 'opts.extensions alone (no #extension in source) → still error');
  // An unknown-extension `#extension ... : enable` in the source is NOT a
  // preprocessor error (GLSL ES §3.4 — the directive is accepted, the
  // extension stays disabled). Only a USE of its functions errors, via the
  // semantics disabled-extension gate.
  const e3 = errs('#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nvoid main() { float d = dFdx(1.0); gl_FragColor = vec4(d); }', 100, 'FRAGMENT');
  check(hasErr(e3, 3, "'dFdx' : requires extension 'GL_OES_standard_derivatives' which is not enabled"), '#extension without opts.extensions: directive OK, dFdx use errors at the use line');
  okInfo('#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nvoid main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT'); // #extension alone compiles

  const e4 = errs('precision mediump float;\nuniform sampler2D s;\nvoid main() { gl_FragColor = texture2DLodEXT(s, vec2(0.0), 0.0); }', 100, 'FRAGMENT');
  check(hasErr(e4, 3, "'texture2DLodEXT' : requires extension 'GL_EXT_shader_texture_lod' which is not enabled"), 'texture2DLodEXT without EXT → error');
  okInfo('#extension GL_EXT_shader_texture_lod : enable\nprecision mediump float;\nuniform sampler2D s;\nvoid main() { gl_FragColor = texture2DLodEXT(s, vec2(0.0), 0.0); }', 100, 'FRAGMENT', ['GL_EXT_shader_texture_lod']);
  const e5 = errs('#extension GL_EXT_shader_texture_lod : enable\nuniform sampler2D s;\nvoid main() { gl_Position = texture2DLodEXT(s, vec2(0.0), 0.0); }', 100, 'VERTEX', ['GL_EXT_shader_texture_lod']);
  check(hasErr(e5, 3, "'texture2DLodEXT' : not available in fragment shaders"), 'texture2DLodEXT in VERTEX → error (fragment-only)');

  const e6 = errs('precision mediump float;\nvoid main() { gl_FragDepthEXT = 0.5; gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
  check(hasErr(e6, 2, "'gl_FragDepthEXT' : undeclared identifier"), 'gl_FragDepthEXT without EXT_frag_depth → error (builtin unknown)');
}

/* ------------------------------------------------------------------ */
/* 9. Version rules                                                    */
/* ------------------------------------------------------------------ */

{
  const e1 = errs('#version 300 es\nvoid main() {}', 100, 'VERTEX');
  check(hasErr(e1, 1, 'invalid #version directive: 300 es not supported in a WebGL 1 context'), '#version 300 es in a 100 context → error line 1');
  const e2 = errs('void main() { uint u = 1u; }', 100, 'VERTEX');
  check(hasErr(e2, 1, "'u' suffix requires GLSL ES 3.00"), 'uint literal in 100 → error');
  const e3 = errs('#version 300 es\nattribute vec4 p;\nvoid main() {}', 300, 'VERTEX');
  check(e3.length > 0 && e3[0].line === 2, 'attribute keyword in 300 → parse error line 2');
  const e4 = errs('#version 300 es\nvarying vec4 v;\nvoid main() {}', 300, 'VERTEX');
  check(e4.length > 0 && e4[0].line === 2, 'varying keyword in 300 → parse error line 2');
  const e5 = errs('#version 300 es\nprecision mediump float;\nvoid main() { gl_FragColor = vec4(1.0); }', 300, 'FRAGMENT');
  check(hasErr(e5, 3, "'gl_FragColor' : undeclared identifier"), 'gl_FragColor in 300 → undeclared');
}

/* ------------------------------------------------------------------ */
/* 10. discard stage rule                                              */
/* ------------------------------------------------------------------ */

{
  const e = errs('void main() { discard; }', 100, 'VERTEX');
  check(hasErr(e, 1, "'discard' : only allowed in fragment shaders"), 'discard in vertex → error');
  okInfo('precision mediump float;\nvoid main() { discard; }', 100, 'FRAGMENT'); // fragment OK
  const e2 = errs('#version 300 es\nvoid main() { discard; }', 300, 'VERTEX');
  check(hasErr(e2, 2, "'discard' : only allowed in fragment shaders"), 'discard in 3.00 vertex → error');
}

/* ------------------------------------------------------------------ */
/* 11. compileShader result shape                                      */
/* ------------------------------------------------------------------ */

{
  const r = comp('attribute vec4 p;\nvoid main() { gl_Position = p; }', 100, 'VERTEX');
  check(r.ok === true, 'compileShader success shape');
  if (r.ok) {
    check(r.shader.infoLog === '', "infoLog '' on success");
    check(r.shader.version === 100, 'shader.version 100');
    check(r.shader.type === 'VERTEX', 'shader.type VERTEX');
    check(r.shader.extensions.size === 0, 'no extensions enabled');
    check(r.shader.source === 'attribute vec4 p;\nvoid main() { gl_Position = p; }', 'shader.source preserved');
    check(r.shader.info.attributes.length === 1, 'shader.info attached');
  }
  const r2 = comp('#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nvoid main() { float d = dFdx(1.0); gl_FragColor = vec4(d); }', 100, 'FRAGMENT', ['GL_OES_standard_derivatives']);
  check(r2.ok === true, 'extension shader compiles');
  if (r2.ok) {
    check(r2.shader.extensions.has('GL_OES_standard_derivatives'), 'shader.extensions contains enabled name');
    check(r2.shader.type === 'FRAGMENT', 'shader.type FRAGMENT');
  }

  // Error shape: exact line of a multi-line source (no float default declared).
  const src = 'precision mediump float;\nvoid main() {\n  float x = 1.0;\n  float y = 2.0;\n  gl_FragColor = vec4(x + y);\n}';
  const e = errs(src.replace('precision mediump float;\n', ''), 100, 'FRAGMENT');
  check(e.length > 0 && e[0].line === 2 && e[0].message === "'x' : No precision specified for (float)", `first error at line 2 — got ${JSON.stringify(e[0])}`);
  check(typeof e[0].message === 'string' && e[0].message.length > 0, 'error message is a non-empty string');

  // Errors are capped at MAX_COMPILE_ERRORS (20).
  const many = Array.from({ length: 40 }, (_, i) => `uniform vec4 u${i};`).join('\n') + '\nvoid main() { gl_FragColor = vec4(1.0); }';
  const e2 = errs(many, 100, 'FRAGMENT');
  check(e2.length <= 20, `errors capped at 20 — got ${e2.length}`);
  check(e2.length === 20, `saturated to 20 errors — got ${e2.length}`);
}

/* ------------------------------------------------------------------ */
/* 12. Direct analyze() entry                                          */
/* ------------------------------------------------------------------ */

{
  // Build an AST via the front-end stages, then call analyze directly.
  const pp = preprocess('attribute vec4 p;\nvoid main() { gl_Position = p; }', { version: 100 });
  if (pp.ok) {
    const lex = tokenize(pp.tokens, pp.version);
    if (lex.ok) {
      const parsed = parse(lex.tokens, { version: pp.version, extensionDirectives: pp.extensionDirectives });
      if (parsed.ok) {
        const r = analyze(parsed.ast, { type: 'VERTEX', extensions: new Set(pp.extensions) });
        check(r.ok === true, 'direct analyze success');
        if (r.ok) {
          check(r.info.attributes.length === 1 && r.info.attributes[0].name === 'p', 'direct analyze: attribute collected');
          check(r.info.uses.pointSize === false && r.info.uses.vertexId === false, 'direct analyze: uses clean');
        }
        const r2 = analyze(parsed.ast, { type: 'FRAGMENT', extensions: new Set(pp.extensions) });
        check(r2.ok === false && r2.errors.length > 0, 'direct analyze honors stage (attribute in fragment → error)');
      }
    }
  }
  const pp2 = preprocess('#version 300 es\nprecision mediump float;\nout vec4 c;\nvoid main() { c = vec4(1.0); }', { version: 300 });
  if (pp2.ok) {
    const lex = tokenize(pp2.tokens, pp2.version);
    if (lex.ok) {
      const parsed = parse(lex.tokens, { version: pp2.version, extensionDirectives: pp2.extensionDirectives });
      if (parsed.ok) {
        const r = analyze(parsed.ast, { type: 'FRAGMENT', extensions: new Set(pp2.extensions) });
        check(r.ok === true, 'direct analyze 3.00 success');
        if (r.ok) check(r.info.outputs.length === 1 && r.info.outputs[0].name === 'c', 'direct analyze: output collected');
      }
    }
  }
  // analyze never mutates the caller's extension set.
  const exts = new Set<string>();
  const pp3 = preprocess('void main() {}', { version: 100 });
  if (pp3.ok) {
    const lex = tokenize(pp3.tokens, pp3.version);
    if (lex.ok) {
      const parsed = parse(lex.tokens, { version: pp3.version, extensionDirectives: pp3.extensionDirectives });
      if (parsed.ok) {
        analyze(parsed.ast, { type: 'VERTEX', extensions: exts });
        check(exts.size === 0, 'analyze does not mutate opts.extensions');
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* 13. Constructor rules (GLSL ES 1.00 §5.4.2) + builtin-name overloads */
/* ------------------------------------------------------------------ */

{
  // --- vecN(matM): first N components of the matrix, column-major order ---
  okInfo('void main() { mat2 m = mat2(1.0); vec4 v = vec4(m); gl_Position = v; }', 100, 'VERTEX'); // vec4(mat2)
  okInfo('void main() { mat4 m = mat4(1.0); vec2 v = vec2(m); gl_Position = vec4(v, 0.0, 1.0); }', 100, 'VERTEX'); // vec2(mat4) shorten
  okInfo('void main() { mat3 m = mat3(1.0); vec3 v = vec3(m); gl_Position = vec4(v, 1.0); }', 100, 'VERTEX'); // vec3(mat3)
  okInfo('void main() { mat3 m = mat3(1.0); vec4 v = vec4(m); gl_Position = v; }', 100, 'VERTEX'); // vec4(mat3) shorten (9 comps)
  okInfo('void main() { mat4 m = mat4(1.0); ivec4 v = ivec4(m); gl_Position = vec4(v); }', 100, 'VERTEX'); // ivec4(mat4)
  okInfo('void main() { mat4 m = mat4(1.0); bvec4 v = bvec4(m); gl_Position = vec4(1.0); }', 100, 'VERTEX'); // bvec4(mat4)
  // Matrix followed by a further argument that contributes nothing → error
  // (matches ANGLE/CTS: vec4(mat2, scalar) is "too many arguments").
  const m1 = errs('void main() { mat2 m = mat2(1.0); vec4 v = vec4(m, 2.0); gl_Position = v; }', 100, 'VERTEX');
  check(m1.length > 0, 'vec4(mat2, scalar) → error (matrix contributes all components)');
  // ... but a matrix LATER in the argument list is fine when earlier args
  // leave room: vec4(2.0, mat2) fills 1 + 3 components.
  okInfo('void main() { mat2 m = mat2(1.0); vec4 v = vec4(2.0, m); gl_Position = v; }', 100, 'VERTEX');

  // --- mixed scalar+vector constructor args (scalars combine with vectors) ---
  okInfo('void main() { vec3 v = vec3(5.0, 4.0, ivec2(2.0, 1.0)); gl_Position = vec4(v, 1.0); }', 100, 'VERTEX'); // ogles CorrectConstruct
  okInfo('void main() { vec3 v = vec3(2, 2.0, 1); gl_Position = vec4(v, 1.0); }', 100, 'VERTEX'); // mixed int/float scalars
  okInfo('void main() { vec3 v3 = vec3(1.0); vec3 v = vec3(1.2, v3); gl_Position = vec4(v, 1.0); }', 100, 'VERTEX'); // shorten: 4 comps into 3
  okInfo('void main() { bvec4 b = bvec4(true); vec3 v = vec3(b); gl_Position = vec4(v, 1.0); }', 100, 'VERTEX'); // bool → float, shorten
  okInfo('void main() { vec4 v4 = vec4(1.0); vec2 v = vec2(v4); gl_Position = vec4(v, 0.0, 1.0); }', 100, 'VERTEX'); // shorten vec4 → vec2
  okInfo('void main() { vec2 v2 = vec2(1.0); vec4 v = vec4(3.0, v2, 4.0); gl_Position = v; }', 100, 'VERTEX'); // 1+2+1 = 4
  const m2 = errs('void main() { vec4 v4 = vec4(1.0); vec4 w = vec4(v4, v4, v4); gl_Position = w; }', 100, 'VERTEX');
  check(m2.length > 0, 'vec4(v, v, v) → error (unused trailing argument)');
  const m3 = errs('void main() { vec4 v = vec4(1.0, 2.0, 3.0); gl_Position = v; }', 100, 'VERTEX');
  check(m3.length > 0, 'vec4(1,2,3) → error (not enough components)');
  const m4 = errs('void main() { vec2 v2 = vec2(1.0); vec4 v = vec4(v2); gl_Position = v; }', 100, 'VERTEX');
  check(m4.length > 0, 'vec4(vec2) → error (not enough components)');
  okInfo('void main() { float f = float(vec2(1.0, 2.0)); gl_Position = vec4(f); }', 100, 'VERTEX'); // scalar ctor takes first element

  // --- matrix constructors from vectors / mixed args ---
  okInfo('void main() { vec3 v = vec3(1.0); mat3 m = mat3(v, v, v); gl_Position = vec4(1.0); }', 100, 'VERTEX'); // 3 columns
  okInfo('void main() { vec3 v = vec3(1.0); mat2 m = mat2(v, 2.0); gl_Position = vec4(1.0); }', 100, 'VERTEX'); // vec3 + scalar
  okInfo('void main() { vec4 v = vec4(1.0); mat2 m = mat2(v); gl_Position = vec4(1.0); }', 100, 'VERTEX'); // single vec4 arg
  okInfo('void main() { vec3 a = vec3(1.0); vec4 b = vec4(1.0); mat4 m = mat4(a, b, b, a, 2.0, 3.0); gl_Position = vec4(1.0); }', 100, 'VERTEX'); // ogles CorrectConstFolding1 pattern
  okInfo('void main() { bool b = true; mat4 m = mat4(b); gl_Position = vec4(1.0); }', 100, 'VERTEX'); // bool scalar diagonal
  okInfo('void main() { mat3 m = mat3(1.0); mat2 n = mat2(m); gl_Position = vec4(1.0); }', 100, 'VERTEX'); // mat2(mat3) single matrix arg
  const m5 = errs('void main() { mat2 a = mat2(1.0); mat3 m = mat3(a, 2.0); gl_Position = vec4(1.0); }', 100, 'VERTEX');
  check(m5.length > 0, 'mat3(mat2, scalar) → error (matrix in multi-arg matrix constructor)');

  // --- user functions may overload builtin names (GLSL ES 1.00 §6.1) ---
  // `radians(3)` must resolve to the USER int radians(int), not the builtin
  // radians(float) (which would return float and fail the int assignment).
  okInfo('int radians(int x) { return x; }\nvoid main() { int f = 45; f = radians(f); }', 100, 'VERTEX');
  okInfo('int radians(int x) { return x; }\nvoid main() { int f = radians(3); gl_Position = vec4(float(f)); }', 100, 'VERTEX');
  // A call only the builtin can take still routes to the builtin table.
  okInfo('int radians(int x) { return x; }\nvoid main() { float f = radians(1.0); gl_Position = vec4(f); }', 100, 'VERTEX');
  const m6 = errs('int radians(int x) { return x; }\nvoid main() { float f = radians(true); gl_Position = vec4(f); }', 100, 'VERTEX');
  check(m6.length > 0, 'radians(bool): neither user overload nor builtin matches → error');
  // Fragment variant of the ogles CorrectBuiltInOveride pattern.
  okInfo('precision mediump float;\nint radians(int x) { return x; }\nvoid main() { int f = 45; f = radians(f); gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
}

/* ------------------------------------------------------------------ */
/* 14. Standalone `invariant <name>;` declarations                     */
/* ------------------------------------------------------------------ */

{
  // GLSL ES 1.00 §4.6.1 / 3.00 §4.6: the short-form `invariant <name>;`
  // statement must FOLLOW the declaration of the named variable (CTS
  // shaders-with-invariance cases 7/8). Wrong order → compile error at the
  // invariant statement's line; correct order compiles. Pins the semantics
  // case in analyzeProgram's global pre-pass (global.lookup order check).
  const w1 = errs(
    'invariant v_varying;\nvarying vec4 v_varying;\nvoid main() { gl_Position = v_varying; }',
    100,
    'VERTEX',
  );
  check(hasErr(w1, 1, "'v_varying' : invariant declaration must follow the variable declaration"), '1.00 VS wrong-order invariant → error line 1');
  const w2 = errs(
    'precision mediump float;\ninvariant v_varying;\nvarying vec4 v_varying;\nvoid main() { gl_FragColor = v_varying; }',
    100,
    'FRAGMENT',
  );
  check(hasErr(w2, 2, "'v_varying' : invariant declaration must follow the variable declaration"), '1.00 FS wrong-order invariant → error line 2');
  const w3 = errs(
    '#version 300 es\ninvariant v;\nout vec4 v;\nvoid main() { v = vec4(1.0); }',
    300,
    'VERTEX',
  );
  check(hasErr(w3, 2, "'v' : invariant declaration must follow the variable declaration"), '3.00 wrong-order invariant → error line 2');
  // Unknown name (no declaration at all) is the same wrong-order error.
  const w4 = errs('invariant nope;\nvoid main() { gl_Position = vec4(1.0); }', 100, 'VERTEX');
  check(hasErr(w4, 1, "'nope' : invariant declaration must follow the variable declaration"), 'invariant on undeclared name → error line 1');

  // Correct order: `varying vec4 v_varying;` THEN `invariant v_varying;`
  // (CTS shaders-with-invariance cases 5/6, invariant-does-not-leak case 1).
  okInfo('varying vec4 v_varying;\ninvariant v_varying;\nvoid main() { gl_Position = v_varying; }', 100, 'VERTEX');
  okInfo('precision mediump float;\nvarying vec4 v_varying;\ninvariant v_varying;\nvoid main() { gl_FragColor = v_varying; }', 100, 'FRAGMENT');

  // Invariant-capable builtins (CTS shaders-with-invariance cases 9-14 +
  // fragcolor-fragdata-invariant.html): gl_Position/gl_PointSize (VS),
  // gl_FragCoord/gl_PointCoord (FS), gl_FragColor/gl_FragData (FS outputs).
  okInfo('invariant gl_Position;\nvoid main() { gl_Position = vec4(0.0); }', 100, 'VERTEX');
  okInfo('invariant gl_PointSize;\nvoid main() { gl_PointSize = 1.0; gl_Position = vec4(0.0); }', 100, 'VERTEX');
  okInfo('precision mediump float;\ninvariant gl_FragCoord;\nvoid main() { gl_FragColor = gl_FragCoord; }', 100, 'FRAGMENT');
  okInfo('precision mediump float;\ninvariant gl_PointCoord;\nvoid main() { gl_FragColor = vec4(gl_PointCoord, 0.0, 0.0); }', 100, 'FRAGMENT');
  okInfo('precision mediump float;\ninvariant gl_FragColor;\ninvariant gl_FragData;\nvoid main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
  okInfo('#version 300 es\ninvariant gl_Position;\nvoid main() { gl_Position = vec4(1.0); }', 300, 'VERTEX');

  // Other builtins cannot be invariant (CTS shaders-with-invariance case 16:
  // `invariant gl_FrontFacing;` must fail compilation).
  const b1 = errs(
    'precision mediump float;\ninvariant gl_FrontFacing;\nvoid main() { gl_FragColor = gl_FrontFacing ? vec4(1.0) : vec4(0.0, 0.0, 0.0, 1.0); }',
    100,
    'FRAGMENT',
  );
  check(hasErr(b1, 2, "'gl_FrontFacing' : invariant cannot be applied to this built-in variable"), 'invariant gl_FrontFacing → error line 2');
  const b2 = errs('precision mediump float;\ninvariant gl_FragDepthEXT;\nvoid main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT', ['GL_EXT_frag_depth']);
  check(b2.length > 0, 'invariant on other builtin (gl_FragDepthEXT) → error');
}

/* ------------------------------------------------------------------ */
/* 15. Scope: for/function-body no-new-scope + struct-with-array rules */
/* ------------------------------------------------------------------ */

{
  // GLSL ES §4.2.2: the for body is a statement-no-new-scope — the for-init
  // declaration's scope extends over the body, so a braced body does NOT push
  // a scope; `for (int i...) { int i; }` is a SAME-SCOPE redefinition (CTS
  // shader-with-for-scoping.html). Blocks nested INSIDE the body still push
  // their own scopes (shadowing legal).
  const f1 = errs(
    'precision mediump float;\nvoid main() {\n  int k = 0;\n  for (int i = 0; i < 10; i++) { int i = k+i; }\n  gl_FragColor = vec4(float(k));\n}',
    100,
    'FRAGMENT',
  );
  check(hasErr(f1, 4, "'i' : redefinition"), 'for body redeclares for-init i → redefinition error line 4');
  okInfo(
    'precision mediump float;\nvoid main() {\n  int k = 0;\n  for (int i = 0; i < 10; i++) { { int i = k+i; } }\n  gl_FragColor = vec4(float(k));\n}',
    100,
    'FRAGMENT',
  );

  // Params + body form a SINGLE scope (CTS shader-with-functional-scoping.
  // html): a body-level redeclaration of a param is a redefinition; a nested
  // block shadowing a param is legal.
  const f2 = errs(
    'precision mediump float;\nint f(int k) {\n  int k = k + 3;\n  return k;\n}\nvoid main() { gl_FragColor = vec4(1.0); }',
    100,
    'FRAGMENT',
  );
  check(hasErr(f2, 3, "'k' : redefinition"), 'function body redeclares param k → redefinition error line 3');
  okInfo(
    'precision mediump float;\nint f(int k) {\n  { int k = k + 3; }\n  return k;\n}\nvoid main() { gl_FragColor = vec4(1.0); }',
    100,
    'FRAGMENT',
  );
  okInfo(
    '#version 300 es\nint f(int k) {\n  { int k = k + 3; }\n  return k;\n}\nvoid main() { }',
    300,
    'FRAGMENT',
  );

  // Structs containing arrays are not assignable / not comparable (GLSL ES
  // 1.00 §5.7/§5.8; CTS struct-assign.html / struct-equals.html) — the rule
  // is version-independent (100 AND 300).
  const s1 = errs(
    'precision mediump float;\nstruct S { float f[3]; };\nvoid main() {\n  S a, b;\n  a = b;\n  gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);\n}',
    100,
    'FRAGMENT',
  );
  check(hasErr(s1, 5, "'=' : cannot assign a struct containing an array"), 'struct-with-array assignment → error line 5');
  const s2 = errs(
    'precision mediump float;\nstruct S { float f[3]; };\nvoid main() {\n  S a, b;\n  bool c = (a == b);\n  gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);\n}',
    100,
    'FRAGMENT',
  );
  check(hasErr(s2, 5, "'==' : cannot compare structs containing an array"), 'struct-with-array == → error line 5');
  const s3 = errs(
    'precision mediump float;\nstruct S { float f[3]; };\nvoid main() {\n  S a, b;\n  bool c = (a != b);\n  gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);\n}',
    100,
    'FRAGMENT',
  );
  check(hasErr(s3, 5, "'!=' : cannot compare structs containing an array"), 'struct-with-array != → error line 5');
  const s4 = errs(
    '#version 300 es\nstruct S { float f[3]; };\nout vec4 o;\nvoid main() {\n  S a, b;\n  a = b;\n  o = vec4(1.0);\n}',
    300,
    'FRAGMENT',
  );
  check(hasErr(s4, 6, "'=' : cannot assign a struct containing an array"), '3.00 struct-with-array assignment → error line 6');
  const s5 = errs(
    '#version 300 es\nstruct S { float f[3]; };\nout vec4 o;\nvoid main() {\n  S a, b;\n  bool c = (a == b);\n  o = vec4(1.0);\n}',
    300,
    'FRAGMENT',
  );
  check(hasErr(s5, 6, "'==' : cannot compare structs containing an array"), '3.00 struct-with-array == → error line 6');
}

/* ------------------------------------------------------------------ */
/* 16. ES 3.00 arrays: returns, ==/!=, assignment, .length()           */
/*     (CTS conformance2/glsl3/{array-as-return-value,array-equality,  */
/*      array-in-complex-expression,array-complex-indexing,            */
/*      array-length-side-effects}.html — semantics-level only;        */
/*      runtime emit is codegen's concern)                             */
/* ------------------------------------------------------------------ */

// Array return types (`float[2] f()`) — accepted at 300 (parser rejects 100).
const a1 = okInfo('#version 300 es\nint[2] f() { return int[2](1, 2); }\nvoid main() { }', 300, 'VERTEX');
check(a1 !== null, 'int[2] return type accepted (ES 3.00)');
const a2 = okInfo('#version 300 es\nfloat[2] f() { return float[2](1.0, 2.0); }\nvoid main() { }', 300, 'VERTEX');
check(a2 !== null, 'float[2] return type accepted (ES 3.00, vertex)');
// Return-value/type mismatch → convertible(vt, int[2]) fails.
const a3 = errs('#version 300 es\nint[2] f() { return 1.0; }\nvoid main() { }', 300, 'VERTEX');
check(hasErr(a3, 2, "cannot convert from 'float' to 'int[2]'"), 'array return type mismatch → error line 2');
// Returning a returned array (array-as-return-value fshaderReturnReturnedArray).
const a3b = okInfo('#version 300 es\nint[2] foo() { return int[2](1, 2); }\nint[2] bar() { return foo(); }\nvoid main() { }', 300, 'VERTEX');
check(a3b !== null, 'returning a returned array accepted (ES 3.00)');
// Array param + array-arg call (array-as-return-value fshaderReturnedArrayAsParameter).
const a4 = okInfo('#version 300 es\nbool isSuccess(int[2] a) { return a[0] == 1; }\nvoid main() { bool b = isSuccess(int[2](1, 2)); gl_Position = vec4(float(b)); }', 300, 'VERTEX');
check(a4 !== null, 'array param + array-arg call accepted (ES 3.00)');
// Array ==/!= at 300 (locals).
const a5 = okInfo('#version 300 es\nvoid main() { int a[2] = int[2](1, 2); int b[2] = int[2](1, 2); bool c = (a == b) && (a != b); gl_Position = vec4(float(c)); }', 300, 'VERTEX');
check(a5 !== null, 'array ==/!= accepted (ES 3.00 locals)');
// Size mismatch → 'cannot be compared'.
const a6 = errs('#version 300 es\nvoid main() { int a[2]; int b[3]; bool c = (a == b); gl_Position = vec4(float(c)); }', 300, 'VERTEX');
check(hasErr(a6, 2, "'==' : operands of type 'int[2]' and 'int[3]' cannot be compared"), 'array == size mismatch → error line 2');
// ES 1.00 array == stays rejected.
const a7 = errs('void main() { int a[2]; int b[2]; bool c = (a == b); gl_Position = vec4(float(c)); }', 100, 'VERTEX');
check(hasErr(a7, 1, "'==' : operands of type 'int[2]' and 'int[2]' cannot be compared"), 'ES 1.00 array == → error line 1');
// Array assignment: 300 accepted, 100 rejected.
const a8 = okInfo('#version 300 es\nvoid main() { int a[3]; int b[3]; a = b; gl_Position = vec4(1.0); }', 300, 'VERTEX');
check(a8 !== null, 'array assignment accepted (ES 3.00)');
const a9 = errs('void main() { int a[3]; int b[3]; a = b; gl_Position = vec4(1.0); }', 100, 'VERTEX');
check(hasErr(a9, 1, "'=' : cannot assign to an array in GLSL ES 1.00"), 'ES 1.00 array assignment → error line 1');
// .length() on every array-valued expression form (array-length-side-effects).
const a10 = okInfo('#version 300 es\nvoid main() { int a[3]; int n = a.length(); gl_Position = vec4(float(n)); }', 300, 'VERTEX');
check(a10 !== null, 'a.length() accepted');
const a11 = okInfo('#version 300 es\nint[2] f() { return int[2](1, 2); }\nvoid main() { int n = (f()).length(); gl_Position = vec4(float(n)); }', 300, 'VERTEX');
check(a11 !== null, '(f()).length() accepted');
const a12 = okInfo('#version 300 es\nvoid main() { int a[3]; int b[3]; int c = (a = b).length(); gl_Position = vec4(float(c)); }', 300, 'VERTEX');
check(a12 !== null, '(a = b).length() accepted');
const a13 = okInfo('#version 300 es\nvoid main() { int a = (int[1](0)).length(); gl_Position = vec4(float(a)); }', 300, 'VERTEX');
check(a13 !== null, '(int[1](0)).length() accepted');
// Array-of-struct equality at 300: plain S accepted; S-with-array-member rejected.
const a14 = okInfo('#version 300 es\nstruct S { int x; };\nvoid main() { S a[3]; S b[3]; bool c = (a == b); gl_Position = vec4(float(c)); }', 300, 'VERTEX');
check(a14 !== null, 'S[3] == S[3] accepted (S without arrays)');
const a15 = errs('#version 300 es\nstruct S { int f[2]; };\nvoid main() { S a[3]; S b[3]; bool c = (a == b); gl_Position = vec4(float(c)); }', 300, 'VERTEX');
check(hasErr(a15, 3, "'==' : cannot compare structs containing an array"), 'S[3] == S[3] with S containing an array → error line 3');
// Indexing an array-valued expression result (array-complex-indexing).
const a16 = okInfo('#version 300 es\nvoid main() { float a[2] = float[2](0.0, 0.0); float b[2] = float[2](2.0, 1.0); float c = (a = b)[0]; gl_Position = vec4(c); }', 300, 'VERTEX');
check(a16 !== null, '(a = b)[0] indexing accepted');
// Complex-expression contexts (array-in-complex-expression): short-circuit
// && / || / ternary / comma with array == operands.
const a17 = okInfo('#version 300 es\nint g = 0;\nint[2] plus() { ++g; return int[2](g, g); }\nbool minus() { --g; return false; }\nvoid main() { int a[2] = int[2](0, 0); minus() && (a == plus()); gl_Position = vec4(float(g)); }', 300, 'VERTEX');
check(a17 !== null, '&& short-circuit with array == accepted');
const a18 = okInfo('#version 300 es\nint g = 0;\nint[2] plus() { ++g; return int[2](g, g); }\nvoid main() { int a[2] = int[2](0, 0); (g == 0) ? true : (a == plus()); gl_Position = vec4(float(g)); }', 300, 'VERTEX');
check(a18 !== null, 'ternary with array == arm accepted');
const a19 = okInfo('#version 300 es\nint[2] func(int param) { return int[2](param, param); }\nvoid main() { int a[2]; int j = 0; bool result = ((++j), (a == func(j))); gl_Position = vec4(float(result)); }', 300, 'VERTEX');
check(a19 !== null, 'comma sequence with array == accepted');
// Multi-dim return types are illegal (arrays of arrays — GLSL ES 3.00 §4.1.9).
const a20 = errs('#version 300 es\nfloat[2][3] f() { return float[2][3](0.0); }\nvoid main() { }', 300, 'VERTEX');
check(hasErr(a20, 2, "'[' : arrays of arrays are not allowed in function return types"), 'multi-dim return type → error line 2');

/* ------------------------------------------------------------------ */
/* Summary                                                             */
/* ------------------------------------------------------------------ */

console.log(`semantics selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
