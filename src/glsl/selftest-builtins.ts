/**
 * selftest-builtins.ts — pinning checks for the GLSL ES 1.00 builtin-table
 * fixes (batch C):
 *
 *  BUG 10 — §8.6 vector relational int/bool variants: ES 1.00 lessThan/
 *  lessThanEqual/greaterThan/greaterThanEqual take vec AND ivec; equal/
 *  notEqual additionally take bvec (GLSL ES 1.00 rev 17 §8.6). The variants
 *  live in `relational100` (merged into the version-100 table by
 *  builtinSignatures(100), kept OUT of builtinFunctions100 so 300.ts's
 *  common100 superset does not duplicate its rel300/eq300 rows).
 *
 *  BUG 11 — §7.6 built-in uniform `gl_DepthRange` (struct
 *  gl_DepthRangeParameters { near, far, diff }), usable in BOTH stages.
 *
 * Run: npx tsx src/glsl/selftest-builtins.ts   (prints "OK", exit 0)
 *
 * Checks are compile-level (the authoritative gate for the OGLEs raw-shader
 * pages) using the VERBATIM OGLEs shader bodies, plus table-shape assertions
 * guarding the 100/300 split (no duplicates anywhere, 3.00 counts unchanged).
 */
import { compileShader } from './compiler.js';
import {
  builtinFunctions100,
  builtinFunctions300,
  builtinSignatures,
  builtinVariables100,
  matches,
  relational100,
} from './builtins/index.js';
import type { BuiltinSignature } from './builtins/index.js';
import { typeEquals } from './types.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function mustCompile(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT', label: string): void {
  checks++;
  const r = compileShader(src, { type, version });
  if (!r.ok) {
    failures++;
    console.error(`FAIL: ${label} — expected success, got ${JSON.stringify(r.errors)}`);
  }
}

function mustFail(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT', label: string): void {
  checks++;
  const r = compileShader(src, { type, version });
  if (r.ok) {
    failures++;
    console.error(`FAIL: ${label} — expected a compile error`);
  }
}

/** Unique signature key (name + canonical param types), same idea as the
 *  builtins/selftest-builtins.ts `sigKey`. */
function sigKey(s: BuiltinSignature): string {
  const canon = (t: unknown): string => JSON.stringify(t);
  return s.name + '(' + s.params.map(canon).join(',') + ')';
}

function noDuplicates(table: BuiltinSignature[], label: string): void {
  const keys = new Set<string>();
  for (const s of table) {
    const k = sigKey(s);
    check(!keys.has(k), `${label}: duplicate signature ${k}`);
    keys.add(k);
  }
}

/* ------------------------------------------------------------------ */
/* 1. Table shape — BUG 10 (relational variants)                       */
/* ------------------------------------------------------------------ */

check(builtinFunctions100.length === 216, `builtinFunctions100 must stay 216 (shared float core), got ${builtinFunctions100.length}`);
check(relational100.length === 24, `relational100 must have 24 signatures (4×ivecN lessThan-family + 2×(ivecN+bvecN) equal/notEqual), got ${relational100.length}`);
check(builtinSignatures(100).length === 240, `builtinSignatures(100) must be 240 (216 + 24), got ${builtinSignatures(100).length}`);

// lessThan* family: vec + ivec = 6 per name in the version-100 table.
for (const name of ['lessThan', 'lessThanEqual', 'greaterThan', 'greaterThanEqual']) {
  check(matches(name, builtinSignatures(100)).length === 6, `${name} must have 6 signatures in 100 (vec+ivec), got ${matches(name, builtinSignatures(100)).length}`);
}
// equal/notEqual: vec + ivec + bvec = 9 per name.
for (const name of ['equal', 'notEqual']) {
  check(matches(name, builtinSignatures(100)).length === 9, `${name} must have 9 signatures in 100 (vec+ivec+bvec), got ${matches(name, builtinSignatures(100)).length}`);
}

// The shared float core keeps vec-only relationals (300.ts rel300/eq300 add
// the non-float rows on top — duplication would break 3.00 overloads).
check(matches('lessThan', builtinFunctions100).length === 3, 'builtinFunctions100 lessThan must stay vec-only (3 signatures)');
check(matches('equal', builtinFunctions100).length === 3, 'builtinFunctions100 equal must stay vec-only (3 signatures)');

// 3.00 regression: the split must leave the 3.00 table EXACTLY as before.
check(matches('lessThan', builtinFunctions300).length === 9, `lessThan must have 9 signatures in 300 (vec+ivec+uvec), got ${matches('lessThan', builtinFunctions300).length}`);
check(matches('equal', builtinFunctions300).length === 12, `equal must have 12 signatures in 300 (vec+ivec+uvec+bvec), got ${matches('equal', builtinFunctions300).length}`);
check(builtinFunctions300.length === 625, `builtinFunctions300 must stay 625, got ${builtinFunctions300.length}`);

// No duplicates anywhere (pickBest reports ambiguous calls on ties).
noDuplicates(builtinSignatures(100), 'builtinSignatures(100)');
noDuplicates(builtinFunctions300, 'builtinFunctions300');

// Spot-check return types: ivec2 lessThan → bvec2, bvec3 notEqual → bvec3.
const ltIvec2 = matches('lessThan', builtinSignatures(100)).find((s) => s.params[0].kind === 'vector' && s.params[0].base === 'int' && s.params[0].size === 2);
check(ltIvec2 !== undefined && typeEquals(ltIvec2.ret, { kind: 'vector', base: 'bool', size: 2 }), 'lessThan(ivec2, ivec2) must return bvec2');
const neBvec3 = matches('notEqual', builtinSignatures(100)).find((s) => s.params[0].kind === 'vector' && s.params[0].base === 'bool' && s.params[0].size === 3);
check(neBvec3 !== undefined && typeEquals(neBvec3.ret, { kind: 'vector', base: 'bool', size: 3 }), 'notEqual(bvec3, bvec3) must return bvec3');

/* ------------------------------------------------------------------ */
/* 2. BUG 10 — compile the verbatim OGLEs relational shaders (100)      */
/* ------------------------------------------------------------------ */

// equal(bvec2(c), bvec2(true)) — the exact original repro.
mustCompile(`attribute vec4 gtf_Color;
attribute vec4 gtf_Vertex;
uniform mat4 gtf_ModelViewProjectionMatrix;
varying vec4 color;

void main (void)
{
	vec2 c = floor(1.5 * gtf_Color.rg);   // 1/3 true, 2/3 false
	vec2 result = vec2(equal(bvec2(c), bvec2(true)));
	color = vec4(result, 0.0, 1.0);
	gl_Position = gtf_ModelViewProjectionMatrix * gtf_Vertex;
}
`, 100, 'VERTEX', 'ogles equal_bvec2_vert.vert');

mustCompile(`attribute vec4 gtf_Color;
attribute vec4 gtf_Vertex;
uniform mat4 gtf_ModelViewProjectionMatrix;
varying vec4 color;

void main (void)
{
	vec2 c = floor(10.0 * gtf_Color.rg - 4.5);   // round to the nearest integer
	vec2 result = vec2(equal(ivec2(c), ivec2(0)));
	color = vec4(result, 0.0, 1.0);
	gl_Position = gtf_ModelViewProjectionMatrix * gtf_Vertex;
}
`, 100, 'VERTEX', 'ogles equal_ivec2_vert.vert');

mustCompile(`#ifdef GL_ES
precision mediump float;
#endif
varying vec4 color;

void main (void)
{
	vec3 c = floor(1.5 * color.rgb);   // 1/3 true, 2/3 false
	vec3 result = vec3(notEqual(bvec3(c), bvec3(true)));
	gl_FragColor = vec4(result, 1.0);
}
`, 100, 'FRAGMENT', 'ogles notEqual_bvec3_frag.frag');

mustCompile(`#ifdef GL_ES
precision mediump float;
#endif
varying vec4 color;

void main (void)
{
	vec2 c = floor(10.0 * color.rg - 4.5);   // round to the nearest integer
	vec2 result = vec2(notEqual(ivec2(c), ivec2(0)));
	gl_FragColor = vec4(result, 0.0, 1.0);
}
`, 100, 'FRAGMENT', 'ogles notEqual_ivec2_frag.frag');

mustCompile(`attribute vec4 gtf_Color;
attribute vec4 gtf_Vertex;
uniform mat4 gtf_ModelViewProjectionMatrix;
varying vec4 color;

void main (void)
{
	vec2 c = floor(10.0 * gtf_Color.rg - 4.5);   // round to the nearest integer
	vec2 result = vec2(lessThan(ivec2(c), ivec2(0)));
	color = vec4(result, 0.0, 1.0);
	gl_Position = gtf_ModelViewProjectionMatrix * gtf_Vertex;
}
`, 100, 'VERTEX', 'ogles lessThan_ivec2_vert.vert');

mustCompile(`#ifdef GL_ES
precision mediump float;
#endif
varying vec4 color;

void main (void)
{
	vec2 c = floor(10.0 * color.rg - 4.5);   // round to the nearest integer
	vec2 result = vec2(lessThanEqual(ivec2(c), ivec2(0)));
	gl_FragColor = vec4(result, 0.0, 1.0);
}
`, 100, 'FRAGMENT', 'ogles lessThanEqual_ivec2_frag.frag');

mustCompile(`attribute vec4 gtf_Color;
attribute vec4 gtf_Vertex;
uniform mat4 gtf_ModelViewProjectionMatrix;
varying vec4 color;

void main (void)
{
	vec3 c = floor(10.0 * gtf_Color.rgb - 4.5);   // round to the nearest integer
	vec3 result = vec3(greaterThan(ivec3(c), ivec3(0)));
	color = vec4(result, 1.0);
	gl_Position = gtf_ModelViewProjectionMatrix * gtf_Vertex;
}
`, 100, 'VERTEX', 'ogles greaterThan_ivec3_vert.vert');

mustCompile(`#ifdef GL_ES
precision mediump float;
#endif
varying vec4 color;

void main (void)
{
	vec3 c = floor(10.0 * color.rgb - 4.5);   // round to the nearest integer
	vec3 result = vec3(greaterThanEqual(ivec3(c), ivec3(0)));
	gl_FragColor = vec4(result, 1.0);
}
`, 100, 'FRAGMENT', 'ogles greaterThanEqual_ivec3_frag.frag');

/* ------------------------------------------------------------------ */
/* 3. BUG 11 — gl_DepthRange built-in uniform (struct, both stages)     */
/* ------------------------------------------------------------------ */

const dr = builtinVariables100.find((v) => v.name === 'gl_DepthRange');
check(dr !== undefined, 'builtinVariables100 must contain gl_DepthRange');
check(dr !== undefined && dr.stage === 'BOTH', 'gl_DepthRange must be available in BOTH stages');
check(dr !== undefined && dr.writable === false, 'gl_DepthRange must be read-only');
check(dr !== undefined && dr.type.kind === 'struct' && dr.type.name === 'gl_DepthRangeParameters', 'gl_DepthRange type must be struct gl_DepthRangeParameters');
check(
  dr !== undefined && dr.type.kind === 'struct' &&
    dr.type.members.length === 3 &&
    dr.type.members.every((m, i) => m.name === ['near', 'far', 'diff'][i] && m.type.kind === 'scalar' && m.type.base === 'float'),
  'gl_DepthRangeParameters members must be float near/far/diff in order',
);

// Verbatim OGLEs biuDepthRange shaders (member reads inside arithmetic).
mustCompile(`attribute vec4 gtf_Vertex;
uniform mat4 gtf_ModelViewProjectionMatrix;
varying vec4  color;

void main(void)
{
	color = vec4(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff, 1.0);

	gl_Position = gtf_ModelViewProjectionMatrix * gtf_Vertex;
}
`, 100, 'VERTEX', 'ogles biuDepthRange DepthRange_vert.vert');

mustCompile(`#ifdef GL_ES
precision mediump float;
#endif

void main(void)
{
	gl_FragColor = vec4(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff, 1.0);
}
`, 100, 'FRAGMENT', 'ogles biuDepthRange DepthRange_frag.frag');

// Negative checks: unknown member and write to the read-only builtin.
mustFail(`void main(void) {
	gl_Position = vec4(gl_DepthRange.nope, 0.0, 0.0, 1.0);
}
`, 100, 'VERTEX', 'gl_DepthRange.nope must be rejected');

mustFail(`void main(void) {
	gl_DepthRange.near = 1.0;
	gl_Position = vec4(0.0);
}
`, 100, 'VERTEX', 'writing gl_DepthRange.near must be rejected (read-only)');

/* ------------------------------------------------------------------ */
/* 4. 3.00 regression — the split must not disturb 3.00 relationals     */
/* ------------------------------------------------------------------ */

mustCompile(`#version 300 es
precision highp float;
in vec4 gtf_Color;
out vec4 color;
void main (void)
{
	vec2 c = floor(10.0 * gtf_Color.rg - 4.5);
	vec2 r1 = vec2(lessThan(ivec2(c), ivec2(0)));
	vec2 r2 = vec2(equal(bvec2(c), bvec2(true)));
	color = vec4(r1, r2);
	gl_Position = vec4(0.0);
}
`, 300, 'VERTEX', '300 relational ivec/bvec calls must still resolve (no ambiguous duplicates)');

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(
  `builtins selftest: ${checks} checks; ` +
  `100 table=${builtinFunctions100.length} (+${relational100.length} relational variants; ` +
  `signatures(100)=${builtinSignatures(100).length}), 300 table=${builtinFunctions300.length}, ` +
  `100 variables=${builtinVariables100.length}`,
);

if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
