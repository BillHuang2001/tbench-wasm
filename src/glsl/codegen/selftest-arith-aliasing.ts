/**
 * selftest-arith-aliasing.ts — sequential-assignment aliasing in matrix
 * arithmetic codegen (emitArith non-dual matrix×matrix / matrix×vector /
 * vector×matrix paths).
 *
 * GLSL assignment has SIMULTANEOUS semantics: every RHS read must observe
 * pre-assignment values. Pre-fix codegen emitted each matrix-multiply result
 * component as a pure expression string over the OPERAND components, and
 * emitAssign wrote targets sequentially — so an in-place pattern
 * (`v = m * v`, `m = m1 * m2`, `v = m1 * m2 * v`, `v = v * M`) read
 * ALREADY-OVERWRITTEN operand components and produced wrong values
 * (three.js instancing_raycast rendered at 36.48% diff). The fix materializes
 * both operands into fresh temps (one shared pre on component 0) so every RHS
 * read is captured before any target write.
 *
 * Each case compiles a real VERTEX shader (compileShader → annotated AST),
 * emits `main` (installUserFunctions + emitStatements), runs the generated JS
 * via `new Function('ctx','R', body)` against a hand-built ctx, and asserts
 * the exact clip values (all chosen values are exact dyadics — `===` safe;
 * expectations are computed by the same left-fold order as the codegen).
 *
 * Run: npx tsx src/glsl/codegen/selftest-arith-aliasing.ts
 *
 * Prints "selftest-arith-aliasing: N checks" and exits 0 only when all pass.
 */
import { compileShader } from '../compiler.js';
import type { ShaderUses } from '../compiler.js';
import { CodegenEnv } from './env.js';
import { emitStatements } from './statements.js';
import { installUserFunctions, installUserGlobals } from './functions.js';
import { generateFragmentStage } from './index.js';
import { R } from './runtime.js';
import type { CodegenLayout } from './index.js';
import type { TranslationUnit, FunctionDefinition } from '../ast.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function findFn(tu: TranslationUnit, name: string): FunctionDefinition {
  for (const d of tu.declarations) {
    if (d.kind === 'function-definition' && d.prototype.name === name) return d;
  }
  throw new Error(`no function '${name}'`);
}

function vertexLayout(version: 100 | 300): CodegenLayout {
  return {
    version,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map([['a_pos', 0]]),
    outputLocations: new Map([['gl_FragColor', 0], ['color', 0]]),
    uses: {
      pointSize: false, fragCoord: false, frontFacing: false, pointCoord: false,
      fragDepth: false, vertexId: false, instanceId: false, drawId: false,
      derivatives: false, depthRange: false,
    },
  };
}

/** Compile a VERTEX shader, emit `main`, run it; returns the body + ctx. */
function runVertex(
  src: string,
  version: 100 | 300,
): { body: string; ctx: Record<string, any> } {
  const r = compileShader(src, { type: 'VERTEX', version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const mainFn = findFn(r.shader.ast, 'main');
  const env = new CodegenEnv('VERTEX', vertexLayout(version));
  const globalInit = installUserGlobals(r.shader.ast, env);
  installUserFunctions(r.shader.ast, env);
  const stmts = emitStatements(mainFn.body.body, env);
  const body = [...(env.temps.length ? [`var ${env.temps.join(', ')};`] : []), ...globalInit, ...stmts].join('\n');
  const fn = new Function('ctx', 'R', body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: { position: [0, 0, 0, 0], pointSize: 0 },
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    scratch: new Float32Array(Math.max(env.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(env.intScratchSize, 16)),
  };
  fn(ctx, R);
  return { body, ctx };
}

/** Dual-mode FRAGMENT layout (derivatives ON) for the (j) guard. */
function fragmentLayout300(): CodegenLayout {
  const uses: ShaderUses = {
    pointSize: false, fragCoord: false, frontFacing: false, pointCoord: false,
    fragDepth: false, vertexId: false, instanceId: false, drawId: false,
    derivatives: true, depthRange: false,
  };
  return {
    version: 300,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map(),
    outputLocations: new Map([['color', 0]]),
    uses,
  };
}

/** Compile a FRAGMENT shader (dual mode via derivatives), run it; returns ctx. */
function runFragment(src: string): { ctx: Record<string, any> } {
  const r = compileShader(src, { type: 'FRAGMENT', version: 300 });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const res = generateFragmentStage(r.shader.ast as TranslationUnit, fragmentLayout300());
  const fn = new Function('ctx', 'R', res.body);
  const ctx: Record<string, any> = {
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(res.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(res.intScratchSize, 16)),
    out: { color: [new Float32Array(4)], fragDepth: 0 },
    discarded: false,
    fragCoord: new Float32Array([0, 0, 0, 0]),
    frontFacing: true,
    pointCoord: new Float32Array([0, 0]),
    varyings: [],
  };
  fn(ctx, R);
  return { ctx };
}

/* ------------------------------------------------------------------ */
/* Exact column-major mat4 helpers (left-fold order matches codegen)   */
/* ------------------------------------------------------------------ */

/** m[c*4+r] = element at column c, row r. Result (bCols=4 × aRows=4). */
function matMul4(a: number[], b: number[]): number[] {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let acc = 0;
      for (let s = 0; s < 4; s++) acc = acc + a[s * 4 + r] * b[c * 4 + s];
      out[c * 4 + r] = acc;
    }
  }
  return out;
}

/** result[r] = Σ_c m[c*4+r] * v[c]. */
function matVec4(m: number[], v: number[]): number[] {
  const out = new Array<number>(4).fill(0);
  for (let r = 0; r < 4; r++) {
    let acc = 0;
    for (let c = 0; c < 4; c++) acc = acc + m[c * 4 + r] * v[c];
    out[r] = acc;
  }
  return out;
}

/** result[c] = Σ_r v[r] * m[c*4+r]. */
function vecMat4(v: number[], m: number[]): number[] {
  const out = new Array<number>(4).fill(0);
  for (let c = 0; c < 4; c++) {
    let acc = 0;
    for (let r = 0; r < 4; r++) acc = acc + v[r] * m[c * 4 + r];
    out[c] = acc;
  }
  return out;
}

/** Check a vec4 result exactly; report the actual values on failure. */
function checkVec4(got: number[], want: number[], label: string): void {
  check(
    got[0] === want[0] && got[1] === want[1] && got[2] === want[2] && got[3] === want[3],
    `${label}: got [${got.join(', ')}], want [${want.join(', ')}]`,
  );
}

/* ------------------------------------------------------------------ */
/* Non-identity, non-diagonal matrices (columns as vec4 in GLSL)       */
/* ------------------------------------------------------------------ */

const M1 = [1, 0.5, 0, 0, 0.5, 1, 0, 0, 0, 0.25, 1, 0, 0, 0, 0.125, 1]; // shear
const M2 = [2, 0, 0, 0, 1, 3, 0, 0, 0, 1, 4, 0, 0, 0, 1, 5]; // lower bidiagonal
const M = [2, 1, 0, 0, 0, 3, 1, 0, 0, 0, 4, 1, 0, 0, 0, 5]; // upper bidiagonal
const M4 = [2, 1, 0, 0, 1, 3, 1, 0, 0, 1, 4, 1, 0, 0, 1, 5]; // for v * M

const GLSL_M1 =
  'mat4(vec4(1.0, 0.5, 0.0, 0.0), vec4(0.5, 1.0, 0.0, 0.0), vec4(0.0, 0.25, 1.0, 0.0), vec4(0.0, 0.0, 0.125, 1.0))';
const GLSL_M2 =
  'mat4(vec4(2.0, 0.0, 0.0, 0.0), vec4(1.0, 3.0, 0.0, 0.0), vec4(0.0, 1.0, 4.0, 0.0), vec4(0.0, 0.0, 1.0, 5.0))';
const GLSL_M =
  'mat4(vec4(2.0, 1.0, 0.0, 0.0), vec4(0.0, 3.0, 1.0, 0.0), vec4(0.0, 0.0, 4.0, 1.0), vec4(0.0, 0.0, 0.0, 5.0))';
const GLSL_M4 =
  'mat4(vec4(2.0, 1.0, 0.0, 0.0), vec4(1.0, 3.0, 1.0, 0.0), vec4(0.0, 1.0, 4.0, 1.0), vec4(0.0, 0.0, 1.0, 5.0))';

const V = [1, 2, 3, 4];

// (a) chain m1 * m2 * v assigned in place to v (the direct aliasing case
// through a mat×mat then mat×vec chain). Pre-fix: the matvec's later
// components read the already-overwritten v components.
{
  const { ctx } = runVertex(
    `void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  mat4 m1 = ${GLSL_M1};
  mat4 m2 = ${GLSL_M2};
  v = m1 * m2 * v;
  gl_Position = vec4(v.x, v.y, v.z, v.w);
}`,
    300,
  );
  checkVec4(ctx.out.position, matVec4(M1, matVec4(M2, V)), 'v = m1 * m2 * v (300 es)');
}

// (a2) same chain at GLSL ES 1.00 (shared non-dual codegen path).
{
  const { ctx } = runVertex(
    `void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  mat4 m1 = ${GLSL_M1};
  mat4 m2 = ${GLSL_M2};
  v = m1 * m2 * v;
  gl_Position = vec4(v.x, v.y, v.z, v.w);
}`,
    100,
  );
  checkVec4(ctx.out.position, matVec4(M1, matVec4(M2, V)), 'v = m1 * m2 * v (100 es)');
}

// (b) chain m1 * m2 * v into gl_Position (NO in-place aliasing) — the
// mathematically correct result must be preserved by the materialization
// (regression guard: the fix must not change correct outputs).
{
  const { ctx } = runVertex(
    `void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  mat4 m1 = ${GLSL_M1};
  mat4 m2 = ${GLSL_M2};
  gl_Position = m1 * m2 * v;
}`,
    300,
  );
  checkVec4(ctx.out.position, matVec4(M1, matVec4(M2, V)), 'gl_Position = m1 * m2 * v');
}

// (c) in-place v = m * v (direct mat×vec aliasing). Pre-fix: result[1] read
// the NEW v[0] and cascaded.
{
  const { ctx } = runVertex(
    `void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  mat4 m = ${GLSL_M};
  v = m * v;
  gl_Position = vec4(v.x, v.y, v.z, v.w);
}`,
    300,
  );
  checkVec4(ctx.out.position, matVec4(M, V), 'v = m * v');
}

// (d) in-place m = m * m2 (mat×mat aliasing: the target m is the LHS operand;
// pre-fix, result[4] read the NEW m[0]).
{
  const { ctx } = runVertex(
    `void main() {
  mat4 m = ${GLSL_M};
  mat4 m2 = ${GLSL_M2};
  m = m * m2;
  gl_Position = m * vec4(1.0, 2.0, 3.0, 4.0);
}`,
    300,
  );
  checkVec4(ctx.out.position, matVec4(matMul4(M, M2), V), 'm = m * m2');
}

// (e) in-place v = v * M (vector×matrix aliasing — pre-fix, result[1] read
// the NEW v[0]).
{
  const { ctx } = runVertex(
    `void main() {
  vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
  mat4 m = ${GLSL_M4};
  v = v * m;
  gl_Position = vec4(v.x, v.y, v.z, v.w);
}`,
    300,
  );
  checkVec4(ctx.out.position, vecMat4(V, M4), 'v = v * m');
}

/* ------------------------------------------------------------------ */
/* Ctor/member-wrapped in-place transforms (skinnormal pattern)        */
/* ------------------------------------------------------------------ */

// (f) THE three.js r185 skinnormal_vertex.glsl.js pattern:
// `objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;` — an
// IN-PLACE vector transform wrapped in a ctor + `.xyz` member access. The
// ctor result is materialized into temps (walkObject's p.pre), but pre-fix
// the non-dual reads() folded that chain into EVERY component's v string and
// emitAssignStmt wrote targets sequentially — component 1 re-ran the chain
// and read the ALREADY-OVERWRITTEN objectNormal.x (component 2 read the new
// x AND the new y). Expected values: M × [1,2,3,0] = [2,7,14,3] (exact
// dyadics — `===` safe), so gl_Position must be [2,7,14,1].
{
  const { ctx } = runVertex(
    `void main() {
  vec3 objectNormal = vec3(1.0, 2.0, 3.0);
  mat4 skinMatrix = ${GLSL_M};
  objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
  gl_Position = vec4(objectNormal.x, objectNormal.y, objectNormal.z, 1.0);
}`,
    300,
  );
  checkVec4(
    ctx.out.position,
    [...matVec4(M, [1, 2, 3, 0]).slice(0, 3), 1],
    'objectNormal = vec4(m * vec4(objectNormal, 0.0)).xyz (300 es)',
  );
}

// (f2) same pattern at GLSL ES 1.00 (shared non-dual codegen path).
{
  const { ctx } = runVertex(
    `void main() {
  vec3 objectNormal = vec3(1.0, 2.0, 3.0);
  mat4 skinMatrix = ${GLSL_M};
  objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
  gl_Position = vec4(objectNormal.x, objectNormal.y, objectNormal.z, 1.0);
}`,
    100,
  );
  checkVec4(
    ctx.out.position,
    [...matVec4(M, [1, 2, 3, 0]).slice(0, 3), 1],
    'objectNormal = vec4(m * vec4(objectNormal, 0.0)).xyz (100 es)',
  );
}

// (g) NON-in-place sanity: the same ctor/member-wrapped transform into a
// FRESH target must equal the reference (regression guard — the fix must not
// change correct outputs).
{
  const { ctx } = runVertex(
    `void main() {
  vec3 other = vec3(1.0, 2.0, 3.0);
  mat4 skinMatrix = ${GLSL_M};
  vec3 fresh = vec4( skinMatrix * vec4( other, 0.0 ) ).xyz;
  gl_Position = vec4(fresh.x, fresh.y, fresh.z, 1.0);
}`,
    300,
  );
  checkVec4(
    ctx.out.position,
    [...matVec4(M, [1, 2, 3, 0]).slice(0, 3), 1],
    'vec3 fresh = vec4(m * vec4(other, 0.0)).xyz (no aliasing)',
  );
}

// (h) NESTED assignment EXPRESSION: the in-place transform as the RHS of
// another assignment (`copy = (objectNormal = vec4(...).xyz);`) — the inner
// assignment's value is consumed per component, so its RHS chain must run
// once BEFORE any of the inner target writes (emitAssign's comp0 hoist).
{
  const { ctx } = runVertex(
    `void main() {
  vec3 objectNormal = vec3(1.0, 2.0, 3.0);
  vec3 copy;
  mat4 skinMatrix = ${GLSL_M};
  copy = (objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz);
  gl_Position = vec4(copy.x, copy.y, copy.z, 1.0);
}`,
    300,
  );
  checkVec4(
    ctx.out.position,
    [...matVec4(M, [1, 2, 3, 0]).slice(0, 3), 1],
    'copy = (objectNormal = vec4(m * vec4(objectNormal, 0.0)).xyz)',
  );
}

// (i) FOR-UPDATE slot: the in-place transform as the loop update expression
// (single-expression slot — the chain folds into ONE comma expression, so it
// must fold once, not once per component).
{
  const { ctx } = runVertex(
    `void main() {
  vec3 objectNormal = vec3(1.0, 2.0, 3.0);
  mat4 skinMatrix = ${GLSL_M};
  for (int i = 0; i < 1; objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz) {
    i++;
  }
  gl_Position = vec4(objectNormal.x, objectNormal.y, objectNormal.z, 1.0);
}`,
    300,
  );
  checkVec4(
    ctx.out.position,
    [...matVec4(M, [1, 2, 3, 0]).slice(0, 3), 1],
    'for-update: objectNormal = vec4(m * vec4(objectNormal, 0.0)).xyz',
  );
}

// (j) DUAL-mode guard (FRAGMENT, derivatives): reads() already attaches pres
// in dual mode — the in-place ctor/member-wrapped pattern must stay correct
// (RHS duals are all 0 here, so color == the same reference).
{
  const r = runFragment(
    `#version 300 es
precision mediump float;
out vec4 color;
void main() {
  vec3 objectNormal = vec3(1.0, 2.0, 3.0);
  mat4 skinMatrix = ${GLSL_M};
  objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
  color = vec4(objectNormal + dFdx(objectNormal) * 0.0, 1.0);
}`,
  );
  checkVec4(
    Array.from(r.ctx.out.color[0]),
    [...matVec4(M, [1, 2, 3, 0]).slice(0, 3), 1],
    'fragment dual: objectNormal = vec4(m * vec4(objectNormal, 0.0)).xyz',
  );
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-arith-aliasing: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
