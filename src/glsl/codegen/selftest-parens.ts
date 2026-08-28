/**
 * selftest-parens.ts — operator-precedence regression checks for int/uint
 * result wraps in GENERATED JS.
 *
 * The codegen wraps every int/uint arithmetic/shift/bitwise result with a
 * `| 0` / `>>> 0` suffix. That wrap binds LOOSER than `+`/`-` (and than
 * `>>`/`>>>`), so a non-self-parenthesized emission like `(X) | 0` embedded in
 * a larger expression mis-parses: `float(a + b) + float(c + d)` emitted
 * `(((a) + (b)) | 0 + ((c) + (d)) >>> 0)` parses as `(a+b) | (0 + (c+d))`
 * instead of `(a+b) + (c+d)`. The invariant this file pins: EVERY `| 0` /
 * `>>> 0` in a generated body is SELF-PARENTHESIZED — `((X) | 0)` — i.e. the
 * character after the wrap is always `)`.
 *
 * Each case compiles + LINKS real shaders via the PUBLIC compileShader /
 * linkProgram API, runs the generated fragment/vertex stage against a
 * hand-built exec ctx, and asserts the runtime value (the primary gate).
 * A shape scan over a regenerated stage body (public generateVertexStage /
 * generateFragmentStage seam, `uses` passed through from the compiler's own
 * ShaderUses — never a hand-written literal) asserts the paren invariant.
 *
 * Run: npx tsx src/glsl/codegen/selftest-parens.ts
 *
 * Prints "selftest-parens: N checks" and exits 0 only when all pass.
 */
import { compileShader, linkProgram } from '../compiler.js';
import type { Program } from '../program.js';
import type { CodegenLayout } from './index.js';
import { generateFragmentStage, generateVertexStage } from './index.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/** Compile+link a VS/FS pair; throws on any error. */
function linkPair(vsSrc: string, fsSrc: string, version: 100 | 300): Program {
  const vs = compileShader(vsSrc, { type: 'VERTEX', version });
  if (!vs.ok) throw new Error(`VS compile failed: ${JSON.stringify(vs.errors)}`);
  const fs = compileShader(fsSrc, { type: 'FRAGMENT', version });
  if (!fs.ok) throw new Error(`FS compile failed: ${JSON.stringify(fs.errors)}`);
  const lp = linkProgram(vs.shader, fs.shader);
  if (!lp.ok) throw new Error(`link failed: ${lp.log}`);
  return lp.program;
}

/**
 * Regenerate one stage's JS body via the PUBLIC codegen seam for the shape
 * scan. The layout mirrors what the linker builds for these minimal shaders
 * (no uniforms/varyings/blocks/attributes); `uses` is the COMPILER's own
 * ShaderUses object (`shader.info.uses`) — never a hand-written literal, so a
 * ShaderUses field added in parallel cannot break this file.
 */
function generateBody(src: string, stage: 'VERTEX' | 'FRAGMENT', version: 100 | 300): string {
  const r = compileShader(src, { type: stage, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const layout: CodegenLayout = {
    version,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map(),
    outputLocations:
      stage === 'FRAGMENT' ? new Map(version === 300 ? [['color', 0]] : [['gl_FragColor', 0]]) : new Map(),
    uses: r.shader.info.uses,
  };
  return stage === 'FRAGMENT'
    ? generateFragmentStage(r.shader.ast, layout).body
    : generateVertexStage(r.shader.ast, layout).body;
}

/** Minimal shared exec-ctx fields (gl/ contract §1 — stores, scratch, blocks). */
function baseCtx(program: Program): Record<string, unknown> {
  return {
    uniforms: program.floatStore,
    intUniforms: program.intStore,
    blockStores: [],
    blockIntStores: [],
    textures: [],
    samplerStates: [],
    scratch: new Float32Array(Math.max(program.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(program.intScratchSize, 16)),
    depthRange: new Float32Array(3),
  };
}

/** Run the fragment stage; returns the location-0 color. */
function runFragment(program: Program): { color: Float32Array } {
  const ctx: Record<string, unknown> = {
    ...baseCtx(program),
    varyings: [],
    fragCoord: new Float32Array(4),
    frontFacing: true,
    pointCoord: new Float32Array(2),
    discarded: false,
    out: { color: [new Float32Array(4)], fragDepth: 0 },
  };
  program.fragment.run(ctx as never);
  return { color: (ctx.out as { color: Float32Array[] }).color[0] };
}

/** Run the vertex stage; returns clip-space position. */
function runVertex(program: Program): { position: Float32Array } {
  const varyingsTotal = program.varyings.reduce((s, v) => s + v.components, 0);
  const ctx: Record<string, unknown> = {
    ...baseCtx(program),
    attribs: [],
    attribIndices: new Int32Array(0),
    vertexId: 0,
    instanceId: 0,
    drawId: 0,
    out: { position: new Float32Array(4), pointSize: 0, varyings: new Float32Array(varyingsTotal) },
  };
  program.vertex.run(ctx as never);
  return { position: (ctx.out as { position: Float32Array }).position };
}

/**
 * Shape invariant: every `| 0` / `>>> 0` in the generated body must be
 * self-parenthesized — the next character is always `)`. Also requires at
 * least one wrap present (so the scan actually exercised the invariant).
 */
function assertAtomicWraps(body: string, label: string): void {
  const re = /\| 0|>>> 0/g;
  let m: RegExpExecArray | null;
  let total = 0;
  let bad = 0;
  while ((m = re.exec(body)) !== null) {
    total++;
    if (body[m.index + m[0].length] !== ')') bad++;
  }
  check(total > 0, `${label}: expected at least one int/uint wrap in the generated body`);
  check(bad === 0, `${label}: ${bad}/${total} wrap(s) NOT atomic (every '| 0'/'>>> 0' must read '((...) | 0)' / '((...) >>> 0)')`);
}

/* ------------------------------------------------------------------ */
/* C1 — the repro: float(int-expr) + float(uint-expr)                  */
/* ------------------------------------------------------------------ */

// (a) The exact repro shape from the report: two COMPUTED int/uint sums
//     converted to float and composed with `+`. 3 + 4 = 7, 1 + 2 = 3 → 10.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  ivec2 a = ivec2(3, 4);
  uvec2 b = uvec2(1, 2);
  float r = float(a.x + a.y) + float(b.x + b.y);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 10, `C1a float(int sum) + float(uint sum) = 10 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C1a body');
}

// (b) Literal-only variant of the repro (float(ivec2(3,4)) takes the FIRST
//     component — GLSL ES scalar ctor rule). No computed wrap is involved, so
//     this is a sanity check for the ctor path, not a precedence gate.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  float r = float(ivec2(3, 4)) + float(uvec2(1, 2));
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 4, `C1b float(ivec2(3,4)) + float(uvec2(1,2)) = 4 (got ${color[0]})`);
}

/* ------------------------------------------------------------------ */
/* C2 — vec2 composition of int/uint conversions                       */
/* ------------------------------------------------------------------ */

// (a) Literal composition vec2(ivec2(3,4)) + vec2(uvec2(1,2)) → (4, 6)
//     (sanity — no computed wraps; passes before and after the fix).
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  vec2 r = vec2(ivec2(3, 4)) + vec2(uvec2(1, 2));
  color = vec4(r, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 4 && color[1] === 6, `C2a vec2(ivec2) + vec2(uvec2) = (4, 6) (got ${color[0]}, ${color[1]})`);
}

// (b) Computed components: (7, 1) + (3, 1) → (10, 2). The vec2(int) ctor
//     leaves the wrapped int strings in place (int→float is a JS no-op), so
//     the float vector add embeds the wraps — the precedence gate.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  ivec2 a = ivec2(3, 4);
  uvec2 b = uvec2(1, 2);
  vec2 r = vec2(a.x + a.y, a.y - a.x) + vec2(b.x + b.y, b.y - b.x);
  color = vec4(r, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 10 && color[1] === 2, `C2b computed vec2 composition = (10, 2) (got ${color[0]}, ${color[1]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C2b body');
}

/* ------------------------------------------------------------------ */
/* C3 — scalar int/uint conversions of computed expressions            */
/* ------------------------------------------------------------------ */

// (a) 1.00 fragment: float(a + b) + float(c + d), all int → 11 + 7 = 18.
{
  const fsSrc = `precision highp float;
void main() {
  int a = 5;
  int b = 6;
  int c = 3;
  int d = 4;
  float r = float(a + b) + float(c + d);
  gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    100,
  );
  const { color } = runFragment(program);
  check(color[0] === 18, `C3a 1.00 float(int+int) + float(int+int) = 18 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 100), 'C3a body');
}

// (b) 1.00 vertex stage — same expression feeding gl_Position.x.
{
  const vsSrc = `precision highp float;
void main() {
  int a = 5;
  int b = 6;
  int c = 3;
  int d = 4;
  float r = float(a + b) + float(c + d);
  gl_Position = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    vsSrc,
    `precision highp float;
void main() { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); }`,
    100,
  );
  const { position } = runVertex(program);
  check(position[0] === 18, `C3b 1.00 vertex float(int+int) + float(int+int) = 18 (got ${position[0]})`);
  assertAtomicWraps(generateBody(vsSrc, 'VERTEX', 100), 'C3b body');
}

// (c) 3.00 fragment: int + uint mix → 11 + 7 = 18.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  int a = 5;
  int b = 6;
  uint c = 3u;
  uint d = 4u;
  float r = float(a + b) + float(c + d);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 18, `C3c 3.00 float(int sum) + float(uint sum) = 18 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C3c body');
}

// (d) 3.00 fragment: uint-only pair → 7 + 11 = 18.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  uint c = 3u;
  uint d = 4u;
  uint e = 5u;
  uint f = 6u;
  float r = float(c + d) + float(e + f);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 18, `C3d 3.00 float(uint+uint) + float(uint+uint) = 18 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C3d body');
}

/* ------------------------------------------------------------------ */
/* C4 — shift results embedded in arithmetic (3.00)                    */
/* ------------------------------------------------------------------ */

// (a) uint >> converted to float and composed with +: float(8 >> 2) + 1 = 3.
//     (Composing inside a uint `+` would re-parenthesize the operand and pass
//     accidentally — the FLOAT add embeds the operand string bare, which is
//     where the wrap's loose binding mis-parses.)
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  uint u = 8u;
  float r = float(u >> 2u) + float(1u);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 3, `C4a float(u >> 2u) + float(1u) = 3 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C4a body');
}

// (b) int << converted to float and composed with +: float(3 << 1) + 2 = 8.
//     (The +1 variant would pass accidentally — (x<<1) is even, so |1 equals
//     +1; use +2.)
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  int x = 3;
  float r = float(x << 1) + float(2);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 8, `C4b float(x << 1) + float(2) = 8 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C4b body');
}

/* ------------------------------------------------------------------ */
/* C5 — bitwise results embedded in arithmetic (3.00)                  */
/* ------------------------------------------------------------------ */

// (a) int &: float(5 & 3) + 1 = 2.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  int a = 5;
  int b = 3;
  int c = 1;
  float r = float(a & b) + float(c);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 2, `C5a float(a & b) + float(c) = 2 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C5a body');
}

// (b) uint &: float(6 & 3) + 2 = 4.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  uint a = 6u;
  uint b = 3u;
  uint c = 2u;
  float r = float(a & b) + float(c);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 4, `C5b float(a & b) + float(c) uint = 4 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C5b body');
}

// (c) int ^: float(6 ^ 3) + 1 = 6.
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  int a = 6;
  int b = 3;
  int c = 1;
  float r = float(a ^ b) + float(c);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 6, `C5c float(a ^ b) + float(c) = 6 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C5c body');
}

/* ------------------------------------------------------------------ */
/* C6 — vector-uint local, dynamic component index (env.ts ensureDynScratch) */
/* ------------------------------------------------------------------ */

// Dynamic WRITE (spills the flat local to ctx.intScratch and copies back with
// the `>>> 0` wrap) + dynamic READS composed with `+`: v[1] = 9, then
// v[0] + v[1] = 3 + 9 = 12. The body scan pins the copy-out wrap atomicity
// (`v__0 = ((ctx.intScratch[...]) >>> 0)` — the pre-fix `... >>> 0;` shape
// violates the invariant).
{
  const fsSrc = `#version 300 es
precision highp float;
out vec4 color;
void main() {
  uvec2 v = uvec2(3u, 4u);
  int i = 1;
  v[i] = 9u;
  float r = float(v[0] + v[1]);
  color = vec4(r, 0.0, 0.0, 1.0);
}`;
  const program = linkPair(
    `#version 300 es
precision highp float;
void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }`,
    fsSrc,
    300,
  );
  const { color } = runFragment(program);
  check(color[0] === 12, `C6 dynamic-indexed uint vec local = 12 (got ${color[0]})`);
  assertAtomicWraps(generateBody(fsSrc, 'FRAGMENT', 300), 'C6 body');
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-parens: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
