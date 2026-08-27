/**
 * selftest-ctor-fixes.ts — regression checks for two codegen bugs.
 *
 * (1) Matrix arguments in VECTOR constructors (expr-ctor.ts emitVectorCtor):
 *     GLSL ES 1.00 §5.4.2 flattens a matrix argument COLUMN-MAJOR
 *     (vec4(mat2) = [m00, m10, m01, m11]); extra components truncate, short
 *     lists pad (0s + final 1). The old code fed the DIAGONAL and threw
 *     "matrix argument in a multi-argument vector constructor" on any
 *     multi-arg ctor containing a matrix. A matrix arg is legal as the SOLE
 *     argument or as the LAST argument of a multi-arg ctor (vec4(scalar,
 *     mat2) valid; vec4(mat2, scalar) invalid — semantics rejects the latter
 *     via the component-count rule in analyzeVectorConstructor, so the
 *     codegen throw is defensive).
 * (2) User struct constructor calls (vertex.ts/fragment.ts): the stage
 *     entries never seeded env.structNames from CodegenLayout.structNames, so
 *     `Foo(...)` fell through to builtin resolution and threw
 *     "codegen: no builtin signature for 'Foo'". Both stage entries now seed
 *     the set from the layout.
 *
 * Each case compiles a real shader (compileShader → annotated AST), emits
 * `main` through generateVertexStage / generateFragmentStage (the ACTUAL
 * stage entries — not direct env driving), compiles the returned body via
 * new Function('ctx','R', body), runs it against a hand-built ctx, and
 * asserts the runtime value. Every value check FAILS on the pre-fix code
 * (wrong diagonal components, or "matrix argument in a multi-argument vector
 * constructor" / "no builtin signature for 'Foo'" crashes).
 *
 * Run: npx tsx src/glsl/codegen/selftest-ctor-fixes.ts
 *
 * Prints "selftest-ctor-fixes: N checks, M failure(s)" and exits 0 only
 * when all pass.
 */
import { compileShader } from '../compiler.js';
import { generateVertexStage, generateFragmentStage, R } from './index.js';
import type { CodegenLayout, UniformSlot } from './index.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/** Run a check body; a thrown crash counts as one failed check (pre-fix the
 *  stage entries crash — capture the message instead of aborting). */
function tryCheck(prefix: string, body: () => void): void {
  try {
    body();
  } catch (e) {
    check(false, `${prefix}: crashed with ${(e as Error).message}`);
  }
}

function baseLayout(version: 100 | 300): CodegenLayout {
  return {
    version,
    uniformSlots: new Map(),
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: new Map(),
    attribLocations: new Map(),
    outputLocations: new Map([['gl_FragColor', 0], ['color', 0]]),
    uses: {
      pointSize: false, fragCoord: false, frontFacing: false, pointCoord: false,
      fragDepth: false, vertexId: false, instanceId: false, derivatives: false,
      drawId: false,
    },
  };
}

interface RunOpts {
  structNames?: string[];
  uniformSlots?: Record<string, UniformSlot>;
  /** Values written into ctx.uniforms at explicit float indices (before the
   *  body runs). Matrix uniforms live with a column stride of 4 floats:
   *  mat2 at slot 0 → m00=u[0], m01=u[1], m10=u[4], m11=u[5]. */
  uniforms?: Record<number, number>;
}

/** Compile + generate the stage body via the REAL stage entry + run; returns
 *  the body and the executed ctx. */
function runMain(
  src: string,
  stage: 'VERTEX' | 'FRAGMENT',
  version: 100 | 300,
  opts: RunOpts = {},
): { body: string; ctx: Record<string, any> } {
  const r = compileShader(src, { type: stage, version });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  const layout = baseLayout(version);
  if (opts.uniformSlots) {
    for (const [k, v] of Object.entries(opts.uniformSlots)) layout.uniformSlots.set(k, v);
  }
  if (opts.structNames) layout.structNames = [...opts.structNames];
  const res = stage === 'VERTEX'
    ? generateVertexStage(r.shader.ast, layout)
    : generateFragmentStage(r.shader.ast, layout);
  const fn = new Function('ctx', 'R', res.body);
  const ctx: Record<string, any> = {
    discarded: false,
    out: stage === 'VERTEX' ? { position: [0, 0, 0, 0], pointSize: 0 } : { color: [[0, 0, 0, 0]] },
    uniforms: new Float32Array(64),
    intUniforms: new Int32Array(64),
    scratch: new Float32Array(Math.max(res.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(res.intScratchSize, 16)),
    fragCoord: [0, 0, 0, 1],
    attribs: [],
    attribIndices: [],
  };
  if (opts.uniforms) {
    for (const [k, v] of Object.entries(opts.uniforms)) ctx.uniforms[+k] = v;
  }
  fn(ctx, R);
  return { body: res.body, ctx };
}

const mUniform = { m: { store: 'float', slot: 0, stride: 0 } as UniformSlot };

/* ------------------------------------------------------------------ */
/* ITEM 1 — matrix argument in vector constructors                     */
/* ------------------------------------------------------------------ */

// (1a) vec4(m): column-major flatten [m00,m10,m01,m11] = [1,2,3,4] (pre-fix
// diagonal gave [1,3,0,1]).
tryCheck('vec4(m)', () => {
  const { ctx } = runMain(
    `precision mediump float;
uniform mat2 m;
void main() {
  gl_FragColor = vec4(m);
}`,
    'FRAGMENT',
    100,
    { uniformSlots: mUniform, uniforms: { 0: 1, 1: 2, 4: 3, 5: 4 } },
  );
  const c = ctx.out.color[0];
  check(c[0] === 1 && c[1] === 2 && c[2] === 3 && c[3] === 4, `vec4(m) → [1,2,3,4] (got [${c.join(',')}])`);
});

// (1b) vec4(1.0, m): matrix as the LAST argument of a multi-arg ctor:
// [1.0] + [1,2,3,4] truncated → [1,1,2,3]. Pre-fix this threw.
tryCheck('vec4(1.0, m)', () => {
  const { ctx } = runMain(
    `precision mediump float;
uniform mat2 m;
void main() {
  gl_FragColor = vec4(1.0, m);
}`,
    'FRAGMENT',
    100,
    { uniformSlots: mUniform, uniforms: { 0: 1, 1: 2, 4: 3, 5: 4 } },
  );
  const c = ctx.out.color[0];
  check(c[0] === 1 && c[1] === 1 && c[2] === 2 && c[3] === 3, `vec4(1.0, m) → [1,1,2,3] (got [${c.join(',')}])`);
});

// (1c) vec4(vec2(1.0,2.0), m): vector + matrix → [1,2] + [1,2,3,4] → [1,2,1,2].
tryCheck('vec4(vec2(1.0,2.0), m)', () => {
  const { ctx } = runMain(
    `precision mediump float;
uniform mat2 m;
void main() {
  gl_FragColor = vec4(vec2(1.0, 2.0), m);
}`,
    'FRAGMENT',
    100,
    { uniformSlots: mUniform, uniforms: { 0: 1, 1: 2, 4: 3, 5: 4 } },
  );
  const c = ctx.out.color[0];
  check(c[0] === 1 && c[1] === 2 && c[2] === 1 && c[3] === 2, `vec4(vec2(1.0,2.0), m) → [1,2,1,2] (got [${c.join(',')}])`);
});

// (1d) vec2(m): truncation → [1,2] (pre-fix diagonal [1,3]).
tryCheck('vec2(m)', () => {
  const { ctx } = runMain(
    `precision mediump float;
uniform mat2 m;
void main() {
  gl_FragColor = vec4(vec2(m), 0.0, 0.0);
}`,
    'FRAGMENT',
    100,
    { uniformSlots: mUniform, uniforms: { 0: 1, 1: 2, 4: 3, 5: 4 } },
  );
  const c = ctx.out.color[0];
  check(c[0] === 1 && c[1] === 2 && c[2] === 0 && c[3] === 0, `vec2(m) → [1,2] (got [${c.join(',')}])`);
});

// (1e) vec3(m): truncation → [1,2,3].
tryCheck('vec3(m)', () => {
  const { ctx } = runMain(
    `precision mediump float;
uniform mat2 m;
void main() {
  gl_FragColor = vec4(vec3(m), 0.0);
}`,
    'FRAGMENT',
    100,
    { uniformSlots: mUniform, uniforms: { 0: 1, 1: 2, 4: 3, 5: 4 } },
  );
  const c = ctx.out.color[0];
  check(c[0] === 1 && c[1] === 2 && c[2] === 3 && c[3] === 0, `vec3(m) → [1,2,3] (got [${c.join(',')}])`);
});

// (1f) mat2x2 spelling (GLSL ES 3.00 only — mat2x2 lexes as an identifier in
// 1.00): same column-major flatten.
tryCheck('vec4(mat2x2)', () => {
  const { ctx } = runMain(
    `#version 300 es
precision mediump float;
uniform mat2x2 m;
out vec4 color;
void main() {
  color = vec4(m);
}`,
    'FRAGMENT',
    300,
    { uniformSlots: mUniform, uniforms: { 0: 1, 1: 2, 4: 3, 5: 4 } },
  );
  const c = ctx.out.color[0];
  check(c[0] === 1 && c[1] === 2 && c[2] === 3 && c[3] === 4, `vec4(mat2x2 m) → [1,2,3,4] (got [${c.join(',')}])`);
});

// (1g) Non-last matrix (vec4(m, 1.0)): INVALID. Semantics rejects it at
// compile time (analyzeVectorConstructor's component-count rule — a matrix
// contributes ≥4 components so the prefix alone fills an n≤4 vector; the
// message is the generic 'too many arguments'). The codegen throw
// ('matrix must be the last argument') is defensive — unreachable via
// compileShader. Accept either surface.
{
  const r = compileShader(
    `precision mediump float;
uniform mat2 m;
void main() {
  gl_FragColor = vec4(m, 1.0);
}`,
    { type: 'FRAGMENT', version: 100 },
  );
  let got: string;
  if (!r.ok) {
    got = r.errors.map((er) => er.message).join('; ');
  } else {
    try {
      generateFragmentStage(r.shader.ast, baseLayout(100));
      got = 'no error';
    } catch (e) {
      got = (e as Error).message;
    }
  }
  check(
    /matrix must be the last argument|too many arguments/.test(got),
    `vec4(m, 1.0) rejected (got: ${got})`,
  );
}

/* ------------------------------------------------------------------ */
/* ITEM 2 — user struct constructor calls through the stage entries    */
/* ------------------------------------------------------------------ */

// (2a) Vertex stage: struct ctor + member reads → position [1,2,3,1].
tryCheck('vertex struct ctor', () => {
  const { ctx } = runMain(
    `struct Foo { vec2 a; float b; };
void main() {
  Foo f = Foo(vec2(1.0, 2.0), 3.0);
  gl_Position = vec4(f.a, f.b, 1.0);
}`,
    'VERTEX',
    100,
    { structNames: ['Foo'] },
  );
  const p = ctx.out.position;
  check(p[0] === 1 && p[1] === 2 && p[2] === 3 && p[3] === 1, `vertex Foo(...) → [1,2,3,1] (got [${p.join(',')}])`);
});

// (2b) Fragment stage: same via gl_FragColor.
tryCheck('fragment struct ctor', () => {
  const { ctx } = runMain(
    `precision mediump float;
struct Foo { vec2 a; float b; };
void main() {
  Foo f = Foo(vec2(1.0, 2.0), 3.0);
  gl_FragColor = vec4(f.a, f.b, 1.0);
}`,
    'FRAGMENT',
    100,
    { structNames: ['Foo'] },
  );
  const c = ctx.out.color[0];
  check(c[0] === 1 && c[1] === 2 && c[2] === 3 && c[3] === 1, `fragment Foo(...) → [1,2,3,1] (got [${c.join(',')}])`);
});

// (2c) Struct-argument-in-struct-ctor: Outer(Inner(1.0, vec2(2.0,3.0)), 4.0)
// flattens to [1,2,3,4] — observed via vec4(o.inner.x, o.inner.y, o.w)
// (exactly 4 components: 1 + 2 + 1).
tryCheck('nested struct ctor', () => {
  const { ctx } = runMain(
    `struct Inner { float x; vec2 y; };
struct Outer { Inner inner; float w; };
void main() {
  Outer o = Outer(Inner(1.0, vec2(2.0, 3.0)), 4.0);
  gl_Position = vec4(o.inner.x, o.inner.y, o.w);
}`,
    'VERTEX',
    100,
    { structNames: ['Inner', 'Outer'] },
  );
  const p = ctx.out.position;
  check(p[0] === 1 && p[1] === 2 && p[2] === 3 && p[3] === 4, `nested Outer(...) → [1,2,3,4] (got [${p.join(',')}])`);
});

// (2d) EMPTY layout (no structNames): the ctor call must FAIL — documents the
// dependency of struct ctor resolution on layout.structNames.
{
  let got = 'no error';
  try {
    runMain(
      `struct Foo { vec2 a; float b; };
void main() {
  Foo f = Foo(vec2(1.0, 2.0), 3.0);
  gl_Position = vec4(f.a, f.b, 1.0);
}`,
      'VERTEX',
      100,
    );
  } catch (e) {
    got = (e as Error).message;
  }
  check(/Foo/.test(got), `struct ctor without structNames fails (got: ${got})`);
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`selftest-ctor-fixes: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
