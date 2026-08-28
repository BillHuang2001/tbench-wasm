/**
 * selftest-overload-globals.ts — regression selftest for two codegen root
 * causes in the ogles functions cluster (codegen/CONTEXT.md BUG A + BUG B):
 *
 *   A. OVERLOADED USER FUNCTIONS: installUserFunctions keyed its registry by
 *      NAME only, so overloads overwrote each other and every call inlined the
 *      LAST definition — `Cannot read properties of undefined (reading 'v')`
 *      at functions.ts:232 when the call's component count differed from the
 *      wrong body's params, and a SILENT MISCOMPILE when counts aligned.
 *   B. USER GLOBAL VARIABLES: `float gray = 0.0;` at file scope is legal
 *      GLSL ES 1.00 (constant initializer); semantics accepts it but codegen
 *      had no storage surface for non-const globals →
 *      `codegen: unknown identifier 'gray'` at link time.
 *
 * All tests drive the REAL pipeline (compileShader → linkProgram → run the
 * linked stages with hand-built exec ctxs) and assert observable behavior
 * (link success + executed values). Link failures are recorded as FAILED
 * checks (the pre-fix code fails to link these shaders — the tests must
 * observe that, not crash on it).
 *
 * Run: npx tsx src/glsl/codegen/selftest-overload-globals.ts
 * Prints "OK" and exits 0 on success; non-zero exit on failure.
 */
import { compileShader, linkProgram } from '../compiler.js';
import type { Program } from '../program.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

function near(got: number, expected: number, msg: string): void {
  check(Math.abs(got - expected) < 1e-6, `${msg} (got ${got}, expected ${expected})`);
}

/* ------------------------------------------------------------------ */
/* Helpers (ctx shapes per program.ts — same as selftest-integration)  */
/* ------------------------------------------------------------------ */

function compile(src: string, type: 'VERTEX' | 'FRAGMENT', version: 100 | 300, extensions?: string[]) {
  const r = compileShader(src, { type, version, extensions: extensions ? new Set(extensions) : undefined });
  if (!r.ok) throw new Error(`compile failed: ${JSON.stringify(r.errors)}`);
  return r.shader;
}

const TRIVIAL_FS = `precision mediump float;
void main() { gl_FragColor = vec4(0.0); }`;

/** Compile+link a VERTEX shader under test against the trivial fragment.
 *  A link failure is recorded as a FAILED check and null is returned. */
function linkVs(src: string, tag: string): Program | null {
  const vs = compile(src, 'VERTEX', 100);
  const fs = compile(TRIVIAL_FS, 'FRAGMENT', 100);
  const l = linkProgram(vs, fs);
  if (!l.ok) {
    check(false, `${tag}: link failed (${l.log.split('\n')[0].slice(0, 90)})`);
    return null;
  }
  return l.program;
}

/** Compile+link a FRAGMENT shader under test against a trivial vertex. */
function linkFs(src: string, tag: string, extensions?: string[]): Program | null {
  const vs = compile(`void main() { gl_Position = vec4(0.0); }`, 'VERTEX', 100);
  const fs = compile(src, 'FRAGMENT', 100, extensions);
  const l = linkProgram(vs, fs);
  if (!l.ok) {
    check(false, `${tag}: link failed (${l.log.split('\n')[0].slice(0, 90)})`);
    return null;
  }
  return l.program;
}

function vertexCtx(prog: Program): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { position: new Float32Array(4), pointSize: 0, varyings: new Float32Array(32) },
    attribs: [],
    attribIndices: new Int32Array(16),
    vertexId: 0,
    instanceId: 0,
  };
}

function fragmentCtx(prog: Program, extra: Record<string, unknown> = {}): any {
  return {
    uniforms: prog.floatStore,
    intUniforms: prog.intStore,
    blockStores: [],
    blockIntStores: [],
    scratch: new Float32Array(Math.max(prog.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(prog.intScratchSize, 16)),
    out: { color: [new Float32Array(4)], fragDepth: 0 },
    discarded: false,
    fragCoord: new Float32Array([0, 0, 0, 1]),
    frontFacing: true,
    pointCoord: new Float32Array([0, 0]),
    varyings: [],
    ...extra,
  };
}

/** Link then, when the link succeeded, run the vertex stage and check one
 *  value against the expectation. */
function runVsCheck(src: string, tag: string, expected: number, msg: string): void {
  const prog = linkVs(src, tag);
  if (!prog) return;
  const ctx = vertexCtx(prog);
  prog.vertex.run(ctx);
  near(ctx.out.position[0], expected, msg);
}

/* ================================================================== */
/* A. Overloaded user functions                                        */
/* ================================================================== */

/* A1. call targets the NON-LAST definition (1 comp vs 2 comp params) —
 *     crashed with undefined-'.v' at HEAD (wrong body inlined). */
runVsCheck(
  `float f(float x) { return 1.0; }
   float f(vec2 x) { return 2.0; }
   void main() { gl_Position.x = f(1.0); }`,
  'overload A1',
  1.0,
  'overload A1: f(1.0) runs the SCALAR body',
);

/* A2. MATCHING component counts — silently miscompiled at HEAD: semantics
 *     resolved `f(ivec2)` to the ivec2 overload (exact), codegen inlined the
 *     LAST-defined vec2 body (2 comps fit 2 temps) → returned 2.0 instead of 1. */
runVsCheck(
  `int f(ivec2 x) { return 1; }
   float f(vec2 y) { return 2.0; }
   void main() { gl_Position.x = float(f(ivec2(1, 2))); }`,
  'overload A2',
  1.0,
  'overload A2: f(ivec2) runs the IVEC2 body',
);

/* A3. ogles is_all shape: scalar overload + array overload (array defined
 *     LAST), BOTH call kinds present — the scalar call must inline the
 *     scalar body, the array call the array body. */
runVsCheck(
  `bool is_all(bvec4 par, bool value) {
     return par[0] == value && par[1] == value && par[2] == value && par[3] == value;
   }
   bool is_all(bvec4 array[3], bvec4 value) {
     return array[0] == value && array[1] == value && array[2] == value;
   }
   void main() {
     bvec4 par[3];
     par[0] = bvec4(true, true, true, true);
     par[1] = bvec4(true, true, true, true);
     par[2] = bvec4(true, true, true, true);
     bvec4 ret = bvec4(true, true, true, true);
     bool r = is_all(par, bvec4(true, true, true, true)); // array overload
     bool s = is_all(ret, true);                          // scalar overload
     gl_Position.x = (r && s) ? 1.0 : 0.0;
   }`,
  'overload A3',
  1.0,
  'overload A3: is_all array + scalar overloads',
);

/* ================================================================== */
/* B. User global variables                                            */
/* ================================================================== */

/* B1. `float gray = 0.0;` read in main (link failed with
 *     `codegen: unknown identifier 'gray'` at HEAD). */
runVsCheck(
  `float gray = 0.0;
   void main() { gl_Position.x = gray + 1.0; }`,
  'global B1',
  1.0,
  'global B1: gray initialized to 0.0',
);

/* B2. uninitialized global `float g2;` reads as 0. */
runVsCheck(
  `float g2;
   void main() { gl_Position.x = g2 + 1.0; }`,
  'global B2',
  1.0,
  'global B2: uninitialized g2 reads 0.0',
);

/* B3. write-to-global through an INLINED function (the ogles
 *     void_empty_empty_void_empty shape: function() sets gray = 1.0). */
runVsCheck(
  `float gray = 0.0;
   void function() { gray = 1.0; }
   void main() {
     gray = 0.0;
     function();
     gl_Position.x = gray;
   }`,
  'global B3',
  1.0,
  'global B3: inlined fn write-back',
);

/* B4. int global (int store) + vector global + init runs per invocation. */
runVsCheck(
  `int g = 3;
   vec2 gv = vec2(0.5, 0.25);
   void main() {
     g = g + 1;
     gl_Position.x = float(g) + gv.x;
   }`,
  'global B4',
  4.5,
  'global B4: int + vec2 globals',
);

/* B5. fragment stage: global read in dual mode (dFdx of a constant global
 *     must be 0; the global's v-plane init must land in the color). */
{
  const prog = linkFs(
    `#extension GL_OES_standard_derivatives : enable
     precision mediump float;
     float gray = 0.25;
     void main() { gl_FragColor = vec4(gray + dFdx(gray), 0.0, 0.0, 1.0); }`,
    'global B5',
    ['GL_OES_standard_derivatives'],
  );
  if (prog) {
    const ctx = fragmentCtx(prog);
    prog.fragment.run(ctx);
    near(ctx.out.color[0][0], 0.25, `global B5: fragment dual-mode global (got ${ctx.out.color[0][0]})`);
  }
}

/* B6. callee-body free name matching a CALLER PARAM resolves to the GLOBAL
 *     (the exact in-parameter-passed-as-inout-argument-and-global shape: G's
 *     `p += q` must accumulate into the global p, NOT into F's param p).
 *     OLD BEHAVIOR (pinned by the failing CTS page): resolveLocal walked ALL
 *     active param frames while emitting the inlined callee BODY, so G's `p`
 *     resolved to F's param p — the write-back landed in F's discarded param
 *     and the global stayed (0,0,0) → black frame. New rule: a callee body
 *     sees only its own frame + globals; caller frames are invisible. */
{
  const prog = linkVs(
    `vec3 p;
     void G(inout vec3 q) { p += q; }
     void F(in vec3 p) { G(p); }
     void main() {
       p = vec3(0.0);
       F(vec3(0.0, 1.0, 0.0));
       gl_Position = vec4(p, 1.0);
     }`,
    'global B6',
  );
  if (prog) {
    const ctx = vertexCtx(prog);
    prog.vertex.run(ctx);
    near(
      ctx.out.position[1],
      1.0,
      `global B6: callee free name vs caller PARAM → global (got y=${ctx.out.position[1]})`,
    );
  }
}

/* B7. same, but the caller's shadow is a LOCAL (frame.locals path) — the
 *     callee body must still land on the global. */
{
  const prog = linkVs(
    `vec3 p;
     void G(inout vec3 q) { p += q; }
     void F(in vec3 a) { vec3 p = a; G(p); }
     void main() {
       p = vec3(0.0);
       F(vec3(0.0, 1.0, 0.0));
       gl_Position = vec4(p, 1.0);
     }`,
    'global B7',
  );
  if (prog) {
    const ctx = vertexCtx(prog);
    prog.vertex.run(ctx);
    near(
      ctx.out.position[1],
      1.0,
      `global B7: callee free name vs caller LOCAL → global (got y=${ctx.out.position[1]})`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`overload-globals selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
