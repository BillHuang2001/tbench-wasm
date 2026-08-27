/**
 * selftest-const.ts — const-expression folding battery (BUGS 2+5).
 *
 * Covers the const-expression evaluator (semantics-const.ts) end-to-end:
 *  - ACCEPTANCE: const initializers that must now compile — the CTS boilerplate
 *    `const vec4 green = vec4(0.0, 1.0, 0.0, 1.0);` (BUG 2), vector/matrix
 *    constructor inits, struct + struct-member reads, comma folding, nested
 *    ctors, conversions (float→int / int→bool ctor forms), splat, ES 3.00
 *    array ctors, binary ops on consts (incl. true matrix products) and
 *    chained member reads (`s.y.x`);
 *  - NEGATIVES: non-constant leaves (uniforms, user-function calls, non-const
 *    operands inside ctors) are rejected with the exact const-initializer
 *    error; strict implicit-conversion context is preserved (`const float f =
 *    1;` still fails — ctors are the only conversion mechanism);
 *  - RUNTIME: link + run (makeVertexCtx/makeFragmentCtx shapes from
 *    tests/unit/glsl.test.ts) — global-const reads (whole, member, matrix
 *    index) emit annotated ctor calls and produce the expected colors; one
 *    dual-mode run (dFdx on a const — constant duals must not crash).
 *
 * Run: npx tsx src/glsl/selftest-const.ts   (prints "OK", exit 0)
 */
import { compileShader, linkProgram } from './compiler.js';
import type { CompileError } from './compiler.js';
import type { Program, VertexExecCtx, FragmentExecCtx } from './program.js';

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

function comp(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT', exts?: string[]) {
  return compileShader(src, { type, version, extensions: exts !== undefined ? new Set(exts) : undefined });
}

/** Assert success and return the shader (for linking). */
function okShader(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT', exts?: string[]) {
  const r = comp(src, version, type, exts);
  if (!r.ok) {
    check(false, `expected success: ${JSON.stringify(src.slice(0, 70))} — got ${JSON.stringify(r.errors)}`);
    return null;
  }
  return r.shader;
}

/** Assert failure and return the errors. */
function errs(src: string, version: 100 | 300, type: 'VERTEX' | 'FRAGMENT', exts?: string[]): CompileError[] {
  const r = comp(src, version, type, exts);
  if (r.ok) {
    check(false, `expected failure: ${JSON.stringify(src.slice(0, 70))}`);
    return [];
  }
  return r.errors;
}

/** Assert that an error with the exact message exists at the exact line. */
function hasErr(errors: CompileError[], line: number, msg: string): boolean {
  return errors.some((e) => e.line === line && e.message === msg);
}

/* ------------------------------------------------------------------ */
/* Runtime exec-context shapes (mirror tests/unit/glsl.test.ts)        */
/* ------------------------------------------------------------------ */

function totalVaryingComponents(program: Program): number {
  return program.varyings.reduce((n, v) => n + v.components, 0);
}

function makeVertexCtx(program: Program, attribs: (Float32Array | number)[]): VertexExecCtx {
  return {
    attribs,
    attribIndices: new Int32Array(attribs.length),
    uniforms: program.floatStore,
    intUniforms: program.intStore,
    blockStores: [],
    blockIntStores: [],
    textures: [],
    samplerStates: [],
    scratch: new Float32Array(Math.max(program.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(program.intScratchSize, 16)),
    vertexId: 0,
    instanceId: 0,
    out: {
      position: new Float32Array(4),
      pointSize: 1,
      varyings: new Float32Array(totalVaryingComponents(program)),
    },
  };
}

function makeFragmentCtx(program: Program): FragmentExecCtx {
  return {
    uniforms: program.floatStore,
    intUniforms: program.intStore,
    blockStores: [],
    blockIntStores: [],
    textures: [],
    samplerStates: [],
    scratch: new Float32Array(Math.max(program.scratchSize, 16)),
    intScratch: new Int32Array(Math.max(program.intScratchSize, 16)),
    varyings: program.varyings.map((v) => ({ v: new Float32Array(v.components) })),
    fragCoord: new Float32Array([0, 0, 0, 1]),
    frontFacing: true,
    pointCoord: new Float32Array(2),
    discarded: false,
    out: {
      color: program.fragment.outputs.map(() => new Float32Array(4)),
      fragDepth: 0,
    },
  };
}

/** Compile VS+FS, link, run one vertex + one fragment invocation, return color. */
function runFragment(
  vsSrc: string,
  fsSrc: string,
  exts?: string[],
): Float32Array | null {
  const vs = okShader(vsSrc, 100, 'VERTEX');
  const fs = okShader(fsSrc, 100, 'FRAGMENT', exts);
  if (vs === null || fs === null) return null;
  const res = linkProgram(vs, fs);
  if (!res.ok) {
    check(false, `link failed: ${res.log}`);
    return null;
  }
  const program = res.program;
  program.vertex.run(makeVertexCtx(program, [new Float32Array([0, 0, 0, 1])]));
  const fctx = makeFragmentCtx(program);
  program.fragment.run(fctx);
  return fctx.discarded ? null : fctx.out.color[0];
}

function close4(c: Float32Array | null, expected: number[]): boolean {
  if (c === null) return false;
  for (let i = 0; i < 4; i++) {
    if (Math.abs(c[i] - expected[i]) > 1e-6) return false;
  }
  return true;
}

const VERT_SIMPLE = 'attribute vec4 a_position;\nvoid main() { gl_Position = a_position; }';

/* ------------------------------------------------------------------ */
/* 1. BUG 2: CTS boilerplate const vec4 + use                          */
/* ------------------------------------------------------------------ */

{
  // The exact CTS constructors-page boilerplate: global const vec4 with a
  // vector-ctor initializer, then read whole (vertex + fragment).
  const v1 = okShader('const vec4 green = vec4(0.0, 1.0, 0.0, 1.0);\nvoid main() { gl_Position = green; }', 100, 'VERTEX');
  check(v1 !== null, 'CTS boilerplate const vec4 ctor init + whole read (vertex)');
  const f1 = okShader('precision mediump float;\nconst vec4 green = vec4(0.0, 1.0, 0.0, 1.0);\nvoid main() { gl_FragColor = green; }', 100, 'FRAGMENT');
  check(f1 !== null, 'CTS boilerplate const vec4 ctor init + whole read (fragment)');
  // Local variant (codegen emits the decl as a JS local — acceptance only).
  const f1b = okShader('precision mediump float;\nvoid main() { const vec4 green = vec4(0.0, 1.0, 0.0, 1.0); gl_FragColor = green; }', 100, 'FRAGMENT');
  check(f1b !== null, 'LOCAL const vec4 ctor init + whole read (fragment)');

  // Member reads of the const: scalar fold and swizzle slice.
  const v2 = okShader('const vec4 green = vec4(0.0, 1.0, 0.0, 1.0);\nconst float r = green.r;\nvoid main() { gl_Position = vec4(r); }', 100, 'VERTEX');
  check(v2 !== null, 'const float r = green.r (scalar member fold)');
  const v3 = okShader('const vec4 green = vec4(0.0, 1.0, 0.0, 1.0);\nvoid main() { gl_Position = vec4(vec2(green.rg), 0.0, 1.0); }', 100, 'VERTEX');
  check(v3 !== null, 'vec2(green.rg) (swizzle slice → ctor mutation)');
}

/* ------------------------------------------------------------------ */
/* 2. BUG 5: matrix consts, comma, structs, nested ctors               */
/* ------------------------------------------------------------------ */

{
  const m1 = okShader('const mat2 m = mat2(1.0, 2.0, 3.0, 4.0);\nvoid main() { gl_Position = vec4(m[0], m[1]); }', 100, 'VERTEX');
  check(m1 !== null, 'const mat2 m = mat2(1,2,3,4); vec4(m[0], m[1]) (matrix ctor + column index)');
  const m2 = okShader('const mat3 mm = mat3(2.0);\nvoid main() { gl_Position = vec4(mm[0], 1.0); }', 100, 'VERTEX');
  check(m2 !== null, 'const mat3 mm = mat3(2.0) (scalar-diagonal matrix ctor)');
  const m3 = okShader('const mat2 a = mat2(1.0, 2.0, 3.0, 4.0);\nconst mat2 b = mat2(1.0, 0.0, 0.0, 1.0);\nconst mat2 p = a * b;\nvoid main() { gl_Position = vec4(p[0], 0.0, 1.0); }', 100, 'VERTEX');
  check(m3 !== null, 'const mat2 p = a * b (true matrix product of consts)');

  const s1 = okShader('struct S { float x; vec2 y; };\nconst S s = S(1.0, vec2(2.0, 3.0));\nconst float f = s.x;\nvoid main() { gl_Position = vec4(f); }', 100, 'VERTEX');
  check(s1 !== null, 'const S s = S(1.0, vec2(2,3)); const float f = s.x (struct ctor + member read)');
  const s2 = okShader('struct S { float x; vec2 y; };\nconst S s = S(1.0, vec2(2.0, 3.0));\nconst float f = s.y.x;\nvoid main() { gl_Position = vec4(f); }', 100, 'VERTEX');
  check(s2 !== null, 'const float f = s.y.x (chained member read)');

  const c1 = okShader('const vec4 v = (vec4(1.0), vec4(2.0));\nvoid main() { gl_Position = v; }', 100, 'VERTEX');
  check(c1 !== null, 'const vec4 v = (vec4(1.0), vec4(2.0)) (comma folds to last operand)');
  const c2 = okShader('const float x = (1.0, 2.0);\nvoid main() { gl_Position = vec4(x); }', 100, 'VERTEX');
  check(c2 !== null, 'const float x = (1.0, 2.0) (scalar comma)');
  const c3 = okShader('const vec4 v = (vec4(1.0), vec4(2.0));\nvoid main() { gl_Position = v + vec4(1.0); }', 100, 'VERTEX');
  check(c3 !== null, 'comma const used in a binary op');

  const n1 = okShader('const vec4 v = vec4(vec2(1.0, 2.0), vec2(3.0, 4.0));\nvoid main() { gl_Position = v; }', 100, 'VERTEX');
  check(n1 !== null, 'nested ctor vec4(vec2, vec2)');
  const n2 = okShader('const ivec4 iv = ivec4(1.0, 2.0, 3.0, 4.0);\nconst bvec4 bv = bvec4(iv);\nvoid main() { gl_Position = vec4(bv); }', 100, 'VERTEX');
  check(n2 !== null, 'conversions: ivec4(float…) and bvec4(ivec4) const ctors');
  const n3 = okShader('const vec4 s = vec4(7.0);\nvoid main() { gl_Position = s; }', 100, 'VERTEX');
  check(n3 !== null, 'splat const vec4(7.0)');

  const b1 = okShader('const float x = 1.0 + 2.0 * 3.0;\nconst int y = (1 + 2) * 3;\nconst bool nb = !true;\nconst float t = true ? 1.0 : 2.0;\nvoid main() { gl_Position = vec4(x + float(y) + float(t)); }', 100, 'VERTEX');
  check(b1 !== null, 'scalar const binary/unary/ternary folding');
  const b2 = okShader('const vec2 a = vec2(1.0, 2.0);\nconst vec2 b = a + vec2(1.0);\nvoid main() { gl_Position = vec4(b, 0.0, 1.0); }', 100, 'VERTEX');
  check(b2 !== null, 'const vector binary op (component-wise)');

  // ES 3.00 array consts.
  const a1 = okShader('#version 300 es\nprecision mediump float;\nconst float a[2] = float[2](1.0, 2.0);\nout vec4 o;\nvoid main() { o = vec4(a[0], a[1], 0.0, 1.0); }', 300, 'FRAGMENT');
  check(a1 !== null, 'ES 3.00 const float a[2] = float[2](1.0, 2.0); a[0]/a[1] reads');
  const a2 = okShader('#version 300 es\nprecision mediump float;\nconst uint u = 0u - 1u;\nout vec4 o;\nvoid main() { o = vec4(float(u)); }', 300, 'FRAGMENT');
  check(a2 !== null, 'ES 3.00 const uint wrap (0u - 1u)');

  // Const-qualified params never fold (no initializer → no constData) but
  // still compile and are usable.
  const p1 = okShader('float f(const float p) { return p + 1.0; }\nvoid main() { gl_Position = vec4(f(1.0)); }', 100, 'VERTEX');
  check(p1 !== null, 'const-qualified param usable (never folded)');
}

/* ------------------------------------------------------------------ */
/* 3. Negatives: non-constant leaves                                    */
/* ------------------------------------------------------------------ */

{
  const e1 = errs('uniform vec4 uX;\nconst vec4 c = vec4(uX);\nvoid main() { gl_Position = c; }', 100, 'VERTEX');
  check(hasErr(e1, 2, "'c' : initializer of const variable must be a constant expression"), 'vec4(uniform) init → const error line 2');
  const e2 = errs('uniform float u;\nconst float x = u + 1.0;\nvoid main() { gl_Position = vec4(x); }', 100, 'VERTEX');
  check(hasErr(e2, 2, "'x' : initializer of const variable must be a constant expression"), 'non-const binary operand → const error');
  const e3 = errs('int f() { return 1; }\nconst int n = f();\nvoid main() { gl_Position = vec4(float(n)); }', 100, 'VERTEX');
  check(hasErr(e3, 2, "'n' : initializer of const variable must be a constant expression"), 'user function call init → const error');
  const e4 = errs('float f() { return 1.0; }\nstruct S { float x; };\nconst S s2 = S(f());\nvoid main() { gl_Position = vec4(s2.x); }', 100, 'VERTEX');
  check(hasErr(e4, 3, "'s2' : initializer of const variable must be a constant expression"), 'struct ctor with non-const arg → const error');
  const e5 = errs('const float f = 1;\nvoid main() { gl_Position = vec4(f); }', 100, 'VERTEX');
  check(e5.length > 0 && !hasErr(e5, 1, "'f' : initializer of const variable must be a constant expression"), 'const float f = 1 still fails (strict conversion — not a const error)');
  const e6 = errs('const vec4 v = vec4(1.0);\nconst vec4 w = v.x;\nvoid main() { gl_Position = w; }', 100, 'VERTEX');
  check(e6.length > 0, 'vec4 w = v.x (scalar→vec4 init) → conversion error, not silently accepted');
}

/* ------------------------------------------------------------------ */
/* 4. Runtime: link + run const reads                                  */
/* ------------------------------------------------------------------ */

{
  // Whole global-const read (mutation path → codegen emits vec4 ctor).
  const c1 = runFragment(
    VERT_SIMPLE,
    'precision mediump float;\nconst vec4 green = vec4(0.0, 1.0, 0.0, 1.0);\nvoid main() { gl_FragColor = green; }',
  );
  check(close4(c1, [0, 1, 0, 1]), `runtime: whole const vec4 read → [0,1,0,1] (got ${JSON.stringify(c1 ? Array.from(c1) : null)})`);

  // Scalar member reads (constValue folds on the member nodes).
  const c2 = runFragment(
    VERT_SIMPLE,
    'precision mediump float;\nconst vec4 green = vec4(0.0, 1.0, 0.0, 1.0);\nvoid main() { gl_FragColor = vec4(green.r, green.g, green.b, green.a); }',
  );
  check(close4(c2, [0, 1, 0, 1]), `runtime: member reads green.rgba → [0,1,0,1] (got ${JSON.stringify(c2 ? Array.from(c2) : null)})`);

  // Matrix column index (m[0]/m[1] mutate into vec2 ctors, column-major).
  const c3 = runFragment(
    VERT_SIMPLE,
    'precision mediump float;\nconst mat2 m = mat2(1.0, 2.0, 3.0, 4.0);\nvoid main() { gl_FragColor = vec4(m[0], m[1]); }',
  );
  check(close4(c3, [1, 2, 3, 4]), `runtime: const mat2 column reads → [1,2,3,4] (got ${JSON.stringify(c3 ? Array.from(c3) : null)})`);

  // Local const (codegen emits a JS local).
  const c4 = runFragment(
    VERT_SIMPLE,
    'precision mediump float;\nvoid main() { const vec4 green = vec4(0.0, 1.0, 0.0, 1.0); gl_FragColor = green; }',
  );
  check(close4(c4, [0, 1, 0, 1]), `runtime: LOCAL const vec4 read → [0,1,0,1] (got ${JSON.stringify(c4 ? Array.from(c4) : null)})`);

  // Dual mode: dFdx/dFdy on const components — constant duals must not crash.
  const c5 = runFragment(
    VERT_SIMPLE,
    '#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nconst vec4 c = vec4(0.25, 0.5, 0.75, 1.0);\nvoid main() { gl_FragColor = vec4(dFdx(c.x), dFdy(c.y), c.z, c.w); }',
    ['GL_OES_standard_derivatives'],
  );
  check(close4(c5, [0, 0, 0.75, 1]), `runtime: dual mode dFdx(const) → [0,0,0.75,1] (got ${JSON.stringify(c5 ? Array.from(c5) : null)})`);
}

/* ------------------------------------------------------------------ */
/* Summary                                                             */
/* ------------------------------------------------------------------ */

console.log(`const selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
