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
 *    dual-mode run (dFdx on a const — constant duals must not crash);
 *  - STRUCT EQUALITY: whole-struct == / != on const operands folds to a single
 *    bool (GLSL ES 1.00 §5.9 / ES 3.00) — link+run value checks; non-const
 *    struct operands ACCEPTED at compile level; wrong struct types and array
 *    operands still rejected; both 100 and 300.
 *  - BUILTIN VECTOR/MATRIX FOLDING: every MATH builtin from CTS
 *    const-variable-initialization.html folds in const initializers over
 *    VECTOR/MATRIX args (GLSL ES §5.10 — "a built-in function call whose
 *    arguments are all constant expressions, with the exception of the texture
 *    lookup functions"): unary elementwise (radians..fract), binary elementwise
 *    (atan(y,x)/pow/mod/min/max/step), vec→bvec predicates (lessThan..notEqual),
 *    geometric (length/distance/dot/cross/normalize/faceforward/reflect/refract)
 *    and matrixCompMult. Texture lookups and derivatives still reject. Link+run
 *    proves folded vector consts reach codegen with correct values.
 *  - GLOBAL INITIALIZERS: ANGLE ValidateGlobalInitializer parity (CTS
 *    global-variable-init.html) — WebGL1 allows const/plain-global/uniform/
 *    math-builtin initializers (legacy), but texture lookups, attribute/varying
 *    reads, l-value ops (`c = 0.0`, `c++`) and builtin non-constants
 *    (gl_FragCoord) are compile errors; ESSL 3.00 allows only const symbols.
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
/* 5. Struct equality (== / !=): const fold + acceptance               */
/* ------------------------------------------------------------------ */

{
  // Whole-struct == / != on CONST operands folds to a single bool; the
  // link+run proves the fold happened (a non-folded struct == would throw in
  // codegen's emitBinary). Values: a==b true, a!=c true, a!=b false, a==c false.
  // GLOBAL const structs: with the fold, the mutated operand nodes are never
  // emitted, so the link avoids the (pre-existing) struct-ctor codegen gap.
  const r1 = runFragment(
    VERT_SIMPLE,
    'precision mediump float;\nstruct S { float x; vec2 y; };\nconst S a = S(1.0, vec2(2.0, 3.0));\nconst S b = S(1.0, vec2(2.0, 3.0));\nconst S c = S(1.0, vec2(2.0, 4.0));\nvoid main() { const bool eq = (a == b); gl_FragColor = vec4(float(eq), float(a != c), float(a != b), float(a == c)); }',
  );
  check(close4(r1, [1, 1, 0, 0]), `runtime: const struct ==/!= fold → [1,1,0,0] (got ${JSON.stringify(r1 ? Array.from(r1) : null)})`);

  // Nested structs: member == (Inner) and whole == / != (the ogles
  // CorrectConstFolding2 shapes: st551/st552/st553 with st6 members).
  const r2 = runFragment(
    VERT_SIMPLE,
    'precision mediump float;\nstruct Inner { vec3 v; };\nstruct Outer { int i; float f; Inner inr; };\nconst Outer a = Outer(2, 4.0, Inner(vec3(7.0)));\nconst Outer b = Outer(2, 4.0, Inner(vec3(7.0)));\nconst Outer c = Outer(2, 4.0, Inner(vec3(8.0)));\nvoid main() { gl_FragColor = vec4(float(a.inr == b.inr), float(a == b), float(a.inr != c.inr), float(a != c)); }',
  );
  check(close4(r2, [1, 1, 1, 1]), `runtime: nested const struct ==/!= (member + whole) → [1,1,1,1] (got ${JSON.stringify(r2 ? Array.from(r2) : null)})`);

  // Local const struct == / != with a const bool result: ACCEPTED at compile
  // level (the const bool folds; linking a LOCAL const struct DECL still hits
  // the pre-existing struct-ctor codegen gap, so no link here).
  const l1 = okShader('precision mediump float;\nstruct S { float x; vec2 y; };\nvoid main() { const S a = S(1.0, vec2(2.0, 3.0)); const S b = S(1.0, vec2(2.0, 3.0)); const bool eq = (a == b); gl_FragColor = vec4(float(eq)); }', 100, 'FRAGMENT');
  check(l1 !== null, 'LOCAL const struct == accepted (fold path, compile-level)');

  // Non-const struct operands: ACCEPTED (compile-level only — runtime emit of
  // a non-folded struct comparison is a codegen concern).
  const n1 = okShader('struct S { float x; };\nuniform S us;\nvoid main() { bool b = (us == us); gl_Position = vec4(float(b)); }', 100, 'VERTEX');
  check(n1 !== null, 'non-const struct == accepted (uniform operands)');
  const n2 = okShader('struct S { float x; };\nvoid main() { S a = S(1.0); S b = S(1.0); bool c = (a != b); gl_Position = vec4(float(c)); }', 100, 'VERTEX');
  check(n2 !== null, 'non-const struct != accepted (local operands)');
  const n3 = okShader('struct S { float x; };\nvoid main() { const S a = S(1.0); S b = S(1.0); bool c = (a == b); gl_Position = vec4(float(c)); }', 100, 'VERTEX');
  check(n3 !== null, 'mixed const/non-const struct == accepted (no fold)');

  // ES 3.00: struct == on const operands (compile-level).
  const v3 = okShader('#version 300 es\nprecision mediump float;\nstruct S { float x; };\nout vec4 o;\nvoid main() { const S a = S(1.0); const S b = S(1.0); const bool eq = (a == b); o = vec4(float(eq)); }', 300, 'FRAGMENT');
  check(v3 !== null, 'ES 3.00 const struct == accepted');

  // Wrong struct types → error; arrays are never comparable → error.
  const e1 = errs('struct A { float x; };\nstruct B { float x; };\nvoid main() { bool b = (A(1.0) == B(1.0)); gl_Position = vec4(float(b)); }', 100, 'VERTEX');
  check(hasErr(e1, 3, "'==' : operands of type 'A' and 'B' cannot be compared"), 'struct == of different struct types → error');
  const e2 = errs('#version 300 es\nvoid main() { float a[2] = float[2](1.0, 2.0); float b[2] = float[2](1.0, 2.0); bool c = (a == b); gl_Position = vec4(float(c)); }', 300, 'VERTEX');
  check(hasErr(e2, 2, "'==' : operands of type 'float[2]' and 'float[2]' cannot be compared"), 'array == → error (arrays not comparable)');
}

/* ------------------------------------------------------------------ */
/* 6. Builtin const folding over VECTOR/MATRIX args & results          */
/*    (CTS const-variable-initialization.html — the 120 formerly-      */
/*    rejected 'not a constant expression' subtests)                   */
/* ------------------------------------------------------------------ */

{
  // Unary elementwise builtins, vec4 form (CTS builtInsGenTypeToGenType).
  // Previously REJECTED ('not a constant expression'); now folded by
  // evalBuiltinConstFold (semantics-const.ts). Vertex + fragment, mirroring
  // the CTS page's compile-status checks.
  const unary = [
    'radians', 'degrees', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
    'abs', 'sign', 'floor', 'ceil', 'fract',
  ];
  for (const b of unary) {
    const fsrc = `precision mediump float;\nconst vec4 c = ${b}(vec4(0.5));\nvoid main() { gl_FragColor = c; }`;
    check(okShader(fsrc, 100, 'FRAGMENT') !== null, `const vec4 c = ${b}(vec4(0.5)) (fragment)`);
    const vsrc = `precision mediump float;\nconst vec4 c = ${b}(vec4(0.5));\nvoid main() { gl_Position = c; }`;
    check(okShader(vsrc, 100, 'VERTEX') !== null, `const vec4 c = ${b}(vec4(0.5)) (vertex)`);
  }

  // vec→bvec comparison predicates (CTS builtIns2VecToBvec): bool-vector
  // RESULTS from vector args fold per component.
  const preds = ['lessThan', 'lessThanEqual', 'greaterThan', 'greaterThanEqual', 'equal', 'notEqual'];
  for (const b of preds) {
    const fsrc = `precision mediump float;\nconst bvec4 c = ${b}(vec4(0.2), vec4(0.5));\nvoid main() { gl_FragColor = vec4(float(c.x)); }`;
    check(okShader(fsrc, 100, 'FRAGMENT') !== null, `const bvec4 c = ${b}(vec4(0.2), vec4(0.5)) (fragment)`);
    const vsrc = `precision mediump float;\nconst bvec4 c = ${b}(vec4(0.2), vec4(0.5));\nvoid main() { gl_Position = vec4(float(c.x)); }`;
    check(okShader(vsrc, 100, 'VERTEX') !== null, `const bvec4 c = ${b}(vec4(0.2), vec4(0.5)) (vertex)`);
  }

  // Two-arg elementwise builtins, vec4 form (CTS builtIns2GenTypeToGenType;
  // atan here is the atan(y, x) two-arg overload).
  const bin2 = ['atan', 'pow', 'mod', 'min', 'max', 'step'];
  for (const b of bin2) {
    const fsrc = `precision mediump float;\nconst vec4 c = ${b}(vec4(0.2), vec4(0.5));\nvoid main() { gl_FragColor = c; }`;
    check(okShader(fsrc, 100, 'FRAGMENT') !== null, `const vec4 c = ${b}(vec4(0.2), vec4(0.5)) (fragment)`);
    const vsrc = `precision mediump float;\nconst vec4 c = ${b}(vec4(0.2), vec4(0.5));\nvoid main() { gl_Position = c; }`;
    check(okShader(vsrc, 100, 'VERTEX') !== null, `const vec4 c = ${b}(vec4(0.2), vec4(0.5)) (vertex)`);
  }

  // Geometric builtins (scalar or vector results) + matrixCompMult — the
  // page's one-of-a-kind cases, exact CTS expression list.
  const geo: [string, string, string][] = [
    ['float l', 'length(vec4(0.5))', 'vec4(l)'],
    ['float d', 'distance(vec4(0.5), vec4(0.2))', 'vec4(d)'],
    ['float dt', 'dot(vec4(0.5), vec4(0.2))', 'vec4(dt)'],
    ['vec3 cr', 'cross(vec3(0.5), vec3(0.2))', 'vec4(cr, 1.0)'],
    ['vec4 n', 'normalize(vec4(0.5))', 'n'],
    ['vec4 ff', 'faceforward(vec4(0.2), vec4(0.3), vec4(0.4))', 'ff'],
    ['vec4 r', 'reflect(vec4(0.2), vec4(0.5))', 'r'],
    ['vec4 rr', 'refract(vec4(0.2), vec4(0.3), 0.4)', 'rr'],
    ['mat4 m', 'matrixCompMult(mat4(0.2), mat4(0.5))', 'm[0]'],
  ];
  for (const [decl, init, use] of geo) {
    const fsrc = `precision mediump float;\nconst ${decl} = ${init};\nvoid main() { gl_FragColor = ${use}; }`;
    check(okShader(fsrc, 100, 'FRAGMENT') !== null, `const ${decl} = ${init} (fragment)`);
    const vsrc = `precision mediump float;\nconst ${decl} = ${init};\nvoid main() { gl_Position = ${use}; }`;
    check(okShader(vsrc, 100, 'VERTEX') !== null, `const ${decl} = ${init} (vertex)`);
  }

  // Scalar forms keep folding via the analysis-time path (regression guard —
  // these passed before the vector work; must not regress).
  for (const init of ['clamp(0.2, 0.3, 0.4)', 'mix(0.2, 0.3, 0.4)', 'smoothstep(0.2, 0.3, 0.4)', 'sin(0.5)']) {
    const fsrc = `precision mediump float;\nconst float c = ${init};\nvoid main() { gl_FragColor = vec4(c); }`;
    check(okShader(fsrc, 100, 'FRAGMENT') !== null, `const float c = ${init} (scalar still folds)`);
  }

  // §5.10 exception: texture lookup functions are NEVER constant expressions,
  // even with fully-const arguments (vector or not).
  const t1 = errs('precision mediump float;\nuniform sampler2D s;\nconst float x = texture2D(s, vec2(0.5)).x;\nvoid main() { gl_FragColor = vec4(x); }', 100, 'FRAGMENT');
  check(hasErr(t1, 3, "'x' : initializer of const variable must be a constant expression"), 'texture2D in const init → still rejected (§5.10 exception)');
  const t2 = errs('#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nconst float d = dFdx(0.5);\nvoid main() { gl_FragColor = vec4(d); }', 100, 'FRAGMENT', ['GL_OES_standard_derivatives']);
  check(hasErr(t2, 3, "'d' : initializer of const variable must be a constant expression"), 'dFdx in const init → still rejected (derivative, non-math builtin)');

  // RUNTIME: folded vector consts must flow to codegen with correct values
  // (global const → reads mutate into annotated ctor calls; codegen bakes).
  const r1 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst vec4 n = normalize(vec4(1.0, 0.0, 0.0, 0.0));\nvoid main() { gl_FragColor = n; }');
  check(close4(r1, [1, 0, 0, 0]), `runtime: const vec4 n = normalize(vec4(1,0,0,0)) → [1,0,0,0] (got ${JSON.stringify(r1 ? Array.from(r1) : null)})`);

  const r2 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst vec4 m = min(vec4(0.7, 0.2, 0.9, 0.4), vec4(0.5));\nvoid main() { gl_FragColor = m; }');
  check(close4(r2, [0.5, 0.2, 0.5, 0.4]), `runtime: const vec4 m = min(...) → [0.5,0.2,0.5,0.4] (got ${JSON.stringify(r2 ? Array.from(r2) : null)})`);

  const r3 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst vec3 c = cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));\nvoid main() { gl_FragColor = vec4(c, 1.0); }');
  check(close4(r3, [0, 0, 1, 1]), `runtime: const vec3 c = cross(x̂, ŷ) → [0,0,1,1] (got ${JSON.stringify(r3 ? Array.from(r3) : null)})`);

  const r4 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst bvec4 p = lessThan(vec4(0.2), vec4(0.5));\nvoid main() { gl_FragColor = vec4(float(p.x), float(p.y), float(p.z), float(p.w)); }');
  check(close4(r4, [1, 1, 1, 1]), `runtime: const bvec4 p = lessThan(...) → [1,1,1,1] (got ${JSON.stringify(r4 ? Array.from(r4) : null)})`);

  // mat2(0.5) is a DIAGONAL matrix (scalar splat), so matrixCompMult with it
  // zeroes the off-diagonal; use an explicit 4-arg ctor for a full pin.
  const r5 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst mat2 m = matrixCompMult(mat2(2.0, 3.0, 4.0, 5.0), mat2(0.5, 0.5, 0.5, 0.5));\nvoid main() { gl_FragColor = vec4(m[0], m[1]); }');
  check(close4(r5, [1, 1.5, 2, 2.5]), `runtime: const mat2 m = matrixCompMult(...) → [1,1.5,2,2.5] (got ${JSON.stringify(r5 ? Array.from(r5) : null)})`);

  const r6 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst vec4 ff = faceforward(vec4(0.2), vec4(0.3), vec4(0.4));\nvoid main() { gl_FragColor = ff; }');
  check(close4(r6, [-0.2, -0.2, -0.2, -0.2]), `runtime: const vec4 ff = faceforward(...) → [-0.2×4] (got ${JSON.stringify(r6 ? Array.from(r6) : null)})`);

  const r7 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst vec4 r = reflect(vec4(0.2), vec4(0.5));\nvoid main() { gl_FragColor = r; }');
  check(close4(r7, [-0.2, -0.2, -0.2, -0.2]), `runtime: const vec4 r = reflect(...) → [-0.2×4] (got ${JSON.stringify(r7 ? Array.from(r7) : null)})`);

  const r8 = runFragment(VERT_SIMPLE, 'precision mediump float;\nconst vec4 rr = refract(vec4(0.2), vec4(0.3), 0.4);\nvoid main() { gl_FragColor = rr; }');
  const eta = 0.4;
  const d = 4 * 0.3 * 0.2; // dot(N, I)
  const k = 1 - eta * eta * (1 - d * d);
  const c = eta * d + Math.sqrt(k);
  const expect = eta * 0.2 - c * 0.3;
  check(close4(r8, [expect, expect, expect, expect]), `runtime: const vec4 rr = refract(...) → [${expect}×4] (got ${JSON.stringify(r8 ? Array.from(r8) : null)})`);
}

/* ------------------------------------------------------------------ */
/* 7. Global variable initializer validation                           */
/*    (CTS global-variable-init.html — the 6 formerly-accepted         */
/*    subtests; exact page shader sources)                             */
/* ------------------------------------------------------------------ */

{
  // REJECTED — the six page-2 cases, exact CTS sources. Error message and
  // line mirror ANGLE ValidateGlobalInitializer ("global variable
  // initializers must be constant expressions").
  const e1 = errs(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nuniform sampler2D s;\nfloat f = texture2DLod(s, vec2(0.5, 0.5), 0.0).x;\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(hasErr(e1, 5, "'=' : global variable initializers must be constant expressions"), 'texture lookup function in global init → error (line 5)');

  const e2 = errs(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nfloat f = aPosition.x;\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(hasErr(e2, 4, "'=' : global variable initializers must be constant expressions"), 'attribute read in global init → error (line 4)');

  const e3 = errs(
    'precision mediump float;\nvarying float v;\nfloat f = v;\nvoid main() {\n    gl_FragColor = vec4(f);\n}',
    100, 'FRAGMENT');
  check(hasErr(e3, 3, "'=' : global variable initializers must be constant expressions"), 'varying read in global init → error (line 3)');

  const e4 = errs(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nfloat c = 1.0;\nfloat f = (c = 0.0);\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(hasErr(e4, 5, "'=' : global variable initializers must be constant expressions"), 'global as l-value (c = 0.0) in global init → error (line 5)');

  const e5 = errs(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nfloat c = 1.0;\nfloat f = (c++);\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(hasErr(e5, 5, "'=' : global variable initializers must be constant expressions"), 'global as l-value (c++) in global init → error (line 5)');

  const e6 = errs(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nfloat foo() {\n    return 1.0;\n}\nfloat f = foo();\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(hasErr(e6, 7, "'=' : global variable initializers must be constant expressions"), 'user function call in global init → error (line 7)');

  const e7 = errs(
    'precision mediump float;\nvec4 v = gl_FragCoord;\nvoid main() {\n    gl_FragColor = v;\n}',
    100, 'FRAGMENT');
  check(hasErr(e7, 2, "'=' : global variable initializers must be constant expressions"), 'builtin non-constant (gl_FragCoord) in global init → error (line 2)');

  // ESSL 3.00: only const symbols allowed (ANGLE: EvqGlobal/Uniform OK only
  // when version < 300) — a uniform initializer that WebGL1 accepts is an
  // error here.
  const e8 = errs(
    '#version 300 es\nprecision mediump float;\nuniform float u;\nfloat f = u;\nout vec4 o;\nvoid main() { o = vec4(f); }',
    300, 'FRAGMENT');
  check(hasErr(e8, 4, "'=' : global variable initializers must be constant expressions"), 'ESSL 3.00 uniform in global init → error (line 4)');

  // ACCEPTED — the page-2 passing subtests (WebGL1 legacy compatibility:
  // initializers need NOT be constant expressions, just per-invocation-state
  // free). Exact CTS sources.
  const a1 = okShader(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nconst float c = 1.0;\nfloat f = c;\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(a1 !== null, 'const global in global init → accepted');
  const a2 = okShader(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nfloat c = 1.0;\nfloat f = c;\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(a2 !== null, 'plain global in global init → accepted (WebGL1 legacy)');
  const a3 = okShader(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nuniform float u;\nfloat f = u;\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(a3 !== null, 'uniform in global init → accepted (WebGL1 legacy)');
  const a4 = okShader(
    'precision mediump float;\nattribute vec4 aPosition;\nvarying float v;\nfloat c = 1.0;\nfloat f = sin(c);\nvoid main() {\n    v = f;\n    gl_Position = aPosition;\n}',
    100, 'VERTEX');
  check(a4 !== null, 'math builtin in global init → accepted');
  const a5 = okShader(
    'precision mediump float;\nfloat green = 1.0;\nfloat black = 0.0;\nfloat f = true ? green : black;\nvoid main() {\n    gl_FragColor = vec4(0.0, f, 0.0, 1.0);\n}',
    100, 'FRAGMENT');
  check(a5 !== null, 'non-const globals in ternary global init → accepted');
  const a6 = okShader(
    'precision mediump float;\nuniform float u_zero;\nfloat green = 1.0 + u_zero;\nfloat f = true ? green : u_zero;\nvoid main() {\n    gl_FragColor = vec4(0.0, f, 0.0, 1.0);\n}',
    100, 'FRAGMENT');
  check(a6 !== null, 'uniform in ternary global init → accepted');
  const a7 = okShader(
    'precision mediump float;\nstruct S {\n    float zero;\n    int one;\n};\nuniform S us;\nS s = us;\nvoid main() {\n    float green = (s.one == 1) ? 1.0 : 0.0;\n    gl_FragColor = vec4(0.0, green, 0.0, 1.0);\n}',
    100, 'FRAGMENT');
  check(a7 !== null, 'global struct initialized with uniform struct → accepted');
  const a8 = okShader(
    'precision mediump float;\nint i = gl_MaxFragmentUniformVectors;\nvoid main() {\n    float green = (i > 0) ? 1.0 : 0.0;\n    gl_FragColor = vec4(0.0, green, 0.0, 1.0);\n}',
    100, 'FRAGMENT');
  check(a8 !== null, 'builtin constant (gl_Max*) in global init → accepted');
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
