/**
 * selftest-semantics-core.ts — smoke battery for the semantics CORE
 * (semantics.ts + semantics-expr.ts + semantics-stmt.ts).
 *
 * Run: npx tsx src/glsl/selftest-semantics-core.ts
 *
 * Drives the full pipeline (preprocess → tokenize → parse → analyzeProgram)
 * and checks: type helpers, implicit conversions, operator type errors,
 * overload resolution (builtin + user, ambiguity, builtin override), scalar
 * constructors + folding, lvalue rules (const / duplicate swizzle / read-only
 * builtins / literals), swizzles, recursion detection (direct, mutual, via
 * prototype), statement rules (if/switch/break/continue/return/case), builtin
 * extension gating, stage filtering, structs, arrays and shadowing.
 *
 * Prints "semantics core selftest: N checks, M failure(s)" then "OK" and
 * exits 0 on success, 1 on any failure.
 */
import { preprocess } from './preprocessor.js';
import { tokenize } from './lexer.js';
import { parse } from './parser.js';
import { analyzeProgram, SemContext } from './semantics.js';
import type { TranslationUnit } from './ast.js';
import { typeEquals, typeName, typeSize, typeComponents, toGLenum } from './types.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/** preprocess → tokenize → parse → analyzeProgram; returns ctx (+ast). */
function analyze(
  src: string,
  version: 100 | 300,
  stage: 'VERTEX' | 'FRAGMENT',
  exts?: Set<string>,
): { ctx: SemContext; ast: TranslationUnit | null } {
  const pp = preprocess(src, { version, extensions: exts });
  const empty = new SemContext(version, stage, exts ?? new Set());
  if (!pp.ok) return { ctx: empty, ast: null };
  const lex = tokenize(pp.tokens, pp.version);
  if (!lex.ok) return { ctx: empty, ast: null };
  const res = parse(lex.tokens, { version: pp.version, extensionDirectives: pp.extensionDirectives });
  if (!res.ok) return { ctx: empty, ast: null };
  const ctx = new SemContext(pp.version, stage, new Set(pp.extensions));
  analyzeProgram(res.ast, ctx);
  return { ctx, ast: res.ast };
}

function expectOk(src: string, version: 100 | 300, stage: 'VERTEX' | 'FRAGMENT', exts?: Set<string>): void {
  const r = analyze(src, version, stage, exts);
  if (r.ast === null) {
    check(false, `parse should succeed (v${version}): ${JSON.stringify(src.slice(0, 64))}`);
    return;
  }
  check(
    r.ctx.errors.length === 0,
    `no errors expected (v${version} ${stage}): ${JSON.stringify(src.slice(0, 64))} — got ${JSON.stringify(r.ctx.errors)}`,
  );
}

function expectErr(
  src: string,
  version: 100 | 300,
  stage: 'VERTEX' | 'FRAGMENT',
  exts?: Set<string>,
  re?: RegExp,
): void {
  const r = analyze(src, version, stage, exts);
  check(r.ast !== null, `parse should succeed for a semantics test (v${version}): ${JSON.stringify(src.slice(0, 64))}`);
  check(
    r.ctx.errors.length > 0,
    `error expected (v${version} ${stage}): ${JSON.stringify(src.slice(0, 64))}`,
  );
  if (re !== undefined && r.ctx.errors.length > 0) {
    check(
      r.ctx.errors.some((e) => re.test(e.message)),
      `error matching ${re} (v${version}): ${JSON.stringify(src.slice(0, 64))} — got ${JSON.stringify(r.ctx.errors.map((e) => e.message))}`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* 1. Type helpers                                                     */
/* ------------------------------------------------------------------ */

{
  const F = { kind: 'scalar', base: 'float' } as const;
  const I = { kind: 'scalar', base: 'int' } as const;
  const V3 = { kind: 'vector', base: 'float', size: 3 } as const;
  const M23 = { kind: 'matrix', cols: 2, rows: 3 } as const;
  const A3 = { kind: 'array', element: F, size: 3 } as const;
  check(toGLenum(F) === 0x1406, 'toGLenum float = 0x1406');
  check(toGLenum(V3) === 0x8b51, 'toGLenum vec3 = 0x8B51');
  check(toGLenum(M23) === 0x8b65, 'toGLenum mat2x3 = 0x8B65');
  check(toGLenum({ kind: 'scalar', base: 'uint' }) === 0x1405, 'toGLenum uint = 0x1405');
  check(toGLenum({ kind: 'sampler', sampler: 'sampler2DArray' }) === 0x8dc1, 'toGLenum sampler2DArray = 0x8DC1');
  check(typeName(M23) === 'mat2x3', `typeName mat2x3 → '${typeName(M23)}'`);
  check(typeName(A3) === 'float[3]', `typeName float[3] → '${typeName(A3)}'`);
  check(typeName({ kind: 'array', element: I, size: null }) === 'int[]', 'typeName int[] (unsized)');
  check(typeName({ kind: 'struct', name: 'S', members: [] }) === 'S', 'typeName struct by name');
  check(typeEquals(A3, { kind: 'array', element: F, size: 3 }), 'typeEquals array element+size');
  check(!typeEquals(A3, { kind: 'array', element: F, size: 4 }), 'typeEquals array size differs');
  check(typeEquals({ kind: 'struct', name: 'S', members: [] }, { kind: 'struct', name: 'S', members: [] }), 'typeEquals struct BY NAME');
  check(typeComponents(V3) === 3, 'typeComponents vec3 = 3');
  check(typeComponents(M23) === 3, 'typeComponents mat2x3 = rows = 3');
  check(typeComponents({ kind: 'sampler', sampler: 'sampler2D' }) === 1, 'typeComponents sampler = 1');
  check(typeSize(A3) === 3, 'typeSize array = 3');
  check(typeSize(V3) === 1, 'typeSize non-array = 1');
}

/* ------------------------------------------------------------------ */
/* 2. Pipeline basics                                                  */
/* ------------------------------------------------------------------ */

expectOk('void main() {}', 100, 'VERTEX');
expectOk('#version 300 es\nvoid main() {}', 300, 'FRAGMENT');

/* ------------------------------------------------------------------ */
/* 3. Implicit conversions                                             */
/* ------------------------------------------------------------------ */

expectOk('void main() { float x = 1 + 2.0; }', 100, 'VERTEX'); // int+float → float
expectOk('#version 300 es\nvoid main() { uint x = 1u + 2; float y = 1u + 2.0; }', 300, 'VERTEX'); // int→uint, uint→float
expectErr('void main() { int x = 1.0 + 2; }', 100, 'VERTEX'); // float→int NOT implicit
expectErr('void main() { uint x = 1; }', 100, 'VERTEX'); // uint is not a 1.00 type

/* ------------------------------------------------------------------ */
/* 4. Operator type errors                                             */
/* ------------------------------------------------------------------ */

expectErr('void main() { float x = 1 + true; }', 100, 'VERTEX');
expectErr('void main() { bool b = 1 && true; }', 100, 'VERTEX');
expectErr('void main() { bool b = vec2(1.0) < vec2(2.0); }', 100, 'VERTEX'); // relational on vectors
expectErr('void main() { vec3 v = vec2(1.0) + vec3(1.0); }', 100, 'VERTEX'); // size mismatch

/* ------------------------------------------------------------------ */
/* 5. Overload resolution                                              */
/* ------------------------------------------------------------------ */

expectOk('void main() { float x = atan(1.0); float y = atan(1.0, 1.0); }', 100, 'VERTEX');
expectErr('float min(float a, float b) { return a; }', 100, 'VERTEX', undefined, /built-in/i); // override builtin
expectErr(
  'float f(int a, float b) { return 1.0; } float f(float a, int b) { return 1.0; } void main() { float x = f(1, 2); }',
  100,
  'VERTEX',
  undefined,
  /ambiguous/i,
);
expectErr('float f(float x) { return x; } void main() { float y = f(1.0, 2.0); }', 100, 'VERTEX', undefined, /no matching/i);

/* ------------------------------------------------------------------ */
/* 6. Constructors                                                     */
/* ------------------------------------------------------------------ */

expectOk('void main() { vec4 v = vec4(vec2(1.0), vec2(2.0)); }', 100, 'VERTEX');
expectOk('void main() { mat3 m = mat3(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0); }', 100, 'VERTEX');
expectErr('void main() { mat2 m = mat2(mat3(1.0)); }', 100, 'VERTEX'); // dims mismatch
expectOk('void main() { int i = int(3.7); }', 100, 'VERTEX'); // scalar conversion ctor
expectErr('void main() { vec3 v = vec3(1.0, 2.0); }', 100, 'VERTEX'); // component count
expectOk('#version 300 es\nvoid main() { float a[3] = float[3](1.0, 2.0, 3.0); }', 300, 'VERTEX');
expectErr('void main() { float a[3] = float[3](1.0, 2.0, 3.0); }', 100, 'VERTEX', undefined, /ES 3\.00/);

/* ------------------------------------------------------------------ */
/* 7. Const folding (end-to-end via array sizes)                       */
/* ------------------------------------------------------------------ */

expectOk('const int N = 2 + 3; float a[N];', 100, 'VERTEX'); // fold: 5
expectOk('const int N = int(3.7); float a[N];', 100, 'VERTEX'); // fold: 3
expectOk('#version 300 es\nconst int N = 1 << 4; float a[N];', 300, 'VERTEX'); // fold: 16
expectErr('const int N = 2 - 3; float a[N];', 100, 'VERTEX'); // size -1 → error

/* ------------------------------------------------------------------ */
/* 8. Lvalue rules                                                     */
/* ------------------------------------------------------------------ */

expectErr('void main() { const float c = 1.0; c = 2.0; }', 100, 'VERTEX', undefined, /lvalue/i);
expectErr('void main() { vec2 v; v.xx = vec2(1.0); }', 100, 'VERTEX', undefined, /lvalue/i);
expectErr('void main() { gl_FragCoord.x = 1.0; }', 100, 'FRAGMENT', undefined, /lvalue/i); // read-only builtin
expectErr('void main() { 1 = 2; }', 100, 'VERTEX', undefined, /lvalue/i);
expectErr('void main() { 1++; }', 100, 'VERTEX', undefined, /lvalue/i);
expectOk('void main() { vec2 v; v[0] = 1.0; }', 100, 'VERTEX'); // index lvalue propagates
expectOk('void main() { int i = 0; i++; ++i; i--; --i; }', 100, 'VERTEX');

/* ------------------------------------------------------------------ */
/* 9. Swizzles                                                         */
/* ------------------------------------------------------------------ */

expectOk('void main() { vec2 v; float x = v.x; vec2 w = v.yx; }', 100, 'VERTEX');
expectErr('void main() { vec2 v; float x = v.xg; }', 100, 'VERTEX', undefined, /mixes component sets/);
expectErr('void main() { vec2 v; float x = v.xyz; }', 100, 'VERTEX', undefined, /out of range/);
expectErr('void main() { vec2 v; vec3 w = v.xy; }', 100, 'VERTEX'); // vec2 → vec3 not convertible

/* ------------------------------------------------------------------ */
/* 10. Recursion detection                                             */
/* ------------------------------------------------------------------ */

expectErr('int f(int x) { return f(x); }', 100, 'VERTEX', undefined, /recursion/);
expectErr('int a(int x) { return b(x); } int b(int x) { return a(x); }', 100, 'VERTEX', undefined, /recursion/);
expectErr(
  'int a(int x); int b(int x) { return a(x); } int a(int x) { return b(x); }',
  100,
  'VERTEX',
  undefined,
  /recursion/,
);

/* ------------------------------------------------------------------ */
/* 11. Statements                                                      */
/* ------------------------------------------------------------------ */

expectErr('void main() { if (1) {} }', 100, 'VERTEX', undefined, /boolean/);
expectErr('void main() { break; }', 100, 'VERTEX', undefined, /break/);
expectErr('void main() { continue; }', 100, 'VERTEX', undefined, /continue/);
expectOk('void main() { while (true) { break; } }', 100, 'VERTEX');
expectOk('void main() { for (int i = 0; i < 3; i++) { continue; } }', 100, 'VERTEX');
expectOk('#version 300 es\nvoid main() { switch (1) { case 1: break; default: break; } }', 300, 'VERTEX');
expectErr('#version 300 es\nvoid main() { switch (1.0) { } }', 300, 'VERTEX', undefined, /switch/);
expectErr('#version 300 es\nvoid main() { case 1: }', 300, 'VERTEX', undefined, /case/);

/* ------------------------------------------------------------------ */
/* 12. Return statements                                               */
/* ------------------------------------------------------------------ */

expectOk('float f() { return 1; }', 100, 'VERTEX'); // int→float implicit
expectErr('int f() { return 1.0; }', 100, 'VERTEX'); // float→int NOT implicit
expectErr('void f() { return 1.0; }', 100, 'VERTEX', undefined, /void/);
expectErr('float f() { return; }', 100, 'VERTEX', undefined, /return/);

/* ------------------------------------------------------------------ */
/* 13. Builtin extension gating                                        */
/* ------------------------------------------------------------------ */

expectErr('precision mediump float;\nvoid main() { float d = dFdx(1.0); }', 100, 'FRAGMENT', undefined, /extension/i);
expectOk(
  '#extension GL_OES_standard_derivatives : enable\nprecision mediump float;\nvoid main() { float d = dFdx(1.0); }',
  100,
  'FRAGMENT',
  new Set(['GL_OES_standard_derivatives']),
);
expectOk('#version 300 es\nprecision mediump float;\nvoid main() { float d = dFdx(1.0); }', 300, 'FRAGMENT'); // core in 3.00
expectErr(
  '#extension GL_OES_standard_derivatives : enable\nvoid main() { float d = dFdx(1.0); }',
  100,
  'VERTEX',
  new Set(['GL_OES_standard_derivatives']),
  /not available/,
);

/* ------------------------------------------------------------------ */
/* 14. Stage filtering                                                 */
/* ------------------------------------------------------------------ */

expectOk('void main() { gl_Position = vec4(1.0); }', 100, 'VERTEX');
expectErr('void main() { gl_Position = vec4(1.0); }', 100, 'FRAGMENT', undefined, /not available/);
expectOk('void main() { gl_FragColor = vec4(1.0); }', 100, 'FRAGMENT');
expectErr('void main() { gl_FragColor = vec4(1.0); }', 100, 'VERTEX', undefined, /not available/);
expectOk('#version 300 es\nvoid main() { int i = gl_VertexID; }', 300, 'VERTEX');

/* ------------------------------------------------------------------ */
/* 15. Structs                                                         */
/* ------------------------------------------------------------------ */

expectOk('struct S { float x; int y; }; void main() { S s; float a = s.x; int b = s.y; }', 100, 'VERTEX');
expectErr('struct S { float x; int y; }; void main() { S s; float a = s.z; }', 100, 'VERTEX', undefined, /no member/);
expectOk('struct S { float x; int y; }; void main() { S s = S(1.0, 2); }', 100, 'VERTEX');
expectErr('struct S { float x; int y; }; void main() { S s = S(1.0); }', 100, 'VERTEX', undefined, /argument/i);

/* ------------------------------------------------------------------ */
/* 16. Misc: gl_Max*, inout, comma, equality, %, shadowing, prototypes */
/* ------------------------------------------------------------------ */

expectOk('void main() { float a[gl_MaxVertexAttribs]; }', 100, 'VERTEX'); // gl_Max* folds
expectOk('#version 300 es\nvoid f(inout float x) { x = 1.0; }', 300, 'VERTEX');
expectOk('void main() { float x = (1.0, 2.0); }', 100, 'VERTEX'); // comma type = last
expectErr('void main() { x = 1.0; }', 100, 'VERTEX', undefined, /undeclared/);
expectOk('void main() { bool b = 1 == 1.0; bool c = vec2(1.0) == vec2(1.0); }', 100, 'VERTEX');
expectErr('void main() { bool b = 1 == true; }', 100, 'VERTEX'); // int vs bool
expectErr('#version 300 es\nvoid main() { float z = ~1.0; }', 300, 'VERTEX'); // ~ on float
expectOk('#version 300 es\nvoid main() { int x = 5 % 2; uint u = 5u % 2u; int s = 1 << 2; }', 300, 'VERTEX');
expectOk('void main() { int x = 5 % 2; }', 100, 'VERTEX'); // % is 1.00-legal for ints
expectErr('float x; void main() { float x; }', 100, 'VERTEX', undefined, /redefinition/);
expectErr('void main() { float x; { float x; } }', 100, 'VERTEX', undefined, /redefinition/);
expectOk('float f(float x); float f(float y) { return y; }', 100, 'VERTEX');
expectOk('float f(float x); float f(float y) { return y; } float g() { return f(1.0); }', 100, 'VERTEX');
expectErr('void f() { } void main() { float x = f(); }', 100, 'VERTEX'); // void call as value
expectOk('void f() { } void main() { f(); }', 100, 'VERTEX'); // void call as statement

/* ------------------------------------------------------------------ */
/* 17. Ternary                                                         */
/* ------------------------------------------------------------------ */

expectOk('void main() { float x = true ? 1 : 2.0; }', 100, 'VERTEX');
expectOk('void main() { vec3 v = true ? vec3(1.0) : vec3(2.0); }', 100, 'VERTEX');
expectErr('void main() { mat2 m = true ? mat2(1.0) : mat2(2.0); }', 100, 'VERTEX', undefined, /incompatible/);

/* ------------------------------------------------------------------ */
/* 18. Arrays: bounds, unsized params, index on scalar                 */
/* ------------------------------------------------------------------ */

expectOk('void main() { float a[2]; float x = a[1]; }', 100, 'VERTEX');
expectErr('void main() { float a[2]; float x = a[2]; }', 100, 'VERTEX', undefined, /out of range/);
expectErr('void main() { float a[2]; float x = a[-1]; }', 100, 'VERTEX', undefined, /out of range/);
expectOk('float f(float a[]) { return a[0]; }', 100, 'VERTEX'); // unsized param array
expectErr('void main() { float x = 1.0; float y = x[0]; }', 100, 'VERTEX', undefined, /index/);

/* ------------------------------------------------------------------ */
/* 19. Builtin-name reuse (BUG 4) + function-body struct scoping (BUG 6) */
/* ------------------------------------------------------------------ */

// BUG 4: GLSL ES allows user VARIABLES with builtin function names (the
// builtin name is only shadowed in the declaring scope).
expectOk('void main() { float sign = 1.0; sign = -sign; }', 100, 'VERTEX'); // local var named after builtin fn
expectOk('float sign; void main() { sign = 1.0; }', 100, 'VERTEX'); // global var
expectOk('void main() { vec2 exp = vec2(1.0); float c = exp.x; }', 100, 'VERTEX');
expectOk('void f(float sin) { sin = sin + 1.0; } void main() { f(1.0); }', 100, 'VERTEX'); // param
// BUG 4: user FUNCTION overloads with a builtin name (DIFFERENT signature)
// are legal; the overload is attached to the builtin placeholder's siblings.
expectOk('int radians(int f) { return f; } void main() { }', 100, 'FRAGMENT'); // ogles CorrectBuiltInOveride
expectOk('float pow(float a, float b, float c) { return a; } void main() { }', 100, 'VERTEX');
expectOk('int radians(int f); int radians(int f) { return f; } void main() { }', 100, 'FRAGMENT'); // proto + def
expectOk('int radians(int f) { return f; } int radians(int f); void main() { }', 100, 'FRAGMENT'); // def + proto
expectOk('int radians(int f) { return f; } int radians(int f, int g) { return f + g; } void main() { }', 100, 'VERTEX'); // 2 overloads
// BUG 4: a same-signature user function with a builtin name is still an error
// (compared against the builtin TABLES, not the void/[] placeholder).
expectErr('float radians(float f) { return f; } void main() { }', 100, 'FRAGMENT', undefined, /built-in/i);
expectErr('int sin(float x) { return 1; } void main() { }', 100, 'VERTEX', undefined, /built-in/i); // same params, diff ret
// (same-signature 'min' override already covered above, section 14)
// BUG 4: builtin VARIABLES (gl_*) and gl_Max* constants stay non-shadowable.
expectErr('float gl_Position; void main() { }', 100, 'VERTEX', undefined, /redefinition/);
expectErr('float gl_FragColor; void main() { }', 100, 'FRAGMENT', undefined, /redefinition/);
// BUG 4: single namespace — once a user function claims a builtin name, a
// later variable with the same name is a redefinition (and vice versa).
expectErr('int radians(int f) { return f; } float radians; void main() { }', 100, 'FRAGMENT', undefined, /redefinition/);
expectErr('float radians; int radians(int f) { return f; } void main() { }', 100, 'FRAGMENT', undefined, /redefinition/);

// BUG 6: struct types declared INSIDE a function body are visible to later
// statements in the same scope (declarations, constructors, nested structs).
expectOk('void main() { struct S { float x; int y; }; S s; float a = s.x; int b = s.y; }', 100, 'VERTEX');
expectOk('void main() { struct S { float x; }; S s = S(1.0); float a = s.x; }', 100, 'VERTEX'); // constructor
expectOk('void main() { struct B { float x; }; struct A { B b; }; A a; float y = a.b.x; }', 100, 'VERTEX'); // nested
expectOk('void main() { struct B { float x; }; struct A { B b; }; A a = A(B(1.0)); float y = a.b.x; }', 100, 'VERTEX'); // nested ctor
expectOk('void main() { { struct S { float x; }; S s; float a = s.x; } }', 100, 'VERTEX'); // block-scoped struct
expectErr('void main() { struct S { float x; }; { struct S { float y; }; } }', 100, 'VERTEX', undefined, /redefinition/); // GLSL: no shadowing
// BUG 4 + BUG 6 combined: local struct with a builtin function name.
expectOk('void main() { struct sign { float x; }; sign s; s.x = 1.0; }', 100, 'VERTEX');

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`semantics core selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
