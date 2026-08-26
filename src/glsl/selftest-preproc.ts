/**
 * selftest-preproc.ts — sanity checks for the GLSL ES preprocessor stage.
 *
 * Run: npx tsx src/glsl/selftest-preproc.ts
 *
 * Verifies: object/function-like macros, `#` stringize, `##` paste, argument
 * pre-expansion, recursion suppression, predefined macros, #if arithmetic /
 * defined() / short-circuit, conditional nesting and skipped-branch token
 * absorption, #version rules, #extension validation (incl. `all`, last-wins),
 * #error, #line remapping, unterminated comment/#if, redefinition errors,
 * backslash-newline splicing and comment stripping. Prints "OK" and exits 0
 * on success.
 */
import { preprocess } from './preprocessor.js';
import type { PreprocessResult } from './preprocessor.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

interface Opts {
  version?: 100 | 300;
  defines?: Record<string, string>;
  extensions?: Set<string>;
}

function run(src: string, opts?: Opts): PreprocessResult {
  return preprocess(src, { version: opts?.version ?? 100, defines: opts?.defines, extensions: opts?.extensions });
}

function texts(src: string, opts?: Opts): string[] {
  const r = run(src, opts);
  check(r.ok, 'expected success: ' + JSON.stringify(src) + (r.ok ? '' : ' errors: ' + JSON.stringify(r.errors)));
  return r.ok ? r.tokens.map((t) => t.text) : [];
}

function expect(src: string, expected: string[], opts?: Opts, msg?: string): void {
  const got = texts(src, opts);
  check(
    JSON.stringify(got) === JSON.stringify(expected),
    (msg ?? 'tokens') + ` for ${JSON.stringify(src)}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(got)}`,
  );
}

function expectErr(src: string, msgPart: string, opts?: Opts): void {
  const r = run(src, opts);
  check(!r.ok, `expected failure: ${JSON.stringify(src)}`);
  if (!r.ok) {
    const joined = r.errors.map((e) => e.message).join(' | ');
    check(joined.includes(msgPart), `error for ${JSON.stringify(src)} should contain '${msgPart}', got: ${joined}`);
  }
}

/* ------------------------------------------------------------------ */
/* Object-like & function-like macros                                  */
/* ------------------------------------------------------------------ */

expect('#define FOO 42\nFOO', ['42']);
expect('#define FOO 1 + 2\nFOO', ['1', '+', '2']);
expect('#define EMPTY\nEMPTY', []);
expect('#define ADD(a,b) ((a)+(b))\nADD(1,2)', ['(', '(', '1', ')', '+', '(', '2', ')', ')']);
expect('#define F() 42\nF()', ['42']);
expect('#define F(x) x\nF(hello)', ['hello']);
expect('#define F(x) x + 1\nF(1)', ['1', '+', '1']);
// Whitespace between the name and `(` prevents function-like invocation.
expect('#define F(x) x\nF (1)', ['F', '(', '1', ')']);
expect('#define F(x) x\nF\n(1)', ['F', '(', '1', ')']);
// Object-like macro whose replacement is a function-like name invoked by the
// following `(` (the re-scan must see across the replacement boundary).
expect('#define A B\n#define B(x) x\nA(1)', ['1']);
expect('#define A B\n#define B(x) x\nA (1)', ['B', '(', '1', ')']);
// Nested parens and multi-line args.
expect('#define G(a,b) a+b\nG((1,2),3)', ['(', '1', ',', '2', ')', '+', '3']);
expect('#define G(a,b) a+b\nG(1,\n2)', ['1', '+', '2']);
expect('#define G(a,b) a+b\nG((1,\n2),3)', ['(', '1', ',', '2', ')', '+', '3']);
expect('#define F(x) x\nF(\n1\n)', ['1']);
// Newline inside a stringize argument collapses to a single space.
expect('#define STR(x) #x\nSTR(a\nb)', ['"a b"']);
// Empty and multi-token args.
expect('#define G(a,b) [a][b]\nG(,x)', ['[', ']', '[', 'x', ']']);
expect('#define G(a,b) [a][b]\nG(1,2 3)', ['[', '1', ']', '[', '2', '3', ']']);
// Wrong argument count is an error.
expectErr('#define F(x) x\nF(1,2)', 'requires 1 argument');

/* ------------------------------------------------------------------ */
/* Stringize and token paste                                           */
/* ------------------------------------------------------------------ */

expect('#define GLUE(a,b) a##b\nGLUE(foo,bar)', ['foobar']);
expect('#define GLUE(a,b) a##b\nGLUE(foo,bar) baz', ['foobar', 'baz']);
expect('#define CAT3(a,b,c) a##b##c\nCAT3(x,y,z)', ['xyz']);
expect('#define STR(x) #x\nSTR(foo)', ['"foo"']);
expect('#define STR(x) #x\nSTR(foo bar)', ['"foo bar"']);
expect('#define STR(x) #x\nSTR(  a  b  )', ['"a b"']);
expect('#define STR(x) #x\nSTR()', ['""']);
expect('#define STR(x) #x\nSTR(a "q")', ['"a \\"q\\""']);
// Pasted result is re-scanned.
expect('#define GLUE(a,b) a##b\n#define FOOBAR 42\nGLUE(FOO,BAR)', ['42']);
// Paste with empty operand.
expect('#define GLUE(a,b) a##b\nGLUE(,x)', ['x']);
expect('#define GLUE(a,b) a##b\nGLUE(x,)', ['x']);

/* ------------------------------------------------------------------ */
/* Argument pre-expansion & recursion suppression                      */
/* ------------------------------------------------------------------ */

expect('#define A 1\n#define F(x) x+1\nF(A)', ['1', '+', '1']);
expect('#define A 1\n#define F(x) x\nF(F(A))', ['1']);
expect('#define X X+1\nX', ['X', '+', '1']);
// Hide sets accumulate: X→Y, Y→X, then `X` expands to Y (X hidden), Y expands
// to X (Y hidden) — the second X is hidden, so the result is `X`.
expect('#define X Y\n#define Y X\nX', ['X']);
expect('#define F(x) F(x)\nF(1)', ['F', '(', '1', ')']);
// Args adjacent to `##` are NOT pre-expanded (the pasted result is re-scanned).
expect('#define A 1\n#define B 2\n#define GLUE(a,b) a##b\nGLUE(A,B)', ['AB']);
// Identical redefinition is legal; different replacement is an error.
expect('#define A 1\n#define A 1\nA', ['1']);
expectErr('#define A 1\n#define A 2', 'redefinition of macro');
expectErr('#define F(x) x\n#define F(y) y', 'redefinition of macro');

/* ------------------------------------------------------------------ */
/* Predefined macros                                                   */
/* ------------------------------------------------------------------ */

expect('__LINE__', ['1']);
expect('\n\n__LINE__', ['3']);
expect('__FILE__', ['0']);
expect('GL_ES', ['1']);
expect('__VERSION__', ['100']);
expect('#version 300 es\n__VERSION__', ['300'], { version: 300 });
expect('#version 100\n__VERSION__', ['100']);
// __LINE__ inside a macro body resolves at the invocation site.
expect('#define L __LINE__\n\nL', ['3']);
// #undef of a predefined macro removes it.
expect('#undef GL_ES\nGL_ES', ['GL_ES']);

/* ------------------------------------------------------------------ */
/* #if / #elif / #ifdef / #ifndef / #else / #endif                     */
/* ------------------------------------------------------------------ */

expect('#if 1\nok\n#endif', ['ok']);
expect('#if 0\nno\n#endif', []);
expect('#if 1+2*3 == 7\nok\n#endif', ['ok']);
expect('#if ((1 + 2) * 3) == 9\nok\n#endif', ['ok']);
expect('#if (1 << 3) == 8\nok\n#endif', ['ok']);
expect('#if 2 << 3 == 16\nok\n#endif', ['ok']);
expect('#if 0x10 == 16\nok\n#endif', ['ok']);
expect('#if 017 == 15\nok\n#endif', ['ok']);
expect('#if 5 % 2 == 1 && 3 > 2\nok\n#endif', ['ok']);
expect('#if 1 && 0 || 1\nok\n#endif', ['ok']);
expect('#if 2 < 3 && 3 < 2\nno\n#else\nok\n#endif', ['ok']);
expect('#if 1\n#if 0\nx\n#elif 1\ny\n#endif\nz\n#endif', ['y', 'z']);
expect('#if 0\nx\n#elif 1\ny\n#else\nz\n#endif', ['y']);
expect('#if 0\nx\n#else\ny\n#endif', ['y']);
expect('#if 1\nx\n#else\ny\n#endif', ['x']);
expect('#ifdef GL_ES\nok\n#endif', ['ok']);
expect('#ifndef GL_ES\nno\n#else\nok\n#endif', ['ok']);
expect('#define FOO\n#ifdef FOO\nok\n#endif', ['ok']);
expect('#undef FOO\n#ifdef FOO\nno\n#else\nok\n#endif', ['ok']);
// defined() / defined NAME; undefined identifiers are 0.
expect('#if defined(FOO)\nno\n#else\nok\n#endif', ['ok']);
expect('#define FOO 1\n#if defined FOO\nok\n#endif', ['ok']);
expect('#if !defined(FOO)\nok\n#endif', ['ok']);
expect('#if UNDEFINED_IDENT == 0\nok\n#endif', ['ok']);
// Short-circuit: unevaluated branches must not error.
expect('#if 0 && UNDEFINED_IDENT\nno\n#endif\nok', ['ok']);
expect('#if 1 || UNDEFINED_IDENT\nok\n#endif', ['ok']);
expect('#if 0 && (1/0)\nno\n#endif\nok', ['ok']);
expect('#if 1 || (1/0)\nok\n#endif', ['ok']);
// Macro expansion inside #if.
expect('#define V 42\n#if V == 42\nok\n#endif', ['ok']);
expect('#define V 42\n#if V * 2 == 84\nok\n#endif', ['ok']);
expect('#define GT(a,b) (a > b)\n#if GT(2,1)\nok\n#endif', ['ok']);
// __LINE__ and __VERSION__ in #if.
expect('#line 50\n#if __LINE__ == 50\nok\n#endif', ['ok']);
expect('#if __VERSION__ == 100\nok\n#endif', ['ok']);
// Skipped branches: tokens consumed, macros NOT defined, no output.
expect('#if 0\n#define BAD 1\nBAD\n#error hidden\n#endif\nok', ['ok']);
expect('#if 1\nok\n#if 0\nBAD\n#endif\n#endif', ['ok']);
// Malformed conditionals.
expectErr('#if 1\nno', 'unterminated #if');
expectErr('#endif', '#endif without #if');
expectErr('#else', '#else without #if');
expectErr('#elif 1', '#elif without #if');
expectErr('#if 1\n#else\n#elif 1\n#endif', '#elif after #else');
expectErr('#if\n#endif', 'invalid expression');
expectErr('#if 1 ; 2\n#endif', 'invalid expression');
// `defined` rules (WebGL): generated by expansion or used as macro name.
expectErr('#if 1\n#endif\n#define defined 1', "'defined' cannot be used as a macro name");
expectErr('#define X defined\n#if X\n#endif', "'defined' generated by macro expansion");

/* ------------------------------------------------------------------ */
/* #version                                                            */
/* ------------------------------------------------------------------ */

expect('#version 100\nfoo', ['foo']);
expect('#version 300 es\nfoo', ['foo'], { version: 300 });
expectErr('#version 300 es\nfoo', 'not supported in a WebGL 1 context');
expectErr('#version 110\nfoo', 'invalid #version');
expectErr('#version 300\nfoo', 'invalid #version');
expectErr('#version 100 es\nfoo', 'invalid #version');
expectErr('foo\n#version 100', 'must appear before any other content');
expectErr('#define X 1\n#version 100', 'must appear before any other content');
// A second #version is accepted (last one wins). GLSL ES 3.00 says only one
// #version may occur in a shader, but the preprocessor does not enforce it.
expect('#version 100\n#version 300 es\nfoo', ['foo'], { version: 300 });
// Comments/whitespace before #version are fine.
expect('// comment\n/* block */\n#version 100\nfoo', ['foo']);

/* ------------------------------------------------------------------ */
/* #extension                                                          */
/* ------------------------------------------------------------------ */

const EXT = new Set(['GL_OES_standard_derivatives']);
expect('#extension GL_OES_standard_derivatives : enable\nGL_OES_standard_derivatives', ['1'], { extensions: EXT });
expect('#extension GL_OES_standard_derivatives : require\nok', ['ok'], { extensions: EXT });
expectErr('#extension GL_UNKNOWN : enable', "extension 'GL_UNKNOWN' is not supported", { extensions: EXT });
expectErr('#extension GL_UNKNOWN : require', "extension 'GL_UNKNOWN' is not supported", { extensions: EXT });
expect('#extension GL_UNKNOWN : warn\nok', ['ok'], { extensions: EXT });
expect('#extension GL_UNKNOWN : disable\nok', ['ok'], { extensions: EXT });
expect('#extension GL_UNKNOWN : warn\nok', ['ok']); // no extensions set at all
// Last directive wins.
expect('#extension GL_OES_standard_derivatives : enable\n#extension GL_OES_standard_derivatives : disable\nGL_OES_standard_derivatives',
  ['GL_OES_standard_derivatives'], { extensions: EXT });
expect('#extension GL_OES_standard_derivatives : disable\n#extension GL_OES_standard_derivatives : enable\nGL_OES_standard_derivatives',
  ['1'], { extensions: EXT });
// `all` : warn/disable OK, require/enable error.
expect('#extension all : warn\nok', ['ok'], { extensions: EXT });
expect('#extension all : disable\nok', ['ok'], { extensions: EXT });
expectErr('#extension all : enable', "cannot use 'require' or 'enable' with 'all'", { extensions: EXT });
// `all : disable` clears previously enabled extensions.
expect('#extension GL_OES_standard_derivatives : enable\n#extension all : disable\nGL_OES_standard_derivatives',
  ['GL_OES_standard_derivatives'], { extensions: EXT });
// Result metadata.
{
  const r = run('#extension GL_OES_standard_derivatives : enable\n#extension GL_UNKNOWN : warn\nok', { extensions: EXT });
  check(r.ok, 'extension metadata run ok');
  if (r.ok) {
    check(r.version === 100, 'declared version is 100 by default');
    check(JSON.stringify(r.extensions) === JSON.stringify(['GL_OES_standard_derivatives']),
      `extensions result: got ${JSON.stringify(r.extensions)}`);
    check(r.extensionDirectives.length === 2, 'two extension directives recorded');
    check(r.extensionDirectives[0].name === 'GL_OES_standard_derivatives' && r.extensionDirectives[0].behavior === 'enable' && r.extensionDirectives[0].line === 1,
      'first directive metadata');
    check(r.extensionDirectives[1].behavior === 'warn' && r.extensionDirectives[1].line === 2, 'second directive metadata');
  }
}

/* ------------------------------------------------------------------ */
/* #error / #pragma / #line                                            */
/* ------------------------------------------------------------------ */

expectErr('#error boom', 'boom');
expectErr('#error "a b"', 'a b');
expectErr('#error\nfoo', '#error');
expect('#pragma STDGL invariant(all)\nok', ['ok']);
expect('#pragma whatever (1,2)\nok', ['ok']);
// #line: the next line becomes the given number.
expectLine('#line 100\nfoo', 100);
expectLine('#line 100\nfoo\nbar', 100);
{
  const r = run('#line 100\nfoo\nbar');
  check(r.ok, 'line remap run ok');
  if (r.ok) {
    check(r.tokens[0].line === 100 && r.tokens[1].line === 101, `line remap: got ${r.tokens[0]?.line}, ${r.tokens[1]?.line}`);
  }
}
// __LINE__ after #line.
expect('#line 7\n#if __LINE__ == 7\nok\n#endif', ['ok']);
expectErr('#line x\nfoo', 'invalid #line');

/* ------------------------------------------------------------------ */
/* Comments and splicing                                               */
/* ------------------------------------------------------------------ */

expect('a /* c */ b', ['a', 'b']);
expect('a/**/b', ['a', 'b']);
expect('a// comment\nb', ['a', 'b']);
expect('/*\n*/\nfoo', ['foo']);
expect('a /* c\nc2 */ b', ['a', 'b']);
expectErr('/* unterminated', 'unterminated comment');
expectErr('foo /* unterminated', 'unterminated comment');
expect('#define FOO \\\n  42\nFOO', ['42']);
expect('#define FOO \\\r\n  42\nFOO', ['42']);
// Comment spanning a spliced line.
expect('a /* c \\\n d */ b', ['a', 'b']);
// Splicing keeps physical line numbers.
{
  const r = run('#define FOO \\\n  42\nFOO');
  check(r.ok, 'splice line check ok');
  if (r.ok) check(r.tokens[0].line === 2, `spliced token line: got ${r.tokens[0]?.line}`);
}
// Tokens: identifiers, numbers, punctuators, columns.
{
  const r = run('foo bar');
  check(r.ok, 'column run ok');
  if (r.ok) check(r.tokens[0].column === 0 && r.tokens[1].column === 4, `columns: got ${r.tokens[0]?.column}, ${r.tokens[1]?.column}`);
}
expect('1.5e3 0x1F 1u 1.0f 1..2 .5', ['1.5e3', '0x1F', '1u', '1.0f', '1.', '.2', '.5']);
expect('a<<=1 a&&b a<b', ['a', '<<=', '1', 'a', '&&', 'b', 'a', '<', 'b']);
expect('a # b', ['a', '#', 'b']); // mid-line # is a token
expectErr('# 123\nfoo', 'invalid directive');
expectErr('#whatever 1\nfoo', 'invalid directive');

/* ------------------------------------------------------------------ */
/* Injected defines                                                    */
/* ------------------------------------------------------------------ */

expect('FOO', ['1'], { defines: { FOO: '1' } });
expect('FOO BAR', ['1', '2'], { defines: { FOO: '1', BAR: '2' } });
expect('FOO', ['1', '+', '2'], { defines: { FOO: '1 + 2' } });
expect('#ifdef FOO\nok\n#endif', ['ok'], { defines: { FOO: '1' } });
// Defines are overridden by explicit #define.
expect('#undef FOO\nFOO', ['FOO'], { defines: { FOO: '1' } });

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`preprocessor selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');

/** Assert the line of the first emitted token. */
function expectLine(src: string, expectedLine: number, opts?: Opts): void {
  const r = run(src, opts);
  check(r.ok, 'expected success for line check: ' + JSON.stringify(src));
  if (r.ok) {
    check(r.tokens[0]?.line === expectedLine, `first token line for ${JSON.stringify(src)}: expected ${expectedLine}, got ${r.tokens[0]?.line}`);
  }
}
