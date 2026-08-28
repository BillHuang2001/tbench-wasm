/**
 * selftest-lexer.ts — sanity checks for the GLSL ES lexer stage.
 *
 * Run: npx tsx src/glsl/selftest-lexer.ts
 *
 * Verifies: number forms (int/octal/hex/uint/float/exp/suffix) in both
 * versions with exact kind+value assertions; operator longest-match results
 * (`<<=`, `++`, `^^` → one op in 100 vs two `^` in 300, `>=`); keyword
 * classification for representative words of each set (attribute in 100 =
 * keyword, in 300 = identifier; in/out/flat/layout/switch in both = keyword);
 * identifier forms; errors (`@` char, `"str"` string token, `1.0u`, octal in
 * 300, paste garbage); multi-error collection; line/column propagation
 * (crafted RawToken positions, incl. the `^^` split columns); and
 * preprocessor→lexer integration. Prints "OK" and exits 0 on success.
 */
import { tokenize } from './lexer.js';
import type { Token, LexResult } from './lexer.js';
import { preprocess } from './preprocessor.js';

let failures = 0;
let checks = 0;

function check(cond: boolean, msg: string): void {
  checks++;
  if (!cond) {
    failures++;
    console.error('FAIL: ' + msg);
  }
}

/** Serialize a token unambiguously: `kind:value` (op uses its text). */
function ser(t: Token): string {
  switch (t.kind) {
    case 'identifier':
      return 'id:' + t.name;
    case 'keyword':
      return 'kw:' + t.name;
    case 'int':
    case 'uint':
    case 'float':
      return t.kind + ':' + t.value;
    case 'op':
      return 'op:' + t.text;
  }
}

function lex(texts: string[], version: 100 | 300): Token[] {
  const raw = texts.map((t, i) => ({ text: t, line: i + 1, column: 0 }));
  const res = tokenize(raw, version);
  check(
    res.ok,
    `lex should succeed for ${JSON.stringify(texts)} (v${version})` + (res.ok ? '' : ': ' + JSON.stringify(res.errors)),
  );
  return res.ok ? res.tokens : [];
}

function expect(texts: string[], version: 100 | 300, expected: string[], msg?: string): void {
  const got = lex(texts, version).map(ser);
  check(
    JSON.stringify(got) === JSON.stringify(expected),
    (msg ?? 'tokens') + ` for ${JSON.stringify(texts)} (v${version}): expected ${JSON.stringify(expected)}, got ${JSON.stringify(got)}`,
  );
}

function lexErr(texts: string[], version: 100 | 300, msgPart: string, expectedErrors = 1): void {
  const raw = texts.map((t, i) => ({ text: t, line: i + 1, column: 0 }));
  const res = tokenize(raw, version);
  check(!res.ok, `expected lex failure for ${JSON.stringify(texts)} (v${version})`);
  if (!res.ok) {
    const joined = res.errors.map((e) => e.message).join(' | ');
    check(
      joined.includes(msgPart),
      `lex error for ${JSON.stringify(texts)} should contain '${msgPart}', got: ${joined}`,
    );
    check(
      res.errors.length === expectedErrors,
      `lex error count for ${JSON.stringify(texts)}: expected ${expectedErrors}, got ${res.errors.length} (${joined})`,
    );
  }
}

/* ------------------------------------------------------------------ */
/* Integer literals (ES 1.00)                                          */
/* ------------------------------------------------------------------ */

expect(['123'], 100, ['int:123']);
expect(['0'], 100, ['int:0']);
expect(['00'], 100, ['int:0']);
expect(['017'], 100, ['int:15']);
expect(['0777'], 100, ['int:511']);
expect(['0x1F'], 100, ['int:31']);
expect(['0X1F'], 100, ['int:31']);
expect(['0x0'], 100, ['int:0']);
expect(['0xdeadBEEF'], 100, ['int:3735928559']);
expect(['0x7fffffff'], 100, ['int:2147483647']);

/* ------------------------------------------------------------------ */
/* Integer literals (ES 3.00: octal removed)                           */
/* ------------------------------------------------------------------ */

expect(['123'], 300, ['int:123']);
expect(['0'], 300, ['int:0']);
expect(['0x1F'], 300, ['int:31']);
expect(['0X1F'], 300, ['int:31']);
lexErr(['017'], 300, 'octal');
lexErr(['00'], 300, 'octal');
lexErr(['0777'], 300, 'octal');

/* ------------------------------------------------------------------ */
/* uint literals (ES 3.00 only)                                        */
/* ------------------------------------------------------------------ */

expect(['1u'], 300, ['uint:1']);
expect(['1U'], 300, ['uint:1']);
expect(['0u'], 300, ['uint:0']);
expect(['0x1Fu'], 300, ['uint:31']);
expect(['0X1Fu'], 300, ['uint:31']);
expect(['123u'], 300, ['uint:123']);
lexErr(['017u'], 300, 'octal');
lexErr(['1u'], 100, "'u' suffix requires GLSL ES 3.00");
lexErr(['1U'], 100, "'u' suffix requires GLSL ES 3.00");
lexErr(['0x1Fu'], 100, "'u' suffix requires GLSL ES 3.00");

/* ------------------------------------------------------------------ */
/* Float literals (both versions)                                      */
/* ------------------------------------------------------------------ */

expect(['1.'], 100, ['float:1']);
expect(['.5'], 100, ['float:0.5']);
expect(['1.5'], 100, ['float:1.5']);
expect(['1e3'], 100, ['float:1000']);
expect(['1E3'], 100, ['float:1000']);
expect(['1.5e-2'], 100, ['float:0.014999999664723873']); // f32 rounding of 0.015
expect(['1E+3'], 100, ['float:1000']);
expect(['1.e3'], 100, ['float:1000']);
expect(['1f'], 100, ['float:1']);
expect(['1F'], 100, ['float:1']);
expect(['1.5f'], 100, ['float:1.5']);
expect(['1.0F'], 100, ['float:1']);
expect(['.5f'], 100, ['float:0.5']);
expect(['1.5'], 300, ['float:1.5']);
expect(['1e3'], 300, ['float:1000']);
expect(['1f'], 300, ['float:1']);
expect(['0.5'], 300, ['float:0.5']);
expect(['0e3'], 300, ['float:0']); // exponent form is a float, not octal
expect(['0.0f'], 300, ['float:0']);
expect(['.25e2'], 100, ['float:25']);

/* ------------------------------------------------------------------ */
/* Number errors                                                       */
/* ------------------------------------------------------------------ */

lexErr(['1.0u'], 100, "'u' suffix is not allowed on float literals");
lexErr(['1.0u'], 300, "'u' suffix is not allowed on float literals");
lexErr(['1e3u'], 300, "'u' suffix is not allowed on float literals");
lexErr(['.5u'], 300, "'u' suffix is not allowed on float literals");
lexErr(['08'], 100, 'invalid octal');
lexErr(['09'], 100, 'invalid octal');
lexErr(['0x'], 100, 'invalid hexadecimal');
lexErr(['0x'], 300, 'invalid hexadecimal');
lexErr(['0xu'], 300, 'invalid hexadecimal');
lexErr(['1uf'], 300, "invalid suffix 'uf'");
lexErr(['1fu'], 300, "invalid suffix 'fu'");
lexErr(['0x1uf'], 300, "invalid suffix 'uf'");
lexErr(['1x'], 100, 'invalid numeric literal'); // paste garbage: 1##x
lexErr(['1..2'], 100, 'invalid numeric literal'); // defensive (preprocessor splits it)

/* ------------------------------------------------------------------ */
/* Operators & punctuation (longest-match results)                     */
/* ------------------------------------------------------------------ */

expect(['<<='], 100, ['op:<<=']);
expect(['>>='], 300, ['op:>>=']);
expect(['<<', '>>'], 100, ['op:<<', 'op:>>']);
expect(['++'], 100, ['op:++']);
expect(['--'], 300, ['op:--']);
expect(['>='], 100, ['op:>=']);
expect(['<=', '==', '!='], 100, ['op:<=', 'op:==', 'op:!=']);
expect(['&&', '||'], 300, ['op:&&', 'op:||']);
expect(['+='], 100, ['op:+=']);
expect(['-=', '*=', '/=', '%=', '&=', '^=', '|='], 100, [
  'op:-=', 'op:*=', 'op:/=', 'op:%=', 'op:&=', 'op:^=', 'op:|=',
]);
expect(
  ['(', ')', '[', ']', '{', '}', ',', '.', ';', ':', '?', '=', '+', '-', '*', '/', '%', '<', '>', '!', '~', '&', '|', '^'],
  100,
  [
    'op:(', 'op:)', 'op:[', 'op:]', 'op:{', 'op:}', 'op:,', 'op:.', 'op:;', 'op::', 'op:?', 'op:=',
    'op:+', 'op:-', 'op:*', 'op:/', 'op:%', 'op:<', 'op:>', 'op:!', 'op:~', 'op:&', 'op:|', 'op:^',
  ],
);

// `^^` is ES 1.00-only: one token in 100, split into two `^` in 300.
expect(['^^'], 100, ['op:^^']);
expect(['^^'], 300, ['op:^', 'op:^']);
expect(['a', '^^', 'b'], 100, ['id:a', 'op:^^', 'id:b']);
expect(['^', '^'], 300, ['op:^', 'op:^']);

/* ------------------------------------------------------------------ */
/* Keywords                                                            */
/* ------------------------------------------------------------------ */

// Common set: keywords in both versions.
for (const w of ['const', 'uniform', 'break', 'continue', 'do', 'for', 'while', 'if', 'else', 'float', 'int', 'void',
  'bool', 'true', 'false', 'lowp', 'mediump', 'highp', 'precision', 'invariant', 'discard', 'return', 'struct',
  'mat2', 'mat3', 'mat4', 'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4', 'bvec2', 'bvec3', 'bvec4',
  'sampler2D', 'samplerCube']) {
  expect([w], 100, ['kw:' + w]);
  expect([w], 300, ['kw:' + w]);
}

// ES 1.00-only: keywords in 100, plain identifiers in 300 (§3.6).
expect(['attribute'], 100, ['kw:attribute']);
expect(['varying'], 100, ['kw:varying']);
expect(['attribute'], 300, ['id:attribute']);
expect(['varying'], 300, ['id:varying']);
expect(['attribute', 'vec4', 'p', ';'], 100, ['kw:attribute', 'kw:vec4', 'id:p', 'op:;']);

// Function-parameter qualifiers + ES 1.00 reserved-for-future-use words that
// are also ES 3.00 keywords: keywords in BOTH versions.
for (const w of ['in', 'out', 'inout', 'switch', 'default', 'flat', 'sampler3D', 'sampler2DShadow']) {
  expect([w], 100, ['kw:' + w]);
  expect([w], 300, ['kw:' + w]);
}

// ES 3.00-only keywords: keywords in 300; plain IDENTIFIERS in 100 per GLSL ES
// 1.00 §3.6 (the CTS shader-with-non-reserved-words tests require them to
// compile as identifiers in WebGL1 shaders — declarations using them as types
// fail type resolution in semantics).
for (const w of ['uint', 'uvec2', 'uvec3', 'uvec4', 'mat2x2', 'mat2x3', 'mat2x4',
  'mat3x2', 'mat3x3', 'mat3x4', 'mat4x2', 'mat4x3', 'mat4x4',
  'sampler2DArray', 'samplerCubeShadow', 'sampler2DArrayShadow',
  'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray',
  'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray',
  'layout', 'centroid', 'smooth', 'noperspective', 'case', 'precise']) {
  expect([w], 100, ['id:' + w]);
  expect([w], 300, ['kw:' + w]);
}

/* ------------------------------------------------------------------ */
/* Identifiers                                                         */
/* ------------------------------------------------------------------ */

expect(['foo'], 100, ['id:foo']);
expect(['_x'], 100, ['id:_x']);
expect(['foo2'], 100, ['id:foo2']);
expect(['gl_Position'], 100, ['id:gl_Position']);
expect(['gl_FragCoord'], 300, ['id:gl_FragCoord']);
expect(['FOO'], 100, ['id:FOO']);
expect(['my_Vec4'], 100, ['id:my_Vec4']);
expect(['a', '_b', 'c1'], 300, ['id:a', 'id:_b', 'id:c1']);
expect(['x', '1', 'y'], 100, ['id:x', 'int:1', 'id:y']);

// ESSL 3.00 §3.6: identifiers are limited to 1024 characters. The
// preprocessor rejects source tokens at this length, so these pin the lexer's
// own check directly (it catches tokens that bypass the preprocessor, e.g.
// `##` paste results). Exactly 1024 characters stays legal
// (attrib-location-length-limits.html compiles a 1024-char attribute name).
lexErr(['a'.repeat(1025)], 300, 'longer than 1024');
lexErr(['a'.repeat(1025)], 100, 'longer than 1024');
expect(['a'.repeat(1024)], 300, ['id:' + 'a'.repeat(1024)]);

/* ------------------------------------------------------------------ */
/* Reserved identifiers (WebGL §6.2 / ANGLE)                           */
/* ------------------------------------------------------------------ */

// GLSL ES 1.00 future-reserved words: identifiers in NEITHER version (they
// are not keywords, so the lexer rejects them as reserved words; previously
// they lexed as identifiers and compiled — CTS reserved-words cluster).
for (const w of ['asm', 'class', 'union', 'enum', 'typedef', 'template', 'this', 'packed',
  'goto', 'inline', 'noinline', 'volatile', 'public', 'static', 'extern', 'external',
  'interface', 'long', 'short', 'double', 'half', 'fixed', 'unsigned', 'superp',
  'input', 'output', 'hvec2', 'hvec3', 'hvec4', 'dvec2', 'dvec3', 'dvec4',
  'fvec2', 'fvec3', 'fvec4', 'sampler1D', 'sampler1DShadow', 'sampler2DRect',
  'sampler3DRect', 'sampler2DRectShadow', 'sizeof', 'cast', 'namespace', 'using']) {
  lexErr([w], 100, 'reserved word');
  lexErr([w], 300, 'reserved word');
}

// Double underscores anywhere in the name: rejected in both versions
// (ANGLE rule; CTS reserved-words `__foo` / `foo__bar`).
lexErr(['__foo'], 100, "'__'");
lexErr(['__foo'], 300, "'__'");
lexErr(['foo__bar'], 100, "'__'");
lexErr(['foo__bar'], 300, "'__'");
lexErr(['foo__bar__baz'], 100, "'__'");

// Reserved prefixes `webgl_` / `_webgl`: rejected in both versions (the 8
// conformance/glsl/reserved/*.vert pages + the 2 misc identifier pages).
lexErr(['webgl_foo'], 100, "'webgl_'");
lexErr(['webgl_foo'], 300, "'webgl_'");
lexErr(['_webgl_foo'], 100, "'_webgl'");
lexErr(['_webgl_foo'], 300, "'_webgl'");

// `gl_` prefixes are NOT lexer errors (builtin shadowing is a semantics
// concern with a builtin whitelist): plain identifiers here.
expect(['gl_foo'], 100, ['id:gl_foo']);
expect(['gl_Foo'], 300, ['id:gl_Foo']);

// The ES 3.00-only keywords stay IDENTIFIERS in 100 (CTS deviation —
// shader-with-non-reserved-words requires them to compile in WebGL1),
// including the sampler2DArray/isampler*/usampler* family and
// samplerCubeArray (not a GLSL ES 3.00 keyword at all).
for (const w of ['uint', 'layout', 'centroid', 'smooth', 'noperspective',
  'uvec2', 'uvec3', 'uvec4', 'mat2x2', 'mat3x2', 'mat4x4',
  'sampler2DArray', 'sampler2DArrayShadow', 'samplerCubeShadow',
  'samplerCubeArray', 'samplerCubeArrayShadow',
  'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray',
  'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray',
  'case', 'precise']) {
  expect([w], 100, ['id:' + w]);
}

/* ------------------------------------------------------------------ */
/* Invalid characters & string tokens                                  */
/* ------------------------------------------------------------------ */

lexErr(['@'], 100, "unexpected character '@'");
lexErr(['@'], 300, "unexpected character '@'");
lexErr(['$'], 100, "unexpected character '$'");
lexErr(['#'], 100, "unexpected character '#'");
lexErr(['\\'], 100, 'unexpected character');
lexErr(['"str"'], 100, 'string literals are not allowed');
lexErr(['"str"'], 300, 'string literals are not allowed');
lexErr(['""'], 100, 'string literals are not allowed');
lexErr(['"abc'], 100, 'string literals are not allowed'); // unterminated
lexErr(['"a b"'], 300, 'string literals are not allowed');
lexErr(['<>'], 100, 'unexpected token'); // paste garbage: <##>
lexErr(['abc$'], 300, 'unexpected token'); // paste garbage

/* ------------------------------------------------------------------ */
/* Error list: continue scanning, collect everything                  */
/* ------------------------------------------------------------------ */

{
  const raw = [
    { text: 'foo', line: 1, column: 0 },
    { text: '@', line: 2, column: 0 },
    { text: '"s"', line: 3, column: 0 },
    { text: '1u', line: 4, column: 0 },
    { text: 'bar', line: 5, column: 0 },
  ];
  const res = tokenize(raw, 100);
  check(!res.ok, 'multi-error run fails');
  if (!res.ok) {
    check(res.errors.length === 3, `expected 3 errors, got ${res.errors.length}: ${JSON.stringify(res.errors)}`);
    check(res.errors[0].line === 2 && res.errors[0].message.includes('@'), `first error: ${JSON.stringify(res.errors[0])}`);
    check(res.errors[1].line === 3, `second error line: ${JSON.stringify(res.errors[1])}`);
    check(res.errors[2].line === 4, `third error line: ${JSON.stringify(res.errors[2])}`);
  }
}

/* ------------------------------------------------------------------ */
/* Line/column propagation (0-based columns, 1-based lines)            */
/* ------------------------------------------------------------------ */

{
  const raw = [
    { text: 'foo', line: 7, column: 3 },
    { text: '123', line: 8, column: 0 },
    { text: '+', line: 8, column: 4 },
    { text: '1.5f', line: 9, column: 1 },
    { text: '017', line: 10, column: 2 },
    { text: '0x1F', line: 11, column: 6 },
  ];
  const res = tokenize(raw, 100);
  check(res.ok, 'line/column run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    const t = res.tokens;
    check(t.length === 6, `line/col token count: got ${t.length}`);
    check(t[0].kind === 'identifier' && t[0].line === 7 && t[0].column === 3, `t0 pos: ${JSON.stringify(t[0])}`);
    check(t[1].kind === 'int' && t[1].line === 8 && t[1].column === 0, `t1 pos: ${JSON.stringify(t[1])}`);
    check(t[2].kind === 'op' && t[2].text === '+' && t[2].line === 8 && t[2].column === 4, `t2 pos: ${JSON.stringify(t[2])}`);
    check(t[3].kind === 'float' && t[3].value === 1.5 && t[3].line === 9 && t[3].column === 1, `t3 pos: ${JSON.stringify(t[3])}`);
    check(t[4].kind === 'int' && t[4].value === 15 && t[4].line === 10 && t[4].column === 2, `t4 pos: ${JSON.stringify(t[4])}`);
    check(t[5].kind === 'int' && t[5].value === 31 && t[5].line === 11 && t[5].column === 6, `t5 pos: ${JSON.stringify(t[5])}`);
  }
}

// `^^` split in 300: both tokens keep the line; the second column is +1.
{
  const raw = [{ text: '^^', line: 5, column: 2 }];
  const res = tokenize(raw, 300);
  check(res.ok, '^^ split run ok');
  if (res.ok) {
    check(res.tokens.length === 2, '^^ split count');
    check(
      res.tokens[0].kind === 'op' && res.tokens[0].text === '^' && res.tokens[0].line === 5 && res.tokens[0].column === 2,
      `^^ first: ${JSON.stringify(res.tokens[0])}`,
    );
    check(
      res.tokens[1].kind === 'op' && res.tokens[1].text === '^' && res.tokens[1].line === 5 && res.tokens[1].column === 3,
      `^^ second: ${JSON.stringify(res.tokens[1])}`,
    );
  }
}

// Error line propagation.
{
  const raw = [{ text: 'ok', line: 3, column: 0 }, { text: '@', line: 4, column: 7 }];
  const res = tokenize(raw, 100);
  check(!res.ok, 'error position run fails');
  if (!res.ok) check(res.errors[0].line === 4, `error line: got ${res.errors[0].line}`);
}

/* ------------------------------------------------------------------ */
/* Preprocessor → lexer integration                                    */
/* ------------------------------------------------------------------ */

function lexPreproc(src: string, version: 100 | 300): LexResult {
  const p = preprocess(src, { version });
  if (!p.ok) return { ok: false, errors: p.errors };
  return tokenize(p.tokens, version);
}

{
  const res = lexPreproc('foo 123 + 1.5e3', 100);
  check(res.ok, 'integration 100 run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    check(
      JSON.stringify(res.tokens.map(ser)) === JSON.stringify(['id:foo', 'int:123', 'op:+', 'float:1500']),
      `integration tokens: ${JSON.stringify(res.tokens.map(ser))}`,
    );
  }
}
{
  const res = lexPreproc('#version 300 es\nuint x = 1u;', 300);
  check(res.ok, 'integration 300 run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    check(
      JSON.stringify(res.tokens.map(ser)) === JSON.stringify(['kw:uint', 'id:x', 'op:=', 'uint:1', 'op:;']),
      `integration 300 tokens: ${JSON.stringify(res.tokens.map(ser))}`,
    );
  }
}
{
  const res = lexPreproc('attribute vec4 p;', 100);
  check(res.ok, 'integration attribute run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    check(
      JSON.stringify(res.tokens.map(ser)) === JSON.stringify(['kw:attribute', 'kw:vec4', 'id:p', 'op:;']),
      `integration attribute tokens: ${JSON.stringify(res.tokens.map(ser))}`,
    );
  }
}
{
  // ES 3.00 keywords used as identifiers in a WebGL1 shader (CTS
  // shader-with-non-reserved-words behavior): `uint`/`layout`/`case` lex as
  // identifiers in 100.
  const res = lexPreproc('uniform vec4 uint;\nstruct layout { vec4 case; };\nvoid main() { gl_Position = uint + layout(0).case; }', 100);
  check(res.ok, 'integration non-reserved run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    const got = res.tokens.map(ser);
    check(got.includes('id:uint'), `non-reserved uint is an identifier: ${JSON.stringify(got)}`);
    check(got.includes('id:layout'), `non-reserved layout is an identifier: ${JSON.stringify(got)}`);
    check(got.includes('id:case'), `non-reserved case is an identifier: ${JSON.stringify(got)}`);
  }
}
{
  // `^^` reaches the lexer as ONE raw token (preprocessor PUNCT2); v300 splits.
  const res = lexPreproc('a ^^ b', 300);
  check(res.ok, 'integration ^^ run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    check(
      JSON.stringify(res.tokens.map(ser)) === JSON.stringify(['id:a', 'op:^', 'op:^', 'id:b']),
      `integration ^^ tokens: ${JSON.stringify(res.tokens.map(ser))}`,
    );
  }
}
{
  // `1..2` is split by the preprocessor into `1.` and `.2`.
  const res = lexPreproc('1..2', 100);
  check(res.ok, 'integration 1..2 run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    check(
      JSON.stringify(res.tokens.map(ser)) === JSON.stringify(['float:1', 'float:0.20000000298023224']),
      `integration 1..2 tokens: ${JSON.stringify(res.tokens.map(ser))}`,
    );
  }
}
{
  // Octal in a 300 shader errors through the full pipeline.
  const res = lexPreproc('017', 300);
  check(!res.ok, 'integration octal 300 fails');
  if (!res.ok) check(res.errors[0].message.includes('octal'), `integration octal msg: ${JSON.stringify(res.errors[0])}`);
}
{
  // Stringize output is a lexer error.
  const res = lexPreproc('#define S(x) #x\nS(foo)', 100);
  check(!res.ok, 'integration stringize fails');
  if (!res.ok) check(res.errors[0].message.includes('string'), `integration stringize msg: ${JSON.stringify(res.errors[0])}`);
}
{
  // uint suffix through the full pipeline in 100.
  const res = lexPreproc('1u', 100);
  check(!res.ok, 'integration 1u v100 fails');
  if (!res.ok) check(res.errors[0].message.includes("'u' suffix requires GLSL ES 3.00"), `integration 1u msg: ${JSON.stringify(res.errors[0])}`);
}
{
  // Full 300 shader sketch.
  const res = lexPreproc('#version 300 es\nlayout(location = 0) in vec2 a;\nout vec4 c;\nvoid main() { c = vec4(a, 0.0, 1.0); }', 300);
  check(res.ok, 'integration shader run ok: ' + (res.ok ? '' : JSON.stringify(res.errors)));
  if (res.ok) {
    const got = res.tokens.map(ser);
    const expected = [
      'kw:layout', 'op:(', 'id:location', 'op:=', 'int:0', 'op:)',
      'kw:in', 'kw:vec2', 'id:a', 'op:;',
      'kw:out', 'kw:vec4', 'id:c', 'op:;',
      'kw:void', 'id:main', 'op:(', 'op:)', 'op:{',
      'id:c', 'op:=', 'kw:vec4', 'op:(', 'id:a', 'op:,',
      'float:0', 'op:,', 'float:1', 'op:)', 'op:;', 'op:}',
    ];
    check(JSON.stringify(got) === JSON.stringify(expected), `integration shader tokens: ${JSON.stringify(got)}`);
  }
}

/* ------------------------------------------------------------------ */
/* Report + exit                                                       */
/* ------------------------------------------------------------------ */

console.log(`lexer selftest: ${checks} checks, ${failures} failure(s)`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('OK');
