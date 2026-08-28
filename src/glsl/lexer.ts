/**
 * lexer.ts — the tokenizer stage of the GLSL ES 1.00 / 3.00 compiler (spec §3.1–§3.6).
 *
 * Pipeline position: `compileShader` runs preprocessor → lexer → parser →
 * semantics. This module consumes the preprocessor's `RawToken[]` stream
 * (preprocessor.ts — identifiers, numbers incl. hex/octal/float with u/f
 * suffixes, punctuators, and string-literal tokens from macro stringize) and
 * classifies every token: identifiers, keywords, numbers (int/uint/float with
 * version rules) and operators/punctuation. Longest-match operator grouping
 * was already done by the preprocessor, so each RawToken maps to at most one
 * op token — except `^^`, which the lexer splits into two `^` tokens in ES 3.00.
 *
 * Version-dependent lexical rules enforced here:
 * - `u`/`U` suffix (uint literals): ES 3.00 only (error in 100).
 * - Octal literals (`0`-prefixed integers): ES 1.00 only (error in 300).
 * - `^^`: ES 1.00 only (split into two `^` in 300; the parser then rejects the
 *   syntax).
 * - `attribute`/`varying`: keywords in ES 1.00; per GLSL ES 3.00 §3.6 they are
 *   NOT keywords in 300 and lex as plain identifiers there (declarations using
 *   them fail type resolution in semantics → compile error).
 *
 * Keyword classification follows the actual spec keyword/reserved lists, not
 * a naive union: in ES 1.00 only the ES 1.00 keywords (incl. `attribute`/
 * `varying`/`in`/`out`/`inout`) and the ES 1.00 reserved-for-future-use words
 * (`switch`/`default`/`flat`/`sampler3D`/`sampler2DShadow`) lex as keywords;
 * every other ES 3.00 keyword (`uint`, `layout`, `case`, `centroid`, `smooth`,
 * `noperspective`, `uvec*`, `mat2x2*`, `sampler2DArray`, `isampler*`,
 * `usampler*`, ...) is a plain identifier in 100 per GLSL ES 1.00 §3.6 — the
 * Khronos CTS `shader-with-non-reserved-words-*-of-8` tests require exactly
 * this (they compile shaders using those words as identifiers in WebGL1).
 * In ES 3.00 all ES 3.00 keywords lex as keywords. All other version-specific
 * rejection is left to the parser, which can emit clear messages ("'X' is
 * reserved in GLSL ES 1.00" / "'X' requires GLSL ES 3.00") instead of generic
 * identifier errors. Bitwise operators lex in both versions too; the
 * parser/semantics rejects them in 100 with proper messages.
 *
 * Errors: 1-based lines (propagated from the preprocessor), 0-based columns,
 * Khronos-style free-form messages; all errors are collected and scanning
 * continues, so a shader reports every lexical problem at once.
 */
import type { CompileError } from './compiler.js';
import type { RawToken } from './preprocessor.js';

/* ------------------------------------------------------------------ */
/* Public interface (consumed by the parser stage)                     */
/* ------------------------------------------------------------------ */

export type Token =
  | { kind: 'identifier'; name: string; line: number; column: number }
  | { kind: 'keyword'; name: string; line: number; column: number }
  | { kind: 'int'; value: number; line: number; column: number }
  | { kind: 'uint'; value: number; line: number; column: number }
  | { kind: 'float'; value: number; line: number; column: number }
  | { kind: 'op'; text: string; line: number; column: number }; // operators & punctuation (single or multi-char)

export type LexResult = { ok: true; tokens: Token[] } | { ok: false; errors: CompileError[] };

/** Tokenize the preprocessor's raw token stream into GLSL tokens. */
export function tokenize(raw: RawToken[], version: 100 | 300): LexResult {
  const tokens: Token[] = [];
  const errors: CompileError[] = [];

  for (const r of raw) {
    const { text, line, column } = r;

    // String literals only arise from macro stringize (`#x`) — GLSL has no
    // string literals, so they are always an error (incl. unterminated ones,
    // which the preprocessor passes through to end-of-line).
    if (text.startsWith('"')) {
      errors.push({ line, message: 'string literals are not allowed in GLSL' });
      continue;
    }

    if (IDENT_RE.test(text)) {
      if (isKeyword(text, version)) {
        tokens.push({ kind: 'keyword', name: text, line, column });
        continue;
      }
      // Reserved-identifier rules (WebGL §6.2 / ANGLE): GLSL ES 1.00
      // future-reserved words, `__` anywhere, and the webgl_/_webgl prefixes
      // are rejected as identifiers in BOTH version modes. `gl_` prefixes are
      // NOT rejected here (needs a builtin whitelist — semantics Scope.declare).
      const reservedMsg = reservedIdentifierError(text);
      if (reservedMsg !== null) {
        errors.push({ line, message: reservedMsg });
        continue;
      }
      tokens.push({ kind: 'identifier', name: text, line, column });
      continue;
    }

    if (OPERATORS.has(text)) {
      if (text === '^^' && version === 300) {
        // ES 3.00 has no `^^` operator: split into two bitwise-xor tokens
        // (the parser rejects the resulting syntax with its own message).
        tokens.push({ kind: 'op', text: '^', line, column });
        tokens.push({ kind: 'op', text: '^', line, column: column + 1 });
      } else {
        tokens.push({ kind: 'op', text, line, column });
      }
      continue;
    }

    if (isNumberStart(text)) {
      const n = parseNumber(text, version);
      if (n.ok) {
        tokens.push({ kind: n.n.kind, value: n.n.value, line, column });
      } else {
        errors.push({ line, message: n.message });
      }
      continue;
    }

    // Invalid character (e.g. `@`, `$`, `#` outside a directive) or garbage
    // produced by token pasting (`a##b` concatenates without re-tokenizing).
    if (text.length === 1) {
      errors.push({ line, message: `unexpected character '${text}'` });
    } else {
      errors.push({ line, message: `unexpected token '${text}'` });
    }
  }

  if (errors.length > 0) return { ok: false, errors };
  return { ok: true, tokens };
}

/* ------------------------------------------------------------------ */
/* Keywords                                                             */
/* ------------------------------------------------------------------ */

/** Keywords common to GLSL ES 1.00 and 3.00. */
const KEYWORDS_COMMON: ReadonlySet<string> = new Set([
  'const', 'uniform', 'break', 'continue', 'do', 'for', 'while', 'if', 'else',
  'float', 'int', 'void', 'bool', 'true', 'false',
  'lowp', 'mediump', 'highp', 'precision', 'invariant', 'discard', 'return',
  'struct', 'mat2', 'mat3', 'mat4',
  'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4', 'bvec2', 'bvec3', 'bvec4',
  'sampler2D', 'samplerCube',
]);

/** Keywords of BOTH versions not in COMMON (function-parameter qualifiers). */
const KEYWORDS_BOTH: ReadonlySet<string> = new Set(['in', 'out', 'inout']);

/** ES 1.00 keywords; NOT keywords in ES 3.00 (§3.6) — plain identifiers there. */
const KEYWORDS_100_ONLY: ReadonlySet<string> = new Set(['attribute', 'varying']);

/**
 * ES 1.00 reserved-for-future-use words (§3.6) that are also ES 3.00 keywords:
 * keywords in BOTH versions (the CTS non-reserved-words tests skip these).
 */
const KEYWORDS_100_RESERVED: ReadonlySet<string> = new Set([
  'switch', 'default', 'flat', 'sampler3D', 'sampler2DShadow',
]);

/**
 * ES 3.00-only keywords. In ES 1.00 they are NOT reserved (spec §3.6) and lex
 * as plain identifiers — the CTS `shader-with-non-reserved-words-*-of-8` tests
 * require them to compile as identifiers in WebGL1 shaders. Declarations using
 * them as types in a 100 shader fail type resolution in semantics.
 */
const KEYWORDS_300_ONLY: ReadonlySet<string> = new Set([
  'uint',
  'uvec2', 'uvec3', 'uvec4',
  'mat2x2', 'mat2x3', 'mat2x4', 'mat3x2', 'mat3x3', 'mat3x4', 'mat4x2', 'mat4x3', 'mat4x4',
  'sampler2DArray', 'samplerCubeShadow', 'sampler2DArrayShadow',
  'isampler2D', 'isampler3D', 'isamplerCube', 'isampler2DArray',
  'usampler2D', 'usampler3D', 'usamplerCube', 'usampler2DArray',
  'layout', 'centroid', 'smooth', 'noperspective',
  'case', 'precise',
]);

function isKeyword(name: string, version: 100 | 300): boolean {
  if (KEYWORDS_COMMON.has(name) || KEYWORDS_BOTH.has(name)) return true;
  if (version === 100) return KEYWORDS_100_ONLY.has(name) || KEYWORDS_100_RESERVED.has(name);
  return KEYWORDS_300_ONLY.has(name) || KEYWORDS_100_RESERVED.has(name);
}

/**
 * GLSL ES 1.00 §3.6 reserved-for-future-use words (the Khronos CTS
 * `GLSL_1_0_17_FutureWords` list) that are NOT already keywords — rejected as
 * identifiers in BOTH version modes (matches ANGLE; CTS reserved-words tests).
 * Excluded here:
 * - `switch`/`default`/`flat`/`sampler3D`/`sampler2DShadow` — already
 *   keywords via KEYWORDS_100_RESERVED above (do not duplicate that path).
 * - The ES 3.00-only keywords (`uint`, `layout`, `centroid`, `smooth`,
 *   `noperspective`, `uvec2-4`, `mat2x2*`, `sampler2DArray`,
 *   `sampler2DArrayShadow`, `samplerCubeShadow`, `isampler*`, `usampler*`,
 *   `case`, `precise`, ...) — in version-100 shaders they lex as IDENTIFIERS
 *   per the documented CTS deviation (shader-with-non-reserved-words tests
 *   require them to compile as identifiers in WebGL1).
 */
const RESERVED_FUTURE: ReadonlySet<string> = new Set([
  'asm', 'class', 'union', 'enum', 'typedef', 'template', 'this', 'packed',
  'goto', 'inline', 'noinline', 'volatile', 'public', 'static', 'extern',
  'external', 'interface', 'long', 'short', 'double', 'half', 'fixed',
  'unsigned', 'superp', 'input', 'output',
  'hvec2', 'hvec3', 'hvec4', 'dvec2', 'dvec3', 'dvec4',
  'fvec2', 'fvec3', 'fvec4',
  'sampler1D', 'sampler1DShadow', 'sampler2DRect', 'sampler3DRect',
  'sampler2DRectShadow', 'sizeof', 'cast', 'namespace', 'using',
]);

/**
 * Reserved-identifier rules (WebGL §6.2 "Identifiers", matching ANGLE):
 * - GLSL ES 1.00 future-reserved words (RESERVED_FUTURE).
 * - names containing `__` anywhere (`__foo`, `foo__bar`, `foo__bar__baz`).
 * - names starting with `webgl_` or `_webgl`.
 * Returns an error message or null. `gl_` prefixes are NOT rejected here —
 * builtin-name shadowing is checked in semantics (Scope.declare) with a
 * builtin whitelist.
 */
function reservedIdentifierError(name: string): string | null {
  if (RESERVED_FUTURE.has(name)) return `'${name}' : reserved word`;
  if (name.includes('__')) return `'${name}' : identifiers may not contain '__'`;
  if (name.startsWith('webgl_') || name.startsWith('_webgl')) {
    return `'${name}' : identifiers may not start with 'webgl_' or '_webgl'`;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Operators & punctuation (GLSL ES 1.00 §3.1 / ES 3.00 §3.1)          */
/* ------------------------------------------------------------------ */

const OPERATORS: ReadonlySet<string> = new Set([
  '(', ')', '[', ']', '{', '}', ',', '.', ';', ':', '?', '=',
  '+', '-', '*', '/', '%', '<', '>', '<=', '>=', '==', '!=',
  '&&', '||', '^^', '!', '~', '&', '|', '^', '<<', '>>', '++', '--',
  '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '^=', '|=',
]);

/* ------------------------------------------------------------------ */
/* Numbers                                                              */
/* ------------------------------------------------------------------ */

const IDENT_RE = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const FLOAT_RE = /^(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$/;
const INT_RE = /^\d+$/;

const isDigit = (c: string): boolean => c >= '0' && c <= '9';
const isHexDigit = (c: string): boolean =>
  (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
const isSuffixChar = (c: string): boolean => c === 'u' || c === 'U' || c === 'f' || c === 'F';

/** Same token-start test the preprocessor uses for numbers. */
const isNumberStart = (text: string): boolean =>
  text.length > 0 && (isDigit(text[0]) || (text[0] === '.' && text.length > 1 && isDigit(text[1])));

interface NumberValue {
  kind: 'int' | 'uint' | 'float';
  value: number;
}

type NumberResult = { ok: true; n: NumberValue } | { ok: false; message: string };

/**
 * Classify and evaluate a number token. The preprocessor already performed
 * max-munch scanning, so the text is either a well-formed literal or garbage
 * from token pasting; malformed shapes are reported as errors (never
 * silently truncated).
 */
function parseNumber(text: string, version: 100 | 300): NumberResult {
  if (text.length >= 2 && text[0] === '0' && (text[1] === 'x' || text[1] === 'X')) {
    // Hexadecimal integer (hex digits greedily, then an optional u/U suffix;
    // f/F after hex digits are hex digits, so a hex float is impossible).
    const rest = text.slice(2);
    let i = 0;
    while (i < rest.length && isHexDigit(rest[i])) i++;
    const digits = rest.slice(0, i);
    const suffix = rest.slice(i);
    if (digits === '') return { ok: false, message: `invalid hexadecimal literal '${text}'` };
    if (suffix === '') return { ok: true, n: { kind: 'int', value: parseInt(digits, 16) } };
    if (suffix === 'u' || suffix === 'U') {
      if (version === 100) return { ok: false, message: "'u' suffix requires GLSL ES 3.00" };
      return { ok: true, n: { kind: 'uint', value: parseInt(digits, 16) } };
    }
    return { ok: false, message: `invalid suffix '${suffix}' on numeric literal` };
  }

  // Decimal/octal integer or float: split the trailing u/U/f/F suffix run.
  let end = text.length;
  while (end > 0 && isSuffixChar(text[end - 1])) end--;
  const core = text.slice(0, end);
  const suffix = text.slice(end);
  const isFloat = core.includes('.') || core.includes('e') || core.includes('E');

  if (suffix === 'u' || suffix === 'U') {
    if (isFloat) return { ok: false, message: "'u' suffix is not allowed on float literals" };
    if (version === 100) return { ok: false, message: "'u' suffix requires GLSL ES 3.00" };
    const n = parseInteger(core, text, version);
    if (!n.ok) return n;
    return { ok: true, n: { kind: 'uint', value: n.value } };
  }
  if (suffix.includes('u') || suffix.includes('U')) {
    return { ok: false, message: `invalid suffix '${suffix}' on numeric literal` };
  }
  if (isFloat || suffix !== '') {
    // Float: f/F suffix, or a fractional/exponent form.
    if (!FLOAT_RE.test(core)) return { ok: false, message: `invalid numeric literal '${text}'` };
    // GLSL float literals are IEEE single-precision (GLSL ES 3.00 §4.1.4):
    // values too large round to ±Infinity, too small to ±0 (or subnormal).
    // Math.fround gives exactly that from the double parse.
    return { ok: true, n: { kind: 'float', value: Math.fround(parseFloat(core)) } };
  }
  const n = parseInteger(core, text, version);
  if (!n.ok) return n;
  return { ok: true, n: { kind: 'int', value: n.value } };
}

/** Evaluate an unsigned integer core: decimal, or octal in ES 1.00. */
function parseInteger(
  core: string,
  text: string,
  version: 100 | 300,
): { ok: true; value: number } | { ok: false; message: string } {
  if (!INT_RE.test(core)) return { ok: false, message: `invalid numeric literal '${text}'` };
  if (core === '0') return { ok: true, value: 0 };
  if (core[0] === '0') {
    // 0-prefixed multi-digit integer = octal, which ES 3.00 removed.
    if (version === 300) return { ok: false, message: 'octal literals are not allowed in GLSL ES 3.00' };
    if (!/^[0-7]+$/.test(core)) return { ok: false, message: `invalid octal literal '${core}'` };
    return { ok: true, value: parseInt(core, 8) };
  }
  return { ok: true, value: parseInt(core, 10) };
}
