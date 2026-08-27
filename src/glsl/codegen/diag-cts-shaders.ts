/**
 * diag-cts-shaders.ts — diagnostic measurement tool for the codegen bugfix wave.
 *
 * Extracts GLSL shader sources from the Khronos CTS pages listed below
 * (READ-ONLY /testsuites/WebGL — never modified) and runs `compileShader`
 * (+ `linkProgram` when the page has both stages) on each, printing per-shader
 * PASS/FAIL lines with the first error message. Classification (POSITIVE =
 * page expects compile/link success, NEGATIVE = page expects failure) is
 * included for the ITEM1 pages so before/after comparison can separate
 * legal-shader regressions from expected failures.
 *
 * This is a measurement tool ONLY — it fixes nothing. Run:
 *   npx tsx src/glsl/codegen/diag-cts-shaders.ts
 * Never throws fatally (per-page try/catch); exits 0 always.
 *
 * ============================================================================
 * BEFORE SUMMARY (measured on the current PRE-FIX code, 2026-03-16, HEAD)
 *   ITEM1 glsl-construct-vec4.html            : VS 25/48 ok | FS 25/48 ok |
 *         LINK 32/50 ok (+46 N/A). Compile side matches page expectations
 *         EXACTLY (25 POSITIVE pass, 23 NEGATIVE fail — no UNEXPECTED-PASS).
 *         All 18 POSITIVE link failures are the codegen matrix-arg guard:
 *         `linker: codegen failed: codegen: matrix argument in a multi-argument
 *         vector constructor` — every generated MULTI-argument vec4(...) case
 *         whose argument list contains a matrix (vec4(s, m2), vec4(s, s, m2),
 *         vec4(s, s, s, m2), ...). SINGLE-matrix-arg cases (vec4(m2)) link OK.
 *   ITEM1 glsl-construct-vec-mat-corner-cases : VS 5/5 ok | FS 5/5 ok |
 *         LINK 2/5 ok. Link failures t1/t3/t4 all from the VERTEX shaders'
 *         `gl_Position = vec4(1.0, m);` (matrix arg in multi-arg vec4 ctor):
 *         `linker: codegen failed: codegen: matrix argument in a
 *         multi-argument vector constructor`.
 *   ITEM2 CorrectComma_frag.test.html         : VS 1/1 | FS 1/1 | LINK 0/1.
 *         LINK: `linker: codegen failed: codegen: no builtin signature for 's'`
 *         (struct ctor `s(9.0, vec3(10,11,12))` in a comma expr; structNames
 *         not yet filled in CodegenLayout).
 *   ITEM2 CorrectConstFolding1_vert.test.html : VS 1/1 | FS 1/1 | LINK 0/1.
 *         LINK: `... no builtin signature for 's2'` (struct ctors s1/s2/s4).
 *   ITEM2 CorrectConstFolding2_vert.test.html : VS 1/1 | FS 1/1 | LINK 0/1.
 *         LINK: `... no builtin signature for 's5'` (struct ctors s/s2/s5...).
 *   ITEM2 'construct_struct' page search      : 0 files match under ogles/GL/
 *         (no dedicated struct-ctor page exists; the three build/ pages above
 *         carry the struct-ctor shaders).
 *   ITEM3 DepthRange_frag / DepthRange_vert   : VS 1/1 | FS 1/1 | LINK 0/1
 *         each. LINK: `linker: codegen failed: codegen: struct member on
 *         builtin path` (gl_DepthRange.near/.far/.diff member access in
 *         DepthRange_frag.frag and DepthRange_vert.vert).
 *   ITEM4 CorrectConstFolding2_vert           : compile+link identical to ITEM2
 *         entry. NOTE: 10 whole-struct ==/!= expressions (6 ==, 4 !=), ALL in
 *         const initializers (`const bool bN = ...` on const struct values
 *         s22/ss/st551/st552/st553) — const-folded at semantics, never reaches
 *         codegen; the only link blocker is the struct-ctor signature issue.
 *   ITEM4 struct_bool_vert (struct/)          : VS 1/1 | FS 1/1 | LINK 0/1
 *         (LINK: `... no builtin signature for 'sabcd'`). NOTE: struct/ dir
 *         (56 files) and equal/ dir (24 files) contain NO whole-struct ==/!=;
 *         struct_bool does member-wise bool compares only.
 *
 * Representative pre-fix failure messages (exact, as printed per LINK line):
 *   - `linker: codegen failed: codegen: matrix argument in a multi-argument vector constructor`
 *   - `linker: codegen failed: codegen: no builtin signature for '<struct name>'`
 *   - `linker: codegen failed: codegen: struct member on builtin path`
 * ============================================================================
 */
import { compileShader, linkProgram } from '../compiler.js';
import type { Shader } from '../compiler.js';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vm from 'node:vm';

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const CTS = '/testsuites/WebGL';
const CONFORMANCE = path.join(CTS, 'sdk/tests/conformance');
const OG = path.join(CONFORMANCE, 'ogles/GL');

/** Default vertex shader used by glsl-conformance-test.js when a fragment-only
 * test case supplies no vShaderSource (glsl-construct-vec4.html). */
const DEFAULT_VERT_100 = [
  'attribute vec4 vPosition;',
  'void main()',
  '{',
  '    gl_Position = vPosition;',
  '}',
].join('\n');

/** Pass-through fragment shader used by the constructor generator for every
 * vertex test case (glsl-construct-vec4.html). */
const PASS_THROUGH_FRAG_100 = [
  'precision mediump float;',
  'varying vec4 vColor;',
  'void main() {',
  '    gl_FragColor = vColor;',
  '}',
].join('\n');

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function readText(p: string): string {
  return fs.readFileSync(p, 'utf8');
}

/** 300 only when the source itself declares `#version 300 es`. */
function detectVersion(src: string): 100 | 300 {
  return /^\s*#\s*version\s+300\s+es\b/m.test(src) ? 300 : 100;
}

function decodeHtml(s: string): string {
  return s
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&');
}

type CompileOutcome = { ok: true; shader: Shader } | { ok: false; errors: { line: number; message: string }[] };

const compileCache = new Map<string, CompileOutcome>();
function compileCached(src: string, stage: 'VERTEX' | 'FRAGMENT', version: 100 | 300): CompileOutcome {
  const key = `${stage}\u0000${version}\u0000${src}`;
  let r = compileCache.get(key);
  if (!r) {
    const res = compileShader(src, { type: stage, version });
    r = res.ok ? { ok: true, shader: res.shader } : { ok: false, errors: res.errors };
    compileCache.set(key, r);
  }
  return r;
}

const linkCache = new Map<string, { ok: boolean; log?: string }>();
function linkCached(vs: ShaderCase, fs: ShaderCase): { ok: boolean; log?: string } {
  const vsVer = vs.version ?? detectVersion(vs.src);
  const fsVer = fs.version ?? detectVersion(fs.src);
  const key = `${vs.stage}\u0000${vsVer}\u0000${vs.src}||${fs.stage}\u0000${fsVer}\u0000${fs.src}`;
  let r = linkCache.get(key);
  if (!r) {
    const vc = compileCached(vs.src, vs.stage, vsVer);
    const fc = compileCached(fs.src, fs.stage, fsVer);
    if (!vc.ok || !fc.ok) {
      r = { ok: false, log: '(stage compile failed)' };
    } else {
      const l = linkProgram(vc.shader, fc.shader);
      r = l.ok ? { ok: true } : { ok: false, log: l.log };
    }
    linkCache.set(key, r);
  }
  return r;
}

function firstCompileErr(r: CompileOutcome): string {
  if (r.ok) return '';
  const e = r.errors[0];
  return `L${e.line}: ${e.message}`;
}

/* ------------------------------------------------------------------ */
/* Reporting                                                           */
/* ------------------------------------------------------------------ */

interface ShaderCase {
  id: string;
  stage: 'VERTEX' | 'FRAGMENT';
  src: string;
  /** Page expectation for the COMPILE of this shader. */
  expected: 'PASS' | 'FAIL';
  /** Default/pass-through shaders: not counted in page VS/FS totals. */
  aux?: boolean;
  version?: 100 | 300;
}

interface LinkCase {
  id: string;
  vs: ShaderCase;
  fs: ShaderCase;
  linkExpected: 'PASS' | 'FAIL';
}

interface PageStats {
  vsOk: number; vsTotal: number;
  fsOk: number; fsTotal: number;
  linkOk: number; linkTotal: number; linkNA: number;
}

const report = {
  lines: [] as string[],
  line(s: string): void {
    this.lines.push(s);
  },
  page(name: string, fn: (stats: PageStats) => void): void {
    const stats: PageStats = { vsOk: 0, vsTotal: 0, fsOk: 0, fsTotal: 0, linkOk: 0, linkTotal: 0, linkNA: 0 };
    let errMsg = '';
    try {
      fn(stats);
    } catch (e) {
      errMsg = e instanceof Error ? e.message : String(e);
    }
    if (errMsg) {
      this.line(`PAGE ${name}: EXTRACT-ERR ${errMsg}`);
      return;
    }
    const fmt = (ok: number, total: number) =>
      total === 0 ? 'N/A' : `${ok === total ? 'PASS' : 'FAIL'} ${ok}/${total}`;
    const linkSuffix = stats.linkNA > 0 ? ` (+${stats.linkNA} N/A)` : '';
    this.line(
      `PAGE ${name}: VS ${fmt(stats.vsOk, stats.vsTotal)} | FS ${fmt(stats.fsOk, stats.fsTotal)} | ` +
      `LINK ${fmt(stats.linkOk, stats.linkTotal)}${linkSuffix}`,
    );
  },
};

/** Compile one shader, print its PASS/FAIL line, count it. */
function compileAndReport(stats: PageStats, c: ShaderCase, extraNote = ''): CompileOutcome {
  const version = c.version ?? detectVersion(c.src);
  const r = compileCached(c.src, c.stage, version);
  const pos = c.expected === 'PASS' ? 'POS' : 'NEG';
  const status = r.ok ? 'PASS' : 'FAIL';
  const unexpected = r.ok && c.expected === 'FAIL' ? ' [UNEXPECTED-PASS]' : '';
  const msg = r.ok ? '' : ` [${firstCompileErr(r)}]`;
  report.line(`  CASE ${c.id} [${pos}] ${c.stage}: ${status}${msg}${unexpected}${extraNote ? ' ' + extraNote : ''}`);
  if (!c.aux) {
    if (c.stage === 'VERTEX') { stats.vsTotal++; if (r.ok) stats.vsOk++; }
    else { stats.fsTotal++; if (r.ok) stats.fsOk++; }
  }
  return r;
}

/** Link a pair (compiling both stages as needed); N/A when a stage fails. */
function linkAndReport(stats: PageStats, lc: LinkCase): void {
  const l = linkCached(lc.vs, lc.fs);
  if (l.log === '(stage compile failed)') {
    stats.linkNA++;
    report.line(`  LINK ${lc.id}: N/A (stage compile failed)`);
    return;
  }
  stats.linkTotal++;
  const status = l.ok ? 'PASS' : 'FAIL';
  const unexpected = l.ok && lc.linkExpected === 'FAIL' ? ' [UNEXPECTED-PASS]' : '';
  const msg = l.ok ? '' : ` [${l.log}]`;
  report.line(`  LINK ${lc.id}: ${status}${msg}${unexpected}`);
  if (l.ok) stats.linkOk++;
}

/* ------------------------------------------------------------------ */
/* AST struct-info walker (local + global declarations)                */
/* ------------------------------------------------------------------ */

interface StructInfo {
  decls: number;        // user struct declarations (global + local)
  ctorCalls: number;    // struct-constructor call sites
  compares: number;     // ==/!= binary exprs with a struct operand
  comparesInConst: number; // ... located inside a `const` initializer
}

function collectStructInfo(shader: Shader | null, src: string): StructInfo {
  const info: StructInfo = { decls: 0, ctorCalls: 0, compares: 0, comparesInConst: 0 };
  if (!shader) {
    // Compile failed — source grep fallback (rough).
    info.decls = (src.match(/\bstruct\s+\w+/g) || []).length;
    return info;
  }
  for (const d of shader.ast.declarations) {
    if (d.kind === 'struct-decl') {
      info.decls++;
    } else if (d.kind === 'global-var-decl') {
      if (d.type.base.kind === 'struct-definition') info.decls++;
      const inConst = d.type.qualifiers.storage === 'const';
      for (const vd of d.declarators) if (vd.init) walkExpr(vd.init, inConst, info);
    } else if (d.kind === 'function-definition') {
      walkStmt(d.body, info);
    }
  }
  return info;
}

function walkStmt(s: import('../ast.js').Stmt, info: StructInfo): void {
  switch (s.kind) {
    case 'compound':
      for (const x of s.body) walkStmt(x, info);
      break;
    case 'decl-stmt': {
      if (s.type.base.kind === 'struct-definition') info.decls++;
      const inConst = s.type.qualifiers.storage === 'const';
      for (const vd of s.declarators) if (vd.init) walkExpr(vd.init, inConst, info);
      break;
    }
    case 'expr-stmt':
      if (s.expr) walkExpr(s.expr, false, info);
      break;
    case 'if':
      walkExpr(s.cond, false, info);
      walkStmt(s.then, info);
      if (s.else) walkStmt(s.else, info);
      break;
    case 'for':
      if (s.init) walkStmt(s.init, info);
      if (s.cond) walkExpr(s.cond, false, info);
      if (s.update) walkExpr(s.update, false, info);
      walkStmt(s.body, info);
      break;
    case 'while':
      walkExpr(s.cond, false, info);
      walkStmt(s.body, info);
      break;
    case 'do-while':
      walkStmt(s.body, info);
      walkExpr(s.cond, false, info);
      break;
    case 'switch':
      walkExpr(s.expr, false, info);
      walkStmt(s.body, info);
      break;
    case 'case':
      if (s.value) walkExpr(s.value, false, info);
      break;
    case 'return':
      if (s.value) walkExpr(s.value, false, info);
      break;
    default:
      break; // break/continue/discard/empty
  }
}

function walkExpr(e: import('../ast.js').Expr, inConst: boolean, info: StructInfo): void {
  switch (e.kind) {
    case 'binary': {
      if (
        (e.op === '==' || e.op === '!=') &&
        (e.left.resolvedType?.kind === 'struct' || e.right.resolvedType?.kind === 'struct')
      ) {
        info.compares++;
        if (inConst) info.comparesInConst++;
      }
      walkExpr(e.left, inConst, info);
      walkExpr(e.right, inConst, info);
      break;
    }
    case 'call': {
      if (e.resolvedType?.kind === 'struct') info.ctorCalls++;
      for (const a of e.args) walkExpr(a, inConst, info);
      break;
    }
    case 'unary':
      walkExpr(e.operand, inConst, info);
      break;
    case 'assign':
      walkExpr(e.target, inConst, info);
      walkExpr(e.value, inConst, info);
      break;
    case 'ternary':
      walkExpr(e.cond, inConst, info);
      walkExpr(e.whenTrue, inConst, info);
      walkExpr(e.whenFalse, inConst, info);
      break;
    case 'index':
      walkExpr(e.object, inConst, info);
      walkExpr(e.index, inConst, info);
      break;
    case 'member':
      walkExpr(e.object, inConst, info);
      break;
    case 'comma':
      for (const x of e.exprs) walkExpr(x, inConst, info);
      break;
    default:
      break; // literal/identifier
  }
}

/* ------------------------------------------------------------------ */
/* ITEM1 — matrix argument in vector constructors                      */
/* ------------------------------------------------------------------ */

/** Evaluate the REAL CTS generator (glsl-constructor-tests-generator.js) in a
 * Node vm with a minimal wtu stub, then generate the vec4 test cases exactly
 * as glsl-construct-vec4.html does at runtime. */
function evalVec4Generator(): any[] {
  const genSrc = readText(path.join(CTS, 'sdk/tests/js/glsl-constructor-tests-generator.js'));
  const wtu = {
    error: (...a: unknown[]) => console.error('WTU-ERR', ...a),
    replaceParams(str: string, ...objs: Record<string, unknown>[]): string {
      return str.replace(/\$\(([^)]+)\)/g, (_m: string, p1: string) => {
        for (const o of objs) if (o[p1] !== undefined) return String(o[p1]);
        throw new Error(`unknown string param '${p1}'`);
      });
    },
  };
  const sandbox: Record<string, unknown> = { WebGLTestUtils: wtu, console };
  vm.createContext(sandbox);
  vm.runInContext(genSrc, sandbox);
  const G = sandbox.GLSLConstructorTestsGenerator as any;
  return G.getConstructorTests('vec4', G.getDefaultTestSet('vec4'));
}

function item1(): void {
  report.line('==== ITEM1: matrix argument in vector constructors ====');

  // --- glsl-construct-vec4.html (programmatic generator, 96 cases) ---
  report.page('glsl-construct-vec4.html', (stats) => {
    const cases = evalVec4Generator();
    let neg = 0;
    for (const c of cases) {
      if (c.vShaderSuccess === false || c.fShaderSuccess === false) neg++;
    }
    report.line(`  NOTE glsl-construct-vec4.html: ${cases.length} generated cases (48 vertex + 48 fragment), ${neg} NEGATIVE (page expects compile failure), ${cases.length - neg} POSITIVE`);
    const passThrough: ShaderCase = { id: '<pass-through frag>', stage: 'FRAGMENT', src: PASS_THROUGH_FRAG_100, expected: 'PASS', aux: true };
    const defaultVert: ShaderCase = { id: '<default vert>', stage: 'VERTEX', src: DEFAULT_VERT_100, expected: 'PASS', aux: true };
    cases.forEach((c, i) => {
      const num = String(i).padStart(2, '0');
      if (typeof c.vShaderSource === 'string') {
        const vs: ShaderCase = { id: `vec4#${num} vs`, stage: 'VERTEX', src: c.vShaderSource, expected: c.vShaderSuccess ? 'PASS' : 'FAIL' };
        compileAndReport(stats, vs);
        linkAndReport(stats, { id: `vec4#${num} vs`, vs, fs: passThrough, linkExpected: c.linkSuccess ? 'PASS' : 'FAIL' });
      } else {
        const fs: ShaderCase = { id: `vec4#${num} fs`, stage: 'FRAGMENT', src: c.fShaderSource, expected: c.fShaderSuccess ? 'PASS' : 'FAIL' };
        compileAndReport(stats, fs);
        linkAndReport(stats, { id: `vec4#${num} fs`, vs: defaultVert, fs, linkExpected: c.linkSuccess ? 'PASS' : 'FAIL' });
      }
    });
  });

  // --- glsl-construct-vec-mat-corner-cases.html (script-block shaders) ---
  report.page('glsl-construct-vec-mat-corner-cases.html', (stats) => {
    const html = readText(path.join(CONFORMANCE, 'glsl/constructors/glsl-construct-vec-mat-corner-cases.html'));
    const scripts = new Map<string, string>();
    for (const m of html.matchAll(/<script id="([^"]+)" type="text\/something-not-javascript">([\s\S]*?)<\/script>/g)) {
      scripts.set(m[1], decodeHtml(m[2]));
    }
    interface T { vId?: string; vOk?: boolean; fId?: string; fOk?: boolean; linkOk?: boolean; msg?: string; }
    const tests: T[] = [];
    for (const m of html.matchAll(/tests\.push\(\{([\s\S]*?)\}\);/g)) {
      const b = m[1];
      const t: T = {
        vId: b.match(/vShaderSource:\s*wtu\.getScript\(\s*"([^"]+)"\s*\)/)?.[1],
        vOk: b.match(/vShaderSuccess:\s*(\w+)/)?.[1] === 'true',
        fId: b.match(/fShaderSource:\s*wtu\.getScript\(\s*"([^"]+)"\s*\)/)?.[1],
        fOk: b.match(/fShaderSuccess:\s*(\w+)/)?.[1] === 'true',
        linkOk: b.match(/linkSuccess:\s*(\w+)/)?.[1] === 'true',
        msg: b.match(/passMsg:\s*"([^"]*)"/)?.[1],
      };
      tests.push(t);
    }
    report.line(`  NOTE glsl-construct-vec-mat-corner-cases.html: ${scripts.size} embedded shader blocks, ${tests.length} test entries (all expect success → all POSITIVE)`);
    tests.forEach((t, i) => {
      const vsSrc = t.vId ? scripts.get(t.vId) : undefined;
      const fsSrc = t.fId ? scripts.get(t.fId) : undefined;
      if (vsSrc === undefined || fsSrc === undefined) {
        throw new Error(`test #${i} references missing script block (vId=${t.vId} fId=${t.fId}); found: ${[...scripts.keys()].join(', ')}`);
      }
      const vs: ShaderCase = { id: t.vId ?? `t${i} vs`, stage: 'VERTEX', src: vsSrc, expected: t.vOk ? 'PASS' : 'FAIL' };
      const fs: ShaderCase = { id: t.fId ?? `t${i} fs`, stage: 'FRAGMENT', src: fsSrc, expected: t.fOk ? 'PASS' : 'FAIL' };
      compileAndReport(stats, vs);
      compileAndReport(stats, fs);
      linkAndReport(stats, { id: `t${i} (${t.msg ?? ''})`, vs, fs, linkExpected: t.linkOk ? 'PASS' : 'FAIL' });
    });
  });
}

/* ------------------------------------------------------------------ */
/* ogles runner-config parsing (shared by ITEM2/ITEM3/ITEM4)           */
/* ------------------------------------------------------------------ */

interface OglesEntry {
  name: string;
  vsPath: string;
  fsPath: string;
  /** `compstat`/`linkstat` are present on build/ pages (true = expect OK). */
  compstat?: boolean;
  linkstat?: boolean;
}

/** Parse an OpenGLESTestRunner.run({...}) config: split the `"tests": [...]`
 * array into brace-balanced entry objects and extract each entry's
 * name / testProgram shader paths / compstat / linkstat. Deterministic —
 * immune to the ordering of "name" vs "testProgram" within an entry
 * (build/ pages put testProgram before the name, biuDepthRange after). */
function parseOglesRunner(html: string): OglesEntry[] {
  const entries: OglesEntry[] = [];
  const testsIdx = html.indexOf('"tests"');
  if (testsIdx === -1) return entries;
  const arrBrace = html.indexOf('[', testsIdx);
  if (arrBrace === -1) return entries;
  let depth = 0;
  let arrEnd = -1;
  for (let i = arrBrace; i < html.length; i++) {
    if (html[i] === '[') depth++;
    else if (html[i] === ']') {
      depth--;
      if (depth === 0) { arrEnd = i; break; }
    }
  }
  if (arrEnd === -1) return entries;
  const arr = html.slice(arrBrace + 1, arrEnd);
  let i = 0;
  while (i < arr.length) {
    while (i < arr.length && arr[i] !== '{') i++;
    if (i >= arr.length) break;
    let d = 0;
    let j = i;
    for (; j < arr.length; j++) {
      if (arr[j] === '{') d++;
      else if (arr[j] === '}') {
        d--;
        if (d === 0) break;
      }
    }
    if (j >= arr.length) break;
    const elem = arr.slice(i, j + 1);
    i = j + 1;
    const name = elem.match(/"name":\s*"([^"]+)"/)?.[1];
    if (!name) continue;
    const tpIdx = elem.indexOf('"testProgram"');
    if (tpIdx === -1) continue;
    const tpBrace = elem.indexOf('{', tpIdx);
    if (tpBrace === -1) continue;
    d = 0;
    let tpEnd = -1;
    for (let k = tpBrace; k < elem.length; k++) {
      if (elem[k] === '{') d++;
      else if (elem[k] === '}') {
        d--;
        if (d === 0) { tpEnd = k; break; }
      }
    }
    if (tpEnd === -1) continue;
    const tpObj = elem.slice(tpBrace, tpEnd + 1);
    const vsPath = tpObj.match(/"vertexShader":\s*"([^"]+)"/)?.[1];
    const fsPath = tpObj.match(/"fragmentShader":\s*"([^"]+)"/)?.[1];
    if (!vsPath || !fsPath) continue;
    const hasComp = elem.includes('"compstat"');
    const hasLink = elem.includes('"linkstat"');
    entries.push({
      name,
      vsPath,
      fsPath,
      compstat: hasComp ? elem.match(/"compstat":\s*(\w+)/)?.[1] === 'true' : undefined,
      linkstat: hasLink ? elem.match(/"linkstat":\s*(\w+)/)?.[1] === 'true' : undefined,
    });
  }
  return entries;
}

/** Compile + link one ogles test pair; `expect` = page expectation (all
 * ogles "build"/"compare" pages expect success). Prints struct info notes. */
function runOglesPair(stats: PageStats, entry: OglesEntry, htmlDir: string, note: string): void {
  const vsPath = path.resolve(htmlDir, entry.vsPath);
  const fsPath = path.resolve(htmlDir, entry.fsPath);
  const vsSrc = readText(vsPath);
  const fsSrc = readText(fsPath);
  const vsExpected = entry.compstat === false ? 'FAIL' : 'PASS';
  const fsExpected = entry.compstat === false ? 'FAIL' : 'PASS';
  const linkExpected = entry.linkstat === false ? 'FAIL' : 'PASS';
  const vs: ShaderCase = { id: path.basename(vsPath), stage: 'VERTEX', src: vsSrc, expected: vsExpected };
  const fs: ShaderCase = { id: path.basename(fsPath), stage: 'FRAGMENT', src: fsSrc, expected: fsExpected };
  const vc = compileAndReport(stats, vs, note);
  const fc = compileAndReport(stats, fs, note);
  linkAndReport(stats, { id: entry.name, vs, fs, linkExpected });
  const vInfo = collectStructInfo(vc.ok ? vc.shader : null, vsSrc);
  const fInfo = collectStructInfo(fc.ok ? fc.shader : null, fsSrc);
  const fmtInfo = (i: StructInfo) =>
    `struct-decls=${i.decls} struct-ctor-calls=${i.ctorCalls} struct-compares=${i.compares}` +
    (i.compares > 0 ? ` (${i.comparesInConst}/${i.compares} in const initializers)` : '');
  report.line(`  NOTE ${entry.name}: VS [${fmtInfo(vInfo)}] | FS [${fmtInfo(fInfo)}]`);
}

/* ------------------------------------------------------------------ */
/* ITEM2 — user struct constructor calls                               */
/* ------------------------------------------------------------------ */

function item2(): void {
  report.line('==== ITEM2: user struct constructor calls ====');

  // Search for a dedicated 'construct_struct' page across ogles/GL.
  let found: string[] = [];
  const walk = (dir: string): void => {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) walk(p);
      else if (/\.(vert|frag|html)$/.test(ent.name)) {
        try {
          if (readText(p).includes('construct_struct')) found.push(p);
        } catch { /* unreadable — skip */ }
      }
    }
  };
  walk(OG);
  report.line(found.length === 0
    ? `  NOTE 'construct_struct' page search: NOT FOUND anywhere under ${OG}`
    : `  NOTE 'construct_struct' page search: found in ${found.length} file(s):\n    ${found.join('\n    ')}`);

  // The three build/ pages (compstat/linkstat true → POSITIVE).
  const buildHtml = readText(path.join(OG, 'build/build_001_to_008.html'));
  const entries = parseOglesRunner(buildHtml).filter((e) => /Correct(Comma|ConstFolding1|ConstFolding2)_/.test(e.name));
  if (entries.length === 0) throw new Error('no CorrectComma/CorrectConstFolding entries found in build_001_to_008.html');
  for (const e of entries) {
    report.page(e.name, (stats) => {
      runOglesPair(stats, e, path.join(OG, 'build'), '');
    });
  }
}

/* ------------------------------------------------------------------ */
/* ITEM3 — gl_DepthRange                                               */
/* ------------------------------------------------------------------ */

function item3(): void {
  report.line('==== ITEM3: gl_DepthRange ====');
  const html = readText(path.join(OG, 'biuDepthRange/biuDepthRange_001_to_002.html'));
  const entries = parseOglesRunner(html).filter((e) => e.name.startsWith('DepthRange'));
  if (entries.length === 0) throw new Error('no DepthRange entries found in biuDepthRange_001_to_002.html');
  for (const e of entries) {
    report.page(e.name, (stats) => {
      runOglesPair(stats, e, path.join(OG, 'biuDepthRange'), '');
    });
  }
}

/* ------------------------------------------------------------------ */
/* ITEM4 — struct ==/!=                                                */
/* ------------------------------------------------------------------ */

function item4(): void {
  report.line('==== ITEM4: struct ==/!=');

  const structDir = path.join(OG, 'struct');
  const equalDir = path.join(OG, 'equal');
  const structFiles = fs.existsSync(structDir) ? fs.readdirSync(structDir).filter((f) => /\.(vert|frag)$/.test(f)) : [];
  const equalFiles = fs.existsSync(equalDir) ? fs.readdirSync(equalDir).filter((f) => /\.(vert|frag)$/.test(f)) : [];
  report.line(`  NOTE struct/ dir: ${structFiles.length} shader files; equal/ dir: ${equalFiles.length} shader files (vector equal() builtin tests — no structs)`);

  // CorrectConstFolding2_vert.vert: whole-struct ==/!= on CONST structs
  // (s22 == s22, ss == ss, st551 == st552, st551 != st553, ...). Reuses the
  // ITEM2 pair; the compile/link cache makes this free.
  const buildHtml = readText(path.join(OG, 'build/build_001_to_008.html'));
  const cf2 = parseOglesRunner(buildHtml).find((e) => e.name.startsWith('CorrectConstFolding2_'));
  if (!cf2) throw new Error('CorrectConstFolding2 entry not found');
  report.page('CorrectConstFolding2_vert (struct ==/!=, const-folded)', (stats) => {
    runOglesPair(stats, cf2, path.join(OG, 'build'), '');
  });

  // struct_bool pair from struct/ (member-wise bool compares — the closest
  // struct-comparison page in struct/; no whole-struct == exists there).
  const structHtml = readText(path.join(structDir, 'struct_001_to_008.html'));
  const sb = parseOglesRunner(structHtml).find((e) => e.name.startsWith('struct_bool_vert'));
  if (!sb) throw new Error('struct_bool_vert entry not found');
  report.page('struct_bool_vert (struct/ member-wise bool compares)', (stats) => {
    runOglesPair(stats, sb, structDir, '');
  });
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

try {
  item1();
  item2();
  item3();
  item4();
} catch (e) {
  console.error(`diag-cts-shaders: FATAL (should not happen): ${e instanceof Error ? e.stack ?? e.message : String(e)}`);
}

for (const l of report.lines) console.log(l);
console.log('diag-cts-shaders: done (exit 0)');
process.exit(0);
