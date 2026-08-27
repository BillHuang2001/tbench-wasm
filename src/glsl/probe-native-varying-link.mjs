#!/usr/bin/env node
/**
 * Probe: native Chromium (SwiftShader via ANGLE) GLSL varying-matching LINK behavior.
 *
 * Empirical ground truth for the glsl linker: what does the native WebGL
 * implementation do when a varying is declared/written in one stage but not
 * declared in the other? The shader sources mirror the mismatched-varying unit
 * test in tests/unit/glsl.test.ts ("reports a link log on failure (mismatched
 * varying)"). PROBE ONLY — does not touch any src/glsl source file.
 *
 * Run: node src/glsl/probe-native-varying-link.mjs
 * Output: JSON array of per-case results on stdout +
 *         src/glsl/probe-results.json ({ userAgent, results })
 *
 * Cases:
 *   A  v100  VS declares+WRITES varying, FS declares none  (webgl1 + webgl2)
 *   B  v100  FS declares+READS varying, VS declares none   (webgl1 + webgl2)
 *   C  v100  FS DECLARES but never reads varying, VS none  (webgl1 + webgl2)
 *   D  v300  VS `out` written, FS no matching `in`         (webgl2)
 *   E  v300  FS `in` read, VS no matching `out`            (webgl2)
 *   F  v300  FS `in` DECLARED but never read, VS no `out`  (webgl2)
 *   CONTROL v100 fully matching pair (must LINK)           (webgl1)
 */
import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

// --- shader sources (exact; mirror the unit test) ---
const VS_POS = `attribute vec4 a_position; void main() { gl_Position = a_position; }`;

const VS_WRITES_VARYING = `attribute vec4 a_position; varying vec2 v_uv; void main() { v_uv = a_position.xy; gl_Position = a_position; }`;
const FS_PLAIN = `precision mediump float; void main() { gl_FragColor = vec4(1.0, 0.5, 0.0, 1.0); }`;
const FS_READS_VARYING = `precision mediump float; varying vec2 v_uv; void main() { gl_FragColor = vec4(v_uv, 0.0, 1.0); }`;
const FS_DECLARES_UNUSED = `precision mediump float; varying vec2 v_uv; void main() { gl_FragColor = vec4(1.0); }`;

const VS300_WRITES_OUT = `#version 300 es\nin vec4 a_position; out vec2 v_uv; void main() { v_uv = a_position.xy; gl_Position = a_position; }`;
const FS300_PLAIN = `#version 300 es\nprecision mediump float; out vec4 outColor; void main() { outColor = vec4(1.0, 0.5, 0.0, 1.0); }`;
const VS300_PLAIN = `#version 300 es\nin vec4 a_position; void main() { gl_Position = a_position; }`;
const FS300_READS_IN = `#version 300 es\nprecision mediump float; in vec2 v_uv; out vec4 outColor; void main() { outColor = vec4(v_uv, 0.0, 1.0); }`;
const FS300_DECLARES_UNUSED = `#version 300 es\nprecision mediump float; in vec2 v_uv; out vec4 outColor; void main() { outColor = vec4(1.0); }`;

const cases = [
  { case: 'A', context: 'webgl', vs: VS_WRITES_VARYING, fs: FS_PLAIN },
  { case: 'A', context: 'webgl2', vs: VS_WRITES_VARYING, fs: FS_PLAIN },
  { case: 'B', context: 'webgl', vs: VS_POS, fs: FS_READS_VARYING },
  { case: 'B', context: 'webgl2', vs: VS_POS, fs: FS_READS_VARYING },
  { case: 'C', context: 'webgl', vs: VS_POS, fs: FS_DECLARES_UNUSED },
  { case: 'C', context: 'webgl2', vs: VS_POS, fs: FS_DECLARES_UNUSED },
  { case: 'D', context: 'webgl2', vs: VS300_WRITES_OUT, fs: FS300_PLAIN },
  { case: 'E', context: 'webgl2', vs: VS300_PLAIN, fs: FS300_READS_IN },
  { case: 'F', context: 'webgl2', vs: VS300_PLAIN, fs: FS300_DECLARES_UNUSED },
  { case: 'CONTROL', context: 'webgl', vs: VS_WRITES_VARYING, fs: FS_READS_VARYING },
];

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  const userAgent = await page.evaluate(() => navigator.userAgent);

  const results = [];
  for (const c of cases) {
    const r = await page.evaluate(
      ({ context, vs, fs }) => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const gl = canvas.getContext(context);
        if (!gl) {
          return { contextError: `getContext('${context}') returned null` };
        }
        const cleanLog = (s) => (s || '').replace(/\u0000/g, '').trim();
        const compile = (type, src) => {
          const sh = gl.createShader(type);
          gl.shaderSource(sh, src);
          gl.compileShader(sh);
          const ok = gl.getShaderParameter(sh, gl.COMPILE_STATUS) === true;
          const log = cleanLog(gl.getShaderInfoLog(sh));
          return { ok, log, shader: sh };
        };
        const vsRes = compile(gl.VERTEX_SHADER, vs);
        const fsRes = compile(gl.FRAGMENT_SHADER, fs);
        let linkStatus = null;
        let linkLog = '';
        if (vsRes.ok && fsRes.ok) {
          const prog = gl.createProgram();
          gl.attachShader(prog, vsRes.shader);
          gl.attachShader(prog, fsRes.shader);
          gl.linkProgram(prog);
          linkStatus = gl.getProgramParameter(prog, gl.LINK_STATUS) === true;
          linkLog = cleanLog(gl.getProgramInfoLog(prog));
          gl.deleteProgram(prog);
        }
        gl.deleteShader(vsRes.shader);
        gl.deleteShader(fsRes.shader);
        return {
          vsCompileOk: vsRes.ok,
          vsLog: vsRes.log,
          fsCompileOk: fsRes.ok,
          fsLog: fsRes.log,
          linkStatus,
          linkLog,
        };
      },
      { context: c.context, vs: c.vs, fs: c.fs },
    );
    if (r.contextError) {
      console.error(`ERROR: case ${c.case} ${c.context}: ${r.contextError}`);
      process.exitCode = 1;
    }
    results.push({ case: c.case, context: c.context, ...r });
  }

  writeFileSync(join(here, 'probe-results.json'), JSON.stringify({ userAgent, results }, null, 2) + '\n');

  // JSON array to stdout
  console.log(JSON.stringify(results, null, 2));

  // plain-text summary table
  console.log('\n=== summary ===');
  console.log(`userAgent: ${userAgent}`);
  const pad = (s, n) => String(s).padEnd(n);
  console.log(`${pad('case', 9)}${pad('context', 8)}${pad('vsCompile', 10)}${pad('fsCompile', 10)}${pad('link', 6)}  log`);
  for (const r of results) {
    const link = r.linkStatus === null || r.linkStatus === undefined ? 'n/a' : String(r.linkStatus);
    const log = r.linkLog || (r.contextError ? `ERROR: ${r.contextError}` : '');
    console.log(
      `${pad(r.case, 9)}${pad(r.context, 8)}${pad(r.vsCompileOk ?? 'n/a', 10)}${pad(r.fsCompileOk ?? 'n/a', 10)}${pad(link, 6)}  ${log}`,
    );
  }

  // control must link (proves the machinery works)
  const ctrl = results.find((r) => r.case === 'CONTROL');
  if (!ctrl || ctrl.linkStatus !== true) {
    console.error('FATAL: CONTROL case did not link — probe machinery broken');
    process.exitCode = 1;
  }
} finally {
  await browser.close();
}
