/**
 * Diagnostic: find WHY the runner (multi-page-per-context) throws
 * "Cyclic __proto__ value" at page load while the single-page debug script
 * does not. Replicates the runner's env exactly (same init scripts, same
 * URL format, same polling loop) and logs, per page:
 *   - page.url() at the moment each pageerror fires
 *   - whether window.__createSoftwareWebGLContext got defined (bundle survived)
 *   - typeof of the native WebGL globals at pageerror time
 * Run: node scripts/debug-cyclic-proto2.mjs [testPath] [numPages]
 */
import { chromium } from 'playwright';
import { startCtsServer } from '../tests/conformance/server.ts';
import { buildInterceptScript } from '../src/context-intercept.ts';
import { HARNESS_SHIM_SCRIPT } from '../tests/conformance/harness.ts';

const testPath = process.argv[2] ?? 'conformance/glsl/misc/shader-with-double-underscore.html';
const numPages = parseInt(process.argv[3] ?? '3', 10);

const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const context = await browser.newContext();
await context.addInitScript(buildInterceptScript());
await context.addInitScript(HARNESS_SHIM_SCRIPT);

const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;

for (let i = 1; i <= numPages; i++) {
  const page = await context.newPage();
  const errs = [];
  page.on('pageerror', (err) => {
    errs.push({ urlAtError: page.url(), message: err.message, stack: err.stack });
  });
  page.on('console', (msg) => {
    if (msg.type() === 'error') console.log(`  [page${i}] console.error: ${msg.text().slice(0, 200)}`);
  });
  console.log(`\n--- page ${i} -> goto ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
  // Runner-style polling snapshot loop
  const poll = () => page.evaluate(() => {
    const s = window.__softglCts;
    const d = document.getElementById('description');
    return {
      finished: s ? s.finished : false,
      description: d && d.textContent ? d.textContent : '',
    };
  });
  for (let t = 0; t < 40; t++) {
    const snap = await poll();
    if (snap.finished || /TEST COMPLETE/.test(snap.description)) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  const state = await page.evaluate(() => ({
    hasFactory: typeof window.__createSoftwareWebGLContext === 'function',
    swglVersion: window.__swglVersion ?? null,
    nativeUniformLocation: typeof globalThis.WebGLUniformLocation,
    nativeActiveInfo: typeof globalThis.WebGLActiveInfo,
    nativeShaderPrecision: typeof globalThis.WebGLShaderPrecisionFormat,
    nativeGL1: typeof globalThis.WebGLRenderingContext,
    uniformLocProtoChain: (() => {
      const p = globalThis.WebGLUniformLocation?.prototype;
      const names = [];
      let cur = p;
      for (let k = 0; k < 8 && cur; k++) { names.push(cur === Object.prototype ? 'Object.prototype' : (cur.constructor?.name ?? '?')); cur = Object.getPrototypeOf(cur); }
      return names;
    })(),
  }));
  console.log(`  page ${i} url: ${page.url()}`);
  console.log(`  state: ${JSON.stringify(state)}`);
  console.log(`  pageerrors (${errs.length}):`);
  for (const e of errs) {
    console.log(`    at url=${e.urlAtError}`);
    console.log(`    ${e.message}`);
    console.log(`    ${(e.stack ?? '').split('\n').slice(0, 8).join('\n    ')}`);
  }
  await page.close().catch(() => {});
}

await browser.close();
await server.close();
