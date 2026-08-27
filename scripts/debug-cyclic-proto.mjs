/**
 * Debug: reproduce the "Cyclic __proto__ value" pageerror with a full stack.
 * Run: node scripts/debug-cyclic-proto.mjs [testPath]
 *
 * GOTCHA: buildInterceptScript() reads WEBGL_SOFTWARE_RENDERER first. The agent
 * shell commonly exports WEBGL_SOFTWARE_RENDERER=/app/renderer.js (nonexistent
 * in worktrees) -> the RENDERER_NOT_FOUND stub is injected instead of the real
 * bundle (getContext throws, tests fail with "Unable to fetch WebGL rendering
 * context", no cyclic error). Run with:
 *   WEBGL_SOFTWARE_RENDERER=./renderer.js node scripts/debug-cyclic-proto.mjs <path>
 * Root cause of the cyclic error (investigated): src/gl/objects/index.ts
 * references WebGLUniformLocation/WebGLActiveInfo/WebGLShaderPrecisionFormat in
 * value position without importing them; esbuild renames the classes to ...2 and
 * leaves the barrel references as global lookups -> setPrototypeOf(nativeProto,
 * nativeProto) -> TypeError at bundle load. Fix: value-import the 3 aux classes
 * in index.ts (see src/gl/CONTEXT.md).
 */
import { chromium } from 'playwright';
import { startCtsServer } from '../tests/conformance/server.ts';
import { buildInterceptScript } from '../src/context-intercept.ts';
import { HARNESS_SHIM_SCRIPT } from '../tests/conformance/harness.ts';

const testPath = process.argv[2] ?? 'conformance/glsl/misc/shader-with-double-underscore.html';

const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const context = await browser.newContext();
await context.addInitScript(buildInterceptScript());
await context.addInitScript(HARNESS_SHIM_SCRIPT);
const page = await context.newPage();
page.on('pageerror', (err) => {
  console.log('=== PAGEERROR ===');
  console.log('message:', err.message);
  console.log('stack:', err.stack);
});
page.on('console', (msg) => {
  if (msg.type() === 'error') console.log('console.error:', msg.text());
});
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;
console.log('Loading:', url);
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(5000);
const desc = await page.evaluate(() => document.getElementById('description')?.textContent ?? '(none)');
console.log('description:', desc.slice(0, 300));
await browser.close();
await server.close();
