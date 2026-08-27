/**
 * Diagnostic: capture the init-script error in-page (bundle wrapped in
 * try/catch storing window.__initError) and compare vs pure-native.
 */
import fs from 'node:fs';
import { chromium } from 'playwright';
import { startCtsServer } from '../tests/conformance/server.ts';
import { HARNESS_SHIM_SCRIPT } from '../tests/conformance/harness.ts';

const testPath = process.argv[2] ?? 'conformance/glsl/misc/shader-with-double-underscore.html';
const mode = process.argv[3] ?? 'wrapped'; // 'wrapped' | 'native' | 'raw'
const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const context = await browser.newContext();

if (mode !== 'native') {
  const bundle = fs.readFileSync('./renderer.js', 'utf-8');
  const script = mode === 'wrapped'
    ? `(function(){ try {\n${bundle}\n} catch (e) { window.__initError = { message: e.message, stack: e.stack }; } })();`
    : bundle; // raw: bundle as-is (like buildInterceptScript)
  await context.addInitScript(script);
  await context.addInitScript(HARNESS_SHIM_SCRIPT);
}

context.on('page', (p) => {
  p.on('pageerror', (err) => console.log(`  PAGEERROR "${err.message}" url=${p.url()}`));
});

const page = await context.newPage();
page.on('pageerror', (err) => console.log(`  [late] PAGEERROR "${err.message}" url=${page.url()}`));
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;
console.log(`mode=${mode} loading ${url}`);
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
const state = await page.evaluate(() => ({
  initError: window.__initError ?? null,
  hasFactory: typeof window.__createSoftwareWebGLContext === 'function',
  description: (document.getElementById('description')?.textContent ?? ''),
}));
console.log('state:', JSON.stringify(state, null, 2));
await browser.close();
await server.close();
