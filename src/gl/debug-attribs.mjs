/**
 * Debug: run a single CTS test page against the software renderer and dump the
 * per-assertion console output (FAIL lines + tail + description).
 *
 * Run (IMPORTANT: env override for the intercept script):
 *   WEBGL_SOFTWARE_RENDERER=./renderer.js node src/gl/debug-attribs.mjs <testPath> [waitMs] [webglVersion]
 *
 * e.g. node src/gl/debug-attribs.mjs conformance/attribs/gl-disabled-vertex-attrib.html 8000 1
 * TEMP debug tool — delete before final commit (or keep as a committed utility).
 */
import { chromium } from 'playwright';
import { startCtsServer } from '../../tests/conformance/server.ts';
import { buildInterceptScript } from '../context-intercept.ts';
import { HARNESS_SHIM_SCRIPT } from '../../tests/conformance/harness.ts';

const testPath = process.argv[2] ?? 'conformance/attribs/gl-disabled-vertex-attrib.html';
const waitMs = parseInt(process.argv[3] ?? '8000', 10);
const version = process.argv[4] ?? '1';

const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'] });
const context = await browser.newContext();
await context.addInitScript(buildInterceptScript());
await context.addInitScript(HARNESS_SHIM_SCRIPT);
const page = await context.newPage();
page.on('pageerror', (err) => {
  console.log('=== PAGEERROR ===');
  console.log('message:', err.message);
  console.log('stack:', (err.stack ?? '').slice(0, 1200));
});
page.on('console', (msg) => {
  if (msg.type() === 'error') console.log('console.error:', msg.text().slice(0, 500));
});
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=${version}`;
console.log('Loading:', url);
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(waitMs);
const out = await page.evaluate(() => {
  const c = document.getElementById('console');
  const d = document.getElementById('description');
  return { console: c?.textContent ?? '(none)', desc: d?.textContent ?? '(none)' };
});
const lines = out.console.split('\n');
const fails = lines.filter((l) => l.includes('FAIL'));
const passes = lines.filter((l) => l.includes('PASS'));
console.log(`=== ${passes.length} PASS / ${fails.length} FAIL ===`);
console.log('=== FAIL lines (first 60) ===');
console.log(fails.slice(0, 60).join('\n'));
console.log('=== tail (last 40) ===');
console.log(lines.slice(-40).join('\n'));
console.log('=== description ===');
console.log(out.desc.slice(0, 500));
await browser.close();
await server.close();
