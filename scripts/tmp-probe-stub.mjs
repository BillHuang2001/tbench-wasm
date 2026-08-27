import { chromium } from 'playwright';
import { startCtsServer } from '../tests/conformance/server.ts';
import { buildInterceptScript } from '../src/context-intercept.ts';
import { HARNESS_SHIM_SCRIPT } from '../tests/conformance/harness.ts';

const testPath = process.argv[2] ?? 'conformance/glsl/misc/shader-with-double-underscore.html';
const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const context = await browser.newContext();
const script = buildInterceptScript();
console.log('intercept script starts with:', JSON.stringify(script.slice(0, 80)));
console.log('intercept script length:', script.length);
await context.addInitScript(script);
await context.addInitScript(HARNESS_SHIM_SCRIPT);
const page = await context.newPage();
page.on('pageerror', (e) => console.log('  PAGEERROR:', e.message.slice(0, 120)));
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(3000);
const state = await page.evaluate(() => {
  const c = document.createElement('canvas');
  let getCtxResult = 'ok';
  try { c.getContext('webgl', { antialias: false }); } catch (e) { getCtxResult = 'threw: ' + e.message; }
  return {
    title: document.title,
    hasFactory: typeof window.__createSoftwareWebGLContext === 'function',
    getCtxResult,
    desc: (document.getElementById('description')?.textContent ?? '').slice(-80),
    consoleDiv: (document.getElementById('console')?.textContent ?? '').slice(-200),
  };
});
console.log(JSON.stringify(state, null, 2));
await browser.close();
await server.close();
