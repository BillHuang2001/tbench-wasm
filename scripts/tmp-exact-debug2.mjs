import { chromium } from 'playwright';
import { startCtsServer } from '../tests/conformance/server.ts';
import { buildInterceptScript } from '../src/context-intercept.ts';
import { HARNESS_SHIM_SCRIPT } from '../tests/conformance/harness.ts';

const testPath = process.argv[2] ?? 'conformance/textures/misc/tex-image-svg-image-no-natural-width-and-height.html';
const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const context = await browser.newContext();
await context.addInitScript(buildInterceptScript());
await context.addInitScript(HARNESS_SHIM_SCRIPT);

const page = await context.newPage();
page.on('pageerror', (err) => console.log(`  PAGEERROR: ${err.message.slice(0, 100)}`));
page.on('console', (msg) => console.log(`  [console.${msg.type()}] ${msg.text().slice(0, 200)}`));
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(2500);
const state = await page.evaluate(() => {
  const c = document.createElement('canvas');
  let r = null;
  try { r = { gl1: !!c.getContext('webgl', { antialias: false }), gl2: !!c.getContext('webgl2', { antialias: false }) }; }
  catch (e) { r = { threw: e.message }; }
  return {
    ...r,
    hasFactory: typeof window.__createSoftwareWebGLContext === 'function',
    hasWrapper: (() => { const s = HTMLCanvasElement.prototype.getContext.toString(); return s.includes('__createSoftwareWebGLContext'); })(),
    hasGL1Global: typeof globalThis.WebGLRenderingContext,
    consoleDiv: (document.getElementById('console')?.textContent ?? '').slice(-300),
  };
});
console.log('state:', JSON.stringify(state, null, 2));
await browser.close();
await server.close();
