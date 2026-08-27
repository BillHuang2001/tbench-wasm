import fs from 'node:fs';
import { chromium } from 'playwright';
import { startCtsServer } from '../tests/conformance/server.ts';
import { HARNESS_SHIM_SCRIPT } from '../tests/conformance/harness.ts';

const mode = process.argv[2] ?? 'none'; // none|shim|bundle|wrapper|both
const testPath = 'conformance/textures/misc/tex-image-svg-image-no-natural-width-and-height.html';
const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const context = await browser.newContext();
const bundle = fs.readFileSync('./renderer.js', 'utf-8');
const wrapper = `(function() {
  var _orig = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(type, attrs) {
    if ((type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') &&
        typeof window.__createSoftwareWebGLContext === 'function') {
      return window.__createSoftwareWebGLContext(this, attrs, type);
    }
    return _orig.call(this, type, attrs);
  };
})();`;
if (mode === 'shim') await context.addInitScript(HARNESS_SHIM_SCRIPT);
if (mode === 'bundle' || mode === 'both') await context.addInitScript(bundle);
if (mode === 'wrapper' || mode === 'both') await context.addInitScript(wrapper);
context.on('page', (p) => p.on('pageerror', (e) => console.log('  PAGEERROR:', e.message.slice(0, 120))));
const page = await context.newPage();
page.on('pageerror', (e) => console.log('  [late] PAGEERROR:', e.message.slice(0, 120)));
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(4000);
const state = await page.evaluate(() => ({
  hasFactory: typeof window.__createSoftwareWebGLContext === 'function',
  desc: (document.getElementById('description')?.textContent ?? '').slice(-80),
  consoleDiv: (document.getElementById('console')?.textContent ?? '').slice(-400),
}));
console.log(`mode=${mode}`);
console.log('hasFactory:', state.hasFactory);
console.log('desc tail:', JSON.stringify(state.desc));
console.log('console tail:', JSON.stringify(state.consoleDiv));
await browser.close();
await server.close();
