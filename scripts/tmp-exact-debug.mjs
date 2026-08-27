// EXACT debug-script setup (buildInterceptScript + shim) + early listeners
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

context.on('page', (p) => {
  console.log('[early listener attached]');
  p.on('pageerror', (err) => console.log(`  [early] PAGEERROR "${err.message.slice(0, 60)}" url=${p.url()}`));
});

const page = await context.newPage();
page.on('pageerror', (err) => console.log(`  [late] PAGEERROR "${err.message.slice(0, 60)}" url=${page.url()}`));
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;
console.log('goto', url);
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(3000);
const state = await page.evaluate(() => ({
  hasFactory: typeof window.__createSoftwareWebGLContext === 'function',
  desc: (document.getElementById('description')?.textContent ?? '').slice(-60),
}));
console.log('state:', JSON.stringify(state));
await browser.close();
await server.close();
