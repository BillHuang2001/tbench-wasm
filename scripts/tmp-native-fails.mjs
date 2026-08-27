import { chromium } from 'playwright';
import { startCtsServer } from '../tests/conformance/server.ts';
const testPath = process.argv[2];
const server = await startCtsServer({ root: '/testsuites/WebGL', host: '127.0.0.1', port: 0 });
const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const page = await browser.newPage();
const url = `${server.baseUrl}sdk/tests/${testPath}?webglVersion=1`;
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForTimeout(4000);
const state = await page.evaluate(() => ({
  desc: (document.getElementById('description')?.textContent ?? '').slice(-200),
  fails: (document.getElementById('console')?.textContent ?? '').split('FAIL').filter(s => s.trim()).map(s => s.trim().slice(0, 150)).slice(-8),
}));
console.log(JSON.stringify(state, null, 2));
await browser.close();
await server.close();
