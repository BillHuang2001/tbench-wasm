import { chromium } from 'playwright';

async function main() {
  const browser = await chromium.launch();
  const ctxN = await browser.newContext();
  const pageN = await ctxN.newPage();
  pageN.on('console', (m) => console.log('PAGE:', m.type(), m.text()));
  pageN.on('pageerror', (e) => console.log('PAGEERROR:', e.message));
  await pageN.goto('file:///probe-page.html');
  await pageN.waitForTimeout(1000);
  const r = await pageN.evaluate(() => (window as any).__probe);
  console.log('probe:', r ? `gl=${r.gl} aniso=${r.anisoExt}` : 'UNDEFINED');
  await browser.close();
}
main().catch(e => { console.error(e); process.exit(1); });
