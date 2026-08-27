import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"] });
const context = await browser.newContext();
await context.addInitScript(() => { throw new Error('INIT_SCRIPT_THROW_MARKER'); });

context.on('page', (p) => {
  console.log('[context.on(page)] new page object, attaching early listener');
  p.on('pageerror', (err) => {
    console.log(`  [early] PAGEERROR "${err.message}" url=${p.url()}`);
  });
});

const page = await context.newPage();
console.log('newPage resolved, url=', page.url());
page.on('pageerror', (err) => console.log(`  [late] PAGEERROR "${err.message}" url=${page.url()}`));
await new Promise(r => setTimeout(r, 500));
console.log('after 500ms, url=', page.url());
await page.goto('data:text/html,<title>t</title>', { waitUntil: 'domcontentloaded' });
console.log('goto done, url=', page.url());
await new Promise(r => setTimeout(r, 500));
await browser.close();
