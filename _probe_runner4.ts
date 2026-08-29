import { chromium } from 'playwright';
import { readFileSync } from 'fs';

const pageUrl = 'file:///probe-page.html';
interface Probe { gl: boolean; anisoExt: boolean; mips: Record<string, number[][]>; renders: Record<string, number[]> }
function compare(a: number[], b: number[]): { same: boolean; meanAbs: number; maxAbs: number; diffs: number } {
  if (!a || !b) return { same: false, meanAbs: -1, maxAbs: -1, diffs: -1 };
  if (a.length !== b.length) return { same: false, meanAbs: -2, maxAbs: -2, diffs: -2 };
  let s = 0, m = 0, d = 0;
  for (let i = 0; i < a.length; i++) { const dd = Math.abs(a[i] - b[i]); if (dd > 0) d++; s += dd; if (dd > m) m = dd; }
  return { same: d === 0, meanAbs: s / a.length, maxAbs: m, diffs: d };
}
async function main() {
  const browser = await chromium.launch();
  const ctxN = await browser.newContext();
  const pageN = await ctxN.newPage();
  pageN.on('pageerror', (e) => console.log('NATIVE PAGEERROR:', e.message));
  await pageN.goto(pageUrl);
  await pageN.waitForTimeout(500);
  const native = (await pageN.evaluate(() => (window as any).__probe)) as Probe | undefined;
  await ctxN.close();
  const ctxS = await browser.newContext();
  const pageS = await ctxS.newPage();
  pageS.on('pageerror', (e) => console.log('SOFT PAGEERROR:', e.message));
  const bundle = readFileSync('/r.js', 'utf8');
  await pageS.addInitScript({ content: `
    ${bundle}
    (function () {
      const orig = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (type, attrs) {
        if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') {
          const r = window.__createSoftwareWebGLContext(this, attrs, type);
          if (r) return r;
        }
        return orig.apply(this, arguments);
      };
    })();
  `});
  await pageS.goto(pageUrl);
  await pageS.waitForTimeout(500);
  const soft = (await pageS.evaluate(() => (window as any).__probe)) as Probe | undefined;
  await ctxS.close();
  await browser.close();
  if (!native || !soft) { console.log('native:', !!native, 'soft:', !!soft); return; }
  console.log('native gl:', native.gl, 'anisoExt:', native.anisoExt, '| soft gl:', soft.gl, 'anisoExt:', soft.anisoExt);
  console.log('\n=== MIP CONTENT (native vs ours) ===');
  for (const key of Object.keys(native.mips)) {
    const nm = native.mips[key], sm = soft.mips[key];
    const parts: string[] = [];
    for (const lk of Object.keys(nm)) {
      const r = compare(nm[lk], sm[lk]);
      parts.push(`${lk}:${r.same ? 'SAME' : `DIFF meanAbs=${r.meanAbs.toFixed(3)} max=${r.maxAbs} n=${r.diffs}`}`);
    }
    console.log(key.padEnd(14), parts.join('  '));
  }
  console.log('\n=== RENDER (native vs ours) ===');
  for (const key of Object.keys(native.renders)) {
    const r = compare(native.renders[key], soft.renders[key]);
    console.log(key.padEnd(22), r.same ? 'SAME' : `DIFF meanAbs=${r.meanAbs.toFixed(3)} max=${r.maxAbs} diffs=${r.diffs}`);
  }
}
main().catch(e => { console.error(e); process.exit(1); });
