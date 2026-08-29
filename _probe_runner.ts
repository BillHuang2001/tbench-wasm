import { chromium } from 'playwright';
import { buildInterceptScript } from './src/context-intercept';

const pageUrl = 'file:///probe-page.html';

interface Probe { gl: boolean; anisoExt: boolean; mips: Record<string, number[][]>; renders: Record<string, number[]> }

function compare(a: number[], b: number[]): { same: boolean; meanAbs: number; maxAbs: number; diffs: number } {
  if (!a || !b) return { same: false, meanAbs: -1, maxAbs: -1, diffs: -1 };
  if (a.length !== b.length) return { same: false, meanAbs: -2, maxAbs: -2, diffs: -2 };
  let s = 0, m = 0, d = 0;
  for (let i = 0; i < a.length; i++) {
    const dd = Math.abs(a[i] - b[i]);
    if (dd > 0) d++;
    s += dd;
    if (dd > m) m = dd;
  }
  return { same: d === 0, meanAbs: s / a.length, maxAbs: m, diffs: d };
}

async function main() {
  const browser = await chromium.launch();
  // Native run
  const ctxN = await browser.newContext();
  const pageN = await ctxN.newPage();
  await pageN.goto(pageUrl);
  const native = (await pageN.evaluate(() => (window as any).__probe)) as Probe;
  await ctxN.close();
  // Software run
  const ctxS = await browser.newContext();
  const pageS = await ctxS.newPage();
  await pageS.addInitScript(buildInterceptScript());
  await pageS.goto(pageUrl);
  const soft = (await pageS.evaluate(() => (window as any).__probe)) as Probe;
  await ctxS.close();
  await browser.close();

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
