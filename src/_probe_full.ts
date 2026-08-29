// Scratch analyzer (task-5): full-image residual maps, 3 scenes, correct stride.
// Delete after use. Run: npx tsx src/_probe_full.ts
import sharp from 'sharp';

const SCENES: { title: string; ours: string; golden: string }[] = [
  { title: 'Hill Valley', ours: '/tmp/post-HV.png', golden: '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/Hillvalley.png' },
  { title: 'Sponza', ours: '/tmp/post-Sponza.png', golden: '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/Sponza.png' },
  { title: 'The car', ours: '/tmp/post-TheCar.png', golden: '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/TheCar.png' },
];

async function load(p: string): Promise<{ w: number; h: number; c: number; data: Buffer }> {
  const { data, info } = await sharp(p).raw().toBuffer({ resolveWithObject: true });
  return { w: info.width, h: info.height, c: info.channels, data };
}

function luma(d: Buffer, i: number, c: number): number {
  return 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
}

async function main(): Promise<void> {
  for (const s of SCENES) {
    const ours = await load(s.ours);
    const gold = await load(s.golden);
    const { w, h } = ours;
    const oc = ours.c, gc = gold.c;
    console.log(`\n===== ${s.title} ===== ${w}x${h} c=${oc} (golden ${gold.w}x${gold.h} c=${gc})`);
    if (gold.w !== w || gold.h !== h) { console.log('SIZE MISMATCH'); continue; }
    const signed = new Float32Array(w * h);
    let over8 = 0, over20 = 0, over50 = 0, maxAbs = 0, meanAbs = 0;
    const hist = new Map<number, number>();
    for (let i = 0; i < w * h; i++) {
      const lp = luma(ours.data, i * oc, oc), lg = luma(gold.data, i * gc, gc);
      const d = Math.abs(lp - lg);
      signed[i] = lp - lg;
      meanAbs += d; if (d > maxAbs) maxAbs = d;
      if (d > 8) over8++; if (d > 20) over20++; if (d > 50) over50++;
      if (d > 8) {
        const b = Math.round((lp - lg) / 10) * 10;
        hist.set(b, (hist.get(b) ?? 0) + 1);
      }
    }
    meanAbs /= w * h;
    console.log(`mean|lumaΔ|=${meanAbs.toFixed(3)} max=${maxAbs.toFixed(1)} px>8: ${over8} px>20: ${over20} px>50: ${over50}`);
    const hk = [...hist.keys()].sort((a, b) => a - b);
    console.log('signed-delta buckets (10-wide) for px>8:', hk.map(k => `${k}:${hist.get(k)}`).join(' '));
    // Full image map, 100x40 cells
    const cw = 100, ch = 40;
    const cellW = w / cw, cellH = h / ch;
    console.log('signed map (# >+15, + >+4, - <-4, X <-15, . |d|>1):');
    for (let cy = 0; cy < ch; cy++) {
      let row = String(cy).padStart(2) + ' ';
      for (let cx = 0; cx < cw; cx++) {
        let sum = 0, n = 0;
        const x0 = Math.floor(cx * cellW), x1 = Math.min(w, Math.floor((cx + 1) * cellW));
        const y0 = Math.floor(cy * cellH), y1 = Math.min(h, Math.floor((cy + 1) * cellH));
        for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { sum += signed[y * w + x]; n++; }
        const m = sum / n;
        row += m > 15 ? '#' : m > 4 ? '+' : m < -15 ? 'X' : m < -4 ? '-' : (Math.abs(m) > 1 ? '.' : ' ');
      }
      console.log(row);
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
