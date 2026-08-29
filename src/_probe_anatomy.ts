// Scratch analyzer (task-5 validation): diff anatomy ours vs golden for 3 Babylon scenes.
// Delete after use. Run: npx tsx src/_probe_anatomy.ts
import sharp from 'sharp';

const SCENES: { title: string; ours: string; golden: string }[] = [
  { title: 'Hill Valley', ours: 'tests/reports/babylon/Hillvalley.png', golden: '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/Hillvalley.png' },
  { title: 'Sponza', ours: 'tests/reports/babylon/Sponza.png', golden: '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/Sponza.png' },
  { title: 'The car', ours: 'tests/reports/babylon/TheCar.png', golden: '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/TheCar.png' },
];

async function load(p: string): Promise<{ w: number; h: number; data: Buffer }> {
  const img = sharp(p);
  const { data, info } = await img.raw().toBuffer({ resolveWithObject: true });
  return { w: info.width, h: info.height, data };
}

function asciiMap(diff: Float32Array, w: number, h: number, thresh: number, label: string): void {
  const cw = 100, ch = Math.max(1, Math.round((h / w) * cw * 0.5));
  const cellW = w / cw, cellH = h / ch;
  console.log(`--- ${label} (thresh=${thresh}) ${w}x${h} ---`);
  for (let cy = 0; cy < ch; cy++) {
    let row = '';
    for (let cx = 0; cx < cw; cx++) {
      let sum = 0, n = 0;
      const x0 = Math.floor(cx * cellW), x1 = Math.min(w, Math.floor((cx + 1) * cellW));
      const y0 = Math.floor(cy * cellH), y1 = Math.min(h, Math.floor((cy + 1) * cellH));
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { sum += diff[y * w + x]; n++; }
      const m = sum / Math.max(1, n);
      row += m > thresh ? (m > 2 * thresh ? '#' : '+') : (m > 0.001 ? '.' : ' ');
    }
    console.log(String(cy).padStart(2) + ' ' + row);
  }
}

async function main(): Promise<void> {
  for (const s of SCENES) {
    const ours = await load(s.ours);
    const gold = await load(s.golden);
    console.log(`\n===== ${s.title} ===== ours ${ours.w}x${ours.h} golden ${gold.w}x${gold.h}`);
    if (ours.w !== gold.w || ours.h !== gold.h) { console.log('SIZE MISMATCH'); continue; }
    const { w, h } = ours;
    const diff = new Float32Array(w * h);
    const lumOurs = new Float32Array(w * h);
    let meanAbs = 0, maxAbs = 0, over8 = 0, over20 = 0, meanDelta = 0;
    for (let i = 0; i < w * h; i++) {
      const o = i * 4;
      const lo = 0.299 * ours.data[o] + 0.587 * ours.data[o + 1] + 0.114 * ours.data[o + 2];
      const lg = 0.299 * gold.data[o] + 0.587 * gold.data[o + 1] + 0.114 * gold.data[o + 2];
      const d = Math.abs(lo - lg);
      diff[i] = d;
      lumOurs[i] = lo;
      meanAbs += d; if (d > maxAbs) maxAbs = d;
      if (d > 8) over8++; if (d > 20) over20++;
      meanDelta += lo - lg;
    }
    meanAbs /= w * h; meanDelta /= w * h;
    console.log(`mean|lumaΔ|=${meanAbs.toFixed(3)}  max=${maxAbs.toFixed(1)}  px>8: ${over8}  px>20: ${over20}  meanDelta=${meanDelta.toFixed(3)}`);
    let mr = 0, mg = 0, mb = 0, ma = 0;
    for (let i = 0; i < w * h; i++) {
      mr += Math.abs(ours.data[i * 4] - gold.data[i * 4]);
      mg += Math.abs(ours.data[i * 4 + 1] - gold.data[i * 4 + 1]);
      mb += Math.abs(ours.data[i * 4 + 2] - gold.data[i * 4 + 2]);
      ma += Math.abs(ours.data[i * 4 + 3] - gold.data[i * 4 + 3]);
    }
    console.log(`mean|Δ| R=${(mr / (w * h)).toFixed(3)} G=${(mg / (w * h)).toFixed(3)} B=${(mb / (w * h)).toFixed(3)} A=${(ma / (w * h)).toFixed(3)}`);
    asciiMap(diff, w, h, 8, 'diff>8');
    asciiMap(diff, w, h, 2, 'diff>2');
    const pos = new Map<number, number>();
    for (let i = 0; i < w * h; i++) {
      if (diff[i] > 8) {
        const d = Math.round(lumOurs[i] - (0.299 * gold.data[i * 4] + 0.587 * gold.data[i * 4 + 1] + 0.114 * gold.data[i * 4 + 2]));
        pos.set(d, (pos.get(d) ?? 0) + 1);
      }
    }
    const keys = [...pos.keys()].sort((a, b) => a - b);
    console.log('delta-luma histogram (ours-golden) for px>8:');
    console.log(keys.map(k => `${k}:${pos.get(k)}`).join(' '));
  }
}

main().catch(e => { console.error(e); process.exit(1); });
