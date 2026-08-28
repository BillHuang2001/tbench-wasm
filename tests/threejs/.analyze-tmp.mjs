// Temp analysis: per-page actual vs expected image stats (stddev = flatness detector)
import sharp from 'sharp';
import { readFileSync } from 'fs';

const report = JSON.parse(readFileSync('tests/reports/threejs/latest.json', 'utf8'));

async function stats(path, gridW = 16, gridH = 10) {
  const { data, info } = await sharp(path).raw().toBuffer({ resolveWithObject: true });
  const { width: w, height: h, channels: ch } = info;
  const px = data.length / ch;
  const sums = new Array(ch).fill(0);
  const sumsq = new Array(ch).fill(0);
  let minV = 255, maxV = 0;
  const grid = new Array(gridW * gridH).fill(0);
  const gridN = new Array(gridW * gridH).fill(0);
  const sample = new Set();
  for (let i = 0; i < px; i++) {
    for (let c = 0; c < ch; c++) {
      const v = data[i * ch + c];
      sums[c] += v; sumsq[c] += v * v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    if (i % 97 === 0) sample.add(data[i * ch] * 65536 + data[i * ch + 1] * 256 + data[i * ch + 2]);
    const gx = Math.floor((i % w) / w * gridW);
    const gy = Math.floor(Math.floor(i / w) / h * gridH);
    const gi = gy * gridW + gx;
    grid[gi] += data[i * ch]; gridN[gi]++;
  }
  const mean = sums.map(s => s / px);
  const stddev = sums.map((s, c) => Math.sqrt(sumsq[c] / px - mean[c] * mean[c]));
  const g = grid.map((v, i) => Math.round(v / gridN[i]));
  return { w, h, mean: mean.map(v => v.toFixed(1)), stddev: stddev.map(v => v.toFixed(2)), minV, maxV, uniq: sample.size, grid: g };
}

console.log('PAGE | diff% | A-mean | A-stddev | E-mean | E-stddev | flat');
for (const p of report.pages) {
  const a = await stats(p.artifacts.actual);
  const e = await stats(p.artifacts.expected);
  const diff = p.diff ? p.diff.diffPercent.toFixed(2) : '-';
  const flat = a.stddev.every(v => parseFloat(v) === 0) ? 'FLAT' : 'no';
  console.log(`${p.name} | ${diff} | ${a.mean.join(',')} | ${a.stddev.join(',')} | ${e.mean.join(',')} | ${e.stddev.join(',')} | ${flat}`);
  for (let gy = 0; gy < 10; gy++) {
    console.log('   A ' + a.grid.slice(gy * 16, gy * 16 + 16).map(v => (v < 32 ? '.' : v < 80 ? '+' : v < 140 ? 'o' : v < 200 ? 'O' : '#')).join(''));
  }
}
