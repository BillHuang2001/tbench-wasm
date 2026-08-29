import sharp from 'sharp';
import { readdirSync } from 'fs';

const dir = 'tests/reports/babylon/';
const goldenDir = '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/';

async function main() {
  const ours = await sharp(dir + 'Hillvalley.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const gold = await sharp(goldenDir + 'Hillvalley.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height } = ours.info;
  const o = ours.data, g = gold.data;
  let sumO = 0, sumG = 0, sumDiff = 0, sumAbsDiff = 0;
  let brighter = 0, darker = 0;
  let over5 = 0, over10 = 0, over20 = 0, over40 = 0, over60 = 0;
  const regionDiff: { [k: string]: { n: number; sum: number; abs: number } } = {};
  const rows = 4, cols = 4;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      const lo = (o[i] + o[i+1] + o[i+2]) / 3;
      const lg = (g[i] + g[i+1] + g[i+2]) / 3;
      sumO += lo; sumG += lg;
      const d = lo - lg;
      sumDiff += d; sumAbsDiff += Math.abs(d);
      const ad = Math.abs(d);
      if (ad > 60) over60++; else if (ad > 40) over40++;
      else if (ad > 20) over20++; else if (ad > 10) over10++; else if (ad > 5) over5++;
      if (d > 1) brighter++; else if (d < -1) darker++;
      const rx = Math.floor(x / width * cols), ry = Math.floor(y / height * rows);
      const k = `${ry},${rx}`;
      regionDiff[k] = regionDiff[k] || { n: 0, sum: 0, abs: 0 };
      regionDiff[k].n++; regionDiff[k].sum += d; regionDiff[k].abs += Math.abs(d);
    }
  }
  const n = width * height;
  console.log('size:', width + 'x' + height);
  console.log('mean luma OURS:', (sumO/n).toFixed(2), 'GOLDEN:', (sumG/n).toFixed(2), 'delta:', (sumDiff/n).toFixed(3));
  console.log('mean |delta|:', (sumAbsDiff/n).toFixed(3));
  console.log('pixels brighter(>1):', brighter, 'darker(<-1):', darker);
  console.log('|d|>5:', over5, '>10:', over10, '>20:', over20, '>40:', over40, '>60:', over60);
  console.log('region mean delta (row=top..bottom, col=left..right):');
  for (let ry = 0; ry < rows; ry++) {
    const line: string[] = [];
    for (let rx = 0; rx < cols; rx++) {
      const r = regionDiff[`${ry},${rx}`];
      line.push((r.sum / r.n).toFixed(2));
    }
    console.log('  row', ry, line.join('  '));
  }
  // Also: luminance correlation in a mid region - histogram of deltas
  const hist: number[] = new Array(41).fill(0);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 3;
    const d = (o[i] + o[i+1] + o[i+2] - g[i] - g[i+1] - g[i+2]) / 3;
    const b = Math.max(0, Math.min(40, Math.round(d) + 20));
    hist[b]++;
  }
  console.log('delta histogram (bin -20..+20, step 1):');
  console.log(hist.join(','));
}
main().catch(e => { console.error(e); process.exit(1); });
