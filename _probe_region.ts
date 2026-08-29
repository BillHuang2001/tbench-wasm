import sharp from 'sharp';

const goldenDir = '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/';

async function main() {
  const ours = await sharp('tests/reports/babylon/Hillvalley.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const gold = await sharp(goldenDir + 'Hillvalley.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height } = ours.info;
  const o = ours.data, g = gold.data;
  // Region of interest: x 0..150, y 100..200
  const x0 = 0, x1 = 150, y0 = 100, y1 = 200;
  let sumOR = 0, sumOG = 0, sumOB = 0, sumGR = 0, sumGG = 0, sumGB = 0;
  let n = 0;
  const map: string[][] = [];
  for (let y = y0; y < y1; y++) {
    const row: string[] = [];
    for (let x = x0; x < x1; x++) {
      const i = (y * width + x) * 3;
      sumOR += o[i]; sumOG += o[i+1]; sumOB += o[i+2];
      sumGR += g[i]; sumGG += g[i+1]; sumGB += g[i+2];
      const d = (o[i]+o[i+1]+o[i+2])/3 - (g[i]+g[i+1]+g[i+2])/3;
      if (d > 15) row.push('#'); else if (d > 5) row.push('+'); else if (d < -15) row.push('@'); else if (d < -5) row.push('-'); else row.push('.');
      n++;
    }
    map.push(row);
  }
  console.log('region mean RGB ours:  ', (sumOR/n).toFixed(1), (sumOG/n).toFixed(1), (sumOB/n).toFixed(1));
  console.log('region mean RGB golden:', (sumGR/n).toFixed(1), (sumGG/n).toFixed(1), (sumGB/n).toFixed(1));
  console.log('delta map (x 0..150 -> 10-char buckets, y 100..200 -> every 4th row):');
  for (let y = y0; y < y1; y += 4) {
    let line = '';
    for (let x = x0; x < x1; x += 15) {
      // bucket of 15 px
      let maxd = 0;
      for (let xx = x; xx < Math.min(x+15, x1); xx++) {
        const i = (y * width + xx) * 3;
        const d = Math.abs((o[i]+o[i+1]+o[i+2])/3 - (g[i]+g[i+1]+g[i+2])/3);
        if (d > maxd) maxd = d;
      }
      line += maxd > 40 ? '#' : maxd > 20 ? '+' : maxd > 8 ? '.' : ' ';
    }
    console.log(`y=${String(y).padStart(3)}: ${line}`);
  }
  // Where exactly are the >20 deltas in this region?
  let bigX0 = 999, bigX1 = 0, bigY0 = 999, bigY1 = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = (y * width + x) * 3;
    const d = (o[i]+o[i+1]+o[i+2])/3 - (g[i]+g[i+1]+g[i+2])/3;
    if (d > 20) { bigX0 = Math.min(bigX0, x); bigX1 = Math.max(bigX1, x); bigY0 = Math.min(bigY0, y); bigY1 = Math.max(bigY1, y); }
  }
  console.log('bbox of delta>20 within region:', bigX0, bigY0, '->', bigX1, bigY1);
}
main().catch(e => { console.error(e); process.exit(1); });
