import sharp from 'sharp';

const goldenDir = '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/';

async function analyze(name: string) {
  const ours = await sharp(`tests/reports/babylon/${name}.png`).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const gold = await sharp(goldenDir + `${name}.png`).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height } = ours.info;
  const o = ours.data, g = gold.data;
  const rows = 5, cols = 5;
  const reg: { [k: string]: { n: number; sum: number; abs: number; big: number } } = {};
  let over10 = 0, over20 = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 3;
    const d = (o[i] + o[i+1] + o[i+2] - g[i] - g[i+1] - g[i+2]) / 3;
    const rx = Math.floor(x / width * cols), ry = Math.floor(y / height * rows);
    const k = `${ry},${rx}`;
    reg[k] = reg[k] || { n: 0, sum: 0, abs: 0, big: 0 };
    reg[k].n++; reg[k].sum += d; reg[k].abs += Math.abs(d);
    if (Math.abs(d) > 10) over10++;
    if (Math.abs(d) > 20) { over20++; reg[k].big++; }
  }
  console.log(`=== ${name} ===  |d|>10: ${over10}  |d|>20: ${over20}`);
  for (let ry = 0; ry < rows; ry++) {
    const line: string[] = [];
    const bigLine: string[] = [];
    for (let rx = 0; rx < cols; rx++) {
      const r = reg[`${ry},${rx}`];
      line.push((r.sum / r.n).toFixed(1));
      bigLine.push(r.big > 50 ? '#' : r.big > 5 ? '+' : '.');
    }
    console.log('  mean delta row', ry, ':', line.join('  '));
    console.log('  big-delta map   ', ry, ':', bigLine.join('   '));
  }
}
async function main() {
  await analyze('Sponza');
  await analyze('The car');
}
main().catch(e => { console.error(e); process.exit(1); });
