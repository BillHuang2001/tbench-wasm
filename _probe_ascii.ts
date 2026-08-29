import sharp from 'sharp';

const goldenDir = '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/';
const files: { name: string; file: string }[] = [
  { name: 'Hill Valley', file: 'Hillvalley' },
  { name: 'Sponza', file: 'Sponza' },
  { name: 'The car', file: 'TheCar' },
];

const ramp = ' .:-=+*#%@';

async function analyze(name: string, file: string) {
  const ours = await sharp(`tests/reports/babylon/${file}.png`).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const gold = await sharp(goldenDir + `${file}.png`).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height } = ours.info;
  const o = ours.data, g = gold.data;
  const W = 64, H = 32;
  console.log(`\n===== ${name} (${width}x${height}) =====`);
  console.log('GOLDEN luma:');
  for (let ry = 0; ry < H; ry++) {
    let line = '';
    for (let rx = 0; rx < W; rx++) {
      // sample center of bucket
      const x = Math.min(width - 1, Math.floor((rx + 0.5) * width / W));
      const y = Math.min(height - 1, Math.floor((ry + 0.5) * height / H));
      const i = (y * width + x) * 3;
      const l = (g[i] + g[i+1] + g[i+2]) / 3;
      line += ramp[Math.min(9, Math.floor(l / 26))];
    }
    console.log(line);
  }
  console.log('DIFF map (|d|>8: #, >4: +, >2: ., else space):');
  for (let ry = 0; ry < H; ry++) {
    let line = '';
    for (let rx = 0; rx < W; rx++) {
      const x = Math.min(width - 1, Math.floor((rx + 0.5) * width / W));
      const y = Math.min(height - 1, Math.floor((ry + 0.5) * height / H));
      const i = (y * width + x) * 3;
      const d = Math.abs((o[i] + o[i+1] + o[i+2] - g[i] - g[i+1] - g[i+2]) / 3);
      line += d > 8 ? '#' : d > 4 ? '+' : d > 2 ? '.' : ' ';
    }
    console.log(line);
  }
}
async function main() {
  for (const f of files) await analyze(f.name, f.file);
}
main().catch(e => { console.error(e); process.exit(1); });
