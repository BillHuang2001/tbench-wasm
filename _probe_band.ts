import sharp from 'sharp';

const goldenDir = '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/';

async function main() {
  const ours = await sharp('tests/reports/babylon/Hillvalley.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const gold = await sharp(goldenDir + 'Hillvalley.png').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height } = ours.info;
  const o = ours.data, g = gold.data;
  // Vertical luma profile at x=75 (middle of band), averaged over x 60..90
  console.log('x=60..90 avg luma, rows y=80..200 (step 4):  OURS vs GOLDEN');
  for (let y = 80; y <= 200; y += 4) {
    let so = 0, sg = 0;
    for (let x = 60; x < 90; x++) {
      const i = (y * width + x) * 3;
      so += (o[i] + o[i+1] + o[i+2]) / 3;
      sg += (g[i] + g[i+1] + g[i+2]) / 3;
    }
    const ao = so / 30, ag = sg / 30;
    console.log(`y=${String(y).padStart(3)}  ours=${ao.toFixed(1).padStart(6)}  gold=${ag.toFixed(1).padStart(6)}  delta=${(ao-ag).toFixed(1).padStart(6)}`);
  }
  // Also horizontal profile at y=140 (mid band), x 0..300 step 10
  console.log('\nx=0..300 (step 10) avg luma at y=140:  OURS vs GOLDEN');
  for (let x = 0; x <= 300; x += 10) {
    let so = 0, sg = 0;
    for (let xx = x; xx < x + 10; xx++) {
      const i = (140 * width + xx) * 3;
      so += (o[i] + o[i+1] + o[i+2]) / 3;
      sg += (g[i] + g[i+1] + g[i+2]) / 3;
    }
    const ao = so / 10, ag = sg / 10;
    console.log(`x=${String(x).padStart(3)}  ours=${ao.toFixed(1).padStart(6)}  gold=${ag.toFixed(1).padStart(6)}  delta=${(ao-ag).toFixed(1).padStart(6)}`);
  }
}
main().catch(e => { console.error(e); process.exit(1); });
