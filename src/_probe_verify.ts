// Scratch verify: pixelmatch on saved files vs harness numbers + sample pixels.
// Run: npx tsx src/_probe_verify.ts
import sharp from 'sharp';
import pixelmatch from 'pixelmatch';

async function rawRGBA(p: string): Promise<{ w: number; h: number; data: Buffer }> {
  const { data, info } = await sharp(p).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { w: info.width, h: info.height, data };
}

async function main(): Promise<void> {
  for (const [name, oursPath, goldenPath] of [
    ['Hill Valley', '/tmp/post-HV.png', '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/Hillvalley.png'],
    ['Sponza', '/tmp/post-Sponza.png', '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/Sponza.png'],
    ['The car', '/tmp/post-TheCar.png', '/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/TheCar.png'],
  ] as [string, string, string][]) {
    const ours = await rawRGBA(oursPath);
    const gold = await rawRGBA(goldenPath);
    const diff = Buffer.alloc(ours.w * ours.h * 4);
    const n = pixelmatch(gold.data, ours.data, diff, ours.w, ours.h, { threshold: 0.035 });
    console.log(`${name}: pixelmatch diffPixels=${n} ratio=${(n / (ours.w * ours.h)).toFixed(4)} (${ours.w}x${ours.h})`);
    // Sample pixels: ours vs golden RGB at a few points
    for (const [x, y] of [[10, 10], [300, 200], [590, 390], [150, 120], [450, 250]] as [number, number][]) {
      const o = (y * ours.w + x) * 4, g = (y * ours.w + x) * 4;
      const oRGB = `${ours.data[o]},${ours.data[o + 1]},${ours.data[o + 2]}a${ours.data[o + 3]}`;
      const gRGB = `${gold.data[g]},${gold.data[g + 1]},${gold.data[g + 2]}a${gold.data[g + 3]}`;
      console.log(`  (${x},${y}) ours=[${oRGB}] golden=[${gRGB}]`);
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
