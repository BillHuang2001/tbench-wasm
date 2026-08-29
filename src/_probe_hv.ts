// Scratch analyzer (task-5): Hill Valley pre-fix vs post-fix vs golden.
// Delete after use. Run: npx tsx src/_probe_hv.ts
import sharp from 'sharp';

async function load(p: string): Promise<{ w: number; h: number; c: number; data: Buffer }> {
  const { data, info } = await sharp(p).raw().toBuffer({ resolveWithObject: true });
  return { w: info.width, h: info.height, c: info.channels, data };
}

function luma(d: Buffer, i: number, c: number): number {
  return 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
}

async function main(): Promise<void> {
  const pre = await load('/tmp/pre-HV.png');
  const post = await load('/tmp/post-HV.png');
  const gold = await load('/testsuites/Babylon.js/packages/tools/tests/test/visualization/ReferenceImages/Hillvalley.png');
  console.log(`pre ${pre.w}x${pre.h} c=${pre.c}  post ${post.w}x${post.h} c=${post.c}  gold ${gold.w}x${gold.h} c=${gold.c}`);
  const { w, h } = post;
  const c = post.c;
  if (pre.w !== w || pre.h !== h || gold.w !== w || gold.h !== h) { console.log('SIZE MISMATCH'); return; }

  // --- pre vs post (isolate the clamp effect) ---
  let changed = 0;
  const deltas = new Map<number, number>();
  let minX = w, minY = h, maxX = 0, maxY = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * c;
    const lp = luma(pre.data, i, c), lt = luma(post.data, i, c);
    if (Math.abs(lp - lt) > 0.5) {
      changed++;
      const d = Math.round(lt - lp);
      deltas.set(d, (deltas.get(d) ?? 0) + 1);
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  console.log(`\nPRE vs POST: ${changed} px changed by the clamp fix; bbox x[${minX},${maxX}] y[${minY},${maxY}]`);
  const dk = [...deltas.keys()].sort((a, b) => a - b);
  console.log('delta (post-pre) histogram:', dk.map(k => `${k}:${deltas.get(k)}`).join(' '));

  // --- post vs golden: residual ---
  const signed = new Float32Array(w * h);
  let over8 = 0, over20 = 0, maxAbs = 0, meanAbs = 0;
  for (let i = 0; i < w * h; i++) {
    const lp = luma(post.data, i * c), lg = luma(gold.data, i * c);
    const d = Math.abs(lp - lg);
    signed[i] = lp - lg;
    meanAbs += d; if (d > maxAbs) maxAbs = d;
    if (d > 8) over8++; if (d > 20) over20++;
  }
  meanAbs /= w * h;
  console.log(`\nPOST vs GOLD: mean|lumaΔ|=${meanAbs.toFixed(3)} max=${maxAbs.toFixed(1)} px>8: ${over8} px>20: ${over20}`);

  // Signed region map: y 90..210, x 0..300, cell=4
  const cell = 4;
  console.log('\npost-vs-gold signed luma delta map, x 0..300, y 90..210 (cell=4px, + bright / - dark / . small / space tiny):');
  for (let y0 = 90; y0 < 210; y0 += cell) {
    let row = String(y0).padStart(3) + ' ';
    for (let x0 = 0; x0 < 300; x0 += cell) {
      let sum = 0, n = 0;
      for (let y = y0; y < Math.min(210, y0 + cell); y++) for (let x = x0; x < Math.min(300, x0 + cell); x++) { sum += signed[y * w + x]; n++; }
      const m = sum / n;
      row += m > 12 ? '#' : m > 3 ? '+' : m < -12 ? 'X' : m < -3 ? '-' : (Math.abs(m) > 0.8 ? '.' : ' ');
    }
    console.log(row);
  }

  // Row profile y 90..210
  console.log('\nrow profile y: mean|lumaΔ| meanSigned (post-gold):');
  for (let y = 90; y <= 210; y += 5) {
    let ma = 0, ms = 0, n = 0;
    for (let x = 0; x < w; x++) { ma += Math.abs(signed[y * w + x]); ms += signed[y * w + x]; n++; }
    console.log(`${y}: ${(ma / n).toFixed(2)} ${(ms / n).toFixed(2)}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
