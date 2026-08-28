// Temp: expected-vs-actual grid comparison for anomaly pages
import sharp from 'sharp';
import { readFileSync } from 'fs';

const report = JSON.parse(readFileSync('tests/reports/threejs/latest.json', 'utf8'));
const focus = ['webgl_animation_skinning_blending', 'webgl_mirror', 'webgl_multisampled_renderbuffers', 'webgl_points_sprites', 'webgl_geometry_terrain', 'webgl_materials_envmaps', 'webgl_postprocessing_afterimage', 'webgl_postprocessing_ssao', 'webgl_instancing_dynamic', 'webgl_buffergeometry_instancing'];

async function grid(path, gw = 16, gh = 10) {
  const { data, info } = await sharp(path).raw().toBuffer({ resolveWithObject: true });
  const { width: w, height: h, channels: ch } = info;
  const px = data.length / ch;
  const sums = new Array(ch).fill(0);
  const g = new Array(gw * gh).fill(0), gn = new Array(gw * gh).fill(0);
  for (let i = 0; i < px; i++) {
    for (let c = 0; c < ch; c++) sums[c] += data[i * ch + c];
    const gx = Math.floor((i % w) / w * gw), gy = Math.floor(Math.floor(i / w) / h * gh);
    g[gy * gw + gx] += data[i * ch]; gn[gy * gw + gx]++;
  }
  const mean = sums.map(s => (s / px).toFixed(1));
  return { mean, grid: g.map((v, i) => Math.round(v / gn[i])) };
}

for (const name of focus) {
  const p = report.pages.find(x => x.name === name);
  if (!p) continue;
  const a = await grid(p.artifacts.actual);
  const e = await grid(p.artifacts.expected);
  const d = p.diff.diffPercent.toFixed(2);
  console.log(`\n=== ${name} diff ${d}% | A-mean ${a.mean} E-mean ${e.mean}`);
  const ch = v => (v < 32 ? '.' : v < 80 ? '+' : v < 140 ? 'o' : v < 200 ? 'O' : '#');
  console.log('   A ' + a.grid.map(ch).join(''));
  console.log('   E ' + e.grid.map(ch).join(''));
}
