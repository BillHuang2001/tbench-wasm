import sharp from 'sharp';

export interface DiffStats {
  numDiffPixels: number;
  diffPercent: number;
  maxDelta: number;
  meanDelta: number;
  pass: boolean;
}

export interface RGBAImage {
  width: number;
  height: number;
  data: Uint8Array;
}

export async function readImage(path: string): Promise<RGBAImage> {
  const { data, info } = await sharp(path).raw().toBuffer({ resolveWithObject: true });
  return { width: info.width, height: info.height, data: new Uint8Array(data) };
}

export function downscale2x(img: RGBAImage): RGBAImage {
  const outW = Math.floor(img.width / 2);
  const outH = Math.floor(img.height / 2);
  const data = new Uint8Array(outW * outH * 4);
  const src = img.data;
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const sx = x * 2;
      const sy = y * 2;
      const o = (y * outW + x) * 4;
      for (let c = 0; c < 4; c++) {
        const i00 = (sy * img.width + sx) * 4 + c;
        const i10 = (sy * img.width + sx + 1) * 4 + c;
        const i01 = ((sy + 1) * img.width + sx) * 4 + c;
        const i11 = ((sy + 1) * img.width + sx + 1) * 4 + c;
        data[o + c] = Math.round((src[i00] + src[i10] + src[i01] + src[i11]) / 4);
      }
    }
  }
  return { width: outW, height: outH, data };
}

const maxDeltaConst = 255 * 255 * 3;

export function compareImages(actual: RGBAImage, expected: RGBAImage, threshold = 0.1): DiffStats {
  if (actual.width !== expected.width || actual.height !== expected.height) {
    throw new Error('Image sizes do not match');
  }
  const w = actual.width;
  const h = actual.height;
  const a = actual.data;
  const e = expected.data;
  let numDiffPixels = 0;
  let maxDelta = 0;
  let sumDelta = 0;
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const dr = a[o] - e[o];
    const dg = a[o + 1] - e[o + 1];
    const db = a[o + 2] - e[o + 2];
    const delta = (dr * dr + dg * dg + db * db) / maxDeltaConst;
    sumDelta += delta;
    if (delta > threshold * threshold) {
      numDiffPixels++;
    }
    if (delta > maxDelta) {
      maxDelta = delta;
    }
  }
  const diffPercent = (numDiffPixels / (w * h)) * 100;
  return {
    numDiffPixels,
    diffPercent,
    maxDelta,
    meanDelta: sumDelta / (w * h),
    pass: diffPercent < 0.1
  };
}

export function makeDiffImage(actual: RGBAImage, expected: RGBAImage, threshold = 0.1): RGBAImage {
  const w = actual.width;
  const h = actual.height;
  const a = actual.data;
  const e = expected.data;
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const dr = a[o] - e[o];
    const dg = a[o + 1] - e[o + 1];
    const db = a[o + 2] - e[o + 2];
    const delta = (dr * dr + dg * dg + db * db) / maxDeltaConst;
    if (delta > threshold * threshold) {
      data[o] = 255;
      data[o + 1] = 0;
      data[o + 2] = 0;
      data[o + 3] = 255;
    } else {
      data[o] = a[o] * 0.2;
      data[o + 1] = a[o + 1] * 0.2;
      data[o + 2] = a[o + 2] * 0.2;
      data[o + 3] = 255;
    }
  }
  return { width: w, height: h, data };
}

export async function writePng(path: string, img: RGBAImage): Promise<void> {
  await sharp(img.data, { raw: { width: img.width, height: img.height, channels: 4 } }).png().toFile(path);
}
