import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import sharp from "sharp";
import { afterAll, describe, expect, it } from "vitest";
import { compareImages, readImage, writePng } from "./compare";

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "threejs-compare-"));

/** Flat RGB image encoded as a 3-channel JPEG (no alpha in the file). */
function makeJpeg(
  w: number,
  h: number,
  r: number,
  g: number,
  b: number,
  quality = 100,
): Promise<Buffer> {
  return sharp({
    create: { width: w, height: h, channels: 3, background: { r, g, b } },
  })
    .jpeg({ quality })
    .toBuffer();
}

/** Flat RGBA image encoded as a 4-channel PNG (alpha in 0-255; sharp's
 *  create() background alpha is a 0-1 float). */
function makePng(
  w: number,
  h: number,
  r: number,
  g: number,
  b: number,
  alpha = 255,
): Promise<Buffer> {
  return sharp({
    create: { width: w, height: h, channels: 4, background: { r, g, b, alpha: alpha / 255 } },
  })
    .png()
    .toBuffer();
}

async function writeFixture(name: string, buf: Buffer): Promise<string> {
  const p = path.join(tmpDir, name);
  await fs.promises.writeFile(p, buf);
  return p;
}

/** Build an RGBA buffer with a flat fill (optionally one pixel overridden). */
function flatRgba(w: number, h: number, r: number, g: number, b: number, alpha = 255): Uint8Array {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    data[o] = r;
    data[o + 1] = g;
    data[o + 2] = b;
    data[o + 3] = alpha;
  }
  return data;
}

describe("readImage", () => {
  it("normalizes a 3-channel JPEG to 4-channel RGBA with alpha 255", async () => {
    // Sanity: the fixture really is 3-channel, so the ensureAlpha path is exercised.
    const jpegBuf = await makeJpeg(16, 16, 120, 80, 200);
    const raw = await sharp(jpegBuf).raw().toBuffer({ resolveWithObject: true });
    expect(raw.info.channels).toBe(3);

    const jpegPath = await writeFixture("flat.jpg", jpegBuf);
    const img = await readImage(jpegPath);
    expect(img.width).toBe(16);
    expect(img.height).toBe(16);
    expect(img.data.length).toBe(16 * 16 * 4);
    for (let i = 3; i < img.data.length; i += 4) {
      expect(img.data[i]).toBe(255);
    }
  });

  it("keeps PNGs at 4 channels with their own alpha", async () => {
    const pngPath = await writeFixture("opaque.png", await makePng(4, 4, 1, 2, 3));
    const img = await readImage(pngPath);
    expect(img.data.length).toBe(4 * 4 * 4);
    for (let i = 3; i < img.data.length; i += 4) {
      expect(img.data[i]).toBe(255);
    }
  });

  it("preserves an existing PNG alpha channel (ensureAlpha must not clobber it)", async () => {
    const pngPath = await writeFixture("half-alpha.png", await makePng(8, 8, 10, 200, 30, 128));
    const img = await readImage(pngPath);
    expect(img.data.length).toBe(8 * 8 * 4);
    for (let i = 3; i < img.data.length; i += 4) {
      expect(img.data[i]).toBe(128);
    }
  });
});

describe("compareImages with normalized layouts", () => {
  it("JPEG (3-ch) vs PNG (4-ch) with identical RGB content → diffPercent 0, pass", async () => {
    const w = 16;
    const h = 16;
    // q100 flat color: lossy error ≤ ±1-2 per channel, far below the per-pixel gate
    // (delta > 0.01² ≈ dr²+dg²+db² > 195).
    const jpegPath = await writeFixture("same.jpg", await makeJpeg(w, h, 120, 80, 200));
    const pngPath = await writeFixture("same.png", await makePng(w, h, 120, 80, 200));
    const jpeg = await readImage(jpegPath);
    const png = await readImage(pngPath);
    expect(jpeg.data.length).toBe(w * h * 4);
    expect(png.data.length).toBe(w * h * 4);
    const res = compareImages(jpeg, png);
    expect(res.numDiffPixels).toBe(0);
    expect(res.diffPercent).toBe(0);
    expect(res.pass).toBe(true);
  });

  it("single pixel red +200 → numDiffPixels 1, exact diffPercent, pass false", async () => {
    const w = 8;
    const h = 8;
    const base = flatRgba(w, h, 10, 200, 30);
    const mutated = flatRgba(w, h, 10, 200, 30);
    mutated[0] += 200; // top-left pixel red: 10 → 210 (no overflow)
    const basePath = path.join(tmpDir, "base.png");
    const mutatedPath = path.join(tmpDir, "mutated.png");
    await writePng(basePath, { width: w, height: h, data: base });
    await writePng(mutatedPath, { width: w, height: h, data: mutated });
    const res = compareImages(await readImage(basePath), await readImage(mutatedPath));
    expect(res.numDiffPixels).toBe(1);
    expect(res.diffPercent).toBeCloseTo(100 / (w * h));
    expect(res.pass).toBe(false);
  });
});

afterAll(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});
