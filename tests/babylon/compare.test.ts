import { existsSync } from "node:fs";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import sharp from "sharp";
import { afterAll, describe, expect, it } from "vitest";
import { compareScreenshots, goldenPath, sanitizeReferenceName } from "./compare.js";

// Load the vendored CommonJS pixelmatch via require() (see compare.ts) — a
// static import would be parsed as ESM under this repo's "type": "module".
const cjsRequire = createRequire(import.meta.url);
const pixelmatch = cjsRequire("./pixelmatch.cjs") as typeof import("./pixelmatch.js");

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "babylon-compare-"));

function makePng(
  w: number,
  h: number,
  r: number,
  g: number,
  b: number,
  alpha = 1,
): Promise<Buffer> {
  return sharp({
    create: { width: w, height: h, channels: 4, background: { r, g, b, alpha } },
  })
    .png()
    .toBuffer();
}

async function writeGolden(name: string, buf: Buffer): Promise<string> {
  const p = path.join(tmpDir, name);
  await fs.promises.writeFile(p, buf);
  return p;
}

/** Decode a PNG to raw RGBA, tweak it, re-encode to PNG. */
async function mutatePng(
  png: Buffer,
  mutate: (data: Buffer, width: number, height: number) => void,
): Promise<Buffer> {
  const { data, info } = await sharp(png).raw().toBuffer({ resolveWithObject: true });
  mutate(data, info.width, info.height);
  return sharp(data, {
    raw: { width: info.width, height: info.height, channels: info.channels },
  })
    .png()
    .toBuffer();
}

describe("sanitizeReferenceName", () => {
  it("replaces runs of invalid chars with '-' and appends .png", () => {
    expect(sanitizeReferenceName("Clustered Lighting (lhs pbr material)")).toBe(
      "Clustered-Lighting-lhs-pbr-material-.png",
    );
  });

  it("appends .png when missing", () => {
    expect(sanitizeReferenceName("Sponza")).toBe("Sponza.png");
  });

  it("does not double the extension", () => {
    expect(sanitizeReferenceName("Sponza.png")).toBe("Sponza.png");
  });

  it("collapses runs of invalid chars into a single '-'", () => {
    expect(sanitizeReferenceName("a  b!!c")).toBe("a-b-c.png");
    expect(sanitizeReferenceName("!!!")).toBe("-.png");
  });

  it("preserves dots and underscores", () => {
    expect(sanitizeReferenceName("my.name_1")).toBe("my.name_1.png");
  });
});

describe("goldenPath", () => {
  const root = "/testsuites/Babylon.js";

  it("resolves to the ReferenceImages dir with the sanitized name", () => {
    expect(goldenPath(root, { title: "Sponza" })).toBe(
      path.join(root, "packages/tools/tests/test/visualization/ReferenceImages", "Sponza.png"),
    );
  });

  it("prefers referenceImage over title", () => {
    const p = goldenPath(root, { title: "Other scene", referenceImage: "Sponza.png" });
    expect(p.endsWith(path.join("ReferenceImages", "Sponza.png"))).toBe(true);
    expect(p).not.toContain("Other-scene");
  });

  it("appends .png to a referenceImage without extension", () => {
    expect(goldenPath(root, { title: "X", referenceImage: "Sponza" })).toBe(
      path.join(root, "packages/tools/tests/test/visualization/ReferenceImages", "Sponza.png"),
    );
  });
});

describe("compareScreenshots", () => {
  const threshold = 0.035;

  it("passes identical images with diffPixels 0", async () => {
    const png = await makePng(4, 4, 10, 200, 30);
    const golden = await writeGolden("identical.png", png);
    const res = await compareScreenshots({
      actualPng: png,
      goldenPath: golden,
      threshold,
      maxDiffPixelRatio: 0.011,
    });
    expect(res.ok).toBe(true);
    expect(res.diffPixels).toBe(0);
    expect(res.totalPixels).toBe(16);
    expect(res.diffRatio).toBe(0);
    expect(res.reason).toBeUndefined();
  });

  it("passes a single differing pixel within maxDiffPixelRatio", async () => {
    const golden = await writeGolden("one-diff.png", await makePng(4, 4, 10, 200, 30));
    const actualPng = await mutatePng(await makePng(4, 4, 10, 200, 30), (data) => {
      data[0] = 200; // top-left pixel → red
      data[1] = 0;
      data[2] = 0;
    });
    // 4x4 = 16 px; maxDiffPixelRatio 0.1 → 1.6 → 1 diff passes.
    const res = await compareScreenshots({
      actualPng,
      goldenPath: golden,
      threshold,
      maxDiffPixelRatio: 0.1,
    });
    expect(res.ok).toBe(true);
    expect(res.diffPixels).toBe(1);
    expect(res.diffRatio).toBeCloseTo(1 / 16);
  });

  it("fails when every pixel differs beyond maxDiffPixelRatio", async () => {
    const golden = await writeGolden("all-diff.png", await makePng(4, 4, 10, 200, 30));
    const actualPng = await makePng(4, 4, 200, 0, 0);
    // 4x4 = 16 px; maxDiffPixelRatio 0.01 → 0.16 → 16 diffs fail.
    const res = await compareScreenshots({
      actualPng,
      goldenPath: golden,
      threshold,
      maxDiffPixelRatio: 0.01,
    });
    expect(res.ok).toBe(false);
    expect(res.diffPixels).toBe(16);
    expect(res.diffRatio).toBe(1);
    expect(res.diffPng).toBeDefined();
  });

  it("does not count a channel delta within the threshold", async () => {
    const golden = await writeGolden("small-delta.png", await makePng(8, 8, 10, 200, 30));
    const actualPng = await mutatePng(await makePng(8, 8, 10, 200, 30), (data) => {
      data[0] += 5; // red +5 in one pixel: maxDelta ≈ 43.14 → not counted
    });
    const res = await compareScreenshots({
      actualPng,
      goldenPath: golden,
      threshold,
      maxDiffPixelRatio: 0.01,
    });
    expect(res.diffPixels).toBe(0);
    expect(res.ok).toBe(true);
  });

  it("counts a channel delta beyond the threshold", async () => {
    const golden = await writeGolden("big-delta.png", await makePng(8, 8, 10, 200, 30));
    const actualPng = await mutatePng(await makePng(8, 8, 10, 200, 30), (data) => {
      data[0] += 200; // red +200 in one pixel → far beyond maxDelta ≈ 43.14
    });
    const res = await compareScreenshots({
      actualPng,
      goldenPath: golden,
      threshold,
      maxDiffPixelRatio: 0.01, // 64 px * 0.01 = 0.64 < 1 diff → fail
    });
    expect(res.diffPixels).toBe(1);
    expect(res.ok).toBe(false);
  });

  it("fails with a size mismatch reason when dimensions differ", async () => {
    const golden = await writeGolden("size.png", await makePng(4, 4, 10, 200, 30));
    const actualPng = await makePng(8, 8, 10, 200, 30);
    const res = await compareScreenshots({
      actualPng,
      goldenPath: golden,
      threshold,
      maxDiffPixelRatio: 0.011,
    });
    expect(res.ok).toBe(false);
    expect(res.reason).toContain("size mismatch");
  });

  it("fails with a missing golden reason when the golden file does not exist", async () => {
    const missing = path.join(tmpDir, "does-not-exist.png");
    expect(existsSync(missing)).toBe(false);
    const res = await compareScreenshots({
      actualPng: await makePng(4, 4, 10, 200, 30),
      goldenPath: missing,
      threshold,
      maxDiffPixelRatio: 0.011,
    });
    expect(res.ok).toBe(false);
    expect(res.reason?.startsWith("missing golden")).toBe(true);
  });
});

describe("pixelmatch parity with playwright-core", () => {
  // Load playwright-core's live vendored copy (npm pixelmatch is NOT installed
  // in this environment; playwright-core ships its own third_party copy).
  // `req(absolutePath)` bypasses the exports map — verified working.
  const req = createRequire(import.meta.url);
  let pwPixelmatch: ((img1: Uint8Array, img2: Uint8Array, output: Uint8Array, width: number, height: number, options?: unknown) => number) | null = null;
  try {
    const pwEntry = req.resolve("playwright-core");
    const thirdParty = path.join(path.dirname(pwEntry), "lib", "third_party", "pixelmatch.js");
    if (existsSync(thirdParty)) {
      pwPixelmatch = req(thirdParty) as typeof pwPixelmatch;
    }
  } catch {
    pwPixelmatch = null;
  }

  // If playwright-core's copy is missing, skip rather than fail.
  const itParity = pwPixelmatch ? it : it.skip;

  function makeRgba(
    w: number,
    h: number,
    fill: (x: number, y: number, c: number) => number,
  ): Uint8Array {
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        for (let c = 0; c < 4; c++) data[(y * w + x) * 4 + c] = fill(x, y, c);
      }
    }
    return data;
  }

  itParity("vendored copy yields identical diff counts to playwright-core's", () => {
    const w = 16;
    const h = 16;
    const opaque = (c: number) => (c === 3 ? 255 : 100);

    const solid = makeRgba(w, h, (_x, _y, c) => opaque(c));
    const solidWithPixel = makeRgba(w, h, (x, y, c) =>
      x === 3 && y === 5 ? (c === 3 ? 255 : 0) : opaque(c),
    );
    const gradient = makeRgba(w, h, (_x, y, c) => (c === 3 ? 255 : Math.floor((y / (h - 1)) * 255)));
    const gradientReversed = makeRgba(w, h, (_x, y, c) =>
      c === 3 ? 255 : Math.floor(((h - 1 - y) / (h - 1)) * 255),
    );
    const checker = makeRgba(w, h, (x, y, c) => (c === 3 ? 255 : (x + y) % 2 === 0 ? 255 : 0));
    const checkerInverted = makeRgba(w, h, (x, y, c) => (c === 3 ? 255 : (x + y) % 2 === 0 ? 0 : 255));

    const cases: Array<[string, Uint8Array, Uint8Array]> = [
      ["identical", solid, solid],
      ["single-pixel diff", solid, solidWithPixel],
      ["vertical gradient vs reversed", gradient, gradientReversed],
      ["checkerboard vs inverted", checker, checkerInverted],
    ];

    for (const [name, img1, img2] of cases) {
      const out1 = new Uint8Array(w * h * 4);
      const out2 = new Uint8Array(w * h * 4);
      const ours = pixelmatch(img1, img2, out1, w, h, { threshold: 0.035 });
      const theirs = pwPixelmatch!(img1, img2, out2, w, h, { threshold: 0.035 });
      expect(ours, `${name}: our count ${ours} vs playwright ${theirs}`).toBe(theirs);
    }

    // Sanity: the differing cases actually register diffs in both copies.
    const out = new Uint8Array(w * h * 4);
    expect(pixelmatch(solid, solidWithPixel, out, w, h, { threshold: 0.035 })).toBe(1);
    expect(pixelmatch(gradient, gradientReversed, out, w, h, { threshold: 0.035 })).toBeGreaterThan(0);
  });
});

afterAll(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});