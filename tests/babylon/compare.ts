import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import sharp from "sharp";

// pixelmatch.cjs is vendored CommonJS (from playwright-core); under this
// repo's "type": "module" a static import would be parsed as ESM, so load it
// via require() — `.cjs` is unambiguous CommonJS for Node. Types come from
// pixelmatch.d.ts (the `.js` specifier is type-only, resolved by tsc only).
const req = createRequire(import.meta.url);
const pixelmatch = req("./pixelmatch.cjs") as typeof import("./pixelmatch.js");

export interface CompareScreenshotOptions {
  actualPng: Buffer;
  goldenPath: string;
  threshold: number; // e.g. 0.035
  maxDiffPixelRatio: number; // e.g. 0.011
}

export interface CompareScreenshotResult {
  ok: boolean;
  diffPixels: number;
  totalPixels: number;
  diffRatio: number;
  maxDiffPixelRatio: number;
  reason?: string;
  diffPng?: Buffer;
}

/**
 * Playwright filepath sanitization applied to the name WITHOUT extension,
 * then ".png" appended: strip a trailing ".png" if present, replace each run
 * of chars outside [A-Za-z0-9._-] with "-", append ".png".
 * e.g. "Clustered Lighting (lhs pbr material)" → "Clustered-Lighting-lhs-pbr-material-.png".
 */
export function sanitizeReferenceName(name: string): string {
  const base = name.replace(/\.png$/, "");
  return base.replace(/[^A-Za-z0-9._-]+/g, "-") + ".png";
}

export function goldenPath(
  babylonRoot: string,
  entry: { referenceImage?: string; title: string },
): string {
  return path.join(
    babylonRoot,
    "packages/tools/tests/test/visualization/ReferenceImages",
    sanitizeReferenceName(entry.referenceImage ?? entry.title),
  );
}

/**
 * Decode a PNG (Buffer or file path) into opaque RGBA raw data — the layout
 * pixelmatch expects (Playwright's pngjs decode is always RGBA).
 * removeAlpha() strips any transparency defensively (a no-op for opaque
 * images); ensureAlpha() restores the 4-channel layout pixelmatch requires
 * (removeAlpha alone yields 3-channel RGB).
 */
async function decodeOpaqueRgba(
  input: Buffer | string,
): Promise<{ data: Buffer; width: number; height: number }> {
  const { data, info } = await sharp(input)
    .removeAlpha()
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
}

export async function compareScreenshots(
  opts: CompareScreenshotOptions,
): Promise<CompareScreenshotResult> {
  const { actualPng, goldenPath: golden, threshold, maxDiffPixelRatio } = opts;

  if (!existsSync(golden)) {
    return {
      ok: false,
      diffPixels: 0,
      totalPixels: 0,
      diffRatio: 0,
      maxDiffPixelRatio,
      reason: `missing golden: ${golden}`,
    };
  }

  const [goldenImg, actualImg] = await Promise.all([
    decodeOpaqueRgba(golden),
    decodeOpaqueRgba(actualPng),
  ]);

  if (goldenImg.width !== actualImg.width || goldenImg.height !== actualImg.height) {
    return {
      ok: false,
      diffPixels: 0,
      totalPixels: goldenImg.width * goldenImg.height,
      diffRatio: 0,
      maxDiffPixelRatio,
      reason: `size mismatch: golden ${goldenImg.width}px by ${goldenImg.height}px, actual ${actualImg.width}px by ${actualImg.height}px`,
    };
  }

  const { width: w, height: h } = goldenImg;
  const totalPixels = w * h;
  const diff = Buffer.alloc(w * h * 4);
  const diffPixels = pixelmatch(goldenImg.data, actualImg.data, diff, w, h, { threshold });
  const diffRatio = diffPixels / totalPixels;
  const ok = diffPixels <= totalPixels * maxDiffPixelRatio;

  if (ok) {
    return { ok, diffPixels, totalPixels, diffRatio, maxDiffPixelRatio };
  }

  const diffPng = await sharp(diff, { raw: { width: w, height: h, channels: 4 } })
    .png()
    .toBuffer();
  return { ok, diffPixels, totalPixels, diffRatio, maxDiffPixelRatio, diffPng };
}
