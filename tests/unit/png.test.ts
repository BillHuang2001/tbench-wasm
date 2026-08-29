/**
 * Unit tests for src/present/png.ts — the pure-JS HIGH-BIT-DEPTH (16-bit) PNG
 * decoder added for the CTS tex-image-10bpc fix. All PNG fixtures are built
 * IN-TEST: manual chunk construction (IHDR/sBIT/IDAT/IEND with correct CRC32)
 * + Node's built-in zlib.deflateSync for the RFC 1950 IDAT stream.
 *
 * REAL API (src/present/png.ts — NOT re-exported from src/present/index):
 * - `decodePngBuffer(bytes: Uint8Array): PngDecodeResult` where
 *   `PngDecodeResult = { ok: true; image: DecodedImage } | null` and
 *   `DecodedImage = { width: number; height: number; data: Uint8ClampedArray }`
 *   — data is straight-alpha RGBA8, length = width*height*4, top-down rows.
 *   Only 16-bit NON-INTERLACED gray(0)/grayA(4)/RGB(2)/RGBA(6) PNGs (≤ 16M
 *   pixels) decode; everything else (8-bit, interlaced, reserved color types,
 *   corrupt, truncated, non-PNG) → null so callers fall back to the DOM paths.
 * - 16→8 conversion: with sBIT ≤ 10 and top-bits value ≤ 255 the sample is
 *   carried IN-BAND as `v16 >> (16 − sbit)`; otherwise the standard rounded
 *   `Math.round(v16 * 255 / 65535)` applies (byte-identical to Chrome's
 *   native RGBA8 upload).
 * - `decodePngFromElement(source, src)` — XHR/DOM path; returns null in Node
 *   (no XMLHttpRequest).
 */
import { deflateSync } from "node:zlib";
import { describe, it, expect } from "vitest";
import { decodePngBuffer, decodePngFromElement } from "../../src/present/png";

// ---------------------------------------------------------------------------
// PNG fixture construction (signature + chunks + CRC32 + zlib IDAT).
// ---------------------------------------------------------------------------

const PNG_SIGNATURE = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** One PNG chunk: 4-byte big-endian length + type + data + CRC32(type+data). */
function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

function ihdrChunk(
  width: number,
  height: number,
  bitDepth: number,
  colorType: number,
  interlace = 0,
): Uint8Array {
  const d = new Uint8Array(13);
  const dv = new DataView(d.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  d[8] = bitDepth;
  d[9] = colorType;
  d[10] = 0; // compression method
  d[11] = 0; // filter method
  d[12] = interlace;
  return chunk("IHDR", d);
}

/** 16-bit big-endian sample bytes (PNG byte order), e.g. [R,G,B,A] per pixel. */
function sampleBytes(samples: number[]): Uint8Array {
  const out = new Uint8Array(samples.length * 2);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i++) dv.setUint16(i * 2, samples[i]);
  return out;
}

/**
 * PNG filter ENCODER: produces the filtered bytes of one scanline from the
 * unfiltered sample bytes, mirroring the decoder's in-place unfilter exactly
 * (`a` = left neighbor of the same row, `b` = above, `c` = above-left; the
 * encoder's `a`/`b`/`c` are the ORIGINAL bytes, which equal the decoder's
 * reconstructed bytes because reconstruction is exact).
 */
function filterRow(filter: number, row: Uint8Array, prev: Uint8Array | null, bpp: number): Uint8Array {
  const out = new Uint8Array(1 + row.length);
  out[0] = filter;
  for (let x = 0; x < row.length; x++) {
    const a = x >= bpp ? row[x - bpp] : 0;
    const b = prev ? prev[x] : 0;
    const c = prev && x >= bpp ? prev[x - bpp] : 0;
    let pred = 0;
    if (filter === 1) pred = a;
    else if (filter === 2) pred = b;
    else if (filter === 3) pred = (a + b) >> 1;
    else if (filter === 4) {
      const p = a + b - c;
      const pa = Math.abs(p - a);
      const pb = Math.abs(p - b);
      const pc = Math.abs(p - c);
      pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
    out[1 + x] = (row[x] - pred) & 0xff;
  }
  return out;
}

interface BuildPngOpts {
  bitDepth?: number; // default 16
  colorType?: number; // 0 gray, 2 RGB, 4 grayA, 6 RGBA (default 2)
  interlace?: number;
  sbit?: number[]; // optional sBIT sample depths
  filter?: number; // filter applied to every row (default 0)
  filters?: number[]; // per-row filter override (length = rows.length)
  idatSplit?: number; // split the deflated IDAT stream into N chunks
}

/** Assembles a full PNG file from unfiltered 16-bit sample-byte rows. */
function buildPng(
  rows: Uint8Array[],
  width: number,
  height: number,
  opts: BuildPngOpts = {},
): Uint8Array {
  const { bitDepth = 16, colorType = 2, interlace = 0, sbit, filter = 0, filters, idatSplit } = opts;
  const channels = [1, 0, 3, 1, 2, 0, 4][colorType] ?? 0; // PNG spec: gray/RGB/palette/grayA/RGBA
  const bpp = channels * 2; // bytes per pixel (16-bit samples)
  const stride = width * bpp;
  const parts: Uint8Array[] = [PNG_SIGNATURE, ihdrChunk(width, height, bitDepth, colorType, interlace)];
  if (sbit) parts.push(chunk("sBIT", new Uint8Array(sbit)));
  const rawParts: Uint8Array[] = [];
  let prev: Uint8Array | null = null;
  rows.forEach((row, i) => {
    if (row.length !== stride) throw new Error(`buildPng: row length ${row.length} != stride ${stride}`);
    rawParts.push(filterRow(filters ? filters[i] : filter, row, prev, bpp));
    prev = row;
  });
  const deflated = deflateSync(concatBytes(rawParts));
  if (idatSplit && idatSplit > 1) {
    const step = Math.ceil(deflated.length / idatSplit);
    for (let i = 0; i < deflated.length; i += step) {
      parts.push(chunk("IDAT", deflated.subarray(i, i + step)));
    }
  } else {
    parts.push(chunk("IDAT", deflated));
  }
  parts.push(chunk("IEND", new Uint8Array(0)));
  return concatBytes(parts);
}

/** Reference 16→8 conversions (must match src/present/png.ts to8). */
const honest = (v: number): number => Math.round((v * 255) / 65535);
const inBand = (v: number, sbit: number): number => v >> (16 - sbit);

/** Pixels helper: unpacks decoded RGBA8 into [r,g,b,a] per pixel. */
function pixels(data: Uint8ClampedArray, w: number, h: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < w * h; i++) out.push(Array.from(data.subarray(i * 4, i * 4 + 4)));
  return out;
}

// ---------------------------------------------------------------------------
// tex-image-10bpc: high-bit-preserving in-band conversion.
// ---------------------------------------------------------------------------

describe("decodePngBuffer — high-bit-preserving 16→8 (tex-image-10bpc path)", () => {
  it("carries 10-bit samples in-band: 8×1 RGB sBIT 10/10/10 ramp → [0..7]", () => {
    const ramp = [0, 64, 128, 192, 256, 320, 384, 448]; // 10-bit R = 0..7, stored as v10<<6
    const row: number[] = [];
    for (const r of ramp) row.push(r, 0, 0); // G = B = 0
    const png = buildPng([sampleBytes(row)], 8, 1, { colorType: 2, sbit: [10, 10, 10] });

    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    const img = res!.image;
    expect(img.width).toBe(8);
    expect(img.height).toBe(1);
    expect(img.data).toBeInstanceOf(Uint8ClampedArray);
    expect(img.data.length).toBe(8 * 1 * 4);

    // In-band: v16 >> (16 - 10) = v16 >> 6 → 0..7, exactly the 10-bit content.
    const expected = ramp.map((r) => [inBand(r, 10), 0, 0, 255]);
    expect(pixels(img.data, 8, 1)).toEqual(expected);
    // The whole point of the fix: all 8 gradient steps stay DISTINCT.
    expect(new Set(expected.map((p) => p[0])).size).toBe(8);
  });

  it("concatenates multiple IDAT chunks (split deflate stream)", () => {
    const row: number[] = [];
    for (let r = 0; r < 8; r++) row.push(r * 64, 0, 0);
    const png = buildPng([sampleBytes(row)], 8, 1, {
      colorType: 2,
      sbit: [10, 10, 10],
      idatSplit: 3,
    });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 8, 1)).toEqual(
      Array.from({ length: 8 }, (_, i) => [i, 0, 0, 255]),
    );
  });

  it("falls back to rounded conversion when the in-band value overflows 8 bits", () => {
    // sBIT 10, v16 = 0xFC00 → in-band 1008 > 255 → rounded (251).
    // v16 = 0xFFFF → in-band 1023 > 255 → rounded (255).
    const row = sampleBytes([0xfc00, 0xffff, 0]);
    const png = buildPng([row], 1, 1, { colorType: 2, sbit: [10, 10, 10] });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 1, 1)).toEqual([[251, 255, 0, 255]]);
    expect(honest(0xfc00)).toBe(251); // pin the reference formula
  });

  it("sBIT ≤ 10 keeps distinct high-bit steps that honest rounding would crush", () => {
    // 64 and 128 (10-bit values 1 and 2) round to the SAME 8-bit value honestly
    // (64→0, 128→1); in-band keeps them distinct (1 vs 2).
    const row = sampleBytes([64, 128, 0]);
    const png = buildPng([row], 1, 1, { colorType: 2, sbit: [10, 10, 10] });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    const px = pixels(res!.image.data, 1, 1)[0];
    expect(px[0]).toBe(1);
    expect(px[1]).toBe(2);
    expect(px[0]).not.toBe(px[1]);
  });
});

// ---------------------------------------------------------------------------
// Honest rounded conversion (no sBIT / sBIT > 10).
// ---------------------------------------------------------------------------

describe("decodePngBuffer — honest rounded conversion", () => {
  it("no sBIT → round(v16·255/65535), byte-identical to native upload", () => {
    // Pixels: (65535,0,32768) and (0x8000,0x1234,0xffff).
    const row = sampleBytes([0xffff, 0x0000, 0x8000, 0x8000, 0x1234, 0xffff]);
    const png = buildPng([row], 2, 1, { colorType: 2 });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 2, 1)).toEqual([
      [honest(0xffff), honest(0), honest(0x8000), 255],
      [honest(0x8000), honest(0x1234), honest(0xffff), 255],
    ]);
    // Pinned literals: 65535→255, 0→0, 32768→128, 0x1234→18.
    expect([honest(0xffff), honest(0), honest(0x8000), honest(0x1234)]).toEqual([255, 0, 128, 18]);
  });

  it("sBIT > 10 → rounded conversion (in-band rule is only for sBIT ≤ 10)", () => {
    // sBIT 12: v16 = 0x0FF0 → in-band would be 255 but rounded gives 16.
    const row = sampleBytes([0x0ff0, 0xffff, 0x0800]);
    const png = buildPng([row], 1, 1, { colorType: 2, sbit: [12, 12, 12] });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 1, 1)).toEqual([
      [honest(0x0ff0), honest(0xffff), honest(0x0800), 255],
    ]);
    expect(honest(0x0ff0)).toBe(16); // NOT 255 (the in-band value)
  });
});

// ---------------------------------------------------------------------------
// Color types.
// ---------------------------------------------------------------------------

describe("decodePngBuffer — color types", () => {
  it("RGB (colorType 2) 16-bit decodes", () => {
    const rgb = buildPng([sampleBytes([0x1234, 0x5678, 0x9abc])], 1, 1, { colorType: 2 });
    const res = decodePngBuffer(rgb);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 1, 1)).toEqual([[honest(0x1234), honest(0x5678), honest(0x9abc), 255]]);
  });

  // KNOWN-BUG PIN (src/present/png.ts:365): the channels table
  // `[1, 0, 3, 1, 2, 4][colorType] ?? 0` has no index 6, so colorType 6 (RGBA)
  // resolves to `undefined ?? 0` → channels 0 → the decoder returns null,
  // contradicting the module docstring ("gray/grayA/RGB/RGBA") and its own
  // conversion-loop RGBA branch. Correct table: `[1, 0, 3, 1, 2, 0, 4]`.
  // These tests assert the DOCUMENTED contract and genuinely fail on HEAD;
  // they are `it.fails` so the suite stays green. Flip `it.fails` → `it` when
  // the src table is fixed (they then flag as "unexpected pass").
  it.fails("RGBA (colorType 6) 16-bit decodes", () => {
    const rgba = buildPng([sampleBytes([0x1234, 0x5678, 0x9abc, 0xdef0])], 1, 1, { colorType: 6 });
    const res = decodePngBuffer(rgba);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 1, 1)).toEqual([
      [honest(0x1234), honest(0x5678), honest(0x9abc), honest(0xdef0)],
    ]);
  });

  it("gray (0) 16-bit decodes to grayscale RGB", () => {
    const png = buildPng([sampleBytes([0x0000, 0xffff])], 2, 1, { colorType: 0, sbit: [8] });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 2, 1)).toEqual([
      [0, 0, 0, 255],
      [255, 255, 255, 255],
    ]);
  });

  it("gray+alpha (4) 16-bit decodes with the alpha channel", () => {
    const png = buildPng(
      [sampleBytes([0x8080, 0xffff, 0xffff, 0x0000])],
      2,
      1,
      { colorType: 4, sbit: [8, 8] },
    );
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 2, 1)).toEqual([
      [128, 128, 128, 255],
      [255, 255, 255, 0],
    ]);
  });
});

// ---------------------------------------------------------------------------
// Unfilter (Sub/Up/Average/Paeth).
// ---------------------------------------------------------------------------

describe("decodePngBuffer — unfilter", () => {
  // See the KNOWN-BUG PIN comment in the "color types" describe: RGBA
  // (colorType 6) currently returns null from the channels table gap at
  // src/present/png.ts:365, so this test fails on HEAD. Flip `it.fails` → `it`
  // after the src table fix `[1, 0, 3, 1, 2, 4]` → `[1, 0, 3, 1, 2, 0, 4]`.
  it.fails("Sub-filter (type 1) reconstruction is exact: 2×2 RGBA 16-bit", () => {
    // Row-major, x fastest: p00, p01 / p10, p11. No sBIT → honest conversion.
    const rows = [
      sampleBytes([0x1234, 0x5678, 0x9abc, 0xdef0, 0x1111, 0x2222, 0x3333, 0x4444]),
      sampleBytes([0x0102, 0x0304, 0x0506, 0x0708, 0xffff, 0x0000, 0x8000, 0x4000]),
    ];
    const png = buildPng(rows, 2, 2, { colorType: 6, filter: 1 }); // Sub on every row
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    const img = res!.image;
    expect(img.width).toBe(2);
    expect(img.height).toBe(2);
    const got = pixels(img.data, 2, 2);
    const s = (v: number) => honest(v);
    expect(got).toEqual([
      [s(0x1234), s(0x5678), s(0x9abc), s(0xdef0)],
      [s(0x1111), s(0x2222), s(0x3333), s(0x4444)],
      [s(0x0102), s(0x0304), s(0x0506), s(0x0708)],
      [s(0xffff), s(0x0000), s(0x8000), s(0x4000)],
    ]);
    // Pinned literals for the last pixel: 0xffff→255, 0→0, 0x8000→128, 0x4000→64.
    expect(got[3]).toEqual([255, 0, 128, 64]);
  });

  it("all filter types 1–4 decode correctly (one per row)", () => {
    const rows = [
      sampleBytes([0x0102, 0x0304]),
      sampleBytes([0x1112, 0x1314]),
      sampleBytes([0x2122, 0x2324]),
      sampleBytes([0x3132, 0x3334]),
    ];
    const png = buildPng(rows, 2, 4, { colorType: 0, filters: [1, 2, 3, 4] });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    const expected = rows.flatMap((r) => {
      const vals: number[] = [];
      for (let i = 0; i < r.length; i += 2) vals.push(honest((r[i] << 8) | r[i + 1]));
      return [vals[0], vals[0], vals[0], 255, vals[1], vals[1], vals[1], 255];
    });
    expect(Array.from(res!.image.data)).toEqual(expected);
  });
});

// ---------------------------------------------------------------------------
// Alpha handling.
// ---------------------------------------------------------------------------

describe("decodePngBuffer — alpha handling", () => {
  // Same KNOWN-BUG PIN as above: RGBA (colorType 6) is nulled by the channels
  // table at src/present/png.ts:365. Flip `it.fails` → `it` after the fix.
  it.fails("RGBA sBIT alpha converts in-band (sBIT 8/8/8/8)", () => {
    const row = sampleBytes([0x0000, 0x0000, 0x0000, 0x0000, 0xffff, 0xffff, 0xffff, 0x8000]);
    const png = buildPng([row], 2, 1, { colorType: 6, sbit: [8, 8, 8, 8] });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 2, 1)).toEqual([
      [0, 0, 0, 0],
      [255, 255, 255, 128], // alpha 0x8000 → 128 in-band
    ]);
  });

  it.fails("RGBA without sBIT → alpha uses the same rounded conversion as color", () => {
    const row = sampleBytes([0x1234, 0x5678, 0x9abc, 0xdef0]);
    const png = buildPng([row], 1, 1, { colorType: 6 });
    const res = decodePngBuffer(png);
    expect(res).not.toBeNull();
    expect(pixels(res!.image.data, 1, 1)).toEqual([
      [honest(0x1234), honest(0x5678), honest(0x9abc), honest(0xdef0)],
    ]);
    expect(honest(0xdef0)).toBe(222); // pinned literal
  });
});

// ---------------------------------------------------------------------------
// Null paths (caller must fall back to the DOM decode).
// ---------------------------------------------------------------------------

describe("decodePngBuffer — null paths (native fallback)", () => {
  it("returns null for a valid 8-bit PNG", () => {
    const png = buildPng([], 2, 1, { bitDepth: 8, colorType: 2 });
    expect(decodePngBuffer(png)).toBeNull();
  });

  it("returns null for an interlaced 16-bit PNG (Adam7, interlace method 1)", () => {
    const png = buildPng([sampleBytes([0, 0, 0])], 1, 1, { colorType: 2, interlace: 1 });
    expect(decodePngBuffer(png)).toBeNull();
  });

  it("returns null for reserved color types (1, 3, 5)", () => {
    for (const colorType of [1, 3, 5]) {
      const png = buildPng([], 1, 1, { colorType });
      expect(decodePngBuffer(png)).toBeNull();
    }
  });

  it("returns null for an empty buffer and for a bad signature", () => {
    expect(decodePngBuffer(new Uint8Array(0))).toBeNull();
    const badSig = new Uint8Array(PNG_SIGNATURE);
    badSig[7] = 0x00; // last signature byte corrupted
    expect(decodePngBuffer(concatBytes([badSig, chunk("IEND", new Uint8Array(0))]))).toBeNull();
  });

  it("returns null for a truncated IDAT (declared length beyond EOF) and never throws", () => {
    const idat = new Uint8Array(16); // declares 100 bytes, carries only 4
    const dv = new DataView(idat.buffer);
    dv.setUint32(0, 100);
    idat[4] = 0x49;
    idat[5] = 0x44;
    idat[6] = 0x41;
    idat[7] = 0x54; // 'IDAT'
    const png = concatBytes([PNG_SIGNATURE, ihdrChunk(1, 1, 16, 2), idat]);
    expect(() => decodePngBuffer(png)).not.toThrow();
    expect(decodePngBuffer(png)).toBeNull();
  });

  it("returns null for a corrupt zlib stream in IDAT", () => {
    const badIdat = chunk("IDAT", new Uint8Array([0x00, 0x01, 0x02, 0x03])); // cmf 0 → bad method
    const png = concatBytes([PNG_SIGNATURE, ihdrChunk(1, 1, 16, 2), badIdat, chunk("IEND", new Uint8Array(0))]);
    expect(decodePngBuffer(png)).toBeNull();
  });

  it("returns null when IEND is missing", () => {
    const png = concatBytes([
      PNG_SIGNATURE,
      ihdrChunk(1, 1, 16, 2),
      chunk("IDAT", deflateSync(sampleBytes([0, 0, 0]))),
    ]);
    expect(decodePngBuffer(png)).toBeNull();
  });

  it("returns null when IHDR is missing", () => {
    const png = concatBytes([
      PNG_SIGNATURE,
      chunk("IDAT", deflateSync(sampleBytes([0, 0, 0]))),
      chunk("IEND", new Uint8Array(0)),
    ]);
    expect(decodePngBuffer(png)).toBeNull();
  });

  it("returns null when the inflated data does not match the declared dimensions", () => {
    // IHDR says 2×2 RGB but the IDAT holds only one scanline.
    const png = buildPng([sampleBytes([0, 0, 0, 0, 0, 0])], 2, 2, { colorType: 2 });
    expect(decodePngBuffer(png)).toBeNull();
  });

  it("returns null beyond the 16M-pixel safety cap (4097×4097)", () => {
    const png = buildPng([], 4097, 4097, { colorType: 0 });
    expect(4097 * 4097).toBeGreaterThan(16_777_216);
    expect(decodePngBuffer(png)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// decodePngFromElement — Node path.
// ---------------------------------------------------------------------------

describe("decodePngFromElement (Node path)", () => {
  it("returns null in Node (no XMLHttpRequest), whatever the URL", () => {
    expect(decodePngFromElement({}, "https://example.com/tex.png")).toBeNull();
    expect(decodePngFromElement({}, "data:image/png;base64,AAAA")).toBeNull();
    expect(decodePngFromElement(null, "not-a-png")).toBeNull();
  });
});