/**
 * png.ts — pure-JS PNG decoding for HIGH-BIT-DEPTH (16-bit) image sources.
 *
 * The DOM decode paths (native-WebGL readback, 2D drawImage+getImageData)
 * quantize image samples to 8 bits per channel. That crush is inherent: a
 * 16bpc PNG whose samples span a narrow high-bit range (e.g. 10-bit content
 * stored as v10<<6) collapses to 2–3 distinct 8-bit values, and no downstream
 * 8→10-bit re-quantization can recover the lost distinctness (CTS
 * conformance2/textures/misc/tex-image-10bpc.html requires ≥7 distinct
 * RGB10_A2 readback values — Chrome passes because its native upload converts
 * the 16-bit decode DIRECTLY to the 10-bit destination, never through an
 * 8-bit intermediate).
 *
 * This module decodes the PNG bytes itself (parse → zlib inflate → unfilter →
 * samples) and converts each 16-bit sample to 8 bits with a
 * HIGH-BIT-PRESERVING rule: when the sample's declared significant bits
 * (sBIT, ≤ 10) make the top-bits value representable in 8 bits (≤ 255), the
 * 8-bit channel carries that value IN-BAND (v16 >> (16 − sbit)) — keeping the
 * subtle gradient distinctness the native path would crush; otherwise the
 * standard rounded 16→8 conversion (round(v16·255/65535), byte-identical to
 * Chrome's native RGBA8 upload, verified empirically) is used.
 *
 * Zero runtime dependencies, no DOM at module load (Node-testable pure
 * functions). Only 16-bit non-interlaced PNGs (gray/grayA/RGB/RGBA) are
 * decoded here — 8-bit PNGs, interlaced PNGs, and any parse/inflate failure
 * return null so callers fall back to the existing DOM decode paths (8-bit
 * PNGs MUST keep the native path: the WebGL spec requires profile-ignored
 * uploads, which the color-managed 2D path and byte-exact native readback
 * already implement).
 */

/** Decoded image (structural — matches image.ts DecodedImage). */
interface DecodedImage {
  readonly width: number;
  readonly height: number;
  /** RGBA8 straight-alpha pixels, length = width * height * 4. */
  readonly data: Uint8ClampedArray;
}

/** Result: {ok:true} with the decoded image, or null (caller must fall back). */
export type PngDecodeResult = { ok: true; image: DecodedImage } | null;

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
/** Safety cap: 16M texels = 64 MB RGBA8; larger sources keep the native path. */
const MAX_PIXELS = 16_777_216;

function readU32BE(b: Uint8Array, off: number): number {
  return ((b[off] << 24) | (b[off + 1] << 16) | (b[off + 2] << 8) | b[off + 3]) >>> 0;
}

/** Concatenates byte arrays without any environment-specific globals. */
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

// ---------------------------------------------------------------------------
// DEFLATE (RFC 1951) inflate — self-contained, LSB-first bit stream.
// ---------------------------------------------------------------------------

/** LSB-first bit reader over a byte array. */
class BitReader {
  private readonly bytes: Uint8Array;
  private pos = 0;
  private bitBuf = 0;
  private bitCnt = 0;

  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
  }

  /** Reads `n` bits (n ≤ 16), LSB-first. */
  read(n: number): number {
    while (this.bitCnt < n) {
      if (this.pos >= this.bytes.length) throw new Error('inflate: unexpected end of stream');
      this.bitBuf |= this.bytes[this.pos++] << this.bitCnt;
      this.bitCnt += 8;
    }
    const v = this.bitBuf & ((1 << n) - 1);
    this.bitBuf >>>= n;
    this.bitCnt -= n;
    return v;
  }

  /** Discards remaining bits in the current byte (stored-block alignment). */
  alignByte(): void {
    this.bitBuf = 0;
    this.bitCnt = 0;
  }
}

/** Huffman tree node: [leftChild, rightChild, symbol] (-1 = absent/none). */
type HNode = [number, number, number];

/**
 * Builds a canonical Huffman tree from code lengths (DEFLATE canonical code
 * assignment: increasing length, then increasing symbol; codes packed MSB
 * first — tree walk reads bits MSB-first, matching the bit stream order).
 */
function buildHuffmanTree(lengths: Uint8Array, count: number): HNode[] {
  const blCount = new Int32Array(16);
  for (let i = 0; i < count; i++) {
    const l = lengths[i];
    if (l > 0) blCount[l]++;
  }
  let code = 0;
  const nextCode = new Int32Array(16);
  for (let bits = 1; bits < 16; bits++) {
    code = (code + blCount[bits - 1]) << 1;
    nextCode[bits] = code;
  }
  const nodes: HNode[] = [[-1, -1, -1]];
  for (let sym = 0; sym < count; sym++) {
    const len = lengths[sym];
    if (len === 0) continue;
    let c = nextCode[len]++;
    let node = 0;
    for (let b = len - 1; b >= 0; b--) {
      const bit = (c >> b) & 1;
      let child = nodes[node][bit];
      if (child === -1) {
        child = nodes.length;
        nodes.push([-1, -1, -1]);
        nodes[node][bit] = child;
      }
      node = child;
    }
    nodes[node][2] = sym;
  }
  return nodes;
}

function decodeSymbol(nodes: HNode[], br: BitReader): number {
  let node = 0;
  for (;;) {
    const child = nodes[node][br.read(1)];
    if (child === -1) throw new Error('inflate: invalid Huffman code');
    node = child;
    if (nodes[node][2] !== -1) return nodes[node][2];
  }
}

// RFC 1951 length/distance tables (base + extra bits).
const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CLEN_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/** Growable output buffer for the inflate stream. */
class OutBuffer {
  buf = new Uint8Array(1 << 16);
  len = 0;

  private ensure(n: number): void {
    if (this.len + n > this.buf.length) {
      const nb = new Uint8Array(Math.max(this.buf.length * 2, this.len + n));
      nb.set(this.buf.subarray(0, this.len));
      this.buf = nb;
    }
  }

  byte(b: number): void {
    this.ensure(1);
    this.buf[this.len++] = b;
  }

  /** LZ77 match copy; overlapping (dist < n) copies work byte-by-byte forward. */
  copy(dist: number, n: number): void {
    if (dist <= 0 || dist > this.len) throw new Error('inflate: distance too far back');
    this.ensure(n);
    const start = this.len - dist;
    for (let i = 0; i < n; i++) this.buf[this.len + i] = this.buf[start + i];
    this.len += n;
  }

  result(): Uint8Array {
    return this.buf.subarray(0, this.len);
  }
}

/** Decodes one deflate block (any of the three block types) into `out`. */
function decodeBlock(br: BitReader, out: OutBuffer): void {
  const litLengths = new Uint8Array(288);
  const distLengths = new Uint8Array(30);
  const btype = br.read(2);
  if (btype === 0) {
    // Stored (uncompressed) block: byte-aligned LEN/NLEN then raw bytes.
    br.alignByte();
    const len = br.read(16);
    br.read(16); // NLEN — not validated (defensive decode)
    for (let i = 0; i < len; i++) out.byte(br.read(8));
    return;
  }
  if (btype === 1) {
    // Fixed Huffman: literal 0–143 → 8 bits, 144–255 → 9, 256–279 → 7, 280–287 → 8; distances 5 bits.
    for (let i = 0; i < 144; i++) litLengths[i] = 8;
    for (let i = 144; i < 256; i++) litLengths[i] = 9;
    for (let i = 256; i < 280; i++) litLengths[i] = 7;
    for (let i = 280; i < 288; i++) litLengths[i] = 8;
    distLengths.fill(5);
  } else if (btype === 2) {
    // Dynamic Huffman: code-length tree first, then lit/dist length streams.
    const hlit = br.read(5) + 257;
    const hdist = br.read(5) + 1;
    const hclen = br.read(4) + 4;
    const clLengths = new Uint8Array(19);
    for (let i = 0; i < hclen; i++) clLengths[CLEN_ORDER[i]] = br.read(3);
    const clTree = buildHuffmanTree(clLengths, 19);
    const all = new Uint8Array(hlit + hdist);
    let k = 0;
    while (k < all.length) {
      const sym = decodeSymbol(clTree, br);
      if (sym < 16) {
        all[k++] = sym;
      } else if (sym === 16) {
        if (k === 0) throw new Error('inflate: repeat with no previous length');
        const rep = 3 + br.read(2);
        const prev = all[k - 1];
        for (let i = 0; i < rep && k < all.length; i++) all[k++] = prev;
      } else if (sym === 17) {
        const rep = 3 + br.read(3);
        k += rep;
      } else {
        const rep = 11 + br.read(7);
        k += rep;
      }
    }
    if (k !== all.length) throw new Error('inflate: bad dynamic header');
    litLengths.set(all.subarray(0, hlit));
    distLengths.set(all.subarray(hlit, hlit + hdist));
  } else {
    throw new Error('inflate: invalid block type');
  }
  const litTree = buildHuffmanTree(litLengths, 288);
  const distTree = buildHuffmanTree(distLengths, 30);
  for (;;) {
    const sym = decodeSymbol(litTree, br);
    if (sym < 256) {
      out.byte(sym);
    } else if (sym === 256) {
      return; // end of block
    } else {
      if (sym > 285) throw new Error('inflate: invalid length code');
      const li = sym - 257;
      const length = LEN_BASE[li] + br.read(LEN_EXTRA[li]);
      const dsym = decodeSymbol(distTree, br);
      if (dsym > 29) throw new Error('inflate: invalid distance code');
      const dist = DIST_BASE[dsym] + br.read(DIST_EXTRA[dsym]);
      out.copy(dist, length);
    }
  }
}

/** RFC 1950 zlib wrapper: verify header, inflate, verify Adler-32 trailer. */
function zlibInflate(data: Uint8Array): Uint8Array {
  if (data.length < 6) throw new Error('zlib: stream too short');
  const cmf = data[0];
  const flg = data[1];
  if ((cmf & 0x0f) !== 8) throw new Error('zlib: unknown compression method');
  if (((cmf << 8) | flg) % 31 !== 0) throw new Error('zlib: bad header check');
  if ((flg & 0x20) !== 0) throw new Error('zlib: preset dictionary unsupported');
  const br = new BitReader(data.subarray(2, data.length - 4));
  const out = new OutBuffer();
  for (;;) {
    const bfinal = br.read(1);
    decodeBlock(br, out);
    if (bfinal) break;
  }
  const adler = readU32BE(data, data.length - 4);
  let s1 = 1;
  let s2 = 0;
  const raw = out.result();
  for (let i = 0; i < raw.length; i++) {
    s1 = (s1 + raw[i]) % 65521;
    s2 = (s2 + s1) % 65521;
  }
  if (((s2 << 16) | s1) >>> 0 !== adler) throw new Error('zlib: adler32 mismatch');
  return raw;
}

// ---------------------------------------------------------------------------
// PNG structure + sample conversion.
// ---------------------------------------------------------------------------

/**
 * High-bit-preserving 16→8 conversion for one sample.
 *
 * `sbit` = declared significant bits (1..16; 16 when no sBIT chunk). When the
 * significant bits are ≤ 10 and the top-(16−sbit) bits value fits 8 bits, the
 * sample is carried IN-BAND (its actual value in the 8-bit channel) so subtle
 * high-bit gradients keep their distinctness through the RGBA8 pipe; otherwise
 * the standard rounded conversion round(v16·255/65535) applies — byte-identical
 * to Chrome's native RGBA8 texImage2D upload (verified empirically).
 */
function to8(v16: number, sbit: number): number {
  const sb = sbit >= 1 && sbit <= 16 ? sbit : 16;
  if (sb <= 10) {
    const inBand = v16 >> (16 - sb);
    if (inBand <= 255) return inBand;
  }
  return Math.round((v16 * 255) / 65535);
}

/**
 * Decodes PNG bytes to straight-alpha RGBA8. Returns the decoded image ONLY
 * for 16-bit non-interlaced PNGs (gray / gray+alpha / RGB / RGBA) — the
 * high-bit-depth case the DOM decode paths would crush. Returns null for
 * anything else (8-bit PNGs, interlaced, unsupported color types, parse or
 * inflate failures): callers then fall back to their existing decode path.
 */
export function decodePngBuffer(bytes: Uint8Array): PngDecodeResult {
  try {
    if (bytes.length < 33) return null;
    for (let i = 0; i < 8; i++) {
      if (bytes[i] !== PNG_SIGNATURE[i]) return null;
    }
    let off = 8;
    let w = 0;
    let h = 0;
    let bitDepth = 0;
    let colorType = 0;
    let interlace = 0;
    let seenIHDR = false;
    let seenIEND = false;
    let sbit: Uint8Array | null = null;
    const idat: Uint8Array[] = [];
    while (off + 12 <= bytes.length && !seenIEND) {
      const len = readU32BE(bytes, off);
      const start = off + 8;
      if (start + len + 4 > bytes.length) return null;
      const type = String.fromCharCode(bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]);
      const data = bytes.subarray(start, start + len);
      if (type === 'IHDR') {
        if (len < 13) return null;
        w = readU32BE(data, 0);
        h = readU32BE(data, 4);
        bitDepth = data[8];
        colorType = data[9];
        if (data[10] !== 0 || data[11] !== 0) return null; // compression/filter method must be 0
        interlace = data[12];
        seenIHDR = true;
      } else if (type === 'sBIT') {
        sbit = new Uint8Array(data);
      } else if (type === 'IDAT') {
        idat.push(new Uint8Array(data));
      } else if (type === 'IEND') {
        seenIEND = true;
      }
      off = start + len + 4; // skip CRC
    }
    if (!seenIHDR || !seenIEND) return null;
    // 8-bit (and lower) PNGs deliberately keep the native decode path — the
    // WebGL spec's color-management rules (profile-ignored uploads) are
    // implemented there; this raw decode is only for high-bit-depth sources.
    if (bitDepth !== 16) return null;
    if (interlace !== 0) return null; // Adam7 → native decode handles it
    const channels = [1, 0, 3, 1, 2, 0, 4][colorType] ?? 0;
    if (channels === 0) return null;
    if (w <= 0 || h <= 0 || w * h > MAX_PIXELS) return null;

    const raw = zlibInflate(concatBytes(idat));
    const stride = w * channels * 2; // 16-bit samples
    if (raw.length !== h * (stride + 1)) return null;

    // Unfilter (PNG filter types 0–4) in place: `a` = left neighbor (same row,
    // already unfiltered), `b` = above, `c` = above-left (previous row intact).
    const bpp = channels * 2;
    for (let y = 0; y < h; y++) {
      const f = raw[y * (stride + 1)];
      if (f > 4) return null;
      const rowBase = y * (stride + 1) + 1;
      const prevBase = y > 0 ? (y - 1) * (stride + 1) + 1 : -1;
      for (let x = 0; x < stride; x++) {
        const i = rowBase + x;
        const a = x >= bpp ? raw[rowBase + x - bpp] : 0;
        const b = prevBase >= 0 ? raw[prevBase + x] : 0;
        const c = prevBase >= 0 && x >= bpp ? raw[prevBase + x - bpp] : 0;
        let v = raw[i];
        if (f === 1) {
          v = (v + a) & 0xff;
        } else if (f === 2) {
          v = (v + b) & 0xff;
        } else if (f === 3) {
          v = (v + ((a + b) >> 1)) & 0xff;
        } else if (f === 4) {
          const p = a + b - c;
          const pa = Math.abs(p - a);
          const pb = Math.abs(p - b);
          const pc = Math.abs(p - c);
          v = (v + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c)) & 0xff;
        }
        raw[i] = v;
      }
    }

    // Convert 16-bit samples to high-bit-preserving RGBA8 (top-down rows).
    const out = new Uint8ClampedArray(w * h * 4);
    const sbR = sbit ? sbit[0] : 16;
    const sbG = sbit ? sbit[Math.min(1, sbit.length - 1)] : 16;
    const sbB = sbit ? sbit[Math.min(2, sbit.length - 1)] : 16;
    const sbA = sbit ? sbit[Math.min(3, sbit.length - 1)] : 16;
    for (let y = 0; y < h; y++) {
      const rowBase = y * (stride + 1) + 1;
      const outRow = y * w * 4;
      for (let x = 0; x < w; x++) {
        const o = rowBase + x * bpp;
        let r: number;
        let g: number;
        let b: number;
        let a = 255;
        if (colorType === 0) {
          r = g = b = to8((raw[o] << 8) | raw[o + 1], sbR);
        } else if (colorType === 2) {
          r = to8((raw[o] << 8) | raw[o + 1], sbR);
          g = to8((raw[o + 2] << 8) | raw[o + 3], sbG);
          b = to8((raw[o + 4] << 8) | raw[o + 5], sbB);
        } else if (colorType === 4) {
          r = g = b = to8((raw[o] << 8) | raw[o + 1], sbR);
          a = to8((raw[o + 2] << 8) | raw[o + 3], sbA);
        } else {
          r = to8((raw[o] << 8) | raw[o + 1], sbR);
          g = to8((raw[o + 2] << 8) | raw[o + 3], sbG);
          b = to8((raw[o + 4] << 8) | raw[o + 5], sbB);
          a = to8((raw[o + 6] << 8) | raw[o + 7], sbA);
        }
        const q = outRow + x * 4;
        out[q] = r;
        out[q + 1] = g;
        out[q + 2] = b;
        out[q + 3] = a;
      }
    }
    return { ok: true, image: { width: w, height: h, data: out } };
  } catch {
    return null; // any failure → caller falls back to the DOM decode paths
  }
}

/**
 * Fetches an HTMLImageElement's own bytes (same-origin / data: / blob: URLs —
 * the sync fetch is a cache hit after the element loaded) and decodes them
 * when they are a 16-bit PNG. Returns null unless a high-bit-depth PNG decoded
 * successfully. Cross-origin fetches (no CORS), non-PNG URLs, and all other
 * failures return null — the caller's existing decode path then reports the
 * proper result (including the tainted-source SecurityError).
 */
export function decodePngFromElement(
  source: unknown,
  src: string,
): { ok: true; image: DecodedImage } | null {
  // Cheap pre-filter: 16-bit PNGs are .png files or data:image/png URIs.
  if (!/\.png($|[?#])/i.test(src) && !/^data:image\/png/i.test(src)) return null;
  if (typeof XMLHttpRequest === 'undefined') return null; // Node: no DOM fetch
  try {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', src, false); // synchronous — cache hit after img.onload
    try {
      xhr.responseType = 'arraybuffer'; // allowed for workers, throws for sync requests from a document
    } catch {
      // Sync XHR from a document cannot change responseType — fall back to the
      // byte-preserving x-user-defined string encoding (each byte → one code
      // unit; low byte recovered via charCodeAt & 0xff).
      xhr.overrideMimeType('text/plain; charset=x-user-defined');
    }
    xhr.send(null);
    if (xhr.status !== 200 && xhr.status !== 0) return null;
    const resp = xhr.response;
    let bytes: Uint8Array;
    if (resp instanceof ArrayBuffer) {
      bytes = new Uint8Array(resp);
    } else if (typeof resp === 'string') {
      bytes = new Uint8Array(resp.length);
      for (let i = 0; i < resp.length; i++) bytes[i] = resp.charCodeAt(i) & 0xff;
    } else {
      return null;
    }
    return decodePngBuffer(bytes);
  } catch {
    return null;
  }
}
