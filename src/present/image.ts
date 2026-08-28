/**
 * image.ts — DOM image-source decoding for texImage2D / texSubImage2D.
 *
 * Decodes HTMLImageElement / HTMLCanvasElement / HTMLVideoElement /
 * ImageBitmap / ImageData into straight-alpha RGBA8 for texture upload.
 * DOM access is lazy and feature-detected (the bundle also runs in Node);
 * failures are reported as result objects, never exceptions — gl/ maps them
 * to GL errors (INVALID_VALUE for incomplete/tainted sources per spec).
 * See ../CONTEXT.md contract §4.
 */

/** Decoded image: straight (non-premultiplied) RGBA8 pixels. */
export interface DecodedImage {
  readonly width: number;
  readonly height: number;
  /** RGBA8 straight-alpha pixels, length = width * height * 4. Owned by the caller. */
  readonly data: Uint8ClampedArray;
}

/** DOM image sources accepted by texImage2D/texSubImage2D (WebGL1+2). */
export type ImageSource =
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement
  | ImageBitmap
  | ImageData
  | OffscreenCanvas;

/** Result of a decode attempt; gl/ maps {ok:false} to INVALID_VALUE. */
export type DecodeResult =
  | { ok: true; image: DecodedImage }
  | { ok: false; reason: string };

/**
 * Shared scratch canvas + 2D context, created lazily on first use and reused
 * across decodes (no per-call DOM allocation). Guarded by feature detection so
 * the bundle runs in Node without any DOM globals. Null when unavailable.
 */
let scratchCanvas: HTMLCanvasElement | null = null;
let scratchCtx: CanvasRenderingContext2D | null = null;

/**
 * Returns the module-level scratch 2D context, creating it on first use.
 * Returns null (and never throws) when no DOM / 2D support is available.
 */
function getScratchContext(): CanvasRenderingContext2D | null {
  if (scratchCtx !== null) {
    return scratchCtx;
  }
  if (typeof document === 'undefined') {
    return null;
  }
  try {
    if (scratchCanvas === null) {
      scratchCanvas = document.createElement('canvas');
    }
    scratchCtx = scratchCanvas.getContext('2d');
  } catch {
    scratchCtx = null;
  }
  return scratchCtx;
}

/**
 * Decodes a DOM-drawable source (image/video/canvas/bitmap) by drawing it at
 * its intrinsic size onto the scratch canvas and reading back via getImageData
 * (already straight RGBA8). Every failure — no 2D support, drawImage errors,
 * SecurityError from a tainted source — is mapped to {ok:false}; never throws.
 */
function decodeViaScratch(source: unknown, width: number, height: number): DecodeResult {
  if (!(width > 0) || !(height > 0) || !Number.isInteger(width) || !Number.isInteger(height)) {
    return { ok: false, reason: `invalid source dimensions ${width}x${height}` };
  }
  const ctx = getScratchContext();
  if (ctx === null) {
    return { ok: false, reason: 'no 2D context available for image decode' };
  }
  try {
    // Size the scratch canvas to the source's intrinsic size (setting
    // width/height resets the 2D context state). Only reallocates when the
    // size actually changes.
    if (scratchCanvas !== null && (scratchCanvas.width !== width || scratchCanvas.height !== height)) {
      scratchCanvas.width = width;
      scratchCanvas.height = height;
    }
    // The scratch is a module-level singleton shared across ALL decodes (and
    // all contexts on the page). drawImage composites source-over, so a
    // same-size decode would blend with the previous decode's pixels. Reset
    // the transform and clear the scratch to transparent before every draw —
    // only then does getImageData return exact straight-alpha pixels.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(source as CanvasImageSource, 0, 0);
    const imageData = ctx.getImageData(0, 0, width, height);
    return { ok: true, image: { width, height, data: imageData.data } };
  } catch (e) {
    // SecurityError (tainted source), broken image, invalid state, etc.
    return { ok: false, reason: e instanceof Error ? e.message : String(e) };
  }
}

/**
 * Decodes any supported DOM image source to straight-alpha RGBA8.
 * ImageData sources take the direct-copy path (no DOM needed); everything
 * else is drawn onto a lazily created scratch 2D canvas and read back via
 * getImageData. Returns {ok:false} — never throws — for incomplete images
 * (image not loaded or broken, video without current data), tainted canvases
 * (SecurityError from getImageData), and unsupported or null sources.
 */
export function decodeImageSource(source: ImageSource): DecodeResult {
  if (source === null || typeof source !== 'object') {
    return { ok: false, reason: 'source is not an object' };
  }
  const v = source as unknown as Record<string, unknown>;

  // (a) ImageData duck-type → direct copy path (no DOM, Node-testable).
  if (v.data instanceof Uint8ClampedArray || v.data instanceof Uint8Array) {
    return decodeImageData(source as ImageData);
  }

  // (b) HTMLImageElement duck-type — always use naturalWidth/naturalHeight
  // (intrinsic size), never clientWidth/offsetWidth (CSS size).
  if (typeof v.naturalWidth === 'number') {
    if (
      typeof v.naturalHeight !== 'number' ||
      v.naturalWidth <= 0 ||
      v.naturalHeight <= 0 ||
      v.complete !== true
    ) {
      return { ok: false, reason: 'image is not loaded or has no intrinsic size' };
    }
    return decodeViaScratch(source, v.naturalWidth, v.naturalHeight);
  }

  // (c) HTMLVideoElement duck-type — require current frame data (readyState
  // >= HAVE_CURRENT_DATA) and a nonzero intrinsic size.
  if (typeof v.videoWidth === 'number' && typeof v.readyState === 'number') {
    if (
      typeof v.videoHeight !== 'number' ||
      v.readyState < 2 ||
      v.videoWidth <= 0 ||
      v.videoHeight <= 0
    ) {
      return { ok: false, reason: 'video has no current frame data' };
    }
    return decodeViaScratch(source, v.videoWidth, v.videoHeight);
  }

  // (d) Canvas-like (HTMLCanvasElement / OffscreenCanvas).
  if (typeof v.getContext === 'function') {
    return decodeViaScratch(source, v.width as number, v.height as number);
  }

  // (e) ImageBitmap duck-type.
  if (typeof v.width === 'number' && typeof v.height === 'number') {
    return decodeViaScratch(source, v.width, v.height);
  }

  // (f) Anything else.
  return { ok: false, reason: 'unsupported image source' };
}

/**
 * Pure path: ImageData → direct copy of its (already straight) RGBA8 data.
 * Requires no DOM, so it is unit-testable in Node; `source` may be a
 * duck-typed { width, height, data: Uint8ClampedArray } in tests.
 */
export function decodeImageData(source: ImageData): DecodeResult {
  if (source === null || typeof source !== 'object') {
    return { ok: false, reason: 'source is not an object' };
  }
  const v = source as unknown as Record<string, unknown>;
  const { width, height, data } = v;
  if (typeof width !== 'number' || typeof height !== 'number') {
    return { ok: false, reason: 'source must have numeric width and height' };
  }
  if (!(data instanceof Uint8ClampedArray) && !(data instanceof Uint8Array)) {
    return { ok: false, reason: 'source data must be a Uint8ClampedArray (or Uint8Array view)' };
  }
  const expected = width * height * 4;
  if (data.length !== expected) {
    return {
      ok: false,
      reason: `data length ${data.length} does not match ${width}x${height} RGBA8 (expected ${expected})`,
    };
  }
  // Fresh copy: the caller keeps ownership of its buffer; mutating the input
  // after decode must not affect the result.
  return { ok: true, image: { width, height, data: new Uint8ClampedArray(data) } };
}

/**
 * True when `value` is a DOM image source this module can decode. Used by gl/
 * to dispatch texImage2D arguments: ArrayBufferView/ArrayBuffer → buffer path;
 * isDecodableImageSource → decode path; anything else → INVALID_VALUE.
 */
export function isDecodableImageSource(value: unknown): boolean {
  if (value === null || typeof value !== 'object') {
    return false;
  }
  // ArrayBuffer / ArrayBufferView belong to the buffer-upload path, not here.
  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    return false;
  }
  const v = value as Record<string, unknown>;
  // ImageData duck-type.
  if (v.data instanceof Uint8ClampedArray || v.data instanceof Uint8Array) {
    return true;
  }
  // HTMLImageElement duck-type.
  if (typeof v.naturalWidth === 'number') {
    return true;
  }
  // HTMLVideoElement duck-type.
  if (typeof v.videoWidth === 'number' && typeof v.readyState === 'number') {
    return true;
  }
  // Canvas-like (HTMLCanvasElement / OffscreenCanvas).
  if (typeof v.getContext === 'function') {
    return true;
  }
  // ImageBitmap duck-type.
  if (typeof v.width === 'number' && typeof v.height === 'number') {
    return true;
  }
  return false;
}
