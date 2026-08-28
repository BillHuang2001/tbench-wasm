/**
 * image.ts — DOM image-source decoding for texImage2D / texSubImage2D.
 *
 * Decodes HTMLImageElement / HTMLCanvasElement / HTMLVideoElement /
 * ImageBitmap / VideoFrame / ImageData into straight-alpha RGBA8 for texture
 * upload. Preferred path: BYTE-EXACT readback through a dedicated scratch
 * NATIVE WebGL1 context (texImage2D → framebufferTexture2D → readPixels),
 * which applies no color management; fallback: 2D drawImage + getImageData.
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
  | OffscreenCanvas
  | VideoFrame;

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
 * NATIVE HTMLCanvasElement.prototype.getContext, captured at module scope.
 * The test harness (src/context-intercept.ts) later overrides
 * HTMLCanvasElement.prototype.getContext to route 'webgl'/'webgl2' requests to
 * the software renderer — but module evaluation runs before that override is
 * installed, so this reference stays the genuine native implementation.
 * Invoking it via .call(scratchCanvas, 'webgl', attrs) on a dedicated scratch
 * canvas bypasses the page's override and never recurses into the software
 * renderer. Null in Node (no DOM).
 */
const NATIVE_GET_CONTEXT: ((type: string, attrs?: unknown) => unknown) | null =
  typeof HTMLCanvasElement !== 'undefined'
    ? (HTMLCanvasElement.prototype.getContext as unknown as (type: string, attrs?: unknown) => unknown)
    : null;

/**
 * Dedicated scratch NATIVE WebGL1 context for byte-exact DOM-source decoding
 * (see decodeViaNativeWebGL). This canvas must NEVER get a 2D context: a
 * canvas that already has a 2D context returns null for getContext('webgl').
 * Lazily created on first use; null when unavailable (Node, no WebGL).
 */
let glScratchCanvas: HTMLCanvasElement | null = null;
let glScratchContext: WebGLRenderingContext | null = null;
let glScratchTexture: WebGLTexture | null = null;
let glScratchFramebuffer: WebGLFramebuffer | null = null;

/**
 * Returns the module-level scratch native WebGL1 context, creating it on first
 * use. Returns null (and never throws) when no DOM / native WebGL is
 * available. Created with premultipliedAlpha:false + preserveDrawingBuffer:true
 * so readPixels returns exact straight RGBA; alpha:true so the alpha channel
 * survives the readback (an {alpha:false} context would force alpha 255).
 */
function getNativeGLContext(): WebGLRenderingContext | null {
  if (glScratchContext !== null) {
    return glScratchContext;
  }
  if (NATIVE_GET_CONTEXT === null || typeof document === 'undefined') {
    return null;
  }
  try {
    if (glScratchCanvas === null) {
      glScratchCanvas = document.createElement('canvas');
    }
    const gl = NATIVE_GET_CONTEXT.call(glScratchCanvas, 'webgl', {
      alpha: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
      antialias: false,
      depth: false,
      stencil: false,
    }) as WebGLRenderingContext | null;
    if (gl === null) {
      return null;
    }
    glScratchContext = gl;
    glScratchTexture = gl.createTexture();
    glScratchFramebuffer = gl.createFramebuffer();
    return gl;
  } catch {
    glScratchContext = null;
    return null;
  }
}

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
 * Byte-exact DOM-source decode via a native WebGL1 scratch context:
 * texImage2D(source) → framebufferTexture2D → readPixels. Unlike the 2D
 * drawImage+getImageData path, the native GL upload applies NO color
 * management (UNPACK_COLORSPACE_CONVERSION_WEBGL = NONE), so RGB levels
 * survive the round trip bit-exact — the 2D path round-trips sRGB→linear→sRGB
 * and collapses levels (~205 unique values instead of 256 on a 256-level
 * ramp), and honors ICC profiles where the WebGL spec says uploads must be
 * profile-ignored (CTS gl-teximage ICC subtests).
 *
 * The readback is TOP-DOWN straight RGBA8 (row 0 = source top row, never
 * premultiplied, never flipped): gl/teximage.ts applies UNPACK_FLIP_Y and
 * UNPACK_PREMULTIPLY_ALPHA itself, so the decode must apply neither.
 *
 * Any failure — no native WebGL, tainted source (SecurityError from
 * texImage2D), incomplete framebuffer, GL error — maps to {ok:false}; never
 * throws. A tainted-source reason is guaranteed to match
 * /security|taint|insecure/i so gl/ rethrows a SecurityError to the page.
 */
function decodeViaNativeWebGL(source: unknown, width: number, height: number): DecodeResult {
  if (!(width > 0) || !(height > 0) || !Number.isInteger(width) || !Number.isInteger(height)) {
    return { ok: false, reason: `invalid source dimensions ${width}x${height}` };
  }
  const gl = getNativeGLContext();
  if (gl === null) {
    return { ok: false, reason: 'no native WebGL context available for image decode' };
  }
  try {
    // Setting canvas.width/height RESETS all context state, so the pixel-store
    // parameters are re-applied on every call (they are cheap) rather than only
    // after a resize. UNPACK_COLORSPACE_CONVERSION_WEBGL = NONE is REQUIRED:
    // the native upload must not color-manage (the 2D fallback path does).
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    // readPixels is scissor-affected and canvas resizes reset scissor state,
    // so keep the scissor test disabled before every readback.
    gl.disable(gl.SCISSOR_TEST);
    if (glScratchCanvas !== null && (glScratchCanvas.width !== width || glScratchCanvas.height !== height)) {
      glScratchCanvas.width = width;
      glScratchCanvas.height = height;
    }
    gl.bindTexture(gl.TEXTURE_2D, glScratchTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source as TexImageSource);
    gl.bindFramebuffer(gl.FRAMEBUFFER, glScratchFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, glScratchTexture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      return { ok: false, reason: 'native framebuffer incomplete for image decode' };
    }
    const buf = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    if (gl.getError() !== gl.NO_ERROR) {
      return { ok: false, reason: 'native readback failed for image decode' };
    }
    return { ok: true, image: { width, height, data: new Uint8ClampedArray(buf) } };
  } catch (e) {
    // SecurityError from native texImage2D (tainted/cross-origin source).
    // Chromium's message ("The image element contains cross-origin data, and
    // may not be loaded.") does NOT contain the words security/taint/insecure,
    // so prefix the DOMException name to keep gl/'s /security|taint|insecure/i
    // mapping working.
    const msg = e instanceof Error ? e.message : String(e);
    const name = (e as { name?: unknown } | null)?.name;
    if (name === 'SecurityError' && !/security|taint|insecure/i.test(msg)) {
      return { ok: false, reason: `security: ${msg}` };
    }
    return { ok: false, reason: msg };
  }
}

/**
 * Native-WebGL readback with the 2D drawImage+getImageData path as fallback.
 * Tainted-source failures propagate unchanged (gl/ must throw a SecurityError
 * to the page); any other native failure — no native WebGL available (Node),
 * a source the GL path cannot upload — falls back to the 2D path, which
 * accepts the same sources and reports the same {ok:false} contract.
 */
function decodeViaNativeWebGLWithFallback(source: unknown, width: number, height: number): DecodeResult {
  const res = decodeViaNativeWebGL(source, width, height);
  if (res.ok || /security|taint|insecure/i.test(res.reason)) {
    return res;
  }
  return decodeViaScratch(source, width, height);
}

/**
 * Decodes any supported DOM image source to straight-alpha RGBA8.
 * ImageData sources take the direct-copy path (no DOM needed); everything else
 * is decoded by the native-WebGL readback path (byte-exact, no color
 * management) with the scratch-2D drawImage+getImageData path as fallback.
 * ImageBitmap stays on the 2D path on purpose (see the (e) branch).
 * Returns {ok:false} — never throws — for incomplete images (image not loaded
 * or broken, video without current data), tainted canvases (SecurityError),
 * and unsupported or null sources.
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
    return decodeViaNativeWebGLWithFallback(source, v.naturalWidth, v.naturalHeight);
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
    return decodeViaNativeWebGLWithFallback(source, v.videoWidth, v.videoHeight);
  }

  // (c2) VideoFrame duck-type (native texImage2D and 2D drawImage both accept
  // VideoFrame). Display size, not coded size — that is what drawImage paints.
  // Placed BEFORE the canvas-like and generic branches.
  if (
    typeof v.format === 'string' &&
    typeof v.displayWidth === 'number' &&
    typeof v.displayHeight === 'number'
  ) {
    if (v.displayWidth <= 0 || v.displayHeight <= 0) {
      return { ok: false, reason: 'video frame has no display size' };
    }
    return decodeViaNativeWebGLWithFallback(source, v.displayWidth, v.displayHeight);
  }

  // (d) Canvas-like (HTMLCanvasElement / OffscreenCanvas).
  if (typeof v.getContext === 'function') {
    return decodeViaNativeWebGLWithFallback(source, v.width as number, v.height as number);
  }

  // (e) ImageBitmap duck-type → 2D path only (NOT native readback): native
  // readback returns the bitmap's RAW STORAGE bytes, which are premultiplied
  // for default-created bitmaps — and gl/teximage.ts premultiplies AGAIN
  // (bitmap.premultiply !== false → premultiply:true), double-premultiplying.
  // The 2D drawImage+getImageData round trip always yields straight alpha,
  // which is exactly what gl/ expects. Bitmaps created with
  // premultiplyAlpha:'none' stay straight on both paths.
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
  // VideoFrame duck-type.
  if (
    typeof v.format === 'string' &&
    typeof v.displayWidth === 'number' &&
    typeof v.displayHeight === 'number'
  ) {
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
