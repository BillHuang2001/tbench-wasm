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
 * Decodes any supported DOM image source to straight-alpha RGBA8.
 * ImageData sources take the direct-copy path (no DOM needed); everything
 * else is drawn onto a lazily created scratch 2D canvas and read back via
 * getImageData. Returns {ok:false} — never throws — for incomplete images
 * (image not loaded or broken, video without current data), tainted canvases
 * (SecurityError from getImageData), and unsupported or null sources.
 */
export function decodeImageSource(source: ImageSource): DecodeResult {
  throw new Error('not implemented: present/image.ts decodeImageSource');
}

/**
 * Pure path: ImageData → direct copy of its (already straight) RGBA8 data.
 * Requires no DOM, so it is unit-testable in Node; `source` may be a
 * duck-typed { width, height, data: Uint8ClampedArray } in tests.
 */
export function decodeImageData(source: ImageData): DecodeResult {
  throw new Error('not implemented: present/image.ts decodeImageData');
}

/**
 * True when `value` is a DOM image source this module can decode. Used by gl/
 * to dispatch texImage2D arguments: ArrayBufferView/ArrayBuffer → buffer path;
 * isDecodableImageSource → decode path; anything else → INVALID_VALUE.
 */
export function isDecodableImageSource(value: unknown): boolean {
  throw new Error('not implemented: present/image.ts isDecodableImageSource');
}
