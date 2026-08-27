/**
 * canvas.ts — CanvasSurface adapters (browser 2D blit / Node pure buffer).
 *
 * The CanvasSurface is the bridge between the software drawing buffer and the
 * outside world. gl/ renders directly into the surface's RGBA8 buffer
 * (obtained via getPixels(), zero-copy) and calls present() after each
 * draw/clear so the frame is visible to headless-Chromium screenshots,
 * canvas.toDataURL, and drawImage-from-canvas.
 *
 * No DOM APIs are touched at module load; the 2D context is acquired lazily
 * and failures degrade to a buffer-only no-op. See ../CONTEXT.md contract §4.
 */

/** RGBA8 pixel surface presented to a canvas (or held as a pure buffer in Node). */
export interface CanvasSurface {
  /** Current surface width in pixels. */
  readonly width: number;
  /** Current surface height in pixels. */
  readonly height: number;
  /**
   * Returns the current RGBA8 (straight alpha) backing buffer,
   * length = width * height * 4. The returned array is owned by the surface
   * and replaced on resize() — callers must re-fetch after every resize.
   */
  getPixels(): Uint8Array;
  /**
   * Blits the current backing buffer to the canvas (browser) or is a no-op
   * (Node). Must never throw: adapter failures degrade to no-op so the GL
   * error queue is the only error channel to the page.
   */
  present(): void;
  /**
   * Reallocates the backing buffer for a new size (zero-filled) and
   * invalidates any cached ImageData. Called by gl/ whenever the drawing
   * buffer resizes (canvas.width/height attribute changes, context creation).
   */
  resize(width: number, height: number): void;
}

/**
 * Browser adapter: blits the software buffer to an HTMLCanvasElement via its
 * native 2D context (putImageData). The 2D context is obtained lazily on the
 * first present(); if it is unavailable or getContext('2d') returns null
 * (canvas already has a native WebGL/other context), present() degrades to a
 * no-op while the buffer stays fully functional.
 */
export class BrowserCanvasSurface implements CanvasSurface {
  /** Drawing buffer state — owned by the surface, replaced on resize(). */
  private _width = 0;
  private _height = 0;
  private _pixels = new Uint8Array(0);

  /** Lazily acquired native 2D context; null when unavailable. */
  private _ctx2d: CanvasRenderingContext2D | null = null;
  /** Cached ImageData wrapping the current buffer, reused across presents. */
  private _imageData: ImageData | null = null;
  /** Set once 2D acquisition failed, to avoid retrying on every present(). */
  private _ctxFailed = false;

  constructor(private readonly canvas: HTMLCanvasElement) {}

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  getPixels(): Uint8Array {
    return this._pixels;
  }

  present(): void {
    try {
      // Lazily acquire the native 2D context on the first present. A null or
      // throwing getContext means the canvas already has a non-2D context (or
      // no 2D support) — record the failure permanently and degrade to no-op.
      if (this._ctx2d === null && !this._ctxFailed) {
        let ctx: CanvasRenderingContext2D | null = null;
        try {
          ctx = this.canvas.getContext('2d') as CanvasRenderingContext2D | null;
        } catch {
          ctx = null;
        }
        if (ctx === null) {
          this._ctxFailed = true;
        } else {
          this._ctx2d = ctx;
        }
      }
      if (this._ctx2d === null) {
        return;
      }
      // Auto-resize safety net: the page may have resized the canvas bitmap
      // between draws; match it so putImageData never clips or misaligns.
      if (this.canvas.width !== this._width || this.canvas.height !== this._height) {
        this.resize(this.canvas.width, this.canvas.height);
      }
      if (this._imageData === null) {
        return;
      }
      this._ctx2d.putImageData(this._imageData, 0, 0);
    } catch {
      // present() never throws — adapter failures degrade to a silent no-op
      // (the GL error queue is the only error channel to the page).
    }
  }

  resize(width: number, height: number): void {
    this._width = width;
    this._height = height;
    this._pixels = new Uint8Array(width * height * 4);
    this._imageData = null;
    // ImageData is a DOM global — absent in Node. Guard by feature detection;
    // if unavailable (or the constructor throws), present() silently no-ops.
    if (typeof ImageData === 'undefined') {
      return;
    }
    try {
      // View over the SAME memory: the ImageData constructor does not copy
      // (spec: "does not set this's data to a copy"), so the cached ImageData
      // always reflects the current buffer contents; putImageData copies at
      // blit time only.
      const view = new Uint8ClampedArray(
        this._pixels.buffer,
        this._pixels.byteOffset,
        this._pixels.byteLength
      );
      this._imageData = new ImageData(view, width, height);
    } catch {
      this._imageData = null;
    }
  }
}

/** Node adapter: pure buffer; present() is a no-op. Used in Node and unit tests. */
export class NodeCanvasSurface implements CanvasSurface {
  private _width = 0;
  private _height = 0;
  private _pixels = new Uint8Array(0);

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  getPixels(): Uint8Array {
    return this._pixels;
  }

  /** Node has no canvas to blit to — intentional no-op. */
  present(): void {
    /* no-op */
  }

  resize(width: number, height: number): void {
    this._width = width;
    this._height = height;
    this._pixels = new Uint8Array(width * height * 4);
  }
}

/**
 * Factory: returns a BrowserCanvasSurface when `canvas` is a canvas-like
 * object (has a getContext function — HTMLCanvasElement in browsers, fakes in
 * tests), otherwise a NodeCanvasSurface (pure buffer). Feature-detection is
 * structural, so no DOM globals are touched.
 */
export function createCanvasSurface(canvas: unknown): CanvasSurface {
  if (
    canvas !== null &&
    typeof canvas === 'object' &&
    typeof (canvas as { getContext?: unknown }).getContext === 'function'
  ) {
    return new BrowserCanvasSurface(canvas as HTMLCanvasElement);
  }
  return new NodeCanvasSurface();
}
