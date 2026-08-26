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
    throw new Error('not implemented: present/canvas.ts BrowserCanvasSurface.getPixels');
  }

  present(): void {
    throw new Error('not implemented: present/canvas.ts BrowserCanvasSurface.present');
  }

  resize(width: number, height: number): void {
    throw new Error('not implemented: present/canvas.ts BrowserCanvasSurface.resize');
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
    throw new Error('not implemented: present/canvas.ts NodeCanvasSurface.getPixels');
  }

  /** Node has no canvas to blit to — intentional no-op. */
  present(): void {
    /* no-op */
  }

  resize(width: number, height: number): void {
    throw new Error('not implemented: present/canvas.ts NodeCanvasSurface.resize');
  }
}

/**
 * Factory: returns a BrowserCanvasSurface when `canvas` is a canvas-like
 * object (has a getContext function — HTMLCanvasElement in browsers, fakes in
 * tests), otherwise a NodeCanvasSurface (pure buffer). Feature-detection is
 * structural, so no DOM globals are touched.
 */
export function createCanvasSurface(canvas: unknown): CanvasSurface {
  throw new Error('not implemented: present/canvas.ts createCanvasSurface');
}
