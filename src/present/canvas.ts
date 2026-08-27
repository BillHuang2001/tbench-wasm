/**
 * canvas.ts — CanvasSurface adapters (browser 2D blit / Node pure buffer).
 *
 * The CanvasSurface is the bridge between the software drawing buffer and the
 * outside world. gl/ renders directly into the surface's RGBA8 buffer
 * (obtained via getPixels(), zero-copy) and calls present() after each
 * draw/clear so the frame is visible to headless-Chromium screenshots,
 * canvas.toDataURL, and drawImage-from-canvas.
 *
 * Presentation semantics (per WebGL spec + CTS canvas tests):
 * - The raster drawing buffer is BOTTOM-UP (GL y-up: buffer row 0 = bottom).
 *   The 2D canvas bitmap is TOP-DOWN, so present() y-flips while blitting.
 * - With premultipliedAlpha:true the buffer holds PREMULTIPLIED values, but
 *   putImageData stores raw bytes and 2D-canvas readback (getImageData,
 *   toDataURL, drawImage) treats them as straight — so present() must
 *   UNPREMULTIPLY before blitting (Chrome-style integer math, clamped).
 * - With alpha:false the presented bitmap is opaque: alpha is forced to 255
 *   AFTER unpremultiplying (the original alpha still governs RGB division).
 * The blit ALWAYS goes through a preallocated scratch buffer — the shared
 * getPixels() buffer is never mutated (gl readPixels aliases it zero-copy).
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
   * Returns the current RGBA8 backing buffer (BOTTOM-UP: row 0 = GL bottom),
   * length = width * height * 4. The returned array is owned by the surface
   * and replaced on resize() — callers must re-fetch after every resize.
   */
  getPixels(): Uint8Array;
  /**
   * Blits the current backing buffer to the canvas (browser) or is a no-op
   * (Node). The blit y-flips (GL bottom-up → canvas top-down) and applies the
   * presentation transform per `opts` (unpremultiply when
   * premultipliedAlpha:true, force alpha 255 when alpha:false); undefined
   * fields fall back to the attrs given at construction, then to the WebGL
   * spec defaults (premultipliedAlpha:true, alpha:true). Must never throw:
   * adapter failures degrade to no-op so the GL error queue is the only error
   * channel to the page.
   */
  present(opts?: PresentOptions): void;
  /**
   * Reallocates the backing buffer for a new size (zero-filled) and
   * invalidates any cached ImageData. Called by gl/ whenever the drawing
   * buffer resizes (canvas.width/height attribute changes, context creation).
   */
  resize(width: number, height: number): void;
}

/**
 * Presentation options for present() — the WebGL context attributes that
 * affect how the drawing buffer is blitted to the 2D canvas. Undefined fields
 * default to the context-creation attrs (see CanvasSurfaceAttrs) and then to
 * the WebGL spec defaults (premultipliedAlpha:true, alpha:true).
 */
export interface PresentOptions {
  /** Buffer holds premultiplied colors; blit must unpremultiply. */
  premultipliedAlpha?: boolean;
  /** Buffer has an alpha channel; the presented bitmap is opaque when false. */
  alpha?: boolean;
}

/**
 * Context-creation attributes the present layer consumes (a subset of
 * WebGLContextAttributes). Passed to createCanvasSurface so present() without
 * opts can honor the context's actual attributes. All fields optional.
 */
export interface CanvasSurfaceAttrs {
  premultipliedAlpha?: boolean;
  alpha?: boolean;
}

/** WebGL spec defaults for the presentation options. */
const DEFAULT_PREMULTIPLIED_ALPHA = true;
const DEFAULT_ALPHA = true;

/**
 * Browser adapter: blits the software buffer to an HTMLCanvasElement via its
 * native 2D context (putImageData). The 2D context is obtained lazily on the
 * first present(); if it is unavailable or getContext('2d') returns null
 * (canvas already has a native WebGL/other context), present() degrades to a
 * no-op while the buffer stays fully functional.
 *
 * The blit is built into a PREALLOCATED scratch buffer (allocated in resize(),
 * reused across presents — zero per-frame allocation): the source buffer
 * (aliased zero-copy by gl/) is only ever READ, never mutated.
 */
export class BrowserCanvasSurface implements CanvasSurface {
  /** Drawing buffer state — owned by the surface, replaced on resize(). */
  private _width = 0;
  private _height = 0;
  /** Raw BOTTOM-UP RGBA8 buffer; gl/ renders into it zero-copy. */
  private _pixels = new Uint8Array(0);
  /** TOP-DOWN presented copy (flipped/unpremultiplied), built per present(). */
  private _scratch = new Uint8Array(0);

  /** Lazily acquired native 2D context; null when unavailable. */
  private _ctx2d: CanvasRenderingContext2D | null = null;
  /** Cached ImageData wrapping the scratch buffer, reused across presents. */
  private _scratchImageData: ImageData | null = null;
  /** Set once 2D acquisition failed, to avoid retrying on every present(). */
  private _ctxFailed = false;

  /** Default presentation attrs from context creation (undefined = spec default). */
  private _premulDefault: boolean | undefined;
  private _alphaDefault: boolean | undefined;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    attrs?: CanvasSurfaceAttrs
  ) {
    this._premulDefault = attrs?.premultipliedAlpha;
    this._alphaDefault = attrs?.alpha;
  }

  get width(): number {
    return this._width;
  }

  get height(): number {
    return this._height;
  }

  getPixels(): Uint8Array {
    return this._pixels;
  }

  present(opts?: PresentOptions): void {
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
      if (this._scratchImageData === null) {
        return;
      }
      // Undefined opts fall back to the creation attrs, then spec defaults.
      const premul = opts?.premultipliedAlpha ?? this._premulDefault ?? DEFAULT_PREMULTIPLIED_ALPHA;
      const alpha = opts?.alpha ?? this._alphaDefault ?? DEFAULT_ALPHA;
      this.buildPresentedCopy(premul, alpha);
      this._ctx2d.putImageData(this._scratchImageData, 0, 0);
    } catch {
      // present() never throws — adapter failures degrade to a silent no-op
      // (the GL error queue is the only error channel to the page).
    }
  }

  resize(width: number, height: number): void {
    this._width = width;
    this._height = height;
    this._pixels = new Uint8Array(width * height * 4);
    this._scratch = new Uint8Array(width * height * 4);
    this._scratchImageData = null;
    // ImageData is a DOM global — absent in Node. Guard by feature detection;
    // if unavailable (or the constructor throws), present() silently no-ops.
    if (typeof ImageData === 'undefined') {
      return;
    }
    try {
      // View over the scratch buffer: the ImageData constructor does not copy
      // (spec: "does not set this's data to a copy"), so mutating _scratch
      // before each putImageData is reflected at blit time. putImageData
      // copies only at blit time.
      const view = new Uint8ClampedArray(
        this._scratch.buffer,
        this._scratch.byteOffset,
        this._scratch.byteLength
      );
      this._scratchImageData = new ImageData(view, width, height);
    } catch {
      this._scratchImageData = null;
    }
  }

  /**
   * Build the presented (TOP-DOWN) copy into _scratch from the raw BOTTOM-UP
   * _pixels: y-flip every row, and per pixel:
   * - premultipliedAlpha → unpremultiply (c' = (c*255 + a/2) / a, Chrome-style
   *   integer math, clamped to 255; a=0 leaves RGB untouched);
   * - alpha:false → force alpha to 255 AFTER the unpremultiply.
   * Never touches _pixels (gl/ readPixels aliases it zero-copy).
   */
  private buildPresentedCopy(premultipliedAlpha: boolean, alpha: boolean): void {
    buildPresentedPixels(
      this._pixels,
      this._scratch,
      this._width,
      this._height,
      premultipliedAlpha,
      alpha
    );
  }
}

/**
 * Build the presented (TOP-DOWN) copy of the drawing buffer into `dst` from
 * the raw BOTTOM-UP `src` (both RGBA8, dst = w*h*4): y-flip every row, and per
 * pixel:
 * - premultipliedAlpha → unpremultiply (c' = (c*255 + a/2) / a, Chrome-style
 *   integer math, clamped to 255; a=0 leaves RGB untouched);
 * - alpha:false → force alpha to 255 AFTER the unpremultiply.
 * Pure function — never mutates `src` (gl/ readPixels aliases it zero-copy).
 * Exported for reuse by gl/ (OffscreenCanvas.transferToImageBitmap snapshots).
 */
export function buildPresentedPixels(
  src: Uint8Array,
  dst: Uint8Array,
  w: number,
  h: number,
  premultipliedAlpha: boolean,
  alpha: boolean
): void {
  if (w === 0 || h === 0) {
    return;
  }
  const rowBytes = w * 4;
  if (!premultipliedAlpha && alpha) {
    // Fast path: pure vertical flip, straight copy.
    for (let i = 0; i < h; i++) {
      const s = (h - 1 - i) * rowBytes;
      const d = i * rowBytes;
      for (let k = 0; k < rowBytes; k++) {
        dst[d + k] = src[s + k];
      }
    }
    return;
  }
  for (let i = 0; i < h; i++) {
    const s = (h - 1 - i) * rowBytes;
    const d = i * rowBytes;
    for (let j = 0; j < w; j++) {
      const si = s + j * 4;
      const di = d + j * 4;
      let r = src[si];
      let g = src[si + 1];
      let b = src[si + 2];
      const a = src[si + 3];
      if (premultipliedAlpha && a > 0) {
        r = ((r * 255 + (a >> 1)) / a) | 0;
        g = ((g * 255 + (a >> 1)) / a) | 0;
        b = ((b * 255 + (a >> 1)) / a) | 0;
        // Clamp (Chrome-style): guards non-premultiplied data, where c > a
        // would otherwise overflow past 255.
        if (r > 255) r = 255;
        if (g > 255) g = 255;
        if (b > 255) b = 255;
      }
      dst[di] = r;
      dst[di + 1] = g;
      dst[di + 2] = b;
      dst[di + 3] = alpha ? a : 255;
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
  present(_opts?: PresentOptions): void {
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
 *
 * `attrs` (optional) carries the WebGL context-creation attributes the
 * presentation transform honors (premultipliedAlpha, alpha); present() calls
 * without opts fall back to them, then to the spec defaults.
 */
export function createCanvasSurface(canvas: unknown, attrs?: CanvasSurfaceAttrs): CanvasSurface {
  if (
    canvas !== null &&
    typeof canvas === 'object' &&
    typeof (canvas as { getContext?: unknown }).getContext === 'function'
  ) {
    return new BrowserCanvasSurface(canvas as HTMLCanvasElement, attrs);
  }
  return new NodeCanvasSurface();
}
