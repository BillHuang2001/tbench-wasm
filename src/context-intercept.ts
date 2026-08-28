import { readFileSync, existsSync } from "node:fs";
import assert from "node:assert/strict";

/**
 * Software renderer context interception for browser-based tests.
 *
 * Set WEBGL_SOFTWARE_RENDERER to a JS file path. That file is injected into
 * browser pages before any scripts run and must define:
 *   window.__createSoftwareWebGLContext(canvas, attrs) → WebGLRenderingContext
 *
 * All canvas.getContext('webgl') calls are then routed through it —
 * both HTMLCanvasElement and OffscreenCanvas (when the global exists).
 * Defaults to ./renderer.js — tests fail until the renderer is built.
 *
 * The injected script is worker-aware: the same [bundle + patch] is also
 * imported into every dedicated worker created from a blob: URL (via a
 * launcher that runs importScripts before the worker's own source), so
 * OffscreenCanvas.getContext('webgl'|'webgl2') inside workers runs the
 * software renderer too (Playwright's addInitScript does not reach workers).
 * Nested workers get the same treatment because the patch executes inside
 * each injected worker and patches that realm's Worker. The patch therefore
 * never references `window` at runtime — it resolves globals via globalThis
 * (fallback self), which exist in both page and worker realms.
 *
 * Mechanism (no placeholders, no runtime fixpoint): the injected script is
 * `rendererSource + "\n" + patchIIFE`. The IIFE embeds the renderer source as
 * a JS string literal (`_bundleSrc`) and defines the whole patch as a NAMED
 * function expression (`_swglPatch`). The worker bundle blob is built at
 * runtime as
 *   `_bundleSrc + "\n(" + _swglPatch.toString() + ")(" + JSON.stringify(_bundleSrc) + ");"`
 * — the function serializes itself, so every realm (page, worker, nested
 * worker) executes byte-identical patch code and produces byte-identical
 * blobs.
 */

export function getRendererPath(): string {
  return process.env.WEBGL_SOFTWARE_RENDERER ?? "./renderer.js";
}

/** Throws if the software renderer file doesn't exist. Use in beforeEach. */
export function assertRendererExists(): void {
  const path = getRendererPath();
  assert(
    existsSync(path),
    `Software renderer not found: ${path}. Build the renderer and place it at ${path}.`
  );
}

/**
 * The shared intercept patch as a NAMED function expression: the IIFE embeds
 * it (page realm) and each injected worker re-invokes its serialized form
 * (`_swglPatch.toString()`, see the Worker patch below), so the function must
 * be fully self-contained:
 *  - `_swglRoot` (globalThis, fallback self) instead of `window` — in workers
 *    `window` does not exist and a bare reference would throw;
 *  - the HTMLCanvasElement patch is guarded (workers have no such global);
 *  - the Worker constructor patch wraps only blob-created dedicated workers
 *    (module workers, non-blob URLs and other call shapes pass through
 *    untouched). It is idempotent via the `Worker.__swglPatched` marker set
 *    on the PATCHED constructor.
 */
function buildInterceptIIFE(bundleSource: string): string {
  return `(function() {
  // Full renderer bundle source, embedded as a string literal so blob workers
  // can be pointed at a bundle blob URL without the bundle being served over
  // HTTP. (In the page realm the original call site was
  // window.__createSoftwareWebGLContext(this, attrs, type) — _swglRoot is the
  // same object there (globalThis === window); the _swglRoot form additionally
  // works inside workers, where \`window\` is undefined.)
  var _bundleSrc = ${JSON.stringify(bundleSource)};

  // The whole patch, as a named function expression whose own name the body
  // uses to serialize itself for worker injection (see the Worker patch
  // below). Self-contained: it defines its own _swglRoot/_softwareContexts so
  // it runs identically in the page realm and in every worker realm.
  var _swglPatch = function _swglPatch(_bundleSrc) {
    var _swglRoot = typeof globalThis !== 'undefined' ? globalThis : self;
    var _softwareContexts = new WeakMap();

    // HTMLCanvasElement patch — page realm only (workers have no HTMLCanvasElement).
    if (typeof HTMLCanvasElement !== 'undefined') {
      var _orig = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(type, attrs) {
        if ((type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') &&
            typeof _swglRoot.__createSoftwareWebGLContext === 'function') {
          var gl = _swglRoot.__createSoftwareWebGLContext(this, attrs, type);
          if (gl) _softwareContexts.set(this, gl);
          return gl;
        }
        return _orig.call(this, type, attrs);
      };
    }

    // OffscreenCanvas patch — page AND worker realms.
    if (typeof OffscreenCanvas !== 'undefined') {
      var _origOff = OffscreenCanvas.prototype.getContext;
      OffscreenCanvas.prototype.getContext = function(type, attrs) {
        if ((type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') &&
            typeof _swglRoot.__createSoftwareWebGLContext === 'function') {
          var gl = _swglRoot.__createSoftwareWebGLContext(this, attrs, type);
          if (gl) _softwareContexts.set(this, gl);
          return gl;
        }
        return _origOff.call(this, type, attrs);
      };
      var _origTransfer = OffscreenCanvas.prototype.transferToImageBitmap;
      if (typeof _origTransfer === 'function') {
        OffscreenCanvas.prototype.transferToImageBitmap = function() {
          var gl = _softwareContexts.get(this);
          if (!gl || typeof gl._transferToImageBitmap !== 'function') {
            return _origTransfer.call(this);
          }
          try {
            var snap = gl._transferToImageBitmap();
            if (!snap) return _origTransfer.call(this);
            // Real ImageBitmap, SYNCHRONOUSLY: native 2D OffscreenCanvas +
            // putImageData + native transferToImageBitmap. This is the only sync
            // path (createImageBitmap is async; ImageBitmap is not constructible).
            var native = new OffscreenCanvas(snap.width, snap.height);
            var ctx2d = native.getContext('2d');
            ctx2d.putImageData(new ImageData(snap.data, snap.width, snap.height), 0, 0);
            return _origTransfer.call(native);
          } catch (e) {
            return _origTransfer.call(this);
          }
        };
      }
    }

    // Worker injection: dedicated workers created from blob: URLs load
    // [bundle + this patch] (via a launcher blob) BEFORE their own source, so
    // OffscreenCanvas.getContext inside workers runs the software renderer.
    // Nested workers get the same treatment: this patch runs inside every
    // injected worker and patches that realm's Worker — blob URLs are
    // same-origin across realms, and the __swglPatched marker keeps
    // re-injection idempotent. Module workers, non-blob URLs and other call
    // shapes pass through untouched.
    if (typeof Worker === 'function' && typeof URL !== 'undefined' &&
        typeof URL.createObjectURL === 'function' && typeof Blob !== 'undefined') {
      var _origWorker = Worker;
      var _origCreateObjectURL = URL.createObjectURL.bind(URL);
      if (!Worker.__swglPatched) {
        var _bundleUrl = null;
        Worker = function(url, opts) {
          if (typeof url === 'string' && url.indexOf('blob:') === 0 &&
              (!opts || opts.type !== 'module')) {
            // Lazily create ONE bundle blob URL; never revoke launcher/bundle
            // URLs (the native worker fetches the launcher async — revocation
            // would race). The bundle blob is this same patch re-invoked with
            // the bundle source as its argument, so every realm (page, worker,
            // nested worker) produces identical, deterministic content.
            if (!_bundleUrl) {
              var _workerBundle = _bundleSrc + '\\n(' + _swglPatch.toString() + ')(' + JSON.stringify(_bundleSrc) + ');';
              _bundleUrl = _origCreateObjectURL(new Blob([_workerBundle], {type: 'text/javascript'}));
            }
            var launcher = 'importScripts(' + JSON.stringify(_bundleUrl) + ',' + JSON.stringify(url) + ');';
            var launcherUrl = _origCreateObjectURL(new Blob([launcher], {type: 'text/javascript'}));
            return new _origWorker(launcherUrl, opts);
          }
          return new _origWorker(url, opts);
        };
        // Marker on the PATCHED constructor (the one later IIFE runs see as
        // \`Worker\`), so re-injection never double-wraps.
        Worker.__swglPatched = true;
      }
    }
  };
  _swglPatch(_bundleSrc);
})();`;
}

/**
 * Returns a JS snippet for Playwright's addInitScript() that overrides
 * canvas.getContext('webgl'). If the renderer file doesn't exist, returns
 * a stub that makes every getContext('webgl') call fail with a clear message.
 */
export function buildInterceptScript(): string {
  const path = getRendererPath();

  if (!existsSync(path)) {
    const msg = `Software renderer not found: ${path}. Build the renderer and place it at ${path}.`;
    // Stub variant: worker-safe (no HTMLCanvasElement/document in workers) but
    // deliberately does NOT patch Worker — there is no renderer to inject.
    return `
(function() {
  var _softwareContexts = new WeakMap();
  if (typeof HTMLCanvasElement !== 'undefined') {
    var _orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, attrs) {
      if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') {
        if (typeof document !== 'undefined') document.title = 'RENDERER_NOT_FOUND';
        var gl = null;
        if (gl) _softwareContexts.set(this, gl);
        throw new Error(${JSON.stringify(msg)});
      }
      return _orig.call(this, type, attrs);
    };
  }
  if (typeof OffscreenCanvas !== 'undefined') {
    var _origOff = OffscreenCanvas.prototype.getContext;
    OffscreenCanvas.prototype.getContext = function(type, attrs) {
      if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') {
        if (typeof document !== 'undefined') document.title = 'RENDERER_NOT_FOUND';
        var gl = null;
        if (gl) _softwareContexts.set(this, gl);
        throw new Error(${JSON.stringify(msg)});
      }
      return _origOff.call(this, type, attrs);
    };
    var _origTransfer = OffscreenCanvas.prototype.transferToImageBitmap;
    if (typeof _origTransfer === 'function') {
      OffscreenCanvas.prototype.transferToImageBitmap = function() {
        var gl = _softwareContexts.get(this);
        if (!gl || typeof gl._transferToImageBitmap !== 'function') {
          return _origTransfer.call(this);
        }
        try {
          var snap = gl._transferToImageBitmap();
          if (!snap) return _origTransfer.call(this);
          // Real ImageBitmap, SYNCHRONOUSLY: native 2D OffscreenCanvas +
          // putImageData + native transferToImageBitmap. This is the only sync
          // path (createImageBitmap is async; ImageBitmap is not constructible).
          var native = new OffscreenCanvas(snap.width, snap.height);
          var ctx2d = native.getContext('2d');
          ctx2d.putImageData(new ImageData(snap.data, snap.width, snap.height), 0, 0);
          return _origTransfer.call(native);
        } catch (e) {
          return _origTransfer.call(this);
        }
      };
    }
  }
})();
`;
  }

  const source = readFileSync(path, "utf-8");
  const iife = buildInterceptIIFE(source);
  return `${source}\n${iife}`;
}
