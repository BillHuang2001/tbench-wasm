/**
 * src/gl/draw.ts — the DRAW PIPELINE ENGINE (internal; api/draw.ts delegates here).
 *
 * Pipeline per draw call (contract §2 with raster/):
 *  1. Cheap preconditions (program linked, enabled attribs have buffers,
 *     indexed bounds — the full validation lives in api/draw.ts).
 *  2. Attribute fetch: per-draw DENSE extraction (glsl contract, see
 *     src/glsl/program.ts VertexExecCtx): each enabled attribute array is
 *     gathered into a packed per-draw scratch array (stride removed, format
 *     conversion applied), `attribs[loc]`, laid out with ONE element per
 *     fetched vertex at stride `program components`, padded (0,0,0,1).
 *     `attribIndices[loc]` is set per vertex: divisor-0 → vertex position
 *     within the draw; instanced → floor(instanceId/divisor); constant → 0.
 *  3. Vertex evaluation: for each instance × vertex, set ctx.vertexId
 *     (gl_VertexID: first+j or the element value) + ctx.instanceId, run
 *     program.vertex.run(ctx), pack the output record
 *     [x,y,z,w, pointSize, clipDistance[8], cullDistance[8], varyings...]
 *     (header = 21 floats: 0-3 position, 4 pointSize, 5-12 clipDistance,
 *     13-20 cullDistance, varyings from 21 — see RECORD_TOTAL_HEADER_FLOATS)
 *     at record index i*count + j
 *     (raster DrawCall instanced addressing: instance i, vertex j at
 *     first + i*count + j; we pack with first = 0).
 *  4. Transform feedback capture (when active && !paused): write the captured
 *     varyings of each processed primitive into the bound TF buffers
 *     (INTERLEAVED_ATTRIBS / SEPARATE_ATTRIBS) from the packed records,
 *     count fully-captured primitives; the rasterizer is bypassed.
 *  5. Occlusion queries: when ANY_SAMPLES_PASSED(*) is active, a
 *     `sampleCountRef` counter is attached to the DrawCall (raster increments
 *     it per passing sample — see "occlusion hook" note below) and summed into
 *     the active query after the draw.
 *  6. Rasterize: build the DrawCall (contract §2) and call rasterizer.draw().
 *
 * All engine functions take the context and mutate only context-owned state.
 *
 * CROSS-AGENT CONTRACT NOTES (stub era):
 *  - rasterizer.draw / clear/blit helpers are STUBS that throw. The engine
 *    calls them inside try/catch and falls back to LOCAL implementations
 *    (direct typed-array fills/copies/packs, "replace when raster lands").
 *    Raster throws are SWALLOWED (no GL error) so the stub era stays
 *    functional; when raster lands, remove the swallow.
 *  - framebuffer-util is a stub: resolveFramebufferTarget/resolveReadSurface/
 *    getAttachmentSurface throw. The engine falls back to resolving the
 *    DEFAULT framebuffer itself (ctx._defaultFB, or a lazily allocated
 *    fallback surface when even lifecycle hasn't landed); FBO targets are
 *    unresolvable until framebuffer-util lands (draws to FBOs then raise
 *    INVALID_FRAMEBUFFER_OPERATION).
 *  - Occlusion hook: raster's DrawCall type has no sample-count field. The
 *    engine attaches `(dc as any).sampleCountRef = { value: 0 }`; the raster
 *    agent must increment `dc.sampleCountRef.value` for every fragment that
 *    passes the depth test (and stencil test, per GLES3 occlusion semantics).
 *    Summed into WebGLQuery._result after the draw; _resultAvailable = true.
 *  - Instanced attributes: glsl codegen knows nothing about divisors (GL
 *    state). The dense extraction for divisor>0 attribs contains
 *    ceil(instanceCount/divisor) elements (buffer element k = k, NO first/
 *    index offset) and attribIndices[loc] = floor(instanceId/divisor) —
 *    exactly the GLES3 instanced fetch rule.
 *  - Transform feedback overflow: capture is terminated at the first write
 *    that would exceed a bound range; no GL error (spec: undefined), only
 *    fully-captured primitives are counted in _primitivesWritten.
 *  - readPixels/blit row order: raster surfaces store row 0 = BOTTOM (GL
 *    convention, src/raster/types.ts) INCLUDING the default framebuffer (the
 *    present() path flips for display). readPixels therefore does NOT flip.
 */

import type { WebGLRenderingContext } from './webgl1';
import type { GLenum, GLint, GLintptr, GLsizei, GLuint } from './types';
import { C1, C2 } from './constants';
import { resolveFramebufferTarget, resolveReadSurface, getAttachmentSurface } from './framebuffer-util';
import { handleCanvasResize } from './lost';
import { getClipControl } from './extensions/clip-state';
import { updateCompleteness, floatLinearExtensionState } from './teximage';
import {
  draw as rasterDraw,
  clearColorSurface,
  clearDepthSurface,
  clearStencilSurface,
  blitColorSurface,
  blitDepthStencilSurface,
  createTextureEnv,
  getPackConverter,
  getFormat,
  floatToHalf,
  linearToSRGB,
} from '../raster';
import type {
  DrawCall, FramebufferTarget, SamplerState, Surface, TextureImage, TextureUnitBinding, TextureEnv,
  ColorMask, ScissorState,
} from '../raster';
import type { VertexExecCtx, AttribSource } from '../glsl/program';
import { buildPresentedPixels } from '../present/canvas';
import type { WebGLProgram } from './objects';
import type { ProgramModel } from './objects';
import type { WebGLBuffer, WebGLQuery, WebGLTexture, WebGLTransformFeedback } from './objects';
import { ensureProgramLinked } from './api/programs';
import { SRC1_BLEND_FACTORS } from './api/state';

/** Clip/cull distance record-slot contract (CROSS-AGENT, see src/gl/CONTEXT.md):
 *  header = [x,y,z,w (0-3), pointSize (4), clipDistance[8] (5-12),
 *  cullDistance[8] (13-20)] → varyings start at 21.
 *  The raster side (src/raster/types.ts RECORD_HEADER_FLOATS/VARYINGS_OFFSET)
 *  is updated to 21 in parallel against this contract. Do NOT size the header
 *  from the imported RECORD_HEADER_FLOATS (5 at HEAD). */
const RECORD_OFFSET_CLIP_DISTANCE = 5;
const RECORD_OFFSET_CULL_DISTANCE = 13;
const RECORD_TOTAL_HEADER_FLOATS = 21; // 4 pos + 1 pointSize + 8 clip + 8 cull

/**
 * WebIDL `unsigned long long` conversion (readPixels dstOffset, clearBuffer*
 * srcOffset — the wasm readpixels CTS pages pass offsets > 2^32, so NO
 * `>>> 0`). NaN/Infinity → 0; negatives wrap modulo 2^64 (a huge value that
 * fails every bounds check, per WebIDL semantics). Shared by api/draw.ts and
 * api/webgl2.ts.
 */
export function toU64(v: unknown): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return n < 0 ? n + 2 ** 64 : n;
}

/** A fully validated, assembled draw request (before rasterizer call). */
export interface DrawRequest {
  mode: GLenum;
  count: GLsizei;
  instanceCount: GLsizei;
  /** For non-indexed: first vertex. For indexed: offset into index buffer (bytes). */
  firstOrOffset: GLint | GLintptr;
  indexed: boolean;
  /** For indexed: UNSIGNED_BYTE | UNSIGNED_SHORT | UNSIGNED_INT. */
  indexType?: GLenum;
  /** For drawRangeElements: [start, end] inclusive index range. */
  range?: [GLuint, GLuint];
  /**
   * gl_DrawID (WEBGL_multi_draw): the 0-based multi-draw subdraw index,
   * constant for every vertex/instance of THIS draw. Omitted (or 0) for
   * single draws. The vertex exec ctx reads it as `gl_DrawID`.
   */
  drawId?: number;
  /**
   * WEBGL_draw_instanced_base_vertex_base_instance /
   * WEBGL_multi_draw_instanced_base_vertex_base_instance:
   *  - baseVertex (GLint, indexed draws only): added to every element index
   *    for attribute fetch AND to gl_VertexID. Per GLES 3.2 §10.5 the vertex
   *    ID of the ith element of an indexed draw is basevertex + elementIndex.
   *  - baseInstance (GLuint): added to the divisor-based instance attribute
   *    fetch index (element = baseInstance + floor(instanceId/divisor)).
   *    gl_InstanceID is NOT offset (always starts at 0).
   * Omitted (0) for regular draws.
   */
  baseVertex?: number;
  baseInstance?: number;
}

/* ================================================================== */
/* Per-context scratch (preallocated, reused across draws)             */
/* ================================================================== */

interface DrawScratch {
  /** Packed vertex records (grows on demand). */
  records: Float32Array;
  /** Dense attribute extraction pool (float path). */
  floatPool: Float32Array;
  /** Dense attribute extraction pool (signed integer path). */
  intPool: Int32Array;
  /** Dense attribute extraction pool (unsigned integer path). */
  uintPool: Uint32Array;
  /** Per-vertex fetch indices (length = MAX_VERTEX_ATTRIBS). */
  attribIndices: Int32Array;
  /** Vertex exec ctx reusable buffers. */
  outPosition: Float32Array;
  outVaryings: Float32Array;
  /** gl_ClipDistance / gl_CullDistance outputs (8 floats each, zeroed per draw). */
  outClipDistance: Float32Array;
  outCullDistance: Float32Array;
  scratch: Float32Array;   // codegen float scratch (program.scratchSize)
  intScratch: Int32Array;  // codegen int scratch
  zeroStore: Float32Array; // fallback uniform-block store (zero-filled)
  zeroIntStore: Int32Array;
  /** Stub-era fallback default-framebuffer color surface (until lifecycle lands). */
  fallbackColor: Surface | null;
  /** preserveDrawingBuffer:false — one pending frame-boundary clear per context. */
  preserveClearPending: boolean;
  /** Float drawing-buffer → RGBA8 present conversion scratch (4 floats). */
  presentTmp: Float32Array;
  /** Stub-era: float/int stores when a fake ProgramModel has none. */
  emptyFloat: Float32Array;
  emptyInt: Int32Array;
}

const scratchMap = new WeakMap<WebGLRenderingContext, DrawScratch>();

function getScratch(ctx: WebGLRenderingContext): DrawScratch {
  let sc = scratchMap.get(ctx);
  if (!sc) {
    sc = {
      records: new Float32Array(0),
      floatPool: new Float32Array(0),
      intPool: new Int32Array(0),
      uintPool: new Uint32Array(0),
      attribIndices: new Int32Array(0),
      outPosition: new Float32Array(4),
      outVaryings: new Float32Array(0),
      outClipDistance: new Float32Array(8),
      outCullDistance: new Float32Array(8),
      scratch: new Float32Array(64),
      intScratch: new Int32Array(64),
      zeroStore: new Float32Array(16384), // MAX_UNIFORM_BLOCK_SIZE / 4
      zeroIntStore: new Int32Array(16384),
      fallbackColor: null,
      preserveClearPending: false,
      presentTmp: new Float32Array(4),
      emptyFloat: new Float32Array(0),
      emptyInt: new Int32Array(0),
    };
    scratchMap.set(ctx, sc);
  }
  return sc;
}

/* ================================================================== */
/* Small helpers                                                       */
/* ================================================================== */

function pushError(ctx: WebGLRenderingContext, err: GLenum): void {
  ctx._errors.push(err);
}

function alignUp(n: number, align: number): number {
  return (((n + align - 1) / align) | 0) * align;
}

/**
 * Extension gate that never throws. Checks the enabled-extension cache
 * (`ctx._extensions`, populated by getExtension) instead of CALLING
 * getExtension — calling it here would cache and thereby SELF-ENABLE the
 * extension, observable as e.g. UNSIGNED_INT indices accepted without
 * OES_element_index_uint ever being requested.
 */
export function extSupported(ctx: WebGLRenderingContext, name: string): boolean {
  try {
    return ctx._extensions.has(name);
  } catch {
    return false;
  }
}

/** Stub-era fallback default-FB color surface (row 0 = BOTTOM, RGBA8). */
function defaultFBSurface(ctx: WebGLRenderingContext): Surface {
  const sc = getScratch(ctx);
  if (!sc.fallbackColor) {
    const w = Math.max(1, ctx._drawingBufferWidth | 0);
    const h = Math.max(1, ctx._drawingBufferHeight | 0);
    sc.fallbackColor = {
      width: w,
      height: h,
      format: C1.RGBA8,
      info: null as never,
      data: new Uint8Array(w * h * 4),
    };
  }
  return sc.fallbackColor;
}

/**
 * Resolve the current DRAW target. framebuffer-util is the single source once
 * it lands; until then the default framebuffer is resolved locally (stub-era
 * fallback surface if lifecycle hasn't allocated _defaultFB).
 */
function resolveDrawTarget(ctx: WebGLRenderingContext): FramebufferTarget | null {
  const s = ctx._state;
  if (s.drawFramebuffer === null) {
    const dfb = ctx._defaultFB;
    if (dfb) {
      return { color: [dfb.color], depth: dfb.depth, stencil: dfb.stencil, width: dfb.width, height: dfb.height, samples: 1 };
    }
    const surf = defaultFBSurface(ctx);
    return { color: [surf], depth: null, stencil: null, width: surf.width, height: surf.height, samples: 1 };
  }
  try {
    return resolveFramebufferTarget(ctx);
  } catch {
    return null; // framebuffer-util stub era: FBOs unresolvable
  }
}

/** Resolve the READ color surface (default FB locally; FBO via framebuffer-util). */
function resolveReadColor(ctx: WebGLRenderingContext): Surface | null {
  const s = ctx._state;
  if (s.readFramebuffer === null) {
    return ctx._defaultFB ? ctx._defaultFB.color : defaultFBSurface(ctx);
  }
  try {
    return resolveReadSurface(ctx);
  } catch {
    return null;
  }
}

/** Resolve the READ depth/stencil surface (default FB locally; FBO via framebuffer-util). */
function resolveReadDepthStencil(ctx: WebGLRenderingContext, attachment: GLenum): Surface | null {
  const s = ctx._state;
  if (s.readFramebuffer === null) {
    const dfb = ctx._defaultFB;
    if (!dfb) return null;
    return attachment === C1.DEPTH_ATTACHMENT ? dfb.depth : dfb.stencil;
  }
  try {
    return getAttachmentSurface(s.readFramebuffer, attachment);
  } catch {
    return null;
  }
}

/**
 * Does the READ framebuffer's attachment point hold an image? (GLES 3.0 §4.3.2
 * blit missing-image validation.) Uses the RAW attachment record, NOT
 * getAttachmentSurface's plane-scoped views: a depth-stencil image attached at
 * DEPTH_ATTACHMENT resolves to null via getAttachmentSurface(DEPTH_ATTACHMENT)
 * (deliberate — stencil-only blits must not copy depth) even though the point
 * HAS an image (blitframebuffer-test.html binds the same DEPTH24_STENCIL8
 * renderbuffer at DEPTH_ATTACHMENT on both sides and expects a DEPTH blit to
 * succeed).
 */
function readAttachmentHasImage(ctx: WebGLRenderingContext, attachment: GLenum): boolean {
  const s = ctx._state;
  const fbo = s.readFramebuffer;
  if (fbo === null) {
    const dfb = ctx._defaultFB;
    if (!dfb) return false;
    return attachment === C1.DEPTH_ATTACHMENT ? !!dfb.depth : !!dfb.stencil;
  }
  const entry = fbo._attachments.get(attachment);
  if (!entry) return false;
  try {
    if (entry.type === 'renderbuffer') return !!entry.renderbuffer._surface;
    const tex = entry.texture;
    const img = tex._image;
    if (!img || entry.level < 0) return false;
    const lvl = img.levels[entry.level];
    if (!lvl) return false;
    const isCube = entry.face !== undefined && entry.face >= 0x8515 && entry.face <= 0x851a;
    const idx = isCube ? entry.face - 0x8515 : entry.layer;
    return !!lvl.data[idx];
  } catch {
    return false;
  }
}

/** Present the drawing buffer when the draw/clear/blit touched the default FB. */
function presentIfDefault(ctx: WebGLRenderingContext): void {
  if (ctx._state.drawFramebuffer !== null) return;
  if (!ctx._presentSurface) return;
  try {
    // A non-RGBA8 drawing buffer (drawingBufferStorage RGBA16F) does NOT alias
    // the present surface's RGBA8 pixel buffer — convert the float store into
    // it (clamped, rounded — the same raw bytes the RGBA8 path would hold) so
    // present() blits the frame (CTS drawingbuffer-storage-test "Drawing
    // transparent RGBA16F to canvas"). Only the color surface needs this; the
    // present transform (unpremultiply + y-flip) runs inside present().
    const dfb = ctx._defaultFB;
    const color = dfb ? dfb.color : null;
    if (color && !(color.data instanceof Uint8Array)) {
      const w = color.width;
      const h = color.height;
      const pixels = ctx._presentSurface.getPixels();
      if (pixels && pixels.length >= w * h * 4) {
        const bpp = surfaceBytesPerPixel(color);
        const tmp = getScratch(ctx).presentTmp;
        for (let y = 0; y < h; y++) {
          const row = y * w;
          for (let x = 0; x < w; x++) {
            decodeSurfaceTexel(color, (row + x) * bpp, tmp);
            const o = (row + x) * 4;
            for (let c = 0; c < 4; c++) {
              const v = tmp[c];
              pixels[o + c] = (v < 0 ? 0 : v > 1 ? 1 : v) * 255 + 0.5 | 0;
            }
          }
        }
      }
    }
    ctx._presentSurface.present();
  } catch {
    // present adapter stub era — never leak to the page
  }
  // preserveDrawingBuffer:false: the drawing buffer must be cleared back to its
  // initial state AFTER compositing — at the next frame boundary, never
  // synchronously (CTS draws → readPixels in the same task and must see the
  // frame; wtu.waitForComposite = 5 rAFs).
  if (ctx._attrs.preserveDrawingBuffer === false) {
    schedulePreserveClear(ctx);
  }
}

/**
 * Frame-boundary clear for preserveDrawingBuffer:false. Coalesced to ONE
 * pending callback per context (draws between schedule and callback just keep
 * the flag). The callback clears the default framebuffer only — the clear must
 * NOT respect scissor/colorMask (spec: the buffer returns to its initial
 * state; CTS buffer-preserve-test enables a scissor before compositing to
 * prove it is ignored). Browser: requestAnimationFrame (runs before the next
 * composite — well within wtu.waitForComposite's 5 frames); Node/headless:
 * setImmediate → microtask → setTimeout (same-task draw→readPixels sequences
 * survive). A canvas NOT connected to the document is never composited, so
 * preserve:false must NOT clear it (CTS buffer-offscreen-test's detached gl2
 * canvas keeps its content). OffscreenCanvas has no isConnected member
 * (undefined) and its content is consumed by the page (texImage2D sources,
 * transferToImageBitmap) rather than composited — treat it as detached too,
 * otherwise the frame-boundary clear wipes source canvases to opaque black
 * (CTS webgl_canvas/tex-2d-rgba-rgba-unsigned_byte.html alpha:false subtests).
 */
function schedulePreserveClear(ctx: WebGLRenderingContext): void {
  const canvas = ctx._canvas as { isConnected?: boolean };
  if (canvas.isConnected === false || canvas.isConnected === undefined) return;
  const sc = getScratch(ctx);
  if (sc.preserveClearPending) return;
  sc.preserveClearPending = true;
  const run = (): void => {
    sc.preserveClearPending = false;
    if (ctx._isLost) return;
    const dfb = ctx._defaultFB;
    if (!dfb) return;
    clearDefaultFramebufferForPreserve(ctx, dfb);
    // Refresh the canvas bitmap with the cleared buffer: putImageData is a
    // SNAPSHOT, and the browser's compositor would show the cleared drawing
    // buffer on the next frame — emulate that composite so drawImage/toDataURL
    // after the frame boundary see the cleared state (CTS
    // context-attribute-preserve-drawing-buffer.html). No-op adapters (Node)
    // make this harmless.
    try {
      ctx._presentSurface?.present();
    } catch {
      // never leak to the page
    }
  };
  const g = globalThis as {
    requestAnimationFrame?: (f: () => void) => number;
    setImmediate?: (f: () => void) => unknown;
    queueMicrotask?: (f: () => void) => void;
    setTimeout?: (f: () => void, ms?: number) => unknown;
  };
  if (typeof g.requestAnimationFrame === 'function') {
    g.requestAnimationFrame(run);
  } else if (typeof g.setImmediate === 'function') {
    g.setImmediate(run);
  } else if (typeof g.queueMicrotask === 'function') {
    g.queueMicrotask(run);
  } else if (typeof g.setTimeout === 'function') {
    g.setTimeout(run, 0);
  } else {
    run(); // pure sandbox without async primitives — clear synchronously
  }
}

/**
 * Clear the default framebuffer to its initial state (color (0,0,0,0), depth
 * 1.0, stencil 0) at the frame boundary. alpha:false drawing buffers have NO
 * alpha channel — the stored alpha byte must be 255 so the composite (raw
 * putImageData) and later reads see an opaque buffer (CTS
 * context-attribute-preserve-drawing-buffer.html expects (0,0,0,255)).
 */
function clearDefaultFramebufferForPreserve(ctx: WebGLRenderingContext, fb: { color: Surface; depth: Surface | null; stencil: Surface | null }): void {
  const d = fb.color.data;
  if (d instanceof Uint8Array) {
    d.fill(0);
    if (ctx._attrs.alpha === false) {
      for (let i = 3; i < d.length; i += 4) d[i] = 255;
    }
  }
  if (fb.depth && fb.depth.data instanceof Float32Array) fb.depth.data.fill(1.0);
  if (fb.stencil && fb.stencil.data instanceof Uint8Array) fb.stencil.data.fill(0);
}

/**
 * OffscreenCanvas.transferToImageBitmap() snapshot: builds the TOP-DOWN,
 * unpremultiplied RGBA8 copy of the drawing buffer (the same presentation
 * transform as present() — y-flip + unpremultiply + alpha handling via
 * buildPresentedPixels), then UNCONDITIONALLY resets the drawing buffer to its
 * initial state (color (0,0,0,0), depth 1.0, stencil 0): per spec the transfer
 * takes the canvas bitmap out of the OffscreenCanvas regardless of
 * preserveDrawingBuffer. The reset is immediate (not scheduled at a frame
 * boundary). Returns null when there is no drawing buffer (context lost / not
 * yet allocated).
 */
export function transferToImageBitmapSnapshot(ctx: WebGLRenderingContext):
    { width: number; height: number; data: Uint8ClampedArray } | null {
  const dfb = ctx._defaultFB;
  if (!dfb || !dfb.color) return null; // context lost / no drawing buffer
  const w = dfb.color.width, h = dfb.color.height;
  const data = new Uint8Array(w * h * 4);
  const src = dfb.color.data; // BOTTOM-UP RGBA8
  if (!(src instanceof Uint8Array)) return null; // default FB is always RGBA8
  // Premultiplied when the context was created with premultipliedAlpha (spec
  // default true); alpha:false buffers are forced opaque after unpremultiply.
  buildPresentedPixels(src, data, w, h,
    ctx._attrs.premultipliedAlpha !== false, ctx._attrs.alpha !== false);
  // transferToImageBitmap UNCONDITIONALLY clears the canvas bitmap (spec):
  clearDefaultFramebufferForPreserve(ctx, dfb);
  // Refresh the canvas bitmap with the cleared buffer (no-op adapters, e.g.
  // Node, make this harmless) — never leak adapter failures to the page.
  try {
    ctx._presentSurface?.present();
  } catch {
    // never leak to the page
  }
  // Uint8ClampedArray view over the same memory (no copy) — ImageBitmap data.
  return { width: w, height: h, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength) };
}

/** Canvas resize guard: keep the default FB in sync (lifecycle owns the realloc). */
function ensureCanvasSize(ctx: WebGLRenderingContext): void {
  const dfb = ctx._defaultFB;
  if (!dfb) return;
  const cw = ctx._canvas.width | 0;
  const ch = ctx._canvas.height | 0;
  if (cw !== dfb.width || ch !== dfb.height) {
    try {
      handleCanvasResize(ctx);
    } catch {
      // lifecycle stub era — draw with stale dims until it lands
    }
  }
}

function scissorState(ctx: WebGLRenderingContext): ScissorState | null {
  const s = ctx._state;
  if (!s.caps.SCISSOR_TEST) return null;
  return { enabled: true, x: s.scissor.x, y: s.scissor.y, w: s.scissor.w, h: s.scissor.h };
}

/* ================================================================== */
/* Surface texel decode/encode (local fallbacks — replace when raster  */
/* formats.ts + fragment-ops.ts land)                                  */
/* ================================================================== */

function decodeSurfaceTexel(surf: Surface, byteOffset: number, out: Float32Array): void {
  const info = surf.info;
  if (info && typeof info.decode === 'function') {
    info.decode(surf.data, byteOffset, out);
    return;
  }
  const d = surf.data as unknown as { constructor: new (...a: never[]) => unknown };
  const f = surf.format;
  const u8 = d as unknown as Uint8Array;
  if (d instanceof Uint8Array) {
    switch (f) {
      case C1.RGBA4: {
        const v = (u8[byteOffset] | (u8[byteOffset + 1] << 8)) & 0xffff;
        out[0] = ((v >> 12) & 0xf) / 15; out[1] = ((v >> 8) & 0xf) / 15;
        out[2] = ((v >> 4) & 0xf) / 15; out[3] = (v & 0xf) / 15;
        return;
      }
      case C1.RGB5_A1: {
        const v = (u8[byteOffset] | (u8[byteOffset + 1] << 8)) & 0xffff;
        out[0] = ((v >> 11) & 0x1f) / 31; out[1] = ((v >> 6) & 0x1f) / 31;
        out[2] = ((v >> 1) & 0x1f) / 31; out[3] = v & 1;
        return;
      }
      case C1.RGB565: {
        const v = (u8[byteOffset] | (u8[byteOffset + 1] << 8)) & 0xffff;
        out[0] = ((v >> 11) & 0x1f) / 31; out[1] = ((v >> 5) & 0x3f) / 63;
        out[2] = (v & 0x1f) / 31; out[3] = 1;
        return;
      }
      default: break;
    }
    const bpp = surf.data.byteLength / (surf.width * surf.height);
    if (bpp === 4) {
      out[0] = u8[byteOffset] / 255; out[1] = u8[byteOffset + 1] / 255;
      out[2] = u8[byteOffset + 2] / 255; out[3] = u8[byteOffset + 3] / 255;
    } else if (bpp === 3) {
      out[0] = u8[byteOffset] / 255; out[1] = u8[byteOffset + 1] / 255;
      out[2] = u8[byteOffset + 2] / 255; out[3] = 1;
    } else if (bpp === 2) {
      out[0] = u8[byteOffset] / 255; out[1] = u8[byteOffset] / 255; // LUMINANCE_ALPHA
      out[2] = u8[byteOffset] / 255; out[3] = u8[byteOffset + 1] / 255;
    } else {
      out[0] = u8[byteOffset] / 255; out[1] = u8[byteOffset] / 255; // LUMINANCE / ALPHA / R8
      out[2] = u8[byteOffset] / 255; out[3] = 1;
    }
    return;
  }
  if (d instanceof Float32Array) {
    const v = (d as Float32Array)[byteOffset >> 2];
    if (f === C1.DEPTH_COMPONENT16 || f === C2.DEPTH_COMPONENT24 || f === C2.DEPTH_COMPONENT32F ||
        f === C1.DEPTH_COMPONENT || f === C1.DEPTH_STENCIL || f === C2.DEPTH24_STENCIL8 || f === C2.DEPTH32F_STENCIL8) {
      out[0] = v; out[1] = v; out[2] = v; out[3] = 1;
    } else {
      // float color formats (stored f32): components from format
      const comps = surfaceComponents(surf);
      for (let c = 0; c < 4; c++) out[c] = c < comps ? (d as Float32Array)[(byteOffset >> 2) + c] : c === 3 ? 1 : 0;
    }
    return;
  }
  if (d instanceof Int8Array) {
    const v = (d as Int8Array)[byteOffset];
    out[0] = v / 127; out[1] = v / 127; out[2] = v / 127; out[3] = 1; return;
  }
  if (d instanceof Uint16Array) {
    const v = (d as Uint16Array)[byteOffset >> 1];
    out[0] = v / 65535; out[1] = v / 65535; out[2] = v / 65535; out[3] = 1; return;
  }
  if (d instanceof Int16Array) {
    const v = (d as Int16Array)[byteOffset >> 1];
    out[0] = v / 32767; out[1] = v / 32767; out[2] = v / 32767; out[3] = 1; return;
  }
  if (d instanceof Int32Array) {
    const v = (d as Int32Array)[byteOffset >> 2];
    out[0] = v; out[1] = v; out[2] = v; out[3] = 1; return;
  }
  if (d instanceof Uint32Array) {
    const v = (d as Uint32Array)[byteOffset >> 2];
    out[0] = v; out[1] = v; out[2] = v; out[3] = 1; return;
  }
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
}

function encodeSurfaceTexel(surf: Surface, byteOffset: number, r: number, g: number, b: number, a: number): void {
  const info = surf.info;
  if (info && typeof info.encode === 'function') {
    info.encode(surf.data, byteOffset, r, g, b, a);
    return;
  }
  const d = surf.data as unknown as { constructor: new (...a: never[]) => unknown };
  const f = surf.format;
  const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
  if (d instanceof Uint8Array) {
    const u8 = d as Uint8Array;
    switch (f) {
      case C1.RGB565: {
        const v = ((clamp01(r) * 31 + 0.5) | 0) << 11 | ((clamp01(g) * 63 + 0.5) | 0) << 5 | (clamp01(b) * 31 + 0.5) | 0;
        u8[byteOffset] = v & 0xff; u8[byteOffset + 1] = (v >> 8) & 0xff;
        return;
      }
      case C1.RGBA4: {
        const v = ((clamp01(r) * 15 + 0.5) | 0) << 12 | ((clamp01(g) * 15 + 0.5) | 0) << 8 |
                  ((clamp01(b) * 15 + 0.5) | 0) << 4 | (clamp01(a) * 15 + 0.5) | 0;
        u8[byteOffset] = v & 0xff; u8[byteOffset + 1] = (v >> 8) & 0xff;
        return;
      }
      case C1.RGB5_A1: {
        const v = ((clamp01(r) * 31 + 0.5) | 0) << 11 | ((clamp01(g) * 31 + 0.5) | 0) << 6 |
                  ((clamp01(b) * 31 + 0.5) | 0) << 1 | (clamp01(a) > 0.5 ? 1 : 0);
        u8[byteOffset] = v & 0xff; u8[byteOffset + 1] = (v >> 8) & 0xff;
        return;
      }
      default: break;
    }
    const bpp = surf.data.byteLength / (surf.width * surf.height);
    if (bpp === 4) {
      u8[byteOffset] = (clamp01(r) * 255 + 0.5) | 0; u8[byteOffset + 1] = (clamp01(g) * 255 + 0.5) | 0;
      u8[byteOffset + 2] = (clamp01(b) * 255 + 0.5) | 0; u8[byteOffset + 3] = (clamp01(a) * 255 + 0.5) | 0;
    } else if (bpp === 3) {
      u8[byteOffset] = (clamp01(r) * 255 + 0.5) | 0; u8[byteOffset + 1] = (clamp01(g) * 255 + 0.5) | 0;
      u8[byteOffset + 2] = (clamp01(b) * 255 + 0.5) | 0;
    } else if (bpp === 2) {
      u8[byteOffset] = (clamp01(r) * 255 + 0.5) | 0; u8[byteOffset + 1] = (clamp01(a) * 255 + 0.5) | 0;
    } else {
      u8[byteOffset] = (clamp01(r) * 255 + 0.5) | 0;
    }
    return;
  }
  if (d instanceof Float32Array) {
    const f32 = d as Float32Array;
    const comps = surfaceComponents(surf);
    for (let c = 0; c < comps; c++) f32[(byteOffset >> 2) + c] = c === 0 ? r : c === 1 ? g : c === 2 ? b : a;
    return;
  }
  if (d instanceof Int32Array) {
    (d as Int32Array)[byteOffset >> 2] = r | 0; return;
  }
  if (d instanceof Uint32Array) {
    (d as Uint32Array)[byteOffset >> 2] = r >>> 0; return;
  }
  if (d instanceof Int8Array) {
    (d as Int8Array)[byteOffset] = r | 0; return;
  }
  if (d instanceof Uint16Array) {
    (d as Uint16Array)[byteOffset >> 1] = (r * 65535 + 0.5) | 0; return;
  }
  if (d instanceof Int16Array) {
    (d as Int16Array)[byteOffset >> 1] = r | 0; return;
  }
}

function surfaceComponents(surf: Surface): number {
  if (surf.info) return surf.info.components;
  const f = surf.format;
  if (f === C1.RGBA || f === C1.RGBA8 || f === C2.RGBA8 || f === C2.RGBA16F || f === C2.RGBA32F ||
      f === C1.RGBA4 || f === C1.RGB5_A1 || f === C2.SRGB8_ALPHA8 || f === C2.RGBA8I || f === C2.RGBA8UI ||
      f === C2.RGBA16I || f === C2.RGBA16UI || f === C2.RGBA32I || f === C2.RGBA32UI || f === C2.RGB10_A2) return 4;
  if (f === C1.RGB || f === C1.RGB8 || f === C2.RGB8 || f === C2.RGB16F || f === C2.RGB32F ||
      f === C1.RGB565 || f === C2.RGB8I || f === C2.RGB8UI || f === C2.RGB16I || f === C2.RGB16UI ||
      f === C2.RGB32I || f === C2.RGB32UI) return 3;
  if (f === C1.LUMINANCE_ALPHA || f === C2.RG || f === C2.RG8 || f === C2.RG16F || f === C2.RG32F ||
      f === C2.RG8I || f === C2.RG8UI || f === C2.RG16I || f === C2.RG16UI || f === C2.RG32I || f === C2.RG32UI) return 2;
  return 1;
}

function surfaceBytesPerPixel(surf: Surface): number {
  if (surf.info) return surf.info.bytesPerPixel;
  return Math.max(1, Math.round(surf.data.byteLength / (surf.width * surf.height)));
}

/** Local color clear (scissor + mask respecting). Replaces raster's clearColorSurface when it lands. */
function clearColorLocal(surf: Surface, r: number, g: number, b: number, a: number, scissor: ScissorState | null, mask: ColorMask): void {
  const w = surf.width;
  const h = surf.height;
  const bpp = surfaceBytesPerPixel(surf);
  const x0 = scissor ? Math.max(0, scissor.x) : 0;
  const y0 = scissor ? Math.max(0, scissor.y) : 0;
  const x1 = scissor ? Math.min(w, scissor.x + scissor.w) : w;
  const y1 = scissor ? Math.min(h, scissor.y + scissor.h) : h;
  const all = mask[0] && mask[1] && mask[2] && mask[3];
  const tmp = new Float32Array(4);
  const data = surf.data as unknown as { constructor: new (...a: never[]) => unknown };
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const off = (y * w + x) * bpp;
      if (all) {
        encodeSurfaceTexel(surf, off, r, g, b, a);
      } else {
        decodeSurfaceTexel(surf, off, tmp);
        encodeSurfaceTexel(surf, off,
          mask[0] ? r : tmp[0], mask[1] ? g : tmp[1], mask[2] ? b : tmp[2], mask[3] ? a : tmp[3]);
      }
    }
  }
  void data;
}

/** Local depth clear (depthMask respecting). */
function clearDepthLocal(surf: Surface, depth: number, scissor: ScissorState | null): void {
  const d = surf.data as Float32Array;
  if (!(d instanceof Float32Array)) return;
  const w = surf.width;
  const h = surf.height;
  const x0 = scissor ? Math.max(0, scissor.x) : 0;
  const y0 = scissor ? Math.max(0, scissor.y) : 0;
  const x1 = scissor ? Math.min(w, scissor.x + scissor.w) : w;
  const y1 = scissor ? Math.min(h, scissor.y + scissor.h) : h;
  for (let y = y0; y < y1; y++) {
    const row = y * w;
    for (let x = x0; x < x1; x++) d[row + x] = depth;
  }
}

/** Local stencil clear (write mask respecting, (value&mask)|(old&~mask)). */
function clearStencilLocal(surf: Surface, value: number, scissor: ScissorState | null, writeMask: number): void {
  const st = surf.stencilData ?? (surf.data as Uint8Array);
  if (!(st instanceof Uint8Array)) return;
  const w = surf.width;
  const h = surf.height;
  const x0 = scissor ? Math.max(0, scissor.x) : 0;
  const y0 = scissor ? Math.max(0, scissor.y) : 0;
  const x1 = scissor ? Math.min(w, scissor.x + scissor.w) : w;
  const y1 = scissor ? Math.min(h, scissor.y + scissor.h) : h;
  const v = value & writeMask;
  if (writeMask === 0xff) {
    for (let y = y0; y < y1; y++) {
      const row = y * w;
      for (let x = x0; x < x1; x++) st[row + x] = v;
    }
  } else {
    for (let y = y0; y < y1; y++) {
      const row = y * w;
      for (let x = x0; x < x1; x++) st[row + x] = v | (st[row + x] & ~writeMask);
    }
  }
}

/* ================================================================== */
/* Attribute fetch (dense extraction, glsl contract)                   */
/* ================================================================== */

interface AttribPlan {
  /** Dense extracted view for enabled buffer-backed attribs; constant view otherwise. */
  source: AttribSource;
  divisor: number;
  instanced: boolean; // divisor > 0
  present: boolean;   // attrib is used by the program
  constant: boolean;  // disabled/no-buffer: source is the CURRENT constant value
}

const MAT2 = 0x8b5a, MAT3 = 0x8b5b, MAT4 = 0x8b5c;

function matrixDims(type: number): { cols: number; rows: number } | null {
  if (type === MAT2) return { cols: 2, rows: 2 };
  if (type === MAT3) return { cols: 3, rows: 3 };
  if (type === MAT4) return { cols: 4, rows: 4 };
  return null;
}

/** One enabled buffer-backed attribute extraction (dense pool fill). */
interface AttribExtraction {
  l: number;
  pool: 'floatPool' | 'intPool' | 'uintPool';
  need: number;      // element count in the dense layout
  elemCount: number; // number of fetched vertices/elements
  comps: number;     // program component count (element stride)
  data: ArrayBuffer;
  dv: DataView;
  typeSize: number;
  stride: number;
  divisor: number;
  integer: boolean;
  unsigned: boolean;
  normalized: boolean;
  aSize: number;
  aType: GLenum;
  aOffset: number;
}

/**
 * Build the per-draw attribute plan: dense extraction into the scratch pools.
 * Returns the attribs array (indexed by location) — constant views for
 * disabled attribs. `indices` non-null for indexed draws.
 *
 * TWO-PASS POOL LAYOUT: ensurePool REASSIGNS sc[which] when it grows, which
 * would invalidate any view created before the grow. So pass 1 walks every
 * enabled buffer-backed location and sums its dense element need per pool;
 * each pool is grown ONCE to the total; pass 2 then hands each extraction a
 * CUMULATIVE byte offset into its pool (running cursor in element units,
 * reset at the top of this call = once per draw; byteOffset = base*4 — all
 * extracted elements are 32-bit, so always 4-byte aligned). Without the
 * cumulative offsets every attribute view aliases byteOffset 0 and the last
 * extraction wins for ALL locations.
 */
function buildAttribs(
  ctx: WebGLRenderingContext,
  pm: ProgramModel,
  req: DrawRequest,
  indices: Uint8Array | Uint16Array | Uint32Array | null,
): { attribs: AttribSource[]; plans: AttribPlan[] } {
  const s = ctx._state;
  const vao = s.vao;
  const maxAttribs = s.limits.MAX_VERTEX_ATTRIBS;
  const sc = getScratch(ctx);
  const attribs: AttribSource[] = new Array(maxAttribs);
  const plans: AttribPlan[] = new Array(maxAttribs);

  const attrs = pm.attributes ?? [];

  // Pass 1 (sizing): plan constants and collect buffer-backed extractions.
  const extract: AttribExtraction[] = [];
  const totals: { floatPool: number; intPool: number; uintPool: number } = { floatPool: 0, intPool: 0, uintPool: 0 };
  for (const pa of attrs) {
    const loc = pa.location;
    if (loc < 0 || loc >= maxAttribs) continue;
    const dims = matrixDims(pa.type);
    const cols = dims ? dims.cols : 1;
    for (let col = 0; col < cols; col++) {
      const l = loc + col;
      if (l >= maxAttribs) break;
      const a = vao.attribs[l];
      if (!a.enabled || !a.buffer || !a.buffer._data) {
        // constant attribute
        plans[l] = { source: 0, divisor: a.divisor, instanced: a.divisor > 0, present: true, constant: true };
        if (pa.integral) {
          // Signedness matters: the uint mirror (constantUI) holds the values
          // written by vertexAttribI4ui — constantI is never touched by those
          // setters (attrib-type-match.html "Correct setup" draw reads the
          // uvec2 constant through constantUI; constantI would be stale zeros).
          const unsigned = pa.type === C1.UNSIGNED_INT ||
            pa.type === C2.UNSIGNED_INT_VEC2 || pa.type === C2.UNSIGNED_INT_VEC3 ||
            pa.type === C2.UNSIGNED_INT_VEC4;
          attribs[l] = unsigned
            ? (a.constantUI as Uint32Array)
            : (a.constantI as Int32Array);
        } else {
          attribs[l] = a.constantF as Float32Array;
        }
        continue;
      }
      plans[l] = { source: 0, divisor: a.divisor, instanced: a.divisor > 0, present: true, constant: false };
      const buf = a.buffer;
      const data = buf._data as ArrayBuffer;
      const typeSize = attribTypeSize(a.type);
      const stride = a.stride === 0 ? a.size * typeSize : a.stride;
      const comps = pa.components; // program component count (element stride in the dense layout)
      // number of elements to extract
      const elemCount = a.divisor > 0
        ? Math.ceil(req.instanceCount / a.divisor)
        : req.count;
      const need = elemCount * comps;
      const integer = a.integer;
      const unsigned = integer && (a.type === C1.UNSIGNED_BYTE || a.type === C1.UNSIGNED_SHORT || a.type === C1.UNSIGNED_INT);
      const pool = integer ? (unsigned ? 'uintPool' : 'intPool') : 'floatPool';
      totals[pool] += need;
      extract.push({
        l, pool, need, elemCount, comps,
        data, dv: new DataView(data), typeSize, stride,
        divisor: a.divisor, integer, unsigned, normalized: a.normalized,
        aSize: a.size, aType: a.type, aOffset: a.offset,
      });
    }
  }
  // Grow each pool once, AFTER sizing (no reassignment during the fill pass).
  if (totals.floatPool > 0) ensurePool(sc, 'floatPool', totals.floatPool);
  if (totals.intPool > 0) ensurePool(sc, 'intPool', totals.intPool);
  if (totals.uintPool > 0) ensurePool(sc, 'uintPool', totals.uintPool);

  // Pass 2 (fill): cumulative per-pool cursor (element units, reset per draw).
  let floatBase = 0, intBase = 0, uintBase = 0;
  for (const ex of extract) {
    const { l, pool, need, elemCount, comps, data, dv, typeSize, stride,
            divisor, integer, unsigned, normalized, aSize, aType, aOffset } = ex;
    if (integer) {
      // raw integer path (vertexAttribIPointer)
      const base = unsigned ? uintBase : intBase;
      const dst = unsigned
        ? new Uint32Array(sc.uintPool.buffer, base * 4, need)
        : new Int32Array(sc.intPool.buffer, base * 4, need);
      if (unsigned) uintBase += need; else intBase += need;
      for (let e = 0; e < elemCount; e++) {
        // baseVertex offsets the element index (indexed draws); baseInstance
        // offsets the instance element (divisor > 0). Both may push the fetch
        // past the end of the buffer — read*Component bounds-checks and yields
        // 0 (ANGLE issue 3764: robust access → NO_ERROR is legal).
        const element = divisor > 0
          ? (req.baseInstance ?? 0) + e
          : (indices ? (req.baseVertex ?? 0) + indices[e] : req.firstOrOffset + e);
        const byteOff = aOffset + element * stride;
        for (let c = 0; c < comps; c++) {
          let v = 0;
          if (c < aSize && byteOff + c * typeSize + typeSize <= data.byteLength) {
            v = readIntComponent(dv, aType, byteOff + c * typeSize);
          }
          dst[e * comps + c] = c < aSize ? v : (c === 3 ? 1 : 0);
        }
      }
      attribs[l] = dst;
    } else {
      // float path with normalization
      const dst = new Float32Array(sc.floatPool.buffer, floatBase * 4, need);
      floatBase += need;
      for (let e = 0; e < elemCount; e++) {
        // baseVertex/baseInstance offsets — see the integer path above.
        const element = divisor > 0
          ? (req.baseInstance ?? 0) + e
          : (indices ? (req.baseVertex ?? 0) + indices[e] : req.firstOrOffset + e);
        const byteOff = aOffset + element * stride;
        for (let c = 0; c < comps; c++) {
          let v = 0;
          if (c < aSize && byteOff + c * typeSize + typeSize <= data.byteLength) {
            v = readFloatComponent(dv, aType, byteOff + c * typeSize, normalized);
          }
          dst[e * comps + c] = c < aSize ? v : (c === 3 ? 1 : 0);
        }
      }
      attribs[l] = dst;
    }
  }
  // Attribs not used by the program: constants (never fetched).
  for (let l = 0; l < maxAttribs; l++) {
    if (!plans[l]) {
      const a = vao.attribs[l];
      plans[l] = { source: 0, divisor: a.divisor, instanced: a.divisor > 0, present: false, constant: true };
      attribs[l] = a.constantF as Float32Array;
    }
  }
  return { attribs, plans };
}

/**
 * Grow a scratch pool on demand. Pools start 0-length; callers create typed-array
 * views over `sc[which].buffer` right after, so the scratch object's field is
 * reassigned here (a bare parameter reassignment would be lost).
 */
function ensurePool(sc: DrawScratch, which: 'floatPool' | 'intPool' | 'uintPool', need: number): void {
  const pool = sc[which];
  if (pool.length >= need) return;
  const len = Math.max(need, 64);
  if (which === 'floatPool') sc.floatPool = new Float32Array(len);
  else if (which === 'intPool') sc.intPool = new Int32Array(len);
  else sc.uintPool = new Uint32Array(len);
}

function attribTypeSize(type: GLenum): number {
  switch (type) {
    case C1.BYTE: case C1.UNSIGNED_BYTE: return 1;
    case C1.SHORT: case C1.UNSIGNED_SHORT: return 2;
    case C1.FLOAT: case C1.INT: case C1.UNSIGNED_INT: return 4;
    default: return 4;
  }
}

function readIntComponent(dv: DataView, type: GLenum, byteOff: number): number {
  switch (type) {
    case C1.BYTE: return dv.getInt8(byteOff);
    case C1.UNSIGNED_BYTE: return dv.getUint8(byteOff);
    case C1.SHORT: return dv.getInt16(byteOff, true);
    case C1.UNSIGNED_SHORT: return dv.getUint16(byteOff, true);
    case C1.INT: return dv.getInt32(byteOff, true);
    case C1.UNSIGNED_INT: return dv.getUint32(byteOff, true);
    default: return 0;
  }
}

function readFloatComponent(dv: DataView, type: GLenum, byteOff: number, normalized: boolean): number {
  switch (type) {
    case C1.FLOAT: return dv.getFloat32(byteOff, true);
    case C1.BYTE: { const v = dv.getInt8(byteOff); return normalized ? v / 127 : v; }
    case C1.UNSIGNED_BYTE: { const v = dv.getUint8(byteOff); return normalized ? v / 255 : v; }
    case C1.SHORT: { const v = dv.getInt16(byteOff, true); return normalized ? v / 32767 : v; }
    case C1.UNSIGNED_SHORT: { const v = dv.getUint16(byteOff, true); return normalized ? v / 65535 : v; }
    case C1.INT: { const v = dv.getInt32(byteOff, true); return normalized ? v / 2147483647 : v; }
    case C1.UNSIGNED_INT: { const v = dv.getUint32(byteOff, true); return normalized ? v / 4294967295 : v; }
    default: return 0;
  }
}

/* ================================================================== */
/* Draw-call assembly                                                  */
/* ================================================================== */

function samplerTargetKey(type: number): 'texture2D' | 'textureCube' | 'texture3D' | 'texture2DArray' {
  switch (type) {
    case C2.SAMPLER_3D: case C2.INT_SAMPLER_3D: case C2.UNSIGNED_INT_SAMPLER_3D:
      return 'texture3D';
    case C2.SAMPLER_2D_ARRAY: case C2.SAMPLER_2D_ARRAY_SHADOW: case 0x8dc4:
    case C2.INT_SAMPLER_2D_ARRAY: case C2.UNSIGNED_INT_SAMPLER_2D_ARRAY:
      return 'texture2DArray';
    case C1.SAMPLER_CUBE: case C2.SAMPLER_CUBE_SHADOW:
    case C2.INT_SAMPLER_CUBE: case C2.UNSIGNED_INT_SAMPLER_CUBE:
      return 'textureCube';
    default:
      return 'texture2D';
  }
}

/**
 * Sampler-uniform GLenum → class (WebGL2 spec "A sampler type must match the
 * internal texture format"; GLES 3.0 §3.8.19). Values per the WebGL 2.0 IDL —
 * NOTE: SAMPLER_2D_ARRAY_SHADOW is 0x8dc4 in WebGL (the spec deviates from
 * GLES 3.0's 0x8dc3; src/glsl/types.ts emits the WebGL value). Returns null
 * for unknown types (WebGL1-only samplers are float-class and are covered by
 * the SAMPLER_2D/SAMPLER_CUBE cases).
 */
function samplerUniformClass(type: GLenum): 'float' | 'signed' | 'unsigned' | 'shadow' | null {
  switch (type) {
    case C1.SAMPLER_2D: case C2.SAMPLER_3D: case C1.SAMPLER_CUBE: case C2.SAMPLER_2D_ARRAY:
      return 'float';
    case C2.SAMPLER_2D_SHADOW: case C2.SAMPLER_CUBE_SHADOW: case 0x8dc4: // SAMPLER_2D_ARRAY_SHADOW (WebGL value)
      return 'shadow';
    case C2.INT_SAMPLER_2D: case C2.INT_SAMPLER_3D: case C2.INT_SAMPLER_CUBE: case C2.INT_SAMPLER_2D_ARRAY:
      return 'signed';
    case C2.UNSIGNED_INT_SAMPLER_2D: case C2.UNSIGNED_INT_SAMPLER_3D:
    case C2.UNSIGNED_INT_SAMPLER_CUBE: case C2.UNSIGNED_INT_SAMPLER_2D_ARRAY:
      return 'unsigned';
    default:
      return null;
  }
}

function effectiveSamplerState(tex: { _params: Record<string, number> }, sampler: { _params: Record<string, number> } | null): SamplerState {
  const p = sampler ? sampler._params : tex._params;
  return {
    minFilter: p[0x2801] ?? 0x2700,
    magFilter: p[0x2800] ?? 0x2601,
    wrapS: p[0x2802] ?? 0x2901,
    wrapT: p[0x2803] ?? 0x2901,
    wrapR: p[0x8072] ?? 0x2901,
    minLod: p[0x813a] ?? -1000,
    maxLod: p[0x813b] ?? 1000,
    compareMode: p[0x884c] ?? 0,
    compareFunc: p[0x884d] ?? 0x0203,
    maxAnisotropy: p[0x84fe] ?? 1,
  };
}

/**
 * Build the per-draw texture environment: images + effective sampler state
 * per unit (indexed by sampler uniform value), and the raster DrawCall
 * bindings. Only units referenced by the program's sampler uniforms are
 * filled; others stay null.
 */
function buildTextureEnv(
  ctx: WebGLRenderingContext,
  pm: ProgramModel,
): { images: (TextureImage | null)[]; samplerStates: SamplerState[]; bindings: (TextureUnitBinding | null)[] } {
  const s = ctx._state;
  const numUnits = s.limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS;
  const images: (TextureImage | null)[] = new Array(numUnits).fill(null);
  const samplerStates: SamplerState[] = new Array(numUnits);
  const bindings: (TextureUnitBinding | null)[] = new Array(numUnits).fill(null);
  const defaults: SamplerState = effectiveSamplerState({ _params: {} }, null);
  for (let u = 0; u < numUnits; u++) samplerStates[u] = defaults;

  const intStore = (pm as unknown as { intStore?: Int32Array | null }).intStore;
  const uniforms = pm.uniforms ?? [];
  if (intStore) {
    for (const u of uniforms) {
      if (!u.sampler) continue;
      // Sampler arrays: `u.size` = array length, elements packed contiguously
      // at intStore[u.location + e] (glsl linker dense packing). Bind each
      // element's unit; the uniform1i write side addresses the same
      // `u.location + e` convention.
      const size = u.size ?? 1;
      for (let e = 0; e < size; e++) {
        const unit = intStore[u.location + e] ?? 0;
        if (unit < 0 || unit >= numUnits) continue;
        const key = samplerTargetKey(u.type);
        const unitState = s.textureUnits[unit];
        // state.ts texture-unit slots are typed with the DOM WebGLTexture
        // interface; they always hold renderer WebGLTexture instances.
        const tex = unitState[key] as unknown as WebGLTexture | null;
        if (!tex || !tex._image) continue;
        // Completeness depends on the CURRENT texParameteri state (MIN_FILTER,
        // BASE/MAX_LEVEL, wrap) plus the levels present — it must be evaluated
        // at DRAW time, not just upload time (a texture uploaded while
        // MIN_FILTER was the default NEAREST_MIPMAP_LINEAR then switched to
        // LINEAR would otherwise sample as incomplete forever). Cheap
        // recompute per draw.
        updateCompleteness(tex, ctx._version, floatLinearExtensionState(ctx));
        const img = tex._image as TextureImage;
        const st = effectiveSamplerState(tex, unitState.sampler);
        images[unit] = img;
        samplerStates[unit] = st;
        bindings[unit] = { img, state: st };
      }
    }
  }
  return { images, samplerStates, bindings };
}

/**
 * WebGL texture feedback-loop check: true when a COMPLETE texture attached to
 * the bound DRAW framebuffer is also bound to a sampler unit referenced by the
 * program's sampler uniforms (unit resolution mirrors buildTextureEnv). Per
 * the WebGL spec such a draw is a no-op + INVALID_OPERATION regardless of the
 * drawBuffers settings (even all-NONE — CTS
 * rendering/rendering-sampling-feedback-loop.html); INCOMPLETE attached
 * textures are ignored (sampling returns (0,0,0,1) — the same page's
 * "incomplete texture" case). Only units the program actually samples matter
 * (a texture bound to an inactive unit is fine — CTS feedback-loop.html).
 *
 * WebGL2 refines the rule to the ATTACHED LEVEL (GLES 3.0 §4.4.3): the draw
 * is invalid only when the attached level falls inside the sampled level
 * window — [level_base, q] for mipmap MIN_FILTERs, [level_base, level_base]
 * for NEAREST/LINEAR — of a complete texture (level window computed per
 * GLES 3.0 §3.8.10 p150, immutable textures clamping into their stored
 * range). CTS conformance2/textures/misc/immutable-tex-render-feedback.html.
 */
function textureFeedbackLoop(ctx: WebGLRenderingContext, pm: ProgramModel): boolean {
  const s = ctx._state;
  const fbo = s.drawFramebuffer;
  if (!fbo || fbo._attachments.size === 0) return false;
  const numUnits = s.limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS;
  const intStore = (pm as unknown as { intStore?: Int32Array | null }).intStore;
  const uniforms = pm.uniforms ?? [];
  if (!intStore) return false;
  // Collect texture attachments once (per-draw; small).
  const textureAtts: { texture: WebGLTexture; level: number }[] = [];
  for (const att of fbo._attachments.values()) {
    if (att.type === 'texture') textureAtts.push({ texture: att.texture, level: att.level });
  }
  if (textureAtts.length === 0) return false;
  for (const u of uniforms) {
    if (!u.sampler) continue;
    // Sampler arrays: elements packed contiguously at intStore[u.location + e]
    // (same convention as buildTextureEnv).
    const size = u.size ?? 1;
    for (let e = 0; e < size; e++) {
      const unit = intStore[u.location + e] ?? 0;
      if (unit < 0 || unit >= numUnits) continue;
      const key = samplerTargetKey(u.type);
      const unitState = s.textureUnits[unit];
      const tex = unitState[key] as unknown as WebGLTexture | null;
      if (!tex || !tex._image) continue;
      if (s.version === 1) {
        // WebGL1: any attached level of a complete texture is a loop (no
        // level-window concept — CTS feedback-loop.html). Completeness at
        // draw time (MIN_FILTER/level chain — cheap per draw).
        updateCompleteness(tex, 1, floatLinearExtensionState(ctx));
        if (!tex._image.complete) continue;
      }
      for (let i = 0; i < textureAtts.length; i++) {
        const att = textureAtts[i];
        if (att.texture !== tex) continue;
        if (s.version === 1) return true;
        // WebGL2: only when the attached level is within the sampled window
        // (samplerFeedbackAtLevel applies the §3.8.10 completeness model with
        // the IMMUTABLE-clamped level window — img.complete uses the raw
        // TEXTURE_MAX_LEVEL and would wrongly gate out clamped-immutable
        // cases, CTS immutable-tex-render-feedback.html).
        if (samplerFeedbackAtLevel(tex, tex._image, att.level, effectiveSamplerState(tex, unitState.sampler))) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * GLES 3.0 §4.4.3 feedback rule, WebGL2: true when a COMPLETE texture sampled
 * at `dstLevel` (the FBO-attached level) is within the level range the sampler
 * actually reads. Mirrors the model CTS
 * immutable-tex-render-feedback.html computes (level_base/level_max per
 * §3.8.10 p150, q = min(floor(log2(maxSize)) + level_base, level_max), and
 * NEAREST/LINEAR MIN_FILTERs sampling only the base level).
 */
function samplerFeedbackAtLevel(tex: WebGLTexture, img: TextureImage, dstLevel: number, st: SamplerState): boolean {
  // Highest stored level (q_tex; holes in `levels` don't count).
  let storedMax = -1;
  for (let l = img.levels.length - 1; l >= 0; l--) {
    if (img.levels[l]) { storedMax = l; break; }
  }
  if (storedMax < 0) return false;
  const rawBase = Math.max(0, tex._params[0x813c] | 0); // TEXTURE_BASE_LEVEL
  const rawMax = Math.max(0, tex._params[0x813d] | 0); // TEXTURE_MAX_LEVEL
  let levelBase = rawBase;
  let levelMax = rawMax;
  if (tex._immutable) {
    // Immutable textures: the effective window clamps into the stored range
    // (GLES 3.0 §3.8.10 p150 — level_base ∈ [0, q], level_max ∈ [level_base, q]).
    levelBase = Math.min(rawBase, storedMax);
    levelMax = Math.min(Math.max(levelBase, rawMax), storedMax);
  }
  if (levelBase > levelMax) return false;
  // Deepest level a full mip chain reaches: floor(log2(maxSize)) + level_base
  // (maxSize excludes 2D_ARRAY layers and cube faces — GLES 3.0 p150).
  // image.width is the level-0-EQUIVALENT size (baseDims·2^base), so use the
  // actual base level's dimensions for the spec q formula.
  const baseLv = img.levels[levelBase];
  const maxSize = Math.max(
    baseLv ? baseLv.width : img.width,
    baseLv ? baseLv.height : img.height,
    img.target === C2.TEXTURE_3D ? (baseLv ? baseLv.depth : img.depth) : 1);
  const q = Math.min(Math.floor(Math.log2(maxSize)) + levelBase, levelMax);
  const minFilter = st.minFilter;
  const useMips = minFilter !== C1.NEAREST && minFilter !== C1.LINEAR;
  const srcMax = useMips ? q : levelBase; // highest level the sampler reads
  // Incomplete textures are safe (sampling yields (0,0,0,1), no feedback).
  if (srcMax > storedMax) return false;
  return levelBase <= dstLevel && dstLevel <= srcMax;
}

/**
 * WebGL2 sampler-type × texture-internal-format compatibility (GLES 3.0
 * §3.8.19 / WebGL2 spec "A sampler type must match the internal texture
 * format"): float samplers accept float + normalized (+ depth with
 * COMPARE_MODE NONE) formats; isampler / usampler only signed/unsigned
 * integer formats; shadow samplers only depth formats with COMPARE_MODE
 * COMPARE_REF_TO_TEXTURE. Returns true when sampling would be incompatible.
 * Only called for COMPLETE textures (incomplete ones sample as (0,0,0,1)
 * without error). CTS uniforms/incompatible-texture-type-for-sampler.html.
 */
function samplerTextureIncompatible(img: TextureImage, samplerClass: 'float' | 'signed' | 'unsigned' | 'shadow', compareMode: number): boolean {
  const info = img.info;
  if (!info) return false; // unknown format class — never block
  if (info.isDepth) {
    // Depth textures: shadow samplers require COMPARE_REF_TO_TEXTURE; float
    // samplers require COMPARE_MODE NONE (the effective mode — sampler object
    // overrides the texture's, per effectiveSamplerState).
    return compareMode === 0x884e /* COMPARE_REF_TO_TEXTURE */
      ? samplerClass !== 'shadow'
      : samplerClass !== 'float';
  }
  if (info.isInteger) {
    return info.isSigned ? samplerClass !== 'signed' : samplerClass !== 'unsigned';
  }
  // Float / normalized (incl. sRGB) formats: float samplers only.
  return samplerClass !== 'float';
}

/** Draw-time driver for samplerTextureIncompatible: true when any ACTIVE
 *  sampler uniform's bound complete texture is format-incompatible. WebGL2
 *  only (WebGL1 has no integer/shadow samplers). */
function incompatibleSamplerTexture(ctx: WebGLRenderingContext, pm: ProgramModel): boolean {
  const s = ctx._state;
  if (s.version !== 2) return false;
  const numUnits = s.limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS;
  const intStore = (pm as unknown as { intStore?: Int32Array | null }).intStore;
  const uniforms = pm.uniforms ?? [];
  if (!intStore) return false;
  for (const u of uniforms) {
    if (!u.sampler) continue;
    const samplerClass = samplerUniformClass(u.type);
    if (!samplerClass) continue;
    const size = u.size ?? 1;
    for (let e = 0; e < size; e++) {
      const unit = intStore[u.location + e] ?? 0;
      if (unit < 0 || unit >= numUnits) continue;
      const key = samplerTargetKey(u.type);
      const unitState = s.textureUnits[unit];
      const tex = unitState[key] as unknown as WebGLTexture | null;
      if (!tex || !tex._image) continue;
      // Incomplete textures are legal (sampling returns (0,0,0,1) — no error).
      updateCompleteness(tex, 2);
      if (!tex._image.complete) continue;
      const st = effectiveSamplerState(tex, unitState.sampler);
      if (samplerTextureIncompatible(tex._image as TextureImage, samplerClass, st.compareMode)) {
        return true;
      }
    }
  }
  return false;
}

/** Build the per-output-location color masks + draw-buffer → attachment map. */
function buildOutputMaps(
  ctx: WebGLRenderingContext,
  pm: ProgramModel,
): { colorMask: ColorMask[]; drawBuffers: number[] } {
  const s = ctx._state;
  const outputs = pm.fragment?.outputs ?? [{ location: 0, type: 0x8b52 }];
  let maxLoc = 0;
  for (const o of outputs) if (o.location > maxLoc) maxLoc = o.location;
  const n = maxLoc + 1;
  const colorMask: ColorMask[] = new Array(n);
  const drawBuffers: number[] = new Array(n).fill(-1);
  for (let l = 0; l < n; l++) {
    // GLES 3.0 §4.2.1: "the draw buffer for fragment colors beyond n is set
    // to NONE" — the fallback for outputs past the drawBuffers() list is NONE
    // (not COLOR_ATTACHMENT0), so such outputs write nowhere.
    const db = l < s.drawBuffers.length ? s.drawBuffers[l] : C1.NONE;
    drawBuffers[l] = db === C1.NONE ? -1 : db - C1.COLOR_ATTACHMENT0;
    // colorMask is keyed by DRAW BUFFER index (the attachment index in the
    // identity mapping — OES_draw_buffers_indexed colorMaskiOES(i, ...) writes
    // the entry for draw buffer i), not by output location: with
    // drawBuffers[l] = COLOR_ATTACHMENTi, output l lands on draw buffer i.
    colorMask[l] = db === C1.NONE ? s.colorMask : (s.colorMaskPerDrawBuffer.get(drawBuffers[l]) ?? s.colorMask);
  }
  return { colorMask, drawBuffers };
}

/**
 * Per-drawbuffer blend state array for the DrawCall (OES_draw_buffers_indexed).
 * Field contract for the raster agent: `dc.blendPerDrawBuffer[d]` =
 * { enabled, srcRGB, dstRGB, srcAlpha, dstAlpha, eqRGB, eqAlpha, color } for
 * draw buffer d ∈ 0..MAX_DRAW_BUFFERS-1. Buffer 0's `enabled` ALWAYS follows
 * the global BLEND cap (spec: draw buffer 0 is governed by enable/disable
 * (BLEND)); buffers > 0 use the extension's per-buffer enable entry when
 * present, falling back to the global cap. Funcs/equations fall back to the
 * base blend state per field. The raster currently blends only output location
 * 0 via `dc.blend` — this array makes per-output blending data available
 * (raster-owned consumption).
 */
function buildBlendPerDrawBuffer(ctx: WebGLRenderingContext): Array<{
  enabled: boolean; srcRGB: number; dstRGB: number; srcAlpha: number; dstAlpha: number;
  eqRGB: number; eqAlpha: number; color: [number, number, number, number];
}> {
  const s = ctx._state;
  const enables = (s as unknown as { blendEnablePerDrawBuffer?: Map<number, boolean> }).blendEnablePerDrawBuffer;
  const arr = new Array<{
    enabled: boolean; srcRGB: number; dstRGB: number; srcAlpha: number; dstAlpha: number;
    eqRGB: number; eqAlpha: number; color: [number, number, number, number];
  }>(s.limits.MAX_DRAW_BUFFERS);
  for (let d = 0; d < s.limits.MAX_DRAW_BUFFERS; d++) {
    const be = s.blendPerDrawBuffer.get(d);
    arr[d] = {
      enabled: d === 0 ? s.caps.BLEND : (enables?.get(d) ?? s.caps.BLEND),
      srcRGB: be?.srcRGB ?? s.blend.srcRGB,
      dstRGB: be?.dstRGB ?? s.blend.dstRGB,
      srcAlpha: be?.srcAlpha ?? s.blend.srcAlpha,
      dstAlpha: be?.dstAlpha ?? s.blend.dstAlpha,
      eqRGB: be?.eqRGB ?? s.blend.eqRGB,
      eqAlpha: be?.eqAlpha ?? s.blend.eqAlpha,
      color: s.blend.color,
    };
  }
  return arr;
}

function buildDrawCall(
  ctx: WebGLRenderingContext,
  pm: ProgramModel,
  req: DrawRequest,
  records: Float32Array,
  stride: number,
  fb: FramebufferTarget,
  env: { images: (TextureImage | null)[]; samplerStates: SamplerState[]; bindings: (TextureUnitBinding | null)[] },
  floatStore: Float32Array,
  uniformBlocks: Record<string, ArrayBufferView> | undefined,
): DrawCall {
  const s = ctx._state;
  const { colorMask, drawBuffers } = buildOutputMaps(ctx, pm);
  const clipControl = getClipControl(ctx);
  const blend0 = s.blendPerDrawBuffer.get(0);
  const blend = {
    enabled: s.caps.BLEND,
    srcRGB: blend0?.srcRGB ?? s.blend.srcRGB,
    dstRGB: blend0?.dstRGB ?? s.blend.dstRGB,
    srcAlpha: blend0?.srcAlpha ?? s.blend.srcAlpha,
    dstAlpha: blend0?.dstAlpha ?? s.blend.dstAlpha,
    eqRGB: blend0?.eqRGB ?? s.blend.eqRGB,
    eqAlpha: blend0?.eqAlpha ?? s.blend.eqAlpha,
    color: s.blend.color,
  };
  const dc: DrawCall = {
    mode: req.mode,
    count: req.count,
    first: 0, // records packed from index 0 (instanced runs at i*count + j)
    instanceCount: req.instanceCount,
    vertices: records,
    vertexStride: stride,
    varyingsOffset: RECORD_TOTAL_HEADER_FLOATS,
    program: pm as never,
    fb,
    viewport: { x: s.viewport.x, y: s.viewport.y, w: s.viewport.w, h: s.viewport.h },
    depthRange: { near: s.depth.range[0], far: s.depth.range[1] },
    clipOrigin: clipControl.origin,
    clipDepthMode: clipControl.depth,
    scissor: {
      enabled: s.caps.SCISSOR_TEST,
      x: s.scissor.x, y: s.scissor.y, w: s.scissor.w, h: s.scissor.h,
    },
    cull: { enabled: s.caps.CULL_FACE, face: s.cullFace, frontFace: s.frontFace },
    polygonOffset: { enabled: s.caps.POLYGON_OFFSET_FILL, factor: s.polygonOffset.factor, units: s.polygonOffset.units },
    dither: s.caps.DITHER,
    colorMask,
    blend,
    depthTest: { enabled: s.caps.DEPTH_TEST, func: s.depth.func },
    depthMask: s.depth.mask,
    stencilTest: {
      enabled: s.caps.STENCIL_TEST,
      front: {
        func: s.stencil.front.func, ref: s.stencil.front.ref, valueMask: s.stencil.front.valueMask,
        writeMask: s.stencil.front.writeMask, fail: s.stencil.front.fail,
        zfail: s.stencil.front.depthFail, zpass: s.stencil.front.depthPass,
      },
      back: {
        func: s.stencil.back.func, ref: s.stencil.back.ref, valueMask: s.stencil.back.valueMask,
        writeMask: s.stencil.back.writeMask, fail: s.stencil.back.fail,
        zfail: s.stencil.back.depthFail, zpass: s.stencil.back.depthPass,
      },
    },
    sampleCoverage: { enabled: s.caps.SAMPLE_COVERAGE, value: s.sampleCoverage.value, invert: s.sampleCoverage.invert },
    rasterizerDiscard: s.caps.RASTERIZER_DISCARD,
    lineWidth: s.lineWidth,
    textures: env.bindings,
    drawBuffers,
    uniforms: floatStore,
    uniformBlocks,
  };
  // Per-drawbuffer blend state (OES_draw_buffers_indexed) — consumed by the
  // raster agent for per-output blending (see buildBlendPerDrawBuffer).
  (dc as unknown as { blendPerDrawBuffer: unknown }).blendPerDrawBuffer = buildBlendPerDrawBuffer(ctx);
  return dc;
}

/* ================================================================== */
/* Transform feedback capture                                          */
/* ================================================================== */

function primitiveInfo(mode: GLenum, count: number): { primCount: number; vertsPerPrim: number } {
  switch (mode) {
    case C1.POINTS: return { primCount: count, vertsPerPrim: 1 };
    case C1.LINES: return { primCount: (count / 2) | 0, vertsPerPrim: 2 };
    case C1.LINE_STRIP: return { primCount: Math.max(0, count - 1), vertsPerPrim: 2 };
    case C1.LINE_LOOP: return { primCount: count > 0 ? count : 0, vertsPerPrim: 2 };
    case C1.TRIANGLES: return { primCount: (count / 3) | 0, vertsPerPrim: 3 };
    default: // TRIANGLE_STRIP / TRIANGLE_FAN
      return { primCount: Math.max(0, count - 2), vertsPerPrim: 3 };
  }
}

/** Vertex position (within the draw) of primitive p, vertex v (count = total vertices, for LINE_LOOP wrap). */
function primVertexIndex(mode: GLenum, p: number, v: number, count: number): number {
  switch (mode) {
    case C1.POINTS: return p;
    case C1.LINES: return p * 2 + v;
    case C1.LINE_STRIP: return p + v;
    case C1.LINE_LOOP: return v === 0 ? p : (p + 1) % count;
    case C1.TRIANGLES: return p * 3 + v;
    case C1.TRIANGLE_STRIP: return p + v;
    default: return v === 0 ? 0 : p + v; // TRIANGLE_FAN
  }
}

function findVarying(varyings: readonly { name: string }[], name: string): number {
  for (let i = 0; i < varyings.length; i++) {
    if (varyings[i].name === name) return i;
    const base = name.replace(/\[0\]$/, '');
    if (varyings[i].name === base) return i;
  }
  return -1;
}

/** True for GLenum varying types that carry integral values (int/uint/bool families). */
function isIntegralVaryingType(glType: number): boolean {
  return (
    glType === 0x1404 /* INT */ ||
    glType === 0x1405 /* UNSIGNED_INT */ ||
    (glType >= 0x8b53 && glType <= 0x8b55) /* INT_VEC2..4 */ ||
    (glType >= 0x8dc6 && glType <= 0x8dc8) /* UNSIGNED_INT_VEC2..4 */ ||
    (glType >= 0x8b56 && glType <= 0x8b59) /* BOOL..BOOL_VEC4 */
  );
}

/** True for unsigned GLenum varying types (uint/uvec families). */
function isUnsignedVaryingType(glType: number): boolean {
  return glType === 0x1405 || (glType >= 0x8dc6 && glType <= 0x8dc8);
}

/** One resolved capture: where the value lives in the vertex record + its component count. */
interface ResolvedTfVarying {
  name: string;
  /** Record float offset of the captured value (gl_Position/gl_PointSize live in the header). */
  recOff: number;
  /** Floats captured per vertex. */
  comps: number;
  /** Integral varying: the record holds float32 VALUES — the TF buffer must get the exact integer, not float bits. */
  integral: boolean;
  /** Unsigned integral varying: captured as uint32. */
  unsigned: boolean;
}

interface ResolvedTfState {
  varyings: ResolvedTfVarying[];
  /** Sum of captured components (interleaved per-vertex stride). */
  totalComps: number;
}

/**
 * Resolve the program's `transformFeedbackVaryings` against the packed vertex
 * record layout (contract §2): each captured varying's SOURCE offset is derived
 * from its position in `pm.varyings` (the record packs varyings contiguously in
 * that order), NOT from the capture list order — the two may differ. The
 * built-ins gl_Position / gl_PointSize are capturable per GLES 3.0 §2.15.2 and
 * live in the record header (offsets 0..3 / 4). Returns null when nothing is
 * captured (no TF varyings, or none resolve into the record).
 */
function resolveTfVaryings(pm: ProgramModel): ResolvedTfState | null {
  const tfVarys = (pm as unknown as { transformFeedbackVaryings?: { name: string }[] }).transformFeedbackVaryings;
  if (!tfVarys || tfVarys.length === 0) return null;
  const varyings = pm.varyings ?? [];
  const out: ResolvedTfVarying[] = [];
  let totalComps = 0;
  for (const tv of tfVarys) {
    let recOff = -1;
    let comps = 0;
    let integral = false;
    let unsigned = false;
    if (tv.name === 'gl_Position') {
      recOff = 0;
      comps = 4;
    } else if (tv.name === 'gl_PointSize') {
      recOff = 4;
      comps = 1;
    } else {
      const vi = findVarying(varyings, tv.name);
      if (vi >= 0) {
        recOff = RECORD_TOTAL_HEADER_FLOATS;
        for (let i = 0; i < vi; i++) recOff += varyings[i].components;
        comps = varyings[vi].components;
        integral = isIntegralVaryingType(varyings[vi].type);
        unsigned = isUnsignedVaryingType(varyings[vi].type);
      }
    }
    if (comps === 0) continue; // not active in the record — skipped (linker normally rejects)
    out.push({ name: tv.name, recOff, comps, integral, unsigned });
    totalComps += comps;
  }
  return out.length === 0 ? null : { varyings: out, totalComps };
}

/**
 * Effective capture capacity (bytes) of a bound TF buffer range.
 * The state model (api/buffers.ts + api/webgl2.ts sync) records a `base`
 * marker per indexed binding: bindBufferBase (whole-buffer, base:true) — the
 * capacity follows the CURRENT data store, so bufferData growth between bind
 * and draw extends it (CTS too-small-buffers.html "Multiple draws success
 * case" re-binds base then grows 31→64 bytes and expects success); an explicit
 * bindBufferRange (base:false) is FIXED at bind time — capacity =
 * min(range.size, store - offset), so "bindBufferRange too small" sections
 * (range 28 < 32 needed, or 12 < 16) generate INVALID_OPERATION and capture
 * NOTHING even though the store is big enough. A size-0 range extends to the
 * end of the buffer (GLES 3.0 §2.10.1) and is recorded base-like (live).
 * GLES 3.0 §2.15.2: a buffer with NO data store (never bufferData'd — e.g.
 * bindBufferBase'd while the generic binding pointed elsewhere) gives
 * UNDEFINED capture results, NO error (CTS too-small-buffers.html separate
 * sections bind buffer 1 while the generic point still holds buffer 0); an
 * unbound binding point captures nothing (varying skipped) — both yield
 * Infinity so the draw proceeds. An explicit bindBufferRange on an
 * unallocated store is a range that cannot exist → too small (the brange
 * cases expect INVALID_OPERATION).
 */
function tfRangeCapacity(tf: WebGLTransformFeedback, bufIdx: number): number {
  const buf = tf._buffers[bufIdx];
  if (!buf) return Infinity; // unbound → varying not captured, no overflow
  const range = tf._bufferRanges[bufIdx] ?? { offset: 0, size: Infinity, base: true };
  if (!buf._data) {
    // Whole-buffer (base) binding over a nonexistent store → undefined results,
    // no error; an explicit range over a nonexistent store → too small.
    return range.base ? Infinity : 0;
  }
  if (range.base) return Math.max(0, buf._data.byteLength - range.offset);
  return Math.min(range.size, Math.max(0, buf._data.byteLength - range.offset));
}

/**
 * GLES 3.0 §2.15.2 draw-time transform-feedback binding conflicts (CTS
 * same-buffer-two-binding-points.html): a draw while the SAME buffer is bound
 * to two or more of the active TF object's indexed binding points, or while a
 * bound point has no corresponding captured varying, generates
 * INVALID_OPERATION and captures NOTHING. NOTE: the CTS test asserts the error
 * right after beginTransformFeedback, so the error must ALSO be generated by
 * begin (api/webgl2.ts — out of this file's allowed set); this check makes the
 * draw itself spec-correct and, crucially, aborts BEFORE capture so the bound
 * buffers stay untouched (the test's checkFloatBuffer assertions).
 */
function tfBindingConflict(tf: WebGLTransformFeedback, prog: WebGLProgram, tfState: ResolvedTfState): boolean {
  const buffers = tf._buffers;
  const n = buffers.length;
  // Same buffer object bound to two or more TF binding points (GLES 3.0
  // §2.15.2; CTS same-buffer-two-binding-points.html cases 1-4). This applies
  // in EVERY mode — INTERLEAVED included.
  for (let i = 0; i < n; i++) {
    const b = buffers[i];
    if (!b) continue;
    for (let j = i + 1; j < n; j++) {
      if (buffers[j] === b) return true;
    }
  }
  // A bound point with no corresponding captured output: SEPARATE_ATTRIBS maps
  // varying i to binding point i, so a bound point ≥ the captured-varying count
  // has no shader output (CTS same-buffer case 4 — the point-2 bind — while
  // its duplicate also trips the check above). INTERLEAVED_ATTRIBS captures
  // everything into point 0 and leftover bindings at higher points are LEGAL
  // (CTS too-small-buffers.html leaves index 1 bound from its separate
  // sections while running interleaved sections and expects NO_ERROR).
  if (prog._tfBufferMode === C2.SEPARATE_ATTRIBS) {
    const used = tfState.varyings.length;
    for (let i = used; i < n; i++) {
      if (buffers[i]) return true;
    }
  }
  return false;
}

/**
 * True when this draw's capture would overflow a bound TF buffer: the spec
 * (GLES 3.0 §2.15.2) makes the draw generate INVALID_OPERATION and capture
 * NOTHING (CTS too-small-buffers.html expects the buffers untouched). The
 * cursor is cumulative across draws of the current TF session.
 */
function tfCaptureOverflows(
  tf: WebGLTransformFeedback,
  prog: WebGLProgram,
  tfState: ResolvedTfState,
  req: DrawRequest,
  startVerts: number,
): boolean {
  const { varyings, totalComps } = tfState;
  if (varyings.length === 0 || totalComps === 0) return false;
  const separate = prog._tfBufferMode === C2.SEPARATE_ATTRIBS;
  const { vertsPerPrim } = primitiveInfo(req.mode, req.count);
  const totalVerts = req.count * req.instanceCount;
  const needed = (startVerts + totalVerts) * 4;
  if (separate) {
    for (let k = 0; k < varyings.length; k++) {
      if (tfRangeCapacity(tf, k) < needed * varyings[k].comps) return true;
    }
  } else {
    if (tfRangeCapacity(tf, 0) < needed * totalComps) return true;
  }
  return false;
}

/**
 * Write the captured varyings of each processed primitive into the bound TF
 * buffers (INTERLEAVED_ATTRIBS / SEPARATE_ATTRIBS) from the packed records.
 * `startVerts` is the per-session vertex cursor (vertices captured by earlier
 * draws of this begin/end session); the write offsets continue from it. Only
 * fully-captured primitives count toward `tf._primitivesWritten` (returned).
 * Unbound buffers are skipped per-varying (the varying is not captured, the
 * primitive still counts); a write that would exceed a bound range aborts the
 * whole draw's capture (the caller's pre-check should have prevented it).
 */
function captureTransformFeedback(
  tf: WebGLTransformFeedback,
  prog: WebGLProgram,
  tfState: ResolvedTfState,
  records: Float32Array,
  stride: number,
  req: DrawRequest,
  startVerts: number,
): number {
  const { varyings: tfVarys, totalComps } = tfState;
  const separate = prog._tfBufferMode === C2.SEPARATE_ATTRIBS;
  const { primCount, vertsPerPrim } = primitiveInfo(req.mode, req.count);
  const buffers = tf._buffers;
  const ranges = tf._bufferRanges;
  let capturedVerts = startVerts;
  let primsCaptured = 0;
  let overflow = false;

  outer: for (let i = 0; i < req.instanceCount && !overflow; i++) {
    for (let p = 0; p < primCount && !overflow; p++) {
      for (let v = 0; v < vertsPerPrim; v++) {
        const vIdx = primVertexIndex(req.mode, p, v, req.count);
        const recBase = (i * req.count + vIdx) * stride;
        for (let k = 0; k < tfVarys.length; k++) {
          const c = tfVarys[k].comps;
          const bufIdx = separate ? k : 0;
          const buf = buffers[bufIdx];
          if (!buf || !buf._data) continue; // unbound → varying not captured
          const range = ranges[bufIdx] ?? { offset: 0, size: buf._size };
          // write offset: interleaved → per-vertex stride of all captured comps;
          // separate → per-varying stride
          let dstByte: number;
          if (separate) {
            dstByte = range.offset + capturedVerts * c * 4;
          } else {
            let before = 0;
            for (let j = 0; j < k; j++) before += tfVarys[j].comps;
            dstByte = range.offset + capturedVerts * totalComps * 4 + before * 4;
          }
          if (dstByte + c * 4 > range.offset + tfRangeCapacity(tf, bufIdx)) {
            overflow = true;
            break outer;
          }
          const recOff = recBase + tfVarys[k].recOff;
          if (tfVarys[k].integral) {
            // Integral varyings ride the float32 record as BIT-PATTERN cells:
            // glsl codegen packs the 32-bit value's bits into the record cell
            // via R.u2f (src/glsl CONTEXT.md "uint varying pack"; the int
            // bit-pack re-apply is paired with this). The TF buffer must hold
            // the exact integer bits (GLES 3.0 §2.15.2), so copy the record
            // cell bits RAW through a reinterpret view — never float→int
            // convert (a denormal bit-pattern cell converts to 0, corrupting
            // uint TF data — get-buffer-sub-data-validity.html). The store
            // into dst wraps via ToInt32/ToUint32, preserving the bit pattern
            // for both signed and unsigned destinations.
            const dst = tfVarys[k].unsigned
              ? new Uint32Array(buf._data, dstByte, c)
              : new Int32Array(buf._data, dstByte, c);
            const recU32 = new Uint32Array(records.buffer, records.byteOffset + recOff * 4, c);
            for (let j = 0; j < c; j++) dst[j] = recU32[j];
          } else {
            new Float32Array(buf._data, dstByte, c).set(records.subarray(recOff, recOff + c));
          }
        }
        capturedVerts++;
      }
      primsCaptured++; // every vertex of this primitive was captured
    }
  }
  tf._primitivesWritten += primsCaptured;
  // Per-session vertex cursor (exact for strips/fans, where prims ≠ verts/vertsPerPrim).
  (tf as unknown as { _tfCaptureCursor?: number })._tfCaptureCursor = capturedVerts;
  return primsCaptured;
}

/** Per-session vertex cursor for a TF object (reset when a new begin/end session starts). */
function tfCaptureCursor(tf: WebGLTransformFeedback): number {
  const cursor = (tf as unknown as { _tfCaptureCursor?: number })._tfCaptureCursor ?? 0;
  // beginTransformFeedback resets _primitivesWritten to 0; a nonzero cursor from
  // a previous session must not carry over.
  return tf._primitivesWritten === 0 && cursor !== 0 ? 0 : cursor;
}

/* ================================================================== */
/* executeDraw                                                         */
/* ================================================================== */

/** True when the draw framebuffer's attachment at `index` (COLOR_ATTACHMENT0+index) has internal format `fmt`. */
function drawBufferAttachmentIs(ctx: WebGLRenderingContext, index: number, fmt: GLenum): boolean {
  const fbo = ctx._state.drawFramebuffer;
  if (!fbo) return false;
  const att = fbo._attachments.get(C1.COLOR_ATTACHMENT0 + index);
  if (!att) return false;
  return (att.type === 'renderbuffer' ? att.renderbuffer._internalformat : att.texture._image?.internalFormat) === fmt;
}

/**
 * Base type family of a fragment output GLenum (ProgramModel.fragment.outputs
 * [].type): 0 = float/normalized, 1 = signed integer, 2 = unsigned integer.
 * GLES 3.0 §4.2.1 — an output may only target a color buffer of the SAME
 * family (int → signed-integer attachment, uint → unsigned, float → any
 * float/normalized buffer).
 */
function outputTypeFamily(type: number): number {
  switch (type) {
    case 0x1404 /* GL_INT */: case 0x8b53 /* GL_INT_VEC2 */:
    case 0x8b54 /* GL_INT_VEC3 */: case 0x8b55 /* GL_INT_VEC4 */:
      return 1;
    case 0x1405 /* GL_UNSIGNED_INT */: case 0x8dc6 /* GL_UNSIGNED_INT_VEC2 */:
    case 0x8dc7 /* GL_UNSIGNED_INT_VEC3 */: case 0x8dc8 /* GL_UNSIGNED_INT_VEC4 */:
      return 2;
    default:
      return 0;
  }
}

/**
 * Execute an assembled draw request: attribute fetch → vertex evaluation →
 * record packing → TF capture → rasterizer.draw (steps above).
 * @internal engine — called by api/draw.ts after validation.
 */
export function executeDraw(ctx: WebGLRenderingContext, req: DrawRequest): void {
  const s = ctx._state;

  // 1. Cheap preconditions (full validation in api/draw.ts).
  const prog = s.currentProgram;
  if (prog !== null) ensureProgramLinked(ctx, prog); // KHR: finish any deferred link
  if (!prog || !prog._linkStatus || !prog._program) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }
  const pm = prog._program as ProgramModel;
  const vao = s.vao;
  const maxAttribs = s.limits.MAX_VERTEX_ATTRIBS;
  for (let loc = 0; loc < maxAttribs; loc++) {
    const a = vao.attribs[loc];
    if (a.enabled && !a.buffer) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
  }
  // WebGL2: integer shader attributes must be backed by integer arrays.
  if (s.version === 2 && pm.attributes) {
    for (const pa of pm.attributes) {
      if (pa.integral && pa.location < maxAttribs) {
        const a = vao.attribs[pa.location];
        if (a.enabled && a.buffer && !a.integer) {
          pushError(ctx, C1.INVALID_OPERATION);
          return;
        }
      }
    }
  }
  // Indexed draws: element buffer bound, offset/bounds valid.
  let indices: Uint8Array | Uint16Array | Uint32Array | null = null;
  if (req.indexed) {
    const eb = vao.elementArrayBuffer;
    const t = req.indexType!;
    const ts = t === C1.UNSIGNED_BYTE ? 1 : t === C1.UNSIGNED_SHORT ? 2 : 4;
    if (!eb) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    // count=0: no elements are read — an unsized element buffer or an offset
    // past the end is NOT an error, only a missing element buffer is (CTS
    // draw-elements-out-of-bounds.html expects NO_ERROR for count=0).
    if (req.count > 0 && (!eb._data || req.firstOrOffset % ts !== 0 || req.firstOrOffset + req.count * ts > eb._size)) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if (req.count > 0) {
      // eb._data is non-null here (count > 0 passed the guard above).
      indices = t === C1.UNSIGNED_BYTE
        ? new Uint8Array(eb._data!, req.firstOrOffset, req.count)
        : t === C1.UNSIGNED_SHORT
          ? new Uint16Array(eb._data!, req.firstOrOffset, req.count)
          : new Uint32Array(eb._data!, req.firstOrOffset, req.count);
    }
  }

  const tf = s.transformFeedback;
  const tfActive = !!tf && tf._active && !tf._paused;

  // Active occlusion query (not counted during transform feedback). The
  // active-query slots are keyed by NUMERIC GLenum target (api/webgl2.ts
  // setActiveQuery writes `slots[target]`), NOT by the named fields.
  let activeQuery: WebGLQuery | null = null;
  if (!tfActive) {
    const q1 = (s.activeQueries as unknown as Record<number, WebGLQuery | null>)[C2.ANY_SAMPLES_PASSED];
    const q2 = (s.activeQueries as unknown as Record<number, WebGLQuery | null>)[C2.ANY_SAMPLES_PASSED_CONSERVATIVE];
    if (q1 && q1._active) activeQuery = q1;
    else if (q2 && q2._active) activeQuery = q2;
  }

  // 2. Resize the drawing buffer BEFORE resolving the draw target: a canvas
  // resize replaces ctx._defaultFB (handleCanvasResize reallocates + clears),
  // so resolving first would capture an orphaned surface and the whole draw
  // would write a buffer nobody presents (CTS to-data-url-test,
  // viewport-unchanged-upon-resize.html).
  ensureCanvasSize(ctx);
  const fb = resolveDrawTarget(ctx);
  if (!fb) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return;
  }

  // Texture feedback-loop detection: drawing while a COMPLETE texture attached
  // to the bound draw framebuffer is also sampled by the program is a no-op +
  // INVALID_OPERATION (spec; CTS renderbuffers/feedback-loop.html +
  // rendering/rendering-sampling-feedback-loop.html). Runs here so it covers
  // arrays/elements/instanced/multi-draw; aborts before vertex evaluation.
  if (textureFeedbackLoop(ctx, pm)) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }

  // WebGL2 sampler-type × texture-format compatibility (GLES 3.0 §3.8.19):
  // an active sampler uniform whose bound COMPLETE texture has an
  // incompatible internal format makes the draw invalid (CTS
  // uniforms/incompatible-texture-type-for-sampler.html). Same placement as
  // the feedback check — covers arrays/elements/instanced/multi-draw before
  // any vertex evaluation.
  if (incompatibleSamplerTexture(ctx, pm)) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }

  if (req.count === 0 || req.instanceCount === 0) return; // nothing to do, no error

  // WebGL2 (GLES 3.0 §4.2.1 + OES_draw_buffers_indexed): a draw is
  // INVALID_OPERATION when any draw buffer with an attached buffer has NO
  // defined fragment-shader output at its index, UNLESS all 4 channels of the
  // draw buffer's effective color mask (per-drawbuffer entry, else the global
  // mask) are false (CTS oes-draw-buffers-indexed.html testColorMaskDrawNoOp:
  // colorMask(1,1,1,1) + shader writing only `validOutput` → INVALID_OPERATION;
  // colorMask(0,0,0,0) → NO_ERROR; colorMaskiOES(invalidOutput, nonzero) →
  // INVALID_OPERATION; those buffers NONE → NO_ERROR). WebGL1 has no
  // multi-draw-buffer/undefined-output rule — skipped (the check also runs
  // before vertex evaluation so the draw is a pure no-op on error).
  if (s.version === 2) {
    const declared = new Set<number>();
    const outs = pm.fragment?.outputs;
    if (outs && outs.length > 0) {
      for (const o of outs) declared.add(o.location);
    } else {
      declared.add(0); // no outputs declared → location 0 is the only candidate
    }
    for (let i = 0; i < s.drawBuffers.length; i++) {
      if (s.drawBuffers[i] === C1.NONE) continue;
      if (declared.has(i)) continue;
      // RASTERIZER_DISCARD discards every fragment before any output write —
      // the undefined-output rule is moot (CTS draw-buffers.html asserts
      // NO_ERROR for an undefined-output draw with discard enabled).
      if (s.caps.RASTERIZER_DISCARD) break;
      // The rule applies ONLY to draw buffers with an attached image: a
      // non-NONE draw buffer whose attachment point has no image is simply
      // skipped by the rasterizer — no error (CTS
      // read-draw-when-missing-image.html draws with
      // drawBuffers=[NONE, COLOR_ATTACHMENT1] and [COLOR_ATTACHMENT0,
      // COLOR_ATTACHMENT1] where CA1 is unattached → NO_ERROR).
      const db = s.drawBuffers[i];
      const dbIdx = db - C1.COLOR_ATTACHMENT0;
      if (dbIdx < 0 || dbIdx >= fb.color.length || !fb.color[dbIdx]) continue;
      const m = s.colorMaskPerDrawBuffer.get(i) ?? s.colorMask;
      if (m[0] || m[1] || m[2] || m[3]) {
        pushError(ctx, C1.INVALID_OPERATION);
        return;
      }
    }
  }
  // GLES 3.0 §4.2.1 (CTS conformance2/rendering/fs-color-type-mismatch-color-
  // buffer-type.html): a draw is INVALID_OPERATION when the base type of a
  // DECLARED fragment shader output does not match the type of the color
  // buffer it writes — float/normalized outputs to integer attachments or
  // integer outputs to float/normalized attachments, including signed-vs-
  // unsigned (an int output to an unsigned attachment mismatches and vice
  // versa). Attachments with no image are skipped (nothing is written); NONE
  // draw buffers are skipped; WebGL1 has no integer color buffers. Placed
  // with the other pre-vertex checks (runs after the count>0 gate above).
  if (s.version === 2) {
    const outs = pm.fragment?.outputs;
    if (outs && outs.length > 0) {
      for (const o of outs) {
        const l = o.location;
        if (l < 0 || l >= s.drawBuffers.length) continue;
        const db = s.drawBuffers[l];
        if (db === C1.NONE) continue;
        const dbIdx = db - C1.COLOR_ATTACHMENT0;
        if (dbIdx < 0 || dbIdx >= fb.color.length || !fb.color[dbIdx]) continue;
        const info = fb.color[dbIdx]!.info;
        // Attachment family: 0 = float/normalized, 1 = signed int, 2 = uint.
        const surfFam = info && info.isInteger ? (info.isSigned ? 1 : 2) : 0;
        if (outputTypeFamily(o.type) !== surfFam) {
          pushError(ctx, C1.INVALID_OPERATION);
          return;
        }
      }
    }
  }
  // WEBGL_render_shared_exponent: a draw is INVALID_OPERATION when any ENABLED
  // draw buffer with an RGB9_E5 attachment has an effective color mask (per-
  // drawbuffer colorMaskiOES entry, else the common mask) that is neither
  // all-true nor all-false — the shared exponent couples all three channels, so
  // individual-channel writes are impossible (CTS webgl-render-shared-exponent
  // colorMaskTest). Placement mirrors the undefined-output check (pre-vertex).
  if (s.version === 2) {
    for (let i = 0; i < s.drawBuffers.length; i++) {
      const db = s.drawBuffers[i];
      if (db === C1.NONE) continue;
      const dbIdx = db - C1.COLOR_ATTACHMENT0;
      if (!drawBufferAttachmentIs(ctx, dbIdx, C2.RGB9_E5)) continue;
      const m = s.colorMaskPerDrawBuffer.get(dbIdx) ?? s.colorMask;
      if ((m[0] || m[1] || m[2] || m[3]) && !(m[0] && m[1] && m[2] && m[3])) {
        pushError(ctx, C1.INVALID_OPERATION);
        return;
      }
    }
  }
  // WEBGL_blend_func_extended (dual-source blending): draw-time restrictions
  // (CTS conformance2/extensions/webgl-blend-func-extended.html + the WebGL1
  // variant — the page matrix is authoritative). A dual-source factor is
  // ACTIVE for a draw buffer when its effective blend state (per-drawbuffer
  // OES_draw_buffers_indexed entry, else the base blend state) contains any
  // SRC1_* factor. Both rules fire with blending ENABLED or DISABLED — the
  // page asserts errors while BLEND is off (js:470-472, js:311-316).
  if (ctx._extensions.has('WEBGL_blend_func_extended')) {
    // (1) Draw-buffer limit: INVALID_OPERATION when the number of ACTIVE draw
    // buffers (non-NONE drawBuffers entries) exceeds
    // MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL and any active draw buffer uses a
    // dual-source factor (page js:398-485: 4 SRC1_* funcs × 4 factor slots ×
    // blend on/off → 32 assertions; non-SRC1 funcs → NO_ERROR).
    let activeBuffers = 0;
    for (let i = 0; i < s.drawBuffers.length; i++) {
      if (s.drawBuffers[i] !== C1.NONE) activeBuffers++;
    }
    if (activeBuffers > s.limits.MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL) {
      for (let i = 0; i < s.drawBuffers.length; i++) {
        if (s.drawBuffers[i] === C1.NONE) continue;
        const be = s.blendPerDrawBuffer.get(i);
        if (
          SRC1_BLEND_FACTORS.includes(be?.srcRGB ?? s.blend.srcRGB) ||
          SRC1_BLEND_FACTORS.includes(be?.dstRGB ?? s.blend.dstRGB) ||
          SRC1_BLEND_FACTORS.includes(be?.srcAlpha ?? s.blend.srcAlpha) ||
          SRC1_BLEND_FACTORS.includes(be?.dstAlpha ?? s.blend.dstAlpha)
        ) {
          pushError(ctx, C1.INVALID_OPERATION);
          return;
        }
      }
    }
    // (2) Missing fragment outputs: for every active draw buffer with a
    // dual-source factor and an effective color mask that is not all false,
    // the fragment shader must write the PRIMARY output (location 0); when
    // blending is enabled for the buffer it must ALSO write a SECONDARY
    // output (location 0, index 1) (page js:304-396 — "no fragment outputs",
    // "only gl_SecondaryFragColorEXT" and "only index 1 output" error even
    // with blending disabled; "only gl_FragColor"/"only index 0 output"
    // error only with blending enabled; all masked-out → NO_ERROR). `index`
    // is not yet part of the glsl output model (layout(index=1) unsupported)
    // — treated as absent, i.e. no secondary output.
    let hasPrimary = false;
    let hasSecondary = false;
    const outs = pm.fragment?.outputs;
    if (outs) {
      for (const o of outs) {
        if (o.location !== 0) continue;
        if ((o as { index?: number }).index === 1) hasSecondary = true;
        else hasPrimary = true;
      }
    }
    const blendEnables = (s as unknown as { blendEnablePerDrawBuffer?: Map<number, boolean> }).blendEnablePerDrawBuffer;
    for (let i = 0; i < s.drawBuffers.length; i++) {
      if (s.drawBuffers[i] === C1.NONE) continue;
      const be = s.blendPerDrawBuffer.get(i);
      if (
        !SRC1_BLEND_FACTORS.includes(be?.srcRGB ?? s.blend.srcRGB) &&
        !SRC1_BLEND_FACTORS.includes(be?.dstRGB ?? s.blend.dstRGB) &&
        !SRC1_BLEND_FACTORS.includes(be?.srcAlpha ?? s.blend.srcAlpha) &&
        !SRC1_BLEND_FACTORS.includes(be?.dstAlpha ?? s.blend.dstAlpha)
      ) {
        continue;
      }
      const m = s.colorMaskPerDrawBuffer.get(i) ?? s.colorMask;
      if (!(m[0] || m[1] || m[2] || m[3])) continue;
      const blendEnabled = i === 0 ? s.caps.BLEND : (blendEnables?.get(i) ?? s.caps.BLEND);
      if (!hasPrimary || (blendEnabled && !hasSecondary)) {
        pushError(ctx, C1.INVALID_OPERATION);
        return;
      }
    }
  }
  // Transform feedback capture pre-check: when the active TF's bound buffers
  // cannot hold this draw's capture (cumulative across the session), the draw
  // generates INVALID_OPERATION and captures NOTHING (GLES 3.0 §2.15.2; CTS
  // too-small-buffers.html). Runs BEFORE attribute fetch / vertex evaluation so
  // huge draws (e.g. 2^16 × 2^16 instances) error out without allocating.
  const tfState = tfActive ? resolveTfVaryings(pm) : null;
  if (tfActive && tfState !== null && tfBindingConflict(tf!, prog, tfState)) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }
  if (tfActive && tfState !== null && tfCaptureOverflows(tf!, prog, tfState, req, tfCaptureCursor(tf!))) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }

  // 3. Attribute fetch (dense extraction).
  const sc = getScratch(ctx);
  const { attribs, plans } = buildAttribs(ctx, pm, req, indices);
  // attribIndices must cover every possible attrib location: it starts
  // 0-length and grows to maxAttribs on first use (typed-array writes past
  // the end are silently dropped — a zero-length array would leave every
  // fetch index undefined and every vertex record NaN).
  if (sc.attribIndices.length < maxAttribs) {
    sc.attribIndices = new Int32Array(maxAttribs);
  }
  const ai = sc.attribIndices;

  // 4. Vertex evaluation loop.
  // Record stride sized from the LOCAL header contract (21 floats) — the
  // imported computeVertexStride/RECORD_HEADER_FLOATS still carry raster's
  // stale 5 until the parallel raster flip lands (see RECORD_TOTAL_HEADER_FLOATS).
  const totalVary = (pm.varyings ?? []).reduce((n, v) => n + v.components, 0);
  const stride = RECORD_TOTAL_HEADER_FLOATS + totalVary;
  const totalVerts = req.count * req.instanceCount;
  const needRecords = totalVerts * stride;
  if (sc.records.length < needRecords) {
    sc.records = new Float32Array(Math.max(needRecords, 64));
  }
  if (sc.outVaryings.length !== totalVary) {
    sc.outVaryings = new Float32Array(totalVary);
  }
  // Zero the clip/cull distance outputs per draw — clip distance 0 = inside,
  // the correct default when the program doesn't write them. UNCONDITIONAL:
  // the record slots must never carry stale values from a previous draw (the
  // current glsl emits scratch-based writes, so they stay zero until the
  // glsl leafWrite flip — that is expected).
  sc.outClipDistance.fill(0);
  sc.outCullDistance.fill(0);
  if (tfActive) {
    // Unwritten varyings capture as 0, not as stale values from a previous draw
    // (the outVaryings/outPosition scratch is reused across draws; CTS
    // unwritten-output-defaults-to-zero.html expects zeros).
    sc.outVaryings.fill(0);
    sc.outPosition.fill(0);
  }
  const scratchSize = (pm as unknown as { scratchSize?: number }).scratchSize ?? 0;
  const intScratchSize = (pm as unknown as { intScratchSize?: number }).intScratchSize ?? 0;
  if (sc.scratch.length < scratchSize) sc.scratch = new Float32Array(Math.max(scratchSize, 64));
  if (sc.intScratch.length < intScratchSize) sc.intScratch = new Int32Array(Math.max(intScratchSize, 64));

  // Uniform-block stores (per block INDEX, from the bound UBO at the block's
  // binding). The blockIndex → binding-point map is per-program state set by
  // uniformBlockBinding() and stored in api/programs.ts's private WeakMap —
  // read it back through the public API (the program is linked here, so the
  // query cannot fail or push errors). Previously this code read a
  // `prog._uniformBlockBindings` property that nothing ever wrote, so every
  // block fell back to binding 0 and UBO-backed draws read zeros.
  const blocks = pm.uniformBlocks ?? [];
  const blockStores: Float32Array[] = new Array(blocks.length);
  const blockIntStores: Int32Array[] = new Array(blocks.length);
  const gl2 = ctx as unknown as {
    getActiveUniformBlockParameter?: (program: WebGLProgram, uniformBlockIndex: GLuint, pname: GLenum) => unknown;
  };
  for (let bi = 0; bi < blocks.length; bi++) {
    const binding =
      gl2.getActiveUniformBlockParameter !== undefined
        ? ((gl2.getActiveUniformBlockParameter(prog, blocks[bi].index, C2.UNIFORM_BLOCK_BINDING) as number) || 0)
        : 0;
    const buf = binding < s.uniformBuffers.length ? s.uniformBuffers[binding] : null;
    const range = binding < s.uniformBufferRanges.length ? s.uniformBufferRanges[binding] : { offset: 0, size: 0 };
    if (buf && buf._data && range.size > 0) {
      const start = Math.min(range.offset, buf._data.byteLength);
      const len = Math.min(range.size, buf._data.byteLength - start);
      blockStores[bi] = new Float32Array(buf._data, start, len >>> 2);
      blockIntStores[bi] = new Int32Array(buf._data, start, len >>> 2);
    } else {
      blockStores[bi] = sc.zeroStore;
      blockIntStores[bi] = sc.zeroIntStore;
    }
  }

  const env = buildTextureEnv(ctx, pm);
  const floatStore = (pm as unknown as { floatStore?: Float32Array | null }).floatStore ?? sc.emptyFloat;
  const intStore = (pm as unknown as { intStore?: Int32Array | null }).intStore ?? sc.emptyInt;

  const vctx: VertexExecCtx & { tex: TextureEnv; out: VertexExecCtx['out'] & { clipDistance: Float32Array; cullDistance: Float32Array } } = {
    attribs,
    attribIndices: ai,
    vertexId: 0,
    instanceId: 0,
    drawId: req.drawId ?? 0, // gl_DrawID: subdraw index; 0 for single draws
    uniforms: floatStore,
    intUniforms: intStore,
    blockStores,
    blockIntStores,
    textures: env.images,
    samplerStates: env.samplerStates,
    // VERTEX shaders can sample textures (GLSL ES allows it; three.js vertex
    // shaders texelFetch morph/skinning textures) — the generated code calls
    // ctx.tex.* exactly like fragment code, so the vertex ctx carries the same
    // per-draw TextureEnv built from the program's sampler bindings.
    tex: createTextureEnv(env.bindings),
    scratch: sc.scratch,
    intScratch: sc.intScratch,
    // gl_DepthRange builtin uniform: [near, far, far - near] (state already
    // clamped to 0..1 by depthRange(); GLES2 §2.11.1).
    depthRange: new Float32Array([s.depth.range[0], s.depth.range[1], s.depth.range[1] - s.depth.range[0]]),
    out: {
      position: sc.outPosition,
      pointSize: 0,
      varyings: sc.outVaryings,
      clipDistance: sc.outClipDistance,
      cullDistance: sc.outCullDistance,
    } as VertexExecCtx['out'] & { clipDistance: Float32Array; cullDistance: Float32Array },
  };

  // Precompute per-loc loops for the inner hot path. Constant (disabled)
  // attribs read the CURRENT constant value: codegen fetches
  // `attribs[loc][attribIndices[loc] * comps + c]`, and constantF/constantI/
  // constantUI are 4-element arrays — their fetch index must stay 0 or the
  // constant view is read out of bounds (NaN → wrong fragment colors).
  const vertexLocs: number[] = [];
  const instancedLocs: { loc: number; divisor: number }[] = [];
  const constantLocs: number[] = [];
  for (let l = 0; l < maxAttribs; l++) {
    const p = plans[l];
    if (!p.present) continue;
    if (p.constant) constantLocs.push(l);
    else if (p.instanced) instancedLocs.push({ loc: l, divisor: p.divisor });
    else vertexLocs.push(l);
  }
  // Constant values never change mid-draw: zero the fetch index once.
  for (let k = 0; k < constantLocs.length; k++) ai[constantLocs[k]] = 0;

  const records = sc.records;
  const pos = sc.outPosition;
  const first = req.indexed ? 0 : req.firstOrOffset;
  const run = pm.vertex.run.bind(pm.vertex);
  let r = 0;
  for (let i = 0; i < req.instanceCount; i++) {
    for (let k = 0; k < instancedLocs.length; k++) {
      ai[instancedLocs[k].loc] = (i / instancedLocs[k].divisor) | 0;
    }
    for (let j = 0; j < req.count; j++, r++) {
      // gl_VertexID = basevertex + element index for indexed draws (GLES 3.2
      // §10.5; the base-vertex-base-instance extensions), first + j otherwise.
      vctx.vertexId = req.indexed ? (req.baseVertex ?? 0) + indices![j] : first + j;
      vctx.instanceId = i;
      for (let k = 0; k < vertexLocs.length; k++) ai[vertexLocs[k]] = j;
      run(vctx);
      const base = r * stride;
      records[base] = pos[0];
      records[base + 1] = pos[1];
      records[base + 2] = pos[2];
      records[base + 3] = pos[3];
      records[base + 4] = vctx.out.pointSize;
      // Clip/cull distance slots (optional-safe — the fields may be absent
      // from the VertexExecCtx out type until the glsl flip; absent = zeroed).
      const cd = vctx.out.clipDistance;
      if (cd) { for (let c = 0; c < 8; c++) records[base + RECORD_OFFSET_CLIP_DISTANCE + c] = cd[c]; }
      const cd2 = vctx.out.cullDistance;
      if (cd2) { for (let c = 0; c < 8; c++) records[base + RECORD_OFFSET_CULL_DISTANCE + c] = cd2[c]; }
      if (totalVary > 0) {
        const vb = base + RECORD_TOTAL_HEADER_FLOATS;
        for (let v = 0; v < totalVary; v++) records[vb + v] = sc.outVaryings[v];
      }
    }
  }

  // 5. Transform feedback capture (bypasses the rasterizer).
  if (tfActive) {
    if (tfState !== null) {
      const prims = captureTransformFeedback(tf!, prog, tfState, records, stride, req, tfCaptureCursor(tf!));
      // TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN query: accumulate the primitives
      // captured by this draw (GLES 3.0 §4.1.4; CTS transform_feedback.html
      // expects 3 for a 3-point POINTS draw).
      const tfQuery = (s.activeQueries as unknown as Record<number, WebGLQuery | null>)[C2.TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN];
      if (tfQuery && tfQuery._active) tfQuery._result += prims;
    }
    return; // TF draws never present (no FB writes) and never count queries
  }

  // 6. Rasterize.
  // Raster DrawCall contract: fragment uniform stores. `uniforms` = default-block
  // float store; `uniformBlocks` = WebGL2 UBO stores keyed by block name
  // (undefined for WebGL1 programs with no blocks).
  const uniformBlocks: Record<string, ArrayBufferView> = {};
  for (let bi = 0; bi < blocks.length; bi++) uniformBlocks[blocks[bi].name] = blockStores[bi];
  const dc = buildDrawCall(ctx, pm, req, records, stride, fb, env, floatStore, blocks.length > 0 ? uniformBlocks : undefined);
  if (activeQuery) {
    (dc as unknown as { sampleCountRef: { value: number } }).sampleCountRef = { value: 0 };
  }
  try {
    rasterDraw(dc);
    if (activeQuery) {
      const ref = (dc as unknown as { sampleCountRef?: { value: number } }).sampleCountRef;
      if (ref) {
        // ANY_SAMPLES_PASSED(*) is a session-scoped BOOLEAN (GLES 3.0 §4.1.4):
        // the result is 1 when any sample passed during the session, else 0.
        // `_result` is reset to 0 at beginQuery; only ever RAISE it so a later
        // all-failed draw cannot clear an earlier pass.
        if (ref.value > 0) activeQuery._result = 1;
        activeQuery._resultAvailable = true;
      }
    }
    presentIfDefault(ctx);
  } catch {
    // rasterizer stub era: swallow "not implemented" throws (remove when raster lands)
  }
}

/* ================================================================== */
/* clear                                                               */
/* ================================================================== */

/** clear(mask): scissor-respecting clear of color/depth/stencil of the draw target. */
export function executeClear(ctx: WebGLRenderingContext, mask: GLuint): void {
  const s = ctx._state;
  // Resize before resolve (canvas resize replaces _defaultFB — see executeDraw).
  ensureCanvasSize(ctx);
  const fb = resolveDrawTarget(ctx);
  if (!fb) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return;
  }
  const scissor = scissorState(ctx);

  if (mask & C1.COLOR_BUFFER_BIT) {
    for (let d = 0; d < s.drawBuffers.length; d++) {
      const db = s.drawBuffers[d];
      if (db === C1.NONE) continue;
      const idx = db - C1.COLOR_ATTACHMENT0;
      const surf = fb.color[idx];
      if (!surf) continue;
      // GLES 3.0 §4.2.1 (CTS clear-func-buffer-type-match.html): a float clear
      // color against an INTEGER color attachment is INVALID_OPERATION — the
      // base type of the clear value must match the buffer family (same
      // surfFam rule as draw-time output matching above).
      if (surf.info.isInteger) {
        pushError(ctx, C1.INVALID_OPERATION);
        return;
      }
      const cm = s.colorMaskPerDrawBuffer.get(d) ?? s.colorMask;
      // WEBGL_render_shared_exponent: Clear is INVALID_OPERATION when an
      // enabled RGB9_E5 draw buffer's effective mask is neither all-true nor
      // all-false (shared exponent couples all channels; CTS colorMaskTest).
      if (drawBufferAttachmentIs(ctx, idx, C2.RGB9_E5) &&
          (cm[0] || cm[1] || cm[2] || cm[3]) && !(cm[0] && cm[1] && cm[2] && cm[3])) {
        pushError(ctx, C1.INVALID_OPERATION);
        return;
      }
      if (!cm[0] && !cm[1] && !cm[2] && !cm[3]) continue;
      let [r, g, b, a] = s.clearColor;
      // The clear color is LINEAR; sRGB color buffers store the ENCODED value
      // (RGB only — GLES 3.0 §4.1.8; CTS clear-srgb-color-buffer.html and
      // read-pixels-from-fbo-test.html assert the encoded bytes). Raster's
      // clearColorSurface stores linear as-is, so encode here.
      if (surf.info.isSRGB) {
        r = linearToSRGB(r); g = linearToSRGB(g); b = linearToSRGB(b);
      }
      try {
        clearColorSurface(surf, r, g, b, a, scissor, cm);
      } catch {
        clearColorLocal(surf, r, g, b, a, scissor, cm);
      }
    }
  }
  if (mask & C1.DEPTH_BUFFER_BIT) {
    if (fb.depth && s.depth.mask) {
      try {
        clearDepthSurface(fb.depth, s.clearDepth, scissor, true);
      } catch {
        clearDepthLocal(fb.depth, s.clearDepth, scissor);
      }
    }
  }
  if (mask & C1.STENCIL_BUFFER_BIT) {
    const stencilSurf = fb.stencil ?? (fb.depth && fb.depth.stencilData ? fb.depth : null);
    if (stencilSurf) {
      const writeMask = (s.stencil.front.writeMask & s.stencil.back.writeMask) & 0xff;
      if (writeMask !== 0) {
        try {
          clearStencilSurface(stencilSurf, s.clearStencil, scissor, writeMask);
        } catch {
          clearStencilLocal(stencilSurf, s.clearStencil, scissor, writeMask);
        }
      }
    }
  }
  presentIfDefault(ctx);
}

/* ================================================================== */
/* readPixels                                                          */
/* ================================================================== */

/** bytes per packed pixel for readPixels (format, type). */
function packBytesPerPixel(format: GLenum, type: GLenum): number {
  const comps =
    format === C1.RGBA || format === C2.RGBA_INTEGER ? 4 :
    format === C1.RGB || format === C2.RGB_INTEGER ? 3 :
    format === C1.LUMINANCE_ALPHA || format === C2.RG || format === C2.RG_INTEGER || format === C2.DEPTH_STENCIL ? 2 :
    1;
  switch (type) {
    case C1.UNSIGNED_BYTE: case C1.BYTE: return comps;
    case C1.UNSIGNED_SHORT_5_6_5: case C1.UNSIGNED_SHORT_4_4_4_4: case C1.UNSIGNED_SHORT_5_5_5_1:
    case C1.UNSIGNED_SHORT: case C1.SHORT: case C2.HALF_FLOAT:
      return comps * 2;
    case C2.UNSIGNED_INT_2_10_10_10_REV: return 4;
    case C2.UNSIGNED_INT_5_9_9_9_REV: return 4; // packed 9/9/9/5 (RGB only)
    case C2.FLOAT_32_UNSIGNED_INT_24_8_REV: return 8;
    default: return comps * 4; // UNSIGNED_INT, INT, FLOAT, UNSIGNED_INT_24_8
  }
}

/**
 * Float triple → UNSIGNED_INT_5_9_9_9_REV bits with the GL readback layout:
 * R in bits 0-8, G 9-17, B 18-26, shared exponent E in 27-31 (GLES 3.0
 * §3.8.21 — CTS webgl-render-shared-exponent.html decodes R from bits 0-8).
 * Quantization matches raster's pack9E5 (E = max(0, floor(log2(maxC)) + 16),
 * mantissa = round(c / 2^(E−24)) clamped to [0, 511]).
 */
function pack9E5Read(r: number, g: number, b: number): number {
  const maxC = Math.max(r, g, b);
  if (maxC <= 2 ** -25) return 0; // ≤ 2^-25 → all-zero (value 0)
  const exp = Math.max(0, Math.floor(Math.log2(maxC)) + 16);
  const scale = 2 ** (exp - 24);
  const m = (v: number): number => Math.min(511, Math.max(0, Math.round(v / scale)));
  return ((exp << 27) | m(r) | (m(g) << 9) | (m(b) << 18)) >>> 0;
}

/** Local pack conversion (replace with raster getPackConverter when it lands). */
function makeLocalPack(surf: Surface, format: GLenum, type: GLenum): ((src: ArrayBufferView, srcOff: number, dst: ArrayBufferView, dstOff: number) => void) | null {
  const tmp = new Float32Array(4);
  const u8 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
  switch (type) {
    case C1.UNSIGNED_BYTE: {
      const comps = format === C1.RGBA ? 4 : format === C1.RGB ? 3 : format === C1.LUMINANCE_ALPHA ? 2 : 1;
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const d8 = dst as Uint8Array;
        if (comps === 4) { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; d8[d + 1] = (u8(tmp[1]) * 255 + 0.5) | 0; d8[d + 2] = (u8(tmp[2]) * 255 + 0.5) | 0; d8[d + 3] = (u8(tmp[3]) * 255 + 0.5) | 0; }
        else if (comps === 3) { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; d8[d + 1] = (u8(tmp[1]) * 255 + 0.5) | 0; d8[d + 2] = (u8(tmp[2]) * 255 + 0.5) | 0; }
        else if (comps === 2) { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; d8[d + 1] = (u8(tmp[3]) * 255 + 0.5) | 0; }
        else { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; }
      };
    }
    case C1.UNSIGNED_SHORT_5_6_5:
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const v = ((u8(tmp[0]) * 31 + 0.5) | 0) << 11 | ((u8(tmp[1]) * 63 + 0.5) | 0) << 5 | (u8(tmp[2]) * 31 + 0.5) | 0;
        const d8 = dst as Uint8Array;
        d8[d] = v & 0xff; d8[d + 1] = (v >> 8) & 0xff;
      };
    case C1.UNSIGNED_SHORT_4_4_4_4:
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const v = ((u8(tmp[0]) * 15 + 0.5) | 0) << 12 | ((u8(tmp[1]) * 15 + 0.5) | 0) << 8 | ((u8(tmp[2]) * 15 + 0.5) | 0) << 4 | (u8(tmp[3]) * 15 + 0.5) | 0;
        const d8 = dst as Uint8Array;
        d8[d] = v & 0xff; d8[d + 1] = (v >> 8) & 0xff;
      };
    case C1.UNSIGNED_SHORT_5_5_5_1:
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const v = ((u8(tmp[0]) * 31 + 0.5) | 0) << 11 | ((u8(tmp[1]) * 31 + 0.5) | 0) << 6 | ((u8(tmp[2]) * 31 + 0.5) | 0) << 1 | (u8(tmp[3]) > 0.5 ? 1 : 0);
        const d8 = dst as Uint8Array;
        d8[d] = v & 0xff; d8[d + 1] = (v >> 8) & 0xff;
      };
    case C2.UNSIGNED_INT_2_10_10_10_REV:
      // RGBA packed 2/10/10/10 (R in bits 0-9, G 10-19, B 20-29, A 30-31).
      // CTS read-pixels-from-fbo-test.html "Special case RGB10_A2" reads with
      // this pair; raster's getPackConverter has no RGBA case for it, so the
      // local pack must (otherwise executeReadPixels errors INVALID_OPERATION).
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const v = (
          (Math.min(1023, Math.max(0, Math.round(tmp[0] * 1023)))) |
          (Math.min(1023, Math.max(0, Math.round(tmp[1] * 1023))) << 10) |
          (Math.min(1023, Math.max(0, Math.round(tmp[2] * 1023))) << 20) |
          (Math.min(3, Math.max(0, Math.round(tmp[3] * 3))) << 30)
        ) >>> 0;
        (dst as Uint32Array)[d >> 2] = v;
      };
    case C2.UNSIGNED_INT_5_9_9_9_REV:
      // Shared-exponent pack (RGB9_E5 storage; WEBGL_render_shared_exponent).
      // The surface stores f32 (isFloat → the raster getPackConverter is
      // skipped), so encode the decoded triple with the GL 9_9_9_5 layout:
      // R in bits 0-8, G 9-17, B 18-26, shared exponent 27-31 (GLES 3.0
      // §3.8.21; CTS webgl-render-shared-exponent.html decodes R from bits
      // 0-8). NOTE: raster's pack9E5 uses the REVERSED layout (R at 18-26) —
      // consistent with its own f32 storage convention but NOT the GL
      // readback layout, so it cannot be reused here.
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        (dst as Uint32Array)[d >> 2] = pack9E5Read(tmp[0], tmp[1], tmp[2]);
      };
    case C1.FLOAT: {
      const comps = format === C1.RGBA ? 4 : format === C1.RGB ? 3 : format === C1.LUMINANCE_ALPHA ? 2 : 1;
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const df = dst as Float32Array;
        for (let c = 0; c < comps; c++) df[(d >> 2) + c] = tmp[c];
      };
    }
    case 0x140b /* HALF_FLOAT */:
    case 0x8d61 /* HALF_FLOAT_OES */: {
      const comps = format === C1.RGBA ? 4 : format === C1.RGB ? 3 : format === C1.LUMINANCE_ALPHA ? 2 : 1;
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const dh = dst as Uint16Array;
        for (let c = 0; c < comps; c++) dh[(d >> 1) + c] = floatToHalf(tmp[c]);
      };
    }
    case C1.UNSIGNED_INT: // DEPTH_COMPONENT / integer formats
      if (format === C1.DEPTH_COMPONENT) {
        return (_src, so, dst, d) => {
          decodeSurfaceTexel(surf, so, tmp);
          const dv = dst as DataView;
          dv.setUint32(d, Math.min(0xffffffff, Math.max(0, Math.round(tmp[0] * 0xffffffff))), true);
        };
      }
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        (dst as Uint32Array)[d >> 2] = tmp[0] >>> 0;
      };
    case C1.UNSIGNED_SHORT:
      if (format === C1.DEPTH_COMPONENT) {
        return (_src, so, dst, d) => {
          decodeSurfaceTexel(surf, so, tmp);
          (dst as Uint16Array)[d >> 1] = Math.min(0xffff, Math.max(0, Math.round(tmp[0] * 0xffff)));
        };
      }
      // Color reads (norm16 surfaces via RGBA/UNSIGNED_SHORT — GL
      // EXT_texture_norm16): pack ALL components ×65535. Previously this branch
      // was depth-only and color reads dropped G/B/A (27189,0,0,0).
      {
        const comps = format === C1.RGBA ? 4 : format === C1.RGB ? 3 : format === C1.LUMINANCE_ALPHA ? 2 : 1;
        return (_src, so, dst, d) => {
          decodeSurfaceTexel(surf, so, tmp);
          const d16 = dst as Uint16Array;
          const base = d >> 1;
          if (comps === 4) {
            d16[base] = (u8(tmp[0]) * 65535 + 0.5) | 0;
            d16[base + 1] = (u8(tmp[1]) * 65535 + 0.5) | 0;
            d16[base + 2] = (u8(tmp[2]) * 65535 + 0.5) | 0;
            d16[base + 3] = (u8(tmp[3]) * 65535 + 0.5) | 0;
          } else if (comps === 3) {
            d16[base] = (u8(tmp[0]) * 65535 + 0.5) | 0;
            d16[base + 1] = (u8(tmp[1]) * 65535 + 0.5) | 0;
            d16[base + 2] = (u8(tmp[2]) * 65535 + 0.5) | 0;
          } else if (comps === 2) {
            d16[base] = (u8(tmp[0]) * 65535 + 0.5) | 0;
            d16[base + 1] = (u8(tmp[3]) * 65535 + 0.5) | 0;
          } else {
            d16[base] = (u8(tmp[0]) * 65535 + 0.5) | 0;
          }
        };
      }
    case C2.UNSIGNED_INT_24_8:
      return (_src, so, dst, d) => {
        decodeSurfaceTexel(surf, so, tmp);
        const st = surf.stencilData ? surf.stencilData[so / surfaceBytesPerPixel(surf)] : 0;
        (dst as Uint32Array)[d >> 2] = ((Math.min(0xffffff, Math.max(0, Math.round(tmp[0] * 0xffffff))) << 8) | (st & 0xff)) >>> 0;
      };
    default:
      return null;
  }
}

/** readPixels with pack-state (alignment/rowLength/skip) + format conversions. */
export function executeReadPixels(
  ctx: WebGLRenderingContext,
  x: GLint, y: GLint, width: GLsizei, height: GLsizei,
  format: GLenum, type: GLenum, pixels: ArrayBufferView,
  dstOffset = 0,
): void {
  const s = ctx._state;
  if (width === 0 || height === 0) return;
  // WebGL2: reading from the DEFAULT framebuffer while READ_BUFFER is NONE is
  // INVALID_OPERATION (GLES 3.0 §4.3.1; CTS renderbuffers/readbuffer.html
  // "should generate INVALID_OPERATION when reading from framebuffer and read
  // buffer is GL_NONE" — the FBO path already errors via the missing
  // attachment in resolveReadColor; the default-FB path resolves unconditionally).
  if (s.version === 2 && s.readFramebuffer === null && s.readBuffer === C1.NONE) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }
  // A page-driven canvas.width/height change with no intervening draw/clear
  // leaves _defaultFB stale (handleCanvasResize reallocates + clears); sync it
  // BEFORE resolving the read surface — every other pipeline entry
  // (draw/clear/blit/clearBuffer) does the same (CTS
  // renderbuffer*-initialization.html read back the pre-resize buffer).
  ensureCanvasSize(ctx);
  const surf = resolveReadColor(ctx);
  if (!surf) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }
  // Pack converter: raster's when available, else local. When the surface is
  // float-storage (W1 unsized float textures promoted to f32), raster's static
  // table maps unsized formats to u8 and would mis-decode — use the local pack,
  // which decodes via surf.info (the real float spec).
  let conv: ((src: ArrayBufferView, srcOff: number, dst: ArrayBufferView, dstOff: number) => void) | null = null;
  if (!surf.info?.isFloat) {
    try {
      const rc = getPackConverter(surf.format, format, type);
      if (rc) {
        conv = (src, so, dst, d) => rc.convert(src, so, dst, d);
      }
    } catch {
      conv = null;
    }
  }
  if (!conv) {
    conv = makeLocalPack(surf, format, type);
    if (!conv) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
  }

  const pack = s.pixelStore.pack;
  const bpp = packBytesPerPixel(format, type);
  const rowLen = pack.rowLength || width;
  const rowStride = alignUp(rowLen * bpp, pack.alignment);
  const sbpp = surfaceBytesPerPixel(surf);

  // Destination: client ArrayBufferView, or PIXEL_PACK_BUFFER when bound.
  // `dstOffset` (WebGL2 readPixels 8th arg, element offset into the view) is
  // validated api-side; here it just shifts the write base.
  let dstBase: number;
  let dstView: ArrayBufferView;
  const packBuf = s.pixelPackBuffer;
  if (packBuf && packBuf._data) {
    dstView = new Uint8Array(packBuf._data);
    dstBase = (pixels as unknown as number) + pack.skipRows * rowStride + pack.skipPixels * bpp;
  } else {
    dstView = pixels;
    dstBase = pixels.byteOffset +
      dstOffset * (pixels as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT +
      pack.skipRows * rowStride + pack.skipPixels * bpp;
  }

  for (let row = 0; row < height; row++) {
    const srcRow = y + row; // GL row (0 = bottom); surfaces store row 0 = bottom
    if (srcRow < 0 || srcRow >= surf.height) continue;
    const dstRow = dstBase + row * rowStride;
    for (let px = 0; px < width; px++) {
      const sx = x + px;
      if (sx < 0 || sx >= surf.width) continue;
      conv(surf.data, (srcRow * surf.width + sx) * sbpp, dstView, dstRow + px * bpp);
    }
  }
}

/* ================================================================== */
/* blitFramebuffer                                                     */
/* ================================================================== */

/** Local nearest blit for color surfaces (replaces raster's blitColorSurface). */
function blitColorLocal(
  src: Surface, dst: Surface,
  srcX: number, srcY: number, srcW: number, srcH: number,
  dstX: number, dstY: number, dstW: number, dstH: number,
): void {
  const sbpp = surfaceBytesPerPixel(src);
  const dbpp = surfaceBytesPerPixel(dst);
  const tmp = new Float32Array(4);
  for (let dy = 0; dy < Math.abs(dstH); dy++) {
    const dstRow = dstY + (dstH < 0 ? -dy : dy);
    if (dstRow < 0 || dstRow >= dst.height) continue;
    const sy = srcY + (srcH < 0 ? -dy : dy) * (Math.abs(srcH) / Math.abs(dstH));
    const srcRow = Math.floor(sy);
    if (srcRow < 0 || srcRow >= src.height) continue;
    for (let dx = 0; dx < Math.abs(dstW); dx++) {
      const dstCol = dstX + (dstW < 0 ? -dx : dx);
      if (dstCol < 0 || dstCol >= dst.width) continue;
      const sx = srcX + (srcW < 0 ? -dx : dx) * (Math.abs(srcW) / Math.abs(dstW));
      const srcCol = Math.floor(sx);
      if (srcCol < 0 || srcCol >= src.width) continue;
      decodeSurfaceTexel(src, (srcRow * src.width + srcCol) * sbpp, tmp);
      encodeSurfaceTexel(dst, (dstRow * dst.width + dstCol) * dbpp, tmp[0], tmp[1], tmp[2], tmp[3]);
    }
  }
}

/** blitFramebuffer (color with filter, depth/stencil nearest-only). */
export function executeBlitFramebuffer(
  ctx: WebGLRenderingContext,
  srcX0: GLint, srcY0: GLint, srcX1: GLint, srcY1: GLint,
  dstX0: GLint, dstY0: GLint, dstX1: GLint, dstY1: GLint,
  mask: GLuint, filter: GLenum,
): void {
  const s = ctx._state;
  const srcW = srcX1 - srcX0;
  const srcH = srcY1 - srcY0;
  const dstW = dstX1 - dstX0;
  const dstH = dstY1 - dstY0;

  // Resize before resolving read/draw targets (a canvas resize replaces
  // _defaultFB — see executeDraw); blitFramebuffer can target the default FB
  // and presentIfDefault runs at the end.
  ensureCanvasSize(ctx);

  if (mask & C1.COLOR_BUFFER_BIT) {
    const src = resolveReadColor(ctx);
    const fb = resolveDrawTarget(ctx);
    // GLES 3.0 §4.3.2 + CTS read-draw-when-missing-image.html: when the read
    // buffer has NO image, a color blit is INVALID_OPERATION only if the
    // blit's destination — the first enabled draw buffer — HAS an image;
    // source- and destination-both-missing → NO_ERROR no-op.
    if (!src && fb) {
      const db0 = s.drawBuffers[0] ?? C1.COLOR_ATTACHMENT0;
      const dst0 = db0 === C1.NONE ? null : fb.color[db0 - C1.COLOR_ATTACHMENT0];
      if (dst0) {
        pushError(ctx, C1.INVALID_OPERATION);
        return;
      }
    }
    if (src && fb) {
      const db0 = s.drawBuffers[0] ?? C1.COLOR_ATTACHMENT0;
      const dst = db0 === C1.NONE ? null : fb.color[db0 - C1.COLOR_ATTACHMENT0];
      if (dst) {
        try {
          blitColorSurface(src, dst, srcX0, srcY0, srcW, srcH, dstX0, dstY0, dstW, dstH,
            filter === C1.LINEAR ? 'linear' : 'nearest');
        } catch {
          blitColorLocal(src, dst, srcX0, srcY0, srcW, srcH, dstX0, dstY0, dstW, dstH);
        }
      }
    }
  }
  if (mask & (C1.DEPTH_BUFFER_BIT | C1.STENCIL_BUFFER_BIT)) {
    const srcDepth = resolveReadDepthStencil(ctx, C1.DEPTH_ATTACHMENT);
    const srcStencil = resolveReadDepthStencil(ctx, C1.STENCIL_ATTACHMENT);
    const fb = resolveDrawTarget(ctx);
    const dstDepth = fb ? fb.depth : null;
    const dstStencil = fb ? fb.stencil ?? (fb.depth && fb.depth.stencilData ? fb.depth : null) : null;
    // GLES 3.0 §4.3.2 + CTS read-draw-when-missing-image.html: a depth/stencil
    // blit is INVALID_OPERATION when exactly one side (source or destination)
    // has the buffer; both-present or both-missing → proceed/no-op. The source
    // side is judged by raw attachment presence (readAttachmentHasImage) —
    // plane-scoped resolution would misreport DS images attached at
    // DEPTH_ATTACHMENT as "missing".
    if ((mask & C1.DEPTH_BUFFER_BIT) !== 0 && readAttachmentHasImage(ctx, C1.DEPTH_ATTACHMENT) !== !!dstDepth) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if ((mask & C1.STENCIL_BUFFER_BIT) !== 0 && readAttachmentHasImage(ctx, C1.STENCIL_ATTACHMENT) !== !!dstStencil) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if (srcDepth && dstDepth) {
      try {
        blitDepthStencilSurface(srcDepth, dstDepth, srcX0, srcY0, srcW, srcH, dstX0, dstY0, dstW, dstH);
      } catch {
        blitColorLocal(srcDepth, dstDepth, srcX0, srcY0, srcW, srcH, dstX0, dstY0, dstW, dstH);
      }
    }
    if (srcStencil && dstStencil) {
      try {
        blitDepthStencilSurface(srcStencil, dstStencil, srcX0, srcY0, srcW, srcH, dstX0, dstY0, dstW, dstH);
      } catch {
        blitColorLocal(srcStencil, dstStencil, srcX0, srcY0, srcW, srcH, dstX0, dstY0, dstW, dstH);
      }
    }
  }
  presentIfDefault(ctx);
}

/* ================================================================== */
/* clearBuffer*                                                        */
/* ================================================================== */

/** Local integer color clear (values as raw ints, mask respecting). */
function clearColorIntLocal(
  surf: Surface, values: Int32Array | Uint32Array, mask: ColorMask, scissor: ScissorState | null,
): void {
  const w = surf.width;
  const h = surf.height;
  const bpp = surfaceBytesPerPixel(surf);
  const x0 = scissor ? Math.max(0, scissor.x) : 0;
  const y0 = scissor ? Math.max(0, scissor.y) : 0;
  const x1 = scissor ? Math.min(w, scissor.x + scissor.w) : w;
  const y1 = scissor ? Math.min(h, scissor.y + scissor.h) : h;
  const d = surf.data as unknown as { constructor: new (...a: never[]) => unknown };
  // Packed integer storage (e.g. RGB10_A2UI: one u32 texel holding 10/10/10/2
  // bits): per-component element writes would overwrite neighboring texels.
  // Detect: one element per texel (element size == bytesPerPixel) with more
  // than one component. Decode-merge-encode via the format registry instead.
  const elBytes = (surf.data as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const packedInt = surf.info !== null && surf.info.isInteger && elBytes === surf.info.bytesPerPixel && surf.info.components > 1;
  if (packedInt) {
    const info = surf.info;
    const out = new Float32Array(4);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const off = (y * w + x) * bpp;
        info.decode(surf.data, off, out);
        if (mask[0]) out[0] = values[0] ?? 0;
        if (mask[1]) out[1] = values[1] ?? 0;
        if (mask[2]) out[2] = values[2] ?? 0;
        if (mask[3]) out[3] = values[3] ?? 1;
        info.encode(surf.data, off, out[0], out[1], out[2], out[3]);
      }
    }
    return;
  }
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const off = (y * w + x) * bpp;
      // integer storage: write raw ints per component
      const write = (comp: number, v: number) => {
        const target = d;
        if (target instanceof Int32Array) target[(off >> 2) + comp] = v;
        else if (target instanceof Uint32Array) target[(off >> 2) + comp] = v >>> 0;
        else if (target instanceof Int16Array) target[(off >> 1) + comp] = v;
        else if (target instanceof Uint16Array) target[(off >> 1) + comp] = v >>> 0;
        else if (target instanceof Int8Array) target[off + comp] = v;
        else if (target instanceof Uint8Array) target[off + comp] = v >>> 0;
      };
      if (mask[0]) write(0, values[0] ?? 0);
      if (mask[1]) write(1, values[1] ?? 0);
      if (mask[2]) write(2, values[2] ?? 0);
      if (mask[3]) write(3, values[3] ?? 1);
    }
  }
}

/** clearBuffer* (WebGL2): clear one attachment of the draw target. */
export function executeClearBuffer(
  ctx: WebGLRenderingContext,
  buffer: GLenum, drawbuffer: GLint, values: Float32Array | Int32Array | Uint32Array | null,
  srcOffset = 0,
  depth?: number, stencil?: number,
): void {
  const s = ctx._state;
  // Resize before resolve (canvas resize replaces _defaultFB — see executeDraw).
  ensureCanvasSize(ctx);
  const fb = resolveDrawTarget(ctx);
  if (!fb) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return;
  }
  const scissor = scissorState(ctx);

  if (buffer === C2.COLOR) {
    if (drawbuffer < 0 || drawbuffer >= s.limits.MAX_DRAW_BUFFERS) {
      pushError(ctx, C1.INVALID_VALUE);
      return;
    }
    const db = s.drawBuffers[drawbuffer] ?? C1.NONE;
    if (db === C1.NONE || !values) return; // no-op
    const idx = db - C1.COLOR_ATTACHMENT0;
    const surf = fb.color[idx];
    if (!surf) return;
    const cm = s.colorMaskPerDrawBuffer.get(drawbuffer) ?? s.colorMask;
    // WEBGL_render_shared_exponent: ClearBuffer* is INVALID_OPERATION when the
    // target draw buffer holds an RGB9_E5 attachment and its effective mask is
    // neither all-true nor all-false (CTS colorMaskTest clearBufferfv checks).
    if (drawBufferAttachmentIs(ctx, idx, C2.RGB9_E5) &&
        (cm[0] || cm[1] || cm[2] || cm[3]) && !(cm[0] && cm[1] && cm[2] && cm[3])) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if (!cm[0] && !cm[1] && !cm[2] && !cm[3]) return;
    if (values instanceof Int32Array || values instanceof Uint32Array) {
      clearColorIntLocal(surf, values.subarray(srcOffset), cm, scissor);
    } else {
      const v = values as Float32Array;
      // The clear color is LINEAR; sRGB color buffers store the ENCODED value
      // (GLES 3.0 §4.1.8 — RGB only, alpha untouched; CTS clear-srgb-color-
      // buffer.html + read-pixels-from-fbo-test.html). Raster's
      // clearColorSurface stores linear as-is, so encode here.
      let r = v[srcOffset] ?? 0, g = v[srcOffset + 1] ?? 0;
      let b = v[srcOffset + 2] ?? 0, a = v[srcOffset + 3] ?? 1;
      if (surf.info.isSRGB) {
        r = linearToSRGB(r); g = linearToSRGB(g); b = linearToSRGB(b);
      }
      try {
        clearColorSurface(surf, r, g, b, a, scissor, cm);
      } catch {
        clearColorLocal(surf, r, g, b, a, scissor, cm);
      }
    }
  } else if (buffer === C2.DEPTH) {
    if (!fb.depth) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if (!s.depth.mask || !values) return;
    try {
      clearDepthSurface(fb.depth, values[srcOffset] ?? 0, scissor, true);
    } catch {
      clearDepthLocal(fb.depth, values[srcOffset] ?? 0, scissor);
    }
  } else if (buffer === C2.STENCIL) {
    const stencilSurf = fb.stencil ?? (fb.depth && fb.depth.stencilData ? fb.depth : null);
    if (!stencilSurf) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if (!values) return;
    const writeMask = (s.stencil.front.writeMask & s.stencil.back.writeMask) & 0xff;
    if (writeMask === 0) return;
    try {
      clearStencilSurface(stencilSurf, values[srcOffset] ?? 0, scissor, writeMask);
    } catch {
      clearStencilLocal(stencilSurf, values[srcOffset] ?? 0, scissor, writeMask);
    }
  } else { // DEPTH_STENCIL (clearBufferfi only)
    if (!fb.depth || !fb.depth.stencilData) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if (s.depth.mask && depth !== undefined) {
      try {
        clearDepthSurface(fb.depth, depth, scissor, true);
      } catch {
        clearDepthLocal(fb.depth, depth, scissor);
      }
    }
    const writeMask = (s.stencil.front.writeMask & s.stencil.back.writeMask) & 0xff;
    if (writeMask !== 0 && stencil !== undefined) {
      try {
        clearStencilSurface(fb.depth, stencil, scissor, writeMask);
      } catch {
        clearStencilLocal(fb.depth, stencil, scissor, writeMask);
      }
    }
  }
  presentIfDefault(ctx);
}

/* ================================================================== */
/* Shared draw validation (api/draw.ts single draws + multiDraw engine) */
/* ================================================================== */

const DRAW_MODES = new Set<number>([
  C1.POINTS, C1.LINE_STRIP, C1.LINE_LOOP, C1.LINES,
  C1.TRIANGLE_STRIP, C1.TRIANGLE_FAN, C1.TRIANGLES,
]);

/** Bytes per index for drawElements types. */
function indexTypeSize(type: GLenum): number {
  return type === C1.UNSIGNED_BYTE ? 1 : type === C1.UNSIGNED_SHORT ? 2 : 4;
}

/**
 * Preconditions shared by every draw (ordered per spec, AFTER the mode/count/
 * type/offset argument checks): program linked + in use (INVALID_OPERATION),
 * transform-feedback active mode mismatch (INVALID_OPERATION), every enabled
 * attrib array backed by a buffer (INVALID_OPERATION), draw target resolvable
 * (INVALID_FRAMEBUFFER_OPERATION), and — GLES 2.0 §4.1.5 — stencil front/back
 * state consistency when STENCIL_TEST is enabled and a stencil buffer is
 * attached (INVALID_OPERATION). The engine's executeDraw re-checks these as a
 * safety net; the api layer checks first so error ordering is exact.
 */
function validateCommonDraw(ctx: WebGLRenderingContext, mode: GLenum, indexed: boolean = false): boolean {
  const s = ctx._state;
  const prog = s.currentProgram;
  if (prog !== null) ensureProgramLinked(ctx, prog); // KHR: finish any deferred link
  if (!prog || !prog._linkStatus || !prog._program) {
    pushError(ctx, C1.INVALID_OPERATION);
    return false;
  }
  // Transform feedback active (bound object; the default TF object is
  // api/webgl2.ts-private — engine scope covers the bound TF, which is also
  // the only TF executeDraw captures from).
  const tf = s.transformFeedback;
  if (tf && tf._active && !tf._paused && tf._primitiveMode !== mode) {
    pushError(ctx, C1.INVALID_OPERATION);
    return false;
  }
  // GLES 3.0 §2.15.2 double-bind rules (CTS
  // transform_feedback/simultaneous_binding.html): a draw is INVALID_OPERATION
  // when a buffer in the BOUND TF object's indexed binding points is also used
  // by the draw through another binding point. Enabled vertex attribute array
  // buffers (and the element-array buffer for indexed draws) and indexed
  // UNIFORM_BUFFER binding points are forbidden REGARDLESS of whether TF is
  // active (the test errors with TF disabled too — "Test ARRAY_BUFFER",
  // "Test UNIFORM_BUFFER"); the other targets (ARRAY_BUFFER, COPY_READ/WRITE,
  // PIXEL_PACK/UNPACK — "Test TF buffer bound to target unused by draw") are
  // forbidden only while TF is active and not paused. The generic
  // TRANSFORM_FEEDBACK_BUFFER point is legal (not part of the TF object, the
  // test draws from it with TF enabled). DISABLED attribs pointing at a TF
  // buffer are legal (not used by the draw), and an UNBOUND TF object's
  // bindings are irrelevant — only s.transformFeedback (the bound object; the
  // default TF is exposed there during default sessions) is checked.
  const vao = s.vao;
  const maxAttribs = s.limits.MAX_VERTEX_ATTRIBS;
  if (tf) {
    const tfBuffers = tf._buffers;
    const tfActive = tf._active && !tf._paused;
    for (let loc = 0; loc < maxAttribs; loc++) {
      const a = vao.attribs[loc];
      if (a.enabled && a.buffer && tfBuffers.includes(a.buffer)) {
        pushError(ctx, C1.INVALID_OPERATION);
        return false;
      }
    }
    if (indexed && vao.elementArrayBuffer && tfBuffers.includes(vao.elementArrayBuffer)) {
      pushError(ctx, C1.INVALID_OPERATION);
      return false;
    }
    for (let i = 0; i < s.uniformBuffers.length; i++) {
      const b = s.uniformBuffers[i];
      if (b && tfBuffers.includes(b)) {
        pushError(ctx, C1.INVALID_OPERATION);
        return false;
      }
    }
    if (tfActive &&
        ((s.arrayBuffer && tfBuffers.includes(s.arrayBuffer)) ||
         (s.copyReadBuffer && tfBuffers.includes(s.copyReadBuffer)) ||
         (s.copyWriteBuffer && tfBuffers.includes(s.copyWriteBuffer)) ||
         (s.pixelPackBuffer && tfBuffers.includes(s.pixelPackBuffer)) ||
         (s.pixelUnpackBuffer && tfBuffers.includes(s.pixelUnpackBuffer)))) {
      pushError(ctx, C1.INVALID_OPERATION);
      return false;
    }
  }
  for (let loc = 0; loc < maxAttribs; loc++) {
    const a = vao.attribs[loc];
    if (a.enabled && !a.buffer) {
      pushError(ctx, C1.INVALID_OPERATION);
      return false;
    }
  }
  // GLES 3.0 §2.11.6 (CTS conformance2/rendering/attrib-type-match.html): at
  // draw time, the base type of every ACTIVE vertex shader input must match the
  // type of its backing data. ENABLED arrays: vertexAttribPointer (float) vs
  // vertexAttribIPointer (integer) must match the shader input's float vs
  // int/uint declaration, INCLUDING int-vs-uint signedness (an INT array on a
  // uvec input is a mismatch, and vice versa). DISABLED arrays: the current
  // generic value's type (most recent setter — vertexAttrib* → float,
  // vertexAttribI* → int, vertexAttribIu* → uint; default float) must match
  // likewise. Inactive attributes are exempt (they are not in pm.attributes).
  {
    const pm = prog._program;
    if (pm) {
      const attrs = pm.attributes ?? [];
      for (const pa of attrs) {
        const loc = pa.location;
        if (loc < 0 || loc >= maxAttribs) continue;
        const a = vao.attribs[loc];
        const shaderInt = !!pa.integral;
        const shaderUnsigned =
          pa.type === C1.UNSIGNED_INT || pa.type === C2.UNSIGNED_INT_VEC2 ||
          pa.type === C2.UNSIGNED_INT_VEC3 || pa.type === C2.UNSIGNED_INT_VEC4;
        const kind = a.genericKind ?? 'f';
        let bad = false;
        if (a.enabled) {
          if (shaderInt !== a.integer) {
            bad = true;
          } else if (shaderInt) {
            const arrUnsigned = a.type === C1.UNSIGNED_BYTE ||
              a.type === C1.UNSIGNED_SHORT || a.type === C1.UNSIGNED_INT;
            if (shaderUnsigned !== arrUnsigned) bad = true;
          }
        } else if (shaderInt) {
          if (kind === 'f') {
            bad = true;
          } else if (shaderUnsigned ? kind !== 'ui' : kind !== 'i') {
            bad = true;
          }
        } else if (kind !== 'f') {
          bad = true;
        }
        if (bad) {
          pushError(ctx, C1.INVALID_OPERATION);
          return false;
        }
      }
    }
  }
  // GLES 3.0 §2.11.6/§2.11.7 (CTS conformance2/rendering/
  // uniform-block-buffer-size.html): a draw is INVALID_OPERATION when any
  // ACTIVE uniform block of the current program is not backed by a buffer
  // range large enough to contain the block. Covers: nothing bound at the
  // block's binding point, a buffer with no data store (never bufferData'd),
  // a store smaller than the block, and a bindBufferRange whose size is
  // smaller than the block. The per-program blockIndex → binding-point map
  // lives in api/programs.ts's private WeakMap — read it back through the
  // public query (same pattern as the engine's block-store build in
  // executeDraw; safe on a linked program, pushes no errors).
  {
    const pm2 = prog._program;
    if (pm2) {
      const blocks = pm2.uniformBlocks ?? [];
      if (blocks.length > 0) {
        const gl2 = ctx as unknown as {
          getActiveUniformBlockParameter?: (program: WebGLProgram, uniformBlockIndex: GLuint, pname: GLenum) => unknown;
        };
        for (let bi = 0; bi < blocks.length; bi++) {
          const binding =
            gl2.getActiveUniformBlockParameter !== undefined
              ? ((gl2.getActiveUniformBlockParameter(prog, blocks[bi].index, C2.UNIFORM_BLOCK_BINDING) as number) || 0)
              : 0;
          const buf = binding < s.uniformBuffers.length ? s.uniformBuffers[binding] : null;
          const range = binding < s.uniformBufferRanges.length ? s.uniformBufferRanges[binding] : null;
          if (!buf || !buf._data || !range || range.size < blocks[bi].size) {
            pushError(ctx, C1.INVALID_OPERATION);
            return false;
          }
        }
      }
    }
  }
  const fb = resolveDrawTarget(ctx);
  if (!fb) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return false;
  }
  // GLES 2.0 §4.1.5 stencil front/back consistency (CTS conformance/misc/
  // webgl-specific-stencil-settings.html): when STENCIL_TEST is enabled AND
  // the draw target has a stencil buffer attached, a draw generates
  // INVALID_OPERATION if the front and back stencil func, ref, or valueMask
  // differ, or if the writeMasks differ. Only the effective 8-bit stencil
  // values matter (ref clamped to [0,255], masks masked with 0xff); the ops
  // (fail/depthFail/depthPass) may differ freely. Checked at draw time, AFTER
  // the FBO-completeness check (an incomplete target still wins with
  // INVALID_FRAMEBUFFER_OPERATION).
  if (s.caps.STENCIL_TEST && (fb.stencil || (fb.depth && fb.depth.stencilData))) {
    const f = s.stencil.front;
    const b = s.stencil.back;
    const ref8 = (v: number): number => (v < 0 ? 0 : v > 0xff ? 0xff : v) | 0;
    if (f.func !== b.func ||
        ref8(f.ref) !== ref8(b.ref) ||
        (f.valueMask & 0xff) !== (b.valueMask & 0xff) ||
        (f.writeMask & 0xff) !== (b.writeMask & 0xff)) {
      pushError(ctx, C1.INVALID_OPERATION);
      return false;
    }
  }
  return true;
}

/**
 * Validate a drawArrays-style call (shared by api/draw.ts drawArrays and the
 * WEBGL_multi_draw engine entry). Returns the assembled DrawRequest, or null
 * after pushing the first error (spec error order: mode → first/count/
 * instanceCount → common preconditions).
 */
export function validateDrawArrays(
  ctx: WebGLRenderingContext,
  mode: GLenum, first: GLint, count: GLsizei,
  instanceCount: GLsizei = 1,
): DrawRequest | null {
  // WebIDL conversion: GLenum is `unsigned long` → ToUint32 (undefined/NaN →
  // 0 = POINTS; -1 → 0xFFFFFFFF; valid enums unchanged). WebGL validation
  // then rejects values outside DRAW_MODES with INVALID_ENUM.
  mode = mode >>> 0;
  if (!DRAW_MODES.has(mode)) { pushError(ctx, C1.INVALID_ENUM); return null; }
  if (first < 0 || count < 0 || instanceCount < 0) { pushError(ctx, C1.INVALID_VALUE); return null; }
  if (!validateCommonDraw(ctx, mode)) return null;
  return { mode, count, instanceCount, firstOrOffset: first, indexed: false };
}

/** Options for validateDrawElements (instanced count + drawRangeElements range). */
export interface DrawElementsOpts {
  instanceCount?: GLsizei;
  range?: [GLuint, GLuint];
}

/**
 * Validate a drawElements-style call (shared by api/draw.ts drawElements and
 * the WEBGL_multi_draw engine entry). Error order per spec: mode → count →
 * type → offset → offset-multiple → (range) → element-array-buffer →
 * common preconditions. UNSIGNED_INT requires WebGL2 or OES_element_index_uint.
 */
export function validateDrawElements(
  ctx: WebGLRenderingContext,
  mode: GLenum, count: GLsizei, type: GLenum, offset: GLintptr,
  opts: DrawElementsOpts = {},
): DrawRequest | null {
  const instanceCount = opts.instanceCount ?? 1;
  // WebIDL conversion: GLenum is `unsigned long` → ToUint32 (undefined/NaN →
  // 0 = POINTS; -1 → 0xFFFFFFFF; valid enums unchanged). The WebGL validation
  // below then rejects values outside the accepted set with INVALID_ENUM.
  mode = mode >>> 0;
  type = type >>> 0;
  if (!DRAW_MODES.has(mode)) { pushError(ctx, C1.INVALID_ENUM); return null; }
  if (count < 0 || instanceCount < 0) { pushError(ctx, C1.INVALID_VALUE); return null; }
  const s = ctx._state;
  const uintOK = s.version === 2 || extSupported(ctx, 'OES_element_index_uint');
  if (type !== C1.UNSIGNED_BYTE && type !== C1.UNSIGNED_SHORT &&
      !(type === C1.UNSIGNED_INT && uintOK)) {
    pushError(ctx, C1.INVALID_ENUM);
    return null;
  }
  if (offset < 0) { pushError(ctx, C1.INVALID_VALUE); return null; }
  const ts = indexTypeSize(type);
  if (offset % ts !== 0) { pushError(ctx, C1.INVALID_OPERATION); return null; }
  const range = opts.range;
  if (range) {
    if (range[0] > range[1]) { pushError(ctx, C1.INVALID_VALUE); return null; }
    if (count > range[1] - range[0] + 1) { pushError(ctx, C1.INVALID_OPERATION); return null; }
  }
  const eb = s.vao.elementArrayBuffer;
  if (!eb) {
    pushError(ctx, C1.INVALID_OPERATION);
    return null;
  }
  // count=0: no elements are read — an unsized element buffer or an offset
  // past the end is NOT an error (CTS draw-elements-out-of-bounds.html
  // expects NO_ERROR for count=0 on a fresh/empty buffer and for count=0 with
  // an offset past the end).
  if (count > 0 && (!eb._data || offset + count * ts > eb._size)) {
    pushError(ctx, C1.INVALID_OPERATION);
    return null;
  }
  if (!validateCommonDraw(ctx, mode, true)) return null;
  const req: DrawRequest = {
    mode, count, instanceCount, firstOrOffset: offset, indexed: true, indexType: type,
  };
  if (range) req.range = range;
  return req;
}

/* ================================================================== */
/* WEBGL_multi_draw engine entries (api/draw.ts + extensions/misc.ts   */
/* delegate here)                                                      */
/* ================================================================== */

/**
 * Shared multi-draw execution: every subdraw was already validated
 * (validate-all-first contract) — run each via executeDraw. gl_DrawID is the
 * 0-based subdraw index i (constant per subdraw, across all vertices/instances).
 */
function runMultiSubdraws(
  ctx: WebGLRenderingContext,
  reqs: DrawRequest[],
): void {
  for (let i = 0; i < reqs.length; i++) {
    try {
      executeDraw(ctx, { ...reqs[i], drawId: i });
    } catch {
      pushError(ctx, C1.INVALID_OPERATION); // engine must not throw; guard anyway
    }
  }
}

/**
 * multiDrawArraysWEBGL engine: validate EVERY subdraw first (any invalid →
 * push the error + NO drawing at all), then execute each subdraw via
 * executeDraw. drawcount ≤ 0 → NO_ERROR no-op (caller guarantees ≥ 0).
 * firsts/counts are Int32Array/sequence (values are WebIDL longs);
 * firstsOffset/countsOffset are ELEMENT offsets into the lists (prototype
 * layer already checked offset + drawcount ≤ list.length).
 *
 * NOTE: `mode` is a parameter here (the objective's shorthand omitted it) —
 * executeDraw requires the per-subdraw mode, and it is validated against the
 * same DRAW_MODES table before any drawing.
 */
export function executeMultiDrawArrays(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  firsts: Int32Array | number[],
  firstsOffset: number,
  counts: Int32Array | number[],
  countsOffset: number,
  drawcount: number,
): void {
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  const reqs: DrawRequest[] = [];
  for (let i = 0; i < n; i++) {
    const req = validateDrawArrays(ctx, mode, firsts[firstsOffset + i], counts[countsOffset + i], 1);
    if (!req) return; // error already pushed; nothing drawn
    reqs.push(req);
  }
  runMultiSubdraws(ctx, reqs);
}

/**
 * multiDrawElementsWEBGL engine entry (same validate-all-first contract as
 * executeMultiDrawArrays). counts/offsets are Int32Array/sequence; offsets are
 * byte offsets into the element array buffer; countsOffset/offsetsOffset are
 * ELEMENT offsets into the lists.
 */
export function executeMultiDrawElements(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  counts: Int32Array | number[],
  countsOffset: number,
  type: GLenum,
  offsets: Int32Array | number[],
  offsetsOffset: number,
  drawcount: number,
): void {
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  const reqs: DrawRequest[] = [];
  for (let i = 0; i < n; i++) {
    const req = validateDrawElements(ctx, mode, counts[countsOffset + i], type, offsets[offsetsOffset + i]);
    if (!req) return; // error already pushed; nothing drawn
    reqs.push(req);
  }
  runMultiSubdraws(ctx, reqs);
}

/**
 * multiDrawArraysInstancedWEBGL engine entry (same validate-all-first
 * contract). Per-subdraw instanceCounts[instanceCountsOffset + i] flows into
 * validateDrawArrays and the executed instanceCount.
 */
export function executeMultiDrawArraysInstanced(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  firsts: Int32Array | number[],
  firstsOffset: number,
  counts: Int32Array | number[],
  countsOffset: number,
  instanceCounts: Int32Array | number[],
  instanceCountsOffset: number,
  drawcount: number,
): void {
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  const reqs: DrawRequest[] = [];
  for (let i = 0; i < n; i++) {
    const req = validateDrawArrays(
      ctx, mode,
      firsts[firstsOffset + i], counts[countsOffset + i],
      instanceCounts[instanceCountsOffset + i],
    );
    if (!req) return; // error already pushed; nothing drawn
    reqs.push(req);
  }
  runMultiSubdraws(ctx, reqs);
}

/**
 * multiDrawElementsInstancedWEBGL engine entry (same validate-all-first
 * contract). Per-subdraw instanceCounts[instanceCountsOffset + i] flows into
 * validateDrawElements and the executed instanceCount.
 */
export function executeMultiDrawElementsInstanced(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  counts: Int32Array | number[],
  countsOffset: number,
  type: GLenum,
  offsets: Int32Array | number[],
  offsetsOffset: number,
  instanceCounts: Int32Array | number[],
  instanceCountsOffset: number,
  drawcount: number,
): void {
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  const reqs: DrawRequest[] = [];
  for (let i = 0; i < n; i++) {
    const req = validateDrawElements(
      ctx, mode,
      counts[countsOffset + i], type, offsets[offsetsOffset + i],
      { instanceCount: instanceCounts[instanceCountsOffset + i] },
    );
    if (!req) return; // error already pushed; nothing drawn
    reqs.push(req);
  }
  runMultiSubdraws(ctx, reqs);
}

/**
 * multiDrawArraysInstancedBaseInstanceWEBGL engine entry (WEBGL_multi_draw_
 * instanced_base_vertex_base_instance; same validate-all-first contract).
 * baseInstances is a Uint32List (GLuint per extension.xml IDL) — each value is
 * converted with ToUint32 (`>>> 0`) and applied to the subdraw's instance
 * attribute fetch (element = baseInstance + floor(instanceId/divisor));
 * gl_InstanceID stays 0-based.
 */
export function executeMultiDrawArraysInstancedBaseInstance(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  firsts: Int32Array | number[],
  firstsOffset: number,
  counts: Int32Array | number[],
  countsOffset: number,
  instanceCounts: Int32Array | number[],
  instanceCountsOffset: number,
  baseInstances: Uint32Array | number[],
  baseInstancesOffset: number,
  drawcount: number,
): void {
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  const reqs: DrawRequest[] = [];
  for (let i = 0; i < n; i++) {
    const req = validateDrawArrays(
      ctx, mode,
      firsts[firstsOffset + i], counts[countsOffset + i],
      instanceCounts[instanceCountsOffset + i],
    );
    if (!req) return; // error already pushed; nothing drawn
    req.baseInstance = baseInstances[baseInstancesOffset + i] >>> 0;
    reqs.push(req);
  }
  runMultiSubdraws(ctx, reqs);
}

/**
 * multiDrawElementsInstancedBaseVertexBaseInstanceWEBGL engine entry
 * (WEBGL_multi_draw_instanced_base_vertex_base_instance; same
 * validate-all-first contract). baseVertices is an Int32List (GLint — may be
 * negative) and baseInstances a Uint32List (GLuint); baseVertex offsets the
 * element indices + gl_VertexID, baseInstance the instance attribute fetch.
 */
export function executeMultiDrawElementsInstancedBaseVertexBaseInstance(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  counts: Int32Array | number[],
  countsOffset: number,
  type: GLenum,
  offsets: Int32Array | number[],
  offsetsOffset: number,
  instanceCounts: Int32Array | number[],
  instanceCountsOffset: number,
  baseVertices: Int32Array | number[],
  baseVerticesOffset: number,
  baseInstances: Uint32Array | number[],
  baseInstancesOffset: number,
  drawcount: number,
): void {
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  const reqs: DrawRequest[] = [];
  for (let i = 0; i < n; i++) {
    const req = validateDrawElements(
      ctx, mode,
      counts[countsOffset + i], type, offsets[offsetsOffset + i],
      { instanceCount: instanceCounts[instanceCountsOffset + i] },
    );
    if (!req) return; // error already pushed; nothing drawn
    req.baseVertex = baseVertices[baseVerticesOffset + i] | 0;
    req.baseInstance = baseInstances[baseInstancesOffset + i] >>> 0;
    reqs.push(req);
  }
  runMultiSubdraws(ctx, reqs);
}