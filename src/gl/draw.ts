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
 *     [x,y,z,w, pointSize, varyings...] at record index i*count + j
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
import {
  computeVertexStride,
  RECORD_HEADER_FLOATS,
  draw as rasterDraw,
  clearColorSurface,
  clearDepthSurface,
  clearStencilSurface,
  blitColorSurface,
  blitDepthStencilSurface,
  getPackConverter,
  getFormat,
} from '../raster';
import type {
  DrawCall, FramebufferTarget, SamplerState, Surface, TextureImage, TextureUnitBinding,
  ColorMask, ScissorState,
} from '../raster';
import type { VertexExecCtx, AttribSource } from '../glsl/program';
import type { WebGLProgram } from './objects';
import type { ProgramModel } from './objects';
import type { WebGLBuffer, WebGLQuery, WebGLTransformFeedback } from './objects';

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
  scratch: Float32Array;   // codegen float scratch (program.scratchSize)
  intScratch: Int32Array;  // codegen int scratch
  zeroStore: Float32Array; // fallback uniform-block store (zero-filled)
  zeroIntStore: Int32Array;
  /** Stub-era fallback default-framebuffer color surface (until lifecycle lands). */
  fallbackColor: Surface | null;
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
      scratch: new Float32Array(64),
      intScratch: new Int32Array(64),
      zeroStore: new Float32Array(16384), // MAX_UNIFORM_BLOCK_SIZE / 4
      zeroIntStore: new Int32Array(16384),
      fallbackColor: null,
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
  return ((n + align - 1) / align) | 0 * align;
}

/** Extension gate that never throws (extension factories are stubs in Phase 2). */
export function extSupported(ctx: WebGLRenderingContext, name: string): boolean {
  try {
    return ctx.getExtension(name) !== null;
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

/** Present the drawing buffer when the draw/clear/blit touched the default FB. */
function presentIfDefault(ctx: WebGLRenderingContext): void {
  if (ctx._state.drawFramebuffer !== null) return;
  if (!ctx._presentSurface) return;
  try {
    ctx._presentSurface.present();
  } catch {
    // present adapter stub era — never leak to the page
  }
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
  const d = surf.data as { constructor: new (...a: never[]) => unknown };
  const f = surf.format;
  const u8 = d as Uint8Array;
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
  const d = surf.data as { constructor: new (...a: never[]) => unknown };
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
  const data = surf.data as { constructor: new (...a: never[]) => unknown };
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
}

const MAT2 = 0x8b5a, MAT3 = 0x8b5b, MAT4 = 0x8b5c;

function matrixDims(type: number): { cols: number; rows: number } | null {
  if (type === MAT2) return { cols: 2, rows: 2 };
  if (type === MAT3) return { cols: 3, rows: 3 };
  if (type === MAT4) return { cols: 4, rows: 4 };
  return null;
}

/**
 * Build the per-draw attribute plan: dense extraction into the scratch pools.
 * Returns the attribs array (indexed by location) — constant views for
 * disabled attribs. `indices` non-null for indexed draws.
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
  for (const pa of attrs) {
    const loc = pa.location;
    if (loc < 0 || loc >= maxAttribs) continue;
    const dims = matrixDims(pa.type);
    const cols = dims ? dims.cols : 1;
    for (let col = 0; col < cols; col++) {
      const l = loc + col;
      if (l >= maxAttribs) break;
      const a = vao.attribs[l];
      plans[l] = { source: 0, divisor: a.divisor, instanced: a.divisor > 0, present: true };
      if (!a.enabled || !a.buffer || !a.buffer._data) {
        // constant attribute
        if (pa.integral) {
          attribs[l] = (a.constantI ?? sc.emptyInt) as Int32Array;
        } else {
          attribs[l] = a.constantF as Float32Array;
        }
        continue;
      }
      const buf = a.buffer;
      const data = buf._data as ArrayBuffer;
      const typeSize = attribTypeSize(a.type);
      const stride = a.stride === 0 ? a.size * typeSize : a.stride;
      const comps = pa.components; // program component count (element stride in the dense layout)
      // number of elements to extract
      const elemCount = a.divisor > 0
        ? Math.ceil(req.instanceCount / a.divisor)
        : req.count;
      const colOffset = col * (dims ? dims.rows * typeSize : 0);
      const dv = new DataView(data);

      if (a.integer) {
        // raw integer path (vertexAttribIPointer)
        const unsigned = a.type === C1.UNSIGNED_BYTE || a.type === C1.UNSIGNED_SHORT || a.type === C1.UNSIGNED_INT;
        const need = elemCount * comps;
        ensurePool(sc, unsigned ? 'uintPool' : 'intPool', need);
        const dst = unsigned
          ? new Uint32Array(sc.uintPool.buffer, 0, need)
          : new Int32Array(sc.intPool.buffer, 0, need);
        for (let e = 0; e < elemCount; e++) {
          const element = a.divisor > 0 ? e : (indices ? indices[e] : req.firstOrOffset + e);
          const byteOff = a.offset + element * stride + colOffset;
          for (let c = 0; c < comps; c++) {
            let v = 0;
            if (c < a.size && byteOff + c * typeSize + typeSize <= data.byteLength) {
              v = readIntComponent(dv, a.type, byteOff + c * typeSize);
            }
            dst[e * comps + c] = c < a.size ? v : (c === 3 ? 1 : 0);
          }
        }
        attribs[l] = dst;
      } else {
        // float path with normalization
        const need = elemCount * comps;
        ensurePool(sc, 'floatPool', need);
        const dst = new Float32Array(sc.floatPool.buffer, 0, need);
        const normalized = a.normalized;
        for (let e = 0; e < elemCount; e++) {
          const element = a.divisor > 0 ? e : (indices ? indices[e] : req.firstOrOffset + e);
          const byteOff = a.offset + element * stride + colOffset;
          for (let c = 0; c < comps; c++) {
            let v = 0;
            if (c < a.size && byteOff + c * typeSize + typeSize <= data.byteLength) {
              v = readFloatComponent(dv, a.type, byteOff + c * typeSize, normalized);
            }
            dst[e * comps + c] = c < a.size ? v : (c === 3 ? 1 : 0);
          }
        }
        attribs[l] = dst;
      }
    }
  }
  // Attribs not used by the program: constants (never fetched).
  for (let l = 0; l < maxAttribs; l++) {
    if (!plans[l]) {
      const a = vao.attribs[l];
      plans[l] = { source: 0, divisor: a.divisor, instanced: a.divisor > 0, present: false };
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
    case C2.SAMPLER_2D_ARRAY: case C2.SAMPLER_2D_ARRAY_SHADOW:
    case C2.INT_SAMPLER_2D_ARRAY: case C2.UNSIGNED_INT_SAMPLER_2D_ARRAY:
      return 'texture2DArray';
    case C1.SAMPLER_CUBE: case C2.SAMPLER_CUBE_SHADOW:
    case C2.INT_SAMPLER_CUBE: case C2.UNSIGNED_INT_SAMPLER_CUBE:
      return 'textureCube';
    default:
      return 'texture2D';
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
      const unit = intStore[u.location] ?? 0;
      if (unit < 0 || unit >= numUnits) continue;
      const key = samplerTargetKey(u.type);
      const unitState = s.textureUnits[unit];
      const tex = unitState[key];
      if (!tex || !tex._image) continue;
      const img = tex._image as TextureImage;
      const st = effectiveSamplerState(tex, unitState.sampler);
      images[unit] = img;
      samplerStates[unit] = st;
      bindings[unit] = { img, state: st };
    }
  }
  return { images, samplerStates, bindings };
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
    colorMask[l] = s.colorMaskPerDrawBuffer.get(l) ?? s.colorMask;
    const db = l < s.drawBuffers.length ? s.drawBuffers[l] : C1.COLOR_ATTACHMENT0;
    drawBuffers[l] = db === C1.NONE ? -1 : db - C1.COLOR_ATTACHMENT0;
  }
  return { colorMask, drawBuffers };
}

function buildDrawCall(
  ctx: WebGLRenderingContext,
  pm: ProgramModel,
  req: DrawRequest,
  records: Float32Array,
  stride: number,
  fb: FramebufferTarget,
  env: { images: (TextureImage | null)[]; samplerStates: SamplerState[]; bindings: (TextureUnitBinding | null)[] },
): DrawCall {
  const s = ctx._state;
  const { colorMask, drawBuffers } = buildOutputMaps(ctx, pm);
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
  return {
    mode: req.mode,
    count: req.count,
    first: 0, // records packed from index 0 (instanced runs at i*count + j)
    instanceCount: req.instanceCount,
    vertices: records,
    vertexStride: stride,
    varyingsOffset: RECORD_HEADER_FLOATS,
    program: pm as never,
    fb,
    viewport: { x: s.viewport.x, y: s.viewport.y, w: s.viewport.w, h: s.viewport.h },
    depthRange: { near: s.depth.range[0], far: s.depth.range[1] },
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
  };
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

/** Vertex position (within the draw) of primitive p, vertex v. */
function primVertexIndex(mode: GLenum, p: number, v: number): number {
  switch (mode) {
    case C1.POINTS: return p;
    case C1.LINES: return p * 2 + v;
    case C1.LINE_STRIP: return p + v;
    case C1.LINE_LOOP: return v === 0 ? p : (p + 1) % (p + 1 + 1) === 0 ? 0 : p + 1;
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

function captureTransformFeedback(
  ctx: WebGLRenderingContext,
  tf: WebGLTransformFeedback,
  prog: WebGLProgram,
  pm: ProgramModel,
  records: Float32Array,
  stride: number,
  req: DrawRequest,
): void {
  const tfVarys = (pm as unknown as { transformFeedbackVaryings?: { name: string }[] }).transformFeedbackVaryings ?? [];
  if (tfVarys.length === 0) return;
  const separate = (prog._tfBufferMode === C2.SEPARATE_ATTRIBS);
  // record offset + components per captured varying
  const offsets: number[] = [];
  const comps: number[] = [];
  let off = RECORD_HEADER_FLOATS;
  for (const tv of tfVarys) {
    const vi = findVarying(pm.varyings, tv.name);
    const c = vi >= 0 ? pm.varyings[vi].components : 0;
    offsets.push(off);
    comps.push(c);
    off += c;
  }
  const totalComps = comps.reduce((a, c) => a + c, 0);
  if (totalComps === 0) return;
  const { primCount, vertsPerPrim } = primitiveInfo(req.mode, req.count);
  const buffers = tf._buffers;
  const ranges = tf._bufferRanges;
  let capturedVerts = 0;
  let overflow = false;

  outer: for (let i = 0; i < req.instanceCount && !overflow; i++) {
    for (let p = 0; p < primCount && !overflow; p++) {
      for (let v = 0; v < vertsPerPrim; v++) {
        const vIdx = primVertexIndex(req.mode, p, v);
        const recBase = (i * req.count + vIdx) * stride;
        for (let k = 0; k < tfVarys.length; k++) {
          const c = comps[k];
          if (c === 0) continue;
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
            const before = comps.slice(0, k).reduce((a, x) => a + x, 0);
            dstByte = range.offset + capturedVerts * totalComps * 4 + before * 4;
          }
          if (dstByte + c * 4 > range.offset + range.size) {
            overflow = true;
            break outer;
          }
          const src = records.subarray(recBase + offsets[k], recBase + offsets[k] + c);
          new Float32Array(buf._data, dstByte, c).set(src);
        }
        capturedVerts++;
      }
    }
  }
  tf._primitivesWritten += Math.floor(capturedVerts / vertsPerPrim);
}

/* ================================================================== */
/* executeDraw                                                         */
/* ================================================================== */

/**
 * Execute an assembled draw request: attribute fetch → vertex evaluation →
 * record packing → TF capture → rasterizer.draw (steps above).
 * @internal engine — called by api/draw.ts after validation.
 */
export function executeDraw(ctx: WebGLRenderingContext, req: DrawRequest): void {
  const s = ctx._state;

  // 1. Cheap preconditions (full validation in api/draw.ts).
  const prog = s.currentProgram;
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
    if (!eb || !eb._data || req.firstOrOffset % ts !== 0 || req.firstOrOffset + req.count * ts > eb._size) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    indices = t === C1.UNSIGNED_BYTE
      ? new Uint8Array(eb._data, req.firstOrOffset, req.count)
      : t === C1.UNSIGNED_SHORT
        ? new Uint16Array(eb._data, req.firstOrOffset, req.count)
        : new Uint32Array(eb._data, req.firstOrOffset, req.count);
  }

  const tf = s.transformFeedback;
  const tfActive = !!tf && tf._active && !tf._paused;

  // Active occlusion query (not counted during transform feedback).
  let activeQuery: WebGLQuery | null = null;
  if (!tfActive) {
    const q1 = s.activeQueries.ANY_SAMPLES_PASSED;
    const q2 = s.activeQueries.ANY_SAMPLES_PASSED_CONSERVATIVE;
    if (q1 && q1._active) activeQuery = q1;
    else if (q2 && q2._active) activeQuery = q2;
  }

  // 2. Resolve the draw target (INVALID_FRAMEBUFFER_OPERATION when incomplete).
  const fb = resolveDrawTarget(ctx);
  if (!fb) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return;
  }
  ensureCanvasSize(ctx);

  if (req.count === 0 || req.instanceCount === 0) return; // nothing to do, no error

  // 3. Attribute fetch (dense extraction).
  const sc = getScratch(ctx);
  const { attribs, plans } = buildAttribs(ctx, pm, req, indices);
  const ai = sc.attribIndices;
  if (ai.length < maxAttribs) {
    // attribIndices is fixed-size per context; maxAttribs never grows.
  }

  // 4. Vertex evaluation loop.
  const stride = computeVertexStride(pm.varyings ?? []);
  const totalVary = stride - RECORD_HEADER_FLOATS;
  const totalVerts = req.count * req.instanceCount;
  const needRecords = totalVerts * stride;
  if (sc.records.length < needRecords) {
    sc.records = new Float32Array(Math.max(needRecords, 64));
  }
  if (sc.outVaryings.length !== totalVary) {
    sc.outVaryings = new Float32Array(totalVary);
  }
  const scratchSize = (pm as unknown as { scratchSize?: number }).scratchSize ?? 0;
  const intScratchSize = (pm as unknown as { intScratchSize?: number }).intScratchSize ?? 0;
  if (sc.scratch.length < scratchSize) sc.scratch = new Float32Array(Math.max(scratchSize, 64));
  if (sc.intScratch.length < intScratchSize) sc.intScratch = new Int32Array(Math.max(intScratchSize, 64));

  // Uniform-block stores (per block INDEX, from the bound UBO at the block's binding).
  const blocks = pm.uniformBlocks ?? [];
  const blockStores: Float32Array[] = new Array(blocks.length);
  const blockIntStores: Int32Array[] = new Array(blocks.length);
  const bindingsMap = (prog as unknown as { _uniformBlockBindings?: Map<number, number> })._uniformBlockBindings;
  for (let bi = 0; bi < blocks.length; bi++) {
    const binding = bindingsMap ? (bindingsMap.get(blocks[bi].index) ?? 0) : 0;
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

  const vctx: VertexExecCtx = {
    attribs,
    attribIndices: ai,
    vertexId: 0,
    instanceId: 0,
    uniforms: floatStore,
    intUniforms: intStore,
    blockStores,
    blockIntStores,
    textures: env.images,
    samplerStates: env.samplerStates,
    scratch: sc.scratch,
    intScratch: sc.intScratch,
    out: { position: sc.outPosition, pointSize: 0, varyings: sc.outVaryings },
  };

  // Precompute per-loc loops for the inner hot path.
  const vertexLocs: number[] = [];
  const instancedLocs: { loc: number; divisor: number }[] = [];
  for (let l = 0; l < maxAttribs; l++) {
    const p = plans[l];
    if (!p.present) continue;
    if (p.instanced) instancedLocs.push({ loc: l, divisor: p.divisor });
    else vertexLocs.push(l);
  }

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
      vctx.vertexId = req.indexed ? indices![j] : first + j;
      vctx.instanceId = i;
      for (let k = 0; k < vertexLocs.length; k++) ai[vertexLocs[k]] = j;
      run(vctx);
      const base = r * stride;
      records[base] = pos[0];
      records[base + 1] = pos[1];
      records[base + 2] = pos[2];
      records[base + 3] = pos[3];
      records[base + 4] = vctx.out.pointSize;
      if (totalVary > 0) {
        const vb = base + RECORD_HEADER_FLOATS;
        for (let v = 0; v < totalVary; v++) records[vb + v] = sc.outVaryings[v];
      }
    }
  }

  // 5. Transform feedback capture (bypasses the rasterizer).
  if (tfActive) {
    captureTransformFeedback(ctx, tf!, prog, pm, records, stride, req);
    return; // TF draws never present (no FB writes) and never count queries
  }

  // 6. Rasterize.
  const dc = buildDrawCall(ctx, pm, req, records, stride, fb, env);
  if (activeQuery) {
    (dc as unknown as { sampleCountRef: { value: number } }).sampleCountRef = { value: 0 };
  }
  try {
    rasterDraw(dc);
    if (activeQuery) {
      const ref = (dc as unknown as { sampleCountRef?: { value: number } }).sampleCountRef;
      if (ref) {
        activeQuery._result += ref.value;
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
  const fb = resolveDrawTarget(ctx);
  if (!fb) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return;
  }
  ensureCanvasSize(ctx);
  const scissor = scissorState(ctx);

  if (mask & C1.COLOR_BUFFER_BIT) {
    for (let d = 0; d < s.drawBuffers.length; d++) {
      const db = s.drawBuffers[d];
      if (db === C1.NONE) continue;
      const idx = db - C1.COLOR_ATTACHMENT0;
      const surf = fb.color[idx];
      if (!surf) continue;
      const cm = s.colorMaskPerDrawBuffer.get(d) ?? s.colorMask;
      if (!cm[0] && !cm[1] && !cm[2] && !cm[3]) continue;
      const [r, g, b, a] = s.clearColor;
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
    case C2.FLOAT_32_UNSIGNED_INT_24_8_REV: return 8;
    default: return comps * 4; // UNSIGNED_INT, INT, FLOAT, UNSIGNED_INT_24_8
  }
}

/** Local pack conversion (replace with raster getPackConverter when it lands). */
function makeLocalPack(format: GLenum, type: GLenum): ((src: ArrayBufferView, srcOff: number, dst: ArrayBufferView, dstOff: number) => void) | null {
  const tmp = new Float32Array(4);
  const u8 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
  switch (type) {
    case C1.UNSIGNED_BYTE: {
      const comps = format === C1.RGBA ? 4 : format === C1.RGB ? 3 : format === C1.LUMINANCE_ALPHA ? 2 : 1;
      return (src, so, dst, d) => {
        decodeSurfaceTexel(src as Surface, so, tmp);
        const d8 = dst as Uint8Array;
        if (comps === 4) { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; d8[d + 1] = (u8(tmp[1]) * 255 + 0.5) | 0; d8[d + 2] = (u8(tmp[2]) * 255 + 0.5) | 0; d8[d + 3] = (u8(tmp[3]) * 255 + 0.5) | 0; }
        else if (comps === 3) { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; d8[d + 1] = (u8(tmp[1]) * 255 + 0.5) | 0; d8[d + 2] = (u8(tmp[2]) * 255 + 0.5) | 0; }
        else if (comps === 2) { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; d8[d + 1] = (u8(tmp[3]) * 255 + 0.5) | 0; }
        else { d8[d] = (u8(tmp[0]) * 255 + 0.5) | 0; }
      };
    }
    case C1.UNSIGNED_SHORT_5_6_5:
      return (src, so, dst, d) => {
        decodeSurfaceTexel(src as Surface, so, tmp);
        const v = ((u8(tmp[0]) * 31 + 0.5) | 0) << 11 | ((u8(tmp[1]) * 63 + 0.5) | 0) << 5 | (u8(tmp[2]) * 31 + 0.5) | 0;
        const d8 = dst as Uint8Array;
        d8[d] = v & 0xff; d8[d + 1] = (v >> 8) & 0xff;
      };
    case C1.UNSIGNED_SHORT_4_4_4_4:
      return (src, so, dst, d) => {
        decodeSurfaceTexel(src as Surface, so, tmp);
        const v = ((u8(tmp[0]) * 15 + 0.5) | 0) << 12 | ((u8(tmp[1]) * 15 + 0.5) | 0) << 8 | ((u8(tmp[2]) * 15 + 0.5) | 0) << 4 | (u8(tmp[3]) * 15 + 0.5) | 0;
        const d8 = dst as Uint8Array;
        d8[d] = v & 0xff; d8[d + 1] = (v >> 8) & 0xff;
      };
    case C1.UNSIGNED_SHORT_5_5_5_1:
      return (src, so, dst, d) => {
        decodeSurfaceTexel(src as Surface, so, tmp);
        const v = ((u8(tmp[0]) * 31 + 0.5) | 0) << 11 | ((u8(tmp[1]) * 31 + 0.5) | 0) << 6 | ((u8(tmp[2]) * 31 + 0.5) | 0) << 1 | (u8(tmp[3]) > 0.5 ? 1 : 0);
        const d8 = dst as Uint8Array;
        d8[d] = v & 0xff; d8[d + 1] = (v >> 8) & 0xff;
      };
    case C1.FLOAT: {
      const comps = format === C1.RGBA ? 4 : format === C1.RGB ? 3 : format === C1.LUMINANCE_ALPHA ? 2 : 1;
      return (src, so, dst, d) => {
        decodeSurfaceTexel(src as Surface, so, tmp);
        const df = dst as Float32Array;
        for (let c = 0; c < comps; c++) df[(d >> 2) + c] = tmp[c];
      };
    }
    case C1.UNSIGNED_INT: // DEPTH_COMPONENT / integer formats
      if (format === C1.DEPTH_COMPONENT) {
        return (src, so, dst, d) => {
          decodeSurfaceTexel(src as Surface, so, tmp);
          const dv = dst as DataView;
          dv.setUint32(d, Math.min(0xffffffff, Math.max(0, Math.round(tmp[0] * 0xffffffff))), true);
        };
      }
      return (src, so, dst, d) => {
        decodeSurfaceTexel(src as Surface, so, tmp);
        (dst as Uint32Array)[d >> 2] = tmp[0] >>> 0;
      };
    case C1.UNSIGNED_SHORT: // DEPTH_COMPONENT
      return (src, so, dst, d) => {
        decodeSurfaceTexel(src as Surface, so, tmp);
        (dst as Uint16Array)[d >> 1] = Math.min(0xffff, Math.max(0, Math.round(tmp[0] * 0xffff)));
      };
    case C2.UNSIGNED_INT_24_8:
      return (src, so, dst, d) => {
        const surf = src as Surface;
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
): void {
  const s = ctx._state;
  if (width === 0 || height === 0) return;
  const surf = resolveReadColor(ctx);
  if (!surf) {
    pushError(ctx, C1.INVALID_OPERATION);
    return;
  }
  // Pack converter: raster's when available, else local.
  let conv: ((src: ArrayBufferView, srcOff: number, dst: ArrayBufferView, dstOff: number) => void) | null = null;
  try {
    const rc = getPackConverter(surf.format, format, type);
    if (rc) {
      conv = (src, so, dst, d) => rc.convert(src, so, dst, d);
    }
  } catch {
    conv = null;
  }
  if (!conv) {
    conv = makeLocalPack(format, type);
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
  let dstBase: number;
  let dstView: ArrayBufferView;
  const packBuf = s.pixelPackBuffer;
  if (packBuf && packBuf._data) {
    dstView = new Uint8Array(packBuf._data);
    dstBase = (pixels as unknown as number) + pack.skipRows * rowStride + pack.skipPixels * bpp;
  } else {
    dstView = pixels;
    dstBase = pixels.byteOffset + pack.skipRows * rowStride + pack.skipPixels * bpp;
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

  if (mask & C1.COLOR_BUFFER_BIT) {
    const src = resolveReadColor(ctx);
    const fb = resolveDrawTarget(ctx);
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
  const d = surf.data as { constructor: new (...a: never[]) => unknown };
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
  depth?: number, stencil?: number,
): void {
  const s = ctx._state;
  const fb = resolveDrawTarget(ctx);
  if (!fb) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return;
  }
  ensureCanvasSize(ctx);
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
    if (!cm[0] && !cm[1] && !cm[2] && !cm[3]) return;
    if (values instanceof Int32Array || values instanceof Uint32Array) {
      clearColorIntLocal(surf, values, cm, scissor);
    } else {
      const v = values as Float32Array;
      try {
        clearColorSurface(surf, v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 1, scissor, cm);
      } catch {
        clearColorLocal(surf, v[0] ?? 0, v[1] ?? 0, v[2] ?? 0, v[3] ?? 1, scissor, cm);
      }
    }
  } else if (buffer === C2.DEPTH) {
    if (!fb.depth) {
      pushError(ctx, C1.INVALID_OPERATION);
      return;
    }
    if (!s.depth.mask || !values) return;
    try {
      clearDepthSurface(fb.depth, values[0] ?? 0, scissor, true);
    } catch {
      clearDepthLocal(fb.depth, values[0] ?? 0, scissor);
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
      clearStencilSurface(stencilSurf, values[0] ?? 0, scissor, writeMask);
    } catch {
      clearStencilLocal(stencilSurf, values[0] ?? 0, scissor, writeMask);
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
 * (INVALID_FRAMEBUFFER_OPERATION). The engine's executeDraw re-checks these as
 * a safety net; the api layer checks first so error ordering is exact.
 */
function validateCommonDraw(ctx: WebGLRenderingContext, mode: GLenum): boolean {
  const s = ctx._state;
  const prog = s.currentProgram;
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
  const vao = s.vao;
  const maxAttribs = s.limits.MAX_VERTEX_ATTRIBS;
  for (let loc = 0; loc < maxAttribs; loc++) {
    const a = vao.attribs[loc];
    if (a.enabled && !a.buffer) {
      pushError(ctx, C1.INVALID_OPERATION);
      return false;
    }
  }
  if (!resolveDrawTarget(ctx)) {
    pushError(ctx, C1.INVALID_FRAMEBUFFER_OPERATION);
    return false;
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
  if (!eb || !eb._data || offset + count * ts > eb._size) {
    pushError(ctx, C1.INVALID_OPERATION);
    return null;
  }
  if (!validateCommonDraw(ctx, mode)) return null;
  const req: DrawRequest = {
    mode, count, instanceCount, firstOrOffset: offset, indexed: true, indexType: type,
  };
  if (range) req.range = range;
  return req;
}

/* ================================================================== */
/* WEBGL_multi_draw engine entries (extensions/misc.ts delegates here)  */
/* ================================================================== */

/**
 * multiDrawArraysWEBGL engine: validate EVERY subdraw first (any invalid →
 * push the error + NO drawing at all), then execute each subdraw via
 * executeDraw. drawcount ≤ 0 → NO_ERROR no-op. firsts/counts are
 * Int32Array/sequence (values are WebIDL longs).
 *
 * NOTE: `mode` is a parameter here (the objective's shorthand omitted it) —
 * executeDraw requires the per-subdraw mode, and the extension validates it
 * against the same DRAW_MODES table before any drawing.
 */
export function executeMultiDrawArrays(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  firsts: Int32Array | number[],
  counts: Int32Array | number[],
  drawcount: number,
): void {
  if (ctx._isLost) { pushError(ctx, C1.CONTEXT_LOST_WEBGL); return; }
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  for (let i = 0; i < n; i++) {
    if (!validateDrawArrays(ctx, mode, firsts[i], counts[i], 1)) return;
  }
  for (let i = 0; i < n; i++) {
    try {
      executeDraw(ctx, { mode, count: counts[i], instanceCount: 1, firstOrOffset: firsts[i], indexed: false });
    } catch {
      pushError(ctx, C1.INVALID_OPERATION); // engine must not throw; guard anyway
    }
  }
}

/**
 * multiDrawElementsWEBGL engine entry (same validate-all-first contract as
 * executeMultiDrawArrays). counts/offsets are Int32Array/sequence; offsets are
 * byte offsets into the element array buffer.
 */
export function executeMultiDrawElements(
  ctx: WebGLRenderingContext,
  mode: GLenum,
  counts: Int32Array | number[],
  type: GLenum,
  offsets: Int32Array | number[],
  drawcount: number,
): void {
  if (ctx._isLost) { pushError(ctx, C1.CONTEXT_LOST_WEBGL); return; }
  const n = drawcount | 0;
  if (n <= 0) return; // NO_ERROR no-op
  for (let i = 0; i < n; i++) {
    if (!validateDrawElements(ctx, mode, counts[i], type, offsets[i])) return;
  }
  for (let i = 0; i < n; i++) {
    try {
      executeDraw(ctx, { mode, count: counts[i], instanceCount: 1, firstOrOffset: offsets[i], indexed: true, indexType: type });
    } catch {
      pushError(ctx, C1.INVALID_OPERATION); // engine must not throw; guard anyway
    }
  }
}
