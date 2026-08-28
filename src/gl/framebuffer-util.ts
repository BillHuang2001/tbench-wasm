/**
 * src/gl/framebuffer-util.ts — shared framebuffer resolution (contract pin).
 *
 * THE single place that resolves WebGLFramebuffer attachments (and the default
 * framebuffer) into raster `FramebufferTarget`/`Surface` objects. Owned and
 * implemented by the framebuffers API module (api/framebuffers.ts); consumed by
 * the draw engine (draw.ts), getters (getters.ts), and teximage's copyTex*.
 *
 * Surface resolution rules:
 *  - Renderbuffer attachment → `rb._surface` (allocated at renderbufferStorage).
 *  - Texture attachment → the level's view: 2D → `_image.levels[level].data[0]`;
 *    cube → `data[CUBE_FACE_TO_INDEX[face]]`; depth-stencil textures expose the
 *    split stencil plane via `levels[level].stencilData`.
 *  - The default framebuffer (drawFramebuffer === null) → `ctx._defaultFB`.
 *
 * Completeness (`checkFramebufferStatus`) implements every rule of the WebGL
 * spec (attachment presence, format/attachment-point compatibility, dimension
 * consistency, cube completeness, sample-count consistency) and is consulted
 * both by the API and by the draw pipeline (INVALID_FRAMEBUFFER_OPERATION).
 *
 * Completeness check order (GLES 3.0 §4.4.5, adapted to WebGL1/2):
 *   1. no attachments                    → FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT
 *   2. sample-count consistency          → FRAMEBUFFER_INCOMPLETE_MULTISAMPLE
 *   3. legal combination of images       → FRAMEBUFFER_UNSUPPORTED
 *      (WebGL1: DEPTH+STENCIL / DEPTH+DS / STENCIL+DS concurrent attachment
 *       points — WebGL1 spec "Framebuffer Object Attachments" hard rule;
 *       WebGL2 (and WebGL1+WEBGL_draw_buffers): same image on more than one
 *       color attachment point; both versions: depth and stencil attachments
 *       that are not the same image.)
 *   4. dimension consistency             → FRAMEBUFFER_INCOMPLETE_DIMENSIONS
 *   5. per-attachment checks             → FRAMEBUFFER_INCOMPLETE_ATTACHMENT
 *      (undefined texture level, cube map not cube-complete, zero-sized
 *       renderbuffer, unallocated renderbuffer storage, and the attached
 *       image's internal format not renderable at its attachment point)
 *   6. layered-attachment consistency    → FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS
 *
 * Renderability tables (documented decisions — CTS is the primary gate):
 *  - Format/attachment-point mismatch (e.g. DEPTH_COMPONENT16 at
 *    COLOR_ATTACHMENT0 or STENCIL_ATTACHMENT) is FRAMEBUFFER_INCOMPLETE_ATTACHMENT,
 *    per the CTS hard expectations in conformance(+2)/renderbuffers/
 *    framebuffer-object-attachment.html. The objective text suggested
 *    UNSUPPORTED for non-renderable formats; CTS demands INCOMPLETE_ATTACHMENT
 *    there, so that wins. The ONE exception: RGB8 attached as color in WebGL2
 *    returns FRAMEBUFFER_UNSUPPORTED (objective-mandated; CTS is tolerant —
 *    read-pixels-from-fbo-test.html skips when the FBO is incomplete).
 *  - WebGL1 color-renderable: RGBA4, RGB565, RGB5_A1, RGBA8 (+ unsized RGBA/RGB,
 *    which resolve to the default sized renderable formats), + RGBA16F
 *    (EXT_color_buffer_half_float), + SRGB8_ALPHA8 (EXT_sRGB). LUMINANCE,
 *    LUMINANCE_ALPHA, ALPHA and sized RGB8 are NOT renderable
 *    (format-filterable-renderable.html, copy-tex-image-2d-formats.html hard
 *    expectations). WebGL1 depth/stencil points are STRICT per the
 *    WebGL1 spec: DEPTH_ATTACHMENT accepts only DEPTH_COMPONENT16 /
 *    DEPTH_COMPONENT (WEBGL_depth_texture texture), STENCIL_ATTACHMENT only
 *    STENCIL_INDEX8, DEPTH_STENCIL_ATTACHMENT only DEPTH_STENCIL /
 *    DEPTH24_STENCIL8 (texture storage) — DEPTH_STENCIL at DEPTH_ATTACHMENT is
 *    INCOMPLETE_ATTACHMENT (CTS hard).
 *  - WebGL2 color-renderable (core): RGBA4, RGB565, RGB5_A1, RGBA8, RGB10_A2,
 *    RGB10_A2UI, SRGB8_ALPHA8, R8/R8I/R8UI, R16I/R16UI, R32I/R32UI,
 *    RG8/RG8I/RG8UI, RG16I/RG16UI, RG32I/RG32UI, RGBA8I/RGBA8UI,
 *    RGBA16I/RGBA16UI, RGBA32I/RGBA32UI, plus the GLES3 RGB integer formats
 *    (RGB8I/RGB8UI/RGB16I/RGB16UI/RGB32I/RGB32UI — GLES3 Table 3.13; WebGL2
 *    only removes RGB8 among the unorm formats), plus unsized RGBA (resolves to
 *    RGBA8 — occlusion-query-scissor.html attaches texImage2D(RGBA) textures
 *    and renders to them). RGB8 is NOT color-renderable in WebGL2 (objective;
 *    matches the known WebGL2 deviation from GLES3) — see checkAttachment for
 *    the texture/renderbuffer split; unsized RGB (→ RGB8) behaves like RGB8.
 *    EXT_color_buffer_float adds R16F/RG16F/RGBA16F/R32F/RG32F/RGBA32F/
 *    R11F_G11F_B10F; EXT_texture_norm16 adds R16_EXT/RG16_EXT/RGBA16_EXT;
 *    WEBGL_render_shared_exponent adds RGB9_E5. Snorm formats are not
 *    renderable (EXT_render_snorm is 'null' status).
 *  - WebGL2 depth-renderable: DEPTH_COMPONENT16/24/32F, DEPTH24_STENCIL8,
 *    DEPTH32F_STENCIL8 (+DEPTH_STENCIL unsized, used by CTS W2 tests);
 *    stencil-renderable: STENCIL_INDEX8 + the depth-stencil formats.
 *  - Texture attachment sample count: texture._msaaSamples only counts when
 *    WEBGL_multisampled_render_to_texture is available (its factory is the
 *    only setter); renderbuffers report rb._samples. All attachments must
 *    agree, else FRAMEBUFFER_INCOMPLETE_MULTISAMPLE.
 *  - There is NO draw-buffer coverage rule: per GLES3, fragments to draw
 *    buffers naming an unattached attachment point are discarded, and
 *    FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER applies only to the DEFAULT
 *    framebuffer. FBOs must be COMPLETE even when drawBuffers names a missing
 *    attachment (rendering-sampling-feedback-loop.html expects COMPLETE with
 *    drawBuffers=[ATT0,ATT1] and only ATT0 attached; no CTS page references
 *    INCOMPLETE_DRAW_BUFFER).
 *  - Layered rule (WebGL2): WebGL2 exposes no whole-level attachment API
 *    (framebufferTextureLayer attaches a single layer; framebufferTexture3D
 *    does not exist), so every attachment is non-layered and the rule is
 *    vacuous. Implemented anyway for spec completeness; treating 3D/2D_ARRAY
 *    attachments as "layered" would break framebuffer-render-to-layer.html
 *    (3D texture + depth renderbuffer must be COMPLETE).
 */

import type { WebGLRenderingContext } from './webgl1';
import type { WebGLFramebuffer, FramebufferAttachment } from './objects';
import type { FramebufferTarget, Surface, PixelFormatInfo } from '../raster';
import { getFormat } from '../raster';
import type { GLenum } from './types';
import { C, C1, C2 } from './constants';

/* ================================================================== */
/* Constants                                                           */
/* ================================================================== */

const COLOR_ATTACHMENT0 = 0x8ce0;
const DEPTH_ATTACHMENT = 0x8d00;
const STENCIL_ATTACHMENT = 0x8d20;
const DEPTH_STENCIL_ATTACHMENT = 0x821a;
const NONE = 0x0000;
const CUBE_POSITIVE_X = 0x8515;
const CUBE_NEGATIVE_Z = 0x851a;

/* ------------------------------------------------------------------ */
/* Renderability tables (see header for the documented decisions)      */
/* ------------------------------------------------------------------ */

/** WebGL1 color-renderable formats. Sized: RGBA4/RGB565/RGB5_A1/RGBA8; the
 *  unsized RGBA/RGB keys are kept because they resolve to the default sized
 *  renderable formats (RGBA4/RGB565) and CTS attaches such textures expecting
 *  FRAMEBUFFER_COMPLETE (texture-attachment-formats.html, copy-tex-image-2d-formats.html).
 *  LUMINANCE/LUMINANCE_ALPHA/ALPHA and sized RGB8 are NOT renderable
 *  (format-filterable-renderable.html, copy-tex-image-2d-formats.html).
 *  Extension formats are additionally gated in isColorRenderable. */
const W1_COLOR_RENDERABLE = new Set<GLenum>([
  0x8056 /* RGBA4 */, 0x8d62 /* RGB565 */, 0x8057 /* RGB5_A1 */,
  0x8058 /* RGBA8 */, 0x1908 /* RGBA (unsized → RGBA4/RGBA8) */, 0x1907 /* RGB (unsized → RGB565) */,
  0x8c43 /* SRGB8_ALPHA8 (EXT_sRGB) */, 0x881a /* RGBA16F (EXT_color_buffer_half_float) */,
]);

/** WebGL2 color-renderable core (GLES3 Table 3.13 minus RGB8; RGB10_A2UI included). */
const W2_COLOR_RENDERABLE_CORE = new Set<GLenum>([
  0x8056 /* RGBA4 */, 0x8d62 /* RGB565 */, 0x8057 /* RGB5_A1 */,
  0x8058 /* RGBA8 */, 0x1908 /* RGBA (unsized → RGBA8) */,
  0x8059 /* RGB10_A2 */, 0x906f /* RGB10_A2UI */,
  0x8c43 /* SRGB8_ALPHA8 */,
  0x8229 /* R8 */, 0x8231 /* R8I */, 0x8232 /* R8UI */,
  0x8233 /* R16I */, 0x8234 /* R16UI */, 0x8235 /* R32I */, 0x8236 /* R32UI */,
  0x822b /* RG8 */, 0x8237 /* RG8I */, 0x8238 /* RG8UI */,
  0x8239 /* RG16I */, 0x823a /* RG16UI */, 0x823b /* RG32I */, 0x823c /* RG32UI */,
  C2.RGBA8I /* 0x8d8e — WebGL2 value (differs from GLES3's 0x8d8a) */, 0x8d7c /* RGBA8UI */,
  0x8d88 /* RGBA16I */, 0x8d76 /* RGBA16UI */,
  0x8d82 /* RGBA32I */, 0x8d70 /* RGBA32UI */,
  // GLES3 RGB integer formats (WebGL2 only removes RGB8 among unorm formats)
  C2.RGB8I /* 0x8d8f — WebGL2 value (differs from GLES3's 0x8d8b) */, 0x8d7d /* RGB8UI */,
  0x8d89 /* RGB16I */, 0x8d77 /* RGB16UI */,
  0x8d83 /* RGB32I */, 0x8d71 /* RGB32UI */,
]);

/** WebGL2 color-renderable via EXT_color_buffer_float (also RGBA16F via EXT_color_buffer_half_float). */
const W2_COLOR_RENDERABLE_EXT_FLOAT = new Set<GLenum>([
  0x822d /* R16F */, 0x822f /* RG16F */, 0x881a /* RGBA16F */,
  0x822e /* R32F */, 0x8230 /* RG32F */, 0x8814 /* RGBA32F */,
  0x8c3a /* R11F_G11F_B10F */,
]);

/** WebGL2 color-renderable via EXT_texture_norm16. */
const W2_COLOR_RENDERABLE_EXT_NORM16 = new Set<GLenum>([
  0x822a /* R16_EXT */, 0x822c /* RG16_EXT */, 0x805b /* RGBA16_EXT */,
]);

/** WebGL1 depth-renderable (renderbuffers + WEBGL_depth_texture textures). */
const W1_DEPTH_RENDERABLE = new Set<GLenum>([
  0x81a5 /* DEPTH_COMPONENT16 */, 0x1902 /* DEPTH_COMPONENT */,
]);

/** WebGL1 stencil-renderable (renderbuffers only). */
const W1_STENCIL_RENDERABLE = new Set<GLenum>([0x8d48 /* STENCIL_INDEX8 */]);

/** WebGL1 DEPTH_STENCIL_ATTACHMENT formats (independent point per WebGL1 spec). */
const W1_DS_RENDERABLE = new Set<GLenum>([
  0x84f9 /* DEPTH_STENCIL */, 0x88f0 /* DEPTH24_STENCIL8 (texture storage) */,
]);

/** WebGL2 depth-renderable (DEPTH_STENCIL unsized accepted — used by CTS W2 tests). */
const W2_DEPTH_RENDERABLE = new Set<GLenum>([
  0x81a5 /* DEPTH_COMPONENT16 */, 0x81a6 /* DEPTH_COMPONENT24 */,
  0x8cac /* DEPTH_COMPONENT32F */, 0x88f0 /* DEPTH24_STENCIL8 */,
  0x8cad /* DEPTH32F_STENCIL8 */, 0x84f9 /* DEPTH_STENCIL */,
]);

/** WebGL2 stencil-renderable (STENCIL_INDEX8 renderbuffer + depth-stencil formats). */
const W2_STENCIL_RENDERABLE = new Set<GLenum>([
  0x8d48 /* STENCIL_INDEX8 */, 0x88f0 /* DEPTH24_STENCIL8 */,
  0x8cad /* DEPTH32F_STENCIL8 */, 0x84f9 /* DEPTH_STENCIL */,
]);

/**
 * Extension availability (engine path — same semantics as ctx.getExtension).
 * Checks the ENABLED-extensions cache (`ctx._extensions`, populated by
 * getExtensionObject) instead of calling getExtension: calling it here would
 * cache and thereby SELF-ENABLE the extension, observable as e.g. float
 * renderbuffers accepted before EXT_color_buffer_half_float is requested.
 */
function hasExtension(ctx: WebGLRenderingContext, name: string): boolean {
  try {
    return ctx._extensions.has(name);
  } catch {
    return false;
  }
}

/** CUBE_FACE_TO_INDEX: face enum (TEXTURE_CUBE_MAP_POSITIVE_X..NEGATIVE_Z) → data index. */
function cubeFaceIndex(face: GLenum): number {
  return face - CUBE_POSITIVE_X;
}

function isCubeFace(face: GLenum): boolean {
  return face >= CUBE_POSITIVE_X && face <= CUBE_NEGATIVE_Z;
}

/** True when two attachment entries reference the identical image. */
function sameImage(a: FramebufferAttachment, b: FramebufferAttachment): boolean {
  if (a.type === 'renderbuffer' && b.type === 'renderbuffer') {
    return a.renderbuffer === b.renderbuffer;
  }
  if (a.type === 'texture' && b.type === 'texture') {
    return (
      a.texture === b.texture &&
      a.level === b.level &&
      a.face === b.face &&
      a.layer === b.layer
    );
  }
  return false; // different attachment types
}

/** Sample count of one attachment (0 for single-sampled textures). */
function attachmentSamples(ctx: WebGLRenderingContext, entry: FramebufferAttachment): number {
  if (entry.type === 'renderbuffer') return entry.renderbuffer._samples;
  // WEBGL_multisampled_render_to_texture is the only setter of _msaaSamples.
  return ctx._version === 2 && hasExtension(ctx, 'WEBGL_multisampled_render_to_texture')
    ? entry.texture._msaaSamples
    : 0;
}

/** Dimensions of one attachment (null when the image is not resolvable). */
function attachmentDims(entry: FramebufferAttachment): { width: number; height: number } | null {
  if (entry.type === 'renderbuffer') {
    const surf = entry.renderbuffer._surface;
    return surf ? { width: surf.width, height: surf.height } : null;
  }
  const image = entry.texture._image;
  if (!image) return null;
  const lvl = image.levels[entry.level];
  return lvl ? { width: lvl.width, height: lvl.height } : null;
}

/** True for a whole-level (layered) texture attachment. WebGL2 cannot create
 *  one (framebufferTextureLayer attaches a single layer) — always false. */
function isLayeredAttachment(_entry: FramebufferAttachment): boolean {
  return false;
}

/* ------------------------------------------------------------------ */
/* Renderability per attachment point                                  */
/* ------------------------------------------------------------------ */

function isColorRenderable(ctx: WebGLRenderingContext, format: GLenum): boolean {
  if (ctx._version === 2) {
    if (W2_COLOR_RENDERABLE_CORE.has(format)) return true;
    if (W2_COLOR_RENDERABLE_EXT_FLOAT.has(format)) {
      if (hasExtension(ctx, 'EXT_color_buffer_float')) return true;
      // EXT_color_buffer_half_float (WebGL2) enables the 16F formats only.
      return (format === 0x822d /* R16F */ || format === 0x822f /* RG16F */ || format === 0x881a /* RGBA16F */) &&
        hasExtension(ctx, 'EXT_color_buffer_half_float');
    }
    if (W2_COLOR_RENDERABLE_EXT_NORM16.has(format)) return hasExtension(ctx, 'EXT_texture_norm16');
    if (format === 0x8c3d /* RGB9_E5 */) return hasExtension(ctx, 'WEBGL_render_shared_exponent');
    return false;
  }
  if (!W1_COLOR_RENDERABLE.has(format)) return false;
  switch (format) {
    case 0x8c43 /* SRGB8_ALPHA8 */: return hasExtension(ctx, 'EXT_sRGB');
    case 0x881a /* RGBA16F */:
    case 0x881b /* RGB16F (renderability optional; report supported) */:
      return hasExtension(ctx, 'EXT_color_buffer_half_float');
    default: return true;
  }
}

function isDepthRenderable(ctx: WebGLRenderingContext, format: GLenum): boolean {
  return ctx._version === 2 ? W2_DEPTH_RENDERABLE.has(format) : W1_DEPTH_RENDERABLE.has(format);
}

function isStencilRenderable(ctx: WebGLRenderingContext, format: GLenum): boolean {
  return ctx._version === 2 ? W2_STENCIL_RENDERABLE.has(format) : W1_STENCIL_RENDERABLE.has(format);
}

function isDepthStencilRenderable(ctx: WebGLRenderingContext, format: GLenum): boolean {
  return ctx._version === 2
    ? isDepthRenderable(ctx, format) && isStencilRenderable(ctx, format)
    : W1_DS_RENDERABLE.has(format);
}

/* ------------------------------------------------------------------ */
/* Attachment resolution                                               */
/* ------------------------------------------------------------------ */

/**
 * WebGL2-spec enum → raster-registry key for Surface.format. The WebGL2 spec
 * defines RGBA8I/RGB8I as the DESKTOP-GL values 0x8d8e/0x8d8f (CTS
 * constants-and-properties-2.html hard-asserts them), while the raster formats
 * registry keys those two formats by the GLES3 values 0x8d8a/0x8d8b
 * (src/raster/gl-enums.ts). Surface.format is documented as the key into the
 * raster formats registry (raster/types.ts) — readPixels' getPackConverter
 * resolves through it — so surfaces for these two formats must carry the
 * raster-registry key. The separate `info` field (decode/encode) is attached
 * from the gl-side spec and is unaffected; both tables describe i8 storage.
 */
function surfaceFormatKey(internalFormat: GLenum): GLenum {
  if (internalFormat === C2.RGBA8I /* 0x8d8e */) return 0x8d8a;
  if (internalFormat === C2.RGB8I /* 0x8d8f */) return 0x8d8b;
  return internalFormat;
}

/** Resolve ONE attachment entry to its raster Surface (null when unset/invalid). */
function resolveAttachmentSurface(entry: FramebufferAttachment): Surface | null {
  if (entry.type === 'renderbuffer') {
    return entry.renderbuffer._surface;
  }
  const tex = entry.texture;
  const image = tex._image;
  if (!image || entry.level < 0) return null;
  const lvl = image.levels[entry.level];
  if (!lvl) return null;
  const faceIndex = isCubeFace(entry.face) ? cubeFaceIndex(entry.face) : 0;
  const data = lvl.data[faceIndex];
  if (!data) return null;
  const info: PixelFormatInfo | null = image.info ?? getFormat(image.internalFormat);
  if (!info) return null;
  return {
    width: lvl.width,
    height: lvl.height,
    format: surfaceFormatKey(image.internalFormat),
    info,
    data,
    stencilData: lvl.stencilData,
  };
}

/**
 * Resolve the current DRAW target: the bound draw framebuffer's attachments,
 * or the default framebuffer when none is bound. Returns null when the bound
 * FBO is incomplete (caller pushes INVALID_FRAMEBUFFER_OPERATION).
 */
export function resolveFramebufferTarget(ctx: WebGLRenderingContext): FramebufferTarget | null {
  const s = ctx._state;
  const fbo = s.drawFramebuffer;
  if (fbo === null) {
    const dfb = ctx._defaultFB;
    if (!dfb) return null; // pre-lifecycle: caller pushes INVALID_FRAMEBUFFER_OPERATION
    return { color: [dfb.color], depth: dfb.depth, stencil: dfb.stencil, width: dfb.width, height: dfb.height, samples: 1 };
  }
  if (checkFramebufferStatus(ctx, fbo) !== C1.FRAMEBUFFER_COMPLETE) return null;

  const maxColor = s.limits.MAX_COLOR_ATTACHMENTS;
  const color: (Surface | null)[] = new Array(maxColor).fill(null);
  let width = 0;
  let height = 0;
  let samples = 0;
  let hasAttach = false;

  const consider = (surf: Surface | null): void => {
    if (!surf) return;
    if (!hasAttach) {
      width = surf.width;
      height = surf.height;
      hasAttach = true;
    }
  };

  for (let i = 0; i < maxColor; i++) {
    const surf = getAttachmentSurface(fbo, COLOR_ATTACHMENT0 + i);
    color[i] = surf;
    consider(surf);
  }
  const depthEntry = fbo._attachments.get(DEPTH_ATTACHMENT) ?? fbo._attachments.get(DEPTH_STENCIL_ATTACHMENT);
  const stencilEntry = fbo._attachments.get(STENCIL_ATTACHMENT) ?? fbo._attachments.get(DEPTH_STENCIL_ATTACHMENT);
  const depth = depthEntry ? resolveAttachmentSurface(depthEntry) : null;
  const stencil = stencilEntry ? resolveAttachmentSurface(stencilEntry) : null;
  consider(depth);
  consider(stencil);

  // samples = common attachment sample count (completeness guarantees equality).
  const allEntries: FramebufferAttachment[] = [];
  for (let i = 0; i < maxColor; i++) {
    const e = fbo._attachments.get(COLOR_ATTACHMENT0 + i);
    if (e) allEntries.push(e);
  }
  if (depthEntry) allEntries.push(depthEntry);
  if (stencilEntry && stencilEntry !== depthEntry) allEntries.push(stencilEntry);
  if (allEntries.length > 0) samples = attachmentSamples(ctx, allEntries[0]);

  return { color, depth, stencil, width, height, samples };
}

/** Resolve the READ surface for readPixels (read framebuffer + readBuffer). */
export function resolveReadSurface(ctx: WebGLRenderingContext): Surface | null {
  const s = ctx._state;
  if (s.readFramebuffer === null) {
    // Default framebuffer read target is the drawing buffer regardless of
    // readBuffer (WebGL2 readBuffer applies to FBOs only).
    return ctx._defaultFB ? ctx._defaultFB.color : null;
  }
  const rb = s.readBuffer;
  if (rb === NONE) return null;
  const idx = rb - COLOR_ATTACHMENT0;
  if (idx < 0 || idx >= s.limits.MAX_COLOR_ATTACHMENTS) return null;
  return getAttachmentSurface(s.readFramebuffer, rb);
}

/**
 * Full WebGL framebuffer completeness check (spec rules; see header).
 * `fbo === null` (default framebuffer) is always FRAMEBUFFER_COMPLETE.
 */
export function checkFramebufferStatus(ctx: WebGLRenderingContext, fbo: WebGLFramebuffer): GLenum {
  if (fbo === null) return C1.FRAMEBUFFER_COMPLETE;
  const version = ctx._version;
  const maxColor = ctx._state.limits.MAX_COLOR_ATTACHMENTS;
  const att = fbo._attachments;

  // ---- collect attachments by point ----
  const colorEntries: { idx: number; entry: FramebufferAttachment }[] = [];
  let depthEntry: FramebufferAttachment | undefined;
  let stencilEntry: FramebufferAttachment | undefined;
  let dsEntry: FramebufferAttachment | undefined;
  let count = 0;
  for (const [point, entry] of att) {
    if (!entry) continue;
    if (point >= COLOR_ATTACHMENT0 && point < COLOR_ATTACHMENT0 + maxColor) {
      colorEntries.push({ idx: point - COLOR_ATTACHMENT0, entry });
      count++;
    } else if (point === DEPTH_ATTACHMENT) {
      depthEntry = entry;
      count++;
    } else if (point === STENCIL_ATTACHMENT) {
      stencilEntry = entry;
      count++;
    } else if (point === DEPTH_STENCIL_ATTACHMENT) {
      dsEntry = entry;
      count++;
    }
    // Unknown points cannot be created through the API (INVALID_ENUM at attach) — ignore.
  }

  // ---- 1. attachment presence ----
  if (count === 0) return C1.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;

  // ---- 2. sample-count consistency ----
  let samples = -1;
  const sampleEntries: FramebufferAttachment[] = [];
  for (const { entry } of colorEntries) sampleEntries.push(entry);
  if (depthEntry) sampleEntries.push(depthEntry);
  if (stencilEntry) sampleEntries.push(stencilEntry);
  if (dsEntry && version === 1) sampleEntries.push(dsEntry);
  for (const entry of sampleEntries) {
    const s = attachmentSamples(ctx, entry);
    if (samples === -1) samples = s;
    else if (samples !== s) return C2.FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;
  }

  // ---- 3. legal combination of images (UNSUPPORTED) ----
  if (version === 1) {
    // WebGL1 spec: error to concurrently attach to DEPTH+DS, STENCIL+DS, DEPTH+STENCIL.
    if ((depthEntry && dsEntry) || (stencilEntry && dsEntry) || (depthEntry && stencilEntry)) {
      return C1.FRAMEBUFFER_UNSUPPORTED;
    }
  }
  const drawBuffersAvail = version === 2 || hasExtension(ctx, 'WEBGL_draw_buffers');
  if (colorEntries.length > 1 && drawBuffersAvail) {
    for (let i = 0; i < colorEntries.length; i++) {
      for (let j = i + 1; j < colorEntries.length; j++) {
        if (sameImage(colorEntries[i].entry, colorEntries[j].entry)) {
          return C1.FRAMEBUFFER_UNSUPPORTED;
        }
      }
    }
  }
  const depth = depthEntry ?? dsEntry;
  const stencil = stencilEntry ?? dsEntry;
  if (depth && stencil && !sameImage(depth, stencil)) {
    return C1.FRAMEBUFFER_UNSUPPORTED;
  }

  // ---- 4. dimension consistency ----
  let dims: { width: number; height: number } | null = null;
  const dimEntries: FramebufferAttachment[] = [];
  for (const { entry } of colorEntries) dimEntries.push(entry);
  if (depthEntry) dimEntries.push(depthEntry);
  if (stencilEntry) dimEntries.push(stencilEntry);
  if (dsEntry && version === 1) dimEntries.push(dsEntry);
  for (const entry of dimEntries) {
    const d = attachmentDims(entry);
    if (!d) continue; // unresolvable image → caught by the per-attachment phase
    if (dims === null) dims = d;
    else if (dims.width !== d.width || dims.height !== d.height) {
      return C1.FRAMEBUFFER_INCOMPLETE_DIMENSIONS;
    }
  }

  // ---- 5. per-attachment checks (level completeness + renderability) ----
  for (const { entry } of colorEntries) {
    const st = checkAttachment(ctx, entry, 'color');
    if (st !== null) return st;
  }
  if (version === 1) {
    // WebGL1: DEPTH/STENCIL/DEPTH_STENCIL are independent attachment points.
    if (depthEntry) {
      const st = checkAttachment(ctx, depthEntry, 'depth');
      if (st !== null) return st;
    }
    if (stencilEntry) {
      const st = checkAttachment(ctx, stencilEntry, 'stencil');
      if (st !== null) return st;
    }
    if (dsEntry) {
      const st = checkAttachment(ctx, dsEntry, 'depthStencil');
      if (st !== null) return st;
    }
  } else {
    // WebGL2: DEPTH_STENCIL_ATTACHMENT aliases DEPTH+STENCIL → check the
    // effective images against the depth and stencil points.
    if (depth) {
      const st = checkAttachment(ctx, depth, 'depth');
      if (st !== null) return st;
    }
    if (stencil && stencil !== depth) {
      const st = checkAttachment(ctx, stencil, 'stencil');
      if (st !== null) return st;
    }
  }

  // ---- 6. layered-attachment consistency (WebGL2; vacuous — see header) ----
  if (version === 2) {
    let layered = -1;
    for (const entry of sampleEntries) {
      const l = isLayeredAttachment(entry) ? 1 : 0;
      if (layered === -1) layered = l;
      else if (layered !== l) return C.FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS;
    }
  }

  return C1.FRAMEBUFFER_COMPLETE;
}

/** Per-attachment completeness: image validity + renderability at its point.
 *  Returns a status GLenum when the attachment is invalid, else null. */
function checkAttachment(
  ctx: WebGLRenderingContext,
  entry: FramebufferAttachment,
  kind: 'color' | 'depth' | 'stencil' | 'depthStencil',
): GLenum | null {
  let format: GLenum;
  if (entry.type === 'renderbuffer') {
    const rb = entry.renderbuffer;
    const surf = rb._surface;
    if (!surf) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT; // attached without allocated storage
    if (surf.width === 0 || surf.height === 0) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    format = rb._internalformat;
  } else {
    const tex = entry.texture;
    const image = tex._image;
    if (!image || entry.level < 0) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    const lvl = image.levels[entry.level];
    if (!lvl) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    if (isCubeFace(entry.face)) {
      // A cube face is framebuffer-attachment-complete only when the cube map
      // is cube-complete at that level (all 6 faces, same size).
      if (lvl.data.length !== 6) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
      for (const face of lvl.data) {
        if (!face) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
      }
    } else if (!lvl.data[0]) {
      return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }
    if (lvl.width === 0 || lvl.height === 0) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    format = image.internalFormat;
    // WebGL1: float-storage LUMINANCE/LUMINANCE_ALPHA/ALPHA textures are NOT
    // color-renderable (EXT_color_buffer_half_float spec; CTS copyTex matrix
    // hard-asserts FRAMEBUFFER_INCOMPLETE_ATTACHMENT for these).
    if (ctx._version === 1 && image.info?.isFloat &&
        (format === 0x1909 /* LUMINANCE */ || format === 0x190a /* LUMINANCE_ALPHA */ || format === 0x1906 /* ALPHA */)) {
      return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }
  }

  switch (kind) {
    case 'color':
      if (!isColorRenderable(ctx, format)) {
        // Objective-mandated WebGL2 deviation: RGB8 (and unsized RGB, which
        // resolves to RGB8) is NOT color-renderable. For TEXTURE attachments
        // the deviation reports FRAMEBUFFER_UNSUPPORTED; RGB8 RENDERBUFFERS are
        // color-renderable per GLES3 Table 3.13 → COMPLETE
        // (read-pixels-from-rgb8-into-pbo-bug.html hard-asserts).
        if (ctx._version === 2 && (format === 0x8051 /* RGB8 */ || format === 0x1907 /* RGB (unsized → RGB8) */)) {
          return entry.type === 'texture' ? C1.FRAMEBUFFER_UNSUPPORTED : null;
        }
        return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
      }
      return null;
    case 'depth':
      return isDepthRenderable(ctx, format) ? null : C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    case 'stencil':
      return isStencilRenderable(ctx, format) ? null : C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    case 'depthStencil':
      return isDepthStencilRenderable(ctx, format) ? null : C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
  }
}

/** Resolve one attachment point of a framebuffer to its surface (null when unset/invalid). */
export function getAttachmentSurface(fbo: WebGLFramebuffer, attachment: GLenum): Surface | null {
  const entry = fbo._attachments.get(attachment);
  if (!entry) return null;
  return resolveAttachmentSurface(entry);
}
