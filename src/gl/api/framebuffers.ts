/**
 * src/gl/api/framebuffers.ts — framebuffer/renderbuffer API (WebGL1 + WebGL2).
 *
 * Owns: createFramebuffer, deleteFramebuffer, isFramebuffer, bindFramebuffer,
 * framebufferRenderbuffer, framebufferTexture2D, framebufferTextureLayer (W2),
 * checkFramebufferStatus, getFramebufferAttachmentParameter,
 * createRenderbuffer, deleteRenderbuffer, isRenderbuffer, bindRenderbuffer,
 * renderbufferStorage, renderbufferStorageMultisample (W2),
 * getRenderbufferParameter, drawBuffers (W2), readBuffer (W2),
 * invalidateFramebuffer (W2), invalidateSubFramebuffer (W2), blitFramebuffer (W2).
 *
 * Behavior notes (implemented):
 *  - bindFramebuffer: WebGL1 target FRAMEBUFFER only; WebGL2 adds DRAW/READ
 *    FRAMEBUFFER. FRAMEBUFFER binds BOTH the draw and read slots (ES3). null =
 *    default framebuffer (stored as a null binding). Deleting a bound FBO
 *    resets the binding (GLES2 glDeleteFramebuffers semantics).
 *  - framebufferRenderbuffer: attachment ∈ {COLOR_ATTACHMENT0..max-1 (1 on W1),
 *    DEPTH, STENCIL, DEPTH_STENCIL} else INVALID_ENUM; renderbuffertarget must
 *    be RENDERBUFFER; WebGL1 requires the renderbuffer to have been bound at
 *    least once (CTS framebuffer-object-attachment.html "bindRenderbuffer must
 *    be called before attachment"); null detaches. WebGL2 DEPTH_STENCIL_ATTACHMENT
 *    is an ALIAS: the same record is written to BOTH the DEPTH and STENCIL
 *    slots; querying DEPTH_STENCIL returns the record only when both slots hold
 *    the identical record (WebGL 2.0 spec "Framebuffer Object Attachments"),
 *    and a depth/stencil mismatch is INVALID_OPERATION for the query.
 *  - framebufferTexture2D: WebGL1 level must be 0 (INVALID_VALUE otherwise),
 *    unless OES_fbo_render_mipmap is enabled (levels < log2(MAX_TEXTURE_SIZE)+1);
 *    WebGL2 level must satisfy 0 ≤ level < numLevelsFromSize(MAX_TEXTURE_SIZE)
 *    (CTS framebuffer-test.html: maxLevels-1 OK, maxLevels → INVALID_VALUE —
 *    the level need NOT exist in the texture; completeness reports it).
 *    Attachment record shape matches WEBGL_multisampled_render_to_texture's
 *    factory: { type:'texture', texture, level, face: textarget, layer: 0 }.
 *  - framebufferTextureLayer (W2): texture must be TEXTURE_2D_ARRAY/TEXTURE_3D
 *    (INVALID_OPERATION); level < numLevelsFromSize(MAX_3D_TEXTURE_SIZE)
 *    (INVALID_VALUE); 0 ≤ layer < texture depth (INVALID_VALUE).
 *  - checkFramebufferStatus delegates to framebuffer-util.checkFramebufferStatus
 *    (try/catch); while that stub throws, a LOCAL fallback applies
 *    (MISSING_ATTACHMENT when empty, INCOMPLETE_ATTACHMENT when any attachment
 *    lacks an image, else COMPLETE). Result cached on fbo._status.
 *  - getFramebufferAttachmentParameter: W1 default framebuffer → INVALID_OPERATION
 *    (spec: querying the default FB is an error on W1; CTS framebuffer-test.html).
 *    W2 default framebuffer: attachment ∈ {BACK, DEPTH, STENCIL} else
 *    INVALID_ENUM; OBJECT_TYPE → FRAMEBUFFER_DEFAULT. No-attachment FBO:
 *    W1 OBJECT_TYPE → NONE, all other pnames → INVALID_ENUM (CTS); W2
 *    OBJECT_TYPE → NONE / OBJECT_NAME → null / LEVEL+FACE+LAYER → 0 /
 *    COLOR_ENCODING+COMPONENT_TYPE+SIZES → INVALID_OPERATION (CTS framebuffer-test.html).
 *  - renderbufferStorage: WebGL1 base {RGBA4, RGB565, RGB5_A1, DEPTH_COMPONENT16,
 *    STENCIL_INDEX8} + gated {SRGB_ALPHA_EXT (EXT_sRGB), DEPTH_COMPONENT,
 *    DEPTH_STENCIL (WEBGL_depth_texture), RGBA16F (EXT_color_buffer_half_float)};
 *    WebGL2 the ES3 sized set (incl. DEPTH_STENCIL — CTS createDepthStencilBuffer)
 *    + gated norm16 formats (EXT_texture_norm16) and RGB9_E5
 *    (WEBGL_render_shared_exponent); else INVALID_ENUM. width/height > MAX_RENDERBUFFER_SIZE
 *    or negative → INVALID_VALUE. Allocates rb._surface via raster
 *    surface.createSurface (try/catch → LOCAL fallback with a minimal
 *    bytes-per-texel table; replace with raster/formats when it lands).
 *  - renderbufferStorageMultisample: samples < 0 → INVALID_VALUE; samples >
 *    MAX_SAMPLES → INVALID_OPERATION; storage stays single-sampled (software
 *    resolve happens at blit); record rb._samples.
 *  - drawBuffers (W2): sequence<GLenum> conversion (TypeError for non-finite);
 *    length > MAX_DRAW_BUFFERS → INVALID_VALUE; length 0 → INVALID_OPERATION;
 *    default FB: exactly [BACK] or [NONE] else INVALID_OPERATION; FBO: entries
 *    NONE or strictly-increasing COLOR_ATTACHMENTi < MAX_COLOR_ATTACHMENTS
 *    (BACK → INVALID_OPERATION, out-of-range → INVALID_ENUM, out-of-order →
 *    INVALID_OPERATION). Stores state.drawBuffers (same field WEBGL_draw_buffers
 *    writes). WebGL1 gets drawBuffersWEBGL from the extension instead.
 *  - readBuffer (W2): default FB {NONE, BACK} else INVALID_OPERATION; FBO
 *    {NONE, COLOR_ATTACHMENT0..<max} else INVALID_OPERATION; src below
 *    COLOR_ATTACHMENT0 (non-NONE) → INVALID_ENUM.
 *  - invalidateFramebuffer/SubFramebuffer (W2): target FRAMEBUFFER/DRAW/READ
 *    else INVALID_ENUM; attachments legal per bound FB (FBO: COLOR_ATTACHMENTi
 *    i<max, DEPTH, STENCIL, DEPTH_STENCIL; default FB: COLOR, DEPTH, STENCIL)
 *    else INVALID_OPERATION (nonexistent-but-legal attachments are ignored,
 *    NO_ERROR); negative width/height → INVALID_VALUE. No-op (software).
 *  - blitFramebuffer (W2): mask stray bits → INVALID_VALUE; filter ∉
 *    {NEAREST, LINEAR} → INVALID_ENUM; LINEAR + depth/stencil mask →
 *    INVALID_OPERATION; same read/draw FBO object (incl. both default) →
 *    INVALID_OPERATION; identical source/dest image (same texture+level+face+
 *    layer, or same renderbuffer) for any mask bit → INVALID_OPERATION;
 *    incomplete read or draw FBO → INVALID_FRAMEBUFFER_OPERATION (GLES3
 *    §4.4.4);
 *    multisample: draw-multisampled + read-single OR both multisampled →
 *    INVALID_OPERATION; read-multisampled + draw-single → OK (resolve, CTS
 *    multisampled-depth-renderbuffer-initialization.html). Then delegates to
 *    draw.executeBlitFramebuffer (try/catch → INVALID_OPERATION while stub).
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';
import { C1, C2, CExt } from '../constants';
import {
  WebGLFramebuffer,
  WebGLRenderbuffer,
  WebGLTexture,
  createObject,
  type FramebufferAttachment,
} from '../objects';
import { validateObject } from '../validation';
import type { GLboolean, GLbitfield, GLenum, GLint, GLsizei } from '../types';
import { executeBlitFramebuffer } from '../draw';
import { checkFramebufferStatus as utilCheckFramebufferStatus } from '../framebuffer-util';
import { createSurface, getFormat } from '../../raster';
import type { PixelFormatInfo, StorageKind, Surface } from '../../raster';

// ---------------------------------------------------------------------------
// Local constants (values not present in constants.ts / CExt)
// ---------------------------------------------------------------------------
const TEXTURE_SAMPLES_EXT = 0x9130; // WEBGL_multisampled_render_to_texture (local in extensions/multisampled.ts too)
const FRAMEBUFFER_DEFAULT = 0x8218; // C2.FRAMEBUFFER_DEFAULT
const COLOR_ATTACHMENT0 = 0x8ce0;
const DEPTH_ATTACHMENT = 0x8d00;
const STENCIL_ATTACHMENT = 0x8d20;
const DEPTH_STENCIL_ATTACHMENT = 0x821a;
const NONE = 0x0000;
const BACK = 0x0405;
const TEXTURE = 0x1702;
const LINEAR = 0x2601;
const SRGB = 0x8c40;
const UNSIGNED_NORMALIZED = 0x8c17;
const SIGNED_NORMALIZED = 0x8f9c;
const FLOAT = 0x1406;
const INT = 0x1404;
const UNSIGNED_INT = 0x1405;
const CUBE_POSITIVE_X = 0x8515;
const CUBE_NEGATIVE_Z = 0x851a;

/** Context-loss guard: no-op while lost WITHOUT generating an error (buffers.ts pattern; CTS context-lost.html asserts NO_ERROR after every void call while lost). */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

// WebGLObject constructors are protected — cast once (buffers.ts pattern).
const FboCtor = WebGLFramebuffer as unknown as new (context: WebGLRenderingContext) => WebGLFramebuffer;
const FboCtorAny = FboCtor as unknown as new (...args: never[]) => WebGLFramebuffer;
const RboCtor = WebGLRenderbuffer as unknown as new (context: WebGLRenderingContext) => WebGLRenderbuffer;
const RboCtorAny = RboCtor as unknown as new (...args: never[]) => WebGLRenderbuffer;
const TexCtor = WebGLTexture as unknown as new (context: WebGLRenderingContext) => WebGLTexture;
const TexCtorAny = TexCtor as unknown as new (...args: never[]) => WebGLTexture;

function validateFbo(ctx: WebGLRenderingContext, obj: unknown): WebGLFramebuffer | null {
  return validateObject<WebGLFramebuffer>(ctx, obj, FboCtorAny);
}
function validateRbo(ctx: WebGLRenderingContext, obj: unknown): WebGLRenderbuffer | null {
  return validateObject<WebGLRenderbuffer>(ctx, obj, RboCtorAny);
}
function validateTex(ctx: WebGLRenderingContext, obj: unknown): WebGLTexture | null {
  return validateObject<WebGLTexture>(ctx, obj, TexCtorAny);
}

/** Valid framebuffer targets for this version. */
function isValidFramebufferTarget(ctx: WebGLRenderingContext, target: GLenum): boolean {
  if (target === C1.FRAMEBUFFER) return true;
  if (ctx._version !== 2) return false;
  return target === C2.DRAW_FRAMEBUFFER || target === C2.READ_FRAMEBUFFER;
}

/** Bound FBO for a validated target (null = default framebuffer). */
function boundFramebufferForTarget(ctx: WebGLRenderingContext, target: GLenum): WebGLFramebuffer | null {
  if (ctx._version === 2 && target === C2.READ_FRAMEBUFFER) return ctx._state.readFramebuffer;
  return ctx._state.drawFramebuffer;
}

/** Number of mip levels for a max dimension (matches CTS numLevelsFromSize). */
function numLevelsFromSize(size: number): number {
  let levels = 0;
  while ((size >> levels) > 0) ++levels;
  return levels;
}

function isCubeFace(textarget: GLenum): boolean {
  return textarget >= CUBE_POSITIVE_X && textarget <= CUBE_NEGATIVE_Z;
}

function isTextureTarget2D(textarget: GLenum): boolean {
  return textarget === C1.TEXTURE_2D || isCubeFace(textarget);
}

/** Attachment-point validation. WebGL1: only COLOR_ATTACHMENT0 (spec); W2: up to MAX_COLOR_ATTACHMENTS. */
function isValidAttachment(ctx: WebGLRenderingContext, attachment: GLenum): boolean {
  const maxColor = ctx._version === 1 ? 1 : ctx._state.limits.MAX_COLOR_ATTACHMENTS;
  if (attachment >= COLOR_ATTACHMENT0 && attachment < COLOR_ATTACHMENT0 + maxColor) return true;
  if (attachment === DEPTH_ATTACHMENT || attachment === STENCIL_ATTACHMENT) return true;
  return attachment === DEPTH_STENCIL_ATTACHMENT; // core in WebGL1 (0x821a) and WebGL2
}

/** Attachment keys a set/query touches (WebGL2 DEPTH_STENCIL aliases depth+stencil). */
function attachmentKeys(ctx: WebGLRenderingContext, attachment: GLenum): GLenum[] {
  if (ctx._version === 2 && attachment === DEPTH_STENCIL_ATTACHMENT) {
    return [DEPTH_ATTACHMENT, STENCIL_ATTACHMENT];
  }
  return [attachment];
}

/**
 * Attachment-OBJECT identity for the WebGL2 DEPTH_STENCIL_ATTACHMENT query:
 * the query succeeds when the depth and stencil points hold the same
 * renderbuffer object or the same texture OBJECT (level/face/layer are NOT
 * compared — CTS framebuffer-texture-layer.html attaches the same 2D_ARRAY
 * texture at depth layer 0 and stencil layer 1 and still expects the query to
 * return the texture; mismatch = different objects or one point empty →
 * INVALID_OPERATION per framebuffer-object-attachment.html).
 */
function sameAttachmentObject(a: FramebufferAttachment, b: FramebufferAttachment): boolean {
  if (a.type === 'renderbuffer' && b.type === 'renderbuffer') return a.renderbuffer === b.renderbuffer;
  if (a.type === 'texture' && b.type === 'texture') return a.texture === b.texture;
  return false;
}

/**
 * GLES3 §4.4.2 (framebufferTexture2D / framebufferTextureLayer): attaching a
 * texture to DEPTH_ATTACHMENT requires its internal format to have a depth
 * component, to STENCIL_ATTACHMENT a stencil component; WebGL2's
 * DEPTH_STENCIL_ATTACHMENT (alias of both points) requires both. A
 * depth-stencil format at DEPTH or STENCIL alone is legal. When the texture
 * has no allocated image yet (internal format unknown — the record is created
 * before the level exists), there is no mismatch: framebuffer completeness
 * reports the unallocated level instead.
 */
function textureAttachmentFormatMismatch(tex: WebGLTexture, attachment: GLenum): boolean {
  const image = tex._image;
  if (!image) return false;
  const d = localFormatDesc(image.internalFormat);
  if (attachment === DEPTH_ATTACHMENT) return !d.isDepth;
  if (attachment === STENCIL_ATTACHMENT) return !d.isStencil;
  if (attachment === DEPTH_STENCIL_ATTACHMENT) return !(d.isDepth && d.isStencil);
  return false;
}

/**
 * Resolve one attachment point's record. For WebGL2 DEPTH_STENCIL_ATTACHMENT
 * the record is returned only when the depth and stencil attachment points
 * hold the same attachment OBJECT (see sameAttachmentObject); a mismatch
 * (different objects, or one point empty) is reported via `conflict` and the
 * caller generates INVALID_OPERATION. On WebGL1 DEPTH_STENCIL_ATTACHMENT is a
 * DISTINCT attachment point from DEPTH/STENCIL — the record stored under key
 * 0x821a is returned directly (misc/expando-loss.html).
 */
function resolveAttachmentRecord(
  ctx: WebGLRenderingContext,
  fbo: WebGLFramebuffer,
  attachment: GLenum,
): { rec: FramebufferAttachment | null; conflict: boolean } {
  if (ctx._version === 2 && attachment === DEPTH_STENCIL_ATTACHMENT) {
    const d = fbo._attachments.get(DEPTH_ATTACHMENT) ?? null;
    const s = fbo._attachments.get(STENCIL_ATTACHMENT) ?? null;
    if (d === null && s === null) return { rec: null, conflict: false };
    // The W2 alias is coherent only when BOTH points hold the same image:
    // one empty point or two different images → INVALID_OPERATION (WebGL2
    // spec; framebuffer-object-attachment.html "DEPTH_ATTACHMENT overwrites
    // depth set by DEPTH_STENCIL_ATTACHMENT" expects the error also when only
    // the depth point is occupied).
    if (d === null || s === null) return { rec: null, conflict: true };
    if (!sameAttachmentObject(d, s)) return { rec: null, conflict: true };
    return { rec: d, conflict: false };
  }
  return { rec: fbo._attachments.get(attachment) ?? null, conflict: false };
}

/** Invalidate the completeness cache + recompute the multisample flag of every FBO. */
function invalidateFboStatuses(ctx: WebGLRenderingContext): void {
  for (const obj of ctx._resources.all) {
    if (obj instanceof WebGLFramebuffer) {
      obj._status = 0;
      let ms = false;
      for (const rec of obj._attachments.values()) {
        if (rec.type === 'renderbuffer') {
          if (rec.renderbuffer._samples > 0) ms = true;
        } else if (rec.texture._msaaSamples > 0) {
          ms = true;
        }
      }
      obj._multisampled = ms;
    }
  }
}

/** Invalidate the status of one FBO (and recompute its multisample flag). */
function invalidateFboStatus(fbo: WebGLFramebuffer): void {
  fbo._status = 0;
  let ms = false;
  for (const rec of fbo._attachments.values()) {
    if (rec.type === 'renderbuffer') {
      if (rec.renderbuffer._samples > 0) ms = true;
    } else if (rec.texture._msaaSamples > 0) {
      ms = true;
    }
  }
  fbo._multisampled = ms;
}

// ---------------------------------------------------------------------------
// Renderbuffer format tables
// ---------------------------------------------------------------------------

/**
 * Local renderbuffer-format validation (raster/formats.isValidRenderbufferFormat
 * throws until it lands). Extension gates check the ENABLED-extensions cache
 * (`ctx._extensions`) — calling getExtension() here would SELF-ENABLE the
 * extension, observable as renderbuffers accepted before it is requested.
 */
function isValidRenderbufferFormat(ctx: WebGLRenderingContext, format: GLenum): boolean {
  if (ctx._version === 1) {
    if (
      format === C1.RGBA4 || format === C1.RGB565 || format === C1.RGB5_A1 ||
      format === C1.DEPTH_COMPONENT16 || format === C1.STENCIL_INDEX8
    ) return true;
    if (format === CExt.SRGB_ALPHA_EXT && ctx._extensions.has('EXT_sRGB')) return true;
    // DEPTH_COMPONENT/DEPTH_STENCIL are core WebGL1 renderbuffer formats
    // (WEBGL_depth_texture gates only the *texture* side of depth formats).
    if (format === C1.DEPTH_COMPONENT || format === C1.DEPTH_STENCIL) return true;
    // EXT_color_buffer_half_float (WebGL1): RGB16F + RGBA16F renderbuffers.
    if ((format === C2.RGB16F || format === C2.RGBA16F) &&
        ctx._extensions.has('EXT_color_buffer_half_float')) return true;
    return false;
  }
  // WebGL2: ES3 sized color/depth/stencil set + DEPTH_STENCIL (CTS uses it).
  if (W2_RB_FORMATS.has(format)) return true;
  // EXT_color_buffer_float (WebGL2): all float renderbuffer formats; the 16F
  // subset is also enabled by EXT_color_buffer_half_float. RGB16F/RGB32F are
  // NEVER legal renderbuffer formats (CTS ext-color-buffer-half-float.html
  // runRGB16FNegativeTest expects INVALID_ENUM).
  if (W2_RB_EXT_FLOAT.has(format)) {
    if (ctx._extensions.has('EXT_color_buffer_float')) return true;
    return (format === C2.R16F || format === C2.RG16F || format === C2.RGBA16F) &&
      ctx._extensions.has('EXT_color_buffer_half_float');
  }
  if (W2_RB_EXT_NORM16.has(format) && ctx._extensions.has('EXT_texture_norm16')) return true;
  if (format === C2.RGB9_E5 && ctx._extensions.has('WEBGL_render_shared_exponent')) return true;
  return false;
}

/** WebGL2 core renderbuffer formats (RGB16F/RGB32F are NOT legal — see above;
 *  STENCIL_INDEX8 is a legal ES3 renderbuffer-only format — CTS
 *  framebuffer-object-attachment.html renderbufferStorage(STENCIL_INDEX8)). */
const W2_RB_FORMATS: ReadonlySet<GLenum> = new Set<GLenum>([
  C2.R8, C2.R8UI, C2.R8I, C2.R16UI, C2.R16I, C2.R32UI, C2.R32I,
  C2.RG8, C2.RG8UI, C2.RG8I, C2.RG16UI, C2.RG16I, C2.RG32UI, C2.RG32I,
  C2.RGB8, C2.RGB8UI, C2.RGB8I, C2.RGB16UI, C2.RGB16I, C2.RGB32UI, C2.RGB32I,
  C2.RGBA8, C2.RGBA8UI, C2.RGBA8I, C2.RGBA16UI, C2.RGBA16I,
  C2.RGBA32UI, C2.RGBA32I,
  C2.RGB10_A2, C1.RGBA4, C1.RGB5_A1, C1.RGB565, C2.SRGB8_ALPHA8,
  C1.DEPTH_COMPONENT16, C2.DEPTH_COMPONENT24, C2.DEPTH_COMPONENT32F,
  C2.DEPTH24_STENCIL8, C2.DEPTH32F_STENCIL8, C1.DEPTH_STENCIL,
  C1.STENCIL_INDEX8,
]);

/** WebGL2 float renderbuffer formats (EXT_color_buffer_float / _half_float). */
const W2_RB_EXT_FLOAT: ReadonlySet<GLenum> = new Set<GLenum>([
  C2.R16F, C2.RG16F, C2.RGBA16F, C2.R32F, C2.RG32F, C2.RGBA32F, C2.R11F_G11F_B10F,
]);

/** WebGL2 renderbuffer formats gated on EXT_texture_norm16. */
const W2_RB_EXT_NORM16: ReadonlySet<GLenum> = new Set<GLenum>([
  CExt.R16_EXT, CExt.RG16_EXT, CExt.RGB16_EXT, CExt.RGBA16_EXT,
]);

/**
 * LOCAL surface description table (bytes/texel + storage class + flags).
 * Replace with raster/formats.ts registry when it lands.
 * Depth formats store Float32Array regardless of nominal bit depth (raster
 * Surface contract: DEPTH_COMPONENT_* data is Float32Array 0..1).
 */
interface LocalFormatDesc {
  bpp: number;
  storage: StorageKind;
  components: number;
  isColor: boolean;
  isDepth: boolean;
  isStencil: boolean;
  isFloat: boolean;
  isSigned: boolean;
  isInteger: boolean;
  isSRGB: boolean;
  normalized: boolean;
}

const DEPTH_STENCIL_DESC: LocalFormatDesc = {
  bpp: 4, storage: 'f32', components: 1, isColor: false, isDepth: true, isStencil: true,
  isFloat: true, isSigned: false, isInteger: false, isSRGB: false, normalized: false,
};
const DEPTH_DESC: LocalFormatDesc = {
  bpp: 4, storage: 'f32', components: 1, isColor: false, isDepth: true, isStencil: false,
  isFloat: true, isSigned: false, isInteger: false, isSRGB: false, normalized: false,
};
const STENCIL_DESC: LocalFormatDesc = {
  bpp: 1, storage: 'u8', components: 1, isColor: false, isDepth: false, isStencil: true,
  isFloat: false, isSigned: false, isInteger: false, isSRGB: false, normalized: false,
};
function colorDesc(bpp: number, storage: StorageKind, components: number, extra?: Partial<LocalFormatDesc>): LocalFormatDesc {
  return {
    bpp, storage, components, isColor: true, isDepth: false, isStencil: false,
    isFloat: storage === 'f32', isSigned: storage === 'i8' || storage === 'i16' || storage === 'i32',
    isInteger: false, isSRGB: false, normalized: false,
    ...extra,
  };
}

/** Normalized-color desc (u8/u16/u32 storage; isInteger must stay false). */
function normColorDesc(bpp: number, storage: StorageKind, components: number, extra?: Partial<LocalFormatDesc>): LocalFormatDesc {
  return colorDesc(bpp, storage, components, { normalized: true, ...extra });
}

/** Integer-color desc (raw storage; normalized must stay false). */
function intColorDesc(bpp: number, storage: StorageKind, components: number): LocalFormatDesc {
  return colorDesc(bpp, storage, components, { isInteger: true });
}

function localFormatDesc(format: GLenum): LocalFormatDesc {
  switch (format) {
    case C1.RGBA4: case C1.RGB5_A1: case C1.RGB565:
      return normColorDesc(2, 'u16', format === C1.RGB565 ? 3 : 4);
    case C1.RGBA8: case C2.SRGB8_ALPHA8: case CExt.SRGB_ALPHA_EXT:
      return normColorDesc(4, 'u8', 4, { isSRGB: format !== C1.RGBA8 });
    case C2.RGB8: return normColorDesc(3, 'u8', 3);
    case C2.R8: return normColorDesc(1, 'u8', 1);
    case C2.RG8: return normColorDesc(2, 'u8', 2);
    case CExt.R16_EXT: return normColorDesc(2, 'u16', 1);
    case CExt.RG16_EXT: return normColorDesc(4, 'u16', 2);
    case CExt.RGB16_EXT: return normColorDesc(6, 'u16', 3);
    case CExt.RGBA16_EXT: return normColorDesc(8, 'u16', 4);
    case C2.R16F: return colorDesc(2, 'f32', 1);
    case C2.RG16F: return colorDesc(4, 'f32', 2);
    case C2.RGB16F: return colorDesc(6, 'f32', 3);
    case C2.RGBA16F: return colorDesc(8, 'f32', 4);
    case C2.R32F: return colorDesc(4, 'f32', 1);
    case C2.RG32F: return colorDesc(8, 'f32', 2);
    case C2.RGB32F: return colorDesc(12, 'f32', 3);
    case C2.RGBA32F: return colorDesc(16, 'f32', 4);
    case C2.RGB9_E5: return colorDesc(12, 'f32', 3);
    case C2.R8I: return intColorDesc(1, 'i8', 1);
    case C2.R8UI: return intColorDesc(1, 'u8', 1);
    case C2.R16I: return intColorDesc(2, 'i16', 1);
    case C2.R16UI: return intColorDesc(2, 'u16', 1);
    case C2.R32I: return intColorDesc(4, 'i32', 1);
    case C2.R32UI: return intColorDesc(4, 'u32', 1);
    case C2.RG8I: return intColorDesc(2, 'i8', 2);
    case C2.RG8UI: return intColorDesc(2, 'u8', 2);
    case C2.RG16I: return intColorDesc(4, 'i16', 2);
    case C2.RG16UI: return intColorDesc(4, 'u16', 2);
    case C2.RG32I: return intColorDesc(8, 'i32', 2);
    case C2.RG32UI: return intColorDesc(8, 'u32', 2);
    case C2.RGB8I: return intColorDesc(3, 'i8', 3);
    case C2.RGB8UI: return intColorDesc(3, 'u8', 3);
    case C2.RGB16I: return intColorDesc(6, 'i16', 3);
    case C2.RGB16UI: return intColorDesc(6, 'u16', 3);
    case C2.RGB32I: return intColorDesc(12, 'i32', 3);
    case C2.RGB32UI: return intColorDesc(12, 'u32', 3);
    case C2.RGBA8I: return intColorDesc(4, 'i8', 4);
    case C2.RGBA8UI: return intColorDesc(4, 'u8', 4);
    case C2.RGBA16I: return intColorDesc(8, 'i16', 4);
    case C2.RGBA16UI: return intColorDesc(8, 'u16', 4);
    case C2.RGBA32I: return intColorDesc(16, 'i32', 4);
    case C2.RGBA32UI: return intColorDesc(16, 'u32', 4);
    case C2.RGB10_A2: return normColorDesc(4, 'u32', 4);
    case C1.DEPTH_COMPONENT16: case C2.DEPTH_COMPONENT24: case C2.DEPTH_COMPONENT32F:
    case C1.DEPTH_COMPONENT:
      return DEPTH_DESC;
    case C2.DEPTH24_STENCIL8: case C2.DEPTH32F_STENCIL8: case C1.DEPTH_STENCIL:
      return DEPTH_STENCIL_DESC;
    case C1.STENCIL_INDEX8:
      return STENCIL_DESC;
    default:
      return normColorDesc(4, 'u8', 4); // unknown color-ish default
  }
}

const noopDecode = (_data: ArrayBufferView, _byteOffset: number, out?: Float32Array): Float32Array => out ?? new Float32Array(4);
const noopEncode = (_data: ArrayBufferView, _byteOffset: number, _r: number, _g: number, _b: number, _a: number): void => { /* placeholder */ };

/** Build a synthetic PixelFormatInfo when the raster registry has no entry yet. */
function syntheticInfo(format: GLenum, d: LocalFormatDesc): PixelFormatInfo {
  return {
    format, components: d.components, bytesPerPixel: d.bpp, storage: d.storage,
    isColor: d.isColor, isDepth: d.isDepth, isStencil: d.isStencil, isFloat: d.isFloat,
    isSigned: d.isSigned, isInteger: d.isInteger, isSRGB: d.isSRGB, normalized: d.normalized,
    decode: noopDecode, encode: noopEncode,
  };
}

/**
 * Allocate a renderbuffer surface. Prefers raster surface.createSurface;
 * falls back to a local allocation (comment: replace with raster/formats when
 * it lands) honoring the raster Surface representation (depth → Float32Array,
 * DEPTH*_STENCIL* → Float32Array + stencilData Uint8Array, STENCIL → Uint8Array).
 */
function allocateRenderbufferSurface(ctx: WebGLRenderingContext, rb: WebGLRenderbuffer, format: GLenum, w: number, h: number): void {
  let surf: Surface | null = null;
  try {
    surf = createSurface(format, w, h);
  } catch {
    surf = null;
  }
  if (surf === null) {
    const d = localFormatDesc(format);
    const n = w * h;
    let data: ArrayBufferView;
    switch (d.storage) {
      case 'f32': data = new Float32Array(n); break;
      case 'i8': data = new Int8Array(n); break;
      case 'i16': data = new Int16Array(n); break;
      case 'i32': data = new Int32Array(n); break;
      case 'u16': data = new Uint16Array(n); break;
      case 'u32': data = new Uint32Array(n); break;
      default: data = new Uint8Array(n);
    }
    const info = getFormat(format) ?? syntheticInfo(format, d);
    const s: Surface = { width: w, height: h, format, info, data };
    if (d.isStencil && d.isDepth) s.stencilData = new Uint8Array(n);
    surf = s;
  }
  rb._surface = surf;
  // Spec: the initial contents of a renderbuffer's depth plane are 1.0
  // (the stencil plane of DEPTH*_STENCIL* stays 0). Only RENDERBUFFERS get
  // this — depth TEXTURE initial contents are legitimately undefined, and this
  // helper is renderbuffer-only (raster createSurface stays zero-filled).
  if (surf.info.isDepth) {
    (surf.data as Float32Array).fill(1.0);
  }
  void ctx;
}

/** Renderbuffer format → [r,g,b,a] component bit counts + depth + stencil bits. */
function formatComponentBits(format: GLenum): [number, number, number, number, number, number] {
  const zero: [number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0];
  switch (format) {
    case C1.RGBA8: case C2.SRGB8_ALPHA8: case CExt.SRGB_ALPHA_EXT:
      return [8, 8, 8, 8, 0, 0];
    case C2.RGB8: case C2.SRGB8: return [8, 8, 8, 0, 0, 0];
    case C1.RGBA4: return [4, 4, 4, 4, 0, 0];
    case C1.RGB5_A1: return [5, 5, 5, 1, 0, 0];
    case C1.RGB565: return [5, 6, 5, 0, 0, 0];
    case C2.R8: case C2.R8I: case C2.R8UI: case C2.R16F: case C2.R32F: case CExt.R16_EXT:
      return [8, 0, 0, 0, 0, 0];
    case C2.RG8: case C2.RG8I: case C2.RG8UI: case C2.RG16F: case C2.RG32F: case CExt.RG16_EXT:
      return [8, 8, 0, 0, 0, 0];
    case C2.R16I: case C2.R16UI: return [16, 0, 0, 0, 0, 0];
    case C2.RG16I: case C2.RG16UI: return [16, 16, 0, 0, 0, 0];
    case C2.R32I: case C2.R32UI: return [32, 0, 0, 0, 0, 0];
    case C2.RG32I: case C2.RG32UI: return [32, 32, 0, 0, 0, 0];
    case C2.RGB8I: case C2.RGB8UI: return [8, 8, 8, 0, 0, 0];
    case C2.RGB16I: case C2.RGB16UI: return [16, 16, 16, 0, 0, 0];
    case C2.RGB32I: case C2.RGB32UI: return [32, 32, 32, 0, 0, 0];
    case C2.RGBA8I: case C2.RGBA8UI: return [8, 8, 8, 8, 0, 0];
    case C2.RGBA16I: case C2.RGBA16UI: case CExt.RGBA16_EXT: return [16, 16, 16, 16, 0, 0];
    case C2.RGBA32I: case C2.RGBA32UI: return [32, 32, 32, 32, 0, 0];
    case C2.RGB16F: return [16, 16, 16, 0, 0, 0];
    case C2.RGB32F: case C2.RGB9_E5: return [32, 32, 32, 0, 0, 0];
    case C2.RGBA16F: return [16, 16, 16, 16, 0, 0];
    case C2.RGBA32F: return [32, 32, 32, 32, 0, 0];
    case C2.R16F: return [16, 0, 0, 0, 0, 0];
    case C2.RG16F: return [16, 16, 0, 0, 0, 0];
    case C2.R32F: return [32, 0, 0, 0, 0, 0];
    case C2.RG32F: return [32, 32, 0, 0, 0, 0];
    case C2.RGB10_A2: return [10, 10, 10, 2, 0, 0];
    case CExt.RGB16_EXT: return [16, 16, 16, 0, 0, 0];
    case C1.DEPTH_COMPONENT16: return [0, 0, 0, 0, 16, 0];
    case C2.DEPTH_COMPONENT24: return [0, 0, 0, 0, 24, 0];
    case C2.DEPTH_COMPONENT32F: return [0, 0, 0, 0, 32, 0];
    case C2.DEPTH24_STENCIL8: case C1.DEPTH_STENCIL: return [0, 0, 0, 0, 24, 8];
    case C2.DEPTH32F_STENCIL8: return [0, 0, 0, 0, 32, 8];
    case C1.STENCIL_INDEX8: return [0, 0, 0, 0, 0, 8];
    default: return zero;
  }
}

/** Attached-image internal format (0 when unallocated). */
function attachmentInternalFormat(rec: FramebufferAttachment): GLenum {
  if (rec.type === 'renderbuffer') return rec.renderbuffer._internalformat;
  return rec.texture._image ? rec.texture._image.internalFormat : 0;
}

/** COLOR_ENCODING (0x8210): LINEAR or SRGB. */
function formatColorEncoding(format: GLenum): GLenum {
  return format === C2.SRGB8_ALPHA8 || format === C2.SRGB8 || format === CExt.SRGB_ALPHA_EXT ? SRGB : LINEAR;
}

/** COMPONENT_TYPE (0x8211). */
function formatComponentType(format: GLenum): GLenum {
  const d = localFormatDesc(format);
  if (d.isDepth || d.isStencil) {
    if (format === C2.DEPTH_COMPONENT32F || format === C2.DEPTH32F_STENCIL8) return FLOAT;
    return UNSIGNED_INT;
  }
  if (d.isInteger) return d.isSigned ? INT : UNSIGNED_INT;
  if (d.isFloat) return FLOAT;
  if (d.isSigned) return SIGNED_NORMALIZED;
  return UNSIGNED_NORMALIZED;
}

// ---------------------------------------------------------------------------
// checkFramebufferStatus — local fallback (full rules land with framebuffer-util)
// ---------------------------------------------------------------------------

function localCheckFramebufferStatus(ctx: WebGLRenderingContext, fbo: WebGLFramebuffer): GLenum {
  void ctx;
  if (fbo._attachments.size === 0) return C1.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
  for (const rec of fbo._attachments.values()) {
    if (rec.type === 'renderbuffer') {
      if (rec.renderbuffer._surface === null) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    } else {
      const tex = rec.texture;
      if (tex._image === null) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
      if (rec.level < 0 || rec.level >= tex._image.levels.length) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
      const lvl = tex._image.levels[rec.level];
      if (!lvl || lvl.data.length === 0) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
      if (isCubeFace(rec.face) && lvl.data.length < 6) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
      if (rec.layer !== 0 && rec.layer >= lvl.depth) return C1.FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
    }
  }
  return C1.FRAMEBUFFER_COMPLETE;
}

/** Compute (and cache) the completeness of a bound FBO. */
function computeFramebufferStatus(ctx: WebGLRenderingContext, fbo: WebGLFramebuffer): GLenum {
  let status: GLenum;
  try {
    status = utilCheckFramebufferStatus(ctx, fbo);
  } catch {
    status = localCheckFramebufferStatus(ctx, fbo);
  }
  fbo._status = status;
  return status;
}

// ---------------------------------------------------------------------------
// Blit identity keys
// ---------------------------------------------------------------------------

type ImageKey =
  | { kind: 'rb'; rb: WebGLRenderbuffer }
  | { kind: 'tex'; texture: WebGLTexture; level: number; face: number; layer: number }
  | { kind: 'default' };

/** Identity key of the image attached at `attachment` (null when nothing attached). */
function attachmentImageKey(ctx: WebGLRenderingContext, fbo: WebGLFramebuffer | null, attachment: GLenum): ImageKey | null {
  if (fbo === null) {
    // Default framebuffer: color image only (BACK / COLOR_ATTACHMENT0 on W2);
    // depth/stencil of the default FB can never collide with an FBO attachment.
    if (attachment === BACK || attachment === COLOR_ATTACHMENT0 || attachment === 0x1800 /* COLOR */) {
      return { kind: 'default' };
    }
    return null;
  }
  const { rec, conflict } = resolveAttachmentRecord(ctx, fbo, attachment);
  if (conflict || rec === null) return null;
  if (rec.type === 'renderbuffer') return { kind: 'rb', rb: rec.renderbuffer };
  return { kind: 'tex', texture: rec.texture, level: rec.level, face: rec.face, layer: rec.layer };
}

function sameImage(a: ImageKey | null, b: ImageKey | null): boolean {
  if (a === null || b === null) return false;
  if (a.kind !== b.kind) return false;
  if (a.kind === 'rb') return b.kind === 'rb' && a.rb === b.rb;
  if (a.kind === 'default') return true;
  if (b.kind !== 'tex') return false;
  return a.texture === b.texture && a.level === b.level && a.face === b.face && a.layer === b.layer;
}

// ---------------------------------------------------------------------------
// Installer
// ---------------------------------------------------------------------------

export function installFramebuffersApi(proto: WebGLRenderingContext): void {
  // ---- Framebuffer objects ----
  proto.createFramebuffer = function (this: WebGLRenderingContext): WebGLFramebuffer | null {
    const ctx = this;
    // No [WebGLHandlesContextLoss]: while lost it still creates an object
    // (CTS context-lost.html nonNullTests) with NO error; isFramebuffer on it
    // → false while lost (isLost guard).
    return createObject(ctx, FboCtor);
  };

  proto.deleteFramebuffer = function (this: WebGLRenderingContext, framebuffer: WebGLFramebuffer | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (framebuffer === null || framebuffer === undefined) return;
    if (!(framebuffer instanceof WebGLFramebuffer)) throw new TypeError("Argument is not of type 'WebGLFramebuffer'");
    if (framebuffer._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (framebuffer._deleted) return;
    framebuffer._deleted = true;
    // Deleting a bound FBO resets the binding(s) to the default framebuffer.
    const s = ctx._state;
    if (s.drawFramebuffer === framebuffer) s.drawFramebuffer = null;
    if (s.readFramebuffer === framebuffer) s.readFramebuffer = null;
    ctx._resources.untrack(framebuffer);
  };

  proto.isFramebuffer = function (this: WebGLRenderingContext, framebuffer: WebGLFramebuffer | null): GLboolean {
    const ctx = this;
    if (ctx._isLost) return false;
    if (framebuffer === null || framebuffer === undefined) return false;
    if (!(framebuffer instanceof WebGLFramebuffer)) throw new TypeError("Argument is not of type 'WebGLFramebuffer'");
    // A cross-context object IS a valid WebGLFramebuffer (WebIDL conversion
    // succeeds) but is not valid for THIS context: is* returns false with NO
    // error (CTS misc/is-object.html; spec §5.14.6).
    if (framebuffer._context !== ctx) return false;
    if (framebuffer._deleted) return false;
    // Spec + CTS: an object must have been bound at least once to be true.
    return everBoundFramebuffers.has(framebuffer);
  };

  proto.bindFramebuffer = function (this: WebGLRenderingContext, target: GLenum, framebuffer: WebGLFramebuffer | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!isValidFramebufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const fbo = framebuffer === null || framebuffer === undefined ? null : validateFbo(ctx, framebuffer);
    if (framebuffer !== null && framebuffer !== undefined && fbo === null) return; // INVALID_OPERATION pushed; binding untouched
    const s = ctx._state;
    const prev = boundFramebufferForTarget(ctx, target);
    if (prev instanceof WebGLFramebuffer) prev._isBound = false;
    if (ctx._version === 2 && target === C2.READ_FRAMEBUFFER) {
      s.readFramebuffer = fbo;
    } else {
      // FRAMEBUFFER (and DRAW_FRAMEBUFFER on W2) bind the draw slot; the
      // FRAMEBUFFER target binds BOTH slots (GLES2 §4.4.1 — the bound
      // framebuffer is used for both reading and writing; WebGL1 only has
      // FRAMEBUFFER, WebGL2 keeps the ES3 both-slot behavior).
      s.drawFramebuffer = fbo;
      if (target === C1.FRAMEBUFFER) s.readFramebuffer = fbo;
    }
    if (fbo) {
      fbo._isBound = true;
      everBoundFramebuffers.add(fbo); // isFramebuffer "has been bound" marker
    }
  };

  proto.framebufferRenderbuffer = function (
    this: WebGLRenderingContext,
    target: GLenum, attachment: GLenum, renderbuffertarget: GLenum, renderbuffer: WebGLRenderbuffer | null,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    // WebIDL converts arguments left-to-right BEFORE any GL semantics run, so
    // the object argument is converted first: a wrong JS type throws TypeError
    // even when target/attachment would otherwise generate a GL error (CTS
    // bad-arguments-test.html, null-object-behaviour.html); null/undefined are
    // legal (detach).
    const rb = renderbuffer === null || renderbuffer === undefined ? null : validateRbo(ctx, renderbuffer);
    if (renderbuffer !== null && renderbuffer !== undefined && rb === null) return;
    if (!isValidFramebufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const fbo = boundFramebufferForTarget(ctx, target);
    if (fbo === null) {
      ctx._errors.push(C1.INVALID_OPERATION); // default framebuffer
      return;
    }
    if (!isValidAttachment(ctx, attachment)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (renderbuffertarget !== C1.RENDERBUFFER) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (rb !== null && ctx._version === 1 && !everBoundRenderbuffers.has(rb)) {
      // WebGL1 CTS: bindRenderbuffer must be called before attachment.
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // WebGL2 DEPTH_STENCIL_ATTACHMENT aliases the DEPTH and STENCIL slots: store
    // the SAME record object under both keys so the query side's image-identity
    // check (resolveAttachmentRecord) never sees a spurious depth/stencil
    // mismatch (identity bug — see repro-fbo-depth-stencil.ts).
    const rec: FramebufferAttachment | null =
      rb === null ? null : { type: 'renderbuffer', renderbuffer: rb };
    for (const key of attachmentKeys(ctx, attachment)) {
      if (rec === null) fbo._attachments.delete(key);
      else fbo._attachments.set(key, rec);
    }
    invalidateFboStatus(fbo);
  };

  proto.framebufferTexture2D = function (
    this: WebGLRenderingContext,
    target: GLenum, attachment: GLenum, textarget: GLenum, texture: WebGLTexture | null, level: GLint,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    // Object arg first (WebIDL left-to-right conversion — see framebufferRenderbuffer).
    const tex = texture === null || texture === undefined ? null : validateTex(ctx, texture);
    if (texture !== null && texture !== undefined && tex === null) return;
    if (!isValidFramebufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const fbo = boundFramebufferForTarget(ctx, target);
    if (fbo === null) {
      ctx._errors.push(C1.INVALID_OPERATION); // default framebuffer
      return;
    }
    if (!isValidAttachment(ctx, attachment)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (!isTextureTarget2D(textarget)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (tex !== null) {
      // The texture must have been bound at least once (to TEXTURE_2D or
      // TEXTURE_CUBE_MAP): `_target` is fixed at the first bindTexture call
      // (api/textures.ts) and stays 0 while never bound — CTS
      // state/fb-attach-implicit-target-assignment.html requires INVALID_OPERATION.
      if (tex._target === 0) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // The texture must be a 2D or cube texture matching textarget (a 3D/
      // 2D_ARRAY/multisample texture → INVALID_OPERATION; target mismatch → INVALID_OPERATION).
      const cubeTex = tex._target === C1.TEXTURE_CUBE_MAP;
      const flatTex = tex._target === C1.TEXTURE_2D;
      if (isCubeFace(textarget)) {
        if (!cubeTex) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
      } else if (!flatTex) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // Level bounds: W1 level 0 only (OES_fbo_render_mipmap relaxes); W2
      // 0 ≤ level < numLevelsFromSize(MAX_TEXTURE_SIZE) (CTS framebuffer-test.html).
      let maxLevel: number;
      if (ctx._version === 1) {
        if (level !== 0 && ctx.getExtension('OES_fbo_render_mipmap') === null) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
        maxLevel = numLevelsFromSize(ctx._state.limits.MAX_TEXTURE_SIZE);
      } else {
        maxLevel = numLevelsFromSize(ctx._state.limits.MAX_TEXTURE_SIZE);
      }
      if (level < 0 || level >= maxLevel) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      // GLES3 §4.4.2: the texture's internal format must be compatible with the
      // attachment point (depth component for DEPTH_ATTACHMENT, stencil for
      // STENCIL_ATTACHMENT, both for the W2 DEPTH_STENCIL alias).
      if (ctx._version === 2 && textureAttachmentFormatMismatch(tex, attachment)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
    }
    // Same-record sharing for DEPTH_STENCIL_ATTACHMENT (see framebufferRenderbuffer).
    const rec: FramebufferAttachment | null =
      tex === null ? null : { type: 'texture', texture: tex, level, face: textarget, layer: 0 };
    for (const key of attachmentKeys(ctx, attachment)) {
      if (rec === null) fbo._attachments.delete(key);
      else fbo._attachments.set(key, rec);
    }
    invalidateFboStatus(fbo);
  };

  proto.checkFramebufferStatus = function (this: WebGLRenderingContext, target: GLenum): GLenum {
    const ctx = this;
    if (ctx._isLost) return C1.FRAMEBUFFER_UNSUPPORTED; // spec: sentinel while lost
    if (!isValidFramebufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return 0;
    }
    const fbo = boundFramebufferForTarget(ctx, target);
    if (fbo === null) return C1.FRAMEBUFFER_COMPLETE; // default framebuffer
    return computeFramebufferStatus(ctx, fbo);
  };

  proto.getFramebufferAttachmentParameter = function (
    this: WebGLRenderingContext,
    target: GLenum, attachment: GLenum, pname: GLenum,
  ): any {
    const ctx = this;
    if (isLost(ctx)) return null;
    if (!isValidFramebufferTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    const fbo = boundFramebufferForTarget(ctx, target);
    if (fbo === null) {
      // Default framebuffer.
      if (ctx._version === 1) {
        // W1: querying the default framebuffer is INVALID_OPERATION (spec + CTS).
        ctx._errors.push(C1.INVALID_OPERATION);
        return null;
      }
      if (attachment !== BACK && attachment !== C2.DEPTH && attachment !== C2.STENCIL) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      return defaultFbAttachmentParameter(ctx, pname);
    }
    if (!isValidAttachment(ctx, attachment)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    const { rec, conflict } = resolveAttachmentRecord(ctx, fbo, attachment);
    if (conflict) {
      // W2: different images on DEPTH and STENCIL → INVALID_OPERATION.
      ctx._errors.push(C1.INVALID_OPERATION);
      return null;
    }
    if (ctx._version === 1) {
      switch (pname) {
        case C1.FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
          return rec === null ? NONE : rec.type === 'renderbuffer' ? C1.RENDERBUFFER : TEXTURE;
        case C1.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
          if (rec === null) {
            ctx._errors.push(C1.INVALID_ENUM); // CTS: only OBJECT_TYPE when no image
            return null;
          }
          return rec.type === 'renderbuffer' ? rec.renderbuffer : rec.texture;
        case C1.FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
          if (rec === null) {
            ctx._errors.push(C1.INVALID_ENUM);
            return null;
          }
          return rec.type === 'renderbuffer' ? 0 : rec.level;
        case C1.FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
          if (rec === null) {
            ctx._errors.push(C1.INVALID_ENUM);
            return null;
          }
          return rec.type === 'renderbuffer' ? 0 : rec.face;
        case CExt.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE_EXT:
          // EXT_color_buffer_half_float (WebGL1): component-type queries become
          // legal with the extension enabled (CTS ext-color-buffer-half-float.html).
          if (!ctx._extensions.has('EXT_color_buffer_half_float')) {
            ctx._errors.push(C1.INVALID_ENUM);
            return null;
          }
          if (rec === null) {
            ctx._errors.push(C1.INVALID_OPERATION);
            return null;
          }
          return attachmentParameterValue(ctx, rec, pname);
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
      }
    }
    // WebGL2 pname table.
    switch (pname) {
      case C1.FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
        return rec === null ? NONE : rec.type === 'renderbuffer' ? C1.RENDERBUFFER : TEXTURE;
      case C1.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
        if (rec === null) return null;
        return rec.type === 'renderbuffer' ? rec.renderbuffer : rec.texture;
      case C1.FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
        if (rec === null) return 0;
        return rec.type === 'renderbuffer' ? 0 : rec.level;
      case C1.FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
        if (rec === null) return 0;
        return rec.type === 'renderbuffer' ? 0 : rec.face;
      case C2.FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
        if (rec === null) return 0;
        return rec.type === 'renderbuffer' ? 0 : rec.layer;
      case C2.FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
      case C2.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
      case C2.FRAMEBUFFER_ATTACHMENT_RED_SIZE:
      case C2.FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
      case C2.FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
      case C2.FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:
      case C2.FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:
      case C2.FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE:
        if (rec === null) {
          ctx._errors.push(C1.INVALID_OPERATION); // CTS framebuffer-test.html
          return null;
        }
        return attachmentParameterValue(ctx, rec, pname);
      case TEXTURE_SAMPLES_EXT:
        // WEBGL_multisampled_render_to_texture (WebGL2-only): sample count of a
        // texture attachment, 0 otherwise (extension spec).
        if (!ctx._extensions.has('WEBGL_multisampled_render_to_texture')) {
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
        }
        if (rec === null) return 0;
        return rec.type === 'texture' ? rec.texture._msaaSamples : 0;
      default:
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
    }
  };

  // ---- Renderbuffer objects ----
  proto.createRenderbuffer = function (this: WebGLRenderingContext): WebGLRenderbuffer | null {
    const ctx = this;
    // No [WebGLHandlesContextLoss]: while lost it still creates an object
    // (CTS context-lost.html nonNullTests) with NO error; isRenderbuffer on it
    // → false while lost (isLost guard).
    return createObject(ctx, RboCtor);
  };

  proto.deleteRenderbuffer = function (this: WebGLRenderingContext, renderbuffer: WebGLRenderbuffer | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (renderbuffer === null || renderbuffer === undefined) return;
    if (!(renderbuffer instanceof WebGLRenderbuffer)) throw new TypeError("Argument is not of type 'WebGLRenderbuffer'");
    if (renderbuffer._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (renderbuffer._deleted) return;
    renderbuffer._deleted = true;
    if (ctx._state.renderbuffer === renderbuffer) ctx._state.renderbuffer = null;
    // Detach ONLY from the currently bound framebuffer(s) (GLES2/WebGL spec:
    // deletion while attached to the bound FBO detaches it there; attachments
    // in UNBOUND framebuffers persist until detached or the FBO is deleted —
    // CTS context/deleted-object-behavior.html testUnboundFBORenderbuffer and
    // misc/object-deletion-behaviour.html both hard-require the persistence).
    const bound = new Set<WebGLFramebuffer>();
    if (ctx._state.drawFramebuffer !== null) bound.add(ctx._state.drawFramebuffer);
    if (ctx._version === 2 && ctx._state.readFramebuffer !== null) bound.add(ctx._state.readFramebuffer);
    for (const fbo of bound) {
      let changed = false;
      for (const [key, rec] of fbo._attachments) {
        if (rec.type === 'renderbuffer' && rec.renderbuffer === renderbuffer) {
          fbo._attachments.delete(key);
          changed = true;
        }
      }
      if (changed) invalidateFboStatus(fbo);
    }
    ctx._resources.untrack(renderbuffer);
  };

  proto.isRenderbuffer = function (this: WebGLRenderingContext, renderbuffer: WebGLRenderbuffer | null): GLboolean {
    const ctx = this;
    if (ctx._isLost) return false;
    if (renderbuffer === null || renderbuffer === undefined) return false;
    if (!(renderbuffer instanceof WebGLRenderbuffer)) throw new TypeError("Argument is not of type 'WebGLRenderbuffer'");
    // Cross-context → false with NO error (see isFramebuffer).
    if (renderbuffer._context !== ctx) return false;
    if (renderbuffer._deleted) return false;
    // Spec + CTS (misc/is-object.html): must have been bound at least once.
    return everBoundRenderbuffers.has(renderbuffer);
  };

  proto.bindRenderbuffer = function (this: WebGLRenderingContext, target: GLenum, renderbuffer: WebGLRenderbuffer | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (target !== C1.RENDERBUFFER) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const rb = renderbuffer === null || renderbuffer === undefined ? null : validateRbo(ctx, renderbuffer);
    if (renderbuffer !== null && renderbuffer !== undefined && rb === null) return;
    if (rb !== null) everBoundRenderbuffers.add(rb); // W1 attach precondition
    ctx._state.renderbuffer = rb;
  };

  proto.renderbufferStorage = function (
    this: WebGLRenderingContext,
    target: GLenum, internalformat: GLenum, width: GLsizei, height: GLsizei,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (target !== C1.RENDERBUFFER) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const rb = ctx._state.renderbuffer;
    if (rb === null) {
      ctx._errors.push(C1.INVALID_OPERATION); // no renderbuffer bound
      return;
    }
    if (width < 0 || height < 0 || width > ctx._state.limits.MAX_RENDERBUFFER_SIZE || height > ctx._state.limits.MAX_RENDERBUFFER_SIZE) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    if (!isValidRenderbufferFormat(ctx, internalformat)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    allocateRenderbufferSurface(ctx, rb, internalformat, width, height);
    rb._internalformat = internalformat;
    rb._width = width;
    rb._height = height;
    rb._samples = 0;
    invalidateFboStatuses(ctx);
  };

  proto.getRenderbufferParameter = function (this: WebGLRenderingContext, target: GLenum, pname: GLenum): any {
    const ctx = this;
    if (isLost(ctx)) return null;
    if (target !== C1.RENDERBUFFER) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    const rb = ctx._state.renderbuffer;
    if (rb === null) {
      ctx._errors.push(C1.INVALID_OPERATION); // no renderbuffer bound (spec)
      return null;
    }
    switch (pname) {
      case C1.RENDERBUFFER_WIDTH: return rb._width;
      case C1.RENDERBUFFER_HEIGHT: return rb._height;
      case C1.RENDERBUFFER_INTERNAL_FORMAT: return rb._internalformat;
      case C2.RENDERBUFFER_SAMPLES: return rb._samples;
      case C1.RENDERBUFFER_RED_SIZE:
      case C1.RENDERBUFFER_GREEN_SIZE:
      case C1.RENDERBUFFER_BLUE_SIZE:
      case C1.RENDERBUFFER_ALPHA_SIZE:
      case C1.RENDERBUFFER_DEPTH_SIZE:
      case C1.RENDERBUFFER_STENCIL_SIZE: {
        const bits = formatComponentBits(rb._internalformat);
        return bits[pname - C1.RENDERBUFFER_RED_SIZE];
      }
      default:
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
    }
  };

  // ---- WebGL2-only methods (guarded: installer runs on both prototypes) ----
  const p2 = proto as unknown as WebGL2RenderingContext;

  if ('framebufferTextureLayer' in proto) {
    p2.framebufferTextureLayer = function (
      this: WebGL2RenderingContext,
      target: GLenum, attachment: GLenum, texture: WebGLTexture | null, level: GLint, layer: GLint,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      // Object arg first (WebIDL left-to-right conversion — see framebufferRenderbuffer).
      const tex = texture === null || texture === undefined ? null : validateTex(ctx, texture);
      if (texture !== null && texture !== undefined && tex === null) return;
      if (!isValidFramebufferTarget(ctx, target)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const fbo = boundFramebufferForTarget(ctx, target);
      if (fbo === null) {
        ctx._errors.push(C1.INVALID_OPERATION); // default framebuffer
        return;
      }
      if (!isValidAttachment(ctx, attachment)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      if (tex !== null) {
        // Only 2D_ARRAY / 3D textures can be attached by layer (CTS: 2D → INVALID_OPERATION),
        // and the texture must have been bound at least once (spec mirrors
        // framebufferTexture2D; `_target` is fixed at first bindTexture).
        if (tex._target === 0 ||
            (tex._target !== C2.TEXTURE_2D_ARRAY && tex._target !== C2.TEXTURE_3D)) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
        const maxLevel = numLevelsFromSize(ctx._state.limits.MAX_3D_TEXTURE_SIZE);
        if (level < 0 || level >= maxLevel) {
          ctx._errors.push(C1.INVALID_VALUE); // CTS framebuffer-texture-layer.html
          return;
        }
        // Layer must be within the texture's depth (CTS: -1 and depth → INVALID_VALUE).
        const depth = tex._image ? tex._image.depth : 1;
        if (layer < 0 || layer >= depth) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
        // GLES3 §4.4.2.4: same depth/stencil component rule as framebufferTexture2D.
        if (textureAttachmentFormatMismatch(tex, attachment)) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
      }
      // Same-record sharing for DEPTH_STENCIL_ATTACHMENT (see framebufferRenderbuffer).
      const rec: FramebufferAttachment | null =
        tex === null ? null : { type: 'texture', texture: tex, level, face: C2.TEXTURE_2D_ARRAY, layer };
      for (const key of attachmentKeys(ctx, attachment)) {
        if (rec === null) fbo._attachments.delete(key);
        else fbo._attachments.set(key, rec);
      }
      invalidateFboStatus(fbo);
    };
  }

  if ('renderbufferStorageMultisample' in proto) {
    p2.renderbufferStorageMultisample = function (
      this: WebGL2RenderingContext,
      target: GLenum, samples: GLsizei, internalformat: GLenum, width: GLsizei, height: GLsizei,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (target !== C1.RENDERBUFFER) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const rb = ctx._state.renderbuffer;
      if (rb === null) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (width < 0 || height < 0 || width > ctx._state.limits.MAX_RENDERBUFFER_SIZE || height > ctx._state.limits.MAX_RENDERBUFFER_SIZE) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (samples < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (samples > ctx._state.limits.MAX_SAMPLES) {
        ctx._errors.push(C1.INVALID_OPERATION); // WebGL2 spec + multisampled.ts factory
        return;
      }
      if (!isValidRenderbufferFormat(ctx, internalformat)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      allocateRenderbufferSurface(ctx, rb, internalformat, width, height);
      rb._internalformat = internalformat;
      rb._width = width;
      rb._height = height;
      rb._samples = samples; // storage stays single-sampled; resolve at blit
      invalidateFboStatuses(ctx);
    };
  }

  if ('drawBuffers' in proto) {
    p2.drawBuffers = function (this: WebGL2RenderingContext, buffers: GLenum[]): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const s = ctx._state;
      // WebIDL sequence<GLenum> conversion (mirrors the WEBGL_draw_buffers factory).
      let arr: number[];
      try {
        arr = Array.from(buffers as unknown as ArrayLike<unknown>, (v) => {
          const n = Number(v);
          if (!Number.isFinite(n)) throw new TypeError('drawBuffers: invalid GLenum in sequence');
          return n >>> 0;
        });
      } catch {
        throw new TypeError('drawBuffers: buffers is not a sequence<GLenum>');
      }
      if (arr.length > s.limits.MAX_DRAW_BUFFERS) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (arr.length === 0) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (s.drawFramebuffer === null) {
        // Default framebuffer: exactly [BACK] or [NONE].
        if (arr.length !== 1 || (arr[0] !== BACK && arr[0] !== NONE)) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
        s.drawBuffers = arr;
        return;
      }
      // FBO: NONE or strictly-increasing COLOR_ATTACHMENTi < MAX_COLOR_ATTACHMENTS.
      let last = -1;
      for (const b of arr) {
        if (b === NONE) continue;
        if (b === BACK) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
        const idx = b - COLOR_ATTACHMENT0;
        if (idx < 0 || idx >= s.limits.MAX_COLOR_ATTACHMENTS) {
          ctx._errors.push(C1.INVALID_ENUM);
          return;
        }
        if (idx <= last) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
        last = idx;
      }
      s.drawBuffers = arr;
    };
  }

  if ('readBuffer' in proto) {
    p2.readBuffer = function (this: WebGL2RenderingContext, src: GLenum): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const s = ctx._state;
      if (src === NONE) {
        s.readBuffer = src;
        return;
      }
      if (s.readFramebuffer === null) {
        // Default framebuffer: NONE or BACK (CTS readbuffer.html).
        if (src === BACK) {
          s.readBuffer = src;
          return;
        }
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (src === BACK) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const idx = src - COLOR_ATTACHMENT0;
      if (idx < 0) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      if (idx >= s.limits.MAX_COLOR_ATTACHMENTS) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      s.readBuffer = src;
    };
  }

  if ('invalidateFramebuffer' in proto) {
    p2.invalidateFramebuffer = function (this: WebGL2RenderingContext, target: GLenum, attachments: GLenum[]): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (!isValidFramebufferTarget(ctx, target)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const arr = toGlenumSequence(attachments);
      if (!validateInvalidateAttachments(ctx, target, arr)) return;
      // No-op (software renderer; contents persist).
    };

    p2.invalidateSubFramebuffer = function (
      this: WebGL2RenderingContext,
      target: GLenum, attachments: GLenum[], x: GLint, y: GLint, width: GLsizei, height: GLsizei,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (!isValidFramebufferTarget(ctx, target)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      const arr = toGlenumSequence(attachments);
      if (!validateInvalidateAttachments(ctx, target, arr)) return;
      if (width < 0 || height < 0) {
        ctx._errors.push(C1.INVALID_VALUE); // CTS invalidate-framebuffer.html
        return;
      }
      void x; void y;
      // No-op (software renderer).
    };
  }

  if ('blitFramebuffer' in proto) {
    p2.blitFramebuffer = function (
      this: WebGL2RenderingContext,
      srcX0: GLint, srcY0: GLint, srcX1: GLint, srcY1: GLint,
      dstX0: GLint, dstY0: GLint, dstX1: GLint, dstY1: GLint,
      mask: GLbitfield, filter: GLenum,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const s = ctx._state;
      const allowedMask = C1.COLOR_BUFFER_BIT | C1.DEPTH_BUFFER_BIT | C1.STENCIL_BUFFER_BIT;
      if ((mask & ~allowedMask) !== 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return;
      }
      if (filter !== C1.NEAREST && filter !== LINEAR) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      if (filter === LINEAR && (mask & (C1.DEPTH_BUFFER_BIT | C1.STENCIL_BUFFER_BIT)) !== 0) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      const readFbo = s.readFramebuffer;
      const drawFbo = s.drawFramebuffer;
      if (readFbo === drawFbo) {
        ctx._errors.push(C1.INVALID_OPERATION); // same FBO object (incl. both default)
        return;
      }
      // Both bound framebuffers must be complete (GLES3 §4.4.4:
      // INVALID_FRAMEBUFFER_OPERATION when either is incomplete).
      if (readFbo !== null && computeFramebufferStatus(ctx, readFbo) !== C1.FRAMEBUFFER_COMPLETE) {
        ctx._errors.push(C1.INVALID_FRAMEBUFFER_OPERATION);
        return;
      }
      if (drawFbo !== null && computeFramebufferStatus(ctx, drawFbo) !== C1.FRAMEBUFFER_COMPLETE) {
        ctx._errors.push(C1.INVALID_FRAMEBUFFER_OPERATION);
        return;
      }
      // Identical source/dest image for any mask bit → INVALID_OPERATION
      // (CTS blitframebuffer-test.html: same texture level/face/layer or same
      // renderbuffer; color also requires the image to be among the draw buffers).
      if ((mask & C1.COLOR_BUFFER_BIT) !== 0) {
        const readKey = attachmentImageKey(ctx, readFbo, readFbo === null ? BACK : s.readBuffer);
        if (readKey !== null) {
          const dbList = drawFbo === null ? [BACK] : s.drawBuffers;
          for (const db of dbList) {
            if (db === NONE) continue;
            if (sameImage(readKey, attachmentImageKey(ctx, drawFbo, db))) {
              ctx._errors.push(C1.INVALID_OPERATION);
              return;
            }
          }
        }
      }
      if ((mask & C1.DEPTH_BUFFER_BIT) !== 0 &&
          sameImage(attachmentImageKey(ctx, readFbo, DEPTH_ATTACHMENT), attachmentImageKey(ctx, drawFbo, DEPTH_ATTACHMENT))) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if ((mask & C1.STENCIL_BUFFER_BIT) !== 0 &&
          sameImage(attachmentImageKey(ctx, readFbo, STENCIL_ATTACHMENT), attachmentImageKey(ctx, drawFbo, STENCIL_ATTACHMENT))) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // Multisample rules (CTS-verified): resolve (read MS + draw single) is
      // legal; draw MS with read single OR read MS → INVALID_OPERATION.
      const readMS = readFbo !== null && readFbo._multisampled;
      const drawMS = drawFbo !== null && drawFbo._multisampled;
      if (drawMS) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      try {
        executeBlitFramebuffer(ctx, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
      } catch {
        ctx._errors.push(C1.INVALID_OPERATION); // engine stub safety
      }
    };
  }
}

// ---------------------------------------------------------------------------
// Module-private helpers (defined after the installer for readability)
// ---------------------------------------------------------------------------

/** WebGL1 rule: a renderbuffer must have been bound before it can be attached. */
const everBoundRenderbuffers = new WeakSet<WebGLRenderbuffer>();

/** isFramebuffer "has been bound at least once" marker (spec + CTS misc/is-object.html). */
const everBoundFramebuffers = new WeakSet<WebGLFramebuffer>();

/** W2 default-framebuffer attachment parameter values. */
function defaultFbAttachmentParameter(ctx: WebGLRenderingContext, pname: GLenum): any {
  switch (pname) {
    case C1.FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE:
      return FRAMEBUFFER_DEFAULT;
    case C1.FRAMEBUFFER_ATTACHMENT_OBJECT_NAME:
      return null;
    case C1.FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
    case C1.FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
    case C2.FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
      return 0;
    case C2.FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
      return LINEAR;
    case C2.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
      return UNSIGNED_NORMALIZED;
    case C2.FRAMEBUFFER_ATTACHMENT_RED_SIZE:
    case C2.FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
    case C2.FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
    case C2.FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE: {
      const fb = ctx._defaultFB;
      const format = fb ? fb.color.format : 0;
      const bits = formatComponentBits(format);
      return bits[pname - C2.FRAMEBUFFER_ATTACHMENT_RED_SIZE];
    }
    case C2.FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE: {
      const fb = ctx._defaultFB;
      const format = fb && fb.depth ? fb.depth.format : 0;
      return formatComponentBits(format)[4];
    }
    case C2.FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: {
      const fb = ctx._defaultFB;
      const format = fb && fb.stencil ? fb.stencil.format : 0;
      return formatComponentBits(format)[5];
    }
    default:
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
  }
}

/** W2 attachment value pnames (COLOR_ENCODING..STENCIL_SIZE) for an attached image. */
function attachmentParameterValue(ctx: WebGLRenderingContext, rec: FramebufferAttachment, pname: GLenum): any {
  const format = attachmentInternalFormat(rec);
  switch (pname) {
    case C2.FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
      return formatColorEncoding(format);
    case C2.FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
      // Combined depth-stencil images have no single component type — the
      // query is INVALID_OPERATION (GLES3 + CTS ext-color-buffer-half-float.html).
      if (format === C1.DEPTH_STENCIL || format === C2.DEPTH24_STENCIL8 || format === C2.DEPTH32F_STENCIL8) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return null;
      }
      // WebGL1 unsized float textures (RGBA/RGB + FLOAT/HALF_FLOAT_OES storage)
      // report FLOAT even though the unsized internal format has no sized desc.
      if (rec.type === 'texture' && !localFormatDesc(format).isFloat &&
          rec.texture._image?.info?.isFloat) {
        return FLOAT;
      }
      return formatComponentType(format);
    default: {
      const bits = formatComponentBits(format);
      const idx = pname - C2.FRAMEBUFFER_ATTACHMENT_RED_SIZE; // RED..ALPHA = 0..3, DEPTH = 4, STENCIL = 5
      if (idx >= 0 && idx <= 5) return bits[idx];
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
  }
}

/** WebIDL sequence<GLenum> conversion (throws TypeError for non-finite elements). */
function toGlenumSequence(attachments: GLenum[]): number[] {
  return Array.from(attachments as unknown as ArrayLike<unknown>, (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) throw new TypeError('attachments: invalid GLenum in sequence');
    return n >>> 0;
  });
}

/** invalidate* attachment validation (CTS invalidate-framebuffer.html). */
function validateInvalidateAttachments(ctx: WebGLRenderingContext, target: GLenum, arr: number[]): boolean {
  const isDefault = boundFramebufferForTarget(ctx, target) === null;
  const maxColor = ctx._state.limits.MAX_COLOR_ATTACHMENTS;
  for (const a of arr) {
    if (isDefault) {
      if (a === C2.COLOR || a === C2.DEPTH || a === C2.STENCIL) continue;
      ctx._errors.push(C1.INVALID_OPERATION);
      return false;
    }
    if (a === DEPTH_ATTACHMENT || a === STENCIL_ATTACHMENT || a === DEPTH_STENCIL_ATTACHMENT) continue;
    const idx = a - COLOR_ATTACHMENT0;
    if (idx >= 0 && idx < maxColor) continue;
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  return true;
}
