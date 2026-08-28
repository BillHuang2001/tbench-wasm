/**
 * src/gl/api/draw.ts — draw calls, clear, readPixels, flush/finish.
 *
 * Owns: drawArrays, drawElements, drawArraysInstanced (W2), drawElementsInstanced
 * (W2), drawRangeElements (W2), clear, clearBuffer{fv,iv,uiv,fi} (W2), flush,
 * finish, readPixels, and the four WEBGL_multi_draw methods (installed
 * unconditionally on both context prototypes; the extension object delegates
 * here too). The multi-draw engine entries — executeMultiDrawArrays/Elements
 * (+Instanced), including the validate-all-subdraws-first contract — live in
 * ../draw.
 *
 * Validation (spec order) before delegating to the draw.ts engine:
 *  - drawArrays/drawElements(+Instanced/drawRangeElements): mode
 *    (INVALID_ENUM), first/count/instanceCount ≥ 0 (INVALID_VALUE), type/
 *    offset for indexed draws (INVALID_ENUM / INVALID_VALUE / INVALID_OPERATION),
 *    drawRangeElements start ≤ end (INVALID_VALUE) + count ≤ end-start+1
 *    (INVALID_OPERATION), element-array-buffer bounds (INVALID_OPERATION), then
 *    shared preconditions (program linked, TF mode match, attrib arrays backed,
 *    FBO complete) — see validateDrawArrays/validateDrawElements in ../draw.
 *  - Transform feedback active: draw mode must equal the beginTransformFeedback
 *    mode (INVALID_OPERATION); TF draws bypass the rasterizer (engine handles).
 *  - clear: mask bits ⊆ {COLOR, DEPTH, STENCIL}_BUFFER_BIT (INVALID_VALUE).
 *  - clearBuffer*: buffer ∈ {COLOR, DEPTH, STENCIL, DEPTH_STENCIL} (INVALID_ENUM
 *    for anything else — GL_COLOR_INT/GL_COLOR_UINT are not WebGL enums);
 *    drawbuffer < MAX_DRAW_BUFFERS (INVALID_VALUE); DEPTH/STENCIL/DEPTH_STENCIL
 *    only target drawbuffer 0 (INVALID_OPERATION); per-function buffer match
 *    (INVALID_OPERATION); for COLOR the function must match the attachment
 *    class (float/fixed → fv, signed int → iv, unsigned int → uiv —
 *    INVALID_OPERATION on mismatch, nothing cleared; missing attachment or
 *    drawBuffers[i] === NONE → no-op NO_ERROR); values length ≥ 4 for color /
 *    ≥ 1 for depth-stencil (INVALID_VALUE); clearBufferfi → DEPTH_STENCIL +
 *    drawbuffer 0.
 *  - readPixels: pixels null → INVALID_VALUE (TypeError for non-view
 *    non-number args); width/height ≥ 0 (INVALID_VALUE); format/type enum
 *    validity (INVALID_ENUM — WebGL1 via format/type sets, WebGL2 via the
 *    GLES3 (format,type) PAIR table); format/type compatibility with the read
 *    buffer (INVALID_OPERATION — default framebuffer accepts only
 *    RGBA/UNSIGNED_BYTE); view class must match type (INVALID_OPERATION);
 *    pixel-store constraints + destination size (INVALID_OPERATION);
 *    PIXEL_PACK_BUFFER bound → the ArrayBufferView overload is
 *    INVALID_OPERATION (WebGL2 spec), the GLintptr-offset overload is
 *    INVALID_OPERATION with NO PIXEL_PACK_BUFFER bound, INVALID_VALUE for
 *    offset < 0, INVALID_OPERATION when the data store is absent or too small
 *    (remainder check per spec) — same enum/combo validation as the view path.
 *  - flush/finish: no-ops (synchronous renderer).
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';
import { C1, C2 } from '../constants';
import {
  executeClear,
  executeClearBuffer,
  executeDraw,
  executeMultiDrawArrays,
  executeMultiDrawArraysInstanced,
  executeMultiDrawElements,
  executeMultiDrawElementsInstanced,
  executeReadPixels,
  extSupported,
  transferToImageBitmapSnapshot,
  validateDrawArrays,
  validateDrawElements,
} from '../draw';
import type {
  GLbitfield, GLenum, GLfloat, GLint, GLintptr, GLsizei, GLuint,
  Float32List, Int32List, Uint32List,
} from '../types';

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

// clearBuffer* buffer enums per WebGL2 §clearBuffer: COLOR/DEPTH/STENCIL/
// DEPTH_STENCIL only (GL_COLOR_INT/GL_COLOR_UINT are GL 4.x enums, NOT WebGL
// enums — they must produce INVALID_ENUM).
const CLEAR_BUFFERS = new Set<number>([
  C2.COLOR, C2.DEPTH, C2.STENCIL, C2.DEPTH_STENCIL,
]);

/** clearBufferfv-compatible buffers (float/fixed color, depth, stencil). */
const CLEAR_FV = new Set<number>([C2.COLOR, C2.DEPTH, C2.STENCIL]);
/** clearBufferiv-compatible buffers (signed-integer color, depth, stencil). */
const CLEAR_IV = new Set<number>([C2.COLOR, C2.DEPTH, C2.STENCIL]);
/** clearBufferuiv-compatible buffers (unsigned-integer color). */
const CLEAR_UIV = new Set<number>([C2.COLOR]);

/** Signed-integer color-renderable internal formats (clearBufferiv targets). */
const SINT_COLOR_FORMATS = new Set<number>([
  C2.R8I, C2.R16I, C2.R32I,
  C2.RG8I, C2.RG16I, C2.RG32I,
  C2.RGB8I, C2.RGB16I, C2.RGB32I,
  C2.RGBA8I, C2.RGBA16I, C2.RGBA32I,
]);
/** Unsigned-integer color-renderable internal formats (clearBufferuiv targets). */
const UINT_COLOR_FORMATS = new Set<number>([
  C2.R8UI, C2.R16UI, C2.R32UI,
  C2.RG8UI, C2.RG16UI, C2.RG32UI,
  C2.RGB8UI, C2.RGB16UI, C2.RGB32UI,
  C2.RGBA8UI, C2.RGBA16UI, C2.RGBA32UI,
  C2.RGB10_A2UI,
]);

/** WebIDL Float32List/Int32List/Uint32List → typed array (TypeError on junk). */
function toList<T extends Float32Array | Int32Array | Uint32Array>(
  values: Float32List | Int32List | Uint32List,
  Ctor: new (src: ArrayLike<number>) => T,
  name: string,
): T {
  if (values instanceof Ctor) return values;
  if (Array.isArray(values) || ArrayBuffer.isView(values)) {
    return new Ctor(values as ArrayLike<number>);
  }
  throw new TypeError(`Argument is not of type '${name}'`);
}

/** Validate buffer enum + drawbuffer for a clearBuffer* call (shared). */
function validateClearBufferArgs(
  ctx: WebGLRenderingContext, buffer: GLenum, drawbuffer: GLint,
): boolean {
  const s = ctx._state;
  if (!CLEAR_BUFFERS.has(buffer)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (drawbuffer < 0 || drawbuffer >= s.limits.MAX_DRAW_BUFFERS) {
    ctx._errors.push(C1.INVALID_VALUE);
    return false;
  }
  // ES3: DEPTH/STENCIL/DEPTH_STENCIL clears only target drawbuffer 0.
  if (buffer !== C2.COLOR && drawbuffer !== 0) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  return true;
}

/**
 * Format class of the color attachment at `drawbuffer` of the DRAW framebuffer:
 * 'sint' / 'uint' / 'float' (float/fixed) — or 'missing' when drawBuffers[i] is
 * NONE or there is no attachment (spec: clearing a missing attachment of a
 * complete framebuffer clears nothing and generates NO error). The default
 * framebuffer is fixed-point RGBA8 → 'float'.
 */
function clearColorAttachmentClass(
  ctx: WebGLRenderingContext, drawbuffer: GLint,
): 'sint' | 'uint' | 'float' | 'missing' {
  const s = ctx._state;
  const fbo = s.drawFramebuffer;
  if (fbo === null) return 'float';
  const db = s.drawBuffers[drawbuffer] ?? C1.NONE;
  if (db === C1.NONE) return 'missing';
  const att = fbo._attachments.get(db);
  if (!att) return 'missing';
  const internalFormat = att.type === 'renderbuffer'
    ? att.renderbuffer._internalformat
    : (att.texture._image?.internalFormat ?? 0);
  if (SINT_COLOR_FORMATS.has(internalFormat)) return 'sint';
  if (UINT_COLOR_FORMATS.has(internalFormat)) return 'uint';
  return 'float';
}

// ---------------------------------------------------------------------------
// readPixels format/type tables
// ---------------------------------------------------------------------------

/** Formats accepted as readPixels `format` (WebGL1). */
const RP_FORMATS_1 = new Set<number>([
  C1.RGBA, C1.RGB, C1.LUMINANCE, C1.LUMINANCE_ALPHA, C1.ALPHA,
]);

/** Types accepted as readPixels `type` (WebGL1; incl. half-float enums). */
const RP_TYPES_1 = new Set<number>([
  C1.UNSIGNED_BYTE, C1.BYTE, C1.UNSIGNED_SHORT, C1.SHORT, C1.UNSIGNED_INT, C1.INT,
  C1.FLOAT, C1.UNSIGNED_SHORT_5_6_5, C1.UNSIGNED_SHORT_4_4_4_4, C1.UNSIGNED_SHORT_5_5_5_1,
  0x140b /* HALF_FLOAT */, 0x8d61 /* HALF_FLOAT_OES */,
]);

/**
 * WebGL2 readPixels (format, type) PAIR table — the GLES3 ReadPixels table
 * (ES 3.0 §4.3.2). Any pair NOT listed → INVALID_ENUM (this replaces the
 * WebGL1-style independent format/type sets for WebGL2; e.g. LUMINANCE as a
 * format, RGBA4 as a format, or RGBA/UNSIGNED_INT_24_8 are all INVALID_ENUM —
 * CTS read-pixels-into-pixel-pack-buffer.html checkFormatAndType).
 */
const RP_PAIRS_2 = new Set<number>();
{
  const P = (format: number, ...types: number[]): void => {
    for (const t of types) RP_PAIRS_2.add((format << 16) | t);
  };
  P(C1.RGBA, C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_5_5_1, C1.UNSIGNED_SHORT_4_4_4_4,
    C2.HALF_FLOAT, C1.FLOAT, C2.UNSIGNED_INT_2_10_10_10_REV);
  P(C2.RGBA_INTEGER, C1.BYTE, C1.UNSIGNED_BYTE, C1.SHORT, C1.UNSIGNED_SHORT, C1.INT, C1.UNSIGNED_INT);
  P(C1.RGB, C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_6_5,
    C2.HALF_FLOAT, C1.FLOAT, C2.UNSIGNED_INT_2_10_10_10_REV);
  P(C2.RGB_INTEGER, C1.BYTE, C1.UNSIGNED_BYTE, C1.SHORT, C1.UNSIGNED_SHORT, C1.INT, C1.UNSIGNED_INT);
  P(C2.RG, C1.UNSIGNED_BYTE, C2.HALF_FLOAT, C1.FLOAT);
  P(C2.RG_INTEGER, C1.BYTE, C1.UNSIGNED_BYTE, C1.SHORT, C1.UNSIGNED_SHORT, C1.INT, C1.UNSIGNED_INT);
  P(C2.RED, C1.UNSIGNED_BYTE, C2.HALF_FLOAT, C1.FLOAT);
  P(C2.RED_INTEGER, C1.BYTE, C1.UNSIGNED_BYTE, C1.SHORT, C1.UNSIGNED_SHORT, C1.INT, C1.UNSIGNED_INT);
  P(C1.DEPTH_COMPONENT, C1.UNSIGNED_SHORT, C1.UNSIGNED_INT);
  P(C2.DEPTH_STENCIL, C2.UNSIGNED_INT_24_8, C2.FLOAT_32_UNSIGNED_INT_24_8_REV);
  // ALPHA is a WebGL1-inherited format (WebGL2 spec: "Only differences from
  // WebGL 1.0 are described here"); the CTS expects ALPHA/UNSIGNED_BYTE to be
  // enum-valid and to fail only the read-buffer compatibility gate
  // (INVALID_OPERATION — read-pixels-into-pixel-pack-buffer.html).
  P(C1.ALPHA, C1.UNSIGNED_BYTE);
}

function rpPairKey(format: GLenum, type: GLenum): number {
  return (format << 16) | type;
}

/** The ArrayBufferView class required by a readPixels `type` ('u8' = Uint8Array|Uint8ClampedArray). */
function expectedViewForType(type: GLenum): (new (n: number) => ArrayBufferView) | 'u8' | null {
  switch (type) {
    case C1.UNSIGNED_BYTE: return 'u8';
    case C1.BYTE: return Int8Array;
    case C1.UNSIGNED_SHORT: case C1.UNSIGNED_SHORT_5_6_5:
    case C1.UNSIGNED_SHORT_4_4_4_4: case C1.UNSIGNED_SHORT_5_5_5_1:
    case 0x140b /* HALF_FLOAT */: case 0x8d61 /* HALF_FLOAT_OES */:
      return Uint16Array;
    case C1.SHORT: return Int16Array;
    case C1.UNSIGNED_INT: case C2.UNSIGNED_INT_2_10_10_10_REV: case C2.UNSIGNED_INT_24_8:
      return Uint32Array;
    case C1.INT: return Int32Array;
    case C1.FLOAT: case C2.FLOAT_32_UNSIGNED_INT_24_8_REV: return Float32Array;
    default: return null;
  }
}

/** Float readPixels combos are gated on the color-buffer-float extensions. */
function floatReadOK(ctx: WebGLRenderingContext, type: GLenum, halfFloat: boolean): boolean {
  if (type === C1.FLOAT) return extSupported(ctx, 'EXT_color_buffer_float');
  if (type === C2.HALF_FLOAT || type === 0x8d61 /* HALF_FLOAT_OES */) {
    return halfFloat &&
      (extSupported(ctx, 'EXT_color_buffer_float') || extSupported(ctx, 'EXT_color_buffer_half_float'));
  }
  if (type === C2.UNSIGNED_INT_10F_11F_11F_REV) return extSupported(ctx, 'EXT_color_buffer_float');
  return false;
}

/**
 * WebGL1 unsized float-storage attachment (RGBA/RGB/LUMINANCE/LA/ALPHA
 * uploaded with FLOAT or HALF_FLOAT_OES): readPixels accepts RGBA with
 * FLOAT (OES_texture_float / EXT_color_buffer_half_float) and with
 * HALF_FLOAT/HALF_FLOAT_OES (OES_texture_half_float / EXT_color_buffer_half_float).
 */
function floatStorageReadOK(ctx: WebGLRenderingContext, format: GLenum, type: GLenum): boolean {
  if (format !== C1.RGBA) return false;
  if (type === C1.FLOAT) {
    return extSupported(ctx, 'OES_texture_float') || extSupported(ctx, 'EXT_color_buffer_half_float');
  }
  if (type === 0x140b /* HALF_FLOAT */ || type === 0x8d61 /* HALF_FLOAT_OES */) {
    return extSupported(ctx, 'OES_texture_half_float') || extSupported(ctx, 'EXT_color_buffer_half_float');
  }
  return false;
}

/** True when (format, type) is a legal readPixels combo for the attachment's
 *  internal format. `floatStorage` marks W1 UNSIZED float-storage levels
 *  (info.isFloat on the attached texture/renderbuffer surface).
 *
 *  Universal expansions per the GLES3 ReadPixels table (ES 3.0 §4.3.2,
 *  Table 3.2 — CTS conformance2/reading/read-pixels-from-fbo-test.html):
 *   - RGBA/UNSIGNED_BYTE is accepted for EVERY normalized color attachment
 *     (R/RG/RGB attachments expand to RGBA on read).
 *   - RGBA_INTEGER/INT (signed) and RGBA_INTEGER/UNSIGNED_INT (unsigned) are
 *     accepted for EVERY integer attachment, with type conversion across
 *     widths (e.g. R8I read as RGBA_INTEGER/INT, R16UI as RGBA_INTEGER/
 *     UNSIGNED_INT). The integer-vs-normalized boundary is NOT relaxed:
 *     integer formats never accept plain RGBA, and normalized/float formats
 *     never accept *_INTEGER formats. */
function readComboOK(
  ctx: WebGLRenderingContext, internalFormat: GLenum, format: GLenum, type: GLenum,
  floatStorage?: boolean,
): boolean {
  const universalRGBA = format === C1.RGBA && type === C1.UNSIGNED_BYTE;
  const universalSInt = format === C2.RGBA_INTEGER && type === C1.INT;
  const universalUInt = format === C2.RGBA_INTEGER && type === C1.UNSIGNED_INT;
  switch (internalFormat) {
    // Unsigned normalized color
    case C1.RGBA: case C1.RGBA8: case C2.RGBA8: case C2.SRGB8_ALPHA8:
      if (floatStorage) return floatStorageReadOK(ctx, format, type);
      return universalRGBA;
    case C1.RGBA4:
      return format === C1.RGBA && (type === C1.UNSIGNED_BYTE || type === C1.UNSIGNED_SHORT_4_4_4_4);
    case C1.RGB5_A1:
      return format === C1.RGBA && (type === C1.UNSIGNED_BYTE || type === C1.UNSIGNED_SHORT_5_5_5_1);
    case C1.RGB565:
      return (format === C1.RGB && (type === C1.UNSIGNED_BYTE || type === C1.UNSIGNED_SHORT_5_6_5)) ||
        universalRGBA;
    case C1.RGB:
      if (floatStorage) return floatStorageReadOK(ctx, format, type);
      // WebGL 1.0 §5.14.12: RGBA/UNSIGNED_BYTE (+ RGBA pack types) must be
      // accepted for ANY complete framebuffer — RGB attachments expand to
      // RGBA on read ((R,G,B,1)). RGB/UNSIGNED_BYTE and RGB/565 remain valid
      // for RGB attachments.
      return (format === C1.RGB && (type === C1.UNSIGNED_BYTE || type === C1.UNSIGNED_SHORT_5_6_5)) ||
        (format === C1.RGBA &&
          (type === C1.UNSIGNED_BYTE || type === C1.UNSIGNED_SHORT_4_4_4_4 || type === C1.UNSIGNED_SHORT_5_5_5_1));
    case C2.RGB8:
      return (format === C1.RGB && type === C1.UNSIGNED_BYTE) || universalRGBA;
    case C1.LUMINANCE: case C1.LUMINANCE_ALPHA: case C1.ALPHA:
      if (floatStorage) return floatStorageReadOK(ctx, format, type);
      // WebGL 1.0 §5.14.12: RGBA/UNSIGNED_BYTE (+ RGBA pack types) must be
      // accepted for ANY complete framebuffer — LUMINANCE/LA/ALPHA attachments
      // expand to RGBA on read (L→(L,L,L,1), A→(0,0,0,A)).
      return format === C1.RGBA &&
        (type === C1.UNSIGNED_BYTE || type === C1.UNSIGNED_SHORT_4_4_4_4 || type === C1.UNSIGNED_SHORT_5_5_5_1);
    case C2.R8: return (format === C2.RED && type === C1.UNSIGNED_BYTE) || universalRGBA;
    case C2.RG8: return (format === C2.RG && type === C1.UNSIGNED_BYTE) || universalRGBA;
    case C2.RGB10_A2:
      return (format === C1.RGBA && type === C2.UNSIGNED_INT_2_10_10_10_REV) || universalRGBA;
    case C2.RGB10_A2UI: return universalUInt;
    // Floating point (renderable only with the color-buffer-float extensions).
    // FLOAT is accepted with format ∈ {RED, RG, RGB, RGBA} for the 16F formats
    // when EITHER float color-buffer extension is enabled (GLES3 ReadPixels
    // table: R16F/RG16F/RGBA16F accept RG/RGB/RGBA expansion with FLOAT; the
    // 32F formats accept the same expansion with FLOAT only — the CTS
    // ext-color-buffer-float.html reads RGBA/FLOAT from R32F/RG32F).
    case C2.R16F: case C2.RG16F: case C2.RGBA16F:
      if (type === C1.FLOAT &&
          (extSupported(ctx, 'EXT_color_buffer_float') || extSupported(ctx, 'EXT_color_buffer_half_float'))) {
        return format === C2.RED || format === C2.RG || format === C1.RGB || format === C1.RGBA;
      }
      return format === (internalFormat === C2.R16F ? C2.RED : internalFormat === C2.RG16F ? C2.RG : C1.RGBA) &&
        floatReadOK(ctx, type, true);
    case C2.R32F: case C2.RG32F: case C2.RGBA32F:
      return (format === C2.RED || format === C2.RG || format === C1.RGB || format === C1.RGBA) &&
        floatReadOK(ctx, type, false);
    case C2.R11F_G11F_B10F:
      // GLES3 adds RGB/FLOAT; the CTS (format-r11f-g11f-b10f, ext-color-buffer-
      // float) additionally reads RGBA/FLOAT from R11F_G11F_B10F attachments.
      return (format === C1.RGB || format === C1.RGBA) && floatReadOK(ctx, type, true);
    // Signed / unsigned integer (native width-matched pair + universal
    // RGBA_INTEGER/INT | RGBA_INTEGER/UNSIGNED_INT expansion).
    case C2.R8I: return (format === C2.RED_INTEGER && type === C1.BYTE) || universalSInt;
    case C2.R8UI: return (format === C2.RED_INTEGER && type === C1.UNSIGNED_BYTE) || universalUInt;
    case C2.R16I: return (format === C2.RED_INTEGER && type === C1.SHORT) || universalSInt;
    case C2.R16UI: return (format === C2.RED_INTEGER && type === C1.UNSIGNED_SHORT) || universalUInt;
    case C2.R32I: return (format === C2.RED_INTEGER && type === C1.INT) || universalSInt;
    case C2.R32UI: return (format === C2.RED_INTEGER && type === C1.UNSIGNED_INT) || universalUInt;
    case C2.RG8I: return (format === C2.RG_INTEGER && type === C1.BYTE) || universalSInt;
    case C2.RG8UI: return (format === C2.RG_INTEGER && type === C1.UNSIGNED_BYTE) || universalUInt;
    case C2.RG16I: return (format === C2.RG_INTEGER && type === C1.SHORT) || universalSInt;
    case C2.RG16UI: return (format === C2.RG_INTEGER && type === C1.UNSIGNED_SHORT) || universalUInt;
    case C2.RG32I: return (format === C2.RG_INTEGER && type === C1.INT) || universalSInt;
    case C2.RG32UI: return (format === C2.RG_INTEGER && type === C1.UNSIGNED_INT) || universalUInt;
    case C2.RGB8I: return (format === C2.RGB_INTEGER && type === C1.BYTE) || universalSInt;
    case C2.RGB8UI: return (format === C2.RGB_INTEGER && type === C1.UNSIGNED_BYTE) || universalUInt;
    case C2.RGB16I: return (format === C2.RGB_INTEGER && type === C1.SHORT) || universalSInt;
    case C2.RGB16UI: return (format === C2.RGB_INTEGER && type === C1.UNSIGNED_SHORT) || universalUInt;
    case C2.RGB32I: return (format === C2.RGB_INTEGER && type === C1.INT) || universalSInt;
    case C2.RGB32UI: return (format === C2.RGB_INTEGER && type === C1.UNSIGNED_INT) || universalUInt;
    case C2.RGBA8I: return (format === C2.RGBA_INTEGER && type === C1.BYTE) || universalSInt;
    case C2.RGBA8UI: return (format === C2.RGBA_INTEGER && type === C1.UNSIGNED_BYTE) || universalUInt;
    case C2.RGBA16I: return (format === C2.RGBA_INTEGER && type === C1.SHORT) || universalSInt;
    case C2.RGBA16UI: return (format === C2.RGBA_INTEGER && type === C1.UNSIGNED_SHORT) || universalUInt;
    case C2.RGBA32I: return universalSInt;
    case C2.RGBA32UI: return universalUInt;
    // Depth / depth-stencil
    case C1.DEPTH_COMPONENT16:
      return format === C1.DEPTH_COMPONENT && (type === C1.UNSIGNED_SHORT || type === C1.UNSIGNED_INT);
    case C2.DEPTH_COMPONENT24:
      return format === C1.DEPTH_COMPONENT && type === C1.UNSIGNED_INT;
    case C2.DEPTH_COMPONENT32F:
      return format === C1.DEPTH_COMPONENT && type === C1.FLOAT;
    case C2.DEPTH24_STENCIL8:
      return format === C2.DEPTH_STENCIL && type === C2.UNSIGNED_INT_24_8;
    case C2.DEPTH32F_STENCIL8:
      return format === C2.DEPTH_STENCIL && type === C2.FLOAT_32_UNSIGNED_INT_24_8_REV;
    default:
      return false;
  }
}

/** Bytes per packed pixel for readPixels size computations (format, type). */
function packBytesPerPixel(format: GLenum, type: GLenum): number {
  const comps =
    format === C1.RGBA || format === C2.RGBA_INTEGER ? 4 :
    format === C1.RGB || format === C2.RGB_INTEGER ? 3 :
    format === C1.LUMINANCE_ALPHA || format === C2.RG || format === C2.RG_INTEGER || format === C2.DEPTH_STENCIL ? 2 :
    1;
  switch (type) {
    case C1.UNSIGNED_BYTE: case C1.BYTE: return comps;
    case C1.UNSIGNED_SHORT_5_6_5: case C1.UNSIGNED_SHORT_4_4_4_4: case C1.UNSIGNED_SHORT_5_5_5_1:
    case C1.UNSIGNED_SHORT: case C1.SHORT: case C2.HALF_FLOAT: case 0x8d61 /* HALF_FLOAT_OES */:
      return comps * 2;
    case C2.UNSIGNED_INT_2_10_10_10_REV: return 4;
    case C2.FLOAT_32_UNSIGNED_INT_24_8_REV: return 8;
    default: return comps * 4; // UNSIGNED_INT, INT, FLOAT, UNSIGNED_INT_24_8
  }
}

function alignUp(n: number, align: number): number {
  return Math.ceil(n / align) * align;
}

/**
 * Format/type compatibility with the READ buffer (INVALID_OPERATION when
 * false). Default framebuffer: only RGBA/UNSIGNED_BYTE (WebGL1+2). FBO: the
 * attachment's internal format decides via readComboOK (missing color
 * attachment → false).
 */
function readPixelsComboOK(ctx: WebGLRenderingContext, format: GLenum, type: GLenum): boolean {
  const s = ctx._state;
  const fbo = s.readFramebuffer;
  if (fbo === null) {
    return format === C1.RGBA && type === C1.UNSIGNED_BYTE;
  }
  const rb = s.version === 2 ? s.readBuffer : C1.COLOR_ATTACHMENT0;
  const att = fbo._attachments.get(rb);
  if (!att) return false; // missing color attachment
  const internalFormat = att.type === 'renderbuffer'
    ? att.renderbuffer._internalformat
    : (att.texture._image?.internalFormat ?? 0);
  // W1 unsized float-storage levels (RGBA/RGB/... + FLOAT/HALF_FLOAT_OES)
  // report their storage class via the attached surface info.
  const floatStorage = att.type === 'renderbuffer'
    ? !!(att.renderbuffer._surface?.info?.isFloat)
    : !!(att.texture._image?.info?.isFloat);
  return readComboOK(ctx, internalFormat, format, type, floatStorage);
}

/**
 * Pixel-store constraints + destination size (INVALID_OPERATION). Returns the
 * number of packed bytes needed, or -1 after pushing the error.
 */
function readPixelsNeeded(ctx: WebGLRenderingContext, format: GLenum, type: GLenum, width: GLsizei, height: GLsizei): number {
  const s = ctx._state;
  const bpp = packBytesPerPixel(format, type);
  const pack = s.pixelStore.pack;
  const rowLen = pack.rowLength || width;
  if (s.version === 2 && pack.skipPixels + width > rowLen) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return -1;
  }
  const rowStride = alignUp(rowLen * bpp, pack.alignment);
  return pack.skipRows * rowStride + pack.skipPixels * bpp +
    (height > 0 ? rowStride * (height - 1) + width * bpp : 0);
}

// ---------------------------------------------------------------------------
// installDrawApi
// ---------------------------------------------------------------------------

export function installDrawApi(proto: WebGLRenderingContext): void {
  // ---- WebGL1 draw calls (inherited by WebGL2 via the prototype chain) ----

  proto.drawArrays = function (this: WebGLRenderingContext, mode: GLenum, first: GLint, count: GLsizei): void {
    const ctx = this;
    if (isLost(ctx)) return;
    // WebIDL: GLint/GLsizei convert via ToInt32 (0xffffffff → -1 → INVALID_VALUE).
    const req = validateDrawArrays(ctx, mode, first | 0, count | 0, 1);
    if (!req) return;
    try { executeDraw(ctx, req); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
  };

  proto.drawElements = function (this: WebGLRenderingContext, mode: GLenum, count: GLsizei, type: GLenum, offset: GLintptr): void {
    const ctx = this;
    if (isLost(ctx)) return;
    // count is GLsizei (ToInt32); offset is GLintptr (64-bit, left untouched).
    const req = validateDrawElements(ctx, mode, count | 0, type, offset);
    if (!req) return;
    try { executeDraw(ctx, req); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
  };

  proto.clear = function (this: WebGLRenderingContext, mask: GLbitfield): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if ((mask & ~(C1.COLOR_BUFFER_BIT | C1.DEPTH_BUFFER_BIT | C1.STENCIL_BUFFER_BIT)) !== 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    try { executeClear(ctx, mask); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
  };

  proto.flush = function (this: WebGLRenderingContext): void {
    if (isLost(this)) return;
    // synchronous renderer: nothing to flush
  };

  proto.finish = function (this: WebGLRenderingContext): void {
    if (isLost(this)) return;
    // synchronous renderer: nothing to wait for
  };

  proto.readPixels = function (
    this: WebGLRenderingContext,
    x: GLint, y: GLint, width: GLsizei, height: GLsizei,
    format: GLenum, type: GLenum, pixels: ArrayBufferView | number | null,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (pixels === null || pixels === undefined) {
      ctx._errors.push(C1.INVALID_VALUE); // nullable per WebIDL; null → INVALID_VALUE
      return;
    }
    const s = ctx._state;
    if (width < 0 || height < 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    // WebGL2: the 7th arg is either an ArrayBufferView (client-memory read) or
    // a GLintptr byte offset into the PIXEL_PACK_BUFFER data store.
    if (typeof pixels === 'number') {
      // ---- PIXEL_PACK_BUFFER offset overload ----
      if (s.version !== 2 || !s.pixelPackBuffer) {
        // No PIXEL_PACK_BUFFER bound (or WebGL1 — no such target): the number
        // overload is illegal → INVALID_OPERATION (CTS asserts this, NOT a
        // TypeError).
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (pixels < 0) {
        ctx._errors.push(C1.INVALID_VALUE); // offset < 0 (spec)
        return;
      }
      const packBuf = s.pixelPackBuffer;
      if (!packBuf._data) {
        ctx._errors.push(C1.INVALID_OPERATION); // no data store
        return;
      }
      // 1. Enum validity: WebGL2 uses the GLES3 (format,type) PAIR table.
      if (!RP_PAIRS_2.has(rpPairKey(format, type))) {
        ctx._errors.push(C1.INVALID_ENUM);
        return;
      }
      // 2. Format/type compatibility with the read buffer (INVALID_OPERATION).
      if (!readPixelsComboOK(ctx, format, type)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // 3. Pixel-store constraints + remainder of the data store
      //    (INVALID_OPERATION).
      const needed = readPixelsNeeded(ctx, format, type, width, height);
      if (needed < 0) return;
      if (pixels + needed > packBuf._data.byteLength) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      // 4. Execute — the engine writes into the PBO at `offset` + pack skips.
      try {
        executeReadPixels(ctx, x, y, width, height, format, type, pixels as unknown as ArrayBufferView);
      } catch { ctx._errors.push(C1.INVALID_OPERATION); }
      return;
    }
    if (!ArrayBuffer.isView(pixels)) {
      throw new TypeError(`Argument is not of type 'ArrayBufferView'`);
    }
    // ---- ArrayBufferView overload ----
    // 1. Enum validity (INVALID_ENUM for values that are never readPixels enums).
    if (s.version === 2) {
      if (!RP_PAIRS_2.has(rpPairKey(format, type))) { ctx._errors.push(C1.INVALID_ENUM); return; }
    } else {
      if (!RP_FORMATS_1.has(format)) { ctx._errors.push(C1.INVALID_ENUM); return; }
      if (!RP_TYPES_1.has(type)) { ctx._errors.push(C1.INVALID_ENUM); return; }
    }
    // 2. WebGL2: the ArrayBufferView overload is invalid while a
    //    PIXEL_PACK_BUFFER is bound (spec; any bound buffer, with or without a
    //    data store).
    if (s.pixelPackBuffer) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // 3. Format/type compatibility with the read buffer (INVALID_OPERATION).
    if (!readPixelsComboOK(ctx, format, type)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // 4. The view's class must match `type` (INVALID_OPERATION, per spec).
    const want = expectedViewForType(type);
    if (want === null) { ctx._errors.push(C1.INVALID_ENUM); return; }
    const viewOK = want === 'u8'
      ? (pixels instanceof Uint8Array || pixels instanceof Uint8ClampedArray)
      : pixels instanceof want;
    if (!viewOK) { ctx._errors.push(C1.INVALID_OPERATION); return; }
    // 5. Pixel-store constraints + destination size (INVALID_OPERATION).
    const needed = readPixelsNeeded(ctx, format, type, width, height);
    if (needed < 0) return;
    if (pixels.byteLength - pixels.byteOffset < needed) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // 6. Execute.
    try {
      executeReadPixels(ctx, x, y, width, height, format, type, pixels);
    } catch { ctx._errors.push(C1.INVALID_OPERATION); }
  };

  /**
   * @internal — OffscreenCanvas.transferToImageBitmap() hook. Installed here
   * (not as a class-body method) because installAll runs on BOTH the WebGL1 and
   * WebGL2 prototypes, and webgl2.ts re-chains the WebGL2 prototype under the
   * NATIVE WebGL2 prototype — class-body methods declared on WebGL1 are LOST
   * for WebGL2 contexts, while api-installed own props survive.
   */
  proto._transferToImageBitmap = function (): { width: number; height: number; data: Uint8ClampedArray } | null {
    return transferToImageBitmapSnapshot(this);
  };

  // ---- WebGL2-only methods (no-op installer on the WebGL1 prototype) ----
  const p2 = proto as unknown as WebGL2RenderingContext;

  if ('drawArraysInstanced' in p2) {
    p2.drawArraysInstanced = function (this: WebGL2RenderingContext, mode: GLenum, first: GLint, count: GLsizei, instanceCount: GLsizei): void {
      const ctx = this;
      if (isLost(ctx)) return;
      // WebIDL: GLint/GLsizei convert via ToInt32.
      const req = validateDrawArrays(ctx, mode, first | 0, count | 0, instanceCount | 0);
      if (!req) return;
      try { executeDraw(ctx, req); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    };
  }

  if ('drawElementsInstanced' in p2) {
    p2.drawElementsInstanced = function (this: WebGL2RenderingContext, mode: GLenum, count: GLsizei, type: GLenum, offset: GLintptr, instanceCount: GLsizei): void {
      const ctx = this;
      if (isLost(ctx)) return;
      // count/instanceCount are GLsizei (ToInt32); offset is GLintptr (untouched).
      const req = validateDrawElements(ctx, mode, count | 0, type, offset, { instanceCount: instanceCount | 0 });
      if (!req) return;
      try { executeDraw(ctx, req); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    };
  }

  if ('drawRangeElements' in p2) {
    p2.drawRangeElements = function (this: WebGL2RenderingContext, mode: GLenum, start: GLuint, end: GLuint, count: GLsizei, type: GLenum, offset: GLintptr): void {
      const ctx = this;
      if (isLost(ctx)) return;
      // start/end are GLuint (ToUint32); count is GLsizei (ToInt32); offset untouched.
      const req = validateDrawElements(ctx, mode, count | 0, type, offset, { range: [start >>> 0, end >>> 0] });
      if (!req) return;
      try { executeDraw(ctx, req); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    };
  }

  if ('clearBufferfv' in p2) {
    p2.clearBufferfv = function (this: WebGL2RenderingContext, buffer: GLenum, drawbuffer: GLint, values: Float32List): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const v = toList(values, Float32Array, 'Float32List');
      if (!validateClearBufferArgs(ctx, buffer, drawbuffer)) return;
      if (!CLEAR_FV.has(buffer)) { ctx._errors.push(C1.INVALID_OPERATION); return; }
      const need = buffer === C2.COLOR ? 4 : 1;
      if (v.length < need) { ctx._errors.push(C1.INVALID_VALUE); return; }
      if (buffer === C2.COLOR) {
        // clearBufferfv clears float/fixed-point color attachments only.
        const cls = clearColorAttachmentClass(ctx, drawbuffer);
        if (cls === 'missing') return; // missing attachment → no-op NO_ERROR
        if (cls !== 'float') { ctx._errors.push(C1.INVALID_OPERATION); return; }
      }
      try { executeClearBuffer(ctx, buffer, drawbuffer, v); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    };
  }

  if ('clearBufferiv' in p2) {
    p2.clearBufferiv = function (this: WebGL2RenderingContext, buffer: GLenum, drawbuffer: GLint, values: Int32List): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const v = toList(values, Int32Array, 'Int32List');
      if (!validateClearBufferArgs(ctx, buffer, drawbuffer)) return;
      if (!CLEAR_IV.has(buffer)) { ctx._errors.push(C1.INVALID_OPERATION); return; }
      const need = buffer === C2.COLOR ? 4 : 1;
      if (v.length < need) { ctx._errors.push(C1.INVALID_VALUE); return; }
      if (buffer === C2.COLOR) {
        // clearBufferiv clears signed-integer color attachments only.
        const cls = clearColorAttachmentClass(ctx, drawbuffer);
        if (cls === 'missing') return; // missing attachment → no-op NO_ERROR
        if (cls !== 'sint') { ctx._errors.push(C1.INVALID_OPERATION); return; }
      }
      try { executeClearBuffer(ctx, buffer, drawbuffer, v); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
      try { executeClearBuffer(ctx, buffer, drawbuffer, v); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    };
  }

  if ('clearBufferuiv' in p2) {
    p2.clearBufferuiv = function (this: WebGL2RenderingContext, buffer: GLenum, drawbuffer: GLint, values: Uint32List): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const v = toList(values, Uint32Array, 'Uint32List');
      if (!validateClearBufferArgs(ctx, buffer, drawbuffer)) return;
      if (!CLEAR_UIV.has(buffer)) { ctx._errors.push(C1.INVALID_OPERATION); return; }
      if (v.length < 4) { ctx._errors.push(C1.INVALID_VALUE); return; }
      if (buffer === C2.COLOR) {
        // clearBufferuiv clears unsigned-integer color attachments only.
        const cls = clearColorAttachmentClass(ctx, drawbuffer);
        if (cls === 'missing') return; // missing attachment → no-op NO_ERROR
        if (cls !== 'uint') { ctx._errors.push(C1.INVALID_OPERATION); return; }
      }
      try { executeClearBuffer(ctx, buffer, drawbuffer, v); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    };
  }

  if ('clearBufferfi' in p2) {
    p2.clearBufferfi = function (this: WebGL2RenderingContext, buffer: GLenum, drawbuffer: GLint, depth: GLfloat, stencil: GLint): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (!CLEAR_BUFFERS.has(buffer)) { ctx._errors.push(C1.INVALID_ENUM); return; }
      if (buffer !== C2.DEPTH_STENCIL) { ctx._errors.push(C1.INVALID_OPERATION); return; }
      if (drawbuffer !== 0) { ctx._errors.push(C1.INVALID_VALUE); return; }
      try { executeClearBuffer(ctx, buffer, drawbuffer, null, depth, stencil); } catch { ctx._errors.push(C1.INVALID_OPERATION); }
    };
  }

  // ---- WEBGL_multi_draw (extension methods; installed UNCONDITIONALLY — the   ----
  // extension registry exposes them on both WebGL1 and WebGL2, and the WebGL2
  // prototype receives them via installAll + the descriptor copy in webgl2.ts).
  //
  // Prototype layer validates ONLY the multi-draw-specific preconditions
  // (WebIDL list conversion, drawcount, list lengths) and delegates to the
  // engine entries, which validate every subdraw first (validate-all-first
  // contract: any invalid subdraw → error + NOTHING drawn) and then execute.
  // The *Offset args are ELEMENT offsets into the lists (NOT bytes); drawcount
  // < 0 → INVALID_VALUE; offset + drawcount > list.length → INVALID_OPERATION.
  //
  // Extension methods are not part of the core class declaration — widen the
  // prototype for installation (same pattern as the `p2` WebGL2 cast above).
  // They are installed NON-ENUMERABLE: per WebGL, extension methods are exposed
  // ONLY via getExtension(), and CTS offscreencanvas/methods.html enumerates
  // `for (var i in gl)` and fails on any enumerable function property outside
  // the spec method list. Non-enumerable own properties are still found by
  // ordinary property lookup, so the extension object's delegation
  // (extensions/misc.ts callInstalled) keeps working unchanged.
  const mdProto = proto as unknown as {
    multiDrawArraysWEBGL(mode: GLenum, firsts: Int32List, firstsOffset: GLuint, counts: Int32List, countsOffset: GLuint, drawcount: GLsizei): void;
    multiDrawElementsWEBGL(mode: GLenum, counts: Int32List, countsOffset: GLuint, type: GLenum, offsets: Int32List, offsetsOffset: GLuint, drawcount: GLsizei): void;
    multiDrawArraysInstancedWEBGL(mode: GLenum, firsts: Int32List, firstsOffset: GLuint, counts: Int32List, countsOffset: GLuint, instanceCounts: Int32List, instanceCountsOffset: GLuint, drawcount: GLsizei): void;
    multiDrawElementsInstancedWEBGL(mode: GLenum, counts: Int32List, countsOffset: GLuint, type: GLenum, offsets: Int32List, offsetsOffset: GLuint, instanceCounts: Int32List, instanceCountsOffset: GLuint, drawcount: GLsizei): void;
  };
  function installExtensionMethod(proto: object, name: string, fn: (...args: never[]) => void): void {
    Object.defineProperty(proto, name, { value: fn, writable: true, configurable: true, enumerable: false });
  }

  installExtensionMethod(mdProto, 'multiDrawArraysWEBGL', function (
    this: WebGLRenderingContext,
    mode: GLenum, firsts: Int32List, firstsOffset: GLuint,
    counts: Int32List, countsOffset: GLuint, drawcount: GLsizei,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const firstsArr = toList(firsts, Int32Array, 'Int32List');
    const countsArr = toList(counts, Int32Array, 'Int32List');
    const dc = drawcount | 0;
    if (dc < 0) { ctx._errors.push(C1.INVALID_VALUE); return; }
    if (dc === 0) return; // NO_ERROR no-op
    const fo = firstsOffset >>> 0;
    const co = countsOffset >>> 0;
    if (fo + dc > firstsArr.length || co + dc > countsArr.length) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    try { executeMultiDrawArrays(ctx, mode, firstsArr, fo, countsArr, co, dc); }
    catch { ctx._errors.push(C1.INVALID_OPERATION); }
  });

  installExtensionMethod(mdProto, 'multiDrawElementsWEBGL', function (
    this: WebGLRenderingContext,
    mode: GLenum, counts: Int32List, countsOffset: GLuint,
    type: GLenum, offsets: Int32List, offsetsOffset: GLuint, drawcount: GLsizei,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const countsArr = toList(counts, Int32Array, 'Int32List');
    const offsetsArr = toList(offsets, Int32Array, 'Int32List');
    const dc = drawcount | 0;
    if (dc < 0) { ctx._errors.push(C1.INVALID_VALUE); return; }
    if (dc === 0) return; // NO_ERROR no-op
    const co = countsOffset >>> 0;
    const oo = offsetsOffset >>> 0;
    if (co + dc > countsArr.length || oo + dc > offsetsArr.length) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    try { executeMultiDrawElements(ctx, mode, countsArr, co, type, offsetsArr, oo, dc); }
    catch { ctx._errors.push(C1.INVALID_OPERATION); }
  });

  installExtensionMethod(mdProto, 'multiDrawArraysInstancedWEBGL', function (
    this: WebGLRenderingContext,
    mode: GLenum, firsts: Int32List, firstsOffset: GLuint,
    counts: Int32List, countsOffset: GLuint,
    instanceCounts: Int32List, instanceCountsOffset: GLuint, drawcount: GLsizei,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const firstsArr = toList(firsts, Int32Array, 'Int32List');
    const countsArr = toList(counts, Int32Array, 'Int32List');
    const instArr = toList(instanceCounts, Int32Array, 'Int32List');
    const dc = drawcount | 0;
    if (dc < 0) { ctx._errors.push(C1.INVALID_VALUE); return; }
    if (dc === 0) return; // NO_ERROR no-op
    const fo = firstsOffset >>> 0;
    const co = countsOffset >>> 0;
    const io = instanceCountsOffset >>> 0;
    if (fo + dc > firstsArr.length || co + dc > countsArr.length || io + dc > instArr.length) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    try { executeMultiDrawArraysInstanced(ctx, mode, firstsArr, fo, countsArr, co, instArr, io, dc); }
    catch { ctx._errors.push(C1.INVALID_OPERATION); }
  });

  installExtensionMethod(mdProto, 'multiDrawElementsInstancedWEBGL', function (
    this: WebGLRenderingContext,
    mode: GLenum, counts: Int32List, countsOffset: GLuint,
    type: GLenum, offsets: Int32List, offsetsOffset: GLuint,
    instanceCounts: Int32List, instanceCountsOffset: GLuint, drawcount: GLsizei,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const countsArr = toList(counts, Int32Array, 'Int32List');
    const offsetsArr = toList(offsets, Int32Array, 'Int32List');
    const instArr = toList(instanceCounts, Int32Array, 'Int32List');
    const dc = drawcount | 0;
    if (dc < 0) { ctx._errors.push(C1.INVALID_VALUE); return; }
    if (dc === 0) return; // NO_ERROR no-op
    const co = countsOffset >>> 0;
    const oo = offsetsOffset >>> 0;
    const io = instanceCountsOffset >>> 0;
    if (co + dc > countsArr.length || oo + dc > offsetsArr.length || io + dc > instArr.length) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    try { executeMultiDrawElementsInstanced(ctx, mode, countsArr, co, type, offsetsArr, oo, instArr, io, dc); }
    catch { ctx._errors.push(C1.INVALID_OPERATION); }
  });
}
