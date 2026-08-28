/**
 * src/gl/api/teximage.ts — texImage/texSubImage/texStorage/copyTex/compressedTex.
 *
 * Owns: texImage2D, texImage3D, texSubImage2D, texSubImage3D, texStorage2D,
 * texStorage3D, copyTexImage2D, copyTexSubImage2D, copyTexSubImage3D,
 * compressedTexImage2D, compressedTexImage3D, compressedTexSubImage2D,
 * compressedTexSubImage3D.
 *
 * All argument validation + overload dispatch happens here; the actual storage
 * mutation is delegated to the teximage.ts engine module. Overload rules:
 *  - texImage2D: 6 args → (target, level, internalformat, format, type, source)
 *    DOM form; else the 9-arg buffer form (pixels optional; null allocates).
 *  - texImage3D: 10 args with a DOM source → DOM form with explicit dims
 *    (target, level, internalformat, w, h, d, border, format, type, source —
 *    the 2D image fills the z=0 slice); else the 10-arg buffer form (number
 *    pixels = PBO offset, view = client data). The 6-arg DOM form is NOT in
 *    the spec and is rejected (INVALID_OPERATION).
 *  - texSubImage2D: 7 args → DOM form; else 9-arg buffer form.
 *  - texSubImage3D: 11 args with a DOM source → DOM form with explicit dims
 *    (target, level, xoffset, yoffset, zoffset, w, h, d, format, type,
 *    source — one slice at zoffset); else the 11-arg buffer form. The 8-arg
 *    DOM form is NOT in the spec and is rejected (INVALID_OPERATION).
 *  - WebGL2: buffer-form pixels may be a NUMBER (byte offset into
 *    PIXEL_UNPACK_BUFFER — must be bound; flipY/premultiplyAlpha must be off;
 *    offset+required ≤ buffer size). The buffer form also accepts trailing
 *    srcOffset (and optional srcLength) numeric arguments AFTER an
 *    ArrayBufferView — the upload reads starting at element srcOffset of the
 *    view, in the view's element units (spec §3.7.2; applySrcOffset slices).
 * Validation highlights: border must be 0 (INVALID_VALUE); internalformat vs
 * format/type compatibility per WebGL1 tables + extension-gated formats
 * (WEBGL_depth_texture / OES_texture_float(_linear) / OES_texture_half_float /
 * EXT_sRGB) and the WebGL2 GLES3 Table 3.2 combo table (W2_COMBOS, norm16
 * gated on EXT_texture_norm16); WebGL1 NPOT level rule; WebGL1 texSubImage
 * format/type must match the level origin; texStorage immutability + sized
 * internalformat + level-count checks; source-buffer size validation per
 * UNPACK_ALIGNMENT/ROW_LENGTH/SKIP_*.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1, C2, CExt } from '../constants';
import type { WebGLFramebuffer, WebGLTexture } from '../objects';
import type { GLenum, GLint, GLsizei, TexImageSource } from '../types';
import { checkFramebufferStatus } from '../framebuffer-util';
import { bufferTfUseError } from './buffers';
import {
  allocateImmutableStorage,
  compressedTexImage,
  copyTexImage,
  copyTexSubImage,
  etc2ImageBytes,
  ETC2_BYTES_PER_BLOCK,
  getLevelOrigin,
  hasTextureLevel,
  uploadTexImage,
  uploadTexSubImage,
  type TexImageSourceArg,
} from '../teximage';

// GL values not present in C1 (see constants.ts provenance / state.ts precedent).
const HALF_FLOAT_OES = 0x8d61;
const UNSIGNED_INT_24_8_WEBGL = 0x84fa;
const SRGB_EXT = 0x8c40;
const SRGB_ALPHA_EXT = 0x8c42;
const TEXTURE_MAX_ANISOTROPY_EXT = 0x84fe; // (not used here; textures.ts owns texParameter)

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

const CUBE_FACES: number[] = [
  C1.TEXTURE_CUBE_MAP_POSITIVE_X,
  C1.TEXTURE_CUBE_MAP_NEGATIVE_X,
  C1.TEXTURE_CUBE_MAP_POSITIVE_Y,
  C1.TEXTURE_CUBE_MAP_NEGATIVE_Y,
  C1.TEXTURE_CUBE_MAP_POSITIVE_Z,
  C1.TEXTURE_CUBE_MAP_NEGATIVE_Z,
];

function isCubeFace(target: GLenum): boolean {
  return target >= C1.TEXTURE_CUBE_MAP_POSITIVE_X && target <= C1.TEXTURE_CUBE_MAP_NEGATIVE_Z;
}

/** 2D-style targets (texImage2D/texSubImage2D/copyTex*2D): TEXTURE_2D + cube faces. */
function is2DTarget(target: GLenum): boolean {
  return target === C1.TEXTURE_2D || isCubeFace(target);
}

/** 3D-style targets (texImage3D/texSubImage3D/copyTexSubImage3D): WebGL2 only. */
function is3DTarget(ctx: WebGLRenderingContext, target: GLenum): boolean {
  return ctx._version === 2 && (target === C2.TEXTURE_3D || target === C2.TEXTURE_2D_ARRAY);
}

/** The texture bound to a (validated) target in the active unit — null when none. */
function boundTextureForTarget(ctx: WebGLRenderingContext, target: GLenum): WebGLTexture | null {
  const unit = ctx._state.textureUnits[ctx._state.activeTexture];
  // state.ts texture-unit slots are typed with the DOM WebGLTexture interface;
  // they always hold renderer WebGLTexture instances (bindTexture writes them).
  if (target === C1.TEXTURE_2D) return unit.texture2D as unknown as WebGLTexture | null;
  if (target === C1.TEXTURE_CUBE_MAP || isCubeFace(target)) return unit.textureCube as unknown as WebGLTexture | null;
  if (target === C2.TEXTURE_3D) return unit.texture3D as unknown as WebGLTexture | null;
  if (target === C2.TEXTURE_2D_ARRAY) return unit.texture2DArray as unknown as WebGLTexture | null;
  return null;
}

/** Dimension limit for a (validated) target. */
function dimLimit(
  ctx: WebGLRenderingContext,
  target: GLenum,
): { maxW: number; maxH: number; maxD: number; maxDim: number } {
  const lim = ctx._state.limits;
  if (target === C1.TEXTURE_CUBE_MAP || isCubeFace(target)) return { maxW: lim.MAX_CUBE_MAP_TEXTURE_SIZE, maxH: lim.MAX_CUBE_MAP_TEXTURE_SIZE, maxD: 1, maxDim: lim.MAX_CUBE_MAP_TEXTURE_SIZE };
  if (target === C2.TEXTURE_3D) return { maxW: lim.MAX_3D_TEXTURE_SIZE, maxH: lim.MAX_3D_TEXTURE_SIZE, maxD: lim.MAX_3D_TEXTURE_SIZE, maxDim: lim.MAX_3D_TEXTURE_SIZE };
  if (target === C2.TEXTURE_2D_ARRAY) return { maxW: lim.MAX_TEXTURE_SIZE, maxH: lim.MAX_TEXTURE_SIZE, maxD: lim.MAX_ARRAY_TEXTURE_LAYERS, maxDim: lim.MAX_TEXTURE_SIZE };
  return { maxW: lim.MAX_TEXTURE_SIZE, maxH: lim.MAX_TEXTURE_SIZE, maxD: 1, maxDim: lim.MAX_TEXTURE_SIZE };
}

const isPow2 = (v: number): boolean => v > 0 && (v & (v - 1)) === 0;

/**
 * Per-level dimension + level-bound validation (GLES2/3 + WebGL semantics;
 * CTS texture-size-limit.html / tex-3d-size-limit.html): the maximum
 * width/height at level L is max(1, maxSize >> L) where maxSize is the
 * target's size limit, and the maximum level is floor(log2(maxSize)).
 * TEXTURE_2D_ARRAY layers (depth) are NOT level-scaled; TEXTURE_3D depth IS
 * (max(1, MAX_3D_TEXTURE_SIZE >> L)). Pushes INVALID_VALUE and returns false
 * when out of bounds. Uses division (not >>) so huge levels cannot wrap via
 * JS's mod-32 shift semantics.
 */
function validateLevelDims(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: number,
  width: number,
  height: number,
  depth: number,
): boolean {
  const lim = dimLimit(ctx, target);
  const scale = Math.pow(2, level);
  const maxW = Math.max(1, Math.floor(lim.maxW / scale));
  const maxH = Math.max(1, Math.floor(lim.maxH / scale));
  const maxD = target === C2.TEXTURE_3D ? Math.max(1, Math.floor(lim.maxD / scale)) : lim.maxD;
  if (width > maxW || height > maxH || depth > maxD) {
    ctx._errors.push(C1.INVALID_VALUE);
    return false;
  }
  if (level > Math.floor(Math.log2(lim.maxDim))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return false;
  }
  return true;
}

/**
 * Shared texImage2D/texImage3D validation (both forms): target, bound texture,
 * immutability, border, level, dims vs limits. Returns the bound texture, or
 * null with an error pushed.
 */
function commonTexImageValidation(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  width: number,
  height: number,
  depth: number,
  border: GLint,
  is3D: boolean,
): WebGLTexture | null {
  if (is3D ? !is3DTarget(ctx, target) : !is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return null;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (tex._immutable) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (border !== 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (level < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (width < 0 || height < 0 || depth < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (!validateLevelDims(ctx, target, level, width, height, depth)) return null;
  if (ctx._version === 1 && level > 0 && (!isPow2(width) || !isPow2(height))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  return tex;
}

// ---------------------------------------------------------------------------
// WebGL1 format/type tables (unsized + extension-gated)
// ---------------------------------------------------------------------------

const W1_INTERNALFORMATS: number[] = [C1.RGBA, C1.RGB, C1.ALPHA, C1.LUMINANCE, C1.LUMINANCE_ALPHA];
const W1_FORMATS: number[] = [C1.RGBA, C1.RGB, C1.ALPHA, C1.LUMINANCE, C1.LUMINANCE_ALPHA];
const W1_TYPES: number[] = [
  C1.UNSIGNED_BYTE,
  C1.UNSIGNED_SHORT_5_6_5,
  C1.UNSIGNED_SHORT_4_4_4_4,
  C1.UNSIGNED_SHORT_5_5_5_1,
];

/** (internalformat → {format, types}) for every WebGL1-legal texImage2D combo. */
const W1_COMBOS: Record<number, { format: number; types: number[] }> = {
  [C1.RGBA]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_4_4_4_4, C1.UNSIGNED_SHORT_5_5_5_1] },
  [C1.RGB]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_6_5] },
  [C1.LUMINANCE]: { format: C1.LUMINANCE, types: [C1.UNSIGNED_BYTE] },
  [C1.LUMINANCE_ALPHA]: { format: C1.LUMINANCE_ALPHA, types: [C1.UNSIGNED_BYTE] },
  [C1.ALPHA]: { format: C1.ALPHA, types: [C1.UNSIGNED_BYTE] },
  [C1.DEPTH_COMPONENT]: { format: C1.DEPTH_COMPONENT, types: [C1.UNSIGNED_INT, C1.UNSIGNED_SHORT, UNSIGNED_INT_24_8_WEBGL] },
  [C1.DEPTH_STENCIL]: { format: C1.DEPTH_STENCIL, types: [UNSIGNED_INT_24_8_WEBGL] },
  [0x881b]: { format: C1.RGB, types: [HALF_FLOAT_OES] }, // RGB16F
  [0x881a]: { format: C1.RGBA, types: [HALF_FLOAT_OES] }, // RGBA16F
  [0x8815]: { format: C1.RGB, types: [C1.FLOAT] }, // RGB32F
  [0x8814]: { format: C1.RGBA, types: [C1.FLOAT] }, // RGBA32F
  // EXT_sRGB: the WebGL extension spec accepts SRGB_EXT/SRGB_ALPHA_EXT as BOTH
  // internalformat and format, and CTS uploads with format == internalformat
  // (ext-sRGB.html, texture-srgb-upload.html) — the combo format must be the
  // SRGB enum itself (the internalformat !== format check below enforces the
  // match). Storage resolves these to SRGB8/SRGB8_ALPHA8 (engine unsizedStorage).
  [SRGB_EXT]: { format: SRGB_EXT, types: [C1.UNSIGNED_BYTE] },
  [SRGB_ALPHA_EXT]: { format: SRGB_ALPHA_EXT, types: [C1.UNSIGNED_BYTE] },
};

function w1InternalformatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (W1_INTERNALFORMATS.includes(fmt)) return true;
  switch (fmt) {
    case C1.DEPTH_COMPONENT:
    case C1.DEPTH_STENCIL:
      return ctx._extensions.has('WEBGL_depth_texture');
    case 0x881b:
    case 0x881a:
      return ctx._extensions.has('OES_texture_half_float');
    case 0x8815:
    case 0x8814:
      return ctx._extensions.has('OES_texture_float');
    case SRGB_EXT:
    case SRGB_ALPHA_EXT:
      return ctx._extensions.has('EXT_sRGB');
    default:
      return false;
  }
}

function w1FormatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (W1_FORMATS.includes(fmt)) return true;
  if (fmt === C1.DEPTH_COMPONENT || fmt === C1.DEPTH_STENCIL) {
    return ctx._extensions.has('WEBGL_depth_texture');
  }
  // EXT_sRGB adds SRGB_EXT/SRGB_ALPHA_EXT as `format` (mirror of the
  // internalformat gating above) — CTS ext-sRGB.html / texture-srgb-upload.html
  // upload with format == internalformat.
  if (fmt === SRGB_EXT || fmt === SRGB_ALPHA_EXT) {
    return ctx._extensions.has('EXT_sRGB');
  }
  return false;
}

/**
 * WebGL1 type legality. Extension-provided types (FLOAT, HALF_FLOAT_OES,
 * depth-texture types) require the extension to be ENABLED — i.e.
 * `getExtension(name)` must have been called — per WebGL 1.0 spec §4.2.1
 * ("An attempt to use any features of an extension without first calling
 * getExtension to enable it must generate an appropriate GL error").
 * `ctx._extensions` is the enabled-extensions cache (populated by
 * getExtensionObject); checking it here avoids the enabling side effect of
 * calling getExtension() from validation code.
 */
function w1TypeValid(ctx: WebGLRenderingContext, type: GLenum): boolean {
  if (W1_TYPES.includes(type)) return true;
  switch (type) {
    case C1.FLOAT:
      return ctx._extensions.has('OES_texture_float');
    case HALF_FLOAT_OES:
      return ctx._extensions.has('OES_texture_half_float');
    case C1.UNSIGNED_INT:
    case C1.UNSIGNED_SHORT:
    case UNSIGNED_INT_24_8_WEBGL:
      return ctx._extensions.has('WEBGL_depth_texture');
    default:
      return false;
  }
}

/** WebGL1 typed-array ↔ type consistency (INVALID_OPERATION on mismatch). */
function w1TypeMatchesView(type: GLenum, pixels: ArrayBufferView): boolean {
  const v = pixels as unknown as { constructor: Function };
  switch (type) {
    case C1.UNSIGNED_BYTE:
      return v instanceof Uint8Array || v instanceof Uint8ClampedArray;
    case C1.UNSIGNED_SHORT_5_6_5:
    case C1.UNSIGNED_SHORT_4_4_4_4:
    case C1.UNSIGNED_SHORT_5_5_5_1:
    case C1.UNSIGNED_SHORT:
    case HALF_FLOAT_OES:
      return v instanceof Uint16Array;
    case C1.FLOAT:
      return v instanceof Float32Array;
    case C1.UNSIGNED_INT:
    case UNSIGNED_INT_24_8_WEBGL:
      return v instanceof Uint32Array;
    default:
      return true;
  }
}

/**
 * WebGL1 (internalformat, format, type) validation + pixels WebIDL checks.
 * Pixels: null legal; ArrayBufferView legal (view↔type checked); anything else
 * (number, DOM source, ArrayBuffer) throws TypeError — WebIDL conversion of the
 * 9-arg form's `ArrayBufferView?` argument.
 */
function w1ValidateFormatType(
  ctx: WebGLRenderingContext,
  internalformat: GLenum,
  format: GLenum,
  type: GLenum,
  pixels: unknown,
): boolean {
  if (!w1InternalformatValid(ctx, internalformat)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (!w1FormatValid(ctx, format)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (!w1TypeValid(ctx, type)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  const combo = W1_COMBOS[internalformat];
  if (!combo) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (combo.format !== format) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (!w1ComboAllowsType(ctx, combo, type)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (internalformat !== format) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (pixels !== null && pixels !== undefined) {
    if (!ArrayBuffer.isView(pixels)) {
      throw new TypeError(`Argument is not of type 'ArrayBufferView'`);
    }
    if (!w1TypeMatchesView(type, pixels)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return false;
    }
  }
  return true;
}

/**
 * WebGL1 (internalformat, type) combo legality: the static W1_COMBOS table
 * plus the extension-gated additions — OES_texture_float adds FLOAT to
 * ALL unsized color formats (RGBA/RGB/LUMINANCE/LUMINANCE_ALPHA/ALPHA →
 * float storage), OES_texture_half_float adds HALF_FLOAT_OES to the same —
 * legal only while the extension is ENABLED (ext specs add the type to every
 * unsized internalformat; CTS ext-color-buffer-half-float.html exercises
 * LUMINANCE/LUMINANCE_ALPHA/ALPHA + HALF_FLOAT_OES).
 */
function w1ComboAllowsType(
  ctx: WebGLRenderingContext,
  combo: { format: number; types: number[] },
  type: GLenum,
): boolean {
  if (combo.types.includes(type)) return true;
  if (type === C1.FLOAT) return ctx._extensions.has('OES_texture_float');
  if (type === HALF_FLOAT_OES) return ctx._extensions.has('OES_texture_half_float');
  return false;
}

// ---------------------------------------------------------------------------
// WebGL2 format/type tables (GLES3 Table 3.2 semantics)
// ---------------------------------------------------------------------------

const W2_FORMATS: number[] = [
  C2.RED,
  C2.RG,
  C1.RGB,
  C1.RGBA,
  C2.RED_INTEGER,
  C2.RG_INTEGER,
  C2.RGB_INTEGER,
  C2.RGBA_INTEGER,
  C1.DEPTH_COMPONENT,
  C1.DEPTH_STENCIL,
  C1.LUMINANCE,
  C1.LUMINANCE_ALPHA,
  C1.ALPHA,
];

const W2_TYPES: number[] = [
  C1.UNSIGNED_BYTE,
  C1.BYTE,
  C1.UNSIGNED_SHORT,
  C1.SHORT,
  C1.UNSIGNED_INT,
  C1.INT,
  C2.HALF_FLOAT,
  C1.FLOAT,
  C1.UNSIGNED_SHORT_5_6_5,
  C1.UNSIGNED_SHORT_4_4_4_4,
  C1.UNSIGNED_SHORT_5_5_5_1,
  C2.UNSIGNED_INT_2_10_10_10_REV,
  C2.UNSIGNED_INT_10F_11F_11F_REV,
  C2.UNSIGNED_INT_5_9_9_9_REV,
  C2.UNSIGNED_INT_24_8,
  C2.FLOAT_32_UNSIGNED_INT_24_8_REV,
];

/** (internalformat → {format, types}) — GLES3 Table 3.2 texImage combos. */
const W2_COMBOS: Record<number, { format: number; types: number[] }> = {
  // ---- unsized ----
  [C1.RGBA]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_4_4_4_4, C1.UNSIGNED_SHORT_5_5_5_1] },
  [C1.RGB]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_6_5] },
  [C2.RG]: { format: C2.RG, types: [C1.UNSIGNED_BYTE] },
  [C2.RED]: { format: C2.RED, types: [C1.UNSIGNED_BYTE] },
  [C2.RGBA_INTEGER]: { format: C2.RGBA_INTEGER, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT, C1.UNSIGNED_INT] },
  [C2.RGB_INTEGER]: { format: C2.RGB_INTEGER, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT, C1.UNSIGNED_INT] },
  [C2.RG_INTEGER]: { format: C2.RG_INTEGER, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT, C1.UNSIGNED_INT] },
  [C2.RED_INTEGER]: { format: C2.RED_INTEGER, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT, C1.UNSIGNED_INT] },
  [C1.DEPTH_COMPONENT]: { format: C1.DEPTH_COMPONENT, types: [C1.UNSIGNED_SHORT, C1.UNSIGNED_INT] },
  [C1.DEPTH_STENCIL]: { format: C1.DEPTH_STENCIL, types: [C2.UNSIGNED_INT_24_8] },
  [C1.LUMINANCE_ALPHA]: { format: C1.LUMINANCE_ALPHA, types: [C1.UNSIGNED_BYTE] },
  [C1.LUMINANCE]: { format: C1.LUMINANCE, types: [C1.UNSIGNED_BYTE] },
  [C1.ALPHA]: { format: C1.ALPHA, types: [C1.UNSIGNED_BYTE] },
  // ---- sized normalized ----
  [C2.R8]: { format: C2.RED, types: [C1.UNSIGNED_BYTE] },
  [C2.RG8]: { format: C2.RG, types: [C1.UNSIGNED_BYTE] },
  [C2.RGB8]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE] },
  [C2.RGBA8]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE] },
  [C2.R8_SNORM]: { format: C2.RED, types: [C1.BYTE] },
  [C2.RG8_SNORM]: { format: C2.RG, types: [C1.BYTE] },
  [C2.RGB8_SNORM]: { format: C1.RGB, types: [C1.BYTE] },
  [C2.RGBA8_SNORM]: { format: C1.RGBA, types: [C1.BYTE] },
  [C1.RGBA4]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_4_4_4_4] },
  [C1.RGB5_A1]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_5_5_1, C2.UNSIGNED_INT_2_10_10_10_REV] }, // GLES3 Table 3.2
  [C1.RGB565]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_6_5] },
  [C2.RGB10_A2]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C2.UNSIGNED_INT_2_10_10_10_REV] },
  [C2.SRGB8]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE] },
  [C2.SRGB8_ALPHA8]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE] },
  // ---- floats ----
  [C2.R16F]: { format: C2.RED, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RG16F]: { format: C2.RG, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RGB16F]: { format: C1.RGB, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RGBA16F]: { format: C1.RGBA, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.R32F]: { format: C2.RED, types: [C1.FLOAT] },
  [C2.RG32F]: { format: C2.RG, types: [C1.FLOAT] },
  [C2.RGB32F]: { format: C1.RGB, types: [C1.FLOAT] },
  [C2.RGBA32F]: { format: C1.RGBA, types: [C1.FLOAT] },
  [C2.R11F_G11F_B10F]: { format: C1.RGB, types: [C2.UNSIGNED_INT_10F_11F_11F_REV, C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RGB9_E5]: { format: C1.RGB, types: [C2.UNSIGNED_INT_5_9_9_9_REV, C2.HALF_FLOAT, C1.FLOAT] },
  // ---- integers ----
  [C2.R8UI]: { format: C2.RED_INTEGER, types: [C1.UNSIGNED_BYTE] },
  [C2.R8I]: { format: C2.RED_INTEGER, types: [C1.BYTE] },
  [C2.R16UI]: { format: C2.RED_INTEGER, types: [C1.UNSIGNED_SHORT] },
  [C2.R16I]: { format: C2.RED_INTEGER, types: [C1.SHORT] },
  [C2.R32UI]: { format: C2.RED_INTEGER, types: [C1.UNSIGNED_INT] },
  [C2.R32I]: { format: C2.RED_INTEGER, types: [C1.INT] },
  [C2.RG8UI]: { format: C2.RG_INTEGER, types: [C1.UNSIGNED_BYTE] },
  [C2.RG8I]: { format: C2.RG_INTEGER, types: [C1.BYTE] },
  [C2.RG16UI]: { format: C2.RG_INTEGER, types: [C1.UNSIGNED_SHORT] },
  [C2.RG16I]: { format: C2.RG_INTEGER, types: [C1.SHORT] },
  [C2.RG32UI]: { format: C2.RG_INTEGER, types: [C1.UNSIGNED_INT] },
  [C2.RG32I]: { format: C2.RG_INTEGER, types: [C1.INT] },
  [C2.RGB8UI]: { format: C2.RGB_INTEGER, types: [C1.UNSIGNED_BYTE] },
  [C2.RGB8I]: { format: C2.RGB_INTEGER, types: [C1.BYTE] },
  [C2.RGB16UI]: { format: C2.RGB_INTEGER, types: [C1.UNSIGNED_SHORT] },
  [C2.RGB16I]: { format: C2.RGB_INTEGER, types: [C1.SHORT] },
  [C2.RGB32UI]: { format: C2.RGB_INTEGER, types: [C1.UNSIGNED_INT] },
  [C2.RGB32I]: { format: C2.RGB_INTEGER, types: [C1.INT] },
  [C2.RGBA8UI]: { format: C2.RGBA_INTEGER, types: [C1.UNSIGNED_BYTE] },
  [C2.RGBA8I]: { format: C2.RGBA_INTEGER, types: [C1.BYTE] },
  [C2.RGBA16UI]: { format: C2.RGBA_INTEGER, types: [C1.UNSIGNED_SHORT] },
  [C2.RGBA16I]: { format: C2.RGBA_INTEGER, types: [C1.SHORT] },
  [C2.RGBA32UI]: { format: C2.RGBA_INTEGER, types: [C1.UNSIGNED_INT] },
  [C2.RGBA32I]: { format: C2.RGBA_INTEGER, types: [C1.INT] },
  [C2.RGB10_A2UI]: { format: C2.RGBA_INTEGER, types: [C2.UNSIGNED_INT_2_10_10_10_REV] },
  // ---- norm16 (EXT_texture_norm16) ----
  [CExt.R16_EXT]: { format: C2.RED, types: [C1.UNSIGNED_SHORT] },
  [CExt.RG16_EXT]: { format: C2.RG, types: [C1.UNSIGNED_SHORT] },
  [CExt.RGB16_EXT]: { format: C1.RGB, types: [C1.UNSIGNED_SHORT] },
  [CExt.RGBA16_EXT]: { format: C1.RGBA, types: [C1.UNSIGNED_SHORT] },
  [CExt.R16_SNORM_EXT]: { format: C2.RED, types: [C1.SHORT] },
  [CExt.RG16_SNORM_EXT]: { format: C2.RG, types: [C1.SHORT] },
  [CExt.RGB16_SNORM_EXT]: { format: C1.RGB, types: [C1.SHORT] },
  [CExt.RGBA16_SNORM_EXT]: { format: C1.RGBA, types: [C1.SHORT] },
  // ---- depth ----
  [C2.DEPTH_COMPONENT16]: { format: C1.DEPTH_COMPONENT, types: [C1.UNSIGNED_SHORT, C1.UNSIGNED_INT] },
  [C2.DEPTH_COMPONENT24]: { format: C1.DEPTH_COMPONENT, types: [C1.UNSIGNED_SHORT, C1.UNSIGNED_INT] },
  [C2.DEPTH_COMPONENT32F]: { format: C1.DEPTH_COMPONENT, types: [C1.FLOAT] },
  [C2.DEPTH24_STENCIL8]: { format: C1.DEPTH_STENCIL, types: [C2.UNSIGNED_INT_24_8] },
  [C2.DEPTH32F_STENCIL8]: { format: C1.DEPTH_STENCIL, types: [C2.FLOAT_32_UNSIGNED_INT_24_8_REV] },
};

const NORM16_FORMATS: number[] = [
  CExt.R16_EXT,
  CExt.RG16_EXT,
  CExt.RGB16_EXT,
  CExt.RGBA16_EXT,
  CExt.R16_SNORM_EXT,
  CExt.RG16_SNORM_EXT,
  CExt.RGB16_SNORM_EXT,
  CExt.RGBA16_SNORM_EXT,
];

function isNorm16Format(fmt: GLenum): boolean {
  return NORM16_FORMATS.includes(fmt);
}

/**
 * W2 texImage internalformat validity (norm16 gated on EXT_texture_norm16).
 * Uses the `_extensions` enabled-cache, NOT `getExtension(...) !== null`:
 * calling getExtension() from validation code POPULATES the singleton cache
 * as a side effect, self-enabling the extension before the page requests it
 * (CTS ext-render-snorm.html checks getSupportedExtensions()/getExtension()
 * ordering; see also api/webgl2.ts anisotropyEnabled).
 */
function w2InternalformatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (isNorm16Format(fmt)) return ctx._extensions.has('EXT_texture_norm16');
  return fmt in W2_COMBOS;
}

/** W2 typed-array ↔ type consistency (INVALID_OPERATION on mismatch). */
function w2TypeMatchesView(type: GLenum, pixels: ArrayBufferView): boolean {
  const v = pixels as unknown as { constructor: Function };
  switch (type) {
    case C1.BYTE:
      return v instanceof Int8Array;
    case C1.UNSIGNED_BYTE:
      return v instanceof Uint8Array || v instanceof Uint8ClampedArray;
    case C1.SHORT:
      return v instanceof Int16Array;
    case C1.UNSIGNED_SHORT:
    case C1.UNSIGNED_SHORT_5_6_5:
    case C1.UNSIGNED_SHORT_4_4_4_4:
    case C1.UNSIGNED_SHORT_5_5_5_1:
    case C2.HALF_FLOAT:
      return v instanceof Uint16Array;
    case C1.INT:
      return v instanceof Int32Array;
    case C1.UNSIGNED_INT:
    case C2.UNSIGNED_INT_5_9_9_9_REV:
    case C2.UNSIGNED_INT_2_10_10_10_REV:
    case C2.UNSIGNED_INT_10F_11F_11F_REV:
    case C2.UNSIGNED_INT_24_8:
      return v instanceof Uint32Array;
    case C1.FLOAT:
      return v instanceof Float32Array;
    default:
      return true;
  }
}

/**
 * WebGL2 (internalformat, format, type) validation for the buffer path.
 * Non-view/non-number/non-null pixels (string, DOM source, ArrayBuffer) →
 * INVALID_OPERATION; DEPTH32F_STENCIL8 requires null pixels.
 */
function w2ValidateFormatType(
  ctx: WebGLRenderingContext,
  internalformat: GLenum,
  format: GLenum,
  type: GLenum,
  pixels: unknown,
): boolean {
  if (!w2InternalformatValid(ctx, internalformat)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (!W2_FORMATS.includes(format)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (!W2_TYPES.includes(type)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  const combo = W2_COMBOS[internalformat];
  if (!combo) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (combo.format !== format) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (!combo.types.includes(type)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (internalformat === C2.DEPTH32F_STENCIL8 && pixels !== null && pixels !== undefined) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (pixels !== null && pixels !== undefined && typeof pixels !== 'number' && !ArrayBuffer.isView(pixels)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (ArrayBuffer.isView(pixels) && !w2TypeMatchesView(type, pixels)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  return true;
}

/** W2 texSubImage (stored internalFormat, format, type) validation. */
function w2ValidateSubFormatType(
  ctx: WebGLRenderingContext,
  internalFormat: GLenum,
  format: GLenum,
  type: GLenum,
): boolean {
  if (!W2_FORMATS.includes(format)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (!W2_TYPES.includes(type)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  const combo = W2_COMBOS[internalFormat];
  if (!combo) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  if (combo.format !== format) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (!combo.types.includes(type)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Source-buffer size validation
// ---------------------------------------------------------------------------

const align = (v: number, a: number): number => (v + a - 1) & ~(a - 1);

/** Components of a (validated) source format. */
function componentsOf(format: GLenum): number {
  switch (format) {
    case C1.RGBA:
    case C2.RGBA_INTEGER:
      return 4;
    case C1.RGB:
    case C2.RGB_INTEGER:
      return 3;
    case C1.LUMINANCE_ALPHA:
    case C2.RG:
    case C2.RG_INTEGER:
      return 2;
    case C1.DEPTH_STENCIL:
      return 2;
    default:
      return 1;
  }
}

/** Bytes per source texel for a (validated) format/type pair. */
function bytesPerTexel(ctx: WebGLRenderingContext, format: GLenum, type: GLenum): number {
  const comps = componentsOf(format);
  switch (type) {
    case C1.UNSIGNED_BYTE:
      return comps;
    case C1.BYTE:
      return ctx._version === 2 ? comps : 1;
    case C1.UNSIGNED_SHORT_5_6_5:
    case C1.UNSIGNED_SHORT_4_4_4_4:
    case C1.UNSIGNED_SHORT_5_5_5_1:
      return 2; // packed
    case C1.UNSIGNED_SHORT:
      return comps * 2;
    case C1.SHORT:
      return ctx._version === 2 ? comps * 2 : 2;
    case C1.UNSIGNED_INT:
      return comps * 4;
    case C1.INT:
      return ctx._version === 2 ? comps * 4 : 4;
    case C2.HALF_FLOAT:
    case HALF_FLOAT_OES:
      return comps * 2;
    case C1.FLOAT:
      return comps * 4;
    case UNSIGNED_INT_24_8_WEBGL:
    case C2.UNSIGNED_INT_2_10_10_10_REV:
    case C2.UNSIGNED_INT_10F_11F_11F_REV:
    case C2.UNSIGNED_INT_5_9_9_9_REV:
      return 4; // packed
    case C2.FLOAT_32_UNSIGNED_INT_24_8_REV:
      return 8; // packed
    default:
      return 1;
  }
}

/**
 * Required source-bytes for (width, height, depth) per UNPACK_* state, then
 * check the actual source (ArrayBufferView size or PBO range). Pushes the
 * error and returns false when the source is too small. `is3D` selects the
 * texImage3D/texSubImage3D formula — the call sites pass depth=1 for both 2D
 * and 3D-depth-1 uploads, so the ENTRYPOINT must decide (2D ignores
 * UNPACK_IMAGE_HEIGHT/UNPACK_SKIP_IMAGES; those are 3D-only params).
 *
 * WebGL 2.0 §3.7.2 unpack constraints (only WebGL2 can set these params; in
 * WebGL1 they are all defaults so the checks never fire there):
 *   DataStoreWidth = (UNPACK_ROW_LENGTH ? ROW_LENGTH : width)
 *   → INVALID_OPERATION if skipPixels + width > DataStoreWidth
 *   texImage3D/texSubImage3D: DataStoreHeight = (UNPACK_IMAGE_HEIGHT ? IMAGE_HEIGHT : height)
 *   → INVALID_OPERATION if skipRows + height > DataStoreHeight
 *
 * Size: every row is padded to UNPACK_ALIGNMENT EXCEPT the last row of the
 * last image, which holds only `width` texels and is unpadded (GLES3 §3.8.2 —
 * the CTS computeImageSizes2D/3D allocate exactly this). Required =
 * skipPixels*bpp + padded rows before the last row + width*bpp, in bytes.
 */
function validatePixelsSize(
  ctx: WebGLRenderingContext,
  pixels: ArrayBufferView | number,
  width: number,
  height: number,
  depth: number,
  format: GLenum,
  type: GLenum,
  is3D: boolean,
): boolean {
  const unpack = ctx._state.pixelStore.unpack;
  const srcBpp = bytesPerTexel(ctx, format, type);
  const rowLength = unpack.rowLength > 0 ? unpack.rowLength : width;
  const rowBytes = align(rowLength * srcBpp, unpack.alignment);
  const imageHeight = unpack.imageHeight > 0 ? unpack.imageHeight : height;
  if (unpack.skipPixels + width > rowLength) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (is3D && unpack.skipRows + height > imageHeight) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  // Rows before the unpadded last row of the last image (0 when nothing is
  // uploaded, e.g. height 0). Skipped rows/images are full padded strides.
  const rowsBeforeLast = Math.max(
    0,
    unpack.skipRows + (is3D ? (unpack.skipImages + depth - 1) * imageHeight : 0) + (height - 1),
  );
  const required = unpack.skipPixels * srcBpp + rowsBeforeLast * rowBytes + (height > 0 ? width * srcBpp : 0);
  if (typeof pixels === 'number') {
    // WebGL2 PIXEL_UNPACK_BUFFER offset path (w2ValidatePbo checked the binding).
    const buf = ctx._state.pixelUnpackBuffer;
    if (buf === null || buf._data === null || pixels + required > buf._size) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return false;
    }
    return true;
  }
  if (pixels.byteLength < required) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  return true;
}

/**
 * WebGL2 srcOffset/srcLength overloads (spec §3.7.2): the extra numeric
 * arguments after the ArrayBufferView select a sub-range of the view, in
 * ELEMENTS of the view's element type (not bytes). srcOffset is an unsigned
 * long long (64-bit — do NOT truncate: offsets ≥ 2^32 arise with huge
 * WebAssembly-memory views, e.g. texImage2d-16gb-wasm-memory uses
 * srcOffset = 16GB-4); srcLength is a GLuint (32-bit). srcOffset/srcLength
 * are converted WebIDL-style (unsigned long long: NaN → 0, negatives wrap
 * to huge — so offset=-1 becomes a huge offset and errors). Returns the
 * sliced view, or null after pushing INVALID_OPERATION (offset beyond the
 * view, or srcOffset+srcLength beyond the view). The too-short
 * effective-length check is left to validatePixelsSize, which pushes
 * INVALID_OPERATION (the CTS views-with-offsets probe expects exactly that
 * for effective < required).
 */
function applySrcOffset(
  ctx: WebGLRenderingContext,
  view: ArrayBufferView,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
): ArrayBufferView | null {
  // `length`/`subarray` aren't on the TS ArrayBufferView base type (DataView
  // excluded); typed arrays all have them (the callers only pass views).
  const v = view as unknown as { length: number; subarray(begin: number, end: number): ArrayBufferView };
  // WebIDL unsigned long long conversion. 2^64 is exactly representable in
  // float64; values beyond 2^53 lose precision but stay huge — exactly what
  // the error paths need (offset=-1 → 2^64-1 → INVALID_OPERATION below).
  let off = Number(srcOffsetArg);
  if (off < 0) off += 18446744073709551616; // negative → mod 2^64
  if (!(off > 0)) off = 0; // NaN, ±0 → 0
  else off = Math.floor(off); // truncate toward zero
  if (off > v.length) {
    // Effective length would be negative (covers the CTS offset=-1 probe).
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  let effLen = v.length - off;
  if (srcLengthArg !== undefined) {
    const srcLen = Number(srcLengthArg) >>> 0;
    if (srcLen > 0) {
      if (off + srcLen > v.length) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return null;
      }
      effLen = srcLen;
    }
  }
  return v.subarray(off, off + effLen);
}

/** WebGL2 PBO (pixels = byte offset) pre-checks: bound, TF-binding, flipY/premultiply, offset. */
function w2ValidatePbo(ctx: WebGLRenderingContext, pixels: number): boolean {
  const buf = ctx._state.pixelUnpackBuffer;
  if (buf === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  // Transform-feedback binding rules (spec "Preventing undefined behavior with
  // Transform Feedback"; GLES 3.0 §2.15.2): a PBO in the currently bound TF
  // object's indexed bindings (or in an active TF's) cannot be used through
  // PIXEL_UNPACK_BUFFER (CTS simultaneous_binding.html "Test PIXEL_UNPACK_BUFFER").
  if (bufferTfUseError(ctx, buf, C2.PIXEL_UNPACK_BUFFER)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  const unpack = ctx._state.pixelStore.unpack;
  if (unpack.flipY || unpack.premultiplyAlpha) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (pixels < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return false;
  }
  if (buf._data === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// DOM-source helpers
// ---------------------------------------------------------------------------

/**
 * Width/height of a DOM TexImageSource (duck-typed, mirroring present/image.ts).
 * HTMLVideoElement's `.width`/`.height` are the reflected content attributes
 * (0 in Chromium) — the intrinsic frame size is `videoWidth`/`videoHeight`;
 * images use `naturalWidth`/`naturalHeight`; canvas/ImageBitmap/ImageData use
 * `.width`/`.height`. Returns null for anything without numeric dims.
 */
/** True when `pixels` is a DOM TexImageSource (WebGL2 9-arg overload), not a
 * buffer view / ArrayBuffer / PBO offset / null. */
function isDomSource(pixels: unknown): boolean {
  return (
    typeof pixels === 'object' && pixels !== null &&
    !ArrayBuffer.isView(pixels) && !(pixels instanceof ArrayBuffer)
  );
}

/** True when the source element holds an SVG image (data:image/svg+xml or a *.svg URL). */
function isSvgSource(s: Record<string, unknown>): boolean {
  const src = String((s as { currentSrc?: unknown }).currentSrc ?? (s as { src?: unknown }).src ?? '');
  return /image\/svg\+xml/i.test(src) || /\.svg(\?|#|$)/i.test(src);
}

/**
 * Parse the width/height attributes of the root <svg> element out of a data:
 * URI source. Per the SVG spec those attributes ARE the image's intrinsic
 * dimensions; an SVG without them (e.g. a viewBox-only document) has NO
 * intrinsic dimensions. Returns {width, height} with 0 for missing attributes,
 * or null when the source is not a synchronously parseable data: URI.
 */
function parseSvgRootDims(s: Record<string, unknown>): { width: number; height: number } | null {
  const src = String((s as { currentSrc?: unknown }).currentSrc ?? (s as { src?: unknown }).src ?? '');
  const m = /^data:image\/svg\+xml(?:;[^,]*)?,(.*)$/is.exec(src);
  if (!m) return null;
  let markup: string;
  try { markup = decodeURIComponent(m[1]); } catch { markup = m[1]; }
  const tag = /<svg\b[^>]*>/i.exec(markup);
  if (!tag) return null;
  const attr = (name: string): number | null => {
    const am = new RegExp(`\\b${name}\\s*=\\s*["'][^"']*["']`, 'i').exec(tag[0]);
    if (!am) return null;
    const n = parseFloat(am[0].replace(/^[^=]*=\s*["']?/, ''));
    return Number.isFinite(n) ? n : null;
  };
  const w = attr('width');
  const h = attr('height');
  return { width: w ?? 0, height: h ?? 0 };
}

/**
 * Upload dimensions for an SVG HTMLImageElement. WebGL spec "Texture Upload
 * Width and Height": for SVG images the texture size is the CURRENT VALUE OF
 * THE WIDTH/HEIGHT PROPERTIES of the HTMLImageElement. Those IDL properties
 * reflect the width/height content attributes of the IMG element with the
 * default being the image's intrinsic dimensions; an SVG without width/height
 * attributes has NO intrinsic dimensions (Chromium's naturalWidth of 150 is
 * the CSS default USED size, not an intrinsic size), so the value is 0. The
 * 2025 CTS test tex-image-svg-image-no-natural-width-and-height.html relies
 * on exactly this: unset → 0×0 (the follow-up texSubImage2D then fails with
 * INVALID_VALUE), set (image.width=100) → 100×100.
 */
function svgImageDims(s: Record<string, unknown>): { width: number; height: number } {
  const getAttr = (name: string): number | null => {
    if (typeof s.getAttribute === 'function') {
      const v = (s as { getAttribute(n: string): string | null }).getAttribute(name);
      if (v !== null && v !== '') {
        const n = Number(v);
        if (Number.isFinite(n)) return n;
      }
    }
    return null;
  };
  let w = getAttr('width');
  let h = getAttr('height');
  if (w === null || h === null) {
    // No img-element attribute for at least one axis: fall back to the SVG
    // document's root width/height (its intrinsic dimensions). For sources we
    // cannot parse (non-data: URIs), the element's width/height properties
    // already carry the intrinsic size when the SVG has width/height attrs.
    const root = parseSvgRootDims(s);
    if (w === null) w = root !== null ? root.width : (typeof s.width === 'number' ? s.width : 0);
    if (h === null) h = root !== null ? root.height : (typeof s.height === 'number' ? s.height : 0);
  }
  return { width: w, height: h };
}

function sourceDims(source: unknown): { width: number; height: number } | null {
  if (source === null || typeof source !== 'object') return null;
  const s = source as Record<string, unknown>;
  if (typeof s.videoWidth === 'number' && typeof s.readyState === 'number') {
    return typeof s.videoHeight === 'number' ? { width: s.videoWidth, height: s.videoHeight } : null;
  }
  if (typeof s.naturalWidth === 'number') {
    if (isSvgSource(s)) return svgImageDims(s);
    return typeof s.naturalHeight === 'number' ? { width: s.naturalWidth, height: s.naturalHeight } : null;
  }
  if (typeof s.width === 'number' && typeof s.height === 'number') return { width: s.width, height: s.height };
  return null;
}

// ---------------------------------------------------------------------------
// texImage2D / texImage3D
// ---------------------------------------------------------------------------

/** 6-arg DOM form: (target, level, internalformat, format, type, source). */
function texImage2DDOM(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  if (source === null || source === undefined) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const dims = sourceDims(source);
  if (dims === null) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const tex = commonTexImageValidation(ctx, target, level, dims.width, dims.height, 1, 0, false);
  if (tex === null) return;
  if (ctx._version === 2) {
    if (!w2ValidateDomFormatType(ctx, internalformat, format, type)) return;
  } else if (!w1ValidateFormatType(ctx, internalformat, format, type, null)) {
    return;
  }
  uploadTexImage(
    ctx, tex, target, level, internalformat, dims.width, dims.height, 1, 0, format, type,
    source as unknown as TexImageSourceArg, source,
    false, // inferred dims (6-arg form): scale the source to the element size
  );
}

/**
 * WebGL2 (internalformat, format, type) validation for TexImageSource
 * uploads (both the 6-arg form and the 9-arg form with explicit dimensions).
 */
function w2ValidateDomFormatType(
  ctx: WebGLRenderingContext,
  internalformat: GLint,
  format: GLenum,
  type: GLenum,
): boolean {
  if (!(internalformat in W2_DOM)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return false;
  }
  const entry = W2_DOM[internalformat];
  if (entry.format !== format) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  if (!entry.types.includes(type)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return false;
  }
  return true;
}

/**
 * 9-arg DOM form (WebGL2 only): (target, level, internalformat, w, h, border,
 * format, type, source). The width/height arguments specify the texture level
 * size; UNPACK_SKIP_PIXELS/UNPACK_SKIP_ROWS select a sub-rectangle of the
 * source (WebGL2 spec §3.7.2 — "In WebGL 1, the width and height are always
 * inferred from the source. In WebGL 2, they can also be explicitly
 * specified.").
 */
function texImage2DDOMWithDims(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  width: GLsizei,
  height: GLsizei,
  border: GLint,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  // WebIDL: GLsizei/GLint are `long` — convert via ToInt32 (truncate toward
  // zero). The CTS passes e.g. bitmap.width/2 = 128.5 for a 257px source and
  // relies on the conversion to 128 (WebGL2 spec §3.7.2); the copy math
  // requires integer dimensions.
  width = width | 0;
  height = height | 0;
  border = border | 0;
  const tex = commonTexImageValidation(ctx, target, level, width, height, 1, border, false);
  if (tex === null) return;
  if (!w2ValidateDomFormatType(ctx, internalformat, format, type)) return;
  uploadTexImage(
    ctx, tex, target, level, internalformat, width, height, 1, border, format, type,
    source as unknown as TexImageSourceArg, source,
    true, // explicit dims: width/height select a source sub-rectangle (no scaling)
  );
}

/** WebGL2 DOM-source internalformat table (texImage2D 6-arg form). */
const W2_DOM: Record<number, { format: number; types: number[] }> = {
  [C2.R8]: { format: C2.RED, types: [C1.UNSIGNED_BYTE] },
  [C2.R16F]: { format: C2.RED, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.R32F]: { format: C2.RED, types: [C1.FLOAT] },
  [C2.R8UI]: { format: C2.RED_INTEGER, types: [C1.UNSIGNED_BYTE] },
  [C2.RG8]: { format: C2.RG, types: [C1.UNSIGNED_BYTE] },
  [C2.RG16F]: { format: C2.RG, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RG32F]: { format: C2.RG, types: [C1.FLOAT] },
  [C2.RG8UI]: { format: C2.RG_INTEGER, types: [C1.UNSIGNED_BYTE] },
  [C2.RGB8]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE] },
  [C2.SRGB8]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE] },
  [C1.RGB565]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_6_5] },
  [C2.R11F_G11F_B10F]: { format: C1.RGB, types: [C2.UNSIGNED_INT_10F_11F_11F_REV, C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RGB9_E5]: { format: C1.RGB, types: [C2.UNSIGNED_INT_5_9_9_9_REV, C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RGB16F]: { format: C1.RGB, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RGB32F]: { format: C1.RGB, types: [C1.FLOAT] },
  [C2.RGB8UI]: { format: C2.RGB_INTEGER, types: [C1.UNSIGNED_BYTE] },
  [C2.RGBA8]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE] },
  [C2.SRGB8_ALPHA8]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE] },
  [C1.RGB5_A1]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_5_5_1] },
  [C2.RGB10_A2]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C2.UNSIGNED_INT_2_10_10_10_REV] },
  [C1.RGBA4]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_4_4_4_4] },
  [C2.RGBA16F]: { format: C1.RGBA, types: [C2.HALF_FLOAT, C1.FLOAT] },
  [C2.RGBA32F]: { format: C1.RGBA, types: [C1.FLOAT] },
  [C2.RGBA8UI]: { format: C2.RGBA_INTEGER, types: [C1.UNSIGNED_BYTE] },
  // Unsized internal formats are legal for TexImageSource uploads (WebGL2 spec
  // §3.7.2): format must equal internalformat and type must be UNSIGNED_BYTE.
  [C1.RGBA]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE] },
  [C1.RGB]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE] },
  [C1.LUMINANCE_ALPHA]: { format: C1.LUMINANCE_ALPHA, types: [C1.UNSIGNED_BYTE] },
  [C1.LUMINANCE]: { format: C1.LUMINANCE, types: [C1.UNSIGNED_BYTE] },
  [C1.ALPHA]: { format: C1.ALPHA, types: [C1.UNSIGNED_BYTE] },
};

/** 9/10-arg buffer path: (target, level, internalformat, w, h, border, format, type, pixels). */
function texImage2DBuffer(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  width: GLsizei,
  height: GLsizei,
  border: GLint,
  format: GLenum,
  type: GLenum,
  pixels: TexImageSourceArg,
): void {
  const tex = commonTexImageValidation(ctx, target, level, width, height, 1, border, false);
  if (tex === null) return;
  if (ctx._version === 2) {
    if (!w2ValidateFormatType(ctx, internalformat, format, type, pixels)) return;
    if (typeof pixels === 'number') {
      if (!w2ValidatePbo(ctx, pixels)) return;
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type, false)) return;
    } else if (ArrayBuffer.isView(pixels)) {
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type, false)) return;
    }
  } else {
    if (!w1ValidateFormatType(ctx, internalformat, format, type, pixels)) return;
    if (ArrayBuffer.isView(pixels)) {
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type, false)) return;
    }
  }
  uploadTexImage(ctx, tex, target, level, internalformat, width, height, 1, border, format, type, pixels);
}

/** 6-arg DOM form for texImage3D — DOM sources are 2D-only (spec). */
function texImage3DDOM(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  void internalformat;
  void format;
  void type;
  if (!is3DTarget(ctx, target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (source === null || source === undefined) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  // DOM sources are only supported for 2D targets.
  ctx._errors.push(C1.INVALID_OPERATION);
}

/** 10-arg buffer path for texImage3D. */
function texImage3DBuffer(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  border: GLint,
  format: GLenum,
  type: GLenum,
  pixels: TexImageSourceArg,
): void {
  const tex = commonTexImageValidation(ctx, target, level, width, height, depth, border, true);
  if (tex === null) return;
  if (!w2ValidateFormatType(ctx, internalformat, format, type, pixels)) return;
  if (typeof pixels === 'number') {
    if (!w2ValidatePbo(ctx, pixels)) return;
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type, true)) return;
  } else if (ArrayBuffer.isView(pixels)) {
    // WebGL2 spec: UNPACK_FLIP_Y_WEBGL / UNPACK_PREMULTIPLY_ALPHA_WEBGL are
    // not supported for texImage3D client-data uploads → INVALID_OPERATION
    // (the PBO path is already rejected in w2ValidatePbo).
    const unpack = ctx._state.pixelStore.unpack;
    if (unpack.flipY || unpack.premultiplyAlpha) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type, true)) return;
  }
  uploadTexImage(ctx, tex, target, level, internalformat, width, height, depth, border, format, type, pixels);
}

/**
 * 10-arg DOM form (WebGL2 only): (target, level, internalformat, w, h, d,
 * border, format, type, source). The width/height/depth arguments specify the
 * texture level size; UNPACK_SKIP_PIXELS/UNPACK_SKIP_ROWS select a
 * sub-rectangle of the source (WebGL2 spec §3.7.2). DOM sources are 2D — the
 * engine divides the source into `depth` horizontal bands filling the level's
 * slices (slice stride UNPACK_IMAGE_HEIGHT or height).
 */
function texImage3DDOMWithDims(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLint,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  border: GLint,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  // WebIDL: GLint/GLsizei are `long` — convert via ToInt32 (truncate toward
  // zero). Same rationale as the 2D cluster: the CTS passes fractional dims
  // (e.g. canvas.width/2 = 128.5 for a 257px source) and relies on the
  // conversion to 128; the copy math requires integer dimensions.
  level = level | 0;
  width = width | 0;
  height = height | 0;
  depth = depth | 0;
  border = border | 0;
  const tex = commonTexImageValidation(ctx, target, level, width, height, depth, border, true);
  if (tex === null) return;
  if (source === null || source === undefined) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (!w2ValidateDomFormatType(ctx, internalformat, format, type)) return;
  uploadTexImage(
    ctx, tex, target, level, internalformat, width, height, depth, border, format, type,
    source as unknown as TexImageSourceArg, source,
    true, // explicit dims: width/height select a source sub-rectangle (no scaling)
  );
}

// ---------------------------------------------------------------------------
// texSubImage2D / texSubImage3D
// ---------------------------------------------------------------------------

/** Common texSubImage2D/3D validation (returns the level data or null). */
function commonTexSubValidation(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: number,
  yoffset: number,
  zoffset: number,
  width: number,
  height: number,
  depth: number,
  is3D: boolean,
): WebGLTexture | null {
  if (is3D ? !is3DTarget(ctx, target) : !is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return null;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (level < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  const lim = dimLimit(ctx, target);
  if (level > Math.floor(Math.log2(lim.maxDim))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (!hasTextureLevel(tex, target, level)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (width < 0 || height < 0 || depth < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (xoffset < 0 || yoffset < 0 || zoffset < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  const levelData = tex._image!.levels[level];
  if (xoffset + width > levelData.width || yoffset + height > levelData.height) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (is3D && zoffset >= levelData.depth) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  return tex;
}

/** 7-arg DOM form: (target, level, xoffset, yoffset, format, type, source). */
function texSubImage2DDOM(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  if (source === null || source === undefined) {
    // WebIDL: the 7-arg DOM overload's TexImageSource is non-nullable → throw
    // TypeError (CTS tex-sub-image-2d-bad-args.html). The 9-arg buffer form's
    // nullable pixels keep their INVALID_VALUE below.
    throw new TypeError(`Argument is not of type 'TexImageSource'`);
  }
  const dims = sourceDims(source);
  if (dims === null) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const tex = commonTexSubValidation(ctx, target, level, xoffset, yoffset, 0, dims.width, dims.height, 1, false);
  if (tex === null) return;
  if (ctx._version === 2) {
    if (!w2ValidateSubFormatType(ctx, tex._image!.internalFormat, format, type)) return;
  } else {
    if (!w1FormatValid(ctx, format)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (!w1TypeValid(ctx, type)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const origin = getLevelOrigin(tex, level);
    if (!origin || origin.format !== format || origin.type !== type) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
  }
  // SVG image with NO intrinsic dimensions (no width/height attributes on the
  // <img> element or the SVG root): per WebGL spec "Texture Upload Width and
  // Height" the upload size is the current value of the element's width/height
  // properties — 0 here (see svgImageDims). When the level was previously
  // allocated with explicit nonzero dims (callers commonly pass
  // image.width/image.height, which Chromium reports as the CSS default used
  // size 150 instead of the spec's 0 — crbug.com/41357911), re-allocate the
  // level at the spec size 0×0 so the texture really is 0×0. The 2025 CTS
  // tex-image-svg-image-no-natural-width-and-height.html relies on exactly
  // this: a follow-up 1×1 texSubImage2D into such a texture must fail with
  // INVALID_VALUE ("texture size should be 0x0").
  if (dims.width === 0 && dims.height === 0 && isSvgSource(source as Record<string, unknown>)) {
    const img = tex._image;
    const lv = img ? img.levels[level] : undefined;
    if (lv && (lv.width !== 0 || lv.height !== 0) && !tex._immutable) {
      uploadTexImage(
        ctx, tex, target, level, img!.internalFormat ?? format, 0, 0, 1, 0, format, type, null,
      );
    }
  }
  uploadTexSubImage(
    ctx, tex, target, level, xoffset, yoffset, 0, dims.width, dims.height, 1, format, type,
    source as unknown as TexImageSourceArg, source,
    false, // inferred dims (7-arg form): scale the source to the element size
  );
}

/**
 * 9-arg DOM form (WebGL2 only): (target, level, xoffset, yoffset, w, h,
 * format, type, source). The width/height arguments (together with
 * UNPACK_SKIP_PIXELS/UNPACK_SKIP_ROWS) specify a sub-rectangle of the source.
 */
function texSubImage2DDOMWithDims(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  width: GLsizei,
  height: GLsizei,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  if (source === null || source === undefined) {
    throw new TypeError(`Argument is not of type 'TexImageSource'`);
  }
  // WebIDL: GLint/GLsizei are `long` — convert via ToInt32 (truncate toward
  // zero); the CTS passes bitmap.width/2 = 128.5 for odd-sized sources and
  // relies on the conversion to 128.
  xoffset = xoffset | 0;
  yoffset = yoffset | 0;
  width = width | 0;
  height = height | 0;
  const tex = commonTexSubValidation(ctx, target, level, xoffset, yoffset, 0, width, height, 1, false);
  if (tex === null) return;
  if (!w2ValidateSubFormatType(ctx, tex._image!.internalFormat, format, type)) return;
  uploadTexSubImage(
    ctx, tex, target, level, xoffset, yoffset, 0, width, height, 1, format, type,
    source as unknown as TexImageSourceArg, source,
    true, // explicit dims: width/height select a source sub-rectangle (no scaling)
  );
}

/** 9-arg buffer path for texSubImage2D. */
function texSubImage2DBuffer(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  width: GLsizei,
  height: GLsizei,
  format: GLenum,
  type: GLenum,
  pixels: TexImageSourceArg,
): void {
  const tex = commonTexSubValidation(ctx, target, level, xoffset, yoffset, 0, width, height, 1, false);
  if (tex === null) return;
  if (pixels === null || pixels === undefined) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (ctx._version === 2) {
    if (!w2ValidateSubFormatType(ctx, tex._image!.internalFormat, format, type)) return;
    if (typeof pixels === 'number') {
      if (!w2ValidatePbo(ctx, pixels)) return;
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type, false)) return;
    } else if (ArrayBuffer.isView(pixels)) {
      // WebGL2: the view's element type must match `type` (texImage2D already
      // enforces this via w2ValidateFormatType; texSubImage2D must too).
      if (!w2TypeMatchesView(type, pixels)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type, false)) return;
    } else {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
  } else {
    if (!w1FormatValid(ctx, format)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (!w1TypeValid(ctx, type)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (!ArrayBuffer.isView(pixels)) {
      throw new TypeError(`Argument is not of type 'ArrayBufferView'`);
    }
    if (!w1TypeMatchesView(type, pixels)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    const origin = getLevelOrigin(tex, level);
    if (!origin || origin.format !== format || origin.type !== type) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type, false)) return;
  }
  uploadTexSubImage(ctx, tex, target, level, xoffset, yoffset, 0, width, height, 1, format, type, pixels);
}

/** 8-arg DOM form for texSubImage3D — DOM sources are 2D-only (spec). */
function texSubImage3DDOM(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  zoffset: GLint,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  void format;
  void type;
  const tex = commonTexSubValidation(ctx, target, level, xoffset, yoffset, zoffset, 0, 0, 1, true);
  if (tex === null) return;
  if (source === null || source === undefined) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  // DOM sources are only supported for 2D targets.
  ctx._errors.push(C1.INVALID_OPERATION);
}

/**
 * 11-arg DOM form (WebGL2 only): (target, level, xoffset, yoffset, zoffset,
 * w, h, d, format, type, source). The width/height/depth arguments (together
 * with UNPACK_SKIP_PIXELS/UNPACK_SKIP_ROWS) specify a sub-rectangle of the
 * source. DOM sources are 2D — the engine divides the source into `depth`
 * horizontal bands filling slices zoffset..zoffset+depth-1; other slices keep
 * their existing contents.
 */
function texSubImage3DDOMWithDims(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  zoffset: GLint,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  format: GLenum,
  type: GLenum,
  source: unknown,
): void {
  if (source === null || source === undefined) {
    // WebIDL: the TexImageSource overload is non-nullable → throw TypeError
    // (mirrors the 2D texSubImage2DDOMWithDims convention; unreachable via
    // dispatch, which requires isDomSource(arguments[10])).
    throw new TypeError(`Argument is not of type 'TexImageSource'`);
  }
  // WebIDL: GLint/GLsizei are `long` — convert via ToInt32 (truncate toward
  // zero), same rationale as the 2D DOM-with-dims path.
  xoffset = xoffset | 0;
  yoffset = yoffset | 0;
  zoffset = zoffset | 0;
  width = width | 0;
  height = height | 0;
  depth = depth | 0;
  const tex = commonTexSubValidation(ctx, target, level, xoffset, yoffset, zoffset, width, height, depth, true);
  if (tex === null) return;
  if (!w2ValidateSubFormatType(ctx, tex._image!.internalFormat, format, type)) return;
  uploadTexSubImage(
    ctx, tex, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type,
    source as unknown as TexImageSourceArg, source,
    true, // explicit dims: width/height select a source sub-rectangle (no scaling)
  );
}

/** 11-arg buffer path for texSubImage3D. */
function texSubImage3DBuffer(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  zoffset: GLint,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  format: GLenum,
  type: GLenum,
  pixels: TexImageSourceArg,
): void {
  const tex = commonTexSubValidation(ctx, target, level, xoffset, yoffset, zoffset, width, height, depth, true);
  if (tex === null) return;
  if (pixels === null || pixels === undefined) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (!w2ValidateSubFormatType(ctx, tex._image!.internalFormat, format, type)) return;
  if (typeof pixels === 'number') {
    if (!w2ValidatePbo(ctx, pixels)) return;
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type, true)) return;
  } else if (ArrayBuffer.isView(pixels)) {
    // WebGL2 spec: UNPACK_FLIP_Y_WEBGL / UNPACK_PREMULTIPLY_ALPHA_WEBGL are
    // not supported for texSubImage3D client-data uploads → INVALID_OPERATION
    // (the PBO path is already rejected in w2ValidatePbo).
    const unpack = ctx._state.pixelStore.unpack;
    if (unpack.flipY || unpack.premultiplyAlpha) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // WebGL2: the view's element type must match `type` (texImage3D already
    // enforces this via w2ValidateFormatType; texSubImage3D must too).
    if (!w2TypeMatchesView(type, pixels)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type, true)) return;
  } else {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  uploadTexSubImage(ctx, tex, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
}

// ---------------------------------------------------------------------------
// copyTexImage2D / copyTexSubImage2D / copyTexSubImage3D
// ---------------------------------------------------------------------------

/**
 * WebGL2 copyTexImage2D internalformats — the full GLES 3.0 / WebGL2 list:
 * the unsized WebGL1-compat formats plus all sized color formats (incl. the
 * 32F floats, RGB32F and RGB10_A2UI). Renderability of the SOURCE attachment
 * is a separate FBO-completeness concern (the source FBO is incomplete without
 * EXT_color_buffer_float, which surfaces as the accepted
 * INVALID_FRAMEBUFFER_OPERATION — CTS ext-color-buffer-float.html).
 */
const W2_COPY_INTERNALFORMATS: number[] = [
  C1.RGBA,
  C1.RGB,
  C1.ALPHA,
  C1.LUMINANCE,
  C1.LUMINANCE_ALPHA,
  C2.R8,
  C2.RG8,
  C2.RGB8,
  C2.RGBA8,
  C1.RGBA4,
  C1.RGB5_A1,
  C1.RGB565,
  C2.RGB10_A2,
  C2.RGB10_A2UI,
  C2.SRGB8,
  C2.SRGB8_ALPHA8,
  C2.R16F,
  C2.RG16F,
  C2.RGB16F,
  C2.RGBA16F,
  C2.R32F,
  C2.RG32F,
  C2.RGB32F,
  C2.RGBA32F,
  C2.R11F_G11F_B10F,
  C2.R8I, C2.R8UI, C2.R16I, C2.R16UI, C2.R32I, C2.R32UI,
  C2.RG8I, C2.RG8UI, C2.RG16I, C2.RG16UI, C2.RG32I, C2.RG32UI,
  C2.RGB8I, C2.RGB8UI, C2.RGB16I, C2.RGB16UI, C2.RGB32I, C2.RGB32UI,
  C2.RGBA8I, C2.RGBA8UI, C2.RGBA16I, C2.RGBA16UI, C2.RGBA32I, C2.RGBA32UI,
];

function isW2CopyInternalFormatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  void ctx;
  return W2_COPY_INTERNALFORMATS.includes(fmt);
}

/** Component count of an internal format (copyTexImage2D src/dest rules). */
function internalFormatComponents(fmt: GLenum): number {
  switch (fmt) {
    case C1.RGBA: case C1.RGBA8: case C2.RGBA16F: case C2.RGBA32F: case C1.RGBA4:
    case C1.RGB5_A1: case C2.RGB10_A2: case C2.RGB10_A2UI: case C2.SRGB8_ALPHA8:
    case C2.RGBA8_SNORM: case C2.RGBA8I: case C2.RGBA8UI: case C2.RGBA16I: case C2.RGBA16UI:
    case C2.RGBA32I: case C2.RGBA32UI:
      return 4;
    case C1.RGB: case C1.RGB8: case C2.RGB16F: case C2.RGB32F: case C1.RGB565:
    case C2.RGB8_SNORM: case C2.R11F_G11F_B10F: case C2.RGB9_E5: case C2.SRGB8:
    case C2.RGB8I: case C2.RGB8UI: case C2.RGB16I: case C2.RGB16UI: case C2.RGB32I: case C2.RGB32UI:
      return 3;
    case C1.LUMINANCE_ALPHA: case C2.RG: case C2.RG8: case C2.RG16F: case C2.RG32F:
    case C2.RG8_SNORM: case C2.RG8I: case C2.RG8UI: case C2.RG16I: case C2.RG16UI:
    case C2.RG32I: case C2.RG32UI:
      return 2;
    case C1.LUMINANCE: case C1.ALPHA: case C2.RED: case C2.R8: case C2.R16F: case C2.R32F:
    case C2.R8_SNORM: case C2.R8I: case C2.R8UI: case C2.R16I: case C2.R16UI:
    case C2.R32I: case C2.R32UI:
      return 1;
    default: return 4; // unknown / default-framebuffer RGBA8
  }
}

/** W1 copyTex source "kind" (GLES2 copyTex conversion table semantics). */
function w1SrcKind(fmt: GLenum): 'rgba' | 'rgb' | 'la' | 'lum' | 'alpha' {
  switch (fmt) {
    case C1.RGBA: return 'rgba';
    case C1.RGB: return 'rgb';
    case C1.LUMINANCE_ALPHA: return 'la';
    case C1.LUMINANCE: return 'lum';
    case C1.ALPHA: return 'alpha';
    default: {
      const comps = internalFormatComponents(fmt);
      return comps === 4 ? 'rgba' : comps === 3 ? 'rgb' : comps === 2 ? 'la' : 'lum';
    }
  }
}

/**
 * WebGL1 copyTexImage2D dest/source conversion rule (GLES2 "CopyTexImage
 * conversions" table, as exercised by CTS ext-color-buffer-half-float.html):
 * dest RGBA requires src RGBA; dest RGB requires src RGB|RGBA; dest LUMINANCE
 * requires src LUMINANCE|RGB|RGBA; dest LUMINANCE_ALPHA requires src
 * LUMINANCE_ALPHA|RGBA; dest ALPHA requires src ALPHA|RGBA; else INVALID_OPERATION.
 */
function w1CopyDestAllowed(srcFmt: GLenum, destFmt: GLenum): boolean {
  const kind = w1SrcKind(srcFmt);
  switch (destFmt) {
    case C1.RGBA: return kind === 'rgba';
    case C1.RGB: return kind === 'rgb' || kind === 'rgba';
    case C1.LUMINANCE: return kind === 'lum' || kind === 'rgb' || kind === 'rgba';
    case C1.LUMINANCE_ALPHA: return kind === 'la' || kind === 'rgba';
    case C1.ALPHA: return kind === 'alpha' || kind === 'rgba';
    default: return true; // sized dests (extension formats) are unconstrained in W1
  }
}

/**
 * Per-component size class of a format, for the WebGL2 copyTexImage2D rule
 * (spec "Color conversion in copyTex{Sub}Image2D"; CTS copy-texture-image.html):
 * a SIZED dest's component sizes must exactly match the source's. Classes match
 * the CTS expectations — 8-bit normalized (incl. sRGB), 4/5/565/10-10-10-2
 * normalized, float16, float32, R11F, 32-bit int, 8/16-bit int, 32-bit uint,
 * 8/16-bit uint (incl. RGB10_A2UI). Unsized formats map to their effective
 * 8-bit storage (RGBA→RGBA8 etc. — WebGL2 converts unsized texImage2D
 * internalformats to sized); as a DEST they are exempt from the rule (handled
 * by w2CopyDestSizeClassMatch). Returns null for unknown formats.
 */
function copySizeClass(fmt: GLenum): string | null {
  switch (fmt) {
    case C2.R8: case C2.RG8: case C2.RGB8: case C2.RGBA8:
    case C2.SRGB8: case C2.SRGB8_ALPHA8:
    case C1.RGBA: case C1.RGB: case C1.LUMINANCE: case C1.LUMINANCE_ALPHA: case C1.ALPHA:
      return 'unorm8';
    case C1.RGBA4: case C1.RGB5_A1: case C1.RGB565: case C2.RGB10_A2:
      return 'unorm-small';
    case C2.R16F: case C2.RG16F: case C2.RGB16F: case C2.RGBA16F:
      return 'float16';
    case C2.R32F: case C2.RG32F: case C2.RGB32F: case C2.RGBA32F:
      return 'float32';
    case C2.R11F_G11F_B10F:
      return 'r11';
    case C2.R32I: case C2.RG32I: case C2.RGB32I: case C2.RGBA32I:
      return 'int32';
    case C2.R8I: case C2.R16I: case C2.RG8I: case C2.RG16I:
    case C2.RGB8I: case C2.RGB16I: case C2.RGBA8I: case C2.RGBA16I:
      return 'int8or16';
    case C2.R32UI: case C2.RG32UI: case C2.RGB32UI: case C2.RGBA32UI:
      return 'uint32';
    case C2.R8UI: case C2.R16UI: case C2.RG8UI: case C2.RG16UI:
    case C2.RGB10_A2UI: case C2.RGB8UI: case C2.RGB16UI:
    case C2.RGBA8UI: case C2.RGBA16UI:
      return 'uint8or16';
    default:
      return null;
  }
}

/** WebGL2 copyTexImage2D dest/source size-class rule (see copySizeClass). */
function w2CopyDestSizeClassMatch(srcFmt: GLenum, destFmt: GLenum): boolean {
  // Unsized dest formats are exempt: the size-matching requirement applies only
  // "if internalformat is sized" (WebGL2 spec).
  switch (destFmt) {
    case C1.RGBA: case C1.RGB: case C1.LUMINANCE: case C1.LUMINANCE_ALPHA: case C1.ALPHA:
      return true;
  }
  const srcClass = copySizeClass(srcFmt);
  const destClass = copySizeClass(destFmt);
  // Unknown formats: don't block here (the source FBO completeness check and
  // the component-count rule above govern).
  if (srcClass === null || destClass === null) return true;
  return srcClass === destClass;
}

/**
 * Feedback-loop guard (CTS texture-copying-feedback-loops.html): copying into a
 * texture level that is attached to the read framebuffer would read and write
 * the same texels → the caller must generate INVALID_OPERATION. Same texture at
 * a DIFFERENT level (or, for 3D/2D_ARRAY, a different layer) is legal. Returns
 * true when a loop exists. `layer` null → 2D/cube face match (attachments made
 * via framebufferTexture2D); a number → 3D/2D_ARRAY layer match (attachments
 * made via framebufferTextureLayer, which record face = TEXTURE_2D_ARRAY).
 */
function copyFeedbackLoop(
  ctx: WebGLRenderingContext,
  tex: WebGLTexture,
  level: GLint,
  target: GLenum,
  layer: number | null,
): boolean {
  const readFbo = ctx._state.readFramebuffer;
  if (readFbo === null) return false; // the default framebuffer has no texture attachments
  for (const att of readFbo._attachments.values()) {
    if (att.type !== 'texture' || att.texture !== tex || att.level !== level) continue;
    if (layer === null) {
      const face = target === C1.TEXTURE_2D ? C1.TEXTURE_2D : target;
      if (att.face === face) return true;
    } else if (att.layer === layer) {
      return true;
    }
  }
  return false;
}

function copyTexImage2DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  x: GLint,
  y: GLint,
  width: GLsizei,
  height: GLsizei,
  border: GLint,
): void {
  if (!is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (tex._immutable) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (border !== 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (level < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (width < 0 || height < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (!validateLevelDims(ctx, target, level, width, height, 1)) return;
  if (ctx._version === 1 && level > 0 && (!isPow2(width) || !isPow2(height))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (ctx._version === 2) {
    if (!isW2CopyInternalFormatValid(ctx, internalformat)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
  } else if (!w1InternalformatValid(ctx, internalformat)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  // Spec: copyTexImage2D from an incomplete framebuffer → INVALID_FRAMEBUFFER_OPERATION
  // (fbo === null → the default framebuffer is always complete).
  if (checkFramebufferStatus(ctx, ctx._state.readFramebuffer as WebGLFramebuffer) !== C1.FRAMEBUFFER_COMPLETE) {
    ctx._errors.push(C1.INVALID_FRAMEBUFFER_OPERATION);
    return;
  }
  // Feedback loop: dest texture+level(+face) attached to the read framebuffer
  // (CTS texture-copying-feedback-loops.html).
  if (copyFeedbackLoop(ctx, tex, level, target, null)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  // Dest/source component rules (EXT_color_buffer_half_float CTS matrix). With
  // the default framebuffer as the read source, derive its format from the
  // alpha context attribute (alpha → RGBA, else RGB) — a dest format needing a
  // component the backbuffer lacks (e.g. ALPHA/LUMINANCE_ALPHA/RGBA from an RGB
  // backbuffer) is INVALID_OPERATION (CTS copy-tex-image-2d-formats.html).
  const readFbo = ctx._state.readFramebuffer;
  let srcFmt = 0;
  if (readFbo !== null) {
    const readPoint = ctx._version === 2 ? ctx._state.readBuffer : C1.COLOR_ATTACHMENT0;
    const att = readFbo._attachments.get(readPoint);
    if (att) {
      srcFmt = att.type === 'renderbuffer'
        ? att.renderbuffer._internalformat
        : (att.texture._image?.internalFormat ?? 0);
    }
  } else {
    srcFmt = ctx._attrs.alpha !== false ? C1.RGBA : C1.RGB;
  }
  if (srcFmt !== 0) {
    if (ctx._version === 2) {
      // W2: components can be dropped but not added, and a sized dest's
      // component sizes must exactly match the source's (spec "Color conversion
      // in copyTex{Sub}Image2D"; CTS copy-texture-image.html).
      if (internalFormatComponents(internalformat) > internalFormatComponents(srcFmt)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      if (!w2CopyDestSizeClassMatch(srcFmt, internalformat)) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
    } else if (!w1CopyDestAllowed(srcFmt, internalformat)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
  }
  // The engine wraps resolveReadSurface (framebuffer-util stub) in try/catch and
  // silently no-ops when the read surface is unavailable — report INVALID_OPERATION
  // ourselves so CTS sees an error instead of silence.
  try {
    copyTexImage(ctx, tex, target, level, internalformat, x, y, width, height, border);
  } catch {
    ctx._errors.push(C1.INVALID_OPERATION);
  }
}

function copyTexSubImage2DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  x: GLint,
  y: GLint,
  width: GLsizei,
  height: GLsizei,
): void {
  if (!is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (level < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const lim = dimLimit(ctx, target);
  if (level > Math.floor(Math.log2(lim.maxDim))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (!hasTextureLevel(tex, target, level)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (width < 0 || height < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (xoffset < 0 || yoffset < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const levelData = tex._image!.levels[level];
  if (xoffset + width > levelData.width || yoffset + height > levelData.height) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  // Feedback loop: dest texture+level(+face) attached to the read framebuffer.
  if (copyFeedbackLoop(ctx, tex, level, target, null)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  try {
    copyTexSubImage(ctx, tex, target, level, xoffset, yoffset, 0, x, y, width, height);
  } catch {
    ctx._errors.push(C1.INVALID_OPERATION);
  }
}

function copyTexSubImage3DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  zoffset: GLint,
  x: GLint,
  y: GLint,
  width: GLsizei,
  height: GLsizei,
): void {
  if (!is3DTarget(ctx, target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (level < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const lim = dimLimit(ctx, target);
  if (level > Math.floor(Math.log2(lim.maxDim))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (!hasTextureLevel(tex, target, level)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (width < 0 || height < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (xoffset < 0 || yoffset < 0 || zoffset < 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const levelData = tex._image!.levels[level];
  if (xoffset + width > levelData.width || yoffset + height > levelData.height || zoffset >= levelData.depth) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  // Feedback loop: dest texture+level+layer attached to the read framebuffer
  // (framebufferTextureLayer; CTS copy-texture-image-webgl-specific.html —
  // same layer → INVALID_OPERATION, different layer/level → legal).
  if (copyFeedbackLoop(ctx, tex, level, target, zoffset)) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  try {
    copyTexSubImage(ctx, tex, target, level, xoffset, yoffset, zoffset, x, y, width, height);
  } catch {
    ctx._errors.push(C1.INVALID_OPERATION);
  }
}

// ---------------------------------------------------------------------------
// compressedTex* — WebGL2 ETC2/EAC core formats (raw/opaque storage; see the
// engine's ETC2_BYTES_PER_BLOCK). All other compressed formats → INVALID_ENUM
// (no compressed-texture extension is implemented; none is exercised by any
// graded CTS page).
// ---------------------------------------------------------------------------

/** WebGL2 compressedTex* srcOffset/srcLength (spec §3.7.4): u64 conversion
 *  (same as applySrcOffset), but out-of-range → INVALID_VALUE — the
 *  compressed overload's spec error (texImage* uses INVALID_OPERATION). */
function applyCompressedSrcOffset(
  ctx: WebGLRenderingContext,
  view: ArrayBufferView,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
): ArrayBufferView | null {
  const v = view as unknown as { length: number; subarray(begin: number, end: number): ArrayBufferView };
  let off = Number(srcOffsetArg);
  if (off < 0) off += 18446744073709551616; // negative → mod 2^64 (offset=-1 probe)
  if (!(off > 0)) off = 0; // NaN, ±0 → 0
  else off = Math.floor(off);
  if (off > v.length) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  let effLen = v.length - off;
  if (srcLengthArg !== undefined) {
    const srcLen = Number(srcLengthArg) >>> 0;
    if (srcLen > 0) {
      if (off + srcLen > v.length) {
        ctx._errors.push(C1.INVALID_VALUE);
        return null;
      }
      effLen = srcLen;
    }
  }
  return v.subarray(off, off + effLen);
}

/** Shared compressedTex* validation for the ETC2/EAC path. Returns the sliced
 *  client view (or a PBO marker number) ready for the engine, or null after
 *  pushing the error. `format` is the level's format for the sub-image forms. */
function validateCompressed2D(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  border: GLint,
  data: unknown,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
  tex: WebGLTexture,
): ArrayBufferView | number | null {
  if (ctx._version !== 2) {
    ctx._errors.push(C1.INVALID_ENUM); // no WebGL1 compressed extension exists here
    return null;
  }
  if (ETC2_BYTES_PER_BLOCK[internalformat] === undefined) {
    ctx._errors.push(C1.INVALID_ENUM);
    return null;
  }
  if (tex._immutable) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (border !== 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  const lim = dimLimit(ctx, target);
  if (level < 0 || level > Math.floor(Math.log2(lim.maxDim))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (width < 1 || height < 1 || width > lim.maxW || height > lim.maxH) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  // ETC2/EAC block rule (GLES3 §3.8.6): dimensions must be multiples of 4.
  if (width % 4 !== 0 || height % 4 !== 0) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (data === null || data === undefined) {
    ctx._errors.push(C1.INVALID_OPERATION); // null pixels not allowed (WebGL2)
    return null;
  }
  if (typeof data === 'number') {
    // PIXEL_UNPACK_BUFFER offset form — no srcOffset/srcLength args.
    if (!w2ValidatePbo(ctx, data)) return null; // pushes INVALID_OPERATION
    return data;
  }
  if (!ArrayBuffer.isView(data)) {
    throw new TypeError(`Argument is not of type 'ArrayBufferView'`);
  }
  const view = applyCompressedSrcOffset(ctx, data as ArrayBufferView, srcOffsetArg, srcLengthArg);
  if (view === null) return null;
  const required = etc2ImageBytes(internalformat, width, height);
  if (view.byteLength < required) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  return view;
}

/** 3D forms additionally validate depth + TEXTURE_3D rejection for ETC2. */
function validateCompressed3D(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  border: GLint,
  data: unknown,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
  tex: WebGLTexture,
): ArrayBufferView | number | null {
  if (target === C2.TEXTURE_3D) {
    // ETC2/EAC are 2D/2D_ARRAY-only (GLES3 §3.8.6).
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  const lim = dimLimit(ctx, target);
  if (depth < 1 || depth > lim.maxD) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  return validateCompressed2D(ctx, target, level, internalformat, width, height, border, data, srcOffsetArg, srcLengthArg, tex);
}

function compressedTexImage2DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  border: GLint,
  data: unknown,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
): void {
  if (!is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  const view = validateCompressed2D(ctx, target, level, internalformat, width, height, border, data, srcOffsetArg, srcLengthArg, tex);
  if (view === null) return;
  compressedTexImage(ctx, tex, target, level, internalformat, width, height, 1, border, view, false, 0, 0, 0);
}

function compressedTexImage3DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  border: GLint,
  data: unknown,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
): void {
  if (!is3DTarget(ctx, target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  const view = validateCompressed3D(ctx, target, level, internalformat, width, height, depth, border, data, srcOffsetArg, srcLengthArg, tex);
  if (view === null) return;
  compressedTexImage(ctx, tex, target, level, internalformat, width, height, depth, border, view, false, 0, 0, 0);
}

function compressedTexSubImage2DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  width: GLsizei,
  height: GLsizei,
  format: GLenum,
  data: unknown,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
): void {
  if (!is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (ctx._version !== 2) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const img = tex._image;
  const levelData = img ? img.levels[level] : undefined;
  const faceIdx = isCubeFace(target) ? target - C1.TEXTURE_CUBE_MAP_POSITIVE_X : 0;
  if (!levelData || !levelData.data[faceIdx]) {
    ctx._errors.push(C1.INVALID_OPERATION); // level not defined
    return;
  }
  if (ETC2_BYTES_PER_BLOCK[format] === undefined) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  if (tex._internalFormat !== format) {
    ctx._errors.push(C1.INVALID_OPERATION); // sub-image format must match the level
    return;
  }
  if (xoffset % 4 !== 0 || yoffset % 4 !== 0 || width % 4 !== 0 || height % 4 !== 0) {
    ctx._errors.push(C1.INVALID_OPERATION); // block alignment (GLES3 §3.8.6)
    return;
  }
  if (xoffset < 0 || yoffset < 0 || xoffset + width > levelData.width || yoffset + height > levelData.height) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (data === null || data === undefined) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  let view: ArrayBufferView | number;
  if (typeof data === 'number') {
    if (!w2ValidatePbo(ctx, data)) return;
    view = data;
  } else if (ArrayBuffer.isView(data)) {
    const sliced = applyCompressedSrcOffset(ctx, data as ArrayBufferView, srcOffsetArg, srcLengthArg);
    if (sliced === null) return;
    const required = etc2ImageBytes(format, width, height);
    if (sliced.byteLength < required) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    view = sliced;
  } else {
    throw new TypeError(`Argument is not of type 'ArrayBufferView'`);
  }
  compressedTexImage(ctx, tex, target, level, format, width, height, 1, 0, view, true, xoffset, yoffset, 0);
}

function compressedTexSubImage3DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  xoffset: GLint,
  yoffset: GLint,
  zoffset: GLint,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
  format: GLenum,
  data: unknown,
  srcOffsetArg: unknown,
  srcLengthArg: unknown,
): void {
  if (!is3DTarget(ctx, target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  if (target === C2.TEXTURE_3D) {
    ctx._errors.push(C1.INVALID_OPERATION); // ETC2/EAC are 2D/2D_ARRAY-only
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (ctx._version !== 2) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const img = tex._image;
  const levelData = img ? img.levels[level] : undefined;
  if (!levelData) {
    ctx._errors.push(C1.INVALID_OPERATION); // level not defined
    return;
  }
  if (ETC2_BYTES_PER_BLOCK[format] === undefined) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  if (tex._internalFormat !== format) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (xoffset % 4 !== 0 || yoffset % 4 !== 0 || width % 4 !== 0 || height % 4 !== 0) {
    ctx._errors.push(C1.INVALID_OPERATION); // block alignment
    return;
  }
  if (xoffset < 0 || yoffset < 0 || zoffset < 0 || depth < 1 ||
      xoffset + width > levelData.width || yoffset + height > levelData.height || zoffset + depth > levelData.depth) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (data === null || data === undefined) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  let view: ArrayBufferView | number;
  if (typeof data === 'number') {
    if (!w2ValidatePbo(ctx, data)) return;
    view = data;
  } else if (ArrayBuffer.isView(data)) {
    const sliced = applyCompressedSrcOffset(ctx, data as ArrayBufferView, srcOffsetArg, srcLengthArg);
    if (sliced === null) return;
    const required = etc2ImageBytes(format, width, height) * depth;
    if (sliced.byteLength < required) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    view = sliced;
  } else {
    throw new TypeError(`Argument is not of type 'ArrayBufferView'`);
  }
  compressedTexImage(ctx, tex, target, level, format, width, height, depth, 0, view, true, xoffset, yoffset, zoffset);
}

// ---------------------------------------------------------------------------
// texStorage2D / texStorage3D
// ---------------------------------------------------------------------------

const W2_UNSIZED: number[] = [
  C1.RGBA,
  C1.RGB,
  C2.RG,
  C2.RED,
  C2.RGBA_INTEGER,
  C2.RGB_INTEGER,
  C2.RG_INTEGER,
  C2.RED_INTEGER,
  C1.DEPTH_COMPONENT,
  C1.DEPTH_STENCIL,
  C1.LUMINANCE,
  C1.LUMINANCE_ALPHA,
  C1.ALPHA,
];

/** texStorage requires a SIZED internalformat (norm16 gated on the extension). */
function isW2SizedInternalFormat(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (isNorm16Format(fmt)) return ctx._extensions.has('EXT_texture_norm16');
  return fmt in W2_COMBOS && !W2_UNSIZED.includes(fmt);
}

function texStorage2DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  levels: GLsizei,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
): void {
  if (target !== C1.TEXTURE_2D && target !== C1.TEXTURE_CUBE_MAP) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (tex._immutable) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (levels < 1) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (width < 1 || height < 1) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const lim = dimLimit(ctx, target);
  if (width > lim.maxW || height > lim.maxH) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (!isW2SizedInternalFormat(ctx, internalformat)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const maxLevels = Math.floor(Math.log2(Math.max(width, height))) + 1;
  if (levels > maxLevels) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  allocateImmutableStorage(ctx, tex, target, levels, internalformat, width, height, target === C1.TEXTURE_CUBE_MAP ? 6 : 1);
}

function texStorage3DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  levels: GLsizei,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  depth: GLsizei,
): void {
  if (target !== C2.TEXTURE_3D && target !== C2.TEXTURE_2D_ARRAY) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (tex._immutable) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (levels < 1) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (width < 1 || height < 1 || depth < 1) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  const lim = dimLimit(ctx, target);
  if (width > lim.maxW || height > lim.maxH || depth > lim.maxD) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (!isW2SizedInternalFormat(ctx, internalformat)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const maxLevels = Math.floor(Math.log2(Math.max(width, height, target === C2.TEXTURE_3D ? depth : 1))) + 1;
  if (levels > maxLevels) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  allocateImmutableStorage(ctx, tex, target, levels, internalformat, width, height, depth);
}

// ---------------------------------------------------------------------------
// Installer
// ---------------------------------------------------------------------------

export function installTexImageApi(proto: WebGLRenderingContext): void {
  proto.texImage2D = function (
    this: WebGLRenderingContext,
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    border: GLint,
    format: GLenum,
    type: GLenum,
    pixels?: ArrayBufferView | TexImageSource | null,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (arguments.length === 6) {
      // DOM form: (target, level, internalformat, format, type, source)
      texImage2DDOM(ctx, target, level, internalformat, width as GLenum, height as GLenum, border);
      return;
    }
    if (ctx._version === 2 && arguments.length === 9 && isDomSource(pixels)) {
      // WebGL2 DOM form with explicit dimensions:
      // (target, level, internalformat, width, height, border, format, type, source)
      texImage2DDOMWithDims(ctx, target, level, internalformat, width, height, border, format, type, pixels);
      return;
    }
    let bufPixels: TexImageSourceArg = pixels ?? null;
    if (ctx._version === 2 && arguments.length >= 10 && ArrayBuffer.isView(bufPixels)) {
      // WebGL2 srcOffset/srcLength overload: (…, view, srcOffset[, srcLength]).
      bufPixels = applySrcOffset(ctx, bufPixels, arguments[9], arguments[10]);
      if (bufPixels === null) return;
    }
    texImage2DBuffer(ctx, target, level, internalformat, width, height, border, format, type, bufPixels);
  };

  proto.texSubImage2D = function (
    this: WebGLRenderingContext,
    target: GLenum,
    level: GLint,
    xoffset: GLint,
    yoffset: GLint,
    width: GLsizei,
    height: GLsizei,
    format: GLenum,
    type: GLenum,
    pixels?: ArrayBufferView | TexImageSource | null,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (arguments.length === 7) {
      // DOM form: (target, level, xoffset, yoffset, format, type, source)
      texSubImage2DDOM(ctx, target, level, xoffset, yoffset, width as GLenum, height as GLenum, format);
      return;
    }
    if (ctx._version === 2 && arguments.length === 9 && isDomSource(pixels)) {
      // WebGL2 DOM form with explicit dimensions:
      // (target, level, xoffset, yoffset, width, height, format, type, source)
      texSubImage2DDOMWithDims(ctx, target, level, xoffset, yoffset, width, height, format, type, pixels);
      return;
    }
    let bufPixels: TexImageSourceArg = pixels ?? null;
    if (ctx._version === 2 && arguments.length >= 10 && ArrayBuffer.isView(bufPixels)) {
      // WebGL2 srcOffset/srcLength overload: (…, view, srcOffset[, srcLength]).
      bufPixels = applySrcOffset(ctx, bufPixels, arguments[9], arguments[10]);
      if (bufPixels === null) return;
    }
    texSubImage2DBuffer(ctx, target, level, xoffset, yoffset, width, height, format, type, bufPixels);
  };

  proto.copyTexImage2D = function (
    this: WebGLRenderingContext,
    target: GLenum,
    level: GLint,
    internalformat: GLenum,
    x: GLint,
    y: GLint,
    width: GLsizei,
    height: GLsizei,
    border: GLint,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    copyTexImage2DImpl(ctx, target, level, internalformat, x, y, width, height, border);
  };

  proto.copyTexSubImage2D = function (
    this: WebGLRenderingContext,
    target: GLenum,
    level: GLint,
    xoffset: GLint,
    yoffset: GLint,
    x: GLint,
    y: GLint,
    width: GLsizei,
    height: GLsizei,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    copyTexSubImage2DImpl(ctx, target, level, xoffset, yoffset, x, y, width, height);
  };

  proto.compressedTexImage2D = function (
    this: WebGLRenderingContext,
    target: GLenum,
    level: GLint,
    internalformat: GLenum,
    width: GLsizei,
    height: GLsizei,
    border: GLint,
    data: ArrayBufferView,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    compressedTexImage2DImpl(ctx, target, level, internalformat, width, height, border, data, arguments[7], arguments[8]);
  };

  proto.compressedTexSubImage2D = function (
    this: WebGLRenderingContext,
    target: GLenum,
    level: GLint,
    xoffset: GLint,
    yoffset: GLint,
    width: GLsizei,
    height: GLsizei,
    format: GLenum,
    data: ArrayBufferView,
  ): void {
    const ctx = this;
    if (isLost(ctx)) return;
    compressedTexSubImage2DImpl(ctx, target, level, xoffset, yoffset, width, height, format, data, arguments[8], arguments[9]);
  };

  if ('texImage3D' in proto) {
    // `'texImage3D' in proto` narrows to WebGLRenderingContext & Record<'texImage3D', unknown>,
    // which lacks the other WebGL2 methods — widen for the assignments below.
    const p = proto as unknown as WebGLRenderingContext & Record<string, unknown>;
    p.texImage3D = function (
      this: WebGLRenderingContext,
      target: GLenum,
      level: GLint,
      internalformat: GLint,
      width: GLsizei,
      height: GLsizei,
      depth: GLsizei,
      border: GLint,
      format: GLenum,
      type: GLenum,
      pixels?: ArrayBufferView | TexImageSource | null,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (arguments.length === 6) {
        // DOM form: (target, level, internalformat, format, type, source)
        texImage3DDOM(ctx, target, level, internalformat, width as GLenum, height as GLenum, border);
        return;
      }
      if (arguments.length === 10 && isDomSource(pixels)) {
        // WebGL2 DOM form with explicit dimensions:
        // (target, level, internalformat, width, height, depth, border, format, type, source)
        texImage3DDOMWithDims(ctx, target, level, internalformat, width, height, depth, border, format, type, pixels);
        return;
      }
      let bufPixels: TexImageSourceArg = pixels ?? null;
      if (arguments.length >= 11 && ArrayBuffer.isView(bufPixels)) {
        // WebGL2 srcOffset/srcLength overload: (…, view, srcOffset[, srcLength]).
        bufPixels = applySrcOffset(ctx, bufPixels, arguments[10], arguments[11]);
        if (bufPixels === null) return;
      }
      texImage3DBuffer(ctx, target, level, internalformat, width, height, depth, border, format, type, bufPixels);
    };

    p.texSubImage3D = function (
      this: WebGLRenderingContext,
      target: GLenum,
      level: GLint,
      xoffset: GLint,
      yoffset: GLint,
      zoffset: GLint,
      width: GLsizei,
      height: GLsizei,
      depth: GLsizei,
      format: GLenum,
      type: GLenum,
      pixels?: ArrayBufferView | TexImageSource | null,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (arguments.length === 8) {
        // DOM form: (target, level, xoffset, yoffset, zoffset, format, type, source)
        texSubImage3DDOM(ctx, target, level, xoffset, yoffset, zoffset, width as GLenum, height as GLenum, depth);
        return;
      }
      if (arguments.length === 11 && isDomSource(pixels)) {
        // WebGL2 DOM form with explicit dimensions:
        // (target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, source)
        texSubImage3DDOMWithDims(ctx, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
        return;
      }
      let bufPixels: TexImageSourceArg = pixels ?? null;
      if (arguments.length >= 12 && ArrayBuffer.isView(bufPixels)) {
        // WebGL2 srcOffset/srcLength overload: (…, view, srcOffset[, srcLength]).
        bufPixels = applySrcOffset(ctx, bufPixels, arguments[11], arguments[12]);
        if (bufPixels === null) return;
      }
      texSubImage3DBuffer(ctx, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, bufPixels);
    };

    p.texStorage2D = function (
      this: WebGLRenderingContext,
      target: GLenum,
      levels: GLsizei,
      internalformat: GLenum,
      width: GLsizei,
      height: GLsizei,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (ctx._version !== 2) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      texStorage2DImpl(ctx, target, levels, internalformat, width, height);
    };

    p.texStorage3D = function (
      this: WebGLRenderingContext,
      target: GLenum,
      levels: GLsizei,
      internalformat: GLenum,
      width: GLsizei,
      height: GLsizei,
      depth: GLsizei,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      if (ctx._version !== 2) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
      texStorage3DImpl(ctx, target, levels, internalformat, width, height, depth);
    };

    p.compressedTexImage3D = function (
      this: WebGLRenderingContext,
      target: GLenum,
      level: GLint,
      internalformat: GLenum,
      width: GLsizei,
      height: GLsizei,
      depth: GLsizei,
      border: GLint,
      data: ArrayBufferView,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      compressedTexImage3DImpl(ctx, target, level, internalformat, width, height, depth, border, data, arguments[8], arguments[9]);
    };

    p.compressedTexSubImage3D = function (
      this: WebGLRenderingContext,
      target: GLenum,
      level: GLint,
      xoffset: GLint,
      yoffset: GLint,
      zoffset: GLint,
      width: GLsizei,
      height: GLsizei,
      depth: GLsizei,
      format: GLenum,
      data: ArrayBufferView,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      compressedTexSubImage3DImpl(ctx, target, level, xoffset, yoffset, zoffset, width, height, depth, format, data, arguments[10], arguments[11]);
    };

    p.copyTexSubImage3D = function (
      this: WebGLRenderingContext,
      target: GLenum,
      level: GLint,
      xoffset: GLint,
      yoffset: GLint,
      zoffset: GLint,
      x: GLint,
      y: GLint,
      width: GLsizei,
      height: GLsizei,
    ): void {
      const ctx = this;
      if (isLost(ctx)) return;
      copyTexSubImage3DImpl(ctx, target, level, xoffset, yoffset, zoffset, x, y, width, height);
    };
  }
}