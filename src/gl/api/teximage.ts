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
 *  - texImage3D: 6 args → DOM form (rejected: DOM sources are 2D-only, spec);
 *    else the 10-arg buffer form.
 *  - texSubImage2D: 7 args → DOM form; else 9-arg buffer form.
 *  - texSubImage3D: 8 args → DOM form (rejected, 2D-only rule); else 11-arg.
 *  - WebGL2: buffer-form pixels may be a NUMBER (byte offset into
 *    PIXEL_UNPACK_BUFFER — must be bound; flipY/premultiplyAlpha must be off;
 *    offset+required ≤ buffer size).
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
import type { WebGLTexture } from '../objects';
import type { GLenum, GLint, GLsizei, TexImageSource } from '../types';
import {
  allocateImmutableStorage,
  compressedTexImage,
  copyTexImage,
  copyTexSubImage,
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

/** Context-loss guard: no-op + one CONTEXT_LOST_WEBGL per call. */
function isLost(ctx: WebGLRenderingContext): boolean {
  if (ctx._isLost) ctx._errors.push(C1.CONTEXT_LOST_WEBGL);
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
  if (isCubeFace(target)) return unit.textureCube as unknown as WebGLTexture | null;
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
  if (isCubeFace(target)) return { maxW: lim.MAX_CUBE_MAP_TEXTURE_SIZE, maxH: lim.MAX_CUBE_MAP_TEXTURE_SIZE, maxD: 1, maxDim: lim.MAX_CUBE_MAP_TEXTURE_SIZE };
  if (target === C2.TEXTURE_3D) return { maxW: lim.MAX_3D_TEXTURE_SIZE, maxH: lim.MAX_3D_TEXTURE_SIZE, maxD: lim.MAX_3D_TEXTURE_SIZE, maxDim: lim.MAX_3D_TEXTURE_SIZE };
  if (target === C2.TEXTURE_2D_ARRAY) return { maxW: lim.MAX_TEXTURE_SIZE, maxH: lim.MAX_TEXTURE_SIZE, maxD: lim.MAX_ARRAY_TEXTURE_LAYERS, maxDim: lim.MAX_TEXTURE_SIZE };
  return { maxW: lim.MAX_TEXTURE_SIZE, maxH: lim.MAX_TEXTURE_SIZE, maxD: 1, maxDim: lim.MAX_TEXTURE_SIZE };
}

const isPow2 = (v: number): boolean => v > 0 && (v & (v - 1)) === 0;

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
  const lim = dimLimit(ctx, target);
  if (width > lim.maxW || height > lim.maxH || depth > lim.maxD) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  if (level > Math.floor(Math.log2(lim.maxDim))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
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
  [SRGB_EXT]: { format: C1.RGB, types: [C1.UNSIGNED_BYTE] },
  [SRGB_ALPHA_EXT]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE] },
};

function w1InternalformatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (W1_INTERNALFORMATS.includes(fmt)) return true;
  switch (fmt) {
    case C1.DEPTH_COMPONENT:
    case C1.DEPTH_STENCIL:
      return ctx.getExtension('WEBGL_depth_texture') !== null;
    case 0x881b:
    case 0x881a:
      return ctx.getExtension('OES_texture_half_float') !== null;
    case 0x8815:
    case 0x8814:
      return ctx.getExtension('OES_texture_float') !== null;
    case SRGB_EXT:
    case SRGB_ALPHA_EXT:
      return ctx.getExtension('EXT_sRGB') !== null;
    default:
      return false;
  }
}

function w1FormatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (W1_FORMATS.includes(fmt)) return true;
  if (fmt === C1.DEPTH_COMPONENT || fmt === C1.DEPTH_STENCIL) {
    return ctx.getExtension('WEBGL_depth_texture') !== null;
  }
  return false;
}

function w1TypeValid(ctx: WebGLRenderingContext, type: GLenum): boolean {
  if (W1_TYPES.includes(type)) return true;
  switch (type) {
    case C1.FLOAT:
      return ctx.getExtension('OES_texture_float') !== null;
    case HALF_FLOAT_OES:
      return ctx.getExtension('OES_texture_half_float') !== null;
    case C1.UNSIGNED_INT:
    case C1.UNSIGNED_SHORT:
    case UNSIGNED_INT_24_8_WEBGL:
      return ctx.getExtension('WEBGL_depth_texture') !== null;
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
  if (!combo.types.includes(type)) {
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
  [C1.RGB5_A1]: { format: C1.RGBA, types: [C1.UNSIGNED_BYTE, C1.UNSIGNED_SHORT_5_5_5_1] },
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

/** W2 texImage internalformat validity (norm16 gated on EXT_texture_norm16). */
function w2InternalformatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (isNorm16Format(fmt)) return ctx.getExtension('EXT_texture_norm16') !== null;
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
 * error and returns false when the source is too small.
 */
function validatePixelsSize(
  ctx: WebGLRenderingContext,
  pixels: ArrayBufferView | number,
  width: number,
  height: number,
  depth: number,
  format: GLenum,
  type: GLenum,
): boolean {
  const unpack = ctx._state.pixelStore.unpack;
  const srcBpp = bytesPerTexel(ctx, format, type);
  const rowLength = unpack.rowLength > 0 ? unpack.rowLength : width;
  const rowBytes = align(rowLength * srcBpp, unpack.alignment);
  const imageHeight = unpack.imageHeight > 0 ? unpack.imageHeight : height;
  const required =
    (unpack.skipRows + (unpack.skipImages + depth) * imageHeight) * rowBytes +
    unpack.skipPixels * srcBpp;
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

/** WebGL2 PBO (pixels = byte offset) pre-checks: bound, flipY/premultiply, offset. */
function w2ValidatePbo(ctx: WebGLRenderingContext, pixels: number): boolean {
  const buf = ctx._state.pixelUnpackBuffer;
  if (buf === null) {
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

/** Width/height of a DOM TexImageSource (numbers only — mocks/ImageData included). */
function sourceDims(source: unknown): { width: number; height: number } | null {
  if (source !== null && typeof source === 'object') {
    const s = source as { width?: unknown; height?: unknown };
    if (typeof s.width === 'number' && typeof s.height === 'number') {
      return { width: s.width, height: s.height };
    }
  }
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
    if (!(internalformat in W2_DOM)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const entry = W2_DOM[internalformat];
    if (entry.format !== format) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (!entry.types.includes(type)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
  } else if (!w1ValidateFormatType(ctx, internalformat, format, type, null)) {
    return;
  }
  uploadTexImage(
    ctx, tex, target, level, internalformat, dims.width, dims.height, 1, 0, format, type,
    source as unknown as TexImageSourceArg, source,
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
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type)) return;
    } else if (ArrayBuffer.isView(pixels)) {
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type)) return;
    }
  } else {
    if (!w1ValidateFormatType(ctx, internalformat, format, type, pixels)) return;
    if (ArrayBuffer.isView(pixels)) {
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type)) return;
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
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type)) return;
  } else if (ArrayBuffer.isView(pixels)) {
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type)) return;
  }
  uploadTexImage(ctx, tex, target, level, internalformat, width, height, depth, border, format, type, pixels);
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
    ctx._errors.push(C1.INVALID_VALUE);
    return;
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
  uploadTexSubImage(
    ctx, tex, target, level, xoffset, yoffset, 0, dims.width, dims.height, 1, format, type,
    source as unknown as TexImageSourceArg, source,
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
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type)) return;
    } else if (ArrayBuffer.isView(pixels)) {
      if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type)) return;
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
    if (!validatePixelsSize(ctx, pixels, width, height, 1, format, type)) return;
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
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type)) return;
  } else if (ArrayBuffer.isView(pixels)) {
    if (!validatePixelsSize(ctx, pixels, width, height, depth, format, type)) return;
  } else {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  uploadTexSubImage(ctx, tex, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
}

// ---------------------------------------------------------------------------
// copyTexImage2D / copyTexSubImage2D / copyTexSubImage3D
// ---------------------------------------------------------------------------

/** WebGL2 copyTexImage2D internalformats (sized color-renderable; 32F gated). */
const W2_COPY_INTERNALFORMATS: number[] = [
  C2.R8,
  C2.RG8,
  C2.RGB8,
  C2.RGBA8,
  C1.RGBA4,
  C1.RGB5_A1,
  C1.RGB565,
  C2.RGB10_A2,
  C2.SRGB8_ALPHA8,
  C2.R16F,
  C2.RG16F,
  C2.RGBA16F,
  C2.R11F_G11F_B10F,
  C2.R8I, C2.R8UI, C2.R16I, C2.R16UI, C2.R32I, C2.R32UI,
  C2.RG8I, C2.RG8UI, C2.RG16I, C2.RG16UI, C2.RG32I, C2.RG32UI,
  C2.RGB8I, C2.RGB8UI, C2.RGB16I, C2.RGB16UI, C2.RGB32I, C2.RGB32UI,
  C2.RGBA8I, C2.RGBA8UI, C2.RGBA16I, C2.RGBA16UI, C2.RGBA32I, C2.RGBA32UI,
];

function isW2CopyInternalFormatValid(ctx: WebGLRenderingContext, fmt: GLenum): boolean {
  if (W2_COPY_INTERNALFORMATS.includes(fmt)) return true;
  if (fmt === C2.R32F || fmt === C2.RG32F || fmt === C2.RGBA32F) {
    return ctx.getExtension('EXT_color_buffer_float') !== null;
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
  const lim = dimLimit(ctx, target);
  if (width > lim.maxW || height > lim.maxH) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
  if (level > Math.floor(Math.log2(lim.maxDim))) {
    ctx._errors.push(C1.INVALID_VALUE);
    return;
  }
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
  try {
    copyTexSubImage(ctx, tex, target, level, xoffset, yoffset, zoffset, x, y, width, height);
  } catch {
    ctx._errors.push(C1.INVALID_OPERATION);
  }
}

// ---------------------------------------------------------------------------
// compressedTex* (no compressed formats implemented — unconditional INVALID_ENUM)
// ---------------------------------------------------------------------------

function compressedTexImage2DImpl(
  ctx: WebGLRenderingContext,
  target: GLenum,
  level: GLint,
  internalformat: GLenum,
  width: GLsizei,
  height: GLsizei,
  border: GLint,
  data: ArrayBufferView,
): void {
  void level; void internalformat; void width; void height; void border; void data;
  if (!is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  ctx._errors.push(C1.INVALID_ENUM); // no compressed format is implemented
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
  data: ArrayBufferView,
): void {
  void level; void internalformat; void width; void height; void depth; void border; void data;
  if (!is3DTarget(ctx, target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  ctx._errors.push(C1.INVALID_ENUM); // no compressed format is implemented
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
  data: ArrayBufferView,
): void {
  void level; void xoffset; void yoffset; void width; void height; void format; void data;
  if (!is2DTarget(target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  ctx._errors.push(C1.INVALID_ENUM); // no compressed format is implemented
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
  data: ArrayBufferView,
): void {
  void level; void xoffset; void yoffset; void zoffset; void width; void height; void depth; void format; void data;
  if (!is3DTarget(ctx, target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  ctx._errors.push(C1.INVALID_ENUM); // no compressed format is implemented
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
  if (isNorm16Format(fmt)) return ctx.getExtension('EXT_texture_norm16') !== null;
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
    texImage2DBuffer(ctx, target, level, internalformat, width, height, border, format, type, pixels ?? null);
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
    texSubImage2DBuffer(ctx, target, level, xoffset, yoffset, width, height, format, type, pixels ?? null);
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
    compressedTexImage2DImpl(ctx, target, level, internalformat, width, height, border, data);
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
    compressedTexSubImage2DImpl(ctx, target, level, xoffset, yoffset, width, height, format, data);
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
      texImage3DBuffer(ctx, target, level, internalformat, width, height, depth, border, format, type, pixels ?? null);
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
      texSubImage3DBuffer(ctx, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels ?? null);
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
      compressedTexImage3DImpl(ctx, target, level, internalformat, width, height, depth, border, data);
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
      compressedTexSubImage3DImpl(ctx, target, level, xoffset, yoffset, zoffset, width, height, depth, format, data);
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
