/**
 * gl-enums.ts — GLenum constants interpreted by the raster module.
 *
 * Raster is a leaf module: it imports nothing from `gl/` or `glsl/`, so the GL
 * constant values it switches on live here. The values are spec-fixed
 * (WebGL 1.0/2.0 + OpenGL ES 2.0/3.0); `gl/` keeps its own (larger) constant
 * table — duplicate definitions cannot drift because the values are defined by
 * the spec, not by us.
 *
 * Only constants that raster actually INTERPRETS are listed (draw modes, test
 * functions, blend factors, texture filters/wraps, pixel formats/types,
 * internal formats, cube faces, attachment points).
 */

export type GLenum = number;

/* ------------------------------------------------------------------ */
/* Primitive modes (DrawCall.mode)                                     */
/* ------------------------------------------------------------------ */
export const POINTS = 0x0000;
export const LINES = 0x0001;
export const LINE_LOOP = 0x0002;
export const LINE_STRIP = 0x0003;
export const TRIANGLES = 0x0004;
export const TRIANGLE_STRIP = 0x0005;
export const TRIANGLE_FAN = 0x0006;

/* ------------------------------------------------------------------ */
/* Cull state                                                          */
/* ------------------------------------------------------------------ */
export const FRONT = 0x0404;
export const BACK = 0x0405;
export const FRONT_AND_BACK = 0x0408;
export const CW = 0x0900;
export const CCW = 0x0901;

/* ------------------------------------------------------------------ */
/* Depth / stencil test functions                                      */
/* ------------------------------------------------------------------ */
export const NEVER = 0x0200;
export const LESS = 0x0201;
export const EQUAL = 0x0202;
export const LEQUAL = 0x0203;
export const GREATER = 0x0204;
export const NOTEQUAL = 0x0205;
export const GEQUAL = 0x0206;
export const ALWAYS = 0x0207;

/* ------------------------------------------------------------------ */
/* Stencil operations                                                  */
/* ------------------------------------------------------------------ */
export const KEEP = 0x1e00;
export const REPLACE = 0x1e01;
export const INCR = 0x1e02;
export const DECR = 0x1e03;
export const INVERT = 0x150a;
export const INCR_WRAP = 0x8507;
export const DECR_WRAP = 0x8508;

/* ------------------------------------------------------------------ */
/* Blend factors                                                       */
/* ------------------------------------------------------------------ */
export const ZERO = 0;
export const ONE = 1;
export const SRC_COLOR = 0x0300;
export const ONE_MINUS_SRC_COLOR = 0x0301;
export const SRC_ALPHA = 0x0302;
export const ONE_MINUS_SRC_ALPHA = 0x0303;
export const DST_ALPHA = 0x0304;
export const ONE_MINUS_DST_ALPHA = 0x0305;
export const DST_COLOR = 0x0306;
export const ONE_MINUS_DST_COLOR = 0x0307;
export const SRC_ALPHA_SATURATE = 0x0308;
export const CONSTANT_COLOR = 0x8001;
export const ONE_MINUS_CONSTANT_COLOR = 0x8002;
export const CONSTANT_ALPHA = 0x8003;
export const ONE_MINUS_CONSTANT_ALPHA = 0x8004;
// Dual-source blending (WEBGL_blend_func_extended, GLES 3.0 table 4.2): the
// SRC1_* factors read the SECONDARY fragment color (output index 1).
export const SRC1_COLOR = 0x88f9;
export const ONE_MINUS_SRC1_COLOR = 0x88fa;
export const SRC1_ALPHA = 0x8589;
export const ONE_MINUS_SRC1_ALPHA = 0x88fb;

/* ------------------------------------------------------------------ */
/* Blend equations                                                     */
/* ------------------------------------------------------------------ */
export const FUNC_ADD = 0x8006;
export const MIN = 0x8007;
export const MAX = 0x8008;
export const FUNC_SUBTRACT = 0x800a;
export const FUNC_REVERSE_SUBTRACT = 0x800b;

/* ------------------------------------------------------------------ */
/* Texture filtering / wrap / compare                                  */
/* ------------------------------------------------------------------ */
export const NEAREST = 0x2600;
export const LINEAR = 0x2601;
export const NEAREST_MIPMAP_NEAREST = 0x2700;
export const LINEAR_MIPMAP_NEAREST = 0x2701;
export const NEAREST_MIPMAP_LINEAR = 0x2702;
export const LINEAR_MIPMAP_LINEAR = 0x2703;
export const REPEAT = 0x2901;
export const CLAMP_TO_EDGE = 0x812f;
export const MIRRORED_REPEAT = 0x8370;
export const NONE = 0;
export const COMPARE_REF_TO_TEXTURE = 0x884e;
export const TEXTURE_MAX_ANISOTROPY_EXT = 0x84fe;

/* ------------------------------------------------------------------ */
/* Texture targets                                                     */
/* ------------------------------------------------------------------ */
export const TEXTURE_2D = 0x0de1;
export const TEXTURE_3D = 0x806f;
export const TEXTURE_CUBE_MAP = 0x8513;
export const TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515;
export const TEXTURE_CUBE_MAP_NEGATIVE_X = 0x8516;
export const TEXTURE_CUBE_MAP_POSITIVE_Y = 0x8517;
export const TEXTURE_CUBE_MAP_NEGATIVE_Y = 0x8518;
export const TEXTURE_CUBE_MAP_POSITIVE_Z = 0x8519;
export const TEXTURE_CUBE_MAP_NEGATIVE_Z = 0x851a;
export const TEXTURE_2D_ARRAY = 0x8c1a;

/* ------------------------------------------------------------------ */
/* Source pixel formats (texImage2D / readPixels)                      */
/* ------------------------------------------------------------------ */
export const ALPHA = 0x1906;
export const RGB = 0x1907;
export const RGBA = 0x1908;
export const LUMINANCE = 0x1909;
export const LUMINANCE_ALPHA = 0x190a;
export const RED = 0x1903;
export const RG = 0x8227;
export const DEPTH_COMPONENT = 0x1902;
export const DEPTH_STENCIL = 0x84f9;
export const STENCIL_INDEX = 0x1901;
export const BGRA = 0x80e1; // WebGL1 (ImageData sources) only

/* ------------------------------------------------------------------ */
/* Source pixel types (texImage2D / readPixels)                        */
/* ------------------------------------------------------------------ */
export const BYTE = 0x1400;
export const UNSIGNED_BYTE = 0x1401;
export const SHORT = 0x1402;
export const UNSIGNED_SHORT = 0x1403;
export const INT = 0x1404;
export const UNSIGNED_INT = 0x1405;
export const FLOAT = 0x1406;
export const HALF_FLOAT = 0x140b;
export const HALF_FLOAT_OES = 0x8d61;
export const UNSIGNED_SHORT_5_6_5 = 0x8363;
export const UNSIGNED_SHORT_4_4_4_4 = 0x8033;
export const UNSIGNED_SHORT_5_5_5_1 = 0x8034;
export const UNSIGNED_INT_2_10_10_10_REV = 0x8368;
export const UNSIGNED_INT_10F_11F_11F_REV = 0x8c3b;
export const UNSIGNED_INT_5_9_9_9_REV = 0x8c3e;
export const FLOAT_32_UNSIGNED_INT_24_8_REV = 0x8fad;
export const UNSIGNED_INT_24_8 = 0x84fa;

/* ------------------------------------------------------------------ */
/* Sized internal formats (WebGL2 textures/renderbuffers;              */
/* also the storage formats chosen for WebGL1 unsized formats)         */
/* ------------------------------------------------------------------ */
export const R8 = 0x8229;
export const R8_SNORM = 0x8f94;
export const R16F = 0x822d;
export const R32F = 0x822e;
export const R8UI = 0x8232;
export const R8I = 0x8231;
export const R16UI = 0x8234;
export const R16I = 0x8233;
export const R32UI = 0x8236;
export const R32I = 0x8235;

export const RG8 = 0x822b;
export const RG8_SNORM = 0x8f95;
export const RG16F = 0x822f;
export const RG32F = 0x8230;
export const RG8UI = 0x8238;
export const RG8I = 0x8237;
export const RG16UI = 0x823a;
export const RG16I = 0x8239;
export const RG32UI = 0x823c;
export const RG32I = 0x823b;

export const RGB8 = 0x8051;
export const RGB8_SNORM = 0x8f96;
export const RGB16F = 0x881b;
export const RGB32F = 0x8815;
export const RGB8UI = 0x8d7d;
export const RGB8I = 0x8d8b;
export const RGB16UI = 0x8d77;
export const RGB16I = 0x8d89;
export const RGB32UI = 0x8d71;
export const RGB32I = 0x8d83;

export const RGBA8 = 0x8058;
export const RGBA8_SNORM = 0x8f97;
export const RGBA16F = 0x881a;
export const RGBA32F = 0x8814;
export const RGBA8UI = 0x8d7c;
export const RGBA8I = 0x8d8a;
export const RGBA16UI = 0x8d76;
export const RGBA16I = 0x8d88;
export const RGBA32UI = 0x8d70;
export const RGBA32I = 0x8d82;

export const RGB10_A2 = 0x8059;
export const RGB10_A2UI = 0x906f;
export const R11F_G11F_B10F = 0x8c3a;
export const RGB9_E5 = 0x8c3d;
export const SRGB8 = 0x8c41;
export const SRGB8_ALPHA8 = 0x8c43;

export const RGBA4 = 0x8056;
export const RGB5_A1 = 0x8057;
export const RGB565 = 0x8d62;

export const DEPTH_COMPONENT16 = 0x81a5;
export const DEPTH_COMPONENT24 = 0x81a6;
export const DEPTH_COMPONENT32F = 0x8cac;
export const DEPTH24_STENCIL8 = 0x88f0;
export const DEPTH32F_STENCIL8 = 0x8cad;
export const STENCIL_INDEX8 = 0x8d48;

/* ------------------------------------------------------------------ */
/* Framebuffer attachment points                                       */
/* ------------------------------------------------------------------ */
export const COLOR_ATTACHMENT0 = 0x8ce0;
export const DEPTH_ATTACHMENT = 0x8d00;
export const STENCIL_ATTACHMENT = 0x8d20;

/* ------------------------------------------------------------------ */
/* EXT_clip_control (clip origin + depth mode)                         */
/* ------------------------------------------------------------------ */
export const LOWER_LEFT_EXT = 0x8ca1;
export const UPPER_LEFT_EXT = 0x8ca2;
export const NEGATIVE_ONE_TO_ONE_EXT = 0x935e;
export const ZERO_TO_ONE_EXT = 0x935f;
