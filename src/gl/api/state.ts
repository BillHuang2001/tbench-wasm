/**
 * src/gl/api/state.ts — capability and simple state setters.
 *
 * Owns: enable, disable, isEnabled, blendColor, blendEquation(separate),
 * blendFunc(separate), clearColor, clearDepth, clearStencil, colorMask,
 * cullFace, depthFunc, depthMask, depthRange, frontFace, hint, lineWidth,
 * pixelStorei, polygonOffset, sampleCoverage, scissor, stencilFunc(separate),
 * stencilMask(separate), stencilOp(separate), viewport.
 *
 * Behavior notes (implemented):
 *  - All setters validate the enum/value FIRST (INVALID_ENUM/INVALID_VALUE) and
 *    are silent no-ops on context loss (NO error — the single CONTEXT_LOST_WEBGL
 *    is delivered by getError's lost-epoch; CTS context-lost.html asserts
 *    NO_ERROR after every void call while lost).
 *    WebIDL conversion failures (wrong argument types) still throw TypeError.
 *  - pixelStorei: UNPACK/PACK_ALIGNMENT ∈ {1,2,4,8} (INVALID_VALUE otherwise);
 *    UNPACK_FLIP_Y/PREMULTIPLY_ALPHA: any nonzero value = true (WebIDL GLboolean, no error);
 *    UNPACK_COLORSPACE_CONVERSION ∈ {BROWSER_DEFAULT_WEBGL, NONE}; WebGL2
 *    ROW_LENGTH, SKIP_ROWS, SKIP_PIXELS, IMAGE_HEIGHT, SKIP_IMAGES pnames
 *    accept any value ≥ 0 and are INVALID_ENUM on WebGL1. Integer pnames are
 *    WebIDL-long converted
 *    (param|0) before range checks.
 *  - lineWidth: only width ≤ 0 or NaN generate INVALID_VALUE (the CTS checks
 *    zero, negative and NaN); any other value — including values above
 *    ALIASED_LINE_WIDTH_RANGE [1,1] — is stored RAW (getParameter returns the
 *    unclamped value per spec; the rasterizer clamps at draw time, draw.ts).
 *  - depthRange: INVALID_OPERATION if zNear > zFar (RAW comparison — the CTS
 *    expects depthRange(20, 10) to error even though both clamp to 1); on
 *    store both values are clamped to [0,1] (GLclampf conversion).
 *  - blendFunc: src factors include SRC_ALPHA_SATURATE, dst factors do not
 *    (both sets include the CONSTANT_* factors). Per GLES2 §4.1.7, pairing a
 *    CONSTANT_COLOR/ONE_MINUS_CONSTANT_COLOR factor with a
 *    CONSTANT_ALPHA/ONE_MINUS_CONSTANT_ALPHA factor across (src, dst) is
 *    INVALID_OPERATION (webgl-specific.html); blendFuncSeparate applies the
 *    same rule to the (srcRGB, dstRGB) and (srcAlpha, dstAlpha) pairs.
 *  - blendEquation: FUNC_ADD/FUNC_SUBTRACT/FUNC_REVERSE_SUBTRACT (+ MIN/MAX on
 *    WebGL2).
 *  - hint: GENERATE_MIPMAP_HINT (both versions) + FRAGMENT_SHADER_DERIVATIVE_HINT
 *    (WebGL2) with mode ∈ {FASTEST, NICEST, DONT_CARE}.
 *  - enable/disable/isEnabled caps: BLEND, CULL_FACE, DEPTH_TEST, DITHER,
 *    POLYGON_OFFSET_FILL, SAMPLE_ALPHA_TO_COVERAGE, SAMPLE_COVERAGE,
 *    SCISSOR_TEST, STENCIL_TEST (+ RASTERIZER_DISCARD on WebGL2); other caps →
 *    INVALID_ENUM.
 *  - polygonOffset: non-finite factor/units → INVALID_VALUE (per wave contract).
 *  - scissor/viewport: width/height must be ≥ 0 (INVALID_VALUE); stored as-is
 *    (no clamping to MAX_VIEWPORT_DIMS).
 *  - activeTexture: TEXTURE0 + i with i < MAX_COMBINED_TEXTURE_IMAGE_UNITS →
 *    state.activeTexture (0-based index).
 *
 * NOTE: ZERO/ONE/CONSTANT_COLOR/ONE_MINUS_CONSTANT_COLOR/CONSTANT_ALPHA/
 * ONE_MINUS_CONSTANT_ALPHA are missing from constants.ts (owned elsewhere);
 * blend-factor validation uses local GL values so semantics are correct even
 * before the constant tables are fixed (gl.ZERO etc. are a constants.ts bug).
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1, C2, CExt } from '../constants';
import type { State, StencilState } from '../state';
import { isClipDistanceEnabled, setClipDistanceEnabled } from '../extensions/clip-state';
import type { GLboolean, GLclampf, GLenum, GLfloat, GLint } from '../types';

type CapKey = keyof State['caps'];

// GL blend-factor values missing from constants.ts (see NOTE above).
const GL_ZERO = 0x0000;
const GL_ONE = 0x0001;
const GL_CONSTANT_COLOR = 0x8001;
const GL_ONE_MINUS_CONSTANT_COLOR = 0x8002;
const GL_CONSTANT_ALPHA = 0x8003;
const GL_ONE_MINUS_CONSTANT_ALPHA = 0x8004;

/** Capability enum → State.caps key (WebGL1 set). */
const CAP_KEYS: Record<number, CapKey> = {
  [C1.BLEND]: 'BLEND',
  [C1.CULL_FACE]: 'CULL_FACE',
  [C1.DEPTH_TEST]: 'DEPTH_TEST',
  [C1.DITHER]: 'DITHER',
  [C1.POLYGON_OFFSET_FILL]: 'POLYGON_OFFSET_FILL',
  [C1.SAMPLE_ALPHA_TO_COVERAGE]: 'SAMPLE_ALPHA_TO_COVERAGE',
  [C1.SAMPLE_COVERAGE]: 'SAMPLE_COVERAGE',
  [C1.SCISSOR_TEST]: 'SCISSOR_TEST',
  [C1.STENCIL_TEST]: 'STENCIL_TEST',
};

/** WebGL2 adds RASTERIZER_DISCARD. */
const CAP_KEYS_V2: Record<number, CapKey> = {
  ...CAP_KEYS,
  [C2.RASTERIZER_DISCARD]: 'RASTERIZER_DISCARD',
};

const SRC_BLEND_FACTORS: number[] = [
  GL_ZERO,
  GL_ONE,
  C1.SRC_COLOR,
  C1.ONE_MINUS_SRC_COLOR,
  C1.SRC_ALPHA,
  C1.ONE_MINUS_SRC_ALPHA,
  C1.DST_ALPHA,
  C1.ONE_MINUS_DST_ALPHA,
  C1.DST_COLOR,
  C1.ONE_MINUS_DST_COLOR,
  C1.SRC_ALPHA_SATURATE,
  GL_CONSTANT_COLOR,
  GL_ONE_MINUS_CONSTANT_COLOR,
  GL_CONSTANT_ALPHA,
  GL_ONE_MINUS_CONSTANT_ALPHA,
];

/** dst factors: the src set minus SRC_ALPHA_SATURATE. */
const DST_BLEND_FACTORS: number[] = SRC_BLEND_FACTORS.filter((f) => f !== C1.SRC_ALPHA_SATURATE);

const BLEND_EQUATIONS: number[] = [C1.FUNC_ADD, C1.FUNC_SUBTRACT, C1.FUNC_REVERSE_SUBTRACT];
const BLEND_EQUATIONS_V2: number[] = [...BLEND_EQUATIONS, C2.MIN, C2.MAX];

// WEBGL_blend_func_extended (versions [1,2]): dual-source factors. Exported
// for the draw engine's dual-source draw-time validation (src/gl/draw.ts).
export const SRC1_BLEND_FACTORS: number[] = [
  0x88f9, // SRC1_COLOR_WEBGL
  0x8589, // SRC1_ALPHA_WEBGL
  0x88fa, // ONE_MINUS_SRC1_COLOR_WEBGL
  0x88fb, // ONE_MINUS_SRC1_ALPHA_WEBGL
];

/** Factor sets per context (extensions widen the WebGL1 sets when enabled). */
function blendFactorSets(ctx: WebGLRenderingContext): { src: number[]; dst: number[] } {
  let src = SRC_BLEND_FACTORS;
  let dst = DST_BLEND_FACTORS;
  if (ctx._extensions.has('WEBGL_blend_func_extended')) {
    // Spec (WEBGL_blend_func_extended): SRC1_* legal for src AND dst; the
    // extension also promotes SRC_ALPHA_SATURATE to the dst side (srcAlpha/
    // dstRGB/dstAlpha positions).
    src = [...src, ...SRC1_BLEND_FACTORS];
    dst = [...dst, C1.SRC_ALPHA_SATURATE, ...SRC1_BLEND_FACTORS];
  }
  return { src, dst };
}

/** CONSTANT_COLOR / ONE_MINUS_CONSTANT_COLOR factor check. */
function isConstColorFactor(f: number): boolean {
  return f === GL_CONSTANT_COLOR || f === GL_ONE_MINUS_CONSTANT_COLOR;
}

/** CONSTANT_ALPHA / ONE_MINUS_CONSTANT_ALPHA factor check. */
function isConstAlphaFactor(f: number): boolean {
  return f === GL_CONSTANT_ALPHA || f === GL_ONE_MINUS_CONSTANT_ALPHA;
}

/**
 * GLES2 §4.1.7: a CONSTANT_*_COLOR factor may not be paired with a
 * CONSTANT_*_ALPHA factor across a blend (src, dst) pair → INVALID_OPERATION
 * (webgl-specific.html). SRC1_* and other factors are unaffected.
 */
function constFactorPairInvalid(src: number, dst: number): boolean {
  return (
    (isConstColorFactor(src) && isConstAlphaFactor(dst)) ||
    (isConstAlphaFactor(src) && isConstColorFactor(dst))
  );
}

/** Per-drawbuffer blend entry shape (mirrors State.blendPerDrawBuffer values). */
interface BlendEntryAll {
  srcRGB: number; dstRGB: number; srcAlpha: number; dstAlpha: number;
  eqRGB: number; eqAlpha: number;
}

/**
 * OES_draw_buffers_indexed: non-indexed blend setters modify the blend state
 * of ALL draw buffers (buffer 0's entry is what getParameter + the draw
 * pipeline read). Overwrites every per-drawbuffer entry with the freshly-set
 * values so previous blendEquationiOES/blendFunciOES entries are replaced
 * (CTS oes-draw-buffers-indexed.html "Non-indexed calls modify all draw
 * buffers state").
 */
function writeBlendAllDrawBuffers(ctx: WebGLRenderingContext, e: BlendEntryAll): void {
  const s = ctx._state;
  for (let i = 0; i < s.limits.MAX_DRAW_BUFFERS; i++) s.blendPerDrawBuffer.set(i, e);
}

/** Same overwrite-all semantics for colorMask (non-indexed colorMask modifies every draw buffer). */
function writeColorMaskAllDrawBuffers(ctx: WebGLRenderingContext, m: [boolean, boolean, boolean, boolean]): void {
  const s = ctx._state;
  for (let i = 0; i < s.limits.MAX_DRAW_BUFFERS; i++) s.colorMaskPerDrawBuffer.set(i, [m[0], m[1], m[2], m[3]]);
}
/** Blend equation set per context (EXT_blend_minmax widens WebGL1). */
function blendEquations(ctx: WebGLRenderingContext): number[] {
  if (ctx._version === 2) return BLEND_EQUATIONS_V2;
  if (ctx._extensions.has('EXT_blend_minmax')) return [...BLEND_EQUATIONS, CExt.MIN_EXT, CExt.MAX_EXT];
  return BLEND_EQUATIONS;
}

const DEPTH_FUNCS: number[] = [
  C1.NEVER,
  C1.LESS,
  C1.EQUAL,
  C1.LEQUAL,
  C1.GREATER,
  C1.NOTEQUAL,
  C1.GEQUAL,
  C1.ALWAYS,
];

/** Stencil test funcs are the same eight as depth (GLES 2.0 §6.2). */
const STENCIL_FUNCS: number[] = DEPTH_FUNCS;

/** Stencil ops (stencilOp/OpSeparate fail/zfail/zpass). */
const STENCIL_OPS: number[] = [
  C1.KEEP,
  C1.ZERO,
  C1.REPLACE,
  C1.INCR,
  C1.DECR,
  C1.INVERT,
  C1.INCR_WRAP,
  C1.DECR_WRAP,
];

/** Stencil faces for the *Separate methods (INVALID_ENUM otherwise). */
const STENCIL_FACES: number[] = [C1.FRONT, C1.BACK, C1.FRONT_AND_BACK];

/** Resolve face → the affected StencilState(s); null → INVALID_ENUM. */
function stencilFaceStates(ctx: WebGLRenderingContext, face: GLenum): StencilState[] | null {
  switch (face) {
    case C1.FRONT:
      return [ctx._state.stencil.front];
    case C1.BACK:
      return [ctx._state.stencil.back];
    case C1.FRONT_AND_BACK:
      return [ctx._state.stencil.front, ctx._state.stencil.back];
    default:
      return null;
  }
}

const HINT_MODES: number[] = [C1.FASTEST, C1.NICEST, C1.DONT_CARE];

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

/** GLclampf conversion: NaN → 0, then clamp to [0,1]. */
function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function installStateApi(proto: WebGLRenderingContext): void {
  proto.activeTexture = function (this: WebGLRenderingContext, texture: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const unit = texture >>> 0; // WebIDL unsigned long conversion
    const max = ctx._state.limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS;
    if (unit < C1.TEXTURE0 || unit >= C1.TEXTURE0 + max) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.activeTexture = unit - C1.TEXTURE0;
  };

  proto.blendColor = function (this: WebGLRenderingContext, red: GLclampf, green: GLclampf, blue: GLclampf, alpha: GLclampf): void {
    const ctx = this;
    if (isLost(ctx)) return;
    ctx._state.blend.color = [clamp01(red), clamp01(green), clamp01(blue), clamp01(alpha)];
  };

  proto.blendEquation = function (this: WebGLRenderingContext, mode: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const eqs = blendEquations(ctx);
    if (!eqs.includes(mode)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.blend.eqRGB = mode;
    ctx._state.blend.eqAlpha = mode;
    // OES_draw_buffers_indexed: non-indexed setters update ALL draw buffers.
    const s = ctx._state;
    writeBlendAllDrawBuffers(ctx, {
      srcRGB: s.blend.srcRGB, dstRGB: s.blend.dstRGB,
      srcAlpha: s.blend.srcAlpha, dstAlpha: s.blend.dstAlpha,
      eqRGB: mode, eqAlpha: mode,
    });
  };

  proto.blendEquationSeparate = function (this: WebGLRenderingContext, modeRGB: GLenum, modeAlpha: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const eqs = blendEquations(ctx);
    if (!eqs.includes(modeRGB) || !eqs.includes(modeAlpha)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.blend.eqRGB = modeRGB;
    ctx._state.blend.eqAlpha = modeAlpha;
    const s = ctx._state;
    writeBlendAllDrawBuffers(ctx, {
      srcRGB: s.blend.srcRGB, dstRGB: s.blend.dstRGB,
      srcAlpha: s.blend.srcAlpha, dstAlpha: s.blend.dstAlpha,
      eqRGB: modeRGB, eqAlpha: modeAlpha,
    });
  };

  proto.blendFunc = function (this: WebGLRenderingContext, sfactor: GLenum, dfactor: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const { src, dst } = blendFactorSets(ctx);
    if (!src.includes(sfactor) || !dst.includes(dfactor)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (constFactorPairInvalid(sfactor, dfactor)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    ctx._state.blend.srcRGB = sfactor;
    ctx._state.blend.dstRGB = dfactor;
    ctx._state.blend.srcAlpha = sfactor;
    ctx._state.blend.dstAlpha = dfactor;
    const s = ctx._state;
    writeBlendAllDrawBuffers(ctx, {
      srcRGB: sfactor, dstRGB: dfactor,
      srcAlpha: sfactor, dstAlpha: dfactor,
      eqRGB: s.blend.eqRGB, eqAlpha: s.blend.eqAlpha,
    });
  };

  proto.blendFuncSeparate = function (this: WebGLRenderingContext, srcRGB: GLenum, dstRGB: GLenum, srcAlpha: GLenum, dstAlpha: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const { src, dst } = blendFactorSets(ctx);
    if (
      !src.includes(srcRGB) ||
      !dst.includes(dstRGB) ||
      !src.includes(srcAlpha) ||
      !dst.includes(dstAlpha)
    ) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (constFactorPairInvalid(srcRGB, dstRGB) || constFactorPairInvalid(srcAlpha, dstAlpha)) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    ctx._state.blend.srcRGB = srcRGB;
    ctx._state.blend.dstRGB = dstRGB;
    ctx._state.blend.srcAlpha = srcAlpha;
    ctx._state.blend.dstAlpha = dstAlpha;
    const s = ctx._state;
    writeBlendAllDrawBuffers(ctx, {
      srcRGB, dstRGB, srcAlpha, dstAlpha,
      eqRGB: s.blend.eqRGB, eqAlpha: s.blend.eqAlpha,
    });
  };

  proto.clearColor = function (this: WebGLRenderingContext, red: GLclampf, green: GLclampf, blue: GLclampf, alpha: GLclampf): void {
    const ctx = this;
    if (isLost(ctx)) return;
    // EXT_color_buffer_half_float spec: "clearColor() will no longer clamp its
    // parameter values on input" — store raw values in WebGL2 (float buffers are
    // core-renderable there) and in WebGL1 once the extension is enabled.
    if (ctx._version === 2 || ctx._extensions.has('EXT_color_buffer_half_float')) {
      ctx._state.clearColor = [Number.isNaN(red) ? 0 : red, Number.isNaN(green) ? 0 : green, Number.isNaN(blue) ? 0 : blue, Number.isNaN(alpha) ? 0 : alpha];
    } else {
      ctx._state.clearColor = [clamp01(red), clamp01(green), clamp01(blue), clamp01(alpha)];
    }
  };

  proto.clearDepth = function (this: WebGLRenderingContext, depth: GLclampf): void {
    const ctx = this;
    if (isLost(ctx)) return;
    ctx._state.clearDepth = clamp01(depth);
  };

  proto.clearStencil = function (this: WebGLRenderingContext, s: GLint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    ctx._state.clearStencil = s | 0; // WebIDL long conversion
  };

  proto.colorMask = function (this: WebGLRenderingContext, red: GLboolean, green: GLboolean, blue: GLboolean, alpha: GLboolean): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const m: [boolean, boolean, boolean, boolean] = [!!red, !!green, !!blue, !!alpha];
    ctx._state.colorMask = m;
    // OES_draw_buffers_indexed: non-indexed colorMask updates ALL draw
    // buffers (overwrites previous colorMaskiOES entries — CTS
    // oes-draw-buffers-indexed.html expects getIndexedParameter(COLOR_WRITEMASK,
    // 0) AND (..., 1) to both reflect the new value after gl.colorMask).
    writeColorMaskAllDrawBuffers(ctx, m);
  };

  proto.cullFace = function (this: WebGLRenderingContext, mode: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (mode !== C1.FRONT && mode !== C1.BACK && mode !== C1.FRONT_AND_BACK) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.cullFace = mode;
  };

  proto.depthFunc = function (this: WebGLRenderingContext, func: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!DEPTH_FUNCS.includes(func)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.depth.func = func;
  };

  proto.depthMask = function (this: WebGLRenderingContext, flag: GLboolean): void {
    const ctx = this;
    if (isLost(ctx)) return;
    ctx._state.depth.mask = !!flag;
  };

  proto.depthRange = function (this: WebGLRenderingContext, zNear: GLclampf, zFar: GLclampf): void {
    const ctx = this;
    if (isLost(ctx)) return;
    // GLES2 §2.11.1: INVALID_OPERATION if zNear > zFar. The comparison is on
    // the RAW values (webgl-specific.html expects depthRange(20, 10) to
    // error even though both clamp to 1); clamping happens on store.
    if (zNear > zFar) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    ctx._state.depth.range = [clamp01(zNear), clamp01(zFar)];
  };

  proto.disable = function (this: WebGLRenderingContext, cap: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const table = ctx._version === 2 ? CAP_KEYS_V2 : CAP_KEYS;
    const key = table[cap];
    if (key === undefined) {
      // WEBGL_clip_cull_distance: CLIP_DISTANCE0..7 become legal caps.
      if (ctx._extensions.has('WEBGL_clip_cull_distance') && cap >= 0x3000 && cap <= 0x3007) {
        setClipDistanceEnabled(ctx, cap - 0x3000, false);
        return;
      }
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.caps[key] = false;
  };

  proto.enable = function (this: WebGLRenderingContext, cap: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const table = ctx._version === 2 ? CAP_KEYS_V2 : CAP_KEYS;
    const key = table[cap];
    if (key === undefined) {
      // WEBGL_clip_cull_distance: CLIP_DISTANCE0..7 become legal caps.
      if (ctx._extensions.has('WEBGL_clip_cull_distance') && cap >= 0x3000 && cap <= 0x3007) {
        setClipDistanceEnabled(ctx, cap - 0x3000, true);
        return;
      }
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.caps[key] = true;
  };

  proto.frontFace = function (this: WebGLRenderingContext, mode: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (mode !== C1.CW && mode !== C1.CCW) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.frontFace = mode;
  };

  proto.hint = function (this: WebGLRenderingContext, target: GLenum, mode: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!HINT_MODES.includes(mode)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (target === C1.GENERATE_MIPMAP_HINT) {
      ctx._state.hints.generateMipmap = mode;
    } else if (
      target === C2.FRAGMENT_SHADER_DERIVATIVE_HINT &&
      (ctx._version === 2 || ctx._extensions.has('OES_standard_derivatives'))
    ) {
      ctx._state.hints.fragmentShaderDerivative = mode;
    } else {
      ctx._errors.push(C1.INVALID_ENUM);
    }
  };

  proto.isEnabled = function (this: WebGLRenderingContext, cap: GLenum): GLboolean {
    const ctx = this;
    if (isLost(ctx)) return false;
    const table = ctx._version === 2 ? CAP_KEYS_V2 : CAP_KEYS;
    const key = table[cap];
    if (key === undefined) {
      // WEBGL_clip_cull_distance: CLIP_DISTANCE0..7 become legal caps.
      if (ctx._extensions.has('WEBGL_clip_cull_distance') && cap >= 0x3000 && cap <= 0x3007) {
        return isClipDistanceEnabled(ctx, cap - 0x3000);
      }
      ctx._errors.push(C1.INVALID_ENUM);
      return false;
    }
    return ctx._state.caps[key];
  };

  proto.lineWidth = function (this: WebGLRenderingContext, width: GLfloat): void {
    const ctx = this;
    if (isLost(ctx)) return;
    // Only width ≤ 0 or NaN generate INVALID_VALUE (CTS limits/gl-line-width
    // checks zero, negative and NaN). Any other value is stored RAW — the
    // rasterizer clamps to ALIASED_LINE_WIDTH_RANGE at draw time (draw.ts),
    // and getParameter must return the unclamped value per spec.
    if (!(width > 0)) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    ctx._state.lineWidth = width;
  };

  proto.pixelStorei = function (this: WebGLRenderingContext, pname: GLenum, param: GLint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const s = ctx._state;
    const v = param | 0; // WebIDL long conversion (pixelStorei's param is GLint)
    switch (pname) {
      case C1.UNPACK_ALIGNMENT:
      case C1.PACK_ALIGNMENT: {
        if (v !== 1 && v !== 2 && v !== 4 && v !== 8) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
        if (pname === C1.UNPACK_ALIGNMENT) s.pixelStore.unpack.alignment = v;
        else s.pixelStore.pack.alignment = v;
        return;
      }
      case C1.UNPACK_FLIP_Y_WEBGL: {
        // WebIDL GLboolean: ANY nonzero value is true — no error.
        s.pixelStore.unpack.flipY = v !== 0;
        return;
      }
      case C1.UNPACK_PREMULTIPLY_ALPHA_WEBGL: {
        // WebIDL GLboolean: ANY nonzero value is true — no error.
        s.pixelStore.unpack.premultiplyAlpha = v !== 0;
        return;
      }
      case C1.UNPACK_COLORSPACE_CONVERSION_WEBGL: {
        if (v !== C1.BROWSER_DEFAULT_WEBGL && v !== C1.NONE) {
          ctx._errors.push(C1.INVALID_VALUE);
          return;
        }
        s.pixelStore.unpack.colorspaceConversion = v;
        return;
      }
      default:
        if (ctx._version === 2) {
          switch (pname) {
            case C2.UNPACK_ROW_LENGTH:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.unpack.rowLength = v;
              return;
            case C2.UNPACK_SKIP_ROWS:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.unpack.skipRows = v;
              return;
            case C2.UNPACK_SKIP_PIXELS:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.unpack.skipPixels = v;
              return;
            case C2.UNPACK_IMAGE_HEIGHT:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.unpack.imageHeight = v;
              return;
            case C2.UNPACK_SKIP_IMAGES:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.unpack.skipImages = v;
              return;
            case C2.PACK_ROW_LENGTH:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.pack.rowLength = v;
              return;
            case C2.PACK_SKIP_ROWS:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.pack.skipRows = v;
              return;
            case C2.PACK_SKIP_PIXELS:
              if (v < 0) ctx._errors.push(C1.INVALID_VALUE);
              else s.pixelStore.pack.skipPixels = v;
              return;
          }
        }
        ctx._errors.push(C1.INVALID_ENUM);
        return;
    }
  };

  proto.polygonOffset = function (this: WebGLRenderingContext, factor: GLfloat, units: GLfloat): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!Number.isFinite(factor) || !Number.isFinite(units)) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    ctx._state.polygonOffset.factor = factor;
    ctx._state.polygonOffset.units = units;
  };

  proto.sampleCoverage = function (this: WebGLRenderingContext, value: GLclampf, invert: GLboolean): void {
    const ctx = this;
    if (isLost(ctx)) return;
    ctx._state.sampleCoverage.value = clamp01(value);
    ctx._state.sampleCoverage.invert = !!invert;
  };

  proto.scissor = function (this: WebGLRenderingContext, x: GLint, y: GLint, width: GLsizei, height: GLsizei): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const xv = x | 0;
    const yv = y | 0;
    const wv = width | 0;
    const hv = height | 0;
    if (wv < 0 || hv < 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    ctx._state.scissor = { x: xv, y: yv, w: wv, h: hv };
  };

  proto.stencilFunc = function (this: WebGLRenderingContext, func: GLenum, ref: GLint, mask: GLuint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!STENCIL_FUNCS.includes(func)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    // WebIDL conversions: ref is a GLint (long), mask a GLuint (unsigned long).
    const r = ref | 0;
    const m = mask >>> 0;
    ctx._state.stencil.front.func = func;
    ctx._state.stencil.front.ref = r;
    ctx._state.stencil.front.valueMask = m;
    ctx._state.stencil.back.func = func;
    ctx._state.stencil.back.ref = r;
    ctx._state.stencil.back.valueMask = m;
  };

  proto.stencilFuncSeparate = function (this: WebGLRenderingContext, face: GLenum, func: GLenum, ref: GLint, mask: GLuint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const states = stencilFaceStates(ctx, face);
    if (states === null) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (!STENCIL_FUNCS.includes(func)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const r = ref | 0;
    const m = mask >>> 0;
    for (const st of states) {
      st.func = func;
      st.ref = r;
      st.valueMask = m;
    }
  };

  proto.stencilMask = function (this: WebGLRenderingContext, mask: GLuint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const m = mask >>> 0;
    ctx._state.stencil.front.writeMask = m;
    ctx._state.stencil.back.writeMask = m;
  };

  proto.stencilMaskSeparate = function (this: WebGLRenderingContext, face: GLenum, mask: GLuint): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const states = stencilFaceStates(ctx, face);
    if (states === null) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const m = mask >>> 0;
    for (const st of states) st.writeMask = m;
  };

  proto.stencilOp = function (this: WebGLRenderingContext, fail: GLenum, zfail: GLenum, zpass: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!STENCIL_OPS.includes(fail) || !STENCIL_OPS.includes(zfail) || !STENCIL_OPS.includes(zpass)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    ctx._state.stencil.front.fail = fail;
    ctx._state.stencil.front.depthFail = zfail;
    ctx._state.stencil.front.depthPass = zpass;
    ctx._state.stencil.back.fail = fail;
    ctx._state.stencil.back.depthFail = zfail;
    ctx._state.stencil.back.depthPass = zpass;
  };

  proto.stencilOpSeparate = function (this: WebGLRenderingContext, face: GLenum, fail: GLenum, zfail: GLenum, zpass: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const states = stencilFaceStates(ctx, face);
    if (states === null) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    if (!STENCIL_OPS.includes(fail) || !STENCIL_OPS.includes(zfail) || !STENCIL_OPS.includes(zpass)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    for (const st of states) {
      st.fail = fail;
      st.depthFail = zfail;
      st.depthPass = zpass;
    }
  };

  proto.viewport = function (this: WebGLRenderingContext, x: GLint, y: GLint, width: GLsizei, height: GLsizei): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const xv = x | 0;
    const yv = y | 0;
    const wv = width | 0;
    const hv = height | 0;
    if (wv < 0 || hv < 0) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    ctx._state.viewport = { x: xv, y: yv, w: wv, h: hv };
  };
}
