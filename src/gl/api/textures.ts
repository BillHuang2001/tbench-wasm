/**
 * src/gl/api/textures.ts — texture objects, parameters, mipmaps.
 *
 * Owns: createTexture, deleteTexture, isTexture, bindTexture, texParameterf,
 * texParameteri, getTexParameter, generateMipmap.
 * (Storage uploads live in api/teximage.ts → engine teximage.ts; activeTexture
 * and pixelStorei are owned by api/state.ts.)
 *
 * Behavior notes (implemented):
 *  - bindTexture: target ∈ {TEXTURE_2D, TEXTURE_CUBE_MAP} (WebGL1) +
 *    {TEXTURE_3D, TEXTURE_2D_ARRAY} (WebGL2). First bind fixes the texture's
 *    target (INVALID_OPERATION on mismatch). Binding updates the active unit's
 *    slot (state.textureUnits[activeTexture]); null unbinds. Re-binding a
 *    delete-pending texture first unbinds it everywhere (deferred deletion).
 *  - deleteTexture: marks _deleted, untracks, unbinds from all texture units;
 *    _deletePending = true when it was bound anywhere. Fake objects throw
 *    TypeError (WebIDL); cross-context → INVALID_OPERATION; already-deleted →
 *    silent no-op. FBO-attachment unbinding is owned by api/framebuffers.ts.
 *  - isTexture: false when lost/null/not-a-texture/cross-context/deleted —
 *    never generates an error.
 *  - texParameterf/i: valid (target, pname, value) table per version;
 *    TEXTURE_MAX_ANISOTROPY_EXT requires EXT_texture_filter_anisotropic.
 *    WebGL2 adds WRAP_R/MIN_LOD/MAX_LOD/BASE_LEVEL/MAX_LEVEL/COMPARE_MODE/
 *    COMPARE_FUNC. BASE_LEVEL/MAX_LEVEL < 0 or MAX_ANISO < 1 → INVALID_VALUE.
 *  - getTexParameter: TEXTURE_IMMUTABLE_FORMAT / TEXTURE_IMMUTABLE_LEVELS
 *    (WebGL2 only), else the texParameter pname table → _params value.
 *  - generateMipmap: complete, non-depth/non-stencil/non-integer base level
 *    required (INVALID_OPERATION otherwise); WebGL1 additionally rejects a
 *    non-power-of-two base level (INVALID_OPERATION — NPOT textures have no
 *    mip chain); float formats need the linear-
 *    filter extensions (W1: OES_texture_float_linear / OES_texture_half_float_linear;
 *    W2: EXT_color_buffer_float; RGB9_E5 always rejected). Delegates to the
 *    engine's generateMipmap for the actual level building.
 */

import type { WebGLRenderingContext } from '../webgl1';
import { C1, C2 } from '../constants';
import { WebGLFramebuffer, WebGLTexture, createObject } from '../objects';
import { validateObject } from '../validation';
import { generateMipmap, updateCompleteness, refreshUnitSamplerBindings, floatLinearExtensionState } from '../teximage';
import type { GLboolean, GLenum, GLfloat, GLint } from '../types';

// GL values not present in C1 (see constants.ts provenance / state.ts precedent).
const COMPARE_REF_TO_TEXTURE = 0x884e;
const TEXTURE_IMMUTABLE_FORMAT = 0x912f;
const TEXTURE_IMMUTABLE_LEVELS = 0x82df;
const RGB9_E5 = 0x8c3d;

const isPow2 = (v: number): boolean => v > 0 && (v & (v - 1)) === 0;

/**
 * Textures that have been bound at least once. isTexture returns false for
 * never-bound objects (CTS is-object.html) — mirrors everBoundRenderbuffers in
 * api/framebuffers.ts.
 */
const everBoundTextures = new WeakSet<WebGLTexture>();

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
  return ctx._isLost;
}

// WebGLTexture inherits WebGLObject's PROTECTED constructor, so its `typeof`
// is not assignable to the generic helpers' `new (...)` constructor params —
// cast once here (runtime behavior is identical; buffers.ts precedent).
const TextureCtor = WebGLTexture as unknown as new (context: WebGLRenderingContext) => WebGLTexture;
const TextureCtorAny = TextureCtor as unknown as new (...args: never[]) => WebGLTexture;

/** validateObject wrapper with the WebGLTexture constructor cast. */
function validateTexture(ctx: WebGLRenderingContext, texture: unknown): WebGLTexture | null {
  return validateObject<WebGLTexture>(ctx, texture, TextureCtorAny);
}

/** Texture-target slot keys in TextureUnitState (bind/unbind via string key). */
type TextureSlotKey = 'texture2D' | 'textureCube' | 'texture3D' | 'texture2DArray';

/** Texture bind targets per context version. */
function isValidTextureTarget(ctx: WebGLRenderingContext, target: GLenum): boolean {
  if (target === C1.TEXTURE_2D || target === C1.TEXTURE_CUBE_MAP) return true;
  if (ctx._version !== 2) return false;
  return target === C2.TEXTURE_3D || target === C2.TEXTURE_2D_ARRAY;
}

/** The active unit's binding slot for a (valid) texture target. */
function slotForTarget(target: GLenum): TextureSlotKey {
  switch (target) {
    case C1.TEXTURE_2D:
      return 'texture2D';
    case C1.TEXTURE_CUBE_MAP:
      return 'textureCube';
    case C2.TEXTURE_3D:
      return 'texture3D';
    default: // TEXTURE_2D_ARRAY
      return 'texture2DArray';
  }
}

/** The texture bound to a (valid) target in the active unit — null when none. */
function boundTextureForTarget(ctx: WebGLRenderingContext, target: GLenum): WebGLTexture | null {
  const unit = ctx._state.textureUnits[ctx._state.activeTexture];
  // state.ts texture-unit slots are typed with the DOM WebGLTexture interface;
  // they always hold renderer WebGLTexture instances (bindTexture writes them).
  return unit[slotForTarget(target)] as unknown as WebGLTexture | null;
}

/** Unbind `texture` from every texture-unit slot. Returns true when it was bound. */
function unbindTextureEverywhere(ctx: WebGLRenderingContext, texture: WebGLTexture): boolean {
  let found = false;
  for (const unit of ctx._state.textureUnits) {
    const u = unit as unknown as Record<TextureSlotKey, WebGLTexture | null>;
    for (const slot of ['texture2D', 'textureCube', 'texture3D', 'texture2DArray'] as const) {
      if (u[slot] === texture) {
        u[slot] = null;
        found = true;
      }
    }
  }
  return found;
}

// ---- texParameter pname / value tables -------------------------------------

const TEX_PARAM_PNAMES_W1: number[] = [
  C1.TEXTURE_MIN_FILTER,
  C1.TEXTURE_MAG_FILTER,
  C1.TEXTURE_WRAP_S,
  C1.TEXTURE_WRAP_T,
];

const TEX_PARAM_PNAMES_W2: number[] = [
  ...TEX_PARAM_PNAMES_W1,
  C2.TEXTURE_WRAP_R,
  C2.TEXTURE_MIN_LOD,
  C2.TEXTURE_MAX_LOD,
  C2.TEXTURE_BASE_LEVEL,
  C2.TEXTURE_MAX_LEVEL,
  C2.TEXTURE_COMPARE_MODE,
  C2.TEXTURE_COMPARE_FUNC,
];

const MIN_FILTER_VALUES: number[] = [
  C1.NEAREST,
  C1.LINEAR,
  C1.NEAREST_MIPMAP_NEAREST,
  C1.NEAREST_MIPMAP_LINEAR,
  C1.LINEAR_MIPMAP_NEAREST,
  C1.LINEAR_MIPMAP_LINEAR,
];
const MAG_FILTER_VALUES: number[] = [C1.NEAREST, C1.LINEAR];
const WRAP_VALUES: number[] = [C1.REPEAT, C1.CLAMP_TO_EDGE, C1.MIRRORED_REPEAT];
const COMPARE_MODE_VALUES: number[] = [C1.NONE, COMPARE_REF_TO_TEXTURE];
const COMPARE_FUNC_VALUES: number[] = [
  C1.NEVER,
  C1.LESS,
  C1.EQUAL,
  C1.LEQUAL,
  C1.GREATER,
  C1.NOTEQUAL,
  C1.GEQUAL,
  C1.ALWAYS,
];

/** pname legality per version + extensions (INVALID_ENUM for unknown pnames). */
function isValidTexParamPname(ctx: WebGLRenderingContext, pname: GLenum): boolean {
  if (TEX_PARAM_PNAMES_W1.includes(pname)) return true;
  if (pname === 0x84fe /* TEXTURE_MAX_ANISOTROPY_EXT */) {
    return ctx._extensions.has('EXT_texture_filter_anisotropic');
  }
  if (ctx._version !== 2) return false;
  return TEX_PARAM_PNAMES_W2.includes(pname);
}

/** Value validation for a (pname, value) pair; pushes the error when invalid. */
function isValidTexParamValue(ctx: WebGLRenderingContext, pname: GLenum, param: number): boolean {
  switch (pname) {
    case C1.TEXTURE_MIN_FILTER:
      if (!MIN_FILTER_VALUES.includes(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return false;
      }
      return true;
    case C1.TEXTURE_MAG_FILTER:
      if (!MAG_FILTER_VALUES.includes(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return false;
      }
      return true;
    case C1.TEXTURE_WRAP_S:
    case C1.TEXTURE_WRAP_T:
    case C2.TEXTURE_WRAP_R:
      if (!WRAP_VALUES.includes(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return false;
      }
      return true;
    case C2.TEXTURE_COMPARE_MODE:
      if (!COMPARE_MODE_VALUES.includes(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return false;
      }
      return true;
    case C2.TEXTURE_COMPARE_FUNC:
      if (!COMPARE_FUNC_VALUES.includes(param)) {
        ctx._errors.push(C1.INVALID_ENUM);
        return false;
      }
      return true;
    case C2.TEXTURE_BASE_LEVEL:
    case C2.TEXTURE_MAX_LEVEL:
      if (param < 0) {
        ctx._errors.push(C1.INVALID_VALUE);
        return false;
      }
      return true;
    case 0x84fe: // TEXTURE_MAX_ANISOTROPY_EXT
      if (param < 1) {
        ctx._errors.push(C1.INVALID_VALUE);
        return false;
      }
      return true;
    default:
      // MIN_LOD / MAX_LOD: any value legal (GL float).
      return true;
  }
}

/** Shared texParameterf/texParameteri implementation. */
function texParameterImpl(ctx: WebGLRenderingContext, target: GLenum, pname: GLenum, param: number): void {
  if (isLost(ctx)) return;
  if (!isValidTextureTarget(ctx, target)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  const tex = boundTextureForTarget(ctx, target);
  if (tex === null) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return;
  }
  if (!isValidTexParamPname(ctx, pname)) {
    ctx._errors.push(C1.INVALID_ENUM);
    return;
  }
  if (!isValidTexParamValue(ctx, pname, param)) return;
  tex._params[pname] = param;
  // MIN_FILTER / MAG_FILTER / BASE_LEVEL / MAX_LEVEL affect completeness
  // (MAG_FILTER doesn't, but recomputing is harmless) — recompute from the
  // CURRENT params so a texture uploaded while MIN_FILTER was the default
  // NEAREST_MIPMAP_LINEAR and later switched to LINEAR becomes complete.
  if (pname === 0x2801 /* MIN_FILTER */ || pname === 0x2800 /* MAG_FILTER */ ||
      pname === 0x813c /* BASE_LEVEL */ || pname === 0x813d /* MAX_LEVEL */) {
    updateCompleteness(tex, ctx._version, floatLinearExtensionState(ctx));
  }
}

export function installTexturesApi(proto: WebGLRenderingContext): void {
  proto.createTexture = function (this: WebGLRenderingContext): WebGLTexture | null {
    const ctx = this;
    // No [WebGLHandlesContextLoss]: while lost it still creates an object
    // (CTS context-lost.html nonNullTests) with NO error; isTexture on it →
    // false while lost (isLost guard).
    return createObject(ctx, TextureCtor);
  };

  proto.deleteTexture = function (this: WebGLRenderingContext, texture: WebGLTexture | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (texture === null || texture === undefined) return;
    if (!(texture instanceof WebGLTexture)) {
      throw new TypeError(`Argument is not of type 'WebGLTexture'`);
    }
    if (texture._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (texture._deleted) return; // already deleted: silent no-op (spec)
    const wasBound = unbindTextureEverywhere(ctx, texture);
    // Detach from every framebuffer currently bound to the context (WebGL spec:
    // deleting a texture attached to the bound framebuffer is as if
    // framebufferTexture2D(null) for each attachment point in THAT framebuffer;
    // UNBOUND framebuffers keep the attachment and the image stays alive —
    // CTS deleted-object-behavior.html testUnboundFBOTexture, object-deletion-
    // behaviour.html "using deleted texture"). Mirrors deleteRenderbuffer.
    const s = ctx._state;
    const boundFbos = new Set<WebGLFramebuffer>();
    if (s.drawFramebuffer instanceof WebGLFramebuffer) boundFbos.add(s.drawFramebuffer);
    if (s.readFramebuffer instanceof WebGLFramebuffer) boundFbos.add(s.readFramebuffer);
    for (const fbo of boundFbos) {
      let changed = false;
      for (const [key, rec] of fbo._attachments) {
        if (rec.type === 'texture' && rec.texture === texture) {
          fbo._attachments.delete(key);
          changed = true;
        }
      }
      if (changed) {
        // Invalidate the completeness cache + recompute the multisample flag
        // (same as invalidateFboStatus in api/framebuffers.ts).
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
    }
    texture._deleted = true;
    texture._deletePending = wasBound;
    ctx._resources.untrack(texture);
  };

  proto.isTexture = function (this: WebGLRenderingContext, texture: WebGLTexture | null): GLboolean {
    const ctx = this;
    if (isLost(ctx)) return false;
    if (texture === null || texture === undefined) return false;
    if (!(texture instanceof WebGLTexture)) return false;
    // Deleted, foreign (different context) and never-bound textures report
    // false with NO error (CTS incorrect-context-object-behaviour.html,
    // is-object.html).
    if (texture._context !== ctx) return false;
    if (texture._deleted) return false;
    return everBoundTextures.has(texture);
  };

  proto.bindTexture = function (this: WebGLRenderingContext, target: GLenum, texture: WebGLTexture | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!isValidTextureTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const s = ctx._state;
    const unitIdx = s.activeTexture;
    const unit = s.textureUnits[unitIdx];
    const slot = slotForTarget(target);
    if (texture === null || texture === undefined) {
      unit[slot] = null;
      // The unbound texture may have lost its sampler association (teximage.ts
      // completeness uses the bound sampler's params for sampling decisions).
      refreshUnitSamplerBindings(s, unitIdx);
      return;
    }
    if (texture instanceof WebGLTexture && texture._deletePending) {
      // Re-binding a pending-delete texture: ensure it is unbound everywhere and
      // release it (deletion was deferred at deleteTexture time). It is already
      // _deleted, so validateObject rejects the bind with INVALID_OPERATION.
      unbindTextureEverywhere(ctx, texture);
      texture._deletePending = false;
    }
    const tex = validateTexture(ctx, texture);
    if (tex === null) return; // cross-context/deleted → INVALID_OPERATION pushed
    if (tex._target === 0) tex._target = target;
    else if (tex._target !== target) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    everBoundTextures.add(tex);
    unit[slot] = tex;
    refreshUnitSamplerBindings(s, unitIdx);
  };

  proto.texParameterf = function (this: WebGLRenderingContext, target: GLenum, pname: GLenum, param: GLfloat): void {
    texParameterImpl(this, target, pname, param);
  };

  proto.texParameteri = function (this: WebGLRenderingContext, target: GLenum, pname: GLenum, param: GLint): void {
    texParameterImpl(this, target, pname, param);
  };

  proto.getTexParameter = function (this: WebGLRenderingContext, target: GLenum, pname: GLenum): any {
    const ctx = this;
    if (isLost(ctx)) return null;
    if (!isValidTextureTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    const tex = boundTextureForTarget(ctx, target);
    if (tex === null) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return null;
    }
    if (pname === TEXTURE_IMMUTABLE_FORMAT || pname === TEXTURE_IMMUTABLE_LEVELS) {
      if (ctx._version !== 2) {
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      }
      // WebGL2 spec: TEXTURE_IMMUTABLE_FORMAT returns a boolean (CTS
      // gl-object-get-calls.html asserts `false` with typeof boolean).
      if (pname === TEXTURE_IMMUTABLE_FORMAT) return tex._immutable === true;
      return tex._immutable ? tex._image!.levels.length : 0;
    }
    if (!isValidTexParamPname(ctx, pname)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    return tex._params[pname];
  };

  proto.generateMipmap = function (this: WebGLRenderingContext, target: GLenum): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (!isValidTextureTarget(ctx, target)) {
      ctx._errors.push(C1.INVALID_ENUM);
      return;
    }
    const tex = boundTextureForTarget(ctx, target);
    if (tex === null) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    const img = tex._image;
    if (!img) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    const base = tex._params[C2.TEXTURE_BASE_LEVEL] ?? 0;
    const lv = img.levels[base];
    if (!lv || lv.width < 1 || lv.height < 1) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    // WebGL 1.0 §5.14.8: generateMipmap on a non-power-of-two base level
    // generates INVALID_OPERATION (NPOT textures have no mip chain in WebGL1).
    if (ctx._version === 1 && (!isPow2(lv.width) || !isPow2(lv.height))) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (img.target === C1.TEXTURE_CUBE_MAP) {
      for (let f = 0; f < 6; f++) {
        if (lv.data[f] === undefined) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
      }
    }
    if (img.info.isDepth || img.info.isStencil || img.info.isInteger || tex._compressed) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (img.info.isFloat) {
      if (ctx._version === 2) {
        // RGB9_E5 is never mipmap-generatable; the 16F formats are
        // mipmap-generatable with either float extension (EXT_color_buffer_half_float
        // makes them color-renderable → their mip chain must be buildable), all
        // other float formats need EXT_color_buffer_float.
        const is16F =
          img.internalFormat === 0x822d /* R16F */ ||
          img.internalFormat === 0x822f /* RG16F */ ||
          img.internalFormat === 0x881a /* RGBA16F */;
        const floatExtOK = ctx._extensions.has('EXT_color_buffer_float') ||
          (is16F && ctx._extensions.has('EXT_color_buffer_half_float'));
        if (img.internalFormat === RGB9_E5 || !floatExtOK) {
          ctx._errors.push(C1.INVALID_OPERATION);
          return;
        }
      } else if (
        !ctx._extensions.has('OES_texture_float_linear') &&
        !ctx._extensions.has('OES_texture_half_float_linear')
      ) {
        ctx._errors.push(C1.INVALID_OPERATION);
        return;
      }
    }
    generateMipmap(ctx, tex, target);
  };
}
