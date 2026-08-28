/**
 * src/gl/getters.ts — getParameter implementation (api/context.ts delegates here).
 *
 * Maps every spec pname to state/limits/drawing-buffer data. Behavior rules:
 *  - Return types per spec: real booleans for caps/masks, Int32Array for
 *    SCISSOR_BOX/VIEWPORT/MAX_VIEWPORT_DIMS, Float32Array for COLOR_CLEAR_VALUE/
 *    BLEND_COLOR/DEPTH_RANGE/ALIASED_*_RANGE, plain boolean[] for COLOR_WRITEMASK,
 *    Uint32Array for COMPRESSED_TEXTURE_FORMATS, numbers otherwise. Typed arrays
 *    are fresh on every call (spec: "All queries returning sequences or typed
 *    arrays return a new object each time").
 *  - Every WebGL1 pname stays legal on WebGL2 (WebGL 2.0 spec §6.2: "If pname
 *    is not in the table above and is not one of parameter names supported by
 *    WebGL 1.0, generates an INVALID_ENUM"). WebGL2-only pnames (MAX_3D_TEXTURE_SIZE,
 *    RASTERIZER_DISCARD, ...) are INVALID_ENUM on WebGL1.
 *  - FBO-dependent pnames (SAMPLE_BUFFERS, SAMPLES, *_BITS,
 *    IMPLEMENTATION_COLOR_READ_*) read the bound DRAW framebuffer via
 *    framebuffer-util.resolveFramebufferTarget (stub-safe: any throw falls back
 *    to the default framebuffer / spec defaults). A bound-but-incomplete FBO
 *    yields 0 bits and INVALID_OPERATION for IMPLEMENTATION_COLOR_READ_*.
 *  - Extension-gated pnames (MAX_TEXTURE_MAX_ANISOTROPY_EXT, UNMASKED_*_WEBGL,
 *    FRAGMENT_SHADER_DERIVATIVE_HINT on WebGL1, WEBGL_draw_buffers / WEBGL_clip_cull_distance
 *    pnames) are INVALID_ENUM until the extension is enabled (retrieved via
 *    getExtension — matches the CTS "extension disabled" checks).
 *  - getParameter(EXTENSIONS) is illegal in WebGL (use getSupportedExtensions).
 *  - String pnames (VENDOR/RENDERER/VERSION/SHADING_LANGUAGE_VERSION) return the
 *    same strings as getString (api/context.ts) via the shared
 *    `stringValueForPname` helper — CTS functions-returning-strings.html requires
 *    non-null strings from getParameter for these.
 *  - Context lost → return null, no error pushed (spec: getters return null
 *    while the context is lost).
 *  - WebGL2 indexed pnames (UNIFORM_BUFFER_BINDING per index, TRANSFORM_FEEDBACK_BUFFER_*)
 *    are handled by getIndexedParameter; getParameter returns the GENERIC
 *    binding for UNIFORM_BUFFER_BINDING / TRANSFORM_FEEDBACK_BUFFER_BINDING
 *    (GLES 3.0 §2.10.1 — the generic point is context state, distinct from the
 *    indexed points; see api/buffers.ts getGenericBufferBinding).
 */

import type { WebGLRenderingContext } from './webgl1';
import type { GLenum } from './types';
import type { WebGLTransformFeedback } from './objects';
import { C } from './constants';
import { getGenericBufferBinding } from './api/buffers';
import { resolveFramebufferTarget } from './framebuffer-util';
import { getClipControl, isClipDistanceEnabled } from './extensions/clip-state';
import {
  ALIASED_LINE_WIDTH_RANGE,
  ALIASED_POINT_SIZE_RANGE,
  getFormat,
  type FramebufferTarget,
  type PixelFormatInfo,
  type Surface,
} from '../raster';

// ---- pnames absent from constants.ts (which only carries the context-visible surface) ----
const CURRENT_PROGRAM = 0x8b8d; // GL_CURRENT_PROGRAM
const SAMPLE_MASK = 0x8e51; // WebGL2 cap (GL_SAMPLE_MASK)
const DRAW_BUFFER0 = 0x8825; // GL_DRAW_BUFFER0 .. GL_DRAW_BUFFER15 (WebGL2 pnames)
const DRAW_BUFFER15 = 0x8834;
const BACK = 0x0405; // default draw buffer of the default framebuffer
const NONE = 0;
const INTERLEAVED_ATTRIBS = 0x8c8c; // default TRANSFORM_FEEDBACK_BUFFER_MODE

/** Full getParameter dispatch. */
export function getParameter(ctx: WebGLRenderingContext, pname: GLenum): unknown {
  if (ctx._isLost) return null;
  const s = ctx._state;
  const lim = s.limits;
  const v2 = ctx._version === 2;

  // String pnames (VENDOR/RENDERER/VERSION/SHADING_LANGUAGE_VERSION): same
  // strings as getString. EXTENSIONS (0x1f03) is NOT legal via getParameter.
  {
    const str = stringValueForPname(ctx, pname);
    if (str !== null) return str;
  }
  // DRAW_BUFFER0..DRAW_BUFFER15: per-drawbuffer output. On WebGL2 always legal;
  // on WebGL1 legal once WEBGL_draw_buffers is enabled (same pname values
  // 0x8825..0x8834). The default framebuffer reports BACK (drawBuffers([BACK]))
  // or NONE (drawBuffers([NONE])); an FBO reports the drawBuffers() values.
  if (pname >= DRAW_BUFFER0 && pname <= DRAW_BUFFER15) {
    if (!v2 && !ctx._extensions.has('WEBGL_draw_buffers')) {
      ctx._errors.push(C.INVALID_ENUM);
      return null;
    }
    const i = pname - DRAW_BUFFER0;
    if (s.drawFramebuffer === null) {
      // Storage is normalized to attachment indices ([COLOR_ATTACHMENT0] for
      // BACK, [NONE], or [] = no color buffers) — report the spec-visible value.
      const db0 = s.drawBuffers[0];
      return db0 === NONE || db0 === undefined ? NONE : BACK;
    }
    return s.drawBuffers[i] ?? NONE;
  }

  switch (pname) {
    // ---- Capabilities (isEnabled/getParameter parity) ----
    case C.BLEND: return s.caps.BLEND;
    case C.CULL_FACE: return s.caps.CULL_FACE;
    case C.DEPTH_TEST: return s.caps.DEPTH_TEST;
    case C.DITHER: return s.caps.DITHER;
    case C.POLYGON_OFFSET_FILL: return s.caps.POLYGON_OFFSET_FILL;
    case C.SAMPLE_ALPHA_TO_COVERAGE: return s.caps.SAMPLE_ALPHA_TO_COVERAGE;
    case C.SAMPLE_COVERAGE: return s.caps.SAMPLE_COVERAGE;
    case C.SCISSOR_TEST: return s.caps.SCISSOR_TEST;
    case C.STENCIL_TEST: return s.caps.STENCIL_TEST;
    case C.RASTERIZER_DISCARD:
      if (!v2) break;
      return s.caps.RASTERIZER_DISCARD;
    case SAMPLE_MASK: // GL_SAMPLE_MASK (WebGL2)
      if (!v2) break;
      return false; // single-sampled software renderer — mask never applied

    // ---- Simple numbers ----
    case C.ACTIVE_TEXTURE: return C.TEXTURE0 + s.activeTexture;
    // Blend state: non-indexed getParameter reads draw buffer 0's
    // per-drawbuffer entry (OES_draw_buffers_indexed — the non-indexed
    // setters mirror their values into every entry, and the indexed setters
    // override buffer 0's entry; CTS oes-draw-buffers-indexed.html
    // "non-indexed getParamter get state from draw buffer 0"), with the base
    // state as fallback (no extension / WebGL1).
    case C.BLEND_SRC_RGB: { const b0 = s.blendPerDrawBuffer.get(0); return b0?.srcRGB ?? s.blend.srcRGB; }
    case C.BLEND_SRC_ALPHA: { const b0 = s.blendPerDrawBuffer.get(0); return b0?.srcAlpha ?? s.blend.srcAlpha; }
    case C.BLEND_DST_RGB: { const b0 = s.blendPerDrawBuffer.get(0); return b0?.dstRGB ?? s.blend.dstRGB; }
    case C.BLEND_DST_ALPHA: { const b0 = s.blendPerDrawBuffer.get(0); return b0?.dstAlpha ?? s.blend.dstAlpha; }
    case C.BLEND_EQUATION_RGB: { const b0 = s.blendPerDrawBuffer.get(0); return b0?.eqRGB ?? s.blend.eqRGB; }
    case C.BLEND_EQUATION_ALPHA: { const b0 = s.blendPerDrawBuffer.get(0); return b0?.eqAlpha ?? s.blend.eqAlpha; }
    case C.DEPTH_CLEAR_VALUE: return s.clearDepth;
    case C.STENCIL_CLEAR_VALUE: return s.clearStencil;
    case C.LINE_WIDTH: return s.lineWidth;
    case C.POLYGON_OFFSET_FACTOR: return s.polygonOffset.factor;
    case C.POLYGON_OFFSET_UNITS: return s.polygonOffset.units;
    case C.SAMPLE_COVERAGE_VALUE: return s.sampleCoverage.value;
    case C.SAMPLE_COVERAGE_INVERT: return s.sampleCoverage.invert;
    case C.CULL_FACE_MODE: return s.cullFace;
    case C.FRONT_FACE: return s.frontFace;
    case C.GENERATE_MIPMAP_HINT: return s.hints.generateMipmap;
    case C.FRAGMENT_SHADER_DERIVATIVE_HINT:
      if (!v2 && !ctx._extensions.has('OES_standard_derivatives')) break;
      return s.hints.fragmentShaderDerivative;
    case C.DEPTH_FUNC: return s.depth.func;
    case C.DEPTH_WRITEMASK: return s.depth.mask;

    // ---- Stencil state (front + back) ----
    case C.STENCIL_FUNC: return s.stencil.front.func;
    case C.STENCIL_REF: return s.stencil.front.ref;
    case C.STENCIL_VALUE_MASK: return s.stencil.front.valueMask;
    case C.STENCIL_FAIL: return s.stencil.front.fail;
    case C.STENCIL_PASS_DEPTH_FAIL: return s.stencil.front.depthFail;
    case C.STENCIL_PASS_DEPTH_PASS: return s.stencil.front.depthPass;
    case C.STENCIL_WRITEMASK: return s.stencil.front.writeMask;
    case C.STENCIL_BACK_FUNC: return s.stencil.back.func;
    case C.STENCIL_BACK_REF: return s.stencil.back.ref;
    case C.STENCIL_BACK_VALUE_MASK: return s.stencil.back.valueMask;
    case C.STENCIL_BACK_FAIL: return s.stencil.back.fail;
    case C.STENCIL_BACK_PASS_DEPTH_FAIL: return s.stencil.back.depthFail;
    case C.STENCIL_BACK_PASS_DEPTH_PASS: return s.stencil.back.depthPass;
    case C.STENCIL_BACK_WRITEMASK: return s.stencil.back.writeMask;

    // ---- Object bindings ----
    case CURRENT_PROGRAM: return s.currentProgram;
    case C.ARRAY_BUFFER_BINDING: return s.arrayBuffer;
    case C.ELEMENT_ARRAY_BUFFER_BINDING: return s.vao.elementArrayBuffer;
    case C.FRAMEBUFFER_BINDING: return s.drawFramebuffer;
    case C.RENDERBUFFER_BINDING: return s.renderbuffer;
    case C.TEXTURE_BINDING_2D: return s.textureUnits[s.activeTexture].texture2D;
    case C.TEXTURE_BINDING_CUBE_MAP: return s.textureUnits[s.activeTexture].textureCube;
    case C.DRAW_FRAMEBUFFER_BINDING:
      if (!v2) break;
      return s.drawFramebuffer;
    case C.READ_FRAMEBUFFER_BINDING:
      if (!v2) break;
      return s.readFramebuffer;
    case C.VERTEX_ARRAY_BINDING:
      if (!v2) break;
      return s.vaoBinding;
    case C.SAMPLER_BINDING:
      if (!v2) break;
      return s.textureUnits[s.activeTexture].sampler;
    case C.TEXTURE_BINDING_3D:
      if (!v2) break;
      return s.textureUnits[s.activeTexture].texture3D;
    case C.TEXTURE_BINDING_2D_ARRAY:
      if (!v2) break;
      return s.textureUnits[s.activeTexture].texture2DArray;
    case C.PIXEL_PACK_BUFFER_BINDING:
      if (!v2) break;
      return s.pixelPackBuffer;
    case C.PIXEL_UNPACK_BUFFER_BINDING:
      if (!v2) break;
      return s.pixelUnpackBuffer;
    case C.UNIFORM_BUFFER_BINDING:
      if (!v2) break;
      // GENERIC binding point (GLES 3.0 §2.10.1) — distinct from the indexed
      // bindings: bindBuffer(UNIFORM_BUFFER, b) sets only this, while
      // bindBufferBase/Range with index 0 also update it. The old-model read
      // (`s.uniformBuffers[0]`) was wrong after bindBuffer vs bindBufferBase
      // splits (CTS uniform-buffers.html, switching-objects.html).
      return getGenericBufferBinding(ctx, C.UNIFORM_BUFFER);
    case C.COPY_READ_BUFFER_BINDING:
      if (!v2) break;
      return s.copyReadBuffer;
    case C.COPY_WRITE_BUFFER_BINDING:
      if (!v2) break;
      return s.copyWriteBuffer;
    case C.TRANSFORM_FEEDBACK_BINDING:
      if (!v2) break;
      return s.transformFeedback;
    case C.TRANSFORM_FEEDBACK_BUFFER_BINDING:
      if (!v2) break;
      // GENERIC binding point (GLES 3.0 §2.10.1; Khronos resolution — CTS
      // switching-objects.html "Generic binding is not changed when switching
      // TF object"): NOT part of the TF object state. bindBuffer(
      // TRANSFORM_FEEDBACK_BUFFER, b) sets only this point; bindBufferBase/
      // Range with index 0 also update it; bindTransformFeedback must NOT
      // change it. The old read (`s.transformFeedback._buffers[0]`) reported
      // the TF object's indexed state instead (switching-objects.html expects
      // null here after bindBuffer(TRANSFORM_FEEDBACK_BUFFER, null)).
      return getGenericBufferBinding(ctx, C.TRANSFORM_FEEDBACK_BUFFER);

    // ---- WebGL2 booleans / misc ----
    case C.TRANSFORM_FEEDBACK_ACTIVE:
      if (!v2) break;
      return activeTransformFeedback(ctx) !== null;
    case C.TRANSFORM_FEEDBACK_PAUSED: {
      if (!v2) break;
      const tf = activeTransformFeedback(ctx);
      return tf !== null && tf._paused;
    }
    case C.TRANSFORM_FEEDBACK_BUFFER_MODE:
      if (!v2) break;
      return s.currentProgram && s.currentProgram._tfBufferMode !== 0
        ? s.currentProgram._tfBufferMode
        : INTERLEAVED_ATTRIBS;
    case C.TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN:
      if (!v2) break;
      return s.transformFeedback ? s.transformFeedback._primitivesWritten : 0;
    case C.READ_BUFFER:
      if (!v2) break;
      // BACK for the default framebuffer, else the readBuffer() setting.
      return s.readFramebuffer === null ? BACK : s.readBuffer;

    // ---- Vectors (fresh arrays every call) ----
    case C.COLOR_CLEAR_VALUE: return new Float32Array(s.clearColor);
    case C.BLEND_COLOR: return new Float32Array(s.blend.color);
    case C.DEPTH_RANGE: return new Float32Array(s.depth.range);
    case C.SCISSOR_BOX: return new Int32Array([s.scissor.x, s.scissor.y, s.scissor.w, s.scissor.h]);
    case C.VIEWPORT: return new Int32Array([s.viewport.x, s.viewport.y, s.viewport.w, s.viewport.h]);
    case C.COLOR_WRITEMASK: { const m = s.colorMaskPerDrawBuffer.get(0) ?? s.colorMask; return [m[0], m[1], m[2], m[3]]; }
    case C.ALIASED_POINT_SIZE_RANGE:
      return new Float32Array([ALIASED_POINT_SIZE_RANGE[0], ALIASED_POINT_SIZE_RANGE[1]]);
    case C.ALIASED_LINE_WIDTH_RANGE:
      return new Float32Array([ALIASED_LINE_WIDTH_RANGE[0], ALIASED_LINE_WIDTH_RANGE[1]]);
    case C.MAX_VIEWPORT_DIMS:
      return new Int32Array([lim.MAX_VIEWPORT_DIMS[0], lim.MAX_VIEWPORT_DIMS[1]]);

    // ---- Pixel storage (getParameter mirrors pixelStorei) ----
    case C.UNPACK_ALIGNMENT: return s.pixelStore.unpack.alignment;
    case C.PACK_ALIGNMENT: return s.pixelStore.pack.alignment;
    case C.UNPACK_FLIP_Y_WEBGL: return s.pixelStore.unpack.flipY;
    case C.UNPACK_PREMULTIPLY_ALPHA_WEBGL: return s.pixelStore.unpack.premultiplyAlpha;
    case C.UNPACK_COLORSPACE_CONVERSION_WEBGL: return s.pixelStore.unpack.colorspaceConversion;
    case C.PACK_ROW_LENGTH:
      if (!v2) break;
      return s.pixelStore.pack.rowLength;
    case C.PACK_SKIP_PIXELS:
      if (!v2) break;
      return s.pixelStore.pack.skipPixels;
    case C.PACK_SKIP_ROWS:
      if (!v2) break;
      return s.pixelStore.pack.skipRows;
    case C.UNPACK_ROW_LENGTH:
      if (!v2) break;
      return s.pixelStore.unpack.rowLength;
    case C.UNPACK_IMAGE_HEIGHT:
      if (!v2) break;
      return s.pixelStore.unpack.imageHeight;
    case C.UNPACK_SKIP_PIXELS:
      if (!v2) break;
      return s.pixelStore.unpack.skipPixels;
    case C.UNPACK_SKIP_ROWS:
      if (!v2) break;
      return s.pixelStore.unpack.skipRows;
    case C.UNPACK_SKIP_IMAGES:
      if (!v2) break;
      return s.pixelStore.unpack.skipImages;

    // ---- Limits (WebGL1 + WebGL2-shared) ----
    case C.MAX_VERTEX_ATTRIBS: return lim.MAX_VERTEX_ATTRIBS;
    case C.MAX_VERTEX_UNIFORM_VECTORS: return lim.MAX_VERTEX_UNIFORM_VECTORS;
    case C.MAX_FRAGMENT_UNIFORM_VECTORS: return lim.MAX_FRAGMENT_UNIFORM_VECTORS;
    case C.MAX_VARYING_VECTORS: return lim.MAX_VARYING_VECTORS;
    case C.MAX_COMBINED_TEXTURE_IMAGE_UNITS: return lim.MAX_COMBINED_TEXTURE_IMAGE_UNITS;
    case C.MAX_VERTEX_TEXTURE_IMAGE_UNITS: return lim.MAX_VERTEX_TEXTURE_IMAGE_UNITS;
    case C.MAX_TEXTURE_IMAGE_UNITS: return lim.MAX_TEXTURE_IMAGE_UNITS;
    case C.MAX_TEXTURE_SIZE: return lim.MAX_TEXTURE_SIZE;
    case C.MAX_CUBE_MAP_TEXTURE_SIZE: return lim.MAX_CUBE_MAP_TEXTURE_SIZE;
    case C.MAX_RENDERBUFFER_SIZE: return lim.MAX_RENDERBUFFER_SIZE;

    // ---- Limits (WebGL2-only) ----
    case C.MAX_3D_TEXTURE_SIZE:
      if (!v2) break;
      return lim.MAX_3D_TEXTURE_SIZE;
    case C.MAX_ARRAY_TEXTURE_LAYERS:
      if (!v2) break;
      return lim.MAX_ARRAY_TEXTURE_LAYERS;
    case C.MAX_SAMPLES:
      if (!v2) break;
      return lim.MAX_SAMPLES;
    case C.MAX_DRAW_BUFFERS:
      if (!v2) {
        // WEBGL_draw_buffers exposes MAX_DRAW_BUFFERS_WEBGL on WebGL1.
        if (!ctx._extensions.has('WEBGL_draw_buffers')) break;
      }
      return lim.MAX_DRAW_BUFFERS;
    case C.MAX_COLOR_ATTACHMENTS:
      if (!v2) {
        // WEBGL_draw_buffers exposes MAX_COLOR_ATTACHMENTS_WEBGL on WebGL1.
        if (!ctx._extensions.has('WEBGL_draw_buffers')) break;
      }
      return lim.MAX_COLOR_ATTACHMENTS;
    case C.MAX_ELEMENT_INDEX:
      if (!v2) break;
      return lim.MAX_ELEMENT_INDEX;
    case C.MAX_UNIFORM_BUFFER_BINDINGS:
      if (!v2) break;
      return lim.MAX_UNIFORM_BUFFER_BINDINGS;
    case C.MAX_UNIFORM_BLOCK_SIZE:
      if (!v2) break;
      return lim.MAX_UNIFORM_BLOCK_SIZE;
    case C.MAX_VERTEX_UNIFORM_BLOCKS:
      if (!v2) break;
      return lim.MAX_VERTEX_UNIFORM_BLOCKS;
    case C.MAX_FRAGMENT_UNIFORM_BLOCKS:
      if (!v2) break;
      return lim.MAX_FRAGMENT_UNIFORM_BLOCKS;
    case C.MAX_COMBINED_UNIFORM_BLOCKS:
      if (!v2) break;
      return lim.MAX_COMBINED_UNIFORM_BLOCKS;
    case C.MAX_VERTEX_OUTPUT_COMPONENTS:
      if (!v2) break;
      return lim.MAX_VERTEX_OUTPUT_COMPONENTS;
    case C.MAX_FRAGMENT_INPUT_COMPONENTS:
      if (!v2) break;
      return lim.MAX_FRAGMENT_INPUT_COMPONENTS;
    case C.MAX_VERTEX_ATTRIB_STRIDE:
      if (!v2) break;
      return lim.MAX_VERTEX_ATTRIB_STRIDE;
    case C.MAX_VERTEX_ATTRIB_RELATIVE_OFFSET:
      if (!v2) break;
      return lim.MAX_VERTEX_ATTRIB_RELATIVE_OFFSET;
    case C.MAX_TEXTURE_LOD_BIAS:
      if (!v2) break;
      return lim.MAX_TEXTURE_LOD_BIAS;
    case C.MIN_PROGRAM_TEXEL_OFFSET:
      if (!v2) break;
      return lim.MIN_PROGRAM_TEXEL_OFFSET;
    case C.MAX_PROGRAM_TEXEL_OFFSET:
      if (!v2) break;
      return lim.MAX_PROGRAM_TEXEL_OFFSET;
    case C.MAX_SERVER_WAIT_TIMEOUT:
      if (!v2) break;
      return lim.MAX_SERVER_WAIT_TIMEOUT;
    case C.MAX_CLIENT_WAIT_TIMEOUT_WEBGL:
      if (!v2) break;
      return lim.MAX_CLIENT_WAIT_TIMEOUT_WEBGL;
    case C.MAX_ELEMENTS_VERTICES:
      if (!v2) break;
      return lim.MAX_ELEMENTS_VERTICES;
    case C.MAX_ELEMENTS_INDICES:
      if (!v2) break;
      return lim.MAX_ELEMENTS_INDICES;
    case C.UNIFORM_BUFFER_OFFSET_ALIGNMENT:
      if (!v2) break;
      return lim.UNIFORM_BUFFER_OFFSET_ALIGNMENT;
    case C.MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS:
      if (!v2) break;
      return lim.MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS;
    case C.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS:
      if (!v2) break;
      return lim.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS;
    case C.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS:
      if (!v2) break;
      return lim.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS;
    case C.MAX_VARYING_COMPONENTS:
      if (!v2) break;
      return lim.MAX_VARYING_VECTORS * 4;
    case C.MAX_VERTEX_UNIFORM_COMPONENTS:
      if (!v2) break;
      return lim.MAX_VERTEX_UNIFORM_VECTORS * 4;
    case C.MAX_FRAGMENT_UNIFORM_COMPONENTS:
      if (!v2) break;
      return lim.MAX_FRAGMENT_UNIFORM_VECTORS * 4;
    case C.MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS:
      if (!v2) break;
      // GLES 3.0 §6.2 invariant: must be ≥ blocks×blockSize/4 + components.
      return Math.max(
        lim.MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS,
        lim.MAX_VERTEX_UNIFORM_BLOCKS * (lim.MAX_UNIFORM_BLOCK_SIZE / 4) +
          lim.MAX_VERTEX_UNIFORM_VECTORS * 4,
      );
    case C.MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS:
      if (!v2) break;
      return Math.max(
        lim.MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS,
        lim.MAX_FRAGMENT_UNIFORM_BLOCKS * (lim.MAX_UNIFORM_BLOCK_SIZE / 4) +
          lim.MAX_FRAGMENT_UNIFORM_VECTORS * 4,
      );

    // ---- Misc state ----
    // SHADER_COMPILER / SHADER_BINARY_FORMATS / NUM_SHADER_BINARY_FORMATS are
    // NOT queryable via getParameter in WebGL: the spec ("No Shader Binaries":
    // "querying shader binary formats and the availability of a shader
    // compiler via getParameter is not supported in the WebGL API") and CTS
    // gl-enum-tests (BOTH WebGL1 and WebGL2 — the pnames are in the shared
    // INVALID_ENUM list) require INVALID_ENUM here, so they intentionally have
    // no case and fall through to the default below. The context constants
    // themselves remain installed (constants-and-properties*).
    case C.COMPRESSED_TEXTURE_FORMATS: return new Uint32Array(0); // no compressed formats
    case C.SUBPIXEL_BITS: return 8;
    case C.IMPLEMENTATION_COLOR_READ_FORMAT:
    case C.IMPLEMENTATION_COLOR_READ_TYPE: {
      if (drawFramebufferIncomplete(ctx)) {
        ctx._errors.push(C.INVALID_OPERATION);
        return null;
      }
      const pair = implementationColorReadPair(ctx);
      return pname === C.IMPLEMENTATION_COLOR_READ_FORMAT ? pair.format : pair.type;
    }

    // ---- FBO-dependent bits (draw framebuffer) ----
    case C.SAMPLE_BUFFERS: return 0; // single-sampled drawing buffer, always
    case C.SAMPLES: return 0;
    case C.RED_BITS:
    case C.GREEN_BITS:
    case C.BLUE_BITS:
    case C.ALPHA_BITS:
    case C.DEPTH_BITS:
    case C.STENCIL_BITS: {
      const b = drawFramebufferBits(ctx);
      switch (pname) {
        case C.RED_BITS: return b.r;
        case C.GREEN_BITS: return b.g;
        case C.BLUE_BITS: return b.b;
        case C.ALPHA_BITS: return b.a;
        case C.DEPTH_BITS: return b.depth;
        default: return b.stencil;
      }
    }

    // ---- Extension-gated pnames ----
    case 0x84ff /* MAX_TEXTURE_MAX_ANISOTROPY_EXT */:
      if (!ctx._extensions.has('EXT_texture_filter_anisotropic')) break;
      return lim.MAX_TEXTURE_MAX_ANISOTROPY_EXT;
    case 0x9245 /* UNMASKED_VENDOR_WEBGL */:
      if (!ctx._extensions.has('WEBGL_debug_renderer_info')) break;
      return 'Software Renderer';
    case 0x9246 /* UNMASKED_RENDERER_WEBGL */:
      if (!ctx._extensions.has('WEBGL_debug_renderer_info')) break;
      return 'Software Renderer (JS)';
    case 0x88fc /* MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL (WEBGL_blend_func_extended) */:
      if (!ctx._extensions.has('WEBGL_blend_func_extended')) break;
      return lim.MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL;
    case 0x0d32 /* MAX_CLIP_DISTANCES_WEBGL (WEBGL_clip_cull_distance) */:
      if (!v2 || !ctx._extensions.has('WEBGL_clip_cull_distance')) break;
      return lim.MAX_CLIP_DISTANCES_WEBGL;
    case 0x82f9 /* MAX_CULL_DISTANCES_WEBGL */:
      if (!v2 || !ctx._extensions.has('WEBGL_clip_cull_distance')) break;
      return lim.MAX_CULL_DISTANCES_WEBGL;
    case 0x82fa /* MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL */:
      if (!v2 || !ctx._extensions.has('WEBGL_clip_cull_distance')) break;
      return lim.MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL;
    case 0x935c /* CLIP_ORIGIN_EXT (EXT_clip_control) */:
      if (!ctx._extensions.has('EXT_clip_control')) break;
      return getClipControl(ctx).origin;
    case 0x935d /* CLIP_DEPTH_MODE_EXT (EXT_clip_control) */:
      if (!ctx._extensions.has('EXT_clip_control')) break;
      return getClipControl(ctx).depth;
    case 0x3000 /* CLIP_DISTANCE0_WEBGL .. CLIP_DISTANCE7_WEBGL (WEBGL_clip_cull_distance) */:
    case 0x3001:
    case 0x3002:
    case 0x3003:
    case 0x3004:
    case 0x3005:
    case 0x3006:
    case 0x3007:
      if (!ctx._extensions.has('WEBGL_clip_cull_distance')) break;
      return isClipDistanceEnabled(ctx, pname - 0x3000);

    default:
      break;
  }
  ctx._errors.push(C.INVALID_ENUM);
  return null;
}

/**
 * The transform feedback object that is currently ACTIVE (or null): the BOUND
 * TF object when it is active. Mirrors api/webgl2.ts's private `activeTF` —
 * begin/end/pause/resume there operate on the bound object, or on a module-
 * private default TF (WeakMap) when none is bound (the default TF is kept OUT
 * of state.transformFeedback so getParameter(TRANSFORM_FEEDBACK_BINDING) stays
 * null per CTS). The default TF object is not reachable from this module
 * (api/webgl2.ts does not export an accessor); the bound-object leg is
 * authoritative for the CTS, which only checks the initial inactive state
 * (conformance2/state/gl-get-calls.html TRANSFORM_FEEDBACK_ACTIVE/PAUSED false).
 * TODO(integration): have api/webgl2.ts export `activeTF(ctx)` and use it here
 * so a begun default TF reports active/paused too.
 */
function activeTransformFeedback(ctx: WebGLRenderingContext): WebGLTransformFeedback | null {
  const bound = ctx._state.transformFeedback;
  if (bound && bound._active) return bound;
  return null; // default-TF leg — see comment above
}

// ---------------------------------------------------------------------------
// Framebuffer-dependent helpers
// ---------------------------------------------------------------------------

interface Bits {
  r: number; g: number; b: number; a: number; depth: number; stencil: number;
}

/** True when the bound DRAW framebuffer is a non-complete FBO (stub-safe). */
function drawFramebufferIncomplete(ctx: WebGLRenderingContext): boolean {
  if (ctx._state.drawFramebuffer === null) return false;
  try {
    return resolveFramebufferTarget(ctx) === null;
  } catch {
    return true; // framebuffer-util stub (parallel agent) — cannot resolve yet
  }
}

/**
 * (format, type) pair for IMPLEMENTATION_COLOR_READ_FORMAT/TYPE — the pair
 * readPixels must accept for the bound READ buffer's color attachment (CTS
 * read-pixels-from-fbo-test.html queries it and reads with it, expecting
 * NO_ERROR + correct colors). Resolution mirrors readPixelsComboOK
 * (api/draw.ts): the READ framebuffer's READ_BUFFER attachment.
 *
 *  - Default framebuffer / no attachment / unknown format → RGBA/UNSIGNED_BYTE
 *    (the universally-accepted pair for normalized attachments).
 *  - Normalized formats (R8..SRGB8_ALPHA8, RGB565, RGBA4, RGB5_A1, RGB10_A2):
 *    RGBA/UNSIGNED_BYTE — readComboOK's universalRGBA rule accepts it for
 *    EVERY normalized color attachment.
 *  - Integer formats: *_INTEGER by channel count (RED/RG/RGB/RGBA_INTEGER)
 *    with the channel-width-matched type (1 → BYTE|UNSIGNED_BYTE,
 *    2 → SHORT|UNSIGNED_SHORT, 4 → INT|UNSIGNED_INT; signed → signed type).
 *    Every such pair is accepted by readComboOK's per-format width-matched
 *    rule (RGB10_A2UI is packed 10/10/10/2 — only the universal
 *    RGBA_INTEGER/UNSIGNED_INT rule accepts it, so it is special-cased).
 *  - Float formats: RGBA/FLOAT only when the color-buffer-float extension is
 *    enabled (readComboOK gates it); otherwise RGBA/UNSIGNED_BYTE (the CTS
 *    float cases are TODO/not graded).
 */
function implementationColorReadPair(ctx: WebGLRenderingContext): { format: number; type: number } {
  const s = ctx._state;
  const fbo = s.readFramebuffer;
  if (fbo === null) return { format: C.RGBA, type: C.UNSIGNED_BYTE };
  const rb = s.version === 2 ? s.readBuffer : C.COLOR_ATTACHMENT0;
  const att = fbo._attachments.get(rb);
  if (!att) return { format: C.RGBA, type: C.UNSIGNED_BYTE };
  const internalFormat = att.type === 'renderbuffer'
    ? att.renderbuffer._internalformat
    : (att.texture._image?.internalFormat ?? 0);
  let info = att.type === 'renderbuffer'
    ? (att.renderbuffer._surface?.info ?? null)
    : (att.texture._image?.info ?? null);
  if (!info) info = getFormat(internalFormat);
  if (info?.isInteger) {
    if (internalFormat === C.RGB10_A2UI) return { format: C.RGBA_INTEGER, type: C.UNSIGNED_INT };
    const fmt = info.components >= 4 ? C.RGBA_INTEGER :
      info.components === 3 ? C.RGB_INTEGER :
      info.components === 2 ? C.RG_INTEGER : C.RED_INTEGER;
    const perComp = info.bytesPerPixel / info.components; // 1 | 2 | 4
    const type = perComp <= 1
      ? (info.isSigned ? C.BYTE : C.UNSIGNED_BYTE)
      : perComp <= 2
        ? (info.isSigned ? C.SHORT : C.UNSIGNED_SHORT)
        : (info.isSigned ? C.INT : C.UNSIGNED_INT);
    return { format: fmt, type };
  }
  if (info?.isFloat) {
    if ((internalFormat === C.R16F || internalFormat === C.RG16F || internalFormat === C.RGBA16F) &&
        (ctx._extensions.has('EXT_color_buffer_float') || ctx._extensions.has('EXT_color_buffer_half_float'))) {
      return { format: C.RGBA, type: C.FLOAT };
    }
    if ((internalFormat === C.R32F || internalFormat === C.RG32F || internalFormat === C.RGBA32F) &&
        ctx._extensions.has('EXT_color_buffer_float')) {
      return { format: C.RGBA, type: C.FLOAT };
    }
    return { format: C.RGBA, type: C.UNSIGNED_BYTE };
  }
  return { format: C.RGBA, type: C.UNSIGNED_BYTE };
}

/**
 * Channel bit depths of the draw framebuffer's attachments. The default
 * framebuffer is attribute-driven (per WebGL spec the drawing buffer's alpha
 * presence follows `alpha`, depth/stencil presence follows `depth`/`stencil`);
 * an FBO reports its attachments' formats (all zero when incomplete).
 */
function drawFramebufferBits(ctx: WebGLRenderingContext): Bits {
  const s = ctx._state;
  if (s.drawFramebuffer === null) {
    return {
      r: 8, g: 8, b: 8,
      a: ctx._attrs.alpha ? 8 : 0,
      depth: ctx._attrs.depth ? (ctx._version === 2 ? 24 : 16) : 0,
      stencil: ctx._attrs.stencil ? 8 : 0,
    };
  }
  let fb: FramebufferTarget | null = null;
  try {
    fb = resolveFramebufferTarget(ctx);
  } catch {
    fb = null; // stub (framebuffers agent) — incomplete
  }
  if (!fb) return { r: 0, g: 0, b: 0, a: 0, depth: 0, stencil: 0 };
  const color = fb.color[0];
  const cb = color ? colorChannelBits(color.format, color.info) : [0, 0, 0, 0];
  return {
    r: cb[0], g: cb[1], b: cb[2], a: cb[3],
    depth: fb.depth ? depthBits(fb.depth.format) : 0,
    stencil: fb.stencil ? stencilBits(fb.stencil.format) : 0,
  };
}

/** Per-channel bit depths [r,g,b,a] of a color attachment's internal format. */
function colorChannelBits(format: GLenum, info: PixelFormatInfo): [number, number, number, number] {
  switch (format) {
    case 0x1906 /* ALPHA */: return [0, 0, 0, 8];
    case 0x1909 /* LUMINANCE */: return [8, 8, 8, 0];
    case 0x190a /* LUMINANCE_ALPHA */: return [8, 8, 8, 8];
    case 0x8d62 /* RGB565 */: return [5, 6, 5, 0];
    case 0x8056 /* RGBA4 */: return [4, 4, 4, 4];
    case 0x8057 /* RGB5_A1 */: return [5, 5, 5, 1];
    case 0x8059 /* RGB10_A2 */:
    case 0x906f /* RGB10_A2UI */: return [10, 10, 10, 2];
    case 0x8c3a /* R11F_G11F_B10F */: return [11, 11, 10, 0];
    default: {
      if (info.isFloat || info.isInteger) {
        const bits = (info.bytesPerPixel * 8) / info.components;
        return [bits, bits, bits, bits];
      }
      // Normalized formats: uniform 8-bit channels unless component count says otherwise.
      switch (info.components) {
        case 1: return [8, 0, 0, 0];
        case 2: return [8, 8, 0, 0];
        case 3: return [8, 8, 8, 0];
        case 4: return [8, 8, 8, 8];
        default: return [0, 0, 0, 0];
      }
    }
  }
}

/** Bit depth of a depth (or depth-stencil) attachment format. */
function depthBits(format: GLenum): number {
  switch (format) {
    case 0x81a5 /* DEPTH_COMPONENT16 */: return 16;
    case 0x81a6 /* DEPTH_COMPONENT24 */: return 24;
    case 0x8cac /* DEPTH_COMPONENT32F */: return 32;
    case 0x88f0 /* DEPTH24_STENCIL8 */: return 24;
    case 0x8cad /* DEPTH32F_STENCIL8 */: return 32;
    default: return 16; // unsized DEPTH_COMPONENT/DEPTH_STENCIL (WebGL1)
  }
}

/** Bit depth of a stencil (or depth-stencil) attachment format. */
function stencilBits(format: GLenum): number {
  switch (format) {
    case 0x8d48 /* STENCIL_INDEX8 */: return 8;
    case 0x88f0 /* DEPTH24_STENCIL8 */: return 8;
    case 0x8cad /* DEPTH32F_STENCIL8 */: return 8;
    default: return 8;
  }
}

/**
 * Single source of truth for the string-valued pnames shared by getString
 * (api/context.ts) and getParameter. Returns null for every other pname —
 * notably EXTENSIONS (0x1f03), which is only legal via getString (the WebGL
 * spec does not allow getParameter(EXTENSIONS)).
 */
export function stringValueForPname(ctx: WebGLRenderingContext, pname: number): string | null {
  switch (pname) {
    case 0x1f02 /* VERSION */:
      return ctx._version === 2
        ? 'WebGL 2.0 (Software Renderer)'
        : 'WebGL 1.0 (Software Renderer)';
    case 0x8b8c /* SHADING_LANGUAGE_VERSION */:
      return ctx._version === 2
        ? 'WebGL GLSL ES 3.00 (Software)'
        : 'WebGL GLSL ES 1.00 (Software)';
    case 0x1f00 /* VENDOR */:
      return 'Software Renderer';
    case 0x1f01 /* RENDERER */:
      return 'Software Renderer (JS)';
    default:
      return null;
  }
}
