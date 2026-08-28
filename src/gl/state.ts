/**
 * src/gl/state.ts — the GL state container (plain mutable data, NO logic).
 *
 * One `State` object per context, created by `createDefaultState(version)` at
 * context creation and reset (re-created) on context restore. All API setters
 * write here; the draw pipeline reads it into a `DrawCall` (contract §2) and
 * hands it to the rasterizer. `getParameter` reads it (see getters.ts).
 *
 * Vertex attrib array state lives inside the VAO (attribs + ELEMENT_ARRAY_BUFFER
 * binding are VAO state per spec; ARRAY_BUFFER binding is context state).
 */

import type { GLenum, GLint, GLintptr, GLsizei, GLsizeiptr, GLuint } from './types';
import type {
  WebGLBuffer,
  WebGLFramebuffer,
  WebGLProgram,
  WebGLQuery,
  WebGLRenderbuffer,
  WebGLSampler,
  WebGLTransformFeedback,
  WebGLVertexArrayObject,
} from './objects';

// ---------------------------------------------------------------------------
// Limits — every MAX_* the context reports. MUST be consistent across
// getParameter, validation (index bounds) and allocation (uniform stores).
// Generous values chosen (software renderer, no GPU constraints) while keeping
// the reported numbers ≥ spec minimums and realistic for tests.
// ---------------------------------------------------------------------------
export interface Limits {
  // WebGL1
  MAX_VERTEX_ATTRIBS: number; // 16
  MAX_VERTEX_UNIFORM_VECTORS: number; // 4096
  MAX_FRAGMENT_UNIFORM_VECTORS: number; // 4096
  MAX_VARYING_VECTORS: number; // 64
  MAX_COMBINED_TEXTURE_IMAGE_UNITS: number; // 32
  MAX_VERTEX_TEXTURE_IMAGE_UNITS: number; // 16
  MAX_TEXTURE_IMAGE_UNITS: number; // 16
  MAX_TEXTURE_SIZE: number; // 8192
  MAX_CUBE_MAP_TEXTURE_SIZE: number; // 8192
  MAX_RENDERBUFFER_SIZE: number; // 8192
  MAX_VIEWPORT_DIMS: [number, number]; // [32767, 32767]
  ALIASED_POINT_SIZE_RANGE: [number, number]; // [1, 1024]
  ALIASED_LINE_WIDTH_RANGE: [number, number]; // [1, 1] — only 1.0 required
  MAX_TEXTURE_MAX_ANISOTROPY_EXT: number; // 16 (EXT_texture_filter_anisotropic)
  // WebGL2
  MAX_3D_TEXTURE_SIZE: number; // 2048
  MAX_ARRAY_TEXTURE_LAYERS: number; // 2048
  MAX_SAMPLES: number; // 4 (software MSAA resolve)
  MAX_DRAW_BUFFERS: number; // 8
  MAX_COLOR_ATTACHMENTS: number; // 8
  MAX_ELEMENT_INDEX: number; // 0xFFFFFFFF
  MAX_UNIFORM_BUFFER_BINDINGS: number; // 72
  MAX_UNIFORM_BLOCK_SIZE: number; // 65536
  MAX_VERTEX_UNIFORM_BLOCKS: number; // 12
  MAX_FRAGMENT_UNIFORM_BLOCKS: number; // 12
  MAX_COMBINED_UNIFORM_BLOCKS: number; // 36
  MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS: number; // 262144 (≥ 12 blocks × 65536/4 + 4096×4, ES 3.0 minimum)
  MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS: number; // 262144
  UNIFORM_BUFFER_OFFSET_ALIGNMENT: number; // 256
  MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: number; // 4
  MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: number; // 4
  MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: number; // 64
  MAX_VERTEX_OUTPUT_COMPONENTS: number; // 128
  MAX_FRAGMENT_INPUT_COMPONENTS: number; // 128
  MAX_VERTEX_ATTRIB_STRIDE: number; // 2048
  MAX_VERTEX_ATTRIB_RELATIVE_OFFSET: number; // 2047
  MIN_PROGRAM_TEXEL_OFFSET: number; // -8
  MAX_PROGRAM_TEXEL_OFFSET: number; // 7
  MAX_TEXTURE_LOD_BIAS: number; // 8
  MAX_SERVER_WAIT_TIMEOUT: number; // 0
  MAX_CLIENT_WAIT_TIMEOUT_WEBGL: number; // 0 (spec: must be 0)
  MAX_ELEMENTS_VERTICES: number; // 0x7FFFFFFF
  MAX_ELEMENTS_INDICES: number; // 0x7FFFFFFF
  // WEBGL_clip_cull_distance (when extension enabled)
  MAX_CLIP_DISTANCES_WEBGL: number; // 8
  MAX_CULL_DISTANCES_WEBGL: number; // 8
  MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL: number; // 16
  // WEBGL_blend_func_extended
  MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL: number; // 1
}

export function defaultLimits(): Limits {
  return {
    MAX_VERTEX_ATTRIBS: 16,
    MAX_VERTEX_UNIFORM_VECTORS: 4096,
    MAX_FRAGMENT_UNIFORM_VECTORS: 4096,
    MAX_VARYING_VECTORS: 64,
    MAX_COMBINED_TEXTURE_IMAGE_UNITS: 32,
    MAX_VERTEX_TEXTURE_IMAGE_UNITS: 16,
    MAX_TEXTURE_IMAGE_UNITS: 16,
    MAX_TEXTURE_SIZE: 8192,
    MAX_CUBE_MAP_TEXTURE_SIZE: 8192,
    MAX_RENDERBUFFER_SIZE: 8192,
    MAX_VIEWPORT_DIMS: [32767, 32767],
    ALIASED_POINT_SIZE_RANGE: [1, 1024],
    ALIASED_LINE_WIDTH_RANGE: [1, 1],
    MAX_TEXTURE_MAX_ANISOTROPY_EXT: 16,
    MAX_3D_TEXTURE_SIZE: 2048,
    MAX_ARRAY_TEXTURE_LAYERS: 2048,
    MAX_SAMPLES: 4,
    MAX_DRAW_BUFFERS: 8,
    MAX_COLOR_ATTACHMENTS: 8,
    MAX_ELEMENT_INDEX: 0xffffffff,
    MAX_UNIFORM_BUFFER_BINDINGS: 72,
    MAX_UNIFORM_BLOCK_SIZE: 65536,
    MAX_VERTEX_UNIFORM_BLOCKS: 12,
    MAX_FRAGMENT_UNIFORM_BLOCKS: 12,
    MAX_COMBINED_UNIFORM_BLOCKS: 36,
    MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS: 262144,
    MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS: 262144,
    UNIFORM_BUFFER_OFFSET_ALIGNMENT: 256,
    MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: 4,
    MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: 4,
    MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: 64,
    MAX_VERTEX_OUTPUT_COMPONENTS: 128,
    MAX_FRAGMENT_INPUT_COMPONENTS: 128,
    MAX_VERTEX_ATTRIB_STRIDE: 2048,
    MAX_VERTEX_ATTRIB_RELATIVE_OFFSET: 2047,
    MIN_PROGRAM_TEXEL_OFFSET: -8,
    MAX_PROGRAM_TEXEL_OFFSET: 7,
    MAX_TEXTURE_LOD_BIAS: 8,
    MAX_SERVER_WAIT_TIMEOUT: 0,
    MAX_CLIENT_WAIT_TIMEOUT_WEBGL: 0,
    MAX_ELEMENTS_VERTICES: 0x7fffffff,
    MAX_ELEMENTS_INDICES: 0x7fffffff,
    MAX_CLIP_DISTANCES_WEBGL: 8,
    MAX_CULL_DISTANCES_WEBGL: 8,
    MAX_COMBINED_CLIP_AND_CULL_DISTANCES_WEBGL: 16,
    MAX_DUAL_SOURCE_DRAW_BUFFERS_WEBGL: 1,
  };
}

// ---------------------------------------------------------------------------
// Vertex attrib array state (per VAO)
// ---------------------------------------------------------------------------
export interface VertexAttribState {
  enabled: boolean;
  size: GLint; // 1..4
  type: GLenum; // BYTE|UNSIGNED_BYTE|SHORT|UNSIGNED_SHORT|FLOAT (+INT/UNSIGNED_INT via vertexAttribIPointer in WebGL2)
  normalized: boolean;
  integer: boolean; // vertexAttribIPointer (WebGL2)
  stride: GLsizei;
  offset: GLintptr;
  divisor: GLuint; // ANGLE_instanced_arrays / WebGL2 vertexAttribDivisor
  buffer: WebGLBuffer | null;
  /** Generic (constant) vertex attrib values — set by vertexAttrib{1,2,3,4}f(v) */
  constantF: Float32Array; // length 4
  constantI: Int32Array; // length 4 (vertexAttribI4i)
  constantUI: Uint32Array; // length 4 (vertexAttribI4ui)
  /**
   * Base type of the CURRENT generic (constant) value — follows the most
   * recent setter: vertexAttrib{1..4}f(v) → 'f', vertexAttribI4{i,iv} → 'i',
   * vertexAttribI4{ui,uiv} → 'ui'. Default 'f' (spec generic attribute
   * default). Consumed by the draw-time attribute type-mismatch check
   * (GLES 3.0 §2.11.6, CTS attrib-type-match.html): a disabled array's generic
   * value type must match the active shader input's float/int/uint base type.
   */
  genericKind: 'f' | 'i' | 'ui';
}

export function defaultVertexAttrib(): VertexAttribState {
  return {
    enabled: false,
    size: 4,
    type: 0x1406, // FLOAT
    normalized: false,
    integer: false,
    stride: 0,
    offset: 0,
    divisor: 0,
    buffer: null,
    constantF: new Float32Array([0, 0, 0, 1]), // spec default (0,0,0,1)
    constantI: new Int32Array([0, 0, 0, 1]),
    constantUI: new Uint32Array([0, 0, 0, 1]),
    genericKind: 'f',
  };
}

/** VAO contents: per-attribute state + element buffer (ELEMENT_ARRAY_BUFFER is VAO state). */
export interface VAOState {
  attribs: VertexAttribState[];
  elementArrayBuffer: WebGLBuffer | null;
}

export function defaultVAOState(numAttribs: number): VAOState {
  const attribs: VertexAttribState[] = new Array(numAttribs);
  for (let i = 0; i < numAttribs; i++) attribs[i] = defaultVertexAttrib();
  return { attribs, elementArrayBuffer: null };
}

// ---------------------------------------------------------------------------
// Sub-state blocks
// ---------------------------------------------------------------------------
export interface BlendState {
  srcRGB: GLenum;
  dstRGB: GLenum;
  srcAlpha: GLenum;
  dstAlpha: GLenum;
  eqRGB: GLenum;
  eqAlpha: GLenum;
  color: [number, number, number, number];
}

export interface StencilState {
  func: GLenum;
  ref: GLint;
  valueMask: GLuint;
  writeMask: GLuint;
  fail: GLenum;
  depthFail: GLenum;
  depthPass: GLenum;
}

export interface PixelStoreUnpack {
  alignment: number; // 1|2|4|8, default 4
  flipY: boolean; // UNPACK_FLIP_Y_WEBGL
  premultiplyAlpha: boolean; // UNPACK_PREMULTIPLY_ALPHA_WEBGL
  colorspaceConversion: GLenum; // UNPACK_COLORSPACE_CONVERSION_WEBGL (BROWSER_DEFAULT_WEBGL|NONE)
  /** gl.unpackColorSpace ('srgb' | 'display-p3'); storage for the IDL accessor. */
  unpackColorSpace: string;
  rowLength: number; // UNPACK_ROW_LENGTH (WebGL2)
  skipRows: number; // UNPACK_SKIP_ROWS (WebGL2)
  skipPixels: number; // UNPACK_SKIP_PIXELS (WebGL2)
  imageHeight: number; // UNPACK_IMAGE_HEIGHT (WebGL2)
  skipImages: number; // UNPACK_SKIP_IMAGES (WebGL2)
}

export interface PixelStorePack {
  alignment: number; // 1|2|4|8, default 4
  rowLength: number; // PACK_ROW_LENGTH (WebGL2)
  skipRows: number; // PACK_SKIP_ROWS (WebGL2)
  skipPixels: number; // PACK_SKIP_PIXELS (WebGL2)
}

/** Per-texture-unit bindings (texture targets + sampler). */
export interface TextureUnitState {
  texture2D: WebGLTexture | null;
  textureCube: WebGLTexture | null;
  texture3D: WebGLTexture | null; // WebGL2
  texture2DArray: WebGLTexture | null; // WebGL2
  texture2DMultisample: WebGLTexture | null; // WebGL2
  sampler: WebGLSampler | null; // WebGL2
}

// ---------------------------------------------------------------------------
// The full state
// ---------------------------------------------------------------------------
export interface State {
  version: 1 | 2;

  // Capabilities (enable/disable/isEnabled/getParameter)
  caps: {
    BLEND: boolean;
    CULL_FACE: boolean;
    DEPTH_TEST: boolean;
    DITHER: boolean;
    POLYGON_OFFSET_FILL: boolean;
    SAMPLE_ALPHA_TO_COVERAGE: boolean;
    SAMPLE_COVERAGE: boolean;
    SCISSOR_TEST: boolean;
    STENCIL_TEST: boolean;
    RASTERIZER_DISCARD: boolean; // WebGL2
  };

  blend: BlendState;
  depth: { func: GLenum; mask: boolean; range: [number, number] };
  stencil: { front: StencilState; back: StencilState };
  clearColor: [number, number, number, number];
  clearDepth: number;
  clearStencil: number;
  colorMask: [boolean, boolean, boolean, boolean];
  /** Per-drawbuffer color masks (WebGL2 + OES_draw_buffers_indexed); base masks in colorMask. */
  colorMaskPerDrawBuffer: Map<number, [boolean, boolean, boolean, boolean]>;
  /**
   * Per-drawbuffer blend state (OES_draw_buffers_indexed). The raster DrawCall
   * carries a single blend state — the draw pipeline uses drawbuffer 0's entry
   * when this map is non-empty (documented known gap for per-drawbuffer blend).
   */
  blendPerDrawBuffer: Map<number, { srcRGB: GLenum; dstRGB: GLenum; srcAlpha: GLenum; dstAlpha: GLenum; eqRGB: GLenum; eqAlpha: GLenum }>;

  viewport: { x: GLint; y: GLint; w: GLsizei; h: GLsizei };
  scissor: { x: GLint; y: GLint; w: GLsizei; h: GLsizei };
  cullFace: GLenum;
  frontFace: GLenum;
  polygonOffset: { factor: number; units: number };
  sampleCoverage: { value: number; invert: boolean };
  lineWidth: number;
  hints: { generateMipmap: GLenum; fragmentShaderDerivative: GLenum };

  activeTexture: number; // texture unit index
  pixelStore: { unpack: PixelStoreUnpack; pack: PixelStorePack };
  textureUnits: TextureUnitState[];

  // Object bindings (context state)
  currentProgram: WebGLProgram | null;
  arrayBuffer: WebGLBuffer | null; // ARRAY_BUFFER binding (NOT part of VAO)
  vaoBinding: WebGLVertexArrayObject | null;
  vao: VAOState; // the CURRENT VAO's contents (default VAO or bound object's)
  drawFramebuffer: WebGLFramebuffer | null; // null = default framebuffer
  readFramebuffer: WebGLFramebuffer | null;
  renderbuffer: WebGLRenderbuffer | null;

  // WebGL2 bindings
  uniformBuffers: (WebGLBuffer | null)[]; // indexed by binding point
  uniformBufferRanges: { offset: GLintptr; size: GLsizeiptr }[];
  pixelPackBuffer: WebGLBuffer | null;
  pixelUnpackBuffer: WebGLBuffer | null;
  copyReadBuffer: WebGLBuffer | null;
  copyWriteBuffer: WebGLBuffer | null;
  transformFeedback: WebGLTransformFeedback | null;
  activeQueries: { ANY_SAMPLES_PASSED: WebGLQuery | null; ANY_SAMPLES_PASSED_CONSERVATIVE: WebGLQuery | null };
  drawBuffers: GLenum[]; // per drawbuffer: COLOR_ATTACHMENTn or NONE
  readBuffer: GLenum; // COLOR_ATTACHMENT0..n | NONE (WebGL2)

  limits: Limits;
}

export function createDefaultState(version: 1 | 2): State {
  const limits = defaultLimits();
  const numUnits = limits.MAX_COMBINED_TEXTURE_IMAGE_UNITS;
  const textureUnits: TextureUnitState[] = new Array(numUnits);
  for (let i = 0; i < numUnits; i++) {
    textureUnits[i] = {
      texture2D: null,
      textureCube: null,
      texture3D: null,
      texture2DArray: null,
      texture2DMultisample: null,
      sampler: null,
    };
  }
  const numUbos = version === 2 ? limits.MAX_UNIFORM_BUFFER_BINDINGS : 0;
  const uniformBuffers: (WebGLBuffer | null)[] = new Array(numUbos).fill(null);
  const uniformBufferRanges = new Array(numUbos).fill({ offset: 0, size: 0 });

  return {
    version,
    caps: {
      BLEND: false,
      CULL_FACE: false,
      DEPTH_TEST: false,
      DITHER: true, // GL default: dither enabled
      POLYGON_OFFSET_FILL: false,
      SAMPLE_ALPHA_TO_COVERAGE: false,
      SAMPLE_COVERAGE: false,
      SCISSOR_TEST: false,
      STENCIL_TEST: false,
      RASTERIZER_DISCARD: false,
    },
    blend: {
      srcRGB: 1, // ONE
      dstRGB: 0, // ZERO
      srcAlpha: 1,
      dstAlpha: 0,
      eqRGB: 0x8006, // FUNC_ADD
      eqAlpha: 0x8006,
      color: [0, 0, 0, 0],
    },
    depth: { func: 0x0201 /* LESS */, mask: true, range: [0, 1] },
    stencil: {
      front: { func: 0x0207 /* ALWAYS */, ref: 0, valueMask: 0xffffffff, writeMask: 0xffffffff, fail: 0x1e00 /* KEEP */, depthFail: 0x1e00, depthPass: 0x1e00 },
      back: { func: 0x0207, ref: 0, valueMask: 0xffffffff, writeMask: 0xffffffff, fail: 0x1e00, depthFail: 0x1e00, depthPass: 0x1e00 },
    },
    clearColor: [0, 0, 0, 0],
    clearDepth: 1,
    clearStencil: 0,
    colorMask: [true, true, true, true],
    colorMaskPerDrawBuffer: new Map(),
    blendPerDrawBuffer: new Map(),
    viewport: { x: 0, y: 0, w: 0, h: 0 }, // sized at context creation
    scissor: { x: 0, y: 0, w: 0, h: 0 },
    cullFace: 0x0405 /* BACK */,
    frontFace: 0x0901 /* CCW */,
    polygonOffset: { factor: 0, units: 0 },
    sampleCoverage: { value: 1, invert: false },
    lineWidth: 1,
    hints: { generateMipmap: 0x1100 /* DONT_CARE */, fragmentShaderDerivative: 0x1100 },
    activeTexture: 0,
    pixelStore: {
      unpack: {
        alignment: 4,
        flipY: false,
        premultiplyAlpha: false,
        colorspaceConversion: 0x9244 /* BROWSER_DEFAULT_WEBGL */,
        unpackColorSpace: 'srgb',
        rowLength: 0,
        skipRows: 0,
        skipPixels: 0,
        imageHeight: 0,
        skipImages: 0,
      },
      pack: { alignment: 4, rowLength: 0, skipRows: 0, skipPixels: 0 },
    },
    textureUnits,
    currentProgram: null,
    arrayBuffer: null,
    vaoBinding: null,
    vao: defaultVAOState(limits.MAX_VERTEX_ATTRIBS),
    drawFramebuffer: null,
    readFramebuffer: null,
    renderbuffer: null,
    uniformBuffers,
    uniformBufferRanges,
    pixelPackBuffer: null,
    pixelUnpackBuffer: null,
    copyReadBuffer: null,
    copyWriteBuffer: null,
    transformFeedback: null,
    activeQueries: { ANY_SAMPLES_PASSED: null, ANY_SAMPLES_PASSED_CONSERVATIVE: null },
    drawBuffers: [0x8ce0 /* COLOR_ATTACHMENT0 */],
    readBuffer: 0x8ce0,
    limits,
  };
}

/** 0x1E00 = KEEP (stencil op). */
export const KEEP = 0x1e00;

/** Alias kept for tests/unit/state.test.ts (createState === createDefaultState). */
export const createState = createDefaultState;
