/**
 * types.ts — shared types and layout constants for the raster module.
 *
 * This file is the COMPILE-TIME CONTRACT between raster and its two callers:
 *  - `gl/` constructs DrawCall objects (contract §2 of src/CONTEXT.md) and owns
 *    surfaces, textures and effective sampler state (contract §3).
 *  - `glsl/` compiles fragment shaders against `FragmentExecCtx` and the
 *    `TextureEnv` sampling entry points (contract §1).
 *
 * Raster imports NOTHING from gl/ or glsl/ at runtime. The program type used
 * here (`RasterProgram`) is a structural subset of glsl's `Program` — glsl
 * must ensure its Program is assignable to it (import this type from
 * `../raster` for a compile-time check).
 *
 * Storage conventions (MANDATORY):
 *  - Vertex records are packed in a Float32Array: `[x, y, z, w, pointSize, varyings...]`.
 *    x/y/z/w are CLIP-space when the record is handed to draw(); after the
 *    viewport transform (clip.applyViewportTransform) x/y/z are WINDOW coords
 *    and w is preserved as clip w (needed for perspective interpolation and
 *    gl_FragCoord.w = 1/w_clip).
 *  - Surface rows: row 0 is the BOTTOM row (GL window coordinates, y up).
 *    The drawing buffer's present() path flips for display; readPixels does
 *    NOT flip (GL semantics).
 *  - No per-fragment allocation anywhere in raster; all scratch lives in
 *    `RasterState` and caller-provided `out` arrays.
 */

import type { PixelFormatInfo } from './formats';
import type { GLenum } from './gl-enums';

/* ================================================================== */
/* Vertex record layout                                               */
/* ================================================================== */

export const RECORD_OFFSET_X = 0;
export const RECORD_OFFSET_Y = 1;
export const RECORD_OFFSET_Z = 2;
export const RECORD_OFFSET_W = 3;
export const RECORD_OFFSET_POINT_SIZE = 4;
/** Index of the first varying component in a vertex record. */
export const VARYINGS_OFFSET = 5;
/** Number of header floats before the varyings. */
export const RECORD_HEADER_FLOATS = 5;

/**
 * Per-varying metadata (structurally identical to glsl's VaryingInfo; raster
 * owns the type so glsl can import it from here).
 */
export interface VaryingInfo {
  name: string;
  /** GLSL type enum (FLOAT_VEC2 etc.) — informational for raster. */
  type: GLenum;
  /** 1..4 component count (packed contiguously in record order). */
  components: number;
  /** Flat interpolation (also implied for integer/uint varyings). */
  flat: boolean;
}

/** Total floats per vertex record for a program's varying list. */
export function computeVertexStride(varyings: readonly VaryingInfo[]): number {
  let n = RECORD_HEADER_FLOATS;
  for (let i = 0; i < varyings.length; i++) n += varyings[i].components;
  return n;
}

/** Writes the header (position + point size) of one vertex record. */
export function writeVertexHeader(
  out: Float32Array, base: number,
  x: number, y: number, z: number, w: number, pointSize: number,
): void {
  out[base + RECORD_OFFSET_X] = x;
  out[base + RECORD_OFFSET_Y] = y;
  out[base + RECORD_OFFSET_Z] = z;
  out[base + RECORD_OFFSET_W] = w;
  out[base + RECORD_OFFSET_POINT_SIZE] = pointSize;
}

/* ================================================================== */
/* Implementation limits (single source of truth — gl/ MUST import     */
/* these for getParameter(ALIASED_POINT_SIZE_RANGE) etc.)              */
/* ================================================================== */

export const ALIASED_POINT_SIZE_RANGE: readonly [number, number] = [1, 1024];
export const ALIASED_LINE_WIDTH_RANGE: readonly [number, number] = [1, 1];

/* ================================================================== */
/* Program (structural subset of glsl's Program)                       */
/* ================================================================== */

/** The fragment execution context — constructed by raster, consumed by glsl codegen. */
export interface FragmentExecCtx {
  /**
   * Interpolated varyings, one entry per program varying, in program order.
   * `v` holds the current fragment's interpolated components (length =
   * varying.components). `ddx`/`ddy` are filled with the 2×2-quad
   * derivatives and are valid only when program.fragment.usesDerivatives.
   * Flat (and integer) varyings have zero derivatives.
   */
  varyings: VaryingInterpolant[];
  /** [x, y, z, w] — window coords; z = depth (0..1); w = 1/w_clip. */
  fragCoord: Float32Array;
  /** True when the fragment faces the front (per current frontFace). */
  frontFacing: boolean;
  /** [s, t] in 0..1 for POINTS primitives; zero otherwise. */
  pointCoord: Float32Array;
  /** Default-block uniform store (float). Integer/sampler uniforms are stored
   *  as their raw bit patterns in the float array (codegen reads via
   *  Int32Array views or `| 0`). */
  uniforms: Float32Array;
  /** Uniform block stores keyed by block name (WebGL2). */
  uniformBlocks?: Record<string, ArrayBufferView>;
  /** Texture sampling entry points + per-unit bindings (see TextureEnv). */
  tex: TextureEnv;
  out: {
    /** Per output-location scratch; codegen writes `ctx.out.color[loc][i]`. */
    color: Float32Array[];
    /** Written by the shader when program.fragment.usesFragDepth. */
    fragDepth: number;
  };
  /**
   * Set to true by generated code when the shader executes `discard`.
   * Raster resets it to false before every invocation and, if true after the
   * invocation, skips ALL fragment ops (stencil/depth/color writes).
   */
  discarded: boolean;
}

/** One interpolated varying for the current fragment. */
export interface VaryingInterpolant {
  /** Interpolated components for the current fragment. */
  v: Float32Array;
  /** ∂v/∂x per component (quad differences); valid when usesDerivatives. */
  ddx?: Float32Array;
  /** ∂v/∂y per component (quad differences); valid when usesDerivatives. */
  ddy?: Float32Array;
}

/** Effective texture unit binding: image + merged sampler state. */
export interface TextureUnitBinding {
  img: TextureImage;
  /** Effective state (texture params merged with any bound WebGLSampler). */
  state: SamplerState;
}

/**
 * Texture sampling environment handed to generated fragment code via
 * `ctx.tex`. All methods write the sampled texel into `out` (float domain)
 * and `outInt`/`outUint` (raw bits — same buffer) and return nothing, so no
 * allocation happens in hot paths. `units` is indexed by sampler uniform value
 * (the texture unit); null entries sample as (0,0,0,1).
 */
export interface TextureEnv {
  /** Per texture unit bindings (index = unit). */
  units: readonly (TextureUnitBinding | null)[];
  /** Shared result scratch (length ≥ 4). */
  out: Float32Array;
  /** Signed-int reinterpretation of `out` (same ArrayBuffer). */
  outInt: Int32Array;
  /** Unsigned-int reinterpretation of `out` (same ArrayBuffer). */
  outUint: Uint32Array;

  /** texture() — implicit LOD from derivatives (fragment). */
  sample2D(unit: number, u: number, v: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void;
  /** textureLod() — explicit level. */
  sample2DLod(unit: number, u: number, v: number, lod: number): void;
  /** texture() on sampler3D. */
  sample3D(unit: number, u: number, v: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void;
  sample3DLod(unit: number, u: number, v: number, w: number, lod: number): void;
  /** texture() on samplerCube. */
  sampleCube(unit: number, u: number, v: number, w: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void;
  sampleCubeLod(unit: number, u: number, v: number, w: number, lod: number): void;
  /** texture() on sampler2DArray; `layer` is NOT filtered/wrapped. */
  sample2DArray(unit: number, u: number, v: number, layer: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void;
  sample2DArrayLod(unit: number, u: number, v: number, layer: number, lod: number): void;
  /** texture() on sampler2DShadow; `ref` is compared against depth texels. */
  sample2DShadow(unit: number, u: number, v: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void;
  /** texture() on samplerCubeShadow (P = (u,v,w), ref separate). */
  sampleCubeShadow(unit: number, u: number, v: number, w: number, ref: number, dux: number, dvx: number, dwx: number, duy: number, dvy: number, dwy: number, bias: number): void;
  /** texture() on sampler2DArrayShadow. */
  sample2DArrayShadow(unit: number, u: number, v: number, layer: number, ref: number, dux: number, dvx: number, duy: number, dvy: number, bias: number): void;
  /** texelFetch() — integer coords, explicit level, no filtering. */
  texelFetch2D(unit: number, x: number, y: number, level: number): void;
  texelFetch3D(unit: number, x: number, y: number, z: number, level: number): void;
  texelFetch2DArray(unit: number, x: number, y: number, layer: number, level: number): void;
}

/** Structural subset of glsl's Program — what raster needs to draw. */
export interface RasterProgram {
  varyings: readonly VaryingInfo[];
  fragment: {
    /** Execute the fragment shader for the current fragment (fills ctx). */
    run(ctx: FragmentExecCtx): void;
    /** Shader uses dFdx/dFdy or implicit-LOD texture sampling → 2×2 quads required. */
    usesDerivatives: boolean;
    /** Shader writes gl_FragDepth → depth test must run after the shader. */
    usesFragDepth: boolean;
    /** Fragment color outputs: { location, type } pairs. */
    outputs: readonly { location: number; type: GLenum }[];
  };
}

/* ================================================================== */
/* Surfaces & framebuffer target                                       */
/* ================================================================== */

/**
 * A render target surface (drawing buffer, renderbuffer, or texture level
 * attached to an FBO). Plain typed-array-backed; gl/ allocates them (via
 * surface.createSurface) and reads them (readPixels); raster writes them.
 *
 * Representation per format family:
 *  - Color formats: `data` holds tightly packed texels (row 0 = BOTTOM).
 *  - DEPTH_COMPONENT_*: `data` is a Float32Array of depths (0..1).
 *  - DEPTH*_STENCIL*: SPLIT representation — `data` = Float32Array depths,
 *    `stencilData` = Uint8Array. The packed 32-bit form exists only at the
 *    readPixels boundary (formats.getPackConverter).
 *  - STENCIL_INDEX8: `data` is a Uint8Array.
 */
export interface Surface {
  width: number;
  height: number;
  /** Internal format GLenum (key into the formats registry). */
  format: GLenum;
  /** Resolved format descriptor (getFormat(format)). */
  info: PixelFormatInfo;
  /** Primary texel storage (see representation rules above). */
  data: ArrayBufferView;
  /** Split stencil plane for DEPTH*_STENCIL* formats. */
  stencilData?: Uint8Array;
}

/** The draw target: color attachment list + depth/stencil attachments. */
export interface FramebufferTarget {
  /** Color attachments (WebGL1: length 1; WebGL2: ≥ drawBuffers length). */
  color: readonly (Surface | null)[];
  /** Depth attachment. A DEPTH_STENCIL surface attached to depth also
   *  occupies the stencil slot (gl/ sets both to the same surface). */
  depth: Surface | null;
  stencil: Surface | null;
  /** Framebuffer dimensions (validated by gl/). */
  width: number;
  height: number;
  /** > 1 for multisample renderbuffers (resolve happens in gl/ via blit). */
  samples: number;
}

/* ================================================================== */
/* Draw state (per-draw snapshot owned by gl/)                         */
/* ================================================================== */

export interface Viewport { x: number; y: number; w: number; h: number; }
export interface DepthRange { near: number; far: number; }
export interface ScissorState { enabled: boolean; x: number; y: number; w: number; h: number; }
export interface CullState { enabled: boolean; face: GLenum; frontFace: GLenum; }
export interface PolygonOffsetState { enabled: boolean; factor: number; units: number; }
export type ColorMask = readonly [boolean, boolean, boolean, boolean];
export interface BlendState {
  enabled: boolean;
  srcRGB: GLenum; dstRGB: GLenum; srcAlpha: GLenum; dstAlpha: GLenum;
  eqRGB: GLenum; eqAlpha: GLenum;
  color: [number, number, number, number];
}
export interface DepthTestState { enabled: boolean; func: GLenum; }
export interface StencilFaceState {
  func: GLenum; ref: number; valueMask: number; writeMask: number;
  fail: GLenum; zfail: GLenum; zpass: GLenum;
}
export interface StencilTestState { enabled: boolean; front: StencilFaceState; back: StencilFaceState; }
export interface SampleCoverageState { enabled: boolean; value: number; invert: boolean; }

/**
 * One draw call. gl/ evaluates the vertex shader into `vertices` (packed
 * records, see the layout constants above) and calls `draw(dc)`.
 *
 * Vertex addressing for instanced draws: instance `i`, vertex `j` (0-based
 * within the draw) lives at record index `first + i * count + j` — i.e. the
 * buffer holds `instanceCount` consecutive runs of `count` records, each run
 * starting at record `first + i*count`. gl/ packs all instances before the
 * single draw() call.
 */
export interface DrawCall {
  mode: GLenum;
  count: number;
  /** Record index of the first vertex (see addressing rule above). */
  first: number;
  instanceCount: number;
  /** Packed post-VS vertex records (see record layout). */
  vertices: Float32Array;
  /** Floats per vertex record (computeVertexStride(program.varyings)). */
  vertexStride: number;
  /** Index of the first varying in a record (must equal VARYINGS_OFFSET). */
  varyingsOffset: number;
  program: RasterProgram;
  fb: FramebufferTarget;
  viewport: Viewport;
  depthRange: DepthRange;
  scissor: ScissorState;
  cull: CullState;
  polygonOffset: PolygonOffsetState;
  dither: boolean;
  /** Per fragment-output-location masks (index = output location). */
  colorMask: readonly ColorMask[];
  blend: BlendState;
  depthTest: DepthTestState;
  depthMask: boolean;
  stencilTest: StencilTestState;
  sampleCoverage: SampleCoverageState;
  rasterizerDiscard: boolean;
  lineWidth: number;
  /** Per texture unit bindings (index = unit; sampler uniforms index here). */
  textures: readonly (TextureUnitBinding | null)[];
  /**
   * Maps fragment output location → color attachment index, or -1 for NONE.
   * Length = number of fragment outputs; raster writes fb.color[drawBuffers[l]]
   * from ctx.out.color[l].
   */
  drawBuffers: readonly number[];
  /**
   * Default-block fragment uniform store (same layout as
   * FragmentExecCtx.uniforms). The rasterizer wires it into the fragment exec
   * ctx. Required — gl/ always provides it for draws with a program.
   */
  uniforms: Float32Array;
  /**
   * WebGL2 uniform block stores keyed by block name. Optional: undefined for
   * WebGL1 draws (no UBOs).
   */
  uniformBlocks?: Record<string, ArrayBufferView>;
  /**
   * Optional WebGL2 occlusion-query counter (SAMPLES_PASSED). Out-param
   * REFERENCE owned by gl/: raster increments `sampleCountRef.value` in place,
   * exactly once per sample that passes the stencil AND depth tests (see
   * FragmentOps). Absent → no counting.
   */
  sampleCountRef?: { value: number };
}

/* ================================================================== */
/* Texture sampling                                                    */
/* ================================================================== */

/**
 * Texture coordinates for the generic sampler entry point.
 * `v` = [u, v] for 2D / 2D_ARRAY (v[2] = layer for 2D_ARRAY), [u, v, w] for
 * 3D and cube. `dx`/`dy` are the per-component derivatives (∂/∂x, ∂/∂y)
 * used for implicit LOD; they may be omitted or zero-filled scratch arrays
 * when the caller has no derivative information (the sampler then treats
 * them as zero — LOD clamps to minLod). NO allocation: callers reuse
 * scratch arrays.
 */
export interface SampleCoord {
  v: Float32Array;
  dx?: Float32Array;
  dy?: Float32Array;
}

/** Effective sampling parameters (merged texture + sampler state by gl/). */
export interface SamplerState {
  minFilter: GLenum;
  magFilter: GLenum;
  wrapS: GLenum;
  wrapT: GLenum;
  wrapR: GLenum;
  /** LOD clamp (GL_TEXTURE_MIN_LOD / MAX_LOD; defaults -1000 / 1000). */
  minLod: number;
  maxLod: number;
  /** NONE or COMPARE_REF_TO_TEXTURE (shadow samplers). */
  compareMode: GLenum;
  /** Comparison function for shadow sampling (default LEQUAL). */
  compareFunc: GLenum;
  /** EXT_texture_filter_anisotropic; 1 = disabled. */
  maxAnisotropy: number;
}

/** One mip level of a texture image. */
export interface TextureLevel {
  width: number;
  height: number;
  /** 3D/2D_ARRAY slice count; 6 for cube maps (one face per entry below). */
  depth: number;
  /**
   * Texel storage: one entry per face for cube maps (order +X, -X, +Y, -Y,
   * +Z, -Z), exactly one entry otherwise. Rows: row 0 = BOTTOM.
   */
  data: ArrayBufferView[];
}

/**
 * A texture image as seen by the sampler. Owned by gl/; `levels`, dimensions
 * and immutable/base/max are updated by gl/ on upload; the sampler only
 * reads.
 */
export interface TextureImage {
  target: GLenum;
  internalFormat: GLenum;
  info: PixelFormatInfo;
  /** Base level dimensions (level 0). */
  width: number;
  height: number;
  /** 3D/2D_ARRAY depth, 6 for cube, 1 for 2D. */
  depth: number;
  levels: TextureLevel[];
  /** LOD range from TEXTURE_BASE_LEVEL / TEXTURE_MAX_LEVEL. */
  baseLevel: number;
  maxLevel: number;
  /** texStorage'd (immutable) — level count fixed. */
  immutable: boolean;
  /** Completeness computed by gl/; incomplete → sampler returns (0,0,0,1). */
  complete: boolean;
}

/* ================================================================== */
/* Fragment ops & per-draw raster state                                */
/* ================================================================== */

/**
 * Per-fragment operations engine (implemented by FragmentOps in
 * fragment-ops.ts). Two-phase protocol so `discard` (detected after the
 * fragment shader runs) suppresses ALL writes:
 *
 *  - test(): scissor → sample coverage → stencil test (+ stencil fail op) →
 *    depth test (read only, when !program.fragment.usesFragDepth; on fail the
 *    stencil zfail op is applied). Returns true if the fragment proceeds to
 *    shading. When the program writes gl_FragDepth, the depth-dependent part
 *    is deferred to finalize() and the depth passed here is unused.
 *  - finalize(): depth write (if depthMask) + stencil zpass op + blend +
 *    dither + sRGB conversion + colorMask + color write, using the
 *    shader-computed depth (which equals the interpolated depth for
 *    non-fragDepth shaders). Called only for passed, non-discarded fragments.
 *
 * Occlusion counting: when `DrawCall.sampleCountRef` is present, the ops
 * engine increments it exactly once per sample that passes the stencil AND
 * depth tests (WebGL2 SAMPLES_PASSED semantics). For shaders that do NOT write
 * gl_FragDepth this happens in test() right after the depth read passes; for
 * gl_FragDepth shaders the depth test is deferred to finalize(), so the
 * increment happens there after the post-shader depth test passes. Fragments
 * discarded by the shader are counted when the depth test already passed
 * before the shader ran. Helper invocations (outside-pixel quad lanes) never
 * count.
 */
export interface FragmentOps {
  test(x: number, y: number, frontFacing: boolean, depth: number): boolean;
  finalize(
    x: number, y: number, frontFacing: boolean, depth: number,
    colors: readonly Float32Array[],
  ): void;
}

/**
 * Per-draw scratch + execution state shared by the primitive rasterizers
 * (triangles/lines/points) and runQuad(). All buffers are allocated once per
 * draw call by rasterizer.createRasterState — never per fragment.
 */
export interface RasterState {
  dc: DrawCall;
  fragCtx: FragmentExecCtx;
  ops: FragmentOps;
  /** Sum of program.varyings[].components. */
  totalVaryComponents: number;
  /** Facing of the primitive currently being rasterized. */
  frontFacing: boolean;
  /** Quad interpolation scratch: 4 × totalVaryComponents, laid out
   *  [pixel][component] for pixels (0,0),(1,0),(0,1),(1,1) of the quad. */
  quadV: Float32Array;
  /** Per-pixel window depth for the quad (4). */
  quadDepth: Float32Array;
  /** Per-pixel 1/w_clip for the quad (4) → gl_FragCoord.w. */
  quadW: Float32Array;
  /** Per-pixel point coords [s,t]×4 (POINTS only; else zeros). */
  quadPointCoord: Float32Array;
}
