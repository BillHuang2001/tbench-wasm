/**
 * program.ts — the Program model and the runtime execution contexts.
 *
 * This file is the compile-time contract between src/glsl/ and src/gl/ and
 * src/raster/ (cross-module contracts §1–§3 in ../CONTEXT.md):
 *
 * - gl/ consumes the info arrays for its whole API surface (getActiveAttrib,
 *   getActiveUniform, getUniformLocation, getUniformBlockIndex,
 *   getActiveUniformBlockParameter, getTransformFeedbackVarying, draw-time
 *   validation), writes uniform values into the stores, sets up the exec ctx
 *   per draw, and drives `vertex.run(ctx)` per vertex.
 * - raster/ consumes `varyings` (packing order for the vertex record and for
 *   interpolation), `fragment.outputs`, the capability flags, and drives
 *   `fragment.run(ctx)` per fragment.
 *
 * Type-only contract: the linker (Phase 2) produces these objects; nothing in
 * this file executes at runtime. TextureImage/SamplerState are raster's types
 * (raster/texture-sampler.ts, contract §3) — the import is type-only, erased
 * at compile time, so there is no runtime dependency cycle.
 */
import type { TextureImage, SamplerState } from '../raster';

/* ------------------------------------------------------------------ */
/* Info types (linker output — introspection for the gl/ API surface)  */
/* ------------------------------------------------------------------ */

/**
 * One active vertex attribute.
 * - `name`: base name (arrays keep the bare name; getActiveAttrib appends
 *   `[0]` per spec — gl/ formats it).
 * - `location`: assigned attrib location. A matC attribute occupies C
 *   consecutive locations starting here (gl/ derives C from `type`).
 * - `type`: GLenum of the element type (GL_FLOAT_VEC3, GL_INT_VEC4, ...).
 * - `size`: array length (1 for non-arrays).
 * - `components`: element components (vector size / matrix rows / 1).
 * - `integral`: int/uint attribute (vertexAttribIPointer data, WebGL2).
 */
export interface AttribInfo {
  name: string;
  location: number;
  type: number;
  size: number;
  components: number;
  integral: boolean;
}

/**
 * One active uniform — the FLATTENED leaf list, matching getActiveUniform
 * semantics:
 * - Plain arrays are ONE entry: name `'u[0]'`, size = array length,
 *   location = first element's slot.
 * - Structs are flattened per leaf member: `'u.m'`, `'u[0].m'`, `'u[2].m'`
 *   (struct arrays expand element-by-element, size = 1 each).
 * - Uniform-block members ARE included (blockIndex >= 0, location = -1);
 *   getUniformLocation must return null for them, so they are NOT in
 *   `Program.uniformMap`.
 *
 * `location`: for default-block uniforms, the STARTING vec4-slot index in the
 * appropriate store (floatStore for float/mat, intStore for int/uint/bool/
 * sampler). `components` = element components; `integral` = int/uint type;
 * `sampler` = sampler type (value = texture unit binding, int store).
 */
export interface UniformInfo {
  name: string;
  location: number; // -1 for uniform-block members
  type: number;
  size: number;
  components: number;
  integral: boolean;
  blockIndex: number; // -1 for default-block uniforms
  sampler: boolean;
}

/**
 * One member of a uniform block (std140 layout, WebGL2) — the data backing
 * getActiveUniformBlockParameter/getActiveUniformsiv(GL_UNIFORM_OFFSET/...).
 * The codegen bakes these offsets/stride values into the generated fragment/
 * vertex bodies at link time.
 */
export interface UniformBlockMemberInfo {
  name: string;
  offset: number; // byte offset within the block
  type: number; // GLenum of the member's element type
  size: number; // array length (1 for non-arrays)
  arrayStride: number; // bytes between array elements (0 when size === 1)
  matrixStride: number; // bytes between columns (column-major) or rows (row-major); 0 for non-matrices
  rowMajor: boolean; // true for layout(row_major) members (default column-major)
}

/**
 * One uniform block (WebGL2). Array blocks (`uniform Blocks {..} b[2]`) get
 * one entry PER ELEMENT, named `'b[0]'`, `'b[1]'`. `index` is the block index
 * used by getUniformBlockIndex/uniformBlockBinding and by the exec ctx
 * (`blockStores[index]`).
 */
export interface UniformBlockInfo {
  name: string;
  index: number;
  size: number; // total std140 byte size (rounded to 16)
  activeUniforms: UniformBlockMemberInfo[];
}

/**
 * One interface varying (vertex out / fragment in), in the packed order both
 * stages agree on.
 * - `type`: GLenum of the element type.
 * - `components`: TOTAL packed component count (array varyings are expanded:
 *   `varying vec3 v[2]` → 6).
 * - `flat`: flat interpolation (flat-qualified, or integral type — integral
 *   varyings are flat-only, enforced at link).
 * The vertex record layout (contract §2) is
 * `[px,py,pz,pw, pointSize, varyings in this order]`; raster interpolates
 * each varying over `components` values.
 */
export interface VaryingInfo {
  name: string;
  type: number;
  components: number;
  flat: boolean;
}

/* ------------------------------------------------------------------ */
/* Program                                                             */
/* ------------------------------------------------------------------ */

/** Compiled vertex stage. `run` evaluates ONE vertex into ctx.out. */
export interface VertexStage {
  run(ctx: VertexExecCtx): void;
}

/**
 * Compiled fragment stage. `run` evaluates ONE fragment into ctx.out; raster
 * commits the outputs after run returns (so helper quad invocations never
 * write) and skips the fragment when `ctx.discarded` is set.
 */
export interface FragmentStage {
  run(ctx: FragmentExecCtx): void;
  /** True when the shader uses dFdx/dFdy/fwidth or implicit-LOD texturing: raster MUST supply ddx/ddy for varyings. */
  usesDerivatives: boolean;
  /** True when the shader writes gl_FragDepth/gl_FragDepthEXT: raster must commit ctx.out.fragDepth. */
  usesFragDepth: boolean;
  /**
   * Fragment color outputs, ONE ENTRY PER SLOT: WebGL1 — gl_FragColor
   * (location 0) or gl_FragData[0..n] with EXT_draw_buffers or
   * gl_SecondaryFragColorEXT (location 0, index 1 — dual-source secondary).
   * WebGL2 — the assigned locations of every output variable, expanded per
   * slot: an array output `out vec4 my_FragData[8]` produces 8 entries
   * (locations base..base+7); scalar/vector outputs produce 1 entry each
   * (explicit layout(location=) honored).
   * `type` = GLenum of the element type (GL_FLOAT_VEC4 | GL_INT_VEC4 |
   * GL_UNSIGNED_INT_VEC4 | ... — any float/int/uint scalar or vector).
   * `index` = the dual-source blend index (0/1): outputs with the same
   * location but indices 0/1 are the primary/secondary pair
   * (GL_EXT_blend_func_extended; gl/ reads index === 1 for dual-source
   * draw validation).
   * Name → location lookups (getFragDataLocation) are NOT part of this array:
   * gl/ builds its map from ShaderInfo.outputs (which carries declaration +
   * per-element '<name>[k]' entries).
   */
  outputs: { location: number; index: number; type: number }[];
}

/** One captured transform-feedback varying (WebGL2, SEPARATE_ATTRIBS or INTERLEAVED). */
export interface TransformFeedbackVarying {
  name: string;
  type: number; // GLenum of the element type
  size: number; // array length (1 for non-arrays)
}

/**
 * The linked program — the single object gl/ registers and raster/ consumes.
 * Created by `linkProgram` (compiler.ts). All fields are link-time immutable
 * except the store contents (gl/ writes uniform values into them; generated
 * code reads them via the exec ctx).
 */
export interface Program {
  attributes: AttribInfo[];
  uniforms: UniformInfo[];
  uniformBlocks: UniformBlockInfo[];
  varyings: VaryingInfo[];

  vertex: VertexStage;
  fragment: FragmentStage;

  /** Vertex shader writes gl_PointSize (raster: honor point size in POINTS mode). */
  usesPointSize: boolean;
  /** Fragment shader reads gl_PointCoord (raster: compute per-fragment point coords). */
  usesGLPointCoord: boolean;
  /** Fragment shader reads gl_FragCoord. */
  usesFragCoord: boolean;
  /** Fragment shader reads gl_FrontFacing. */
  usesFrontFacing: boolean;

  /**
   * getUniformLocation lookup table, keyed by every canonical lookup path of
   * DEFAULT-BLOCK uniforms: `'u'`, `'u[2]'`, `'u.m'`, `'u[0].m'`, `'u[2].m'`.
   * Bare array names (`'u'`) map to the first element (spec: getUniformLocation
   * of an array name returns u[0]'s location). Uniform-block members are NOT
   * present (spec: returns null). gl/ returns null for any miss.
   */
  uniformMap: ReadonlyMap<string, UniformInfo>;

  /**
   * Default-block uniform stores. Layout: vec4-slot packing — every uniform
   * occupies `ceil(components/4)` consecutive slots (matC = C slots,
   * column-major), `UniformInfo.location` = start slot, NO sharing between
   * uniforms (conservative WebGL1-compatible packing; WebGL2-legal).
   * gl/ writes via glUniform* (float values into floatStore, int/uint/bool/
   * sampler into intStore) and passes the same arrays in the exec ctx.
   */
  floatStore: Float32Array;
  intStore: Int32Array;

  /**
   * Builtin uniform gl_DepthRange reflection (GLSL ES 1.00 §7.6 / 3.00 §7.7):
   * float-store indices of [near, far, far − near] — the members
   * 'gl_DepthRange.near/far/diff' are ACTIVE UNIFORMS in `uniforms` +
   * `uniformMap` (GL_FLOAT, size 1) backed by these REAL slots, appended after
   * all user uniforms. gl/ MUST write the CURRENT depth range into them
   * (float write at index `depthRangeSlots[i]`, byte `slot*4`) at link/adopt
   * time and on every glDepthRangef so getUniform returns live values.
   * codegen does NOT read these slots (it lowers member reads to
   * `ctx.depthRange[i]` — draw-time state filled per draw); the slots exist
   * purely for the reflection API. null when neither shader reads
   * gl_DepthRange (no entries, no slots).
   */
  depthRangeSlots: [number, number, number] | null;

  /**
   * Scratch buffer sizes (in elements) the generated code needs for local
   * arrays / complex builtin temporaries. gl/ allocates ctx.scratch /
   * ctx.intScratch of at least this size per draw and reuses them across
   * vertices/fragments (zero per-invocation allocation).
   */
  scratchSize: number;
  intScratchSize: number;

  /** Captured varyings (WebGL2 transform feedback) — empty for WebGL1 programs. */
  transformFeedbackVaryings: TransformFeedbackVarying[];
}

/* ------------------------------------------------------------------ */
/* Execution contexts                                                  */
/* ------------------------------------------------------------------ */

/** Source of one attribute, by location. */
export type AttribSource = Float32Array | Int32Array | Uint32Array | number;

/**
 * State shared by vertex and fragment execution. All arrays are preallocated
 * per draw by gl/ and reused for every invocation (no per-vertex/per-fragment
 * allocation anywhere in the hot path).
 */
export interface BaseExecCtx {
  /** Default-block float/matrix uniforms (program.floatStore). */
  uniforms: Float32Array;
  /** Default-block int/uint/bool/sampler uniforms (program.intStore). */
  intUniforms: Int32Array;
  /**
   * Uniform-block data per BLOCK INDEX (indexed by UniformBlockInfo.index,
   * NOT by binding point): at draw time gl/ maps each block's current binding
   * to the bound GL buffer and stores a view of its data here (a zero-filled
   * fallback when nothing is bound). Generated code bakes the block index
   * and member byte offset; int/uint members read blockIntStores.
   */
  blockStores: Float32Array[];
  blockIntStores: Int32Array[];
  /** Texture images per texture unit (null = nothing bound). */
  textures: (TextureImage | null)[];
  /** Effective per-unit sampling state (texture params + sampler object override), computed by gl/ per draw. */
  samplerStates: SamplerState[];
  /** Codegen scratch for float locals/arrays (size ≥ program.scratchSize). */
  scratch: Float32Array;
  /** Codegen scratch for int locals/arrays (size ≥ program.intScratchSize). */
  intScratch: Int32Array;
  /**
   * Builtin uniform gl_DepthRange state (GLSL ES 1.00 §7.6 / 3.00 §7.7):
   * [near, far, far − near] from the current depthRange. gl/ fills it on the
   * VERTEX exec ctx (gl/draw.ts) and raster fills it on the FRAGMENT exec ctx
   * (rasterizer.ts) per draw; codegen lowers `gl_DepthRange.near/far/diff` to
   * `ctx.depthRange[0/1/2]`. Absent (layers not yet updated) → shaders reading
   * gl_DepthRange throw at runtime — the gl/raster managers must fill it.
   */
  depthRange?: Float32Array;
}

/**
 * Vertex execution context.
 *
 * Attribute fetch model (contract with gl/): gl/ performs a per-draw DENSE
 * extraction of each enabled attribute array (stride removed, ordered by
 * vertex position within the draw; format conversion applied) into
 * `attribs[loc]`. `attribIndices[loc]` is the current fetch index:
 * - divisor-0 attributes: the 0-based vertex position within the draw;
 * - instanced attributes (divisor > 0): floor(instanceId / divisor);
 * - constant attributes: 0 (unused).
 * Generated code reads `attribs[loc][attribIndices[loc] * components + c]`.
 * Constant attributes may also be passed as a plain `number` (scalar
 * constant; components fill (v,0,0,1)) — codegen handles the typeof guard,
 * but gl/ SHOULD pass 4-element views for speed.
 *
 * `vertexId` is gl_VertexID: first+i for glDrawArrays (i = vertex position),
 * the element value for glDrawElements — it may differ from the fetch index.
 */
export interface VertexExecCtx extends BaseExecCtx {
  attribs: AttribSource[];
  attribIndices: Int32Array;
  vertexId: number;
  instanceId: number;
  /**
   * gl_DrawID (WEBGL_multi_draw / GL_ANGLE_multi_draw): the multi-draw
   * subdraw index, constant for every vertex of the draw; 0 for single
   * draws. Optional so selftest ctx literals that don't exercise gl_DrawID
   * stay valid — gl/draw.ts ALWAYS sets it (req.drawId ?? 0) when executing.
   */
  drawId?: number;
  out: {
    /** Clip-space position [x, y, z, w] (preallocated; written per vertex). */
    position: Float32Array;
    /** gl_PointSize (0 when the shader doesn't write it). */
    pointSize: number;
    /** Packed varying values per Program.varyings order (preallocated). */
    varyings: Float32Array;
  };
}

/** Interpolated varying values for one varying, provided by raster. */
export interface VaryingValues {
  /** Interpolated components (Float32Array of VaryingInfo.components). */
  v: Float32Array;
  /** Screen-space derivatives; present when program.fragment.usesDerivatives. */
  ddx?: Float32Array;
  ddy?: Float32Array;
}

/**
 * Fragment execution context. raster evaluates fragments in 2×2 quads and
 * provides derivatives for all varyings when the shader needs them; it resets
 * `discarded` per fragment and commits ctx.out only for non-discarded,
 * in-primitive fragments after run() returns.
 */
export interface FragmentExecCtx extends BaseExecCtx {
  /** Interpolated varyings per Program.varyings order. */
  varyings: VaryingValues[];
  /** gl_FragCoord [x, y, z, w] (window coords; w = 1/w_clip). */
  fragCoord: Float32Array;
  frontFacing: boolean;
  /** gl_PointCoord [s, t] in [0,1] (POINTS mode only; valid when usesGLPointCoord). */
  pointCoord: Float32Array;
  /** Reset by raster before run(); `discard` compiles to `discarded = true; return`. */
  discarded: boolean;
  out: {
    /** Color per output location (preallocated Float32Array(4) each). */
    color: Float32Array[];
    /** gl_FragDepth (written when fragment.usesFragDepth). */
    fragDepth: number;
  };
}
