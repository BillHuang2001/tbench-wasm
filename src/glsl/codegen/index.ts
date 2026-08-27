/**
 * codegen/index.ts — the codegen ↔ linker contract (part of contract §1).
 *
 * The LINKER computes the layout (attribute locations, uniform vec4-slot
 * positions, std140 block offsets, varying packing) and calls
 * generateVertexStage / generateFragmentStage with the shader's annotated AST
 * plus this layout. The generated `body` is the JS function body compiled via
 * `new Function('ctx', 'R', body)` — R is the shared runtime object from
 * runtime.ts (never embedded per shader).
 *
 * The DUAL seam: `Value` is the currency of expression codegen.
 *   - non-dual mode: only `v` is set (a JS expression string).
 *   - dual mode (layout.uses.derivatives): every FLOAT value also carries
 *     `dx`/`dy` (screen-space derivative expressions). Integer/uint/bool
 *     values never carry duals.
 * The dual mode is implemented by a follow-up task on this seam.
 *
 * The internal codegen environment (var table, scratch allocator, naming) is
 * designed by the implementation tasks (expressions.ts etc.) — this file only
 * pins the linker-facing surface + the Value seam.
 */
import type { TranslationUnit } from '../ast.js';
import type { ShaderUses } from '../compiler.js';

/** One flattened default-block uniform path → store position. */
export interface UniformSlot {
  /** 'float' store = ctx.uniforms, 'int' store = ctx.intUniforms. */
  store: 'float' | 'int';
  /** Start vec4-slot index (location) of the path's first element. */
  slot: number;
  /**
   * Per-element slot stride for paths whose LAST array dimension may be
   * dynamically indexed (canonicalized to `[0]` in the path key). 0 = not
   * dynamically indexable. Struct-array members: stride = whole-struct slots
   * (e.g. 'u[0].m' → slot(u[0].m) + i*stride).
   */
  stride: number;
}

/** One std140 uniform-block member (byte offsets; codegen divides by 4). */
export interface BlockMemberLayout {
  offset: number;        // byte offset of the member's first element
  arrayStride: number;   // bytes between array elements (0 when not an array)
  matrixStride: number;  // bytes between matrix columns (0 when not a matrix)
  rowMajor: boolean;     // always false (std140 column-major is mandatory)
  /**
   * std140 byte size of ONE block instance (bytes between consecutive
   * instances of an ARRAYED block, e.g. `uniform Blocks {..} b[2]`).
   * Absent on every member entry = single-instance block (a dynamic
   * instance index is then a linker error). Present on ALL member entries
   * of an arrayed block (the linker stamps the same value on each).
   */
  blockStride?: number;
}

/** One interface varying's packed position (both stages agree). */
export interface VaryingLayout {
  /** Fragment read index: ctx.varyings[i].v. */
  index: number;
  /** Vertex write offset: ctx.out.varyings[offset + c]. */
  offset: number;
  /** Total packed component count (array varyings expanded). */
  components: number;
  /** Component count of one array element (== components for non-arrays). */
  elemComponents: number;
  /** Flat interpolation (flat-qualified or integral type). */
  flat: boolean;
}

/**
 * Everything the codegen needs beyond the annotated AST. Built by the linker
 * from the two Shaders' ShaderInfo (see linker.ts) — the codegen itself never
 * computes layout.
 */
export interface CodegenLayout {
  /** Shader language version (100 | 300) — from the Shader's declared #version.
   *  Needed by codegen for builtin re-resolution (builtinSignatures(version))
   *  and for version-dependent call lowering (texture2D vs texture). The
   *  linker fills this from the Shader. */
  version: 100 | 300;
  /** Flattened default-block uniform paths: 'u', 'u.m', 'u[0]', 'u[0].m', 'u[2].m'. */
  uniformSlots: Map<string, UniformSlot>;
  /** Uniform-block layouts: block index → member path → byte layout. */
  blocks: Map<number, Map<string, BlockMemberLayout>>;
  /**
   * Block access name → block index. For INSTANCE-ARRAYED blocks
   * (`uniform B {..} b[2]`) the key is the instance name ('b'); for
   * INSTANCE-LESS blocks the keys are the bare MEMBER names ('m') — GLSL
   * accesses instance-less block members by bare name. Codegen resolves a
   * member identifier through this map; the member path keys then live in
   * `blocks.get(index)` (prefixed with the instance name + index for
   * instance arrays, e.g. 'b[0].m'). Arrayed blocks additionally need a
   * PREFIX entry 'b[0]' (offset 0, blockStride set) so codegen can resolve
   * dynamic instance indices.
   */
  blockIndices: Map<string, number>;
  /** Interface varyings by base name. */
  varyings: Map<string, VaryingLayout>;
  /** Vertex attributes: base name → first location (matC occupies C consecutive). */
  attribLocations: Map<string, number>;
  /** Fragment outputs: name → location ('gl_FragColor', 'gl_FragData', user outs). */
  outputLocations: Map<string, number>;
  /** Shader capability flags (ShaderInfo.uses) — `derivatives` drives dual mode. */
  uses: ShaderUses;
}

/** Result of generating one stage's function body. */
export interface StageCodegenResult {
  /** JS function BODY source (no wrapper); compiled via new Function('ctx','R', body). */
  body: string;
  /** Required ctx.scratch size in float elements (program.scratchSize = max over stages). */
  scratchSize: number;
  /** Required ctx.intScratch size in int elements. */
  intScratchSize: number;
}

/**
 * One JS expression: `v` = value; `dx`/`dy` = dual parts (dual mode only).
 *
 * `pre` = PURE materialization expressions (texture-sample chains, temp
 * assignments of multi-use pure math) that MUST run once, in order, before
 * `v`/`dx`/`dy` are evaluated. Every component of one multi-component result
 * shares the SAME pre-array instance; statement emitters dedupe pres by array
 * identity and hoist them before the statement's expression. Pres are always
 * PURE (no assignments to GLSL state, no out-param writes) — GLSL side
 * effects (assignments, out params) fold inline into `v` instead, so running
 * a pre unconditionally is always safe. The dual-mode task (C5) treats pre
 * the same way (pres run once; dx/dy strings reference the same temps).
 */
export interface Value {
  v: string;
  dx?: string;
  dy?: string;
  pre?: string[];
}

/** Generate the vertex stage body (writes ctx.out.position/pointSize/varyings). */
export function generateVertexStage(ast: TranslationUnit, layout: CodegenLayout): StageCodegenResult {
  throw new Error('not implemented');
}

/** Generate the fragment stage body (writes ctx.out.color[loc], ctx.out.fragDepth, ctx.discarded). */
export function generateFragmentStage(ast: TranslationUnit, layout: CodegenLayout): StageCodegenResult {
  throw new Error('not implemented');
}

/** The shared runtime object passed as the 2nd arg of new Function('ctx','R', body). */
export { R } from './runtime.js';
