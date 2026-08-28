/**
 * linker.ts — the program linker (cross-module contract §1: linker phase).
 *
 * `linkProgram(vs, fs, opts)` validates the shader pair and produces the
 * `Program` object gl/ and raster/ consume:
 *
 *   1. version compatibility + deferred-feature rejection (transform
 *      feedback, varying interface blocks — later tasks);
 *   2. uniform merge (same name in both stages must be type-identical) and
 *      default-block layout (conservative vec4-slot packing);
 *   3. uniform-block layout (WebGL2, std140): per-stage member offsets,
 *      shared-index merge for same-named blocks, blockIndices for codegen;
 *   4. varying matching (name / element type / array size / flat) and dense
 *      packing in VERTEX declaration order;
 *   5. attribute location assignment (explicit layout(location=) →
 *      bindAttribLocation → first-free automatic);
 *   6. fragment output location assignment;
 *   7. limit checks; JS codegen for both stages; Program assembly.
 *
 * UNIFORM STORE LAYOUT (must match codegen/env.ts exactly):
 * - ONE unified vec4-slot cursor for the whole default block. Each uniform
 *   occupies `slotCount` vec4 slots; `UniformSlot.slot` / `UniformInfo.location`
 *   = cursor*4 — a FLOAT index into the store (NOT vec4×4). A vec4 at location
 *   L occupies floats [L..L+3]; a matC occupies C*4 consecutive floats with
 *   column `col` at L + col*4 + row (column stride 4 — GLSL memory order).
 * - slotCount: scalar/vector/sampler → 1 slot; matC → C slots; struct → Σ
 *   member slots; array → element slots × size, EXCEPT scalar/sampler arrays
 *   which pack DENSELY (elements at consecutive floats, stride 1, footprint
 *   ceil(size/4) slots).
 * - int/uint/bool/sampler leaves live in the INT store (ctx.intUniforms) with
 *   the same slot numbering (1 int per component, stride 1). Both stores share
 *   the numeric slot space; each store is sized only to its own high-water
 *   mark (floatMax / intMax).
 * - Dynamic array indexing resolves through the '[0]' prefix key whose
 *   `stride` = FLOAT elements per element of the LAST dimension: scalar/
 *   sampler → 1, vector → 4, matrix → cols*4, struct → whole-struct slots,
 *   nested array → inner footprint.
 * - uniformSlots contains EVERY reachable path: all leaves ('u', 'u[0]',
 *   'u.m', 'u[0].m', 'u[2].m', 'u[0].m[1]') plus all ancestor prefixes
 *   ('u', 'u[0]') — codegen throws on missing entries.
 *
 * VARYING PACKING: dense flat components, cumulative offsets in vertex
 * declaration order (matched fs inputs read the same offsets). Struct varyings
 * flatten to per-member leaves ('v.m'), each with its own (index, offset).
 */
import type { Stmt, StructDefinition, StructMemberDecl, TranslationUnit } from './ast.js';
import type { LinkLimits, LinkOptions, LinkResult, Shader, ShaderUses, TransformFeedbackSpec, UniformBlockDecl, UniformDecl } from './compiler.js';
import type { AttribInfo, FragmentExecCtx, Program, TransformFeedbackVarying, UniformBlockInfo, UniformBlockMemberInfo, UniformInfo, VaryingInfo, VertexExecCtx } from './program.js';
import { generateFragmentStage, generateVertexStage, R } from './codegen/index.js';
import type { BlockMemberLayout, CodegenLayout, StageCodegenResult, UniformSlot, VaryingLayout } from './codegen/index.js';
import { flatComponents, isIntegralFamily } from './codegen/env.js';
import { isIntegral, isSampler, toGLenum, typeComponents, typeEquals, typeName } from './types.js';
import type { GLSLType, Precision, TypeQualifiers } from './types.js';

/* ------------------------------------------------------------------ */
/* Limits (WebGL minimums per version; gl/ passes its own via opts)    */
/* ------------------------------------------------------------------ */

const LIMITS_100: LinkLimits = {
  maxVertexAttribs: 16,
  maxVertexUniformVectors: 128,
  maxFragmentUniformVectors: 16,
  maxVaryingVectors: 8,
  maxVertexTextureImageUnits: 0,
  maxTextureImageUnits: 8,
  maxCombinedTextureImageUnits: 8,
  maxDrawBuffers: 1,
  maxUniformBufferBindings: 0,
  maxUniformBlockSize: 0,
  maxVertexUniformBlocks: 0,
  maxFragmentUniformBlocks: 0,
  maxCombinedUniformBlocks: 0,
  maxTransformFeedbackSeparateAttribs: 0,
  maxTransformFeedbackInterleavedComponents: 0,
  maxTransformFeedbackSeparateComponents: 0,
};

const LIMITS_300: LinkLimits = {
  maxVertexAttribs: 16,
  maxVertexUniformVectors: 256,
  maxFragmentUniformVectors: 224,
  maxVaryingVectors: 15,
  maxVertexTextureImageUnits: 16,
  maxTextureImageUnits: 16,
  maxCombinedTextureImageUnits: 32,
  maxDrawBuffers: 4,
  maxUniformBufferBindings: 24,
  maxUniformBlockSize: 16384,
  maxVertexUniformBlocks: 12,
  maxFragmentUniformBlocks: 12,
  maxCombinedUniformBlocks: 24,
  maxTransformFeedbackSeparateAttribs: 4,
  maxTransformFeedbackInterleavedComponents: 64,
  maxTransformFeedbackSeparateComponents: 4,
};

function resolveLimits(version: 100 | 300, opts?: LinkOptions): LinkLimits {
  return { ...(version === 100 ? LIMITS_100 : LIMITS_300), ...opts?.limits };
}

/* ------------------------------------------------------------------ */
/* Uniform layout helpers                                              */
/* ------------------------------------------------------------------ */

/** vec4 slots occupied by a type in the default-block store (conservative
 *  vec4-slot packing — never shares slots between uniforms). */
function slotCount(t: GLSLType): number {
  switch (t.kind) {
    case 'scalar':
    case 'vector':
    case 'sampler':
      return 1;
    case 'matrix':
      return t.cols;
    case 'struct':
      return t.members.reduce((n, m) => n + slotCount(m.type), 0);
    case 'array': {
      const n = t.size ?? 0;
      const e = t.element;
      // Scalar/sampler elements pack densely (stride 1 float); other elements
      // occupy whole vec4 slots each.
      if (e.kind === 'scalar' || e.kind === 'sampler') return Math.ceil(n / 4);
      return slotCount(e) * n;
    }
    case 'void':
      return 0;
  }
}

/**
 * Per-element rows for the API resource-limit accounting (GLSL ES 1.00
 * Appendix A packing rules, as CTS conformance/glsl/misc/
 * shader-uniform-packing-restrictions.html / shader-varying-packing-restrictions.html
 * demand): a scalar/vector element occupies ONE row, a matC element C rows
 * ("the spec says mat2 takes 4 columns, 2 rows"), structs sum their members,
 * arrays multiply the element rows by the element count. This is the API
 * LIMIT CHECK ONLY — it is deliberately independent of the (denser) packed
 * storage layout in slotCount/emitType: scalar arrays pack 4-per-vec4 in the
 * store, but the limit must still count every element (uniform float[4097]
 * exceeds MAX_*_UNIFORM_VECTORS 4096 even though it packs into 1025 slots).
 */
function limitRows(t: GLSLType): number {
  switch (t.kind) {
    case 'scalar':
    case 'vector':
    case 'sampler':
      return 1;
    case 'matrix':
      return t.cols;
    case 'struct':
      return t.members.reduce((n, m) => n + limitRows(m.type), 0);
    case 'array': {
      const n = t.size ?? 0;
      return limitRows(t.element) * n;
    }
    case 'void':
      return 0;
  }
}

/** FLOAT elements between consecutive elements of the LAST array dimension
 *  (the `stride` codegen applies for dynamic indices). Nested arrays align
 *  each outer element to its vec4-slot footprint (slotCount*4) so the dynamic
 *  stride agrees with the const-index leaf positions. */
function elemFloatStride(e: GLSLType): number {
  switch (e.kind) {
    case 'scalar':
    case 'sampler':
      return 1;
    case 'vector':
      return 4;
    case 'matrix':
      return e.cols * 4;
    case 'struct':
      return slotCount(e) * 4;
    case 'array':
      return slotCount(e) * 4;
    case 'void':
      return 0;
  }
}

/** True when the type lives in the INT store (int/uint/bool/sampler — the
 *  codegen's isIntegralFamily || sampler rule). */
function isIntStoreType(t: GLSLType): boolean {
  return isIntegralFamily(t) || isSampler(t);
}

/** Mutable allocation state shared by the flatten walk. */
interface AllocState {
  slots: Map<string, UniformSlot>;
  floatMax: number; // high-water float index (+1) in floatStore
  intMax: number; // high-water int index (+1) in intStore
}

/** Emit uniformSlots entries for `t` at `path`, starting at vec4-slot `cursor`.
 *  Returns the advance (vec4 slots) or an error. */
function emitType(path: string, t: GLSLType, cursor: number, st: AllocState): { advance: number } | { error: string } {
  switch (t.kind) {
    case 'scalar':
    case 'vector':
    case 'sampler':
    case 'matrix': {
      const slot = cursor * 4;
      const int = isIntStoreType(t);
      st.slots.set(path, { store: int ? 'int' : 'float', slot, stride: 0 });
      // Matrix storage is COLUMN-MAJOR with vec4 columns: each column occupies
      // 4 floats in the store (slot = cursor*4, column stride 4), so a matC
      // leaf needs cols*4 floats — cols*rows under-sizes the store (getUniform
      // then reads past the end → NaN for matrix arrays).
      const comps = t.kind === 'matrix' ? t.cols * 4 : t.kind === 'vector' ? t.size : 1;
      if (int) st.intMax = Math.max(st.intMax, slot + comps);
      else st.floatMax = Math.max(st.floatMax, slot + comps);
      return { advance: slotCount(t) };
    }
    case 'array': {
      const n = t.size ?? 0;
      const e = t.element;
      if (n === 0) return { advance: 0 };
      const dense = e.kind === 'scalar' || e.kind === 'sampler';
      const stride = elemFloatStride(e);
      const int = isIntStoreType(e);
      const base = cursor * 4;
      let advance: number;
      if (dense) {
        // Scalar/sampler arrays pack densely: element k at base + k floats.
        for (let k = 0; k < n; k++) {
          const slot = base + k;
          st.slots.set(`${path}[${k}]`, { store: int ? 'int' : 'float', slot, stride: 0 });
          if (int) st.intMax = Math.max(st.intMax, slot + 1);
          else st.floatMax = Math.max(st.floatMax, slot + 1);
        }
        advance = Math.ceil(n / 4);
      } else {
        // Non-scalar elements (vector/matrix/struct/nested array): one whole
        // (aligned) element block per index; recursion places the inner leaves.
        advance = 0;
        for (let k = 0; k < n; k++) {
          const r = emitType(`${path}[${k}]`, e, (base + k * stride) / 4, st);
          if ('error' in r) return r;
          advance += r.advance;
        }
      }
      // '[0]' prefix: dynamic-index resolution (codegen reads its stride).
      // MUST be set AFTER the element entries — the element-0 leaf entry
      // ('u[0]' for vector/matrix/scalar/sampler elements, 'u[0][0]' for
      // nested arrays) would otherwise overwrite it with stride 0, so every
      // dynamically indexed element read would resolve to element 0 (BUG d:
      // `uni[ii]` on `uniform vec4 uni[8]` read only element 0). Struct
      // elements self-key with elemFloatStride, so they never collide.
      st.slots.set(`${path}[0]`, { store: int ? 'int' : 'float', slot: base, stride });
      return { advance };
    }
    case 'struct': {
      // Members are laid out sequentially (each at its own slot advance);
      // the struct itself gets an ancestor-prefix entry ('u', 'u[0]').
      let c = cursor;
      for (const m of t.members) {
        const r = emitType(`${path}.${m.name}`, m.type, c, st);
        if ('error' in r) return r;
        c += r.advance;
      }
      st.slots.set(path, { store: isIntStoreType(t) ? 'int' : 'float', slot: cursor * 4, stride: elemFloatStride(t) });
      return { advance: c - cursor };
    }
    case 'void':
      return { advance: 0 };
  }
}

/** One merged default-block uniform (same name in both stages). */
interface MergedUniform {
  /** The vertex-stage declaration when declared in both stages (the first
   *  seen); the fragment twin is kept separately for precision/struct checks. */
  decl: UniformDecl;
  fsDecl: UniformDecl | null;
  inVs: boolean;
  inFs: boolean;
}

/* ------------------------------------------------------------------ */
/* Cross-stage precision + struct-identity matching                    */
/* ------------------------------------------------------------------ */

/**
 * Replay the stage's `precision` statements to get the effective default
 * precision map (GLSL ES §4.5.3). NOTE the deliberate divergence from
 * semantics' defaultPrecisions: semantics seeds the VERTEX int default as
 * 'mediump', but the SPEC default is highp, and
 * shader-with-global-variable-precision-mismatch.html depends on the spec
 * default (VS `uniform int foo;` = highp vs FS default mediump must fail).
 * Sampler defaults are never consulted (sampler precision is exempt from the
 * comparison).
 */
function stageDefaultPrecisions(shader: Shader): Map<string, Precision> {
  const m = new Map<string, Precision>();
  if (shader.type === 'VERTEX') {
    m.set('float', 'highp');
    m.set('int', 'highp');
  } else {
    m.set('int', 'mediump');
  }
  for (const d of shader.ast.declarations) {
    if (d.kind === 'precision-decl') m.set(d.base, d.precision);
  }
  return m;
}

/** Effective precision of a type: the explicit qualifier, else the default in
 *  effect at the declaration (arrays unwrap to their element; uint shares the
 *  int default). bool/sampler/struct → null (bool has no precision; sampler
 *  precision is exempt from the cross-stage comparison). */
function effectivePrecision(explicit: Precision | undefined, t: GLSLType, defaults: Map<string, Precision>): Precision | null {
  if (explicit !== undefined) return explicit;
  let e = t;
  while (e.kind === 'array') e = e.element;
  switch (e.kind) {
    case 'scalar':
    case 'vector':
      return e.base === 'float' || e.base === 'int' || e.base === 'uint'
        ? (defaults.get(e.base === 'uint' ? 'int' : e.base) ?? null)
        : null;
    case 'matrix':
      return defaults.get('float') ?? null;
    default:
      return null; // sampler / struct / bool
  }
}

/**
 * Locate the definition of user struct `name` in one stage's AST (either a
 * bare `struct S { ... };` or an inline `struct S { ... } v;` definition) and
 * return the effective precision of each member, in declaration order, or
 * null when the struct is not defined in that stage. Struct-member precision
 * is NOT part of ShaderInfo (UniformDecl carries only the uniform's own
 * precision) — the AST qualifiers + replayed defaults are the only source.
 */
function structMemberPrecisions(shader: Shader, structName: string): (Precision | null)[] | null {
  const defaults = stageDefaultPrecisions(shader);
  for (const d of shader.ast.declarations) {
    let members: StructMemberDecl[] | null = null;
    if (d.kind === 'struct-decl' && d.name === structName) members = d.members;
    else if (d.kind === 'global-var-decl' && d.type.base.kind === 'struct-definition' && d.type.base.name === structName) {
      members = d.type.base.members;
    }
    if (members !== null) {
      return members.map((m) => effectivePrecision(m.type.qualifiers.precision, m.type.resolved ?? { kind: 'void' }, defaults));
    }
  }
  return null;
}

/**
 * Unwrap every array layer of a type (struct arrays compare their element
 * struct; nested arrays are legal in ES 3.00).
 */
function unwrapArrays(t: GLSLType): GLSLType {
  let e = t;
  while (e.kind === 'array') e = e.element;
  return e;
}

/**
 * Member-level structural identity for matched struct uniforms (GLSL ES 1.00
 * §4.2.4 — same name, same sequence of type names, same type definitions and
 * field names; the CTS also requires matching member precision). The top-level
 * `typeEquals` check only compares struct NAMES, so two same-named structs
 * with different members link today — this rejects them. `a` is the vertex
 * decl, `b` the fragment decl (both have the SAME struct name — typeEquals
 * passed). Returns an error string or null when the definitions agree.
 */
function structUniformConflict(a: UniformDecl, b: UniformDecl, vs: Shader, fs: Shader): string | null {
  const sa = unwrapArrays(a.type);
  const sb = unwrapArrays(b.type);
  if (sa.kind !== 'struct' || sb.kind !== 'struct') return null; // typeEquals already rejected
  if (sa.members.length !== sb.members.length) {
    return `linker: uniform '${a.name}' struct '${sa.name}' member count mismatch`;
  }
  const pa = structMemberPrecisions(vs, sa.name);
  const pb = structMemberPrecisions(fs, sb.name);
  for (let i = 0; i < sa.members.length; i++) {
    const ma = sa.members[i];
    const mb = sb.members[i];
    if (ma.name !== mb.name) {
      return `linker: uniform '${a.name}' struct '${sa.name}' member name mismatch ('${ma.name}' vs '${mb.name}')`;
    }
    if (!typeEquals(ma.type, mb.type)) {
      return `linker: uniform '${a.name}' struct '${sa.name}' member '${ma.name}' type mismatch`;
    }
    const precA = pa !== null && pa.length === sa.members.length ? pa[i] : null;
    const precB = pb !== null && pb.length === sb.members.length ? pb[i] : null;
    if (precA !== null && precB !== null && precA !== precB) {
      return `linker: uniform '${a.name}' struct '${sa.name}' member '${ma.name}' precision mismatch (${precA} vs ${precB})`;
    }
  }
  return null;
}

/**
 * Effective precision of a top-level default-block uniform: the explicit
 * qualifier on its declaration (AST), else the stage default for its base
 * type. Samplers/bool return null (exempt / no precision).
 */
function uniformPrecision(shader: Shader, name: string, type: GLSLType): Precision | null {
  const defaults = stageDefaultPrecisions(shader);
  for (const d of shader.ast.declarations) {
    if (d.kind !== 'global-var-decl') continue;
    for (const dec of d.declarators) {
      if (dec.name === name) return effectivePrecision(d.type.qualifiers.precision, type, defaults);
    }
  }
  // No AST declaration found (should not happen for a merged uniform) — fall
  // back to the recorded effective precision.
  return null;
}

/** Cross-stage precision consistency for one matched default-block uniform:
 *  float/int/uint types compare their effective precision; structs compare
 *  member-by-member (name/type/precision); samplers and bools are exempt. */
function uniformPrecisionConflict(a: UniformDecl, b: UniformDecl, vs: Shader, fs: Shader): string | null {
  const ta = unwrapArrays(a.type);
  if (ta.kind === 'struct') {
    return structUniformConflict(a, b, vs, fs);
  }
  const pa = uniformPrecision(vs, a.name, a.type);
  const pb = uniformPrecision(fs, b.name, b.type);
  if (pa !== null && pb !== null && pa !== pb) {
    return `linker: uniform '${a.name}' precision mismatch (${pa} vs ${pb})`;
  }
  return null;
}

/** Combine vs + fs default-block uniforms by name; same name must be
 *  type-identical in both stages (GLSL link rule) and, when the type is
 *  float/int/uint or a struct, must carry the same precision (structs compare
 *  member-by-member — see uniformPrecisionConflict). */
function mergeUniforms(vs: Shader, fs: Shader): MergedUniform[] | { error: string } {
  const byName = new Map<string, MergedUniform>();
  const order: string[] = [];
  const add = (decl: UniformDecl, stage: 'vs' | 'fs'): string | null => {
    const ex = byName.get(decl.name);
    if (ex) {
      if (!typeEquals(ex.decl.type, decl.type)) {
        return `linker: uniform '${decl.name}' type conflict (${typeName(ex.decl.type)} vs ${typeName(decl.type)})`;
      }
      if (stage === 'vs') {
        ex.inVs = true;
      } else {
        ex.inFs = true;
        ex.fsDecl = decl;
        // Both stages now known: cross-stage precision/struct-identity check.
        return uniformPrecisionConflict(ex.decl, decl, vs, fs);
      }
      return null;
    }
    byName.set(decl.name, { decl, fsDecl: stage === 'fs' ? decl : null, inVs: stage === 'vs', inFs: stage === 'fs' });
    order.push(decl.name);
    return null;
  };
  for (const d of vs.info.uniforms) {
    const e = add(d, 'vs');
    if (e !== null) return { error: e };
  }
  for (const d of fs.info.uniforms) {
    const e = add(d, 'fs');
    if (e !== null) return { error: e };
  }
  return order.map((n) => byName.get(n)!);
}

/** Visit every flattened LEAF path of a type in allocation order (struct
 *  members recurse; TOP-LEVEL array elements expand per index — matching
 *  emitType's uniformSlots keys). STRUCT MEMBER arrays stay WHOLE: one leaf
 *  '<p>.m[0]' carrying the array type, so getActiveUniform reports a single
 *  entry with size = array length — only the top-level array dimension expands
 *  (CTS shader-with-array-of-structs-containing-arrays.html expects 4 entries
 *  for `my_struct { vec4 color1[2]; vec4 color2[2]; } u_colors[2];`, not 8).
 *  Arrays whose element chain leads to a struct still expand per element: a
 *  struct-array member has no GLenum, so it cannot be reported as one entry
 *  (leafInfo would throw on toGLenum(struct)). */
function walkLeaves(path: string, t: GLSLType, cb: (path: string, leaf: GLSLType) => void): void {
  switch (t.kind) {
    case 'scalar':
    case 'vector':
    case 'sampler':
    case 'matrix':
      cb(path, t);
      return;
    case 'struct':
      for (const m of t.members) walkMember(`${path}.${m.name}`, m.type, cb);
      return;
    case 'array': {
      const n = t.size ?? 0;
      for (let k = 0; k < n; k++) walkLeaves(`${path}[${k}]`, t.element, cb);
      return;
    }
    case 'void':
      return;
  }
}

/** Member walk: non-array types recurse through walkLeaves (nested structs
 *  keep expanding); arrays of leaf types (scalar/vector/matrix/sampler) stay
 *  whole as ONE leaf at '<p>.m[0]' — getActiveUniform expands only the
 *  top-level array dimension of a struct uniform. Struct-element arrays expand
 *  per element (see walkLeaves). */
function walkMember(path: string, t: GLSLType, cb: (path: string, leaf: GLSLType) => void): void {
  if (t.kind === 'array' && unwrapArrays(t).kind !== 'struct') {
    cb(`${path}[0]`, t);
    return;
  }
  walkLeaves(path, t, cb);
}

interface UniformLayoutResult {
  slots: Map<string, UniformSlot>;
  uniforms: UniformInfo[];
  uniformMap: Map<string, UniformInfo>;
  floatMax: number;
  intMax: number;
}

/** Allocate the default-block stores for the merged uniform set (vs order
 *  first, then fs-only), building uniformSlots + Program.uniforms +
 *  Program.uniformMap, and checking the per-stage uniform-vector limits. */
function layoutUniforms(merged: MergedUniform[], limits: LinkLimits): UniformLayoutResult | { error: string } {
  const st: AllocState = { slots: new Map(), floatMax: 0, intMax: 0 };
  const uniforms: UniformInfo[] = [];
  const uniformMap = new Map<string, UniformInfo>();
  let cursor = 0;
  let vertexSlots = 0;
  let fragmentSlots = 0;
  for (const u of merged) {
    const t = u.decl.type;
    // API limit accounting is PER ELEMENT (limitRows), not per packed slot:
    // a 4097-element scalar array must exceed 4096 uniform vectors even
    // though the store packs it densely (see limitRows).
    const rows = limitRows(t);
    if (u.inVs) vertexSlots += rows;
    if (u.inFs) fragmentSlots += rows;
    const r = emitType(u.decl.name, t, cursor, st);
    if ('error' in r) return { error: r.error };
    // Ancestor prefix for the root name (arrays/structs; leaves already keyed it).
    if (!st.slots.has(u.decl.name)) {
      st.slots.set(u.decl.name, { store: isIntStoreType(t) ? 'int' : 'float', slot: cursor * 4, stride: 0 });
    }
    const leaves: { path: string; type: GLSLType }[] = [];
    walkLeaves(u.decl.name, t, (p, lt) => leaves.push({ path: p, type: lt }));
    const leafInfo = (path: string, type: GLSLType, size: number): UniformInfo => ({
      name: path,
      location: st.slots.get(path)!.slot,
      type: toGLenum(type),
      size,
      components: typeComponents(type),
      integral: isIntegral(type),
      blockIndex: -1,
      sampler: isSampler(type),
    });
    const isStructRoot = t.kind === 'struct' || (t.kind === 'array' && t.element.kind === 'struct');
    if (!isStructRoot) {
      // Plain (non-struct) uniform: ONE getActiveUniform entry — non-arrays
      // 'u', arrays 'u[0]' with size = array length. uniformMap holds
      // per-element infos + the bare name → first element.
      const e = t.kind === 'array' ? t.element : t;
      uniforms.push(
        leafInfo(t.kind === 'array' ? `${u.decl.name}[0]` : u.decl.name, e, t.kind === 'array' ? t.size ?? 1 : 1),
      );
      for (const l of leaves) uniformMap.set(l.path, leafInfo(l.path, l.type, 1));
      uniformMap.set(u.decl.name, uniformMap.get(leaves[0].path)!);
    } else {
      // Struct / struct-array: getActiveUniform entries per flattened leaf
      // ('u.m', 'u[0].m', 'u[2].m'). MEMBER arrays stay whole: one entry
      // '<p>.m[0]' with size = array length (only the TOP-LEVEL array expands —
      // CTS shader-with-array-of-structs-containing-arrays.html). uniformMap:
      // leaf paths + bare names ('u', 'u[0]', '<p>.m') → first leaf; member
      // array elements '<p>.m[k]' (k < size) alias the SAME leaf — gl's
      // getUniformLocation derives the element from the QUERY name.
      for (const l of leaves) {
        const size = l.type.kind === 'array' ? l.type.size ?? 1 : 1;
        const info = leafInfo(l.path, l.type, size);
        uniforms.push(info);
        uniformMap.set(l.path, info);
        if (l.type.kind === 'array') {
          const bare = l.path.slice(0, -3); // strip the trailing '[0]'
          uniformMap.set(bare, info);
          for (let k = 0; k < size; k++) uniformMap.set(`${bare}[${k}]`, info);
        }
      }
      const first = uniformMap.get(leaves[0].path)!;
      uniformMap.set(u.decl.name, first);
      if (t.kind === 'array') uniformMap.set(`${u.decl.name}[0]`, first);
    }
    cursor += r.advance;
  }
  if (vertexSlots > limits.maxVertexUniformVectors) {
    return {
      error: `linker: too many vertex uniform vectors (${vertexSlots}, max ${limits.maxVertexUniformVectors})`,
    };
  }
  if (fragmentSlots > limits.maxFragmentUniformVectors) {
    return {
      error: `linker: too many fragment uniform vectors (${fragmentSlots}, max ${limits.maxFragmentUniformVectors})`,
    };
  }
  return { slots: st.slots, uniforms, uniformMap, floatMax: st.floatMax, intMax: st.intMax };
}

/* ------------------------------------------------------------------ */
/* Uniform blocks (WebGL2, std140)                                     */
/* ------------------------------------------------------------------ */

/** roundUp(x, align) — std140 alignment rounding (align is a power of two). */
function roundUp(x: number, align: number): number {
  if (align <= 0) return x;
  const r = x % align;
  return r === 0 ? x : x + align - r;
}

/** std140 base alignment (bytes): scalar 4, vec2 8, vec3/vec4 16, matrix 16,
 *  array = element alignment, struct = max member alignment. Row-major does
 *  not change alignments (matrix rows are vec4-aligned either way). */
function std140Align(t: GLSLType): number {
  switch (t.kind) {
    case 'scalar':
    case 'sampler':
      return 4;
    case 'vector':
      return t.size === 2 ? 8 : 16;
    case 'matrix':
      return 16;
    case 'array':
      return std140Align(t.element);
    case 'struct': {
      let a = 0;
      for (const m of t.members) a = Math.max(a, std140Align(m.type));
      return a === 0 ? 4 : a;
    }
    case 'void':
      return 4;
  }
}

/** std140 size of a type: scalar 4, vector 4×n (vec3 = 12), matrix 16×cols
 *  column-major / 16×rows row-major (stride 16 either way), array = count ×
 *  element stride, struct = members at aligned offsets, rounded to the struct
 *  align. `rowMajor` applies to every matrix inside the type (a block member
 *  declared `layout(row_major)` carries it into struct members too). */
function std140Size(t: GLSLType, rowMajor: boolean): number {
  switch (t.kind) {
    case 'scalar':
    case 'sampler':
      return 4;
    case 'vector':
      return 4 * t.size;
    case 'matrix':
      return 16 * (rowMajor ? t.rows : t.cols);
    case 'array': {
      const n = t.size ?? 0;
      return n * std140ArrayStride(t.element, rowMajor);
    }
    case 'struct': {
      let off = 0;
      for (const m of t.members) {
        off = roundUp(off, std140Align(m.type));
        off += std140Size(m.type, rowMajor);
      }
      return roundUp(off, std140Align(t));
    }
    case 'void':
      return 0;
  }
}

/** std140 stride between consecutive array elements: roundUp(element size,
 *  element alignment). */
function std140ArrayStride(t: GLSLType, rowMajor: boolean): number {
  return roundUp(std140Size(t, rowMajor), std140Align(t));
}

/** Emit BlockMemberLayout entries for type `t` at `byteOffset`, keyed by
 *  `path`. Emits the path root, every nested member leaf, every const-indexed
 *  array element, and the '[0]' dynamic-index prefix (arrayStride set).
 *  When `blockStride` is set (arrayed block) it is stamped on EVERY entry —
 *  codegen resolves any dynamic index inside an arrayed block through it.
 *  `rowMajor` is the member's EFFECTIVE layout qualifier (member-level
 *  overrides block-level; default column-major) — stamped on every entry so
 *  codegen's matrix element math picks the right byte order. */
function emitBlockLayout(
  path: string,
  t: GLSLType,
  byteOffset: number,
  out: Map<string, BlockMemberLayout>,
  blockStride: number | undefined,
  rowMajor: boolean,
): void {
  const mk = (offset: number, arrayStride: number, matrixStride: number): BlockMemberLayout => {
    const base = { offset, arrayStride, matrixStride, rowMajor };
    return blockStride === undefined ? base : { ...base, blockStride };
  };
  switch (t.kind) {
    case 'scalar':
    case 'vector':
    case 'sampler':
      out.set(path, mk(byteOffset, 0, 0));
      return;
    case 'matrix':
      out.set(path, mk(byteOffset, 0, 16));
      return;
    case 'array': {
      const n = t.size ?? 0;
      const stride = std140ArrayStride(t.element, rowMajor);
      out.set(path, mk(byteOffset, stride, 0));
      if (n > 0) {
        for (let k = 0; k < n; k++) {
          emitBlockLayout(`${path}[${k}]`, t.element, byteOffset + k * stride, out, blockStride, rowMajor);
        }
        // '[0]' prefix AFTER the element entries — the element-0 leaf entry
        // ('u[0]') would otherwise overwrite its arrayStride with 0, so every
        // dynamically indexed element read would resolve to element 0 (BUG d,
        // block twin). matrixStride keeps the element's column/row stride
        // (matrix elements: const-index reads of element 0 use the same entry).
        out.set(`${path}[0]`, mk(byteOffset, stride, t.element.kind === 'matrix' ? 16 : 0));
      }
      return;
    }
    case 'struct': {
      out.set(path, mk(byteOffset, 0, 0));
      let off = 0;
      for (const m of t.members) {
        off = roundUp(off, std140Align(m.type));
        emitBlockLayout(`${path}.${m.name}`, m.type, byteOffset + off, out, blockStride, rowMajor);
        off += std140Size(m.type, rowMajor);
      }
      return;
    }
    case 'void':
      return;
  }
}

/** One flattened block leaf (getActiveUniform semantics: structs descend,
 *  arrays stop as ONE entry with size = array length, name '[0]'-suffixed). */
interface BlockLeaf {
  path: string;
  type: GLSLType;
  offset: number;
  arrayStride: number;
  matrixStride: number;
  rowMajor: boolean;
  size: number;
}

/** Descend the members of struct `t` at `path`/`byteOffset` (member offsets
 *  come from the std140 walk). Never emits a leaf for the struct itself. */
function descendStructLeaves(
  path: string,
  t: Extract<GLSLType, { kind: 'struct' }>,
  byteOffset: number,
  out: BlockLeaf[],
  rowMajor: boolean,
): void {
  let off = 0;
  for (const m of t.members) {
    off = roundUp(off, std140Align(m.type));
    collectBlockLeaves(`${path}.${m.name}`, m.type, byteOffset + off, out, rowMajor);
    off += std140Size(m.type, rowMajor);
  }
}

/** Collect the flattened leaves of member type `t` at `path`/`byteOffset`.
 *  INVARIANT: no leaf ever carries a struct type — arrays of structs expand
 *  EVERY element into per-member leaves ('lights[0].intensity', ...), so
 *  toGLenum/typeComponents/isIntegral/typeName never see a struct. */
function collectBlockLeaves(
  path: string,
  t: GLSLType,
  byteOffset: number,
  out: BlockLeaf[],
  rowMajor: boolean,
): void {
  if (t.kind === 'struct') {
    descendStructLeaves(path, t, byteOffset, out, rowMajor);
    return;
  }
  if (t.kind === 'array') {
    if (t.element.kind === 'struct') {
      // Array of structs: one leaf group PER ELEMENT at path `${path}[k]`
      // (getActiveUniform convention — a struct has no GLenum, so a single
      // '[0]' entry cannot represent it). Nested array members recurse and
      // keep their own '[0]' leaf with size = array length.
      const n = t.size ?? 0;
      const stride = std140ArrayStride(t.element, rowMajor);
      for (let k = 0; k < n; k++) {
        descendStructLeaves(`${path}[${k}]`, t.element, byteOffset + k * stride, out, rowMajor);
      }
      return;
    }
    out.push({
      path: `${path}[0]`,
      type: t.element,
      offset: byteOffset,
      arrayStride: std140ArrayStride(t.element, rowMajor),
      matrixStride: 0,
      rowMajor,
      size: t.size ?? 0,
    });
    return;
  }
  out.push({
    path,
    type: t,
    offset: byteOffset,
    arrayStride: 0,
    matrixStride: t.kind === 'matrix' ? 16 : 0,
    rowMajor,
    size: 1,
  });
}

/** The full std140 layout of ONE block declaration. */
interface BlockLayout {
  blockName: string;
  instanceName: string | null;
  arraySize: number;
  /** True when the instance declares an explicit array dim (`b[1]` IS arrayed). */
  instanceArray: boolean;
  /** std140 byte size of ONE block instance (rounded up to 16). */
  size: number;
  /** Member path → byte layout (codegen's `blocks.get(index)` map). */
  members: Map<string, BlockMemberLayout>;
  /** Flattened leaves per block instance element (one group for non-arrayed). */
  leafGroups: BlockLeaf[][];
  /** The declaration's members (for the type-identity check on shared blocks). */
  declMembers: UniformBlockDecl['members'];
}

/**
 * Compute the std140 layout of one `UniformBlockDecl`. Member paths follow
 * the codegen contract (codegen/index.ts):
 * - instance-less blocks: bare member names ('m', 'm[0]', 'm.s.x');
 * - named non-arrayed blocks: instance name + member ('b.m');
 * - arrayed blocks (instanceArray, size ≥ 1): per-element paths
 *   ('b[0].m', 'b[1].m') with every entry stamped blockStride = block size,
 *   plus the 'b[0]' PREFIX entry {offset 0, blockStride} for dynamic instance
 *   indexing.
 * Row-major: a block-level `layout(row_major)` applies to every member; a
 * member-level `layout(row_major)`/`layout(column_major)` overrides it for
 * that member (ES 3.00 §4.3.9). Row-major matrices occupy 16 bytes PER ROW
 * (mat4x3 = 48 bytes) with matrixStride 16 (the row stride); column-major
 * occupy 16 bytes per column. Codegen bakes the resulting offsets/stride and
 * the `rowMajor` flag into the matrix element reads.
 */
function layoutBlock(decl: UniformBlockDecl): BlockLayout {
  const memberRowMajor = decl.members.map((m) => m.rowMajor ?? decl.rowMajor);
  const memberOffsets: number[] = [];
  let off = 0;
  for (let i = 0; i < decl.members.length; i++) {
    off = roundUp(off, std140Align(decl.members[i].type));
    memberOffsets.push(off);
    off += std140Size(decl.members[i].type, memberRowMajor[i]);
  }
  const blockSize = roundUp(off, 16);
  const members = new Map<string, BlockMemberLayout>();
  const leafGroups: BlockLeaf[][] = [];
  const isArrayed = decl.instanceArray;
  const emitMember = (prefix: string, baseOffset: number, blockStride: number | undefined, leaves: BlockLeaf[]): void => {
    for (let i = 0; i < decl.members.length; i++) {
      const m = decl.members[i];
      const path = `${prefix}${m.name}`;
      emitBlockLayout(path, m.type, baseOffset + memberOffsets[i], members, blockStride, memberRowMajor[i]);
      collectBlockLeaves(path, m.type, baseOffset + memberOffsets[i], leaves, memberRowMajor[i]);
    }
  };
  if (isArrayed) {
    members.set(`${decl.instanceName}[0]`, {
      offset: 0,
      arrayStride: 0,
      matrixStride: 0,
      rowMajor: false,
      blockStride: blockSize,
    });
    for (let k = 0; k < decl.arraySize; k++) {
      const leaves: BlockLeaf[] = [];
      emitMember(`${decl.instanceName}[${k}].`, k * blockSize, blockSize, leaves);
      leafGroups.push(leaves);
    }
  } else if (decl.instanceName !== null) {
    const leaves: BlockLeaf[] = [];
    emitMember(`${decl.instanceName}.`, 0, undefined, leaves);
    leafGroups.push(leaves);
  } else {
    const leaves: BlockLeaf[] = [];
    emitMember('', 0, undefined, leaves);
    leafGroups.push(leaves);
  }
  return {
    blockName: decl.name,
    instanceName: decl.instanceName,
    arraySize: decl.arraySize,
    instanceArray: isArrayed,
    size: blockSize,
    members,
    leafGroups,
    declMembers: decl.members,
  };
}

/** Two blocks with the same name must have IDENTICAL layouts (member path
 *  sets, every byte offset/stride, the member types AND their precision — a
 *  `mediump vec4` vs a `highp vec4` member is a mismatch (ES 3.00 §4.3.7;
 *  shader-with-mis-matching-uniform-block.html) and their row-major flags)
 *  to share an index. */
function blockLayoutsEqual(a: BlockLayout, b: BlockLayout): boolean {
  if (a.size !== b.size || a.members.size !== b.members.size) return false;
  if (a.declMembers.length !== b.declMembers.length) return false;
  for (let i = 0; i < a.declMembers.length; i++) {
    const ma = a.declMembers[i];
    const mb = b.declMembers[i];
    if (ma.name !== mb.name || !typeEquals(ma.type, mb.type) || ma.rowMajor !== mb.rowMajor) {
      return false;
    }
    if (ma.precision !== mb.precision) {
      return false;
    }
  }
  for (const [k, ea] of a.members) {
    const eb = b.members.get(k);
    if (
      eb === undefined ||
      ea.offset !== eb.offset ||
      ea.arrayStride !== eb.arrayStride ||
      ea.matrixStride !== eb.matrixStride ||
      ea.rowMajor !== eb.rowMajor ||
      (ea.blockStride ?? 0) !== (eb.blockStride ?? 0)
    ) {
      return false;
    }
  }
  return true;
}

interface BlockLayoutResult {
  blocks: Map<number, Map<string, BlockMemberLayout>>;
  blockIndices: Map<string, number>;
  infos: UniformBlockInfo[];
  /** Block members as Program.uniforms entries (location -1, blockIndex ≥ 0). */
  uniformInfos: UniformInfo[];
}

/**
 * Compute the program's uniform-block layout: vs blocks first (declaration
 * order), then fs-only blocks; same-named blocks must have identical layouts
 * and SHARE one index. Checks per-stage block counts, block sizes and the
 * combined count against the limits. Block members are kept out of
 * uniformSlots/uniformMap (getUniformLocation returns null for them — spec).
 */
function layoutBlocks(vs: Shader, fs: Shader, limits: LinkLimits): BlockLayoutResult | { error: string } {
  const byName = new Map<string, BlockLayout>();
  const order: string[] = [];
  let vsCount = 0;
  let fsCount = 0;
  let combined = 0;
  const addStage = (shader: Shader, stage: 'vs' | 'fs'): string | null => {
    for (const decl of shader.info.uniformBlocks) {
      const lay = layoutBlock(decl);
      if (stage === 'vs') vsCount++;
      else fsCount++;
      if (lay.size > limits.maxUniformBlockSize) {
        return `linker: uniform block '${decl.name}' exceeds maxUniformBlockSize (${lay.size}, max ${limits.maxUniformBlockSize})`;
      }
      const ex = byName.get(decl.name);
      if (ex !== undefined) {
        if (!blockLayoutsEqual(ex, lay)) {
          return `linker: uniform block '${decl.name}' layout mismatch`;
        }
      } else {
        byName.set(decl.name, lay);
        order.push(decl.name);
        combined++;
      }
    }
    return null;
  };
  const ev = addStage(vs, 'vs');
  if (ev !== null) return { error: ev };
  const ef = addStage(fs, 'fs');
  if (ef !== null) return { error: ef };
  if (vsCount > limits.maxVertexUniformBlocks) {
    return { error: `linker: too many vertex uniform blocks (${vsCount}, max ${limits.maxVertexUniformBlocks})` };
  }
  if (fsCount > limits.maxFragmentUniformBlocks) {
    return { error: `linker: too many fragment uniform blocks (${fsCount}, max ${limits.maxFragmentUniformBlocks})` };
  }
  if (combined > limits.maxCombinedUniformBlocks) {
    return { error: `linker: too many uniform blocks (${combined}, max ${limits.maxCombinedUniformBlocks})` };
  }

  const blocks = new Map<number, Map<string, BlockMemberLayout>>();
  const blockIndices = new Map<string, number>();
  const infos: UniformBlockInfo[] = [];
  const uniformInfos: UniformInfo[] = [];
  for (let idx = 0; idx < order.length; idx++) {
    const lay = byName.get(order[idx])!;
    blocks.set(idx, lay.members);
    if (lay.instanceName !== null) {
      // Named blocks (arrayed or not): the instance name resolves the block.
      blockIndices.set(lay.instanceName, idx);
    } else {
      // Instance-less blocks: every member ROOT name resolves the block.
      for (const m of lay.members.keys()) {
        const root = m.split('.')[0].split('[')[0];
        blockIndices.set(root, idx);
      }
    }
    const memberInfo = (l: BlockLeaf): UniformBlockMemberInfo => ({
      name: l.path,
      offset: l.offset,
      type: toGLenum(l.type),
      size: l.size,
      arrayStride: l.arrayStride,
      matrixStride: l.matrixStride,
      rowMajor: l.rowMajor,
    });
    const leafInfos = (group: BlockLeaf[]): UniformBlockMemberInfo[] => group.map(memberInfo);
    if (lay.instanceArray) {
      // Arrayed blocks: one UniformBlockInfo PER ELEMENT ('b[0]', 'b[1]'...).
      for (let k = 0; k < lay.arraySize; k++) {
        infos.push({ name: `${lay.instanceName}[${k}]`, index: idx, size: lay.size, activeUniforms: leafInfos(lay.leafGroups[k]) });
      }
      for (const group of lay.leafGroups) {
        for (const l of group) {
          uniformInfos.push({
            name: l.path,
            location: -1,
            type: toGLenum(l.type),
            size: l.size,
            components: typeComponents(l.type),
            integral: isIntegral(l.type),
            blockIndex: idx,
            sampler: false,
          });
        }
      }
    } else {
      // Non-arrayed blocks: one UniformBlockInfo named by the BLOCK name
      // (getUniformBlockIndex/getActiveUniformBlockName query block names).
      infos.push({ name: lay.blockName, index: idx, size: lay.size, activeUniforms: leafInfos(lay.leafGroups[0]) });
      for (const l of lay.leafGroups[0]) {
        uniformInfos.push({
          name: l.path,
          location: -1,
          type: toGLenum(l.type),
          size: l.size,
          components: typeComponents(l.type),
          integral: isIntegral(l.type),
          blockIndex: idx,
          sampler: false,
        });
      }
    }
  }
  return { blocks, blockIndices, infos, uniformInfos };
}

/* ------------------------------------------------------------------ */
/* Varying matching + packing                                          */
/* ------------------------------------------------------------------ */

/** One packed varying leaf (plain varyings are one leaf; struct varyings
 *  flatten to one leaf per member). `type` is the element type (arrays carry
 *  their size in `arraySize` for plain varyings and inside `type` for struct
 *  members). */
interface VaryingLeaf {
  key: string;
  type: GLSLType;
  arraySize: number;
}

/** Flatten a struct varying into per-member leaves ('v.m'; nested structs
 *  recurse; member arrays keep their array structure inside the type). */
function flattenVaryingStruct(prefix: string, t: GLSLType, out: VaryingLeaf[]): void {
  if (t.kind !== 'struct') {
    out.push({ key: prefix, type: t, arraySize: 1 });
    return;
  }
  for (const m of t.members) flattenVaryingStruct(`${prefix}.${m.name}`, m.type, out);
}

interface VaryingLayoutResult {
  map: Map<string, VaryingLayout>;
  infos: VaryingInfo[];
}

/** Names of the `invariant <name>;` SHORT-FORM declarations in a shader
 *  (parser.ts produces 'invariant-decl' AST nodes; semantics gives them no
 *  ShaderInfo effect — the qualifier form `invariant varying vec4 v;` is
 *  already recorded on VaryingDecl.invariant, so this only adds the short
 *  form: `varying vec4 v; invariant v;`). */
function invariantDeclNames(s: Shader): Set<string> {
  const names = new Set<string>();
  for (const d of s.ast.declarations) {
    if (d.kind === 'invariant-decl') names.add(d.name);
  }
  return names;
}

/** True when the shader source contains `#pragma STDGL invariant(all)`.
 *  The preprocessor DROPS pragma lines entirely (preprocessor.ts case
 *  'pragma'), so the directive is undetectable in the AST — scan the
 *  ORIGINAL source (Shader.source) instead. */
function hasInvariantAllPragma(s: Shader): boolean {
  return /^\s*#\s*pragma\s+STDGL\s+invariant\s*\(\s*all\s*\)\s*$/m.test(s.source);
}

/** The linker-side match identity of one varying: plain varyings match by
 *  NAME; varying-interface-block members match by (blockName, memberName) —
 *  instance names may differ between stages (`out VS_OUT { vec4 c; } a;` vs
 *  `in VS_OUT { vec4 c; } b;`). ':' cannot appear in GLSL identifiers, so the
 *  composite keys never collide. */
function varyingMatchKey(v: { blockName: string | null; name: string }): string {
  if (v.blockName === null) return `v:${v.name}`;
  const idx = v.name.lastIndexOf('.');
  const member = idx >= 0 ? v.name.slice(idx + 1) : v.name;
  return `b:${v.blockName}:${member}`;
}

/** Match every fragment input against a vertex output (name / element type /
 *  array size / flat; block members by (blockName, memberName)) and pack in
 *  VERTEX declaration order (dense flat components, cumulative offsets).
 *  Extra vertex outputs are packed too — they are active varyings (raster
 *  interpolates the whole record). Struct varyings flatten to per-member
 *  leaves, each with its own (index, offset).
 *
 *  FRAGMENT-USED MATCHING (native Chromium behavior, both GLSL versions):
 *  only fragment varyings whose value is actually READ (VaryingDecl.used)
 *  must match a vertex output; fragment varyings that are declared but never
 *  read impose NO constraint (they are stripped before linking, so no match
 *  / type / flat checks). Vertex-side extras are always allowed. `used` is
 *  tracked by semantics (scanUses) and is false on vertex-stage entries.
 *
 *  VaryingLayout keys: plain varyings + instance-less block members use their
 *  bare name; named block members use the FULL '<instance>.<member>' path.
 *  Because block instance names may differ between stages, every matched block
 *  member emits layout entries for BOTH the vertex key (write side) and the
 *  fragment key (read side), sharing one (index, offset) — Program.varyings
 *  keeps ONE entry per vertex leaf. */
function layoutVaryings(vs: Shader, fs: Shader, limits: LinkLimits): VaryingLayoutResult | { error: string } {
  // INVARIANCE (GLSL ES 1.00 §4.6.4 "Invariance and linkage"): a matched
  // varying must be invariant in BOTH stages or NEITHER. The invariant flag
  // is the qualifier-form `invariant varying vec4 v;` (VaryingDecl.invariant,
  // recorded by semantics) OR the short form `invariant v;` (an
  // 'invariant-decl' AST node — semantics records no ShaderInfo effect for
  // it, so the linker reads the AST). `#pragma STDGL invariant(all)` marks
  // every OUTPUT of the stage invariant — the preprocessor DROPS pragma
  // lines, so it is detected in the original source. The pragma affects ONLY
  // outputs: in the vertex stage that is the varyings + gl_Position +
  // gl_PointSize; in the fragment stage it is gl_FragColor/gl_FragData only,
  // so fragment varyings (INPUTS) and gl_FragCoord/gl_PointCoord are NOT
  // made invariant by it (shaders-with-invariance case 17: variant VS +
  // pragma-only FS links).
  const vsInvNames = invariantDeclNames(vs);
  const fsInvNames = invariantDeclNames(fs);
  const vsPragma = hasInvariantAllPragma(vs);
  const invariantOf = (v: (typeof vs.info.varyings)[number], stage: 'vs' | 'fs'): boolean =>
    v.invariant || (stage === 'vs' ? vsInvNames : fsInvNames).has(v.name) || (stage === 'vs' && vsPragma);
  // Builtin cross-stage invariance: gl_FragCoord derives from gl_Position and
  // gl_PointCoord from gl_PointSize — `invariant gl_FragCoord` / `invariant
  // gl_PointCoord` in the fragment shader require the vertex output to be
  // invariant (cases 10/13 of shaders-with-invariance.html). The reverse
  // direction is NOT required (cases 11/14 link).
  if (fsInvNames.has('gl_FragCoord') && !(vsInvNames.has('gl_Position') || vsPragma)) {
    return { error: `linker: 'invariant gl_FragCoord' requires 'invariant gl_Position'` };
  }
  if (fsInvNames.has('gl_PointCoord') && !(vsInvNames.has('gl_PointSize') || vsPragma)) {
    return { error: `linker: 'invariant gl_PointCoord' requires 'invariant gl_PointSize'` };
  }
  const vsByKey = new Map<string, typeof vs.info.varyings[number]>();
  for (const v of vs.info.varyings) vsByKey.set(varyingMatchKey(v), v);
  const fsByKey = new Map<string, typeof fs.info.varyings[number]>();
  for (const f of fs.info.varyings) {
    const key = varyingMatchKey(f);
    fsByKey.set(key, f);
    // Declared-but-unread fragment varyings are inactive: no match/type/flat
    // constraint (native behavior — see the doc comment above).
    if (!f.used) continue;
    const v = vsByKey.get(key);
    if (!v) return { error: `linker: varying '${f.name}' not matched` };
    if (!typeEquals(f.type, v.type) || f.arraySize !== v.arraySize) {
      return { error: `linker: varying '${f.name}' type mismatch (${typeName(v.type)} vs ${typeName(f.type)})` };
    }
    if (f.flat !== v.flat) {
      return { error: `linker: varying '${f.name}' flat qualifier mismatch` };
    }
    if (invariantOf(f, 'fs') !== invariantOf(v, 'vs')) {
      return { error: `linker: varying '${f.name}' invariance mismatch` };
    }
  }
  const map = new Map<string, VaryingLayout>();
  const infos: VaryingInfo[] = [];
  let offset = 0;
  let vectors = 0;
  for (const v of vs.info.varyings) {
    let leaves: VaryingLeaf[];
    if (v.type.kind === 'struct') {
      if (v.arraySize > 1) {
        // Struct-array varyings need a codegen-side element-offset channel
        // (subP resets the const-index flatOff on member descent) — the
        // codegen walker cannot resolve them today.
        return { error: `linker: struct-array varying '${v.name}' not supported` };
      }
      leaves = [];
      flattenVaryingStruct(v.name, v.type, leaves);
    } else {
      leaves = [{ key: v.name, type: v.type, arraySize: v.arraySize }];
    }
    // Fragment-side twin leaves (block members only — the fs instance name may
    // differ; the matched decl's types equal the vertex's, so the leaf shapes
    // agree). UNUSED fragment twins (declared but never read) are skipped: the
    // codegen only looks up the fragment read-side key when the FS actually
    // reads the member, so no read-side key or layout-mismatch check is needed.
    let fsLeaves: VaryingLeaf[] | null = null;
    if (v.blockName !== null) {
      const f = fsByKey.get(varyingMatchKey(v));
      if (f !== undefined && f.used) {
        fsLeaves = [];
        if (f.type.kind === 'struct') flattenVaryingStruct(f.name, f.type, fsLeaves);
        else fsLeaves.push({ key: f.name, type: f.type, arraySize: f.arraySize });
        if (fsLeaves.length !== leaves.length) {
          // Struct type equality is BY NAME — same-named structs with
          // different members would flatten differently; treat as a mismatch.
          return { error: `linker: varying '${v.name}' struct layout mismatch` };
        }
      }
    }
    for (let li = 0; li < leaves.length; li++) {
      const leaf = leaves[li];
      const elemComps = leaf.type.kind === 'array' ? flatComponents(leaf.type.element) : flatComponents(leaf.type);
      const comps = flatComponents(leaf.type) * leaf.arraySize;
      const layout: VaryingLayout = { index: infos.length, offset, components: comps, elemComponents: elemComps, flat: v.flat };
      map.set(leaf.key, layout);
      if (fsLeaves !== null) map.set(fsLeaves[li].key, layout);
      infos.push({ name: leaf.key, type: toGLenum(leaf.type), components: comps, flat: v.flat });
      offset += comps;
      // API limit accounting is PER ELEMENT (matC = C rows), matching the
      // CTS varying-packing page (float[65] must exceed 64 varying vectors
      // even though 65 floats pack into 17 vec4s).
      vectors += limitRows(leaf.type) * leaf.arraySize;
    }
  }
  if (vectors > limits.maxVaryingVectors) {
    return { error: `linker: too many varying vectors (${vectors}, max ${limits.maxVaryingVectors})` };
  }
  return { map, infos };
}

/* ------------------------------------------------------------------ */
/* Transform feedback (WebGL2, opts.transformFeedback)                 */
/* ------------------------------------------------------------------ */

const GL_FLOAT_VEC4 = 0x8b52;

/**
 * Validate the transform-feedback capture spec and compute
 * `Program.transformFeedbackVaryings`. Every name must be an ACTIVE vertex
 * varying (block members accept the full '<instance>.<member>' path or the
 * bare member name — the instance prefix is stripped; arrays capture the
 * whole array) or 'gl_Position'. SEPARATE_ATTRIBS: count ≤
 * maxTransformFeedbackSeparateAttribs and each varying's total components ≤
 * maxTransformFeedbackSeparateComponents. INTERLEAVED_ATTRIBS: total
 * components ≤ maxTransformFeedbackInterleavedComponents.
 */
function layoutTransformFeedback(
  vs: Shader,
  spec: TransformFeedbackSpec | undefined,
  limits: LinkLimits,
): { varyings: TransformFeedbackVarying[] } | { error: string } {
  if (spec === undefined) return { varyings: [] };
  if (vs.version !== 300) {
    return { error: 'linker: transform feedback requires GLSL ES 3.00' };
  }
  const captured: TransformFeedbackVarying[] = [];
  let totalComponents = 0;
  for (const name of spec.varyings) {
    // Arrays may be specified with or without the '[0]' suffix (GL practice).
    const base = name.endsWith('[0]') ? name.slice(0, -3) : name;
    let found: (typeof vs.info.varyings)[number] | 'gl_Position' | null = null;
    if (base === 'gl_Position') {
      found = 'gl_Position';
    } else {
      for (const v of vs.info.varyings) {
        if (v.blockName === null) {
          if (v.name === base) {
            found = v;
            break;
          }
        } else {
          // Block member: accept the full '<instance>.<member>' key or the
          // bare member name (instance prefix stripped).
          const idx = v.name.lastIndexOf('.');
          const member = idx >= 0 ? v.name.slice(idx + 1) : v.name;
          if (v.name === base || member === base) {
            found = v;
            break;
          }
        }
      }
    }
    if (found === null) {
      return { error: `linker: transform feedback varying '${name}' is not an active vertex varying` };
    }
    const type = found === 'gl_Position' ? GL_FLOAT_VEC4 : toGLenum(found.type);
    const size = found === 'gl_Position' ? 1 : found.arraySize;
    const components = found === 'gl_Position' ? 4 : flatComponents(found.type) * found.arraySize;
    if (spec.bufferMode === 'SEPARATE_ATTRIBS') {
      if (captured.length >= limits.maxTransformFeedbackSeparateAttribs) {
        return {
          error: `linker: transform feedback separate attribs exceed maxTransformFeedbackSeparateAttribs (${limits.maxTransformFeedbackSeparateAttribs})`,
        };
      }
      if (components > limits.maxTransformFeedbackSeparateComponents) {
        return {
          error: `linker: transform feedback varying '${name}' exceeds maxTransformFeedbackSeparateComponents (${components}, max ${limits.maxTransformFeedbackSeparateComponents})`,
        };
      }
    }
    totalComponents += components;
    captured.push({ name, type, size });
  }
  if (spec.bufferMode === 'INTERLEAVED_ATTRIBS' && totalComponents > limits.maxTransformFeedbackInterleavedComponents) {
    return {
      error: `linker: transform feedback interleaved components exceed maxTransformFeedbackInterleavedComponents (${totalComponents}, max ${limits.maxTransformFeedbackInterleavedComponents})`,
    };
  }
  return { varyings: captured };
}

/* ------------------------------------------------------------------ */
/* Sampler explicit-binding conflicts (ES 3.00 layout(binding=))       */
/* ------------------------------------------------------------------ */

/**
 * Two ACTIVE samplers of DIFFERENT types with the SAME EXPLICIT binding
 * (`UniformDecl.binding` non-null) are a link error (a texture unit cannot
 * serve two sampler kinds). Samplers WITHOUT an explicit binding default to
 * unit 0 — the strict GLSL ES 3.00 rule would flag two such samplers as
 * conflicting too, but WebGL practice (and the CTS) only rejects EXPLICIT
 * binding conflicts, so default-0 samplers link fine here. Also enforces
 * maxCombinedTextureImageUnits (arrays occupy `size` units).
 * "Active" is approximated as "declared" — no usage analysis exists.
 */
function checkSamplerBindings(merged: MergedUniform[], limits: LinkLimits): string | null {
  const byBinding = new Map<number, { type: GLSLType; name: string }>();
  let totalUnits = 0;
  for (const u of merged) {
    if (!isSampler(u.decl.type)) continue;
    const t = u.decl.type;
    const elem = t.kind === 'array' ? t.element : t;
    totalUnits += t.kind === 'array' ? (t.size ?? 1) : 1;
    const binding = u.decl.binding;
    if (binding !== null) {
      const ex = byBinding.get(binding);
      if (ex !== undefined && !typeEquals(ex.type, elem)) {
        return `linker: sampler binding conflict: units ${binding} (${ex.name} vs ${u.decl.name})`;
      }
      if (ex === undefined) byBinding.set(binding, { type: elem, name: u.decl.name });
    }
  }
  if (totalUnits > limits.maxCombinedTextureImageUnits) {
    return `linker: too many texture units (${totalUnits}, max ${limits.maxCombinedTextureImageUnits})`;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Attributes                                                          */
/* ------------------------------------------------------------------ */

interface AttribLayoutResult {
  map: Map<string, number>;
  infos: AttribInfo[];
}

/** Assign attribute locations: explicit layout(location=) first, then
 *  bindAttribLocation (names WITHOUT explicit locations only), then first-free
 *  in declaration order. A matC attribute occupies C consecutive locations;
 *  an array of N elements occupies N × elemLocations. */
function layoutAttributes(vs: Shader, opts: LinkOptions, limits: LinkLimits): AttribLayoutResult | { error: string } {
  const occupied: { start: number; end: number; name: string }[] = [];
  const map = new Map<string, number>();
  const infos: AttribInfo[] = [];
  const bindings = opts.attribBindings;

  const claim = (name: string, start: number, end: number): string | null => {
    for (const o of occupied) {
      if (start < o.end && o.start < end) {
        return `linker: attribute '${name}' location ${start} conflicts with '${o.name}'`;
      }
    }
    occupied.push({ start, end, name });
    return null;
  };
  const firstFree = (need: number): number => {
    let loc = 0;
    for (;;) {
      let ok = true;
      for (const o of occupied) {
        if (loc < o.end && o.start < loc + need) {
          loc = o.end;
          ok = false;
          break;
        }
      }
      if (ok) return loc;
    }
  };

  for (const a of vs.info.attributes) {
    if (a.builtin === true) {
      // Built-in vertex inputs (gl_VertexID / gl_InstanceID / gl_DrawID,
      // WebGL2 — pushed by semantics only when the shader READS them,
      // appended after all user attributes): ACTIVE with location -1 per spec
      // (ACTIVE_ATTRIBUTES/getActiveAttrib count them; getAttribLocation
      // returns -1). They consume no generic attrib slots and stay OUT of
      // attribLocations — codegen reads them from ctx.vertexId /
      // ctx.instanceId / ctx.drawId, never ctx.attribs[loc]. `integral: false`
      // deliberately: the flag means "needs integer vertex-array backing" for
      // the gl draw-validation guard (src/gl/draw.ts L1592-1602:
      // `pa.integral && pa.location < maxAttribs` → vao.attribs[pa.location]).
      // Built-ins are never fetched from vertex arrays, so integral=true with
      // location -1 would index vao.attribs[-1] (undefined) → TypeError on
      // every draw of a gl_VertexID program. Type stays INT (getActiveAttrib
      // reports a.type).
      infos.push({
        name: a.name,
        location: -1,
        type: toGLenum(a.type),
        size: a.arraySize,
        components: typeComponents(a.type),
        integral: false,
      });
      continue;
    }
    if (!a.used) continue; // inactive attributes consume no generic slots (native behavior; getActiveAttrib omits them)
    const elemLocations = a.type.kind === 'matrix' ? a.type.cols : 1;
    const need = elemLocations * a.arraySize;
    let loc: number;
    if (a.location !== null) loc = a.location;
    else if (bindings !== undefined && bindings.has(a.name)) loc = bindings.get(a.name)!;
    else loc = firstFree(need);
    if (loc < 0) return { error: `linker: attribute '${a.name}' has a negative location` };
    if (loc + need > limits.maxVertexAttribs) {
      return { error: `linker: attribute '${a.name}' exceeds maxVertexAttribs (${limits.maxVertexAttribs})` };
    }
    const err = claim(a.name, loc, loc + need);
    if (err !== null) return { error: err };
    map.set(a.name, loc);
    infos.push({
      name: a.name,
      location: loc,
      type: toGLenum(a.type),
      size: a.arraySize,
      components: typeComponents(a.type),
      integral: isIntegral(a.type),
    });
  }
  return { map, infos };
}

/* ------------------------------------------------------------------ */
/* Fragment outputs                                                    */
/* ------------------------------------------------------------------ */

interface OutputLayoutResult {
  map: Map<string, number>;
  outputs: { location: number; type: number }[];
}

/** Fragment output locations. ES 1.00: gl_FragColor → 0, gl_FragData[i] → i
 *  (layout key 'gl_FragData' → base 0; codegen adds the index). ES 3.00: user
 *  outs with explicit layout(location=) or the single-output default 0; ARRAY
 *  outputs expand to one Program output entry PER SLOT (ShaderInfo carries the
 *  declaration entry + per-element '<name>[k]' entries — see compiler.ts
 *  OutputDecl). */
function layoutOutputs(fs: Shader, limits: LinkLimits): OutputLayoutResult | { error: string } {
  const map = new Map<string, number>();
  const outputs: { location: number; type: number }[] = [];
  if (fs.version === 100) {
    for (const o of fs.info.outputs) {
      if (o.name === 'gl_FragColor') {
        map.set('gl_FragColor', 0);
        outputs.push({ location: 0, type: toGLenum(o.type) });
      } else if (o.name.startsWith('gl_FragData')) {
        const idx = o.index ?? 0;
        if (idx >= limits.maxDrawBuffers) {
          return { error: `linker: gl_FragData[${idx}] exceeds maxDrawBuffers (${limits.maxDrawBuffers})` };
        }
        map.set('gl_FragData', 0);
        outputs.push({ location: idx, type: toGLenum(o.type) });
      }
    }
  } else {
    // GLSL ES 3.00 §4.3.8.2 (WebGL2): every output variable needs an explicit
    // layout(location=) unless it is the shader's ONLY output variable (which
    // then auto-assigns location 0 — CTS draw-buffers / gl-get-frag-data).
    let declCount = 0;
    for (const o of fs.info.outputs) if (parseOutputElement(o.name) === null) declCount++;
    const occupied = new Map<number, string>(); // location → owning declaration
    for (const o of fs.info.outputs) {
      const el = parseOutputElement(o.name);
      if (el !== null) {
        // Per-element entry of an array output: location = base + k. The
        // declaration entry (processed first — semantics emits it before the
        // elements) owns the whole range.
        const base = map.get(el.base);
        if (base === undefined) {
          return { error: `linker: output '${o.name}' has no declaration entry` };
        }
        const loc = base + el.k;
        if (loc >= limits.maxDrawBuffers) {
          return { error: `linker: output '${o.name}' location ${loc} exceeds maxDrawBuffers (${limits.maxDrawBuffers})` };
        }
        const owner = occupied.get(loc);
        if (owner !== undefined && owner !== el.base) {
          return { error: `linker: output '${o.name}' location ${loc} conflicts with another output` };
        }
        occupied.set(loc, el.base);
        map.set(o.name, loc);
        outputs.push({ location: loc, type: toGLenum(o.type) });
        continue;
      }
      // Declaration entry.
      let base: number;
      if (o.location === null) {
        if (declCount > 1) {
          return { error: `linker: output '${o.name}' must declare layout(location=) when the fragment shader has multiple outputs` };
        }
        base = 0; // single output variable → location 0
      } else {
        base = o.location;
      }
      if (base >= limits.maxDrawBuffers) {
        return { error: `linker: output '${o.name}' location ${base} exceeds maxDrawBuffers (${limits.maxDrawBuffers})` };
      }
      map.set(o.name, base);
      if (o.arraySize === 1) {
        const owner = occupied.get(base);
        if (owner !== undefined) {
          return { error: `linker: output '${o.name}' location ${base} conflicts with another output` };
        }
        occupied.set(base, o.name);
        outputs.push({ location: base, type: toGLenum(o.type) });
      } else {
        // Claim the whole [base, base+arraySize) range; the per-slot Program
        // output entries come from the element entries below.
        for (let k = 0; k < o.arraySize; k++) {
          const loc = base + k;
          if (loc >= limits.maxDrawBuffers) {
            return { error: `linker: output '${o.name}' location ${loc} exceeds maxDrawBuffers (${limits.maxDrawBuffers})` };
          }
          const owner = occupied.get(loc);
          if (owner !== undefined && owner !== o.name) {
            return { error: `linker: output '${o.name}' location ${loc} conflicts with another output` };
          }
          occupied.set(loc, o.name);
        }
      }
    }
  }
  return { map, outputs };
}

/** '<name>[k]' suffix of an array-output ELEMENT entry (null = declaration). */
function parseOutputElement(name: string): { base: string; k: number } | null {
  const m = /^(.*)\[(\d+)\]$/.exec(name);
  if (m === null) return null;
  return { base: m[1], k: Number(m[2]) };
}

/* ------------------------------------------------------------------ */
/* Capability merge + deferred-feature rejection                       */
/* ------------------------------------------------------------------ */

function mergeUses(vs: Shader, fs: Shader): ShaderUses {
  return {
    pointSize: vs.info.uses.pointSize || fs.info.uses.pointSize,
    fragCoord: vs.info.uses.fragCoord || fs.info.uses.fragCoord,
    frontFacing: vs.info.uses.frontFacing || fs.info.uses.frontFacing,
    pointCoord: vs.info.uses.pointCoord || fs.info.uses.pointCoord,
    fragDepth: vs.info.uses.fragDepth || fs.info.uses.fragDepth,
    vertexId: vs.info.uses.vertexId || fs.info.uses.vertexId,
    instanceId: vs.info.uses.instanceId || fs.info.uses.instanceId,
    drawId: vs.info.uses.drawId || fs.info.uses.drawId,
    derivatives: vs.info.uses.derivatives || fs.info.uses.derivatives,
    depthRange: vs.info.uses.depthRange || fs.info.uses.depthRange,
  };
}

/** Varying interface blocks (ES 3.00 `out`/`in`): reject the combos the
 *  packer/codegen do not support — vertex `in` blocks, fragment `out` blocks,
 *  and ARRAYED blocks (block arrays: the codegen walker loses const element
 *  indices on member descent (subP resets flatOff), so `a[i].c` cannot be
 *  packed correctly without codegen changes). Plain (non-arrayed) vertex
 *  `out` / fragment `in` blocks are fully supported. */
function varyingBlockError(vs: Shader, fs: Shader): string | null {
  const stages: [Shader, 'VERTEX' | 'FRAGMENT'][] = [
    [vs, 'VERTEX'],
    [fs, 'FRAGMENT'],
  ];
  for (const [shader, stage] of stages) {
    for (const d of shader.ast.declarations) {
      if (d.kind !== 'interface-block') continue;
      const storage = d.qualifiers.storage;
      if (storage !== 'out' && storage !== 'in') continue;
      const valid = (stage === 'VERTEX' && storage === 'out') || (stage === 'FRAGMENT' && storage === 'in');
      if (!valid) {
        return `linker: ${stage === 'VERTEX' ? 'vertex input' : 'fragment output'} interface blocks not supported`;
      }
      if (d.arrayDims.length > 0) {
        return `linker: arrayed varying interface blocks not supported`;
      }
    }
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Struct type names (codegen struct-constructor dispatch)              */
/* ------------------------------------------------------------------ */

/**
 * Collect every user struct type name declared in a shader AST, in source
 * order, deduplicated. Codegen resolves `Foo(...)` constructor calls only for
 * type names present in `CodegenLayout.structNames` (seeded into
 * env.structNames) — the linker fills that field from BOTH stages' ASTs, since
 * one layout serves both and a struct declared in only one stage must still
 * resolve there. Sources:
 *  - top-level bare `struct S {...};` (ExternalDecl 'struct-decl' — the
 *    common case);
 *  - top-level struct-with-declarators `struct S {...} v;` / `uniform struct
 *    S {...} u;` (GlobalVarDecl whose TypeSpec.base is an inline struct
 *    definition);
 *  - local struct declarations inside function bodies (DeclStmt with an
 *    inline struct-definition base, in any nested block).
 * No descent into uniform-block members or other declarations is needed:
 * parser.ts rejects struct definitions inside structs, in function parameters
 * and as function return types, and builtin struct types (gl_DepthRange
 * parameters) live in the builtin tables, never in the AST — so every
 * StructDecl/StructDefinition node here is a user type.
 */
export function collectStructNames(tu: TranslationUnit): string[] {
  const names = new Set<string>();
  const addDef = (def: StructDefinition | null): void => {
    // Parser rejects anonymous structs; '' guard is defensive only.
    if (def !== null && def.name !== null && def.name !== '') names.add(def.name);
  };
  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'compound':
        for (const c of s.body) walkStmt(c);
        return;
      case 'decl-stmt':
        if (s.type.base.kind === 'struct-definition') addDef(s.type.base);
        return;
      case 'if':
        walkStmt(s.then);
        if (s.else !== null) walkStmt(s.else);
        return;
      case 'for':
        if (s.init !== null) walkStmt(s.init);
        walkStmt(s.body);
        return;
      case 'while':
        walkStmt(s.body);
        return;
      case 'do-while':
        walkStmt(s.body);
        return;
      case 'switch':
        walkStmt(s.body);
        return;
      default:
        return;
    }
  };
  for (const d of tu.declarations) {
    if (d.kind === 'struct-decl') {
      if (d.name !== '') names.add(d.name);
    } else if (d.kind === 'global-var-decl') {
      if (d.type.base.kind === 'struct-definition') addDef(d.type.base);
    } else if (d.kind === 'function-definition') {
      walkStmt(d.body);
    }
  }
  return [...names];
}

/* ------------------------------------------------------------------ */
/* linkProgram                                                         */
/* ------------------------------------------------------------------ */

/**
 * Link a vertex+fragment shader pair. See the file header for the layout
 * conventions. On success the returned Program is fully executable:
 * `program.vertex.run(ctx)` / `program.fragment.run(ctx)` evaluate one
 * vertex / one fragment with the stores, scratch and out buffers provided in
 * the exec ctx (see program.ts).
 */
export function linkProgram(vs: Shader, fs: Shader, opts?: LinkOptions): LinkResult {
  if (vs.version !== fs.version) {
    return { ok: false, log: 'linker: vertex and fragment shader versions differ' };
  }
  const vbe = varyingBlockError(vs, fs);
  if (vbe !== null) return { ok: false, log: vbe };

  const limits = resolveLimits(vs.version, opts);

  // Capability merge first: the gl_DepthRange allocation below is usage-gated.
  const uses = mergeUses(vs, fs);

  const merged = mergeUniforms(vs, fs);
  if ('error' in merged) return { ok: false, log: merged.error };
  const sb = checkSamplerBindings(merged, limits);
  if (sb !== null) return { ok: false, log: sb };
  const ul = layoutUniforms(merged, limits);
  if ('error' in ul) return { ok: false, log: ul.error };

  // Builtin uniform gl_DepthRange (GLSL ES 1.00 §7.6 / 3.00 §7.7): when EITHER
  // stage reads it, expose the struct members as ACTIVE UNIFORMS backed by 3
  // REAL float-store slots appended AFTER all user uniforms (existing uniform
  // offsets unchanged — the allocation uses the same float-index convention as
  // the vec4-slot layout: `location` = index into floatStore, gl/ writes at
  // byte location*4). codegen keeps reading ctx.depthRange (draw-time state);
  // these entries exist so getActiveUniform / getUniformLocation / getUniform
  // report the builtin and return the CURRENT glDepthRangef values. The store
  // is zero-initialized by `new Float32Array` in the assembly below. Usage
  // gating is REQUIRED: never report the builtin when neither shader reads it
  // (active-uniform enumeration stays exactly as before).
  let depthRangeSlots: [number, number, number] | null = null;
  const depthRangeUniforms: UniformInfo[] = [];
  if (uses.depthRange) {
    const base = ul.floatMax; // one float per member (near, far, far − near)
    depthRangeSlots = [base, base + 1, base + 2];
    const mk = (name: string, slot: number): UniformInfo => ({
      name,
      location: slot,
      type: 0x1406, // GL_FLOAT
      size: 1,
      components: 1,
      integral: false,
      blockIndex: -1,
      sampler: false,
    });
    const near = mk('gl_DepthRange.near', base);
    const far = mk('gl_DepthRange.far', base + 1);
    const diff = mk('gl_DepthRange.diff', base + 2);
    depthRangeUniforms.push(near, far, diff);
    // uniformMap entries carry the name (gl/ resolves getUniformLocation via
    // `uniformMap.get(nm)` then matches `.name` against program.uniforms).
    ul.uniformMap.set('gl_DepthRange.near', near);
    ul.uniformMap.set('gl_DepthRange.far', far);
    ul.uniformMap.set('gl_DepthRange.diff', diff);
    ul.floatMax = base + 3; // floatStore grows to include the 3 slots
  }

  const bl = layoutBlocks(vs, fs, limits);
  if ('error' in bl) return { ok: false, log: bl.error };

  const vl = layoutVaryings(vs, fs, limits);
  if ('error' in vl) return { ok: false, log: vl.error };

  const tf = layoutTransformFeedback(vs, opts?.transformFeedback, limits);
  if ('error' in tf) return { ok: false, log: tf.error };

  const al = layoutAttributes(vs, opts ?? {}, limits);
  if ('error' in al) return { ok: false, log: al.error };

  const ol = layoutOutputs(fs, limits);
  if ('error' in ol) return { ok: false, log: ol.error };

  const structNames = [...new Set([...collectStructNames(vs.ast), ...collectStructNames(fs.ast)])];
  const layout: CodegenLayout = {
    version: vs.version,
    uniformSlots: ul.slots,
    blocks: bl.blocks,
    blockIndices: bl.blockIndices,
    varyings: vl.map,
    attribLocations: al.map,
    outputLocations: ol.map,
    uses,
    structNames,
  };

  let vsRes: StageCodegenResult;
  let fsRes: StageCodegenResult;
  try {
    vsRes = generateVertexStage(vs.ast, layout);
    fsRes = generateFragmentStage(fs.ast, layout);
  } catch (e) {
    return { ok: false, log: `linker: codegen failed: ${(e as Error).message}` };
  }
  let vertexFn: (ctx: VertexExecCtx, R: unknown) => void;
  let fragmentFn: (ctx: FragmentExecCtx, R: unknown) => void;
  try {
    vertexFn = new Function('ctx', 'R', vsRes.body) as (ctx: VertexExecCtx, R: unknown) => void;
    fragmentFn = new Function('ctx', 'R', fsRes.body) as (ctx: FragmentExecCtx, R: unknown) => void;
  } catch (e) {
    return { ok: false, log: `linker: generated code failed to compile: ${(e as Error).message}` };
  }

  return {
    ok: true,
    program: {
      attributes: al.infos,
      uniforms: [...ul.uniforms, ...bl.uniformInfos, ...depthRangeUniforms],
      uniformBlocks: bl.infos,
      varyings: vl.infos,
      vertex: { run: (ctx) => vertexFn(ctx, R) },
      fragment: {
        run: (ctx) => fragmentFn(ctx, R),
        usesDerivatives: uses.derivatives,
        usesFragDepth: uses.fragDepth,
        outputs: ol.outputs,
      },
      usesPointSize: uses.pointSize,
      usesGLPointCoord: uses.pointCoord,
      usesFragCoord: uses.fragCoord,
      usesFrontFacing: uses.frontFacing,
      uniformMap: ul.uniformMap,
      floatStore: new Float32Array(ul.floatMax),
      intStore: new Int32Array(ul.intMax),
      depthRangeSlots,
      scratchSize: Math.max(vsRes.scratchSize, fsRes.scratchSize),
      intScratchSize: Math.max(vsRes.intScratchSize, fsRes.intScratchSize),
      transformFeedbackVaryings: tf.varyings,
    },
  };
}
