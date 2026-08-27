/**
 * linker.ts — the program linker (cross-module contract §1: linker phase).
 *
 * `linkProgram(vs, fs, opts)` validates the shader pair and produces the
 * `Program` object gl/ and raster/ consume:
 *
 *   1. version compatibility + deferred-feature rejection (UBOs, transform
 *      feedback, varying interface blocks — later tasks);
 *   2. uniform merge (same name in both stages must be type-identical) and
 *      default-block layout (conservative vec4-slot packing);
 *   3. varying matching (name / element type / array size / flat) and dense
 *      packing in VERTEX declaration order;
 *   4. attribute location assignment (explicit layout(location=) →
 *      bindAttribLocation → first-free automatic);
 *   5. fragment output location assignment;
 *   6. limit checks; JS codegen for both stages; Program assembly.
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
import type { TranslationUnit } from './ast.js';
import type { LinkLimits, LinkOptions, LinkResult, Shader, ShaderUses, UniformDecl } from './compiler.js';
import type { AttribInfo, FragmentExecCtx, Program, UniformInfo, VaryingInfo, VertexExecCtx } from './program.js';
import { generateFragmentStage, generateVertexStage, R } from './codegen/index.js';
import type { CodegenLayout, StageCodegenResult, UniformSlot, VaryingLayout } from './codegen/index.js';
import { flatComponents, isIntegralFamily } from './codegen/env.js';
import { isIntegral, isSampler, toGLenum, typeComponents, typeEquals, typeName } from './types.js';
import type { GLSLType } from './types.js';

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

/** FLOAT elements between consecutive elements of the LAST array dimension
 *  (the `stride` codegen applies for dynamic indices). */
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
      return elemFloatStride(e.element) * (e.size ?? 1);
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
      const comps = t.kind === 'matrix' ? t.cols * t.rows : t.kind === 'vector' ? t.size : 1;
      if (int) st.intMax = Math.max(st.intMax, slot + comps);
      else st.floatMax = Math.max(st.floatMax, slot + comps);
      return { advance: slotCount(t) };
    }
    case 'array': {
      const n = t.size ?? 0;
      const e = t.element;
      if (n === 0) return { advance: 0 };
      if (e.kind === 'struct' || e.kind === 'array') {
        // Struct / nested-array uniform support lands with the flattening
        // extension (commit 2 of E5a).
        return { error: `linker: array of '${typeName(e)}' uniforms not supported` };
      }
      const dense = e.kind === 'scalar' || e.kind === 'sampler';
      const stride = elemFloatStride(e);
      const int = isIntStoreType(e);
      const comps = e.kind === 'matrix' ? e.cols * e.rows : e.kind === 'vector' ? e.size : 1;
      const base = cursor * 4;
      // '[0]' prefix: dynamic-index resolution (codegen reads its stride).
      st.slots.set(`${path}[0]`, { store: int ? 'int' : 'float', slot: base, stride });
      for (let k = 0; k < n; k++) {
        const slot = base + k * stride;
        st.slots.set(`${path}[${k}]`, { store: int ? 'int' : 'float', slot, stride: 0 });
        if (int) st.intMax = Math.max(st.intMax, slot + comps);
        else st.floatMax = Math.max(st.floatMax, slot + comps);
      }
      return { advance: dense ? Math.ceil(n / 4) : slotCount(e) * n };
    }
    case 'struct':
      // Struct uniform support lands with the flattening extension (commit 2).
      return { error: `linker: struct uniform '${path}' not supported` };
    case 'void':
      return { advance: 0 };
  }
}

/** One merged default-block uniform (same name in both stages). */
interface MergedUniform {
  decl: UniformDecl;
  inVs: boolean;
  inFs: boolean;
}

/** Combine vs + fs default-block uniforms by name; same name must be
 *  type-identical in both stages (GLSL link rule). */
function mergeUniforms(vs: Shader, fs: Shader): MergedUniform[] | { error: string } {
  const byName = new Map<string, MergedUniform>();
  const order: string[] = [];
  const add = (decl: UniformDecl, stage: 'vs' | 'fs'): string | null => {
    const ex = byName.get(decl.name);
    if (ex) {
      if (!typeEquals(ex.decl.type, decl.type)) {
        return `linker: uniform '${decl.name}' type conflict (${typeName(ex.decl.type)} vs ${typeName(decl.type)})`;
      }
      if (stage === 'vs') ex.inVs = true;
      else ex.inFs = true;
      return null;
    }
    byName.set(decl.name, { decl, inVs: stage === 'vs', inFs: stage === 'fs' });
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

interface UniformLayoutResult {
  slots: Map<string, UniformSlot>;
  uniforms: UniformInfo[];
  floatMax: number;
  intMax: number;
}

/** Allocate the default-block stores for the merged uniform set (vs order
 *  first, then fs-only), building uniformSlots + the Program.uniforms list. */
function layoutUniforms(merged: MergedUniform[]): UniformLayoutResult | { error: string } {
  const st: AllocState = { slots: new Map(), floatMax: 0, intMax: 0 };
  const uniforms: UniformInfo[] = [];
  let cursor = 0;
  for (const u of merged) {
    const t = u.decl.type;
    const r = emitType(u.decl.name, t, cursor, st);
    if ('error' in r) return { error: r.error };
    // Ancestor prefix for the root name (arrays; the leaf case already keyed it).
    if (!st.slots.has(u.decl.name)) {
      st.slots.set(u.decl.name, { store: isIntStoreType(t) ? 'int' : 'float', slot: cursor * 4, stride: 0 });
    }
    // getActiveUniform entries: plain arrays are ONE entry 'u[0]' (size N);
    // plain non-arrays one entry 'u'. Structs flatten per leaf (extension).
    if (t.kind === 'array') {
      const e = t.element;
      uniforms.push({
        name: `${u.decl.name}[0]`,
        location: st.slots.get(`${u.decl.name}[0]`)!.slot,
        type: toGLenum(e),
        size: t.size ?? 1,
        components: typeComponents(e),
        integral: isIntegral(e),
        blockIndex: -1,
        sampler: isSampler(e),
      });
    } else {
      uniforms.push({
        name: u.decl.name,
        location: st.slots.get(u.decl.name)!.slot,
        type: toGLenum(t),
        size: 1,
        components: typeComponents(t),
        integral: isIntegral(t),
        blockIndex: -1,
        sampler: isSampler(t),
      });
    }
    cursor += r.advance;
  }
  return { slots: st.slots, uniforms, floatMax: st.floatMax, intMax: st.intMax };
}

/* ------------------------------------------------------------------ */
/* Varying matching + packing                                          */
/* ------------------------------------------------------------------ */

interface VaryingLayoutResult {
  map: Map<string, VaryingLayout>;
  infos: VaryingInfo[];
}

/** Match every fragment input against a vertex output (name / element type /
 *  array size / flat) and pack in VERTEX declaration order (dense flat
 *  components, cumulative offsets). Extra vertex outputs are packed too —
 *  they are active varyings (raster interpolates the whole record). */
function layoutVaryings(vs: Shader, fs: Shader, limits: LinkLimits): VaryingLayoutResult | { error: string } {
  const vsByName = new Map<string, typeof vs.info.varyings[number]>();
  for (const v of vs.info.varyings) vsByName.set(v.name, v);
  for (const f of fs.info.varyings) {
    const v = vsByName.get(f.name);
    if (!v) return { error: `linker: varying '${f.name}' not matched` };
    if (!typeEquals(f.type, v.type) || f.arraySize !== v.arraySize) {
      return { error: `linker: varying '${f.name}' type mismatch (${typeName(v.type)} vs ${typeName(f.type)})` };
    }
    if (f.flat !== v.flat) {
      return { error: `linker: varying '${f.name}' flat qualifier mismatch` };
    }
  }
  const map = new Map<string, VaryingLayout>();
  const infos: VaryingInfo[] = [];
  let offset = 0;
  let vectors = 0;
  for (const v of vs.info.varyings) {
    if (v.type.kind === 'struct') {
      // Struct-varying flattening lands with the extension (commit 2).
      return { error: `linker: struct varying '${v.name}' not supported` };
    }
    const elemComps = flatComponents(v.type);
    const comps = elemComps * v.arraySize;
    map.set(v.name, { index: infos.length, offset, components: comps, elemComponents: elemComps, flat: v.flat });
    infos.push({ name: v.name, type: toGLenum(v.type), components: comps, flat: v.flat });
    offset += comps;
    vectors += Math.ceil(comps / 4);
  }
  if (vectors > limits.maxVaryingVectors) {
    return { error: `linker: too many varying vectors (${vectors}, max ${limits.maxVaryingVectors})` };
  }
  return { map, infos };
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
 *  outs with explicit layout(location=) or auto-assigned 0,1,2,... */
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
    const occupied = new Set<number>();
    let next = 0;
    for (const o of fs.info.outputs) {
      let loc = o.location;
      if (loc === null) {
        while (occupied.has(next)) next++;
        loc = next;
      }
      if (loc >= limits.maxDrawBuffers) {
        return { error: `linker: output '${o.name}' location ${loc} exceeds maxDrawBuffers (${limits.maxDrawBuffers})` };
      }
      if (occupied.has(loc)) {
        return { error: `linker: output '${o.name}' location ${loc} conflicts with another output` };
      }
      occupied.add(loc);
      map.set(o.name, loc);
      outputs.push({ location: loc, type: toGLenum(o.type) });
    }
  }
  return { map, outputs };
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
    derivatives: vs.info.uses.derivatives || fs.info.uses.derivatives,
  };
}

/** Features explicitly deferred to later tasks: return a short feature name
 *  (the caller emits 'linker: <feature> not supported') or null. */
function deferredFeature(vs: Shader, fs: Shader, opts?: LinkOptions): string | null {
  if (opts?.transformFeedback !== undefined) return 'transform feedback';
  if (vs.info.uniformBlocks.length > 0 || fs.info.uniformBlocks.length > 0) return 'uniform blocks';
  for (const ast of [vs.ast, fs.ast]) {
    for (const d of ast.declarations) {
      if (d.kind === 'interface-block') {
        const storage = d.qualifiers.storage;
        if (storage === 'out' || storage === 'in') return 'varying interface blocks';
      }
    }
  }
  return null;
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
  const feat = deferredFeature(vs, fs, opts);
  if (feat !== null) return { ok: false, log: `linker: ${feat} not supported` };

  const limits = resolveLimits(vs.version, opts);

  const merged = mergeUniforms(vs, fs);
  if ('error' in merged) return { ok: false, log: merged.error };
  const ul = layoutUniforms(merged);
  if ('error' in ul) return { ok: false, log: ul.error };

  const vl = layoutVaryings(vs, fs, limits);
  if ('error' in vl) return { ok: false, log: vl.error };

  const al = layoutAttributes(vs, opts ?? {}, limits);
  if ('error' in al) return { ok: false, log: al.error };

  const ol = layoutOutputs(fs, limits);
  if ('error' in ol) return { ok: false, log: ol.error };

  const uses = mergeUses(vs, fs);
  const layout: CodegenLayout = {
    version: vs.version,
    uniformSlots: ul.slots,
    blocks: new Map(),
    blockIndices: new Map(),
    varyings: vl.map,
    attribLocations: al.map,
    outputLocations: ol.map,
    uses,
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
      uniforms: ul.uniforms,
      uniformBlocks: [],
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
      uniformMap: new Map(),
      floatStore: new Float32Array(ul.floatMax),
      intStore: new Int32Array(ul.intMax),
      scratchSize: Math.max(vsRes.scratchSize, fsRes.scratchSize),
      intScratchSize: Math.max(vsRes.intScratchSize, fsRes.intScratchSize),
      transformFeedbackVaryings: [],
    },
  };
}
