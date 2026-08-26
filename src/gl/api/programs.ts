/**
 * src/gl/api/programs.ts — shader/program lifecycle, linking, introspection.
 *
 * Owns: createShader, deleteShader, isShader, shaderSource, compileShader,
 * createProgram, deleteProgram, isProgram, attachShader, detachShader,
 * linkProgram, useProgram, validateProgram, getShaderParameter,
 * getProgramParameter, getShaderInfoLog, getProgramInfoLog, getShaderSource,
 * getShaderPrecisionFormat, getAttribLocation, getUniformLocation,
 * bindAttribLocation, getActiveAttrib, getActiveUniform, getUniform +
 * WebGL2: uniformBlockBinding, getUniformBlockIndex, getActiveUniformBlockParameter,
 * getActiveUniformBlockName, getUniformIndices, getActiveUniformsiv,
 * getFragDataLocation, transformFeedbackVaryings, getTransformFeedbackVarying.
 *
 * Pipeline (contract §1 with glsl/):
 *  - compileShader → glsl.compileShader(source, {type, version, defines,
 *    extensions}); the full glsl Shader result is kept in the module-level
 *    `shaderResults` WeakMap (the pinned WebGLShader._compiled is too narrow);
 *    COMPILE_STATUS + info log are set on the WebGLShader.
 *  - linkProgram → glsl.linkProgram(vs, fs, {limits, attribBindings,
 *    transformFeedback}); on success stores the glsl Program in `programModels`
 *    (WeakMap) and `_program`, allocates `_uniformStore` and block stores.
 *  - useProgram (kept here: uniform* writes need state.currentProgram):
 *    validates the program is linked (INVALID_OPERATION otherwise).
 *  - getActiveAttrib/Uniform return WebGLActiveInfo {size, type, name};
 *    getUniformLocation supports array-name suffixes ('u[2]'); getUniform
 *    reads the uniform store back (types per WebGL spec).
 *  - KHR_parallel_shader_compile: COMPLETION_STATUS_KHR (0x91B1) reports true
 *    (synchronous compile) for both shaders and programs when the extension is
 *    exposed, INVALID_ENUM otherwise.
 *
 * UNIFORM STORE LAYOUT (single source of truth, matches src/glsl/program.ts):
 * glsl Program.floatStore (Float32Array) / intStore (Int32Array) hold the
 * default-block uniforms with vec4-slot packing: a uniform occupies
 * `elementSlots()` consecutive slots per array element (1 for scalars/vectors,
 * `cols` for matrices — column-major, 4 floats per slot); UniformInfo.location
 * is the START slot. gl/ writes via:
 *  - floats: `_uniformStore` = DataView over floatStore's ArrayBuffer
 *    (byte offset = slot*16 + component*4, little-endian);
 *  - ints/uints/bools/samplers: `programModels.get(p).intStore[slot*4+c]`.
 * The draw engine passes the SAME arrays as ctx.uniforms / ctx.intUniforms,
 * so writes land in the memory the generated shaders read.
 *
 * getProgramParameter pre-link behavior (documented decision): LINK_STATUS /
 * DELETE_STATUS / VALIDATE_STATUS / ATTACHED_SHADERS return live values on an
 * unlinked program; ACTIVE_ATTRIBUTES / ACTIVE_UNIFORMS / ACTIVE_UNIFORM_BLOCKS /
 * TRANSFORM_FEEDBACK_VARYINGS return 0 (no error) while `_program === null`;
 * TRANSFORM_FEEDBACK_BUFFER_MODE returns `_tfBufferMode` (0 pre-link).
 * ACTIVE_*_MAX_LENGTH and INFO_LOG_LENGTH are INVALID_ENUM (WebGL removed the
 * *_LENGTH queries; CTS program-test.html expects INVALID_ENUM + null).
 *
 * Relink invalidation: each linkProgram call bumps a per-program generation
 * counter (`linkGen`); getUniformLocation captures it on the returned
 * WebGLUniformLocation (`locGen`). uniform* / getUniform reject locations from
 * a previous link with INVALID_OPERATION (uniform-location.html relink tests).
 *
 * Error semantics: WebIDL conversion failures throw TypeError; cross-context
 * objects → INVALID_OPERATION; DELETED shaders/programs that are no longer
 * attached/in-use → INVALID_VALUE for query methods (ES2 "not the name of an
 * object") and INVALID_OPERATION for lifecycle methods; deleted-but-attached
 * shaders and deleted-but-in-use programs remain valid until detached/unused.
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGL2RenderingContext } from '../webgl2';
import { C1, C2, CExt } from '../constants';
import {
  WebGLShader,
  WebGLProgram,
  WebGLUniformLocation,
  WebGLActiveInfo,
  WebGLShaderPrecisionFormat,
  validateUniformLocation,
  createObject,
} from '../objects';
import type { ProgramModel } from '../objects';
import { validateObject, requireString } from '../validation';
import { compileShader, linkProgram } from '../../glsl';
import type { Shader as GlslShader, Program as GlslProgram, LinkOptions } from '../../glsl';
import type { GLboolean, GLenum, GLint, GLuint } from '../types';

// ---- Module-level per-program / per-shader state (objects files are pinned; keep here) ----

/** Full glsl compile result per WebGLShader (link input). */
const shaderResults = new WeakMap<WebGLShader, GlslShader>();
/** glsl Program per linked WebGLProgram (stores, uniformMap, TF varyings). */
const programModels = new WeakMap<WebGLProgram, GlslProgram>();
/** Attach refcount per WebGLShader (deferred deletion while attached). */
const shaderAttachCounts = new WeakMap<WebGLShader, number>();
/** Link generation counter per WebGLProgram (relink invalidation). */
const linkGen = new WeakMap<WebGLProgram, number>();
/** Link generation captured at getUniformLocation (per WebGLUniformLocation). */
export const locGen = new WeakMap<WebGLUniformLocation, number>();
/** validateProgram result per WebGLProgram (VALIDATE_STATUS). */
const validateStatus = new WeakMap<WebGLProgram, boolean>();
/** uniformBlockBinding: blockIndex → binding point (default 0), per program. */
const uniformBlockBindings = new WeakMap<WebGLProgram, Uint32Array>();
/** Block referenced flags per program: bit 1 = VS, bit 2 = FS. */
const blockReferenced = new WeakMap<WebGLProgram, Uint32Array>();
/** Fragment output name → location (getFragDataLocation), per program. */
const fragDataMaps = new WeakMap<WebGLProgram, Map<string, number>>();

/** Context-loss guard: no-op + one CONTEXT_LOST_WEBGL per call. */
function isLost(ctx: WebGLRenderingContext): boolean {
  if (ctx._isLost) ctx._errors.push(C1.CONTEXT_LOST_WEBGL);
  return ctx._isLost;
}

// WebGLShader/WebGLProgram inherit WebGLObject's PROTECTED constructor — cast
// once (runtime behavior is identical; pattern from api/buffers.ts).
const ShaderCtor = WebGLShader as unknown as new (context: WebGLRenderingContext) => WebGLShader;
const ShaderCtorAny = ShaderCtor as unknown as new (...args: never[]) => WebGLShader;
const ProgramCtor = WebGLProgram as unknown as new (context: WebGLRenderingContext) => WebGLProgram;
const ProgramCtorAny = ProgramCtor as unknown as new (...args: never[]) => WebGLProgram;

/** Lifecycle validation: cross-context/deleted → INVALID_OPERATION (validateObject). */
function validateShader(ctx: WebGLRenderingContext, shader: unknown): WebGLShader | null {
  return validateObject<WebGLShader>(ctx, shader, ShaderCtorAny);
}
function validateProgram(ctx: WebGLRenderingContext, program: unknown): WebGLProgram | null {
  return validateObject<WebGLProgram>(ctx, program, ProgramCtorAny);
}

/**
 * Query validation for shaders: deleted shaders still attached to a program are
 * valid (delete is deferred until detached); a deleted, unattached shader is
 * "not the name of a shader object" → INVALID_VALUE (ES2 / CTS program-test).
 */
function validateShaderQuery(ctx: WebGLRenderingContext, shader: unknown): WebGLShader | null {
  if (!(shader instanceof WebGLShader)) throw new TypeError(`Argument is not of type 'WebGLShader'`);
  if (shader._context !== ctx) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (shader._deleted && (shaderAttachCounts.get(shader) ?? 0) === 0) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  return shader;
}

/** Query validation for programs (deleted-but-in-use programs remain valid). */
function validateProgramQuery(ctx: WebGLRenderingContext, program: unknown): WebGLProgram | null {
  if (!(program instanceof WebGLProgram)) throw new TypeError(`Argument is not of type 'WebGLProgram'`);
  if (program._context !== ctx) {
    ctx._errors.push(C1.INVALID_OPERATION);
    return null;
  }
  if (program._deleted && !program._inUse && !program._inTransformFeedback) {
    ctx._errors.push(C1.INVALID_VALUE);
    return null;
  }
  return program;
}

// ---- Name validation (WebGL 1.0 spec §5.14.10 / WebGL2: limits 256 vs 1024) ----

/** Characters forbidden in attribute/uniform names: `" $ ` @ \ '`. */
function hasBadNameChars(name: string): boolean {
  for (let i = 0; i < name.length; i++) {
    const c = name.charCodeAt(i);
    if (c === 0x22 || c === 0x24 || c === 0x60 || c === 0x40 || c === 0x5c || c === 0x27) return true;
  }
  return false;
}

/** True for names beginning with the reserved prefixes gl_/webgl_/_webgl_. */
function isReservedPrefix(name: string): boolean {
  return name.startsWith('gl_') || name.startsWith('webgl_') || name.startsWith('_webgl_');
}

/** Name too long → push INVALID_VALUE (WebGL1: 256, WebGL2: 1024). */
function nameTooLong(ctx: WebGLRenderingContext, name: string): boolean {
  if (name.length > (ctx._version === 2 ? 1024 : 256)) {
    ctx._errors.push(C1.INVALID_VALUE);
    return true;
  }
  return false;
}

/** Bad characters → push INVALID_VALUE. */
function badNameChars(ctx: WebGLRenderingContext, name: string): boolean {
  if (hasBadNameChars(name)) {
    ctx._errors.push(C1.INVALID_VALUE);
    return true;
  }
  return false;
}

// ---- GLSL type helpers (uniform store layout) ----

/** True for FLOAT_MAT* GLenums (missing non-square constants in constants.ts → literals). */
export function isMatrixType(type: number): boolean {
  switch (type) {
    case C1.FLOAT_MAT2:
    case C1.FLOAT_MAT3:
    case C1.FLOAT_MAT4:
    case 0x8b65 /* FLOAT_MAT2x3 */:
    case 0x8b66 /* FLOAT_MAT2x4 */:
    case 0x8b67 /* FLOAT_MAT3x2 */:
    case 0x8b68 /* FLOAT_MAT3x4 */:
    case 0x8b69 /* FLOAT_MAT4x2 */:
    case 0x8b6a /* FLOAT_MAT4x3 */:
      return true;
    default:
      return false;
  }
}

/** Matrix column count (matCxR → C). */
export function matrixCols(type: number): number {
  switch (type) {
    case C1.FLOAT_MAT2:
    case 0x8b65 /* FLOAT_MAT2x3 */:
    case 0x8b66 /* FLOAT_MAT2x4 */:
      return 2;
    case C1.FLOAT_MAT3:
    case 0x8b67 /* FLOAT_MAT3x2 */:
    case 0x8b68 /* FLOAT_MAT3x4 */:
      return 3;
    default:
      return 4;
  }
}

/** Matrix row count (matCxR → R). */
export function matrixRows(type: number): number {
  switch (type) {
    case C1.FLOAT_MAT2:
    case 0x8b67 /* FLOAT_MAT3x2 */:
    case 0x8b69 /* FLOAT_MAT4x2 */:
      return 2;
    case C1.FLOAT_MAT3:
    case 0x8b65 /* FLOAT_MAT2x3 */:
    case 0x8b6a /* FLOAT_MAT4x3 */:
      return 3;
    default:
      return 4;
  }
}

/**
 * Consecutive store slots per array element: 1 for scalars/vectors
 * (ceil(components/4)), `cols` for matrices (column-major, 4 floats per slot).
 */
export function elementSlots(uniform: { type: number; components: number }): number {
  return isMatrixType(uniform.type) ? matrixCols(uniform.type) : Math.ceil(uniform.components / 4);
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/** Shared compileShader body — also resets state on shaderSource. */
function doCompileShader(ctx: WebGLRenderingContext, shader: WebGLShader): void {
  shader._compileStatus = false;
  shader._infoLog = '';
  shader._compiled = null;
  shaderResults.delete(shader);
  let result: ReturnType<typeof compileShader>;
  try {
    result = compileShader(shader._source, {
      type: shader._type === C1.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT',
      version: ctx._version === 2 ? 300 : 100,
      defines: {},
      extensions: new Set(ctx.getSupportedExtensions()),
    });
  } catch (e) {
    // glsl/ is stubbed in the parallel wave — never let the exception reach
    // the page: behave as a compile failure with a diagnostic info log.
    shader._infoLog = `internal compiler error: ${e instanceof Error ? e.message : String(e)}`;
    return;
  }
  if (result.ok) {
    shader._compileStatus = true;
    shader._compiled = { ok: true };
    shaderResults.set(shader, result.shader);
  } else {
    shader._infoLog = result.errors.map((e) => `ERROR: 0:${e.line}: ${e.message}`).join('\n');
    shader._compiled = { ok: false, errors: result.errors };
  }
}

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------

/** Map context limits → glsl LinkLimits. */
function buildLinkLimits(ctx: WebGLRenderingContext): LinkOptions['limits'] {
  const l = ctx._state.limits;
  return {
    maxVertexAttribs: l.MAX_VERTEX_ATTRIBS,
    maxVertexUniformVectors: l.MAX_VERTEX_UNIFORM_VECTORS,
    maxFragmentUniformVectors: l.MAX_FRAGMENT_UNIFORM_VECTORS,
    maxVaryingVectors: l.MAX_VARYING_VECTORS,
    maxVertexTextureImageUnits: l.MAX_VERTEX_TEXTURE_IMAGE_UNITS,
    maxTextureImageUnits: l.MAX_TEXTURE_IMAGE_UNITS,
    maxCombinedTextureImageUnits: l.MAX_COMBINED_TEXTURE_IMAGE_UNITS,
    maxDrawBuffers: l.MAX_DRAW_BUFFERS,
    maxUniformBufferBindings: l.MAX_UNIFORM_BUFFER_BINDINGS,
    maxUniformBlockSize: l.MAX_UNIFORM_BLOCK_SIZE,
    maxVertexUniformBlocks: l.MAX_VERTEX_UNIFORM_BLOCKS,
    maxFragmentUniformBlocks: l.MAX_FRAGMENT_UNIFORM_BLOCKS,
    maxCombinedUniformBlocks: l.MAX_COMBINED_UNIFORM_BLOCKS,
    maxTransformFeedbackSeparateAttribs: l.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS,
    maxTransformFeedbackInterleavedComponents: l.MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS,
    maxTransformFeedbackSeparateComponents: l.MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS,
  };
}

/** Mark `refs` bits for every Program.uniformBlock matching a shader block decl. */
function markBlockRefs(refs: Uint32Array, blocks: GlslShader['info']['uniformBlocks'], bit: number, pm: GlslProgram): void {
  for (const bd of blocks) {
    for (let i = 0; i < pm.uniformBlocks.length; i++) {
      const bn = pm.uniformBlocks[i].name;
      if (bn === bd.name || (bd.arraySize > 1 && bn.startsWith(`${bd.name}[`))) refs[i] |= bit;
    }
  }
}

/** linkProgram core (both failure paths clear the executable). */
function failLink(p: WebGLProgram, log: string): void {
  p._linkStatus = false;
  p._infoLog = log;
  p._program = null;
  p._uniformStore = null;
  p._blockStores = [];
  programModels.delete(p);
}

function doLinkProgram(ctx: WebGLRenderingContext, p: WebGLProgram): void {
  linkGen.set(p, (linkGen.get(p) ?? 0) + 1); // invalidate old uniform locations
  const shaders = [...p._attachedShaders];
  const vs = shaders.find((s) => s._type === C1.VERTEX_SHADER);
  const fs = shaders.find((s) => s._type === C1.FRAGMENT_SHADER);
  const vsR = vs !== undefined ? shaderResults.get(vs) : undefined;
  const fsR = fs !== undefined ? shaderResults.get(fs) : undefined;
  if (vs === undefined || fs === undefined || !vs._compileStatus || !fs._compileStatus || !vsR || !fsR) {
    failLink(p, 'missing vertex/fragment shader');
    return;
  }
  const opts: LinkOptions = { limits: buildLinkLimits(ctx), attribBindings: p._bindAttribLocations };
  if (p._transformFeedbackVaryings !== null) {
    opts.transformFeedback = {
      varyings: p._transformFeedbackVaryings,
      bufferMode: p._tfBufferMode === C2.SEPARATE_ATTRIBS ? 'SEPARATE_ATTRIBS' : 'INTERLEAVED_ATTRIBS',
    };
  }
  let result: ReturnType<typeof linkProgram>;
  try {
    result = linkProgram(vsR, fsR, opts);
  } catch (e) {
    failLink(p, `internal compiler error: ${e instanceof Error ? e.message : String(e)}`);
    return;
  }
  if (!result.ok) {
    failLink(p, result.log);
    return;
  }
  const pm = result.program;
  p._program = pm as unknown as ProgramModel;
  p._linkStatus = true;
  p._infoLog = '';
  // _uniformStore aliases floatStore's ArrayBuffer (see header "STORE LAYOUT").
  p._uniformStore = new DataView(pm.floatStore.buffer, pm.floatStore.byteOffset, pm.floatStore.byteLength);
  p._blockStores = pm.uniformBlocks.map((b) => new DataView(new ArrayBuffer(Math.max(b.size, 16))));
  if (p._tfBufferMode === 0) p._tfBufferMode = C2.INTERLEAVED_ATTRIBS; // spec default
  programModels.set(p, pm);
  uniformBlockBindings.set(p, new Uint32Array(pm.uniformBlocks.length));
  const refs = new Uint32Array(pm.uniformBlocks.length);
  if (vsR) markBlockRefs(refs, vsR.info.uniformBlocks, 1, pm);
  if (fsR) markBlockRefs(refs, fsR.info.uniformBlocks, 2, pm);
  blockReferenced.set(p, refs);
  const fmap = new Map<string, number>();
  for (const o of fsR.info.outputs) {
    // gl_FragColor / gl_FragData[N] are not user outputs (ES 3.00 only).
    if (o.index !== null || o.name.startsWith('gl_Frag')) continue;
    fmap.set(o.name, o.location ?? 0);
  }
  fragDataMaps.set(p, fmap);
}

// ---------------------------------------------------------------------------
// getUniform
// ---------------------------------------------------------------------------

/** Read one component of one array element from the store. */
function readStoreValue(pm: GlslProgram, uniform: GlslProgram['uniforms'][number], elementIdx: number, comp: number, asUint: boolean): number {
  const slot = uniform.location + elementIdx * elementSlots(uniform);
  if (isMatrixType(uniform.type)) return pm.floatStore[slot * 4 + comp];
  if (isFloatType(uniform.type)) return pm.floatStore[slot * 4 + comp];
  const v = pm.intStore[slot * 4 + comp];
  return asUint ? v >>> 0 : v;
}

function isFloatType(type: number): boolean {
  return type === C1.FLOAT || type === C1.FLOAT_VEC2 || type === C1.FLOAT_VEC3 || type === C1.FLOAT_VEC4 || isMatrixType(type);
}
function isIntType(type: number): boolean {
  return type === C1.INT || type === C1.INT_VEC2 || type === C1.INT_VEC3 || type === C1.INT_VEC4;
}
function isUintType(type: number): boolean {
  return type === C2.UNSIGNED_INT || type === C2.UNSIGNED_INT_VEC2 || type === C2.UNSIGNED_INT_VEC3 || type === C2.UNSIGNED_INT_VEC4;
}
function isBoolType(type: number): boolean {
  return type === C1.BOOL || type === C1.BOOL_VEC2 || type === C1.BOOL_VEC3 || type === C1.BOOL_VEC4;
}
function isSamplerType(type: number): boolean {
  switch (type) {
    case C1.SAMPLER_2D:
    case C1.SAMPLER_CUBE:
    case C2.SAMPLER_3D:
    case C2.SAMPLER_2D_ARRAY:
    case C2.SAMPLER_2D_SHADOW:
    case C2.SAMPLER_CUBE_SHADOW:
    case C2.SAMPLER_2D_ARRAY_SHADOW:
    case C2.INT_SAMPLER_2D:
    case C2.INT_SAMPLER_3D:
    case C2.INT_SAMPLER_CUBE:
    case C2.INT_SAMPLER_2D_ARRAY:
    case C2.UNSIGNED_INT_SAMPLER_2D:
    case C2.UNSIGNED_INT_SAMPLER_3D:
    case C2.UNSIGNED_INT_SAMPLER_CUBE:
    case C2.UNSIGNED_INT_SAMPLER_2D_ARRAY:
      return true;
    default:
      return false;
  }
}

/** getUniform result for one uniform at (possibly) an array element. */
function readUniform(pm: GlslProgram, uniform: GlslProgram['uniforms'][number], isBase: boolean): any {
  const count = isBase ? uniform.size : 1; // base location → whole array
  const baseIdx = isBase ? 0 : 0; // element offset handled via location delta below
  const elementOffset = isBase ? 0 : 0;
  void baseIdx;
  void elementOffset;
  const slots = elementSlots(uniform);
  const startElem = isBase ? 0 : 0;
  void startElem;
  // element index = (location - uniform.location) / slots (integer, validated by caller)
  const read = (elem: number, comp: number): number => readStoreValue(pm, uniform, elem, comp, isUintType(uniform.type));
  const components = uniform.components;
  if (uniform.type === C1.FLOAT || uniform.type === C1.INT || uniform.type === C2.UNSIGNED_INT) {
    if (count === 1) {
      const v = read(0, 0);
      return uniform.type === C1.INT ? v : uniform.type === C2.UNSIGNED_INT ? v >>> 0 : v;
    }
    const out = uniform.type === C1.FLOAT ? new Float32Array(count) : uniform.type === C1.INT ? new Int32Array(count) : new Uint32Array(count);
    for (let e = 0; e < count; e++) out[e] = read(e, 0);
    return out;
  }
  if (isMatrixType(uniform.type)) {
    const c = matrixCols(uniform.type);
    const r = matrixRows(uniform.type);
    const out = new Float32Array(count * c * r);
    for (let e = 0; e < count; e++)
      for (let col = 0; col < c; col++)
        for (let row = 0; row < r; row++) out[e * c * r + col * r + row] = pm.floatStore[(uniform.location + (elementIdx + e) * slots + col) * 4 + row];
    return out;
  }
  if (isFloatType(uniform.type) || uniform.type === C1.BOOL_VEC2) {
    // FLOAT_VEC* (and BOOL_VEC2 per gl-uniform-arrays.html → Float32Array)
    const n = components;
    const out = new Float32Array(count * n);
    for (let e = 0; e < count; e++) for (let i = 0; i < n; i++) out[e * n + i] = read(e, i);
    return out;
  }
  if (isIntType(uniform.type) || uniform.type === C1.BOOL_VEC3) {
    const n = components;
    const out = new Int32Array(count * n);
    for (let e = 0; e < count; e++) for (let i = 0; i < n; i++) out[e * n + i] = read(e, i);
    return out;
  }
  if (isUintType(uniform.type) || uniform.type === C1.BOOL_VEC4) {
    const n = components;
    const out = new Uint32Array(count * n);
    for (let e = 0; e < count; e++) for (let i = 0; i < n; i++) out[e * n + i] = read(e, i);
    return out;
  }
  if (uniform.type === C1.BOOL) {
    if (count === 1) return read(0, 0) !== 0;
    const out = new Uint32Array(count);
    for (let e = 0; e < count; e++) out[e] = read(e, 0) !== 0 ? 1 : 0;
    return out;
  }
  if (isSamplerType(uniform.type)) {
    if (count === 1) return pm.intStore[uniform.location * 4];
    const out = new Int32Array(count);
    for (let e = 0; e < count; e++) out[e] = pm.intStore[(uniform.location + elementIdx + e) * 4];
    return out;
  }
  return null; // unreachable for active uniforms
}

// ---------------------------------------------------------------------------
// installProgramsApi
// ---------------------------------------------------------------------------

export function installProgramsApi(proto: WebGLRenderingContext): void {
  // ---- Shaders ----
  proto.createShader = function (this: WebGLRenderingContext, type: GLenum): WebGLShader | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    if (type !== C1.VERTEX_SHADER && type !== C1.FRAGMENT_SHADER) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    const shader = createObject(ctx, ShaderCtor);
    shader._type = type;
    return shader;
  };

  proto.deleteShader = function (this: WebGLRenderingContext, shader: WebGLShader | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (shader === null || shader === undefined) return;
    if (!(shader instanceof WebGLShader)) throw new TypeError(`Argument is not of type 'WebGLShader'`);
    if (shader._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (shader._deleted) return; // already deleted: silent no-op
    // Deletion is deferred while attached (spec); isShader → false immediately.
    shader._deleted = true;
    ctx._resources.untrack(shader);
  };

  proto.isShader = function (this: WebGLRenderingContext, shader: WebGLShader | null): GLboolean {
    const ctx = this;
    if (isLost(ctx)) return false;
    if (shader === null || shader === undefined) return false;
    if (!(shader instanceof WebGLShader)) throw new TypeError(`Argument is not of type 'WebGLShader'`);
    if (shader._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return false;
    }
    return !shader._deleted;
  };

  proto.shaderSource = function (this: WebGLRenderingContext, shader: WebGLShader, source: string): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const s = validateShaderQuery(ctx, shader);
    if (s === null) return;
    const src = requireString(source, 'source'); // WebIDL DOMString (TypeError for null/undefined)
    s._source = src;
    s._compileStatus = false;
    s._infoLog = '';
    s._compiled = null;
    s._translatedSource = null;
    shaderResults.delete(s);
  };

  proto.compileShader = function (this: WebGLRenderingContext, shader: WebGLShader): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const s = validateShaderQuery(ctx, shader);
    if (s === null) return;
    doCompileShader(ctx, s);
  };

  proto.getShaderParameter = function (this: WebGLRenderingContext, shader: WebGLShader, pname: GLenum): any {
    const ctx = this;
    if (isLost(ctx)) return null;
    const s = validateShaderQuery(ctx, shader);
    if (s === null) return null;
    switch (pname) {
      case C1.COMPILE_STATUS:
        return s._compileStatus;
      case C1.DELETE_STATUS:
        return s._deleted;
      case C1.SHADER_TYPE:
        return s._type;
      case CExt.COMPLETION_STATUS_KHR:
        if (ctx.getExtension('KHR_parallel_shader_compile') !== null) return true;
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
      default:
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
    }
  };

  proto.getShaderInfoLog = function (this: WebGLRenderingContext, shader: WebGLShader): string | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    const s = validateShaderQuery(ctx, shader);
    if (s === null) return null;
    return s._infoLog;
  };

  proto.getShaderSource = function (this: WebGLRenderingContext, shader: WebGLShader): string | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    const s = validateShaderQuery(ctx, shader);
    if (s === null) return null;
    return s._source;
  };

  proto.getShaderPrecisionFormat = function (this: WebGLRenderingContext, shadertype: GLenum, precisiontype: GLenum): WebGLShaderPrecisionFormat | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    if (shadertype !== C1.VERTEX_SHADER && shadertype !== C1.FRAGMENT_SHADER) {
      ctx._errors.push(C1.INVALID_ENUM);
      return null;
    }
    switch (precisiontype) {
      case C1.LOW_FLOAT:
      case C1.MEDIUM_FLOAT:
      case C1.HIGH_FLOAT:
        // WebGL spec minimums for float: range 2^127, 23-bit mantissa.
        return new WebGLShaderPrecisionFormat(127, 127, 23);
      case C1.LOW_INT:
      case C1.MEDIUM_INT:
      case C1.HIGH_INT:
        // CTS requires rangeMax ≥ 8 for LOW_INT (WebGL1 minimums: 2^7..2^7,
        // but the conformance shader-precision-format test enforces ≥ 8 for
        // lowp/mediump int → report the 32-bit range 31/30).
        return new WebGLShaderPrecisionFormat(31, 30, 0);
      default:
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
    }
  };

  // ---- Programs ----
  proto.createProgram = function (this: WebGLRenderingContext): WebGLProgram | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    return createObject(ctx, ProgramCtor);
  };

  proto.deleteProgram = function (this: WebGLRenderingContext, program: WebGLProgram | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (program === null || program === undefined) return;
    if (!(program instanceof WebGLProgram)) throw new TypeError(`Argument is not of type 'WebGLProgram'`);
    if (program._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    if (program._deleted) return; // silent no-op
    program._deleted = true;
    if (!program._inUse && !program._inTransformFeedback) {
      // Not in use: release immediately (spec: deletion deferred only while in use).
      ctx._resources.untrack(program);
      if (ctx._state.currentProgram === program) ctx._state.currentProgram = null;
    }
    // while _inUse: kept tracked; isProgram → false immediately.
  };

  proto.isProgram = function (this: WebGLRenderingContext, program: WebGLProgram | null): GLboolean {
    const ctx = this;
    if (isLost(ctx)) return false;
    if (program === null || program === undefined) return false;
    if (!(program instanceof WebGLProgram)) throw new TypeError(`Argument is not of type 'WebGLProgram'`);
    if (program._context !== ctx) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return false;
    }
    return !program._deleted;
  };

  proto.useProgram = function (this: WebGLRenderingContext, program: WebGLProgram | null): void {
    const ctx = this;
    if (isLost(ctx)) return;
    if (program === null || program === undefined) {
      if (ctx._state.currentProgram !== null) {
        ctx._state.currentProgram._inUse = false;
        if (ctx._state.currentProgram._deleted) ctx._resources.untrack(ctx._state.currentProgram);
        ctx._state.currentProgram = null;
      }
      return;
    }
    const p = validateProgram(ctx, program);
    if (p === null) return;
    if (!p._linkStatus || p._program === nul