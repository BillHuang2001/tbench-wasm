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
 *  - KHR_parallel_shader_compile: when the extension is ENABLED (getExtension
 *    called — the singleton cache `ctx._extensions` is the gate, NOT
 *    getExtension() which would self-fulfill), compileShader/linkProgram become
 *    ASYNC: the work is chunked into per-stage macrotasks by
 *    api/parallel-compile.ts (shader: preprocess/tokenize/parse/analyze+finalize
 *    = 4 chunks; link: 1 chunk that first finishes the snapshot shaders'
 *    pending compiles) and COMPLETION_STATUS_KHR (0x91B1) reports the boolean
 *    "no pending work" WITHOUT triggering anything. Every other query/draw/
 *    uniform-write triggers the pending work synchronously (ensureShaderCompiled/
 *    ensureProgramLinked). While the context is lost, COMPLETION_STATUS_KHR
 *    returns true (checked before the lost/validation guards — the objects were
 *    created before loss and are invalidated). When the extension is not
 *    exposed: INVALID_ENUM + null (unchanged).
 *
 * UNIFORM STORE LAYOUT (single source of truth, matches src/glsl/program.ts):
 * glsl Program.floatStore (Float32Array) / intStore (Int32Array) hold the
 * default-block uniforms; UniformInfo.location is a FLOAT index into the store
 * (NOT a vec4 slot — see src/glsl/linker.ts "UNIFORM STORE LAYOUT"). A
 * scalar/sampler occupies 1 float at L; a vecN occupies floats [L..L+N-1]; a
 * matCxR occupies C*4 consecutive floats with column `col` at L + col*4 + row
 * (column stride 4 — GLSL memory order). Arrays: scalar/sampler elements pack
 * DENSELY (element k at L+k, stride 1); vector elements stride 4 floats;
 * matrix elements stride cols*4 (`elementSlots()` below returns this float
 * stride, matching the linker's elemFloatStride / codegen's dynamic stride).
 * gl/ writes via:
 *  - floats: `_uniformStore` = DataView over floatStore's ArrayBuffer
 *    (byte offset = floatIndex*4, little-endian);
 *  - ints/uints/bools/samplers: `programModels.get(p).intStore[floatIndex]`
 *    (uint stored as raw int32 bits; the generated code reinterprets via >>> 0).
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
import type { Shader as GlslShader, Program as GlslProgram, LinkOptions, CompileResult } from '../../glsl';
import { preprocess } from '../../glsl/preprocessor';
import type { PreprocessResult } from '../../glsl/preprocessor';
import { tokenize } from '../../glsl/lexer';
import type { Token } from '../../glsl/lexer';
import { parse } from '../../glsl/parser';
import type { TranslationUnit } from '../../glsl/ast';
import { analyze } from '../../glsl/semantics';
import {
  enqueueShaderCompile,
  enqueueProgramLink,
  isShaderPending,
  isProgramPending,
  cancelPendingShader,
  ensureShaderCompiled,
  ensureProgramLinked,
} from './parallel-compile';
import type { PendingItem } from './parallel-compile';
import type { GLboolean, GLenum, GLint, GLuint } from '../types';

// Re-export the KHR trigger helpers for the other trigger sites (draw.ts,
// api/uniforms.ts, api/webgl2.ts, extensions/misc.ts).
export { isShaderPending, isProgramPending, ensureShaderCompiled, ensureProgramLinked } from './parallel-compile';

/**
 * MAX_COMPILE_ERRORS: the glsl compiler caps per-stage error lists at 20
 * (src/glsl/compiler.ts — not exported; hardcoded here to mirror it for the
 * deferred path's per-stage slicing).
 */
const MAX_COMPILE_ERRORS = 20;

/** KHR_parallel_shader_compile enabled? Gate on the singleton CACHE (only
 * getExtensionObject populates it) — getExtension() would create the singleton
 * and self-fulfill the COMPLETION_STATUS_KHR queries. */
function khrEnabled(ctx: WebGLRenderingContext): boolean {
  return ctx._extensions.has('KHR_parallel_shader_compile');
}

// ---- Module-level per-program / per-shader state (objects files are pinned; keep here) ----

/** Full glsl compile result per WebGLShader (link input). */
const shaderResults = new WeakMap<WebGLShader, GlslShader>();
/** glsl Program per linked WebGLProgram (stores, uniformMap, TF varyings). */
export const programModels = new WeakMap<WebGLProgram, GlslProgram>();
/** Attach refcount per WebGLShader (deferred deletion while attached). */
const shaderAttachCounts = new WeakMap<WebGLShader, number>();
/** Link generation counter per WebGLProgram (relink invalidation). */
export const linkGen = new WeakMap<WebGLProgram, number>();
/** Link generation captured at getUniformLocation (per WebGLUniformLocation). */
export const locGen = new WeakMap<WebGLUniformLocation, number>();
/** Array-element + whole-array info per WebGLUniformLocation (getUniform). */
const uniformLocInfo = new WeakMap<WebGLUniformLocation, { elem: number; whole: boolean }>();

/**
 * Array-element + whole-array info per WebGLUniformLocation, shared with the
 * uniform* write paths (api/uniforms.ts). Locations from a previous link are
 * rejected before this is consulted (locGen/linkGen), so the entry is always
 * current. Default: element 0, whole array (bare-name locations).
 */
export function getUniformLocationInfo(loc: WebGLUniformLocation): { elem: number; whole: boolean } {
  return uniformLocInfo.get(loc) ?? { elem: 0, whole: true };
}
/** validateProgram result per WebGLProgram (VALIDATE_STATUS). */
const validateStatus = new WeakMap<WebGLProgram, boolean>();
/** uniformBlockBinding: blockIndex → binding point (default 0), per program. */
const uniformBlockBindings = new WeakMap<WebGLProgram, Uint32Array>();
/** Block referenced flags per program: bit 1 = VS, bit 2 = FS. */
const blockReferenced = new WeakMap<WebGLProgram, Uint32Array>();
/** Fragment output name → location (getFragDataLocation), per program. */
const fragDataMaps = new WeakMap<WebGLProgram, Map<string, number>>();

/**
 * Context-loss guard: no-op while lost WITHOUT generating an error (CTS
 * context-lost.html asserts NO_ERROR after every void call while lost — the
 * single CONTEXT_LOST_WEBGL is delivered via getError's lost-epoch, lost.ts).
 */
function isLost(ctx: WebGLRenderingContext): boolean {
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
 * Context-lost INVALIDATED shaders (flag set by lost.ts loseContext) →
 * INVALID_OPERATION per spec — checked AFTER deleted so a page-deleted object
 * keeps its INVALID_VALUE (CTS gl-get-attrib-location-errors.html post-restore
 * expects INVALID_VALUE for deletedProgram).
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
  if ((shader as unknown as { _invalidated?: boolean })._invalidated) {
    ctx._errors.push(C1.INVALID_OPERATION); // invalidated by context loss (spec)
    return null;
  }
  return shader;
}

/**
 * Query validation for programs (deleted-but-in-use programs remain valid).
 * Context-lost INVALIDATED programs → INVALID_OPERATION even while in use —
 * checked AFTER deleted so a page-deleted object keeps INVALID_VALUE (CTS
 * gl-get-attrib-location-errors.html post-restore expects INVALID_VALUE for
 * deletedProgram, INVALID_OPERATION for a pre-loss program still in use).
 */
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
  if ((program as unknown as { _invalidated?: boolean })._invalidated) {
    ctx._errors.push(C1.INVALID_OPERATION); // invalidated by context loss (spec)
    return null;
  }
  return program;
}

// ---- Name validation (WebGL 1.0 spec §5.14.10 / WebGL2: limits 256 vs 1024) ----

/**
 * Characters forbidden in attribute/uniform/block names: anything outside the
 * GLSL source character set (WebGL spec "Characters Outside the GLSL Source
 * Character Set" — ISO/IEC 646/ASCII subset of GLSL ES §3.1). That is: the six
 * ASCII symbols `" $ ' @ \ `` plus EVERY non-ASCII character (e.g. 'à' — CTS
 * gl-get-attrib-location-errors.html expects INVALID_VALUE). All other ASCII —
 * letters, digits, underscore, the GLSL symbols `. + - / * % < > [ ] ( ) { }
 * ^ | & ~ = ! : ; , ? #`, and whitespace — is allowed so array suffixes
 * ('u[2]') and struct paths ('u.m') keep working at the getUniformLocation
 * call site.
 */
function hasBadNameChars(name: string): boolean {
  for (let i = 0; i < name.length; i++) {
    const c = name.charCodeAt(i);
    if (c === 0x09 || c === 0x0a || c === 0x0b || c === 0x0c || c === 0x0d || c === 0x20) {
      continue; // whitespace (tab/LF/VT/FF/CR/space) is in the GLSL char set
    }
    if (c < 0x21 || c >= 0x7f) return true; // control / non-ASCII (e.g. 'à')
    if (
      (c >= 0x30 && c <= 0x39) || // 0-9
      (c >= 0x41 && c <= 0x5a) || // A-Z
      (c >= 0x61 && c <= 0x7a) || // a-z
      c === 0x5f // _
    ) {
      continue;
    }
    switch (c) {
      // GLSL symbols: . + - / * % < > [ ] ( ) { } ^ | & ~ = ! : ; , ? #
      case 0x2e: case 0x2b: case 0x2d: case 0x2f: case 0x2a: case 0x25:
      case 0x3c: case 0x3e: case 0x5b: case 0x5d: case 0x28: case 0x29:
      case 0x7b: case 0x7d: case 0x5e: case 0x7c: case 0x26: case 0x7e:
      case 0x3d: case 0x21: case 0x3a: case 0x3b: case 0x2c: case 0x3f:
      case 0x23:
        continue;
      default:
        return true; // `" $ ' @ \` and anything else not in the set
    }
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
 * FLOAT stride between consecutive array elements in the default-block store
 * (the linker's elemFloatStride — also codegen's dynamic-index stride):
 * scalar/sampler → 1 (dense packing), vector → 4, matrix → cols*4
 * (column-major). With UniformInfo.location being a FLOAT index, element k of
 * an array starts at float `location + k * elementSlots(uniform)`.
 */
export function elementSlots(uniform: { type: number; components: number }): number {
  if (isMatrixType(uniform.type)) return matrixCols(uniform.type) * 4;
  return uniform.components > 1 ? 4 : 1;
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

/** Reset compile state (compileShader entry + shaderSource). */
function resetShaderCompileState(shader: WebGLShader): void {
  shader._compileStatus = false;
  shader._infoLog = '';
  shader._compiled = null;
  shader._translatedSource = null;
  shaderResults.delete(shader);
}

/**
 * Shared compile finalize — replicates doCompileShader's observable state
 * writes exactly. Used by the sync path AND the deferred path (async chunks
 * and the ensureShaderCompiled trigger run the SAME step closures, so both
 * produce identical results).
 */
function finalizeShaderCompile(shader: WebGLShader, source: string, result: CompileResult): void {
  if (result.ok) {
    shader._compileStatus = true;
    shader._compiled = { ok: true };
    shader._translatedSource = source; // WEBGL_debug_shaders (no real translation)
    shaderResults.set(shader, result.shader);
  } else {
    shader._infoLog = result.errors.map((e) => `ERROR: 0:${e.line}: ${e.message}`).join('\n');
    shader._compiled = { ok: false, errors: result.errors };
    shader._translatedSource = '';
  }
}

/** Synchronous compile from the shader's current source (extension disabled — unchanged behavior). */
function doCompileShader(ctx: WebGLRenderingContext, shader: WebGLShader): void {
  let result: ReturnType<typeof compileShader>;
  try {
    result = compileShader(shader._source, {
      type: shader._type === C1.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT',
      version: ctx._version === 2 ? 300 : 100,
      defines: {},
      extensions: new Set(ctx.getSupportedExtensions()),
    });
  } catch (e) {
    // glsl/ must never let the exception reach the page: behave as a compile
    // failure with a diagnostic info log.
    shader._infoLog = `internal compiler error: ${e instanceof Error ? e.message : String(e)}`;
    return;
  }
  finalizeShaderCompile(shader, shader._source, result);
}

/** Deferred compile inputs, snapshotted at compileShader call time. */
interface ShaderCompileSnapshot {
  source: string;
  type: 'VERTEX' | 'FRAGMENT';
  version: 100 | 300;
  defines: Record<string, string>;
  extensions: Set<string>;
}

/**
 * Enqueue the 4-chunk deferred compile (extension enabled): preprocess →
 * tokenize → parse → analyze+finalize, sharing one `inter` object. The stage
 * orchestration mirrors src/glsl/compiler.ts compileShader exactly (per-stage
 * errors sliced to MAX_COMPILE_ERRORS); a stage failure finalizes early
 * (item.complete). A glsl-stage exception → onError (only _infoLog set,
 * _compileStatus stays false — mirroring doCompileShader's catch).
 */
function enqueueCompile(ctx: WebGLRenderingContext, shader: WebGLShader): void {
  const snapshot: ShaderCompileSnapshot = {
    source: shader._source,
    type: shader._type === C1.VERTEX_SHADER ? 'VERTEX' : 'FRAGMENT',
    version: ctx._version === 2 ? 300 : 100,
    defines: {},
    extensions: new Set(ctx.getSupportedExtensions()),
  };
  const inter: {
    pp?: Extract<PreprocessResult, { ok: true }>;
    tokens?: Token[];
    ast?: TranslationUnit;
  } = {};
  // Steps only ever run AFTER enqueueShaderCompile assigns `item` (the
  // scheduler is macrotask-based; triggers run later), so the capture is safe.
  let item: PendingItem;
  const finish = (result: CompileResult): void => {
    finalizeShaderCompile(shader, snapshot.source, result);
    item.complete = true;
  };
  item = enqueueShaderCompile(
    ctx,
    shader,
    [
      () => {
        if (item.complete) return;
        const pp = preprocess(snapshot.source, { version: snapshot.version, defines: snapshot.defines, extensions: snapshot.extensions });
        if (!pp.ok) {
          finish({ ok: false, errors: pp.errors.slice(0, MAX_COMPILE_ERRORS) });
          return;
        }
        inter.pp = pp;
      },
      () => {
        if (item.complete) return;
        const lex = tokenize(inter.pp!.tokens, inter.pp!.version);
        if (!lex.ok) {
          finish({ ok: false, errors: lex.errors.slice(0, MAX_COMPILE_ERRORS) });
          return;
        }
        inter.tokens = lex.tokens;
      },
      () => {
        if (item.complete) return;
        const parsed = parse(inter.tokens!, { version: inter.pp!.version, extensionDirectives: inter.pp!.extensionDirectives });
        if (!parsed.ok) {
          finish({ ok: false, errors: parsed.errors.slice(0, MAX_COMPILE_ERRORS) });
          return;
        }
        inter.ast = parsed.ast;
      },
      () => {
        if (item.complete) return;
        const analyzed = analyze(inter.ast!, { type: snapshot.type, extensions: new Set(inter.pp!.extensions) });
        if (!analyzed.ok) {
          finish({ ok: false, errors: analyzed.errors });
          return;
        }
        finish({
          ok: true,
          shader: {
            type: snapshot.type,
            version: inter.ast!.version,
            source: snapshot.source,
            infoLog: '',
            extensions: new Set(inter.pp!.extensions),
            ast: inter.ast!,
            info: analyzed.info,
          },
        });
      },
    ],
    (e) => {
      shader._infoLog = `internal compiler error: ${e instanceof Error ? e.message : String(e)}`;
    },
  );
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

/** Snapshot of link inputs at linkProgram call time (deferred link). */
interface LinkSnapshot {
  shaders: WebGLShader[];
  attribBindings: Map<string, number>;
  transformFeedback: LinkOptions['transformFeedback'];
}

/** Build the LinkOptions for a snapshot (context limits read at execution time). */
function linkOptsFrom(ctx: WebGLRenderingContext, snap: LinkSnapshot): LinkOptions {
  const opts: LinkOptions = { limits: buildLinkLimits(ctx), attribBindings: snap.attribBindings };
  if (snap.transformFeedback !== undefined) opts.transformFeedback = snap.transformFeedback;
  return opts;
}

/**
 * Shared link success finalize — replicates doLinkProgram's success writes
 * exactly (both the sync path and the deferred link step call this, so their
 * observable results are identical).
 */
function finalizeLinkSuccess(p: WebGLProgram, vsR: GlslShader, fsR: GlslShader, result: { ok: true; program: GlslProgram }): void {
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

/** Execute a link from a snapshot + finalize — shared by the sync path and the deferred link step. */
function executeLink(ctx: WebGLRenderingContext, p: WebGLProgram, snap: LinkSnapshot): void {
  const vs = snap.shaders.find((s) => s._type === C1.VERTEX_SHADER);
  const fs = snap.shaders.find((s) => s._type === C1.FRAGMENT_SHADER);
  const vsR = vs !== undefined ? shaderResults.get(vs) : undefined;
  const fsR = fs !== undefined ? shaderResults.get(fs) : undefined;
  if (vs === undefined || fs === undefined || !vs._compileStatus || !fs._compileStatus || !vsR || !fsR) {
    failLink(p, 'missing vertex/fragment shader');
    return;
  }
  let result: ReturnType<typeof linkProgram>;
  try {
    result = linkProgram(vsR, fsR, linkOptsFrom(ctx, snap));
  } catch (e) {
    failLink(p, `internal compiler error: ${e instanceof Error ? e.message : String(e)}`);
    return;
  }
  if (!result.ok) {
    failLink(p, result.log);
    return;
  }
  finalizeLinkSuccess(p, vsR, fsR, result);
}

/** Synchronous link (extension disabled — unchanged behavior). */
function doLinkProgram(ctx: WebGLRenderingContext, p: WebGLProgram): void {
  linkGen.set(p, (linkGen.get(p) ?? 0) + 1); // invalidate old uniform locations
  executeLink(ctx, p, {
    shaders: [...p._attachedShaders],
    attribBindings: new Map(p._bindAttribLocations),
    transformFeedback: p._transformFeedbackVaryings !== null
      ? {
          varyings: p._transformFeedbackVaryings,
          bufferMode: p._tfBufferMode === C2.SEPARATE_ATTRIBS ? 'SEPARATE_ATTRIBS' : 'INTERLEAVED_ATTRIBS',
        }
      : undefined,
  });
}

/** Enqueue the 1-chunk deferred link (extension enabled). */
function enqueueLink(ctx: WebGLRenderingContext, p: WebGLProgram): void {
  const snap: LinkSnapshot = {
    shaders: [...p._attachedShaders],
    attribBindings: new Map(p._bindAttribLocations),
    transformFeedback: p._transformFeedbackVaryings !== null
      ? {
          varyings: [...p._transformFeedbackVaryings],
          bufferMode: p._tfBufferMode === C2.SEPARATE_ATTRIBS ? 'SEPARATE_ATTRIBS' : 'INTERLEAVED_ATTRIBS',
        }
      : undefined,
  };
  enqueueProgramLink(
    ctx,
    p,
    [
      () => {
        // Ordering guarantee (CTS: program-true implies shader-true): finish
        // the snapshot shaders' in-flight compiles BEFORE linking.
        for (const s of snap.shaders) ensureShaderCompiled(ctx, s);
        // linkGen bumps at link EXECUTION time (not call time) for the
        // deferred path.
        linkGen.set(p, (linkGen.get(p) ?? 0) + 1);
        executeLink(ctx, p, snap);
      },
    ],
    (e) => {
      failLink(p, `internal compiler error: ${e instanceof Error ? e.message : String(e)}`);
    },
  );
}

// ---------------------------------------------------------------------------
// getUniform
// ---------------------------------------------------------------------------

/** Read one component of one array element from the store (float-indexed). */
function readStoreValue(pm: GlslProgram, uniform: GlslProgram['uniforms'][number], elementIdx: number, comp: number, asUint: boolean): number {
  const base = uniform.location + elementIdx * elementSlots(uniform);
  if (isMatrixType(uniform.type)) {
    // comp is the flat column-major index (col*rows+row); the store lays
    // columns at stride 4 floats.
    const r = matrixRows(uniform.type);
    return pm.floatStore[base + Math.floor(comp / r) * 4 + (comp % r)];
  }
  if (isFloatType(uniform.type)) return pm.floatStore[base + comp];
  const v = pm.intStore[base + comp];
  return asUint ? v >>> 0 : v;
}

function isFloatType(type: number): boolean {
  return type === C1.FLOAT || type === C1.FLOAT_VEC2 || type === C1.FLOAT_VEC3 || type === C1.FLOAT_VEC4 || isMatrixType(type);
}
function isIntType(type: number): boolean {
  return type === C1.INT || type === C1.INT_VEC2 || type === C1.INT_VEC3 || type === C1.INT_VEC4;
}
function isUintType(type: number): boolean {
  return type === C1.UNSIGNED_INT || type === C2.UNSIGNED_INT_VEC2 || type === C2.UNSIGNED_INT_VEC3 || type === C2.UNSIGNED_INT_VEC4;
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
function readUniform(pm: GlslProgram, uniform: GlslProgram['uniforms'][number], elem: number, whole: boolean): any {
  const count = whole ? uniform.size : 1; // base location → whole array; element location → single value
  const slots = elementSlots(uniform);
  const read = (e: number, comp: number): number => readStoreValue(pm, uniform, e, comp, isUintType(uniform.type));
  const components = uniform.components;
  if (uniform.type === C1.FLOAT || uniform.type === C1.INT || uniform.type === C1.UNSIGNED_INT) {
    if (count === 1) {
      const v = read(elem, 0);
      return uniform.type === C1.INT ? v : uniform.type === C1.UNSIGNED_INT ? v >>> 0 : v;
    }
    const out = uniform.type === C1.FLOAT ? new Float32Array(count) : uniform.type === C1.INT ? new Int32Array(count) : new Uint32Array(count);
    for (let e = 0; e < count; e++) out[e] = read(elem + e, 0);
    return out;
  }
  if (isMatrixType(uniform.type)) {
    const c = matrixCols(uniform.type);
    const r = matrixRows(uniform.type);
    const out = new Float32Array(count * c * r);
    // slots = float stride per element (cols*4); column col at +col*4 floats.
    for (let e = 0; e < count; e++)
      for (let col = 0; col < c; col++)
        for (let row = 0; row < r; row++) out[e * c * r + col * r + row] = pm.floatStore[uniform.location + (elem + e) * slots + col * 4 + row];
    return out;
  }
  if (isFloatType(uniform.type) || uniform.type === C1.BOOL_VEC2) {
    // FLOAT_VEC* (and BOOL_VEC2 per gl-uniform-arrays.html → Float32Array)
    const n = components;
    const out = new Float32Array(count * n);
    for (let e = 0; e < count; e++) for (let i = 0; i < n; i++) out[e * n + i] = read(elem + e, i);
    return out;
  }
  if (isIntType(uniform.type) || uniform.type === C1.BOOL_VEC3) {
    const n = components;
    const out = new Int32Array(count * n);
    for (let e = 0; e < count; e++) for (let i = 0; i < n; i++) out[e * n + i] = read(elem + e, i);
    return out;
  }
  if (isUintType(uniform.type) || uniform.type === C1.BOOL_VEC4) {
    const n = components;
    const out = new Uint32Array(count * n);
    for (let e = 0; e < count; e++) for (let i = 0; i < n; i++) out[e * n + i] = read(elem + e, i);
    return out;
  }
  if (uniform.type === C1.BOOL) {
    if (count === 1) return read(elem, 0) !== 0;
    const out = new Uint32Array(count);
    for (let e = 0; e < count; e++) out[e] = read(elem + e, 0) !== 0 ? 1 : 0;
    return out;
  }
  if (isSamplerType(uniform.type)) {
    if (count === 1) return pm.intStore[uniform.location + elem];
    const out = new Int32Array(count);
    for (let e = 0; e < count; e++) out[e] = pm.intStore[uniform.location + elem + e];
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
    // No [WebGLHandlesContextLoss] on createShader: while lost it still creates
    // an object (CTS context-lost.html asserts createShader → non-null and
    // isShader → false) and generates NO error — the type check is skipped too
    // (the implementation is not run per the spec's lost-context algorithm).
    if (ctx._isLost) {
      if (type !== C1.VERTEX_SHADER && type !== C1.FRAGMENT_SHADER) return null;
      return createObject(ctx, ShaderCtor);
    }
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
    cancelPendingShader(ctx, s); // drop any in-flight deferred compile (new source)
    s._source = src;
    resetShaderCompileState(s);
  };

  proto.compileShader = function (this: WebGLRenderingContext, shader: WebGLShader): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const s = validateShaderQuery(ctx, shader);
    if (s === null) return;
    resetShaderCompileState(s);
    if (khrEnabled(ctx)) {
      enqueueCompile(ctx, s); // deferred: 4 chunks, COMPLETION_STATUS_KHR stays false meanwhile
    } else {
      doCompileShader(ctx, s); // synchronous (unchanged)
    }
  };

  proto.getShaderParameter = function (this: WebGLRenderingContext, shader: WebGLShader, pname: GLenum): any {
    const ctx = this;
    // KHR: COMPLETION_STATUS_KHR is TRUE while the context is lost — checked
    // BEFORE the lost/validation guards (the objects were created before loss
    // and are invalidated; the pending work is moot).
    if (pname === CExt.COMPLETION_STATUS_KHR && ctx._isLost) return true;
    if (isLost(ctx)) return null;
    const s = validateShaderQuery(ctx, shader);
    if (s === null) return null;
    switch (pname) {
      case C1.COMPILE_STATUS:
        ensureShaderCompiled(ctx, s); // trigger: status reflects real compilation
        return s._compileStatus;
      case C1.DELETE_STATUS:
        ensureShaderCompiled(ctx, s);
        return s._deleted;
      case C1.SHADER_TYPE:
        ensureShaderCompiled(ctx, s);
        return s._type;
      case CExt.COMPLETION_STATUS_KHR:
        // Boolean "no pending work" — MUST NOT trigger anything (the CTS page
        // measures query latency ≤ 100ms). Extension not enabled →
        // INVALID_ENUM + null (checked BEFORE getExtension is called).
        if (!khrEnabled(ctx)) {
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
        }
        return !isShaderPending(s);
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
    ensureShaderCompiled(ctx, s); // trigger: log reflects real compilation
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
    // No [WebGLHandlesContextLoss] on createProgram: while lost it still
    // creates an object (CTS context-lost.html: createProgram → non-null,
    // isProgram → false, no error).
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

  // getAttachedShaders is not declared on the context class (implementation
  // gap) — installed via cast, same pattern as the getActiveUniforms alias
  // below. Returns the shaders attached to the program (spec §5.14.10).
  (proto as unknown as { getAttachedShaders?: (program: WebGLProgram) => WebGLShader[] | null }).getAttachedShaders =
    function (this: WebGLRenderingContext, program: WebGLProgram): WebGLShader[] | null {
      const ctx = this;
      if (isLost(ctx)) return null;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return null;
      ensureProgramLinked(ctx, p);
      return Array.from(p._attachedShaders);
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
    ensureProgramLinked(ctx, p); // trigger: only a linked program can be used
    if (!p._linkStatus || p._program === null) {
      // Spec: useProgram on a program that has not been successfully linked
      // generates INVALID_OPERATION (ES 2.0 §2.10.3 / WebGL spec).
      ctx._errors.push(C1.INVALID_OPERATION);
      return;
    }
    const prev = ctx._state.currentProgram;
    if (prev !== null) {
      prev._inUse = false;
      if (prev._deleted) ctx._resources.untrack(prev);
    }
    p._inUse = true;
    ctx._state.currentProgram = p;
  };

  proto.attachShader = function (this: WebGLRenderingContext, program: WebGLProgram, shader: WebGLShader): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const p = validateProgram(ctx, program);
    if (p === null) return;
    const s = validateShader(ctx, shader);
    if (s === null) return;
    if (p._attachedShaders.has(s)) {
      ctx._errors.push(C1.INVALID_OPERATION); // already attached (CTS program-test)
      return;
    }
    for (const a of p._attachedShaders) {
      if (a._type === s._type) {
        ctx._errors.push(C1.INVALID_OPERATION); // same shader type (CTS program-test)
        return;
      }
    }
    p._attachedShaders.add(s);
    shaderAttachCounts.set(s, (shaderAttachCounts.get(s) ?? 0) + 1);
  };

  proto.detachShader = function (this: WebGLRenderingContext, program: WebGLProgram, shader: WebGLShader): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const p = validateProgram(ctx, program);
    if (p === null) return;
    const s = validateShader(ctx, shader);
    if (s === null) return;
    if (!p._attachedShaders.has(s)) {
      ctx._errors.push(C1.INVALID_OPERATION); // not attached (CTS program-test)
      return;
    }
    p._attachedShaders.delete(s);
    const n = (shaderAttachCounts.get(s) ?? 1) - 1;
    if (n <= 0) shaderAttachCounts.delete(s);
    else shaderAttachCounts.set(s, n);
    // A delete-marked shader becomes fully deleted once its last attachment is
    // removed (deleteShader already untracked it; the object is now invalid).
  };

  proto.linkProgram = function (this: WebGLRenderingContext, program: WebGLProgram): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const p = validateProgram(ctx, program);
    if (p === null) return;
    if (khrEnabled(ctx)) {
      enqueueLink(ctx, p); // deferred: 1 chunk, COMPLETION_STATUS_KHR stays false meanwhile
    } else {
      doLinkProgram(ctx, p); // synchronous (unchanged)
    }
  };

  proto.validateProgram = function (this: WebGLRenderingContext, program: WebGLProgram): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const p = validateProgram(ctx, program);
    if (p === null) return;
    ensureProgramLinked(ctx, p); // trigger: VALIDATE_STATUS reflects real link state
    // Software renderer: VALIDATE_STATUS reflects the same link preconditions
    // the draw path enforces (linked program with both stages attached).
    const ok = p._linkStatus && p._program !== null && p._attachedShaders.size > 0;
    validateStatus.set(p, ok);
  };

  proto.getProgramParameter = function (this: WebGLRenderingContext, program: WebGLProgram, pname: GLenum): any {
    const ctx = this;
    if (pname === CExt.COMPLETION_STATUS_KHR && ctx._isLost) return true;
    if (isLost(ctx)) return null;
    const p = validateProgramQuery(ctx, program);
    if (p === null) return null;
    switch (pname) {
      case C1.DELETE_STATUS:
        ensureProgramLinked(ctx, p);
        return p._deleted;
      case C1.LINK_STATUS:
        // Trigger: the CTS growth loop measures REAL link duration via this
        // query — the deferred work must run synchronously here.
        ensureProgramLinked(ctx, p);
        return p._linkStatus;
      case C1.VALIDATE_STATUS:
        ensureProgramLinked(ctx, p);
        return validateStatus.get(p) ?? false;
      case C1.ATTACHED_SHADERS:
        ensureProgramLinked(ctx, p);
        return p._attachedShaders.size;
      case C1.ACTIVE_ATTRIBUTES: {
        ensureProgramLinked(ctx, p);
        const pm = programModels.get(p);
        return pm === undefined ? 0 : pm.attributes.length;
      }
      case C1.ACTIVE_UNIFORMS: {
        ensureProgramLinked(ctx, p);
        const pm = programModels.get(p);
        return pm === undefined ? 0 : pm.uniforms.length;
      }
      case C2.ACTIVE_UNIFORM_BLOCKS: {
        ensureProgramLinked(ctx, p);
        const pm = programModels.get(p);
        return pm === undefined ? 0 : pm.uniformBlocks.length;
      }
      case C2.TRANSFORM_FEEDBACK_VARYINGS: {
        ensureProgramLinked(ctx, p);
        const pm = programModels.get(p);
        return pm === undefined ? 0 : pm.transformFeedbackVaryings.length;
      }
      case C2.TRANSFORM_FEEDBACK_BUFFER_MODE:
        ensureProgramLinked(ctx, p);
        return p._tfBufferMode;
      case CExt.COMPLETION_STATUS_KHR:
        // Boolean "no pending work" — MUST NOT trigger anything. Extension not
        // enabled → INVALID_ENUM + null.
        if (!khrEnabled(ctx)) {
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
        }
        return !isProgramPending(p);
      default:
        // INFO_LOG_LENGTH / ACTIVE_*_MAX_LENGTH removed from WebGL (CTS
        // program-test expects INVALID_ENUM + null for these).
        ctx._errors.push(C1.INVALID_ENUM);
        return null;
    }
  };

  proto.getProgramInfoLog = function (this: WebGLRenderingContext, program: WebGLProgram): string | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    const p = validateProgramQuery(ctx, program);
    if (p === null) return null;
    ensureProgramLinked(ctx, p); // trigger: log reflects real link
    return p._infoLog;
  };

  proto.getAttribLocation = function (this: WebGLRenderingContext, program: WebGLProgram, name: string): GLint {
    const ctx = this;
    if (isLost(ctx)) return -1;
    const p = validateProgramQuery(ctx, program);
    if (p === null) return -1;
    ensureProgramLinked(ctx, p); // trigger before reading link state
    if (!p._linkStatus || p._program === null) {
      ctx._errors.push(C1.INVALID_OPERATION); // not linked (spec §5.14.10)
      return -1;
    }
    const nm = requireString(name, 'name');
    if (nameTooLong(ctx, nm) || badNameChars(ctx, nm)) return -1;
    if (isReservedPrefix(nm)) return -1; // no error (spec)
    const pm = programModels.get(p);
    if (pm === undefined) return -1;
    // Arrays: bare name, or 'a[0]' (location of the first element).
    const target = nm.endsWith('[0]') ? nm.slice(0, -3) : nm;
    const a = pm.attributes.find((at) => at.name === target);
    return a === undefined ? -1 : a.location;
  };

  proto.getUniformLocation = function (this: WebGLRenderingContext, program: WebGLProgram, name: string): WebGLUniformLocation | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    const p = validateProgramQuery(ctx, program);
    if (p === null) return null;
    ensureProgramLinked(ctx, p); // trigger before reading link state
    if (!p._linkStatus || p._program === null) {
      ctx._errors.push(C1.INVALID_OPERATION); // not linked (spec §5.14.10)
      return null;
    }
    const nm = requireString(name, 'name');
    if (nameTooLong(ctx, nm) || badNameChars(ctx, nm)) return null;
    if (isReservedPrefix(nm)) return null; // no error (spec)
    const pm = programModels.get(p);
    if (pm === undefined) return null;
    // glsl Program.uniformMap is keyed by canonical lookup paths ('u', 'u[2]',
    // 'u.m', 'u[0].m', ...); uniform-block members are absent → null per spec.
    // NOTE: glsl layoutUniforms builds uniformMap with FRESH UniformInfo
    // instances (per-element leaves, size 1) while Program.uniforms holds the
    // flattened entries — resolve the array entry BY NAME (object identity
    // across the two maps is not guaranteed; a -1 index would poison every
    // later uniform* call with INVALID_OPERATION).
    const u = pm.uniformMap.get(nm);
    if (u === undefined) return null;
    let entry = pm.uniforms.find((x) => x.name === u.name);
    if (entry === undefined) entry = pm.uniforms.find((x) => x.name === u.name.replace(/\[\d+\]/, '[0]'));
    if (entry === undefined) return null;
    let elem = 0;
    let whole = false;
    // elem/whole come from the ENTRY (map leaves carry size 1).
    if (entry.size > 1) {
      const m = /\[(\d+)\]$/.exec(nm);
      if (m !== null) {
        elem = parseInt(m[1], 10);
        if (elem >= entry.size) return null; // out-of-range index → null, no error
      } else {
        whole = true; // bare array name → location of the first element
      }
    }
    const idx = pm.uniforms.indexOf(entry);
    const loc = new WebGLUniformLocation(p, idx, entry.name);
    locGen.set(loc, linkGen.get(p) ?? 0);
    uniformLocInfo.set(loc, { elem, whole });
    return loc;
  };

  proto.bindAttribLocation = function (this: WebGLRenderingContext, program: WebGLProgram, index: GLuint, name: string): void {
    const ctx = this;
    if (isLost(ctx)) return;
    const p = validateProgram(ctx, program);
    if (p === null) return;
    const nm = requireString(name, 'name');
    if (isReservedPrefix(nm)) {
      ctx._errors.push(C1.INVALID_OPERATION); // reserved prefix (spec §5.14.10)
      return;
    }
    if (nameTooLong(ctx, nm) || badNameChars(ctx, nm)) return;
    if (index >= ctx._state.limits.MAX_VERTEX_ATTRIBS) {
      ctx._errors.push(C1.INVALID_VALUE);
      return;
    }
    p._bindAttribLocations.set(nm, index); // applied at the next link
  };

  proto.getActiveAttrib = function (this: WebGLRenderingContext, program: WebGLProgram, index: GLuint): WebGLActiveInfo | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    const p = validateProgramQuery(ctx, program);
    if (p === null) return null;
    ensureProgramLinked(ctx, p); // trigger before reading the program model
    const pm = programModels.get(p);
    if (pm === undefined || index >= pm.attributes.length) {
      ctx._errors.push(C1.INVALID_VALUE); // out of range (spec §5.14.10)
      return null;
    }
    const a = pm.attributes[index];
    const name = a.size > 1 ? `${a.name}[0]` : a.name; // arrays: 'a[0]' (spec)
    return new WebGLActiveInfo(a.size, a.type, name);
  };

  proto.getActiveUniform = function (this: WebGLRenderingContext, program: WebGLProgram, index: GLuint): WebGLActiveInfo | null {
    const ctx = this;
    if (isLost(ctx)) return null;
    const p = validateProgramQuery(ctx, program);
    if (p === null) return null;
    ensureProgramLinked(ctx, p); // trigger before reading the program model
    const pm = programModels.get(p);
    if (pm === undefined || index >= pm.uniforms.length) {
      ctx._errors.push(C1.INVALID_VALUE); // out of range (spec §5.14.10)
      return null;
    }
    const u = pm.uniforms[index];
    // glsl flattens leaf uniforms: name already formatted ('u[0]', 'u.m').
    return new WebGLActiveInfo(u.size, u.type, u.name);
  };

  proto.getUniform = function (this: WebGLRenderingContext, program: WebGLProgram, location: WebGLUniformLocation): any {
    const ctx = this;
    if (isLost(ctx)) return null;
    const p = validateProgramQuery(ctx, program);
    if (p === null) return null;
    if (!(location instanceof WebGLUniformLocation)) throw new TypeError(`Argument is not of type 'WebGLUniformLocation'`);
    if (location._program !== p) {
      ctx._errors.push(C1.INVALID_OPERATION); // location from a different program
      return null;
    }
    ensureProgramLinked(ctx, p); // trigger before reading link state
    if (!p._linkStatus || p._program === null) {
      ctx._errors.push(C1.INVALID_OPERATION);
      return null;
    }
    if ((locGen.get(location) ?? -1) !== (linkGen.get(p) ?? 0)) {
      ctx._errors.push(C1.INVALID_OPERATION); // location from a previous link
      return null;
    }
    const pm = programModels.get(p);
    if (pm === undefined) return null;
    const uniform = pm.uniforms[location._index];
    if (uniform === undefined || uniform.blockIndex >= 0) {
      ctx._errors.push(C1.INVALID_OPERATION); // block members have no getUniform value
      return null;
    }
    const info = uniformLocInfo.get(location) ?? { elem: 0, whole: true };
    try {
      return readUniform(pm, uniform, info.elem, info.whole);
    } catch {
      return null; // internal read failure must never escape to the page
    }
  };

  // ---- WebGL2 additions (gated: only present on the WebGL2 prototype) ----
  if ('uniformBlockBinding' in proto) {
    const p2 = proto as unknown as WebGL2RenderingContext;

    p2.transformFeedbackVaryings = function (this: WebGL2RenderingContext, program: WebGLProgram, varyings: string[], bufferMode: GLenum): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const p = validateProgram(ctx, program);
      if (p === null) return;
      const vs = Array.from(varyings ?? []).map((v) => requireString(v, 'varyings'));
      if (bufferMode !== C2.INTERLEAVED_ATTRIBS && bufferMode !== C2.SEPARATE_ATTRIBS) {
        ctx._errors.push(C1.INVALID_VALUE); // GLES 3.0 §2.12.8
        return;
      }
      if (bufferMode === C2.SEPARATE_ATTRIBS && vs.length > ctx._state.limits.MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS) {
        ctx._errors.push(C1.INVALID_VALUE); // GLES 3.0 §2.12.8
        return;
      }
      p._transformFeedbackVaryings = vs; // applied at the next link
      p._tfBufferMode = bufferMode;
    };

    p2.getTransformFeedbackVarying = function (this: WebGL2RenderingContext, program: WebGLProgram, index: GLuint): WebGLActiveInfo | null {
      const ctx = this;
      if (isLost(ctx)) return null;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return null;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const pm = programModels.get(p);
      if (pm === undefined || index >= pm.transformFeedbackVaryings.length) {
        ctx._errors.push(C1.INVALID_VALUE); // out of range (GLES 3.0 §2.12.8)
        return null;
      }
      const v = pm.transformFeedbackVaryings[index];
      return new WebGLActiveInfo(v.size, v.type, v.name);
    };

    p2.getUniformBlockIndex = function (this: WebGL2RenderingContext, program: WebGLProgram, uniformBlockName: string): GLuint {
      const ctx = this;
      if (isLost(ctx)) return C2.INVALID_INDEX;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return C2.INVALID_INDEX;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const nm = requireString(uniformBlockName, 'uniformBlockName');
      if (nameTooLong(ctx, nm) || badNameChars(ctx, nm)) return C2.INVALID_INDEX;
      const pm = programModels.get(p);
      if (pm === undefined) return C2.INVALID_INDEX;
      // Array blocks are indexed per element: 'UBOData[0]' (CTS uniform-buffers.html).
      const b = pm.uniformBlocks.find((blk) => blk.name === nm);
      return b === undefined ? C2.INVALID_INDEX : b.index;
    };

    p2.getActiveUniformBlockParameter = function (this: WebGL2RenderingContext, program: WebGLProgram, uniformBlockIndex: GLuint, pname: GLenum): any {
      const ctx = this;
      if (isLost(ctx)) return null;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return null;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const pm = programModels.get(p);
      if (pm === undefined || uniformBlockIndex >= pm.uniformBlocks.length) {
        ctx._errors.push(C1.INVALID_VALUE); // not an active block (spec §5.14.10)
        return null;
      }
      const block = pm.uniformBlocks[uniformBlockIndex];
      const refs = blockReferenced.get(p) ?? new Uint32Array(0);
      switch (pname) {
        case C2.UNIFORM_BLOCK_BINDING: {
          const bindings = uniformBlockBindings.get(p);
          return bindings !== undefined ? bindings[uniformBlockIndex] : 0;
        }
        case C2.UNIFORM_BLOCK_DATA_SIZE:
          return block.size;
        case C2.UNIFORM_BLOCK_ACTIVE_UNIFORMS:
          return block.activeUniforms.length;
        case C2.UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES: {
          const idxs: number[] = [];
          for (const m of block.activeUniforms) {
            const i = pm.uniforms.findIndex((u) => u.blockIndex === uniformBlockIndex && u.name === m.name);
            if (i >= 0) idxs.push(i);
          }
          return new Uint32Array(idxs);
        }
        case C2.UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER:
          return ((refs[uniformBlockIndex] ?? 0) & 1) !== 0;
        case C2.UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER:
          return ((refs[uniformBlockIndex] ?? 0) & 2) !== 0;
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
      }
    };

    p2.getActiveUniformBlockName = function (this: WebGL2RenderingContext, program: WebGLProgram, uniformBlockIndex: GLuint): string | null {
      const ctx = this;
      if (isLost(ctx)) return null;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return null;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const pm = programModels.get(p);
      if (pm === undefined || uniformBlockIndex >= pm.uniformBlocks.length) {
        ctx._errors.push(C1.INVALID_VALUE);
        return null;
      }
      return pm.uniformBlocks[uniformBlockIndex].name;
    };

    p2.uniformBlockBinding = function (this: WebGL2RenderingContext, program: WebGLProgram, uniformBlockIndex: GLuint, uniformBlockBinding: GLuint): void {
      const ctx = this;
      if (isLost(ctx)) return;
      const p = validateProgram(ctx, program);
      if (p === null) return;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const pm = programModels.get(p);
      if (pm === undefined || uniformBlockIndex >= pm.uniformBlocks.length) {
        ctx._errors.push(C1.INVALID_VALUE); // GLES 3.0 §2.12.6.5
        return;
      }
      if (uniformBlockBinding >= ctx._state.limits.MAX_UNIFORM_BUFFER_BINDINGS) {
        ctx._errors.push(C1.INVALID_VALUE); // GLES 3.0 §2.12.6.5
        return;
      }
      const bindings = uniformBlockBindings.get(p);
      if (bindings !== undefined) bindings[uniformBlockIndex] = uniformBlockBinding;
    };

    p2.getUniformIndices = function (this: WebGL2RenderingContext, program: WebGLProgram, uniformNames: string[]): GLuint[] | null {
      const ctx = this;
      if (isLost(ctx)) return null;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return null;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const pm = programModels.get(p);
      if (pm === undefined) return null;
      const names = Array.from(uniformNames ?? []).map((n) => requireString(n, 'uniformNames'));
      for (const n of names) {
        if (nameTooLong(ctx, n) || badNameChars(ctx, n)) return null;
      }
      // GLES 3.0 glGetUniformIndices: not-found names yield GL_INVALID_INDEX
      // (0xFFFFFFFF), no error (CTS uniform-buffers.html checks INVALID_INDEX).
      // Resolve BY NAME (uniformMap holds fresh leaf instances — identity
      // with Program.uniforms entries is not guaranteed).
      const out: number[] = [];
      for (const n of names) {
        let u = pm.uniformMap.get(n);
        if (u !== undefined) {
          const entry = pm.uniforms.find((x) => x.name === u!.name) ??
            pm.uniforms.find((x) => x.name === u!.name.replace(/\[\d+\]/, '[0]'));
          if (entry !== undefined) u = entry;
          else u = undefined;
        }
        if (u === undefined) u = pm.uniforms.find((un) => un.name === n);
        out.push(u === undefined ? C2.INVALID_INDEX : pm.uniforms.indexOf(u));
      }
      return out;
    };

    // Spec name (webgl2.idl §5.14: getActiveUniforms). The class stub declares
    // the GLES3-ish alias getActiveUniformsiv — expose the implementation under
    // both so CTS (uniform-buffers.html) and the declared stub name work.
    const activeUniformsImpl = function (this: WebGL2RenderingContext, program: WebGLProgram, uniformIndices: GLuint[], pname: GLenum): GLint[] | null {
      const ctx = this;
      if (isLost(ctx)) return null;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return null;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const pm = programModels.get(p);
      if (pm === undefined) return null;
      switch (pname) {
        case C2.UNIFORM_TYPE:
        case C2.UNIFORM_SIZE:
        case C2.UNIFORM_BLOCK_INDEX:
        case C2.UNIFORM_OFFSET:
        case C2.UNIFORM_ARRAY_STRIDE:
        case C2.UNIFORM_MATRIX_STRIDE:
        case C2.UNIFORM_IS_ROW_MAJOR:
          break;
        default:
          ctx._errors.push(C1.INVALID_ENUM);
          return null;
      }
      const indices = Array.from(uniformIndices ?? []).map((i) => i >>> 0); // WebIDL GLuint
      const out: number[] = [];
      for (const i of indices) {
        const u = pm.uniforms[i];
        if (u === undefined) {
          ctx._errors.push(C1.INVALID_VALUE); // index >= ACTIVE_UNIFORMS
          return null;
        }
        switch (pname) {
          case C2.UNIFORM_TYPE:
            out.push(u.type);
            break;
          case C2.UNIFORM_SIZE:
            out.push(u.size);
            break;
          case C2.UNIFORM_BLOCK_INDEX:
            out.push(u.blockIndex);
            break;
          default: {
            // UNIFORM_OFFSET/ARRAY_STRIDE/MATRIX_STRIDE/IS_ROW_MAJOR: -1 for
            // default-block uniforms (GLES 3.0 §2.12.6), std140 member data
            // for block members (glsl bakes these at link time).
            if (u.blockIndex < 0 || u.blockIndex >= pm.uniformBlocks.length) {
              out.push(-1);
              break;
            }
            const m = pm.uniformBlocks[u.blockIndex].activeUniforms.find((mm) => mm.name === u.name);
            if (m === undefined) {
              out.push(-1);
              break;
            }
            switch (pname) {
              case C2.UNIFORM_OFFSET:
                out.push(m.offset);
                break;
              case C2.UNIFORM_ARRAY_STRIDE:
                out.push(m.arrayStride);
                break;
              case C2.UNIFORM_MATRIX_STRIDE:
                out.push(m.matrixStride);
                break;
              default:
                out.push(m.rowMajor ? 1 : 0);
                break;
            }
            break;
          }
        }
      }
      return out;
    };
    (p2 as unknown as { getActiveUniforms: typeof activeUniformsImpl }).getActiveUniforms = activeUniformsImpl;
    (p2 as unknown as { getActiveUniformsiv: typeof activeUniformsImpl }).getActiveUniformsiv = activeUniformsImpl;

    p2.getFragDataLocation = function (this: WebGL2RenderingContext, program: WebGLProgram, name: string): GLint {
      const ctx = this;
      if (isLost(ctx)) return -1;
      const p = validateProgramQuery(ctx, program);
      if (p === null) return -1;
      ensureProgramLinked(ctx, p); // trigger before reading the program model
      const nm = requireString(name, 'name');
      if (nameTooLong(ctx, nm) || badNameChars(ctx, nm)) return -1;
      const pm = programModels.get(p);
      if (pm === undefined) return -1;
      const fmap = fragDataMaps.get(p);
      if (fmap === undefined) return -1;
      return fmap.get(nm) ?? -1;
    };
  }
}