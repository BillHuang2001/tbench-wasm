/**
 * compiler.ts — the public entry API of the GLSL compiler (contract §1).
 *
 * Two-phase model matching WebGL semantics:
 * - `compileShader` runs preprocessor → lexer → parser → semantic analysis
 *   and returns a `Shader` carrying the annotated AST + resolved declaration
 *   summaries (`info`) and an empty `infoLog` on success. NO codegen happens
 *   here — the generated JavaScript is produced at LINK time.
 * - `linkProgram` validates the pair (version match, varying matching,
 *   attribute/uniform/block layout, limits, transform feedback), generates
 *   the JS function bodies via `new Function('ctx', 'R', body)`, and assembles
 *   the `Program` (program.ts) that gl/ and raster/ consume.
 *
 * Error reporting: 1-based line numbers, Khronos-style messages
 * ("ERROR: 0:<line>: <message>"); `linkProgram` failures return a single
 * formatted log string (getProgramInfoLog).
 *
 * PHASE 2 STATUS: `compileShader` is implemented (preprocess → lexer → parse
 * → semantics). `linkProgram` remains a stub: the linker + codegen pipeline
 * (varying matching, attribute/uniform/block layout, std140, limits,
 * transform feedback, JS codegen) is the next executor's work.
 */
import type { TranslationUnit } from './ast.js';
import type { GLSLType, Precision } from './types.js';
import type { Program } from './program.js';
import { preprocess } from './preprocessor.js';
import { tokenize } from './lexer.js';
import { parse } from './parser.js';
import { analyze } from './semantics.js';
import { linkProgram as linkProgramImpl } from './linker.js';

/** Shader stage. */
export type ShaderStage = 'VERTEX' | 'FRAGMENT';

/* ------------------------------------------------------------------ */
/* compileShader                                                       */
/* ------------------------------------------------------------------ */

export interface CompileOptions {
  /** Shader stage. */
  type: ShaderStage;
  /**
   * Highest GLSL version the CONTEXT supports (100 = WebGL1, 300 = WebGL2).
   * The shader's actual language version comes from its `#version` directive
   * (default 100 when absent). A shader declaring a version higher than this
   * fails to compile (WebGL1 rejects `#version 300 es`). A 300 context
   * compiles both 100 and 300 es shaders.
   */
  version: 100 | 300;
  /** Extra preprocessor defines injected before the source (default: none). */
  defines?: Record<string, string>;
  /**
   * Extensions the context supports (e.g. GL_OES_standard_derivatives).
   * `#extension` directives are validated against this set; requesting an
   * unsupported extension with require/enable is a compile error.
   */
  extensions?: Set<string>;
}

export interface CompileError {
  /** 1-based source line (after #line remapping). */
  line: number;
  /** GLSL-style message, e.g. "'foo' : undeclared identifier". */
  message: string;
}

export type CompileResult =
  | { ok: true; shader: Shader }
  | { ok: false; errors: CompileError[] };

/** Maximum number of compile errors returned per stage (parser/semantics cap
 * their internal collection at the same count). */
const MAX_COMPILE_ERRORS = 20;

/**
 * A compiled shader — the opaque handle gl/ stores (getShaderSource,
 * getShaderParameter, getShaderInfoLog all read from it) and the input to
 * linkProgram.
 */
export interface Shader {
  readonly type: ShaderStage;
  /** Declared language version (from #version; default 100). */
  readonly version: 100 | 300;
  /** Original source text (as passed to compileShader). */
  readonly source: string;
  /** '' on success; the formatted log otherwise (getShaderInfoLog). */
  readonly infoLog: string;
  /** Extensions successfully enabled via #extension in this shader. */
  readonly extensions: ReadonlySet<string>;
  /** Annotated parse tree (semantics has resolved types; used by the linker). */
  readonly ast: TranslationUnit;
  /** Resolved declaration summaries — the linker's input (see below). */
  readonly info: ShaderInfo;
}

/* ------------------------------------------------------------------ */
/* ShaderInfo — resolved declarations (produced by semantics)          */
/* ------------------------------------------------------------------ */

/** Vertex input attribute (attribute/in). `location` = explicit layout(location=) or null (auto-assigned at link). */
export interface AttributeDecl {
  name: string;
  type: GLSLType; // element type (arrays: element)
  arraySize: number; // 1 for non-arrays
  location: number | null;
  /** Set to true by scanUses when the VERTEX shader reads the attribute.
   *  Inactive attributes consume no generic slots at link (native behavior)
   *  and are omitted from getActiveAttrib/getAttribLocation. */
  used: boolean;
}

/**
 * Interface varying (vertex out / fragment in), in declaration order.
 * `flat` = flat-qualified OR integral element type (integral varyings are
 * flat-only — link error otherwise). `centroid`/`noperspective`/`invariant`
 * are carried for completeness (invariant affects nothing in software).
 *
 * VARYING INTERFACE BLOCK MEMBERS (ES 3.00 `out`/`in` blocks): `blockName` is
 * the interface BLOCK's name (null for plain varyings) and `name` is the
 * '<instance>.<member>' path (bare member name for instance-less blocks).
 * Instance names may differ between stages — the linker matches block members
 * by (blockName, memberName) and emits layout keys for BOTH stages' paths.
 */
export interface VaryingDecl {
  name: string;
  /** Interface block name for block members; null for plain varyings. */
  blockName: string | null;
  type: GLSLType; // element type
  arraySize: number;
  flat: boolean;
  centroid: boolean;
  noperspective: boolean;
  invariant: boolean;
  /**
   * FRAGMENT stage only: true iff the fragment shader READS this varying's
   * value (a load of its value — a pure `=`-write target or a never-read
   * declaration stays false). The linker matches only USED fragment varyings
   * against vertex outputs (native behavior); declared-but-unread fragment
   * varyings impose no constraint. Vertex-stage entries are never read (the
   * linker ignores `used` there), but the field exists on all constructions.
   */
  used: boolean;
}

/** Default-block uniform (top-level only; the linker flattens structs/arrays). */
export interface UniformDecl {
  name: string;
  type: GLSLType; // full type incl. struct/array structure
  precision: Precision | null;
  /** layout(binding=) for samplers (ES 3.00) or null (default assigned at link). */
  binding: number | null;
}

/** Uniform block declaration (ES 3.00). */
export interface UniformBlockDecl {
  /** Block name (e.g. 'Lights'); array blocks get arraySize > 1. */
  name: string;
  instanceName: string | null;
  arraySize: number;
  /** layout(binding=) or null (default 0). */
  binding: number | null;
  members: { name: string; type: GLSLType; precision: Precision | null }[];
}

/**
 * Fragment color output.
 * - ES 1.00: 'gl_FragColor' (index null) or 'gl_FragData'[N] (index set,
 *   EXT_draw_buffers only).
 * - ES 3.00: user out variable; location = layout(location=) or null
 *   (auto-assigned at link).
 */
export interface OutputDecl {
  name: string;
  index: number | null; // gl_FragData[N] index (1.00), else null
  location: number | null; // explicit location (3.00) or null
  type: GLSLType; // FLOAT_VEC4 / INT_VEC4 / UINT_VEC4
}

/** Capability flags: which stage-specific built-ins the shader references. */
export interface ShaderUses {
  /** Vertex writes gl_PointSize. */
  pointSize: boolean;
  /** Fragment reads gl_FragCoord. */
  fragCoord: boolean;
  /** Fragment reads gl_FrontFacing. */
  frontFacing: boolean;
  /** Fragment reads gl_PointCoord. */
  pointCoord: boolean;
  /** Fragment writes gl_FragDepth (3.00) / gl_FragDepthEXT (1.00 + EXT_frag_depth). */
  fragDepth: boolean;
  /** Vertex reads gl_VertexID. */
  vertexId: boolean;
  /** Vertex reads gl_InstanceID. */
  instanceId: boolean;
  /** Vertex reads gl_DrawID (GL_ANGLE_multi_draw / WEBGL_multi_draw). */
  drawId: boolean;
  /** Fragment uses dFdx/dFdy/fwidth or any implicit-LOD texture function (→ dual-number codegen). */
  derivatives: boolean;
  /**
   * Either stage reads gl_DepthRange (builtin struct uniform, GLSL ES 1.00
   * §7.6 / 3.00 §7.7). The linker then exposes the members as ACTIVE UNIFORMS
   * ('gl_DepthRange.near/far/diff', GL_FLOAT, size 1) backed by 3 float-store
   * slots appended after the user uniforms.
   */
  depthRange: boolean;
}

/** Resolved declaration summaries in source order — the linker's input. */
export interface ShaderInfo {
  attributes: AttributeDecl[]; // vertex only
  varyings: VaryingDecl[]; // vertex outs / fragment ins
  uniforms: UniformDecl[];
  uniformBlocks: UniformBlockDecl[];
  outputs: OutputDecl[]; // fragment only
  uses: ShaderUses;
}

/* ------------------------------------------------------------------ */
/* linkProgram                                                         */
/* ------------------------------------------------------------------ */

/**
 * Link-time limits, checked against active resources (attribute locations,
 * uniform vectors, varying vectors, texture units, transform feedback).
 * When omitted from LinkOptions, the WebGL minimums for the shader version
 * are assumed (gl/ passes its context's actual getParameter values).
 */
export interface LinkLimits {
  maxVertexAttribs: number; // min 16
  maxVertexUniformVectors: number; // min 128 (ES 1.00) / 256 (ES 3.00)
  maxFragmentUniformVectors: number; // min 16 (ES 1.00) / 224 (ES 3.00)
  maxVaryingVectors: number; // min 8 (ES 1.00) / 15 (ES 3.00)
  maxVertexTextureImageUnits: number; // ES 1.00 min 0; software supports 16
  maxTextureImageUnits: number; // min 8 (ES 1.00) / 16 (ES 3.00)
  maxCombinedTextureImageUnits: number; // min 8 (ES 1.00) / 32 (ES 3.00)
  maxDrawBuffers: number; // 1 (ES 1.00) / 4 (ES 3.00)
  maxUniformBufferBindings: number; // 24 (ES 3.00)
  maxUniformBlockSize: number; // 16384 (ES 3.00)
  maxVertexUniformBlocks: number; // 12 (ES 3.00)
  maxFragmentUniformBlocks: number; // 12 (ES 3.00)
  maxCombinedUniformBlocks: number; // 24 (ES 3.00)
  maxTransformFeedbackSeparateAttribs: number; // 4 (ES 3.00)
  maxTransformFeedbackInterleavedComponents: number; // 64 (ES 3.00)
  maxTransformFeedbackSeparateComponents: number; // 4 (ES 3.00)
}

/** WebGL2 transform feedback capture specification (gl.transformFeedbackVaryings). */
export interface TransformFeedbackSpec {
  varyings: string[];
  bufferMode: 'SEPARATE_ATTRIBS' | 'INTERLEAVED_ATTRIBS';
}

export interface LinkOptions {
  /** Context limits; omitted fields default to WebGL minimums. */
  limits?: Partial<LinkLimits>;
  /** Explicit attribute bindings from bindAttribLocation (name → location). */
  attribBindings?: ReadonlyMap<string, number>;
  /** Transform feedback capture spec (WebGL2 only). */
  transformFeedback?: TransformFeedbackSpec;
}

export type LinkResult = { ok: true; program: Program } | { ok: false; log: string };

/* ------------------------------------------------------------------ */
/* Entry points (Phase 2 implements; signatures are the contract)      */
/* ------------------------------------------------------------------ */

/**
 * Compile a GLSL ES source string. Runs the full front-end
 * (preprocessor → lexer → parser → semantics) WITHOUT code generation.
 * Returns the annotated Shader on success or 1-based compile errors.
 */
export function compileShader(source: string, opts: CompileOptions): CompileResult {
  // Pass the stage through so the preprocessor defines GL_FRAGMENT_PRECISION_HIGH
  // (=1) for FRAGMENT shaders only (see preprocessor.ts `preprocess`).
  const pp = preprocess(source, { version: opts.version, type: opts.type, defines: opts.defines, extensions: opts.extensions });
  if (!pp.ok) return { ok: false, errors: pp.errors.slice(0, MAX_COMPILE_ERRORS) };
  const lex = tokenize(pp.tokens, pp.version);
  if (!lex.ok) return { ok: false, errors: lex.errors.slice(0, MAX_COMPILE_ERRORS) };
  const parsed = parse(lex.tokens, { version: pp.version, extensionDirectives: pp.extensionDirectives });
  if (!parsed.ok) return { ok: false, errors: parsed.errors.slice(0, MAX_COMPILE_ERRORS) };
  const analyzed = analyze(parsed.ast, { type: opts.type, extensions: new Set(pp.extensions) });
  if (!analyzed.ok) return { ok: false, errors: analyzed.errors };
  return {
    ok: true,
    shader: {
      type: opts.type,
      version: parsed.ast.version,
      source,
      infoLog: '',
      extensions: new Set(pp.extensions),
      ast: parsed.ast,
      info: analyzed.info,
    },
  };
}

/**
 * Link a vertex+fragment shader pair: version compatibility, varying
 * matching (name/type/size, flat rules), attribute location assignment
 * (explicit layout → attribBindings → automatic), uniform store layout
 * (vec4-slot packing, std140 for UBOs), limit checks, transform-feedback
 * validation, JS codegen for both stages, and Program assembly.
 *
 * Implemented in linker.ts.
 */
export function linkProgram(vs: Shader, fs: Shader, opts?: LinkOptions): LinkResult {
  return linkProgramImpl(vs, fs, opts);
}
