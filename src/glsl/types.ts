/**
 * types.ts — GLSL ES type system shared by every stage of the compiler.
 *
 * This module is part of the PUBLIC API of src/glsl/. The `GLSLType` tagged
 * union below is the language the AST annotation, semantic analysis, linker
 * and codegen all speak in; the helper functions are declared here as stubs
 * and implemented by the semantics module (Phase 2).
 *
 * NOTE: the GLSL→GLenum mapping for the info arrays (`AttribInfo.type`,
 * `UniformInfo.type`, ...) is NOT exported from here — consumers (gl/) own
 * their GL constant tables and compare against them. `toGLenum` is internal.
 */

/** Base scalar component types of the language. */
export type BaseScalar = 'float' | 'int' | 'uint' | 'bool';

/**
 * Sampler kinds.
 *
 * Version availability:
 * - WebGL1 (GLSL ES 1.00): only `sampler2D` and `samplerCube`. WebGL1
 *   deliberately does NOT expose the ES 1.00 shadow samplers, and
 *   `GL_OES_texture_3D` is not a WebGL extension, so `sampler3D` etc. must
 *   compile-error in a 100 shader.
 * - WebGL2 (GLSL ES 3.00): all kinds listed here are available.
 */
export type SamplerKind =
  | 'sampler2D'
  | 'samplerCube'
  | 'sampler3D'
  | 'sampler2DArray'
  | 'sampler2DShadow'
  | 'samplerCubeShadow'
  | 'sampler2DArrayShadow'
  | 'isampler2D'
  | 'isampler3D'
  | 'isamplerCube'
  | 'isampler2DArray'
  | 'usampler2D'
  | 'usampler3D'
  | 'usamplerCube'
  | 'usampler2DArray';

/**
 * A resolved GLSL type (tagged union, immutable).
 *
 * - `{ kind: 'array', element, size }`: `size` is the element count;
 *   `size: null` marks an UNSIZED array, legal only for function parameters
 *   (ES 3.00) and constructor arguments. Nested arrays are legal in ES 3.00
 *   (`float a[2][3]` — flattened to `array(array(float,3),2)` by the parser).
 * - `{ kind: 'struct', name, members }`: equality is BY NAME (GLSL semantics).
 *   `members` is kept for std140 layout / uniform flattening.
 * - Matrices are always float; `cols`/`rows` cover ES 3.00 non-square types
 *   (mat2x3 = 2 columns, 3 rows). Codegen stores matrices column-major.
 */
export type GLSLType =
  | { kind: 'void' }
  | { kind: 'scalar'; base: BaseScalar }
  | { kind: 'vector'; base: BaseScalar; size: 2 | 3 | 4 }
  | { kind: 'matrix'; cols: 2 | 3 | 4; rows: 2 | 3 | 4 }
  | { kind: 'sampler'; sampler: SamplerKind }
  | { kind: 'struct'; name: string; members: StructMember[] }
  | { kind: 'array'; element: GLSLType; size: number | null };

/** A member of a struct type or uniform block (layout-relevant info only). */
export interface StructMember {
  name: string;
  type: GLSLType;
}

/** Precision qualifiers. All are accepted; execution is uniformly highp (float64). */
export type Precision = 'highp' | 'mediump' | 'lowp';

/**
 * Storage qualifiers. `attribute`/`varying` are the GLSL ES 1.00 spellings of
 * `in`/`out` (a 300 shader using `attribute` is a compile error, and vice
 * versa — version-dependent keyword validation lives in lexer/semantics).
 */
export type StorageClass = 'const' | 'in' | 'out' | 'uniform' | 'attribute' | 'varying';

/** Interpolation qualifiers (flat is mandatory for integral varyings). */
export type Interpolation = 'smooth' | 'flat' | 'noperspective';

/** Full qualifier bag attached to a declaration's type spec. */
export interface TypeQualifiers {
  storage?: StorageClass;
  precision?: Precision;
  interpolation?: Interpolation;
  /** `centroid` auxiliary qualifier (ES 3.00). */
  centroid?: boolean;
  /** `invariant` (linker/codegen may ignore; affects only rounding semantics). */
  invariant?: boolean;
  /** `precise` (ES 3.00; accepted, no codegen effect). */
  precise?: boolean;
  /** `layout(...)` — `location` for attributes/outputs, `binding` for samplers/UBOs. */
  layout?: LayoutQualifiers;
}

export interface LayoutQualifiers {
  location?: number;
  binding?: number;
}

/* ------------------------------------------------------------------ */
/* Declared helpers (implemented by semantics — Phase 2).              */
/* ------------------------------------------------------------------ */

/**
 * GLenum for a GLSL type (GL_FLOAT_VEC3 = 0x8B51, ...). Throws for `void`.
 * Used by getActiveAttrib/getActiveUniform/getTransformFeedbackVarying
 * (`AttribInfo.type` etc. are GLenums).
 */
export function toGLenum(type: GLSLType): number {
  throw new Error('not implemented');
}

/**
 * Component count per element: scalar → 1, vector → size, matrix → rows,
 * sampler → 1, struct/array → components of the element type (structs are
 * flattened by the linker, so this is only meaningful for scalar/vector/
 * matrix leaves). This is the `components` field of AttribInfo/UniformInfo
 * and the width raster uses when interpolating varyings.
 */
export function typeComponents(type: GLSLType): number {
  throw new Error('not implemented');
}

/** True for int/uint types (integral attributes/varyings/uniforms). */
export function isIntegral(type: GLSLType): boolean {
  throw new Error('not implemented');
}

/** True for float scalar/vector/matrix types. */
export function isFloat(type: GLSLType): boolean {
  throw new Error('not implemented');
}

/** True for sampler types. */
export function isSampler(type: GLSLType): boolean {
  throw new Error('not implemented');
}

/** Array element count: 1 for non-arrays, the size for arrays (throws for unsized). */
export function typeSize(type: GLSLType): number {
  throw new Error('not implemented');
}

/** Structural equality (used for varying matching, overload resolution, ...). */
export function typeEquals(a: GLSLType, b: GLSLType): boolean {
  throw new Error('not implemented');
}

/**
 * Canonical GLSL spelling of a type ('vec3', 'sampler2D', 'mat2x3',
 * 'S[3]', ...) for error messages and shader info logs.
 */
export function typeName(type: GLSLType): string {
  throw new Error('not implemented');
}
