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
 *
 * `inout` is a glsl-internal storage class (ES 3.00 function parameters only;
 * not part of the WebGL API surface). It is recorded on the param's TypeSpec
 * so semantics/codegen can distinguish by-reference parameters from plain
 * `in` parameters.
 */
export type StorageClass = 'const' | 'in' | 'out' | 'inout' | 'uniform' | 'attribute' | 'varying';

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
  /** Explicit uniform-block layout id (`std140`/`shared`/`packed`), if any. */
  blockLayout?: string;
}

/* ------------------------------------------------------------------ */
/* Declared helpers (implemented by semantics — Phase 2).              */
/* ------------------------------------------------------------------ */

/** GLenum values (spec-fixed; cross-checked against src/raster/gl-enums.ts and src/gl/constants.ts). */
const SCALAR_ENUM: Record<BaseScalar, number> = {
  float: 0x1406, // GL_FLOAT
  int: 0x1404, // GL_INT
  uint: 0x1405, // GL_UNSIGNED_INT
  bool: 0x8b56, // GL_BOOL
};

/** Vector enum base per component base: GL_FLOAT_VEC2=0x8B50, GL_INT_VEC2=0x8B53, GL_BOOL_VEC2=0x8B57, GL_UNSIGNED_INT_VEC2=0x8DC6. */
const VECTOR_ENUM_BASE: Record<BaseScalar, number> = {
  float: 0x8b50,
  int: 0x8b53,
  uint: 0x8dc6,
  bool: 0x8b57,
};

/** GL_FLOAT_MAT2=0x8B5A .. GL_FLOAT_MAT4x3=0x8B6A, keyed by "cols,rows". */
const MATRIX_ENUM: Readonly<Record<string, number>> = {
  '2,2': 0x8b5a,
  '3,3': 0x8b5b,
  '4,4': 0x8b5c,
  '2,3': 0x8b65, // mat2x3
  '2,4': 0x8b66, // mat2x4
  '3,2': 0x8b67, // mat3x2
  '3,4': 0x8b68, // mat3x4
  '4,2': 0x8b69, // mat4x2
  '4,3': 0x8b6a, // mat4x3
};

const SAMPLER_ENUM: Record<SamplerKind, number> = {
  sampler2D: 0x8b5e,
  sampler3D: 0x8b5f,
  samplerCube: 0x8b60,
  sampler2DShadow: 0x8b62,
  sampler2DArray: 0x8dc1,
  sampler2DArrayShadow: 0x8dc4,
  samplerCubeShadow: 0x8dc5,
  isampler2D: 0x8dca,
  isampler3D: 0x8dcb,
  isamplerCube: 0x8dcc,
  isampler2DArray: 0x8dcf,
  usampler2D: 0x8dd2,
  usampler3D: 0x8dd3,
  usamplerCube: 0x8dd4,
  usampler2DArray: 0x8dd7,
};

/**
 * GLenum for a GLSL type (GL_FLOAT_VEC3 = 0x8B51, ...). Throws for `void` and
 * structs; arrays map to their ELEMENT type (getActiveAttrib/getActiveUniform
 * report element GLenums for arrays). Used by getActiveAttrib/getActiveUniform/
 * getTransformFeedbackVarying (`AttribInfo.type` etc. are GLenums).
 */
export function toGLenum(type: GLSLType): number {
  switch (type.kind) {
    case 'void':
      throw new Error("toGLenum: 'void' has no GLenum");
    case 'scalar':
      return SCALAR_ENUM[type.base];
    case 'vector':
      return VECTOR_ENUM_BASE[type.base] + (type.size - 2);
    case 'matrix': {
      const e = MATRIX_ENUM[`${type.cols},${type.rows}`];
      if (e === undefined) throw new Error(`toGLenum: unsupported matrix ${typeName(type)}`);
      return e;
    }
    case 'sampler':
      return SAMPLER_ENUM[type.sampler];
    case 'struct':
      throw new Error(`toGLenum: struct type '${type.name}' has no GLenum`);
    case 'array':
      return toGLenum(type.element);
  }
}

/**
 * Component count per element: scalar → 1, vector → size, matrix → rows,
 * sampler → 1, struct/array → components of the element type (structs are
 * flattened by the linker, so this is only meaningful for scalar/vector/
 * matrix leaves). This is the `components` field of AttribInfo/UniformInfo
 * and the width raster uses when interpolating varyings.
 */
export function typeComponents(type: GLSLType): number {
  switch (type.kind) {
    case 'void':
      return 0;
    case 'scalar':
    case 'sampler':
      return 1;
    case 'vector':
      return type.size;
    case 'matrix':
      return type.rows;
    case 'struct':
      return 0;
    case 'array':
      return typeComponents(type.element);
  }
}

/** True for int/uint types (integral attributes/varyings/uniforms). */
export function isIntegral(type: GLSLType): boolean {
  switch (type.kind) {
    case 'scalar':
      return type.base === 'int' || type.base === 'uint';
    case 'vector':
      return type.base === 'int' || type.base === 'uint';
    case 'array':
      return isIntegral(type.element);
    default:
      return false;
  }
}

/** True for float scalar/vector/matrix types. */
export function isFloat(type: GLSLType): boolean {
  switch (type.kind) {
    case 'scalar':
    case 'vector':
      return type.base === 'float';
    case 'matrix':
      return true;
    case 'array':
      return isFloat(type.element);
    default:
      return false;
  }
}

/** True for sampler types. */
export function isSampler(type: GLSLType): boolean {
  switch (type.kind) {
    case 'sampler':
      return true;
    case 'array':
      return isSampler(type.element);
    default:
      return false;
  }
}

/** Array element count: 1 for non-arrays, the size for arrays (throws for unsized). */
export function typeSize(type: GLSLType): number {
  if (type.kind === 'array') {
    if (type.size === null) throw new Error(`typeSize: unsized array of ${typeName(type.element)}`);
    return type.size;
  }
  return 1;
}

/**
 * Structural equality (used for varying matching, overload resolution, ...).
 * Structs compare BY NAME; arrays compare element + size (two unsized arrays
 * are equal); matrices compare cols+rows.
 */
export function typeEquals(a: GLSLType, b: GLSLType): boolean {
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case 'void':
      return true;
    case 'scalar':
      return b.kind === 'scalar' && a.base === b.base;
    case 'vector':
      return b.kind === 'vector' && a.base === b.base && a.size === b.size;
    case 'matrix':
      return b.kind === 'matrix' && a.cols === b.cols && a.rows === b.rows;
    case 'sampler':
      return b.kind === 'sampler' && a.sampler === b.sampler;
    case 'struct':
      return b.kind === 'struct' && a.name === b.name;
    case 'array':
      return b.kind === 'array' && typeEquals(a.element, b.element) && a.size === b.size;
  }
}

const VECTOR_PREFIX: Record<BaseScalar, string> = { float: '', int: 'i', uint: 'u', bool: 'b' };

/**
 * Canonical GLSL spelling of a type ('vec3', 'sampler2D', 'mat2x3',
 * 'S[3]', ...) for error messages and shader info logs.
 */
export function typeName(type: GLSLType): string {
  switch (type.kind) {
    case 'void':
      return 'void';
    case 'scalar':
      return type.base;
    case 'vector':
      return `${VECTOR_PREFIX[type.base]}vec${type.size}`;
    case 'matrix':
      return `mat${type.cols}${type.cols === type.rows ? '' : `x${type.rows}`}`;
    case 'sampler':
      return type.sampler;
    case 'struct':
      return type.name;
    case 'array':
      return `${typeName(type.element)}[${type.size === null ? '' : String(type.size)}]`;
  }
}
