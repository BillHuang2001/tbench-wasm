/**
 * builtins/types.ts — GLSL ES builtin table interfaces + type-literal helpers.
 *
 * The tables in `100.ts`, `extensions.ts` and `300.ts` are pure data consumed
 * by the semantics pass (overload resolution, builtin variable scoping,
 * gl_Max* constants) and by codegen (call lowering). This module only defines
 * the data shape plus tiny pure construction helpers — no runtime logic beyond
 * that.
 *
 * Overload resolution contract (semantics): when matching a call to a
 * BuiltinSignature, ES 1.00 allows implicit int→float conversions of
 * arguments; ES 3.00 additionally allows int→uint and uint→float.
 * `extension` gates the whole signature (a signature tagged with an extension
 * is only visible when that extension is enabled in the shader). `stage`
 * restricts the signature to one shader stage.
 */
import type { BaseScalar, GLSLType, SamplerKind } from '../types.js';

export interface BuiltinSignature {
  name: string;
  /** Parameter types; overload resolution allows implicit int→float (and, ES 3.00, int→uint, uint→float) conversions. */
  params: GLSLType[];
  ret: GLSLType;
  /** Extension required (name only; e.g. 'GL_OES_standard_derivatives'). Absent = core. */
  extension?: string;
  /** Stage restriction: 'VERTEX'|'FRAGMENT' (absent = both). */
  stage?: 'VERTEX' | 'FRAGMENT';
}

export interface BuiltinVariable {
  name: string; // 'gl_Position', 'gl_FragColor', ...
  type: GLSLType;
  stage: 'VERTEX' | 'FRAGMENT' | 'BOTH';
  /** writable (gl_Position, gl_PointSize, gl_FragColor, gl_FragData, gl_FragDepthEXT, gl_FragDepth) vs read-only */
  writable: boolean;
  extension?: string; // gl_FragData → 'GL_EXT_draw_buffers', gl_FragDepthEXT → 'GL_EXT_frag_depth'
}

export interface BuiltinConstant {
  name: string;
  value: number;
  /**
   * Extension that introduces/overrides this constant (e.g. GL_EXT_draw_buffers
   * raises gl_MaxDrawBuffers 1 → 4). Absent = core value.
   */
  extension?: string;
}

/** All signatures in `table` whose name matches (exact match). */
export function matches(name: string, table: BuiltinSignature[]): BuiltinSignature[] {
  return table.filter((s) => s.name === name);
}

/* ------------------------------------------------------------------ */
/* GLSLType literal helpers (pure construction — shared by all tables) */
/* ------------------------------------------------------------------ */

export const F: GLSLType = { kind: 'scalar', base: 'float' };
export const I: GLSLType = { kind: 'scalar', base: 'int' };
export const U: GLSLType = { kind: 'scalar', base: 'uint' };
export const B: GLSLType = { kind: 'scalar', base: 'bool' };
/** void — legal only as a return type (e.g. umulExtended/imulExtended). */
export const VOID: GLSLType = { kind: 'void' };

/** Vector of `base` with `size` components. */
export const v = (base: BaseScalar, size: 2 | 3 | 4): GLSLType => ({ kind: 'vector', base, size });

export const vec = (size: 2 | 3 | 4): GLSLType => v('float', size);
export const ivec = (size: 2 | 3 | 4): GLSLType => v('int', size);
export const uvec = (size: 2 | 3 | 4): GLSLType => v('uint', size);
export const bvec = (size: 2 | 3 | 4): GLSLType => v('bool', size);

export const V2: GLSLType = vec(2);
export const V3: GLSLType = vec(3);
export const V4: GLSLType = vec(4);

/** Square or rectangular float matrix (column-major; cols × rows). */
export const mat = (cols: 2 | 3 | 4, rows: 2 | 3 | 4 = cols): GLSLType => ({ kind: 'matrix', cols, rows });

export const smp = (sampler: SamplerKind): GLSLType => ({ kind: 'sampler', sampler });

export const arr = (element: GLSLType, size: number): GLSLType => ({ kind: 'array', element, size });

/** genType = float/vec2/vec3/vec4 (the classic 1.00 placeholder set). */
export const genType: GLSLType[] = [F, V2, V3, V4];
/** genIType = int/ivec2/ivec3/ivec4. */
export const genIType: GLSLType[] = [I, ivec(2), ivec(3), ivec(4)];
/** genUType = uint/uvec2/uvec3/uvec4. */
export const genUType: GLSLType[] = [U, uvec(2), uvec(3), uvec(4)];
/** genBType = bool/bvec2/bvec3/bvec4. */
export const genBType: GLSLType[] = [B, bvec(2), bvec(3), bvec(4)];

/** Extra fields carried by a signature (extension gate / stage restriction). */
export interface SigExtra {
  extension?: string;
  stage?: 'VERTEX' | 'FRAGMENT';
}

/** Build one signature. */
export function sig(name: string, params: GLSLType[], ret: GLSLType, extra?: SigExtra): BuiltinSignature {
  return { name, params, ret, ...extra };
}

/** One-arg genX family: f(t) → t for each t in `types`. */
export function gen(name: string, types: GLSLType[], extra?: SigExtra): BuiltinSignature[] {
  return types.map((t) => sig(name, [t], t, extra));
}

/** Two-arg genX family: f(t, t) → t for each t in `types`. */
export function gen2(name: string, types: GLSLType[], extra?: SigExtra): BuiltinSignature[] {
  return types.map((t) => sig(name, [t, t], t, extra));
}

/** Three-arg genX family: f(t, t, t) → t for each t in `types`. */
export function gen3(name: string, types: GLSLType[], extra?: SigExtra): BuiltinSignature[] {
  return types.map((t) => sig(name, [t, t, t], t, extra));
}
