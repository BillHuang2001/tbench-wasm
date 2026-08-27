/**
 * builtins/index.ts — public surface of the GLSL ES builtin tables.
 *
 * Consumers:
 * - semantics: overload resolution uses `builtinSignatures(version)` (+ the
 *   extension tables when the shader enables the extension); builtin variable
 *   scoping uses `builtinVariables(version)` (+ `extensionVariables`);
 *   gl_Max* constant folding uses `builtinConstants(version)` (+
 *   `extensionConstants` merged on top when the matching extension is
 *   enabled — EXT_draw_buffers overrides the core gl_MaxDrawBuffers = 1).
 *   `builtinSignatures(100)` = builtinFunctions100 + relational100 (the ES
 *   1.00 relational int/bool variants kept out of builtinFunctions100 so
 *   300.ts's common100 superset does not duplicate its rel300/eq300 rows).
 * - codegen: call lowering for texture functions, derivatives, pack/unpack,
 *   bitfield ops, etc. uses the same tables via `matches()`.
 */
export type {
  BuiltinSignature,
  BuiltinVariable,
  BuiltinConstant,
  SigExtra,
} from './types.js';

export {
  matches,
  // type-literal helpers (shared by the table files)
  F,
  I,
  U,
  B,
  VOID,
  v,
  vec,
  ivec,
  uvec,
  bvec,
  V2,
  V3,
  V4,
  mat,
  smp,
  arr,
  genType,
  genIType,
  genUType,
  genBType,
  sig,
  gen,
  gen2,
  gen3,
} from './types.js';

export { builtinFunctions100, relational100, builtinVariables100, builtinConstants100 } from './100.js';
export { extensionFunctions, extensionVariables, extensionConstants } from './extensions.js';
export { builtinFunctions300, builtinVariables300, builtinConstants300 } from './300.js';

import type { BuiltinConstant, BuiltinSignature, BuiltinVariable } from './types.js';
import {
  builtinConstants100,
  builtinFunctions100,
  builtinVariables100,
  relational100,
} from './100.js';
import { builtinConstants300, builtinFunctions300, builtinVariables300 } from './300.js';

/**
 * The complete version-100 signature table: the shared float core
 * (builtinFunctions100 — also the base of 300.ts's superset) plus the ES 1.00
 * relational int/bool variants (relational100), which are NOT in
 * builtinFunctions100 so 300.ts's rel300/eq300 additions stay duplication-free.
 */
const builtinSignatures100: BuiltinSignature[] = [...builtinFunctions100, ...relational100];

/** Function signature table for a shader version (1.00 or 3.00 core only). */
export const builtinSignatures = (version: 100 | 300): BuiltinSignature[] =>
  version === 300 ? builtinFunctions300 : builtinSignatures100;

/** Builtin variable table for a shader version (1.00 or 3.00 core only). */
export const builtinVariables = (version: 100 | 300): BuiltinVariable[] =>
  version === 300 ? builtinVariables300 : builtinVariables100;

/** gl_Max* constant table for a shader version (1.00 or 3.00 core only). */
export const builtinConstants = (version: 100 | 300): BuiltinConstant[] =>
  version === 300 ? builtinConstants300 : builtinConstants100;
