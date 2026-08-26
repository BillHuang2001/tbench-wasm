/**
 * semantics.ts — semantic analysis (the "S" in the GLSL ES 1.00 / 3.00
 * compile pipeline: preprocess → lexer → parse → SEMANTICS → link).
 *
 * Responsibilities (CORE, Phase 2a — the follow-up task adds declaration
 * qualifier/stage/precision rules + ShaderInfo assembly on top of this):
 * - symbol tables (`Scope`): ONE namespace for variables/functions/structs
 *   (GLSL ES §4.2), GLSL shadowing rules (declaring a name that exists in ANY
 *   enclosing scope is an error), builtin registration (functions, variables,
 *   gl_Max* constants — gated by the shader's enabled extensions).
 * - a global PRE-PASS over `TranslationUnit.declarations` registering
 *   structs, global variables and function signatures (with overloads) in
 *   source order, then per-function BODY analysis (params + statements) and
 *   RECURSION DETECTION over the user-function call graph.
 * - type-spec resolution (`resolveTypeSpec`): builtin types per version,
 *   user structs, inline struct definitions, array dims — shared by globals,
 *   locals and params.
 *
 * Expression analysis lives in semantics-expr.ts (`analyzeExpr`), statement
 * analysis in semantics-stmt.ts (`analyzeStatement`); both are re-exported
 * here. Entry point for a whole shader: `analyzeProgram(ast, ctx)`.
 */
import type {
  ExternalDecl, Expr, FunctionDefinition, FunctionPrototype, GlobalVarDecl,
  InterfaceBlockDecl, ParamDecl, StructDecl, StructDefinition, TranslationUnit,
  TypeName, TypeSpec, VarDeclarator,
} from './ast.js';
import type { GLSLType, SamplerKind, StorageClass, StructMember } from './types.js';
import { typeEquals, typeName } from './types.js';
import { analyzeExpr, convertible } from './semantics-expr.js';
import { analyzeStatement } from './semantics-stmt.js';
import {
  builtinConstants, builtinSignatures, builtinVariables,
  extensionConstants, extensionFunctions, extensionVariables,
} from './builtins/index.js';
import type { BuiltinVariable } from './builtins/index.js';

export { analyzeExpr, analyzeStatement, convertible };

/* ------------------------------------------------------------------ */
/* Errors & analysis context                                           */
/* ------------------------------------------------------------------ */

/** One semantic error: 1-based source line + Khronos-style message. */
export interface SemError {
  line: number;
  message: string;
}

/** Maximum number of errors collected (parser parity: MAX_ERRORS = 20). */
const MAX_ERRORS = 20;

/**
 * Per-shader analysis context. Create one per shader and pass it to
 * `analyzeProgram`; `errors` is the collected error list (capped at 20).
 *
 * The `loopDepth`/`breakableDepth`/`switchDepth`/`currentFunction`/`userFns`
 * fields are INTERNAL analysis state managed by the semantics pass — reset
 * per shader, do not touch from outside.
 */
export class SemContext {
  readonly version: 100 | 300;
  readonly stage: 'VERTEX' | 'FRAGMENT';
  readonly enabledExtensions: Set<string>;
  errors: SemError[] = [];

  /** Internal: nested loop depth (continue legality). */
  loopDepth = 0;
  /** Internal: nested loop+switch depth (break legality). */
  breakableDepth = 0;
  /** Internal: nested switch depth (case-label legality). */
  switchDepth = 0;
  /** Internal: the FnSymbol whose body is being analyzed (call-graph edges). */
  currentFunction: FnSymbol | null = null;
  /** Internal: every user function signature (recursion detection). */
  userFns: FnSymbol[] = [];

  constructor(version: 100 | 300, stage: 'VERTEX' | 'FRAGMENT', enabledExtensions: Set<string>) {
    this.version = version;
    this.stage = stage;
    this.enabledExtensions = enabledExtensions;
  }

  /** Record an error (1-based line); stops collecting at MAX_ERRORS. */
  error(line: number, message: string): void {
    if (this.errors.length >= MAX_ERRORS) return;
    this.errors.push({ line, message });
  }
}

/* ------------------------------------------------------------------ */
/* Symbols                                                             */
/* ------------------------------------------------------------------ */

export type SymbolKind = 'var' | 'fn' | 'struct' | 'builtin-var' | 'builtin-const';

export interface BaseSymbol {
  kind: SymbolKind;
  name: string;
}

/** A user variable: global, local or function parameter. */
export interface VarSymbol extends BaseSymbol {
  kind: 'var';
  type: GLSLType;
  storage?: StorageClass;
  /** const globals/locals with constant initializers: the folded value. */
  constValue?: number | boolean;
}

/** One function parameter (name may be '' for unnamed params). */
export interface ParamInfo {
  name: string;
  type: GLSLType;
  /** in/out/inout/const param qualifiers (inout → by-reference marshaling). */
  storage?: StorageClass;
}

/**
 * A function signature. User overloads share one name via `siblings`
 * (the scope maps the name to the FIRST signature; siblings[0] === self).
 * `builtin` entries are placeholders (signature tables drive builtin
 * overload resolution) used only to make builtin names unreservable.
 */
export interface FnSymbol extends BaseSymbol {
  kind: 'fn';
  retType: GLSLType;
  params: ParamInfo[];
  /** true once a definition exists (prototypes leave it false). */
  defined: boolean;
  /** declaration line (definition line when defined). */
  line: number;
  builtin: boolean;
  /** user function NAMES this function's body calls (recursion detection). */
  calls: Set<string>;
  /** all signatures of this name (overloads); includes self. */
  siblings: FnSymbol[];
}

/** A struct type declaration (global or local). */
export interface StructSymbol extends BaseSymbol {
  kind: 'struct';
  type: GLSLType; // { kind: 'struct', name, members }
}

/** A builtin variable (gl_Position, gl_FragColor, gl_FragData, ...). */
export interface BuiltinVarSymbol extends BaseSymbol {
  kind: 'builtin-var';
  type: GLSLType;
  writable: boolean;
  stage: 'VERTEX' | 'FRAGMENT' | 'BOTH';
}

/** A gl_Max* constant (folded to its integer value at use sites). */
export interface BuiltinConstSymbol extends BaseSymbol {
  kind: 'builtin-const';
  type: GLSLType;
  value: number;
}

export type Symbol = VarSymbol | FnSymbol | StructSymbol | BuiltinVarSymbol | BuiltinConstSymbol;

/* ------------------------------------------------------------------ */
/* Scope                                                               */
/* ------------------------------------------------------------------ */

/**
 * A lexical scope: a symbol table with a parent chain (global scope has
 * parent null).
 *
 * API (used by semantics-expr/semantics-stmt and the follow-up task):
 * - `new Scope(parent | null)` — a scope with an optional enclosing scope.
 * - `push(): Scope` — create and return a child scope (block/for/function).
 * - `pop(): Scope | null` — return the parent (children are also dropped by
 *   simply discarding them; pop is provided for symmetry).
 * - `lookup(name): Symbol | undefined` — nearest declaration, walking up.
 * - `lookupLocal(name): Symbol | undefined` — this scope only.
 * - `declare(sym, ctx, line): boolean` — register with GLSL shadowing rules:
 *   a name already declared in THIS scope or ANY enclosing scope is an error
 *   (GLSL ES forbids shadowing — single namespace) and nothing is
 *   registered. Returns false on failure.
 * - `forceDeclare(sym)` — register without checks (builtin pre-pass).
 */
export class Scope {
  readonly parent: Scope | null;
  private readonly symbols = new Map<string, Symbol>();

  constructor(parent: Scope | null = null) {
    this.parent = parent;
  }

  /** Nearest declaration of `name` walking up the scope chain. */
  lookup(name: string): Symbol | undefined {
    for (let s: Scope | null = this; s !== null; s = s.parent) {
      const sym = s.symbols.get(name);
      if (sym !== undefined) return sym;
    }
    return undefined;
  }

  /** Declaration of `name` in THIS scope only. */
  lookupLocal(name: string): Symbol | undefined {
    return this.symbols.get(name);
  }

  /**
   * Declare `sym`. GLSL ES forbids shadowing: if the name exists in this
   * scope or any enclosing scope → error `'name' : redefinition` and the
   * symbol is NOT registered. Returns true on success.
   */
  declare(sym: Symbol, ctx: SemContext, line: number): boolean {
    if (this.lookup(sym.name) !== undefined) {
      ctx.error(line, `'${sym.name}' : redefinition`);
      return false;
    }
    this.symbols.set(sym.name, sym);
    return true;
  }

  /** Register without shadowing checks (builtin pre-pass only). */
  forceDeclare(sym: Symbol): void {
    this.symbols.set(sym.name, sym);
  }

  /** Create and return a child scope. */
  push(): Scope {
    return new Scope(this);
  }

  /** Return the parent scope (null for the global scope). */
  pop(): Scope | null {
    return this.parent;
  }
}

/* ------------------------------------------------------------------ */
/* Builtin type-name resolution (version-dependent)                    */
/* ------------------------------------------------------------------ */

const SAMPLER_100: ReadonlySet<string> = new Set(['sampler2D', 'samplerCube']);
const SAMPLER_300: ReadonlySet<string> = new Set([
  'sampler2D', 'samplerCube', 'sampler3D', 'sampler2DArray', 'sampler2DShadow',
  'samplerCubeShadow', 'sampler2DArrayShadow', 'isampler2D', 'isampler3D',
  'isamplerCube', 'isampler2DArray', 'usampler2D', 'usampler3D', 'usamplerCube',
  'usampler2DArray',
]);

/**
 * The GLSLType of a builtin TYPE NAME in `version`, or undefined when the
 * name is not a builtin type there (e.g. `uint`/`mat2x3`/`sampler3D` in a
 * 100 shader lex as plain identifiers — the CTS non-reserved-words tests
 * require them to be usable as user identifiers).
 */
export function builtinType(name: string, version: 100 | 300): GLSLType | undefined {
  switch (name) {
    case 'void':
      return { kind: 'void' };
    case 'float':
    case 'int':
    case 'bool':
      return { kind: 'scalar', base: name };
    case 'uint':
      return version === 300 ? { kind: 'scalar', base: 'uint' } : undefined;
  }
  const vm = /^(b|i|u)?vec([234])$/.exec(name);
  if (vm) {
    const size = Number(vm[2]) as 2 | 3 | 4;
    const prefix = vm[1] ?? '';
    if (prefix === 'u' && version === 100) return undefined;
    const base = prefix === 'i' ? 'int' : prefix === 'u' ? 'uint' : prefix === 'b' ? 'bool' : 'float';
    return { kind: 'vector', base, size };
  }
  const mm = /^mat([234])(?:x([234]))?$/.exec(name);
  if (mm) {
    const cols = Number(mm[1]) as 2 | 3 | 4;
    const rows = mm[2] ? (Number(mm[2]) as 2 | 3 | 4) : cols;
    if (version === 100) {
      // ES 1.00: only square mat2/mat3/mat4 are keywords; mat2x2 etc. lex as
      // identifiers there and must be usable as user names.
      if (mm[2] !== undefined) return undefined;
      return { kind: 'matrix', cols, rows };
    }
    return { kind: 'matrix', cols, rows };
  }
  if (version === 100 ? SAMPLER_100.has(name) : SAMPLER_300.has(name)) {
    return { kind: 'sampler', sampler: name as SamplerKind };
  }
  return undefined;
}

/* ------------------------------------------------------------------ */
/* Type-spec resolution                                                */
/* ------------------------------------------------------------------ */

/**
 * Resolve a declared type: qualifiers are ignored here (the follow-up's
 * declaration rules validate them); the BASE type (with struct members
 * resolved) is returned and stored in `spec.resolved`. Array dims are NOT
 * part of the TypeSpec — declarators/params wrap dims separately via
 * `wrapArrayDims`. Returns null (with an error recorded) on failure.
 */
export function resolveTypeSpec(spec: TypeSpec, scope: Scope, ctx: SemContext): GLSLType | null {
  if (spec.resolved !== undefined) return spec.resolved;
  let t: GLSLType | null;
  if (spec.base.kind === 'type-name') {
    t = resolveTypeName(spec.base, scope, ctx);
  } else {
    t = resolveStructDef(spec.base, scope, ctx);
  }
  if (t !== null) spec.resolved = t;
  return t;
}

/** Resolve a bare type name: builtin type or user struct (single namespace). */
function resolveTypeName(name: TypeName, scope: Scope, ctx: SemContext): GLSLType | null {
  const builtin = builtinType(name.name, ctx.version);
  if (builtin !== undefined) return builtin;
  const sym = scope.lookup(name.name);
  if (sym === undefined) {
    ctx.error(name.loc.line, `'${name.name}' : undeclared identifier`);
    return null;
  }
  if (sym.kind === 'struct') return sym.type;
  ctx.error(name.loc.line, `'${name.name}' : not a type`);
  return null;
}

/**
 * Resolve an inline struct definition into a struct type (members resolved
 * recursively). Does NOT register the name — callers do. Recursive
 * self-reference is an error.
 */
function resolveStructDef(def: StructDefinition, scope: Scope, ctx: SemContext): GLSLType | null {
  if (def.name === null) return null; // parser already errored (anonymous struct)
  const members: StructMember[] = [];
  for (const m of def.members) {
    const mt = resolveTypeSpec(m.type, scope, ctx);
    if (mt === null) continue;
    if (mt.kind === 'void') {
      ctx.error(m.loc.line, "'void' : invalid struct member type");
      continue;
    }
    if (mt.kind === 'struct' && mt.name === def.name) {
      ctx.error(m.loc.line, `'${def.name}' : recursive struct definition`);
      continue;
    }
    members.push({ name: m.name, type: mt });
  }
  return { kind: 'struct', name: def.name, members };
}

/**
 * Wrap a base type with array dims. `dims[0]` is the OUTERMOST dimension
 * (GLSL: `a[2][3]` = array of 2 arrays of 3). Entries may be null (unsized
 * `[]`), which is legal only for function parameters (`allowUnsized`).
 * Dim expressions are analyzed (annotated) and must fold to a positive
 * integer constant; failures are reported and size 1 is used for recovery.
 */
export function wrapArrayDims(
  base: GLSLType,
  dims: Expr[],
  scope: Scope,
  ctx: SemContext,
  allowUnsized: boolean,
  line: number,
): GLSLType {
  let t = base;
  for (let i = dims.length - 1; i >= 0; i--) {
    const dim = dims[i] as Expr | null;
    if (dim === null) {
      if (!allowUnsized) {
        ctx.error(line, 'unsized array declarations are only allowed for function parameters');
        t = { kind: 'array', element: t, size: 1 };
      } else {
        t = { kind: 'array', element: t, size: null };
      }
      continue;
    }
    analyzeExpr(dim, scope, ctx);
    const v = dim.constValue;
    if (typeof v !== 'number' || !Number.isInteger(v)) {
      ctx.error(dim.loc.line, 'array size must be a constant integer expression');
      t = { kind: 'array', element: t, size: 1 };
    } else if (v <= 0) {
      ctx.error(dim.loc.line, 'array size must be greater than zero');
      t = { kind: 'array', element: t, size: 1 };
    } else {
      t = { kind: 'array', element: t, size: v };
    }
  }
  return t;
}

/* ------------------------------------------------------------------ */
/* Declaration analysis (globals + locals, shared)                     */
/* ------------------------------------------------------------------ */

/**
 * Analyze an initializer + register each declarator as a VarSymbol.
 * Shared by global declarations and local decl-statements. Initializers are
 * analyzed as expressions and must convert (implicitly) to the declared
 * type; `const` variables require a constant initializer (scalar/vector/
 * matrix: a folded constValue; array/struct: a constructor call). Array dims
 * are validated via wrapArrayDims.
 */
export function declareVariables(
  baseType: GLSLType | null,
  spec: TypeSpec,
  declarators: VarDeclarator[],
  scope: Scope,
  ctx: SemContext,
  allowUnsized: boolean,
): void {
  if (baseType === null) return;
  for (const d of declarators) {
    if (d.name === '') continue; // parser error-recovery placeholder
    const type = wrapArrayDims(baseType, d.arrayDims, scope, ctx, allowUnsized, d.loc.line);
    if (type.kind === 'void') {
      ctx.error(d.loc.line, "'void' : cannot declare a variable of type void");
      continue;
    }
    let constValue: number | boolean | undefined;
    if (d.init !== null) {
      analyzeExpr(d.init, scope, ctx);
      const it = d.init.resolvedType;
      if (it !== undefined && !convertible(it, type, ctx.version)) {
        ctx.error(d.init.loc.line, `cannot convert from '${typeName(it)}' to '${typeName(type)}'`);
      }
    }
    if (spec.qualifiers.storage === 'const') {
      if (d.init === null) {
        ctx.error(d.loc.line, `'${d.name}' : const variable must be initialized`);
      } else if (type.kind === 'array' || type.kind === 'struct') {
        // const arrays/structs: accept constructor-call initializers (the
        // scalar constValue model cannot represent aggregates).
        if (d.init.kind !== 'call') {
          ctx.error(d.init.loc.line, `'${d.name}' : initializer of const variable must be a constant expression`);
        }
      } else if (d.init.constValue === undefined) {
        ctx.error(d.init.loc.line, `'${d.name}' : initializer of const variable must be a constant expression`);
      } else {
        constValue = d.init.constValue;
      }
    }
    scope.declare({ kind: 'var', name: d.name, type, storage: spec.qualifiers.storage, constValue }, ctx, d.loc.line);
  }
}

/* ------------------------------------------------------------------ */
/* Global pre-pass                                                     */
/* ------------------------------------------------------------------ */

/** Register builtin functions/variables/constants into the GLOBAL scope.
 * Registered FIRST so any user declaration with a builtin name errors
 * (GLSL: builtin names cannot be redeclared). Extension-gated entries are
 * merged in when the shader enables the extension (EXT_draw_buffers
 * overrides gl_FragData's size and gl_MaxDrawBuffers, EXT_frag_depth adds
 * gl_FragDepthEXT). */
function registerBuiltins(scope: Scope, ctx: SemContext): void {
  const vars = new Map<string, BuiltinVariable>();
  for (const v of builtinVariables(ctx.version)) vars.set(v.name, v);
  for (const v of extensionVariables) {
    if (v.extension !== undefined && ctx.enabledExtensions.has(v.extension)) vars.set(v.name, v);
  }
  for (const v of vars.values()) {
    scope.forceDeclare({ kind: 'builtin-var', name: v.name, type: v.type, writable: v.writable, stage: v.stage });
  }

  const consts = new Map<string, number>();
  for (const c of builtinConstants(ctx.version)) consts.set(c.name, c.value);
  for (const c of extensionConstants) {
    if (c.extension !== undefined && ctx.enabledExtensions.has(c.extension)) consts.set(c.name, c.value);
  }
  for (const [name, value] of consts) {
    scope.forceDeclare({ kind: 'builtin-const', name, type: { kind: 'scalar', base: 'int' }, value });
  }

  // Function names are reserved: register a placeholder FnSymbol per name so
  // user declarations with those names fail the shadowing check. Overload
  // RESOLUTION for builtins uses the signature tables, not these placeholders.
  const fnames = new Set<string>();
  for (const s of builtinSignatures(ctx.version)) fnames.add(s.name);
  for (const s of extensionFunctions) {
    if (s.extension !== undefined && ctx.enabledExtensions.has(s.extension)) fnames.add(s.name);
  }
  for (const name of fnames) {
    const sym = makeFnSymbol(name, { kind: 'void' }, [], true, 0);
    sym.builtin = true;
    scope.forceDeclare(sym);
  }
}

/** Build a fresh user-function signature symbol (siblings = [self]). */
function makeFnSymbol(name: string, retType: GLSLType, params: ParamInfo[], defined: boolean, line: number): FnSymbol {
  const sym: FnSymbol = {
    kind: 'fn', name, retType, params, defined, line, builtin: false,
    calls: new Set(), siblings: [],
  };
  sym.siblings = [sym];
  return sym;
}

/** Register a bare `struct S {...};` declaration. */
function registerStructDecl(d: StructDecl, scope: Scope, ctx: SemContext): void {
  const t = resolveStructDef({ kind: 'struct-definition', name: d.name, members: d.members, loc: d.loc }, scope, ctx);
  if (t === null || t.kind !== 'struct') return;
  scope.declare({ kind: 'struct', name: t.name, type: t }, ctx, d.loc.line);
}

/** Analyze a global variable declaration (incl. inline struct definitions). */
function analyzeGlobalVarDecl(d: GlobalVarDecl, scope: Scope, ctx: SemContext): void {
  const baseType = resolveTypeSpec(d.type, scope, ctx);
  if (baseType === null) return;
  if (d.type.base.kind === 'struct-definition' && baseType.kind === 'struct') {
    // `struct S {...} s;` — register the struct before the variables.
    scope.declare({ kind: 'struct', name: baseType.name, type: baseType }, ctx, d.loc.line);
  }
  declareVariables(baseType, d.type, d.declarators, scope, ctx, false);
}

/** Resolve a param list into ParamInfo[] (unsized array dims allowed). */
function resolveParams(params: ParamDecl[], scope: Scope, ctx: SemContext): ParamInfo[] {
  const out: ParamInfo[] = [];
  const placeholder: GLSLType = { kind: 'scalar', base: 'float' };
  for (const p of params) {
    const base = resolveTypeSpec(p.type, scope, ctx);
    if (base === null) {
      out.push({ name: p.name, type: placeholder });
      continue;
    }
    if (base.kind === 'void') {
      ctx.error(p.loc.line, "'void' : invalid parameter type");
      out.push({ name: p.name, type: placeholder });
      continue;
    }
    const t = wrapArrayDims(base, p.arrayDims, scope, ctx, true, p.loc.line);
    out.push({ name: p.name, type: t, storage: p.type.qualifiers.storage });
  }
  return out;
}

/** Exact signature match between a registered FnSymbol and a new signature. */
function sameSignature(a: FnSymbol, retType: GLSLType, params: ParamInfo[]): boolean {
  if (!typeEquals(a.retType, retType)) return false;
  if (a.params.length !== params.length) return false;
  for (let i = 0; i < a.params.length; i++) {
    if (!sameParamType(a.params[i].type, params[i].type)) return false;
  }
  return true;
}

/** Param-type equality; unsized array params match any size of the same element. */
function sameParamType(a: GLSLType, b: GLSLType): boolean {
  if (typeEquals(a, b)) return true;
  if (a.kind === 'array' && b.kind === 'array' && (a.size === null || b.size === null)) {
    return typeEquals(a.element, b.element);
  }
  return false;
}

/** Register a function prototype (no body). Duplicate prototypes are OK; new
 * signatures become overloads; colliding builtin names error. */
function registerPrototype(d: FunctionPrototype, scope: Scope, ctx: SemContext): void {
  const retType = resolveTypeSpec(d.returnType, scope, ctx);
  if (retType === null) return;
  const params = resolveParams(d.params, scope, ctx);
  const existing = scope.lookupLocal(d.name);
  if (existing === undefined) {
    const sym = makeFnSymbol(d.name, retType, params, false, d.loc.line);
    scope.forceDeclare(sym);
    ctx.userFns.push(sym);
    return;
  }
  if (existing.kind !== 'fn') {
    ctx.error(d.loc.line, `'${d.name}' : redefinition`);
    return;
  }
  if (existing.builtin) {
    ctx.error(d.loc.line, `'${d.name}' : redefinition of built-in function`);
    return;
  }
  for (const sig of existing.siblings) {
    if (sameSignature(sig, retType, params)) return; // repeated prototype: OK
  }
  const sym = makeFnSymbol(d.name, retType, params, false, d.loc.line);
  existing.siblings.push(sym);
  ctx.userFns.push(sym);
}

/**
 * Register a function definition: must match a prior prototype exactly or
 * stand alone; duplicate definitions error; new signatures become overloads.
 * Returns the signature symbol (for body analysis).
 */
function registerDefinition(d: FunctionDefinition, scope: Scope, ctx: SemContext): FnSymbol | null {
  const proto = d.prototype;
  const retType = resolveTypeSpec(proto.returnType, scope, ctx);
  if (retType === null) return null;
  const params = resolveParams(proto.params, scope, ctx);
  const existing = scope.lookupLocal(proto.name);
  if (existing === undefined) {
    const sym = makeFnSymbol(proto.name, retType, params, true, d.loc.line);
    scope.forceDeclare(sym);
    ctx.userFns.push(sym);
    return sym;
  }
  if (existing.kind !== 'fn') {
    ctx.error(d.loc.line, `'${proto.name}' : redefinition`);
    return null;
  }
  if (existing.builtin) {
    ctx.error(d.loc.line, `'${proto.name}' : redefinition of built-in function`);
    return null;
  }
  for (const sig of existing.siblings) {
    if (sameSignature(sig, retType, params)) {
      if (sig.defined) {
        ctx.error(d.loc.line, `'${proto.name}' : redefinition of function`);
        return null;
      }
      sig.defined = true;
      sig.line = d.loc.line;
      sig.params = params; // adopt the definition's param names
      return sig;
    }
  }
  const sym = makeFnSymbol(proto.name, retType, params, true, d.loc.line);
  existing.siblings.push(sym);
  ctx.userFns.push(sym);
  return sym;
}

/** Register an interface block minimally: block name → struct type, instance
 * name → variable of that (possibly array) type. Member-via-instance access
 * then resolves through the struct type. (Full UBO rules are the follow-up's
 * declaration work.) */
function registerInterfaceBlock(d: InterfaceBlockDecl, scope: Scope, ctx: SemContext): void {
  const members: StructMember[] = [];
  for (const m of d.members) {
    const mt = resolveTypeSpec(m.type, scope, ctx);
    if (mt !== null && mt.kind !== 'void') members.push({ name: m.name, type: mt });
  }
  const blockType: GLSLType = { kind: 'struct', name: d.blockName, members };
  scope.declare({ kind: 'struct', name: d.blockName, type: blockType }, ctx, d.loc.line);
  if (d.instanceName !== null && d.instanceName !== '') {
    const t = wrapArrayDims(blockType, d.arrayDims, scope, ctx, false, d.loc.line);
    scope.declare({ kind: 'var', name: d.instanceName, type: t, storage: 'uniform' }, ctx, d.loc.line);
  }
}

/* ------------------------------------------------------------------ */
/* Function bodies                                                     */
/* ------------------------------------------------------------------ */

/** Analyze one function body: params + statements in a fresh function scope. */
function analyzeFunctionBody(d: FunctionDefinition, sig: FnSymbol, global: Scope, ctx: SemContext): void {
  const fnScope = global.push();
  const savedFn = ctx.currentFunction;
  const savedLoop = ctx.loopDepth;
  const savedBreak = ctx.breakableDepth;
  const savedSwitch = ctx.switchDepth;
  ctx.currentFunction = sig;
  ctx.loopDepth = 0;
  ctx.breakableDepth = 0;
  ctx.switchDepth = 0;
  for (const p of sig.params) {
    if (p.name === '') continue; // unnamed param: no symbol
    fnScope.declare({ kind: 'var', name: p.name, type: p.type, storage: p.storage }, ctx, d.loc.line);
  }
  analyzeStatement(d.body, fnScope, ctx, { returnType: sig.retType });
  ctx.currentFunction = savedFn;
  ctx.loopDepth = savedLoop;
  ctx.breakableDepth = savedBreak;
  ctx.switchDepth = savedSwitch;
}

/* ------------------------------------------------------------------ */
/* Recursion detection                                                 */
/* ------------------------------------------------------------------ */

/**
 * DFS over the user-function call graph (edges recorded by name during body
 * analysis). Direct and mutual recursion → one error per detected cycle:
 * "'name' : recursion is not allowed" (both versions).
 */
function detectRecursion(ctx: SemContext): void {
  const adj = new Map<string, Set<string>>();
  const lines = new Map<string, number>();
  for (const f of ctx.userFns) {
    if (!adj.has(f.name)) {
      adj.set(f.name, new Set());
      lines.set(f.name, f.line);
    }
    for (const callee of f.calls) adj.get(f.name)!.add(callee);
  }
  const colors = new Map<string, number>(); // 0 white, 1 gray, 2 black
  const visit = (name: string): boolean => {
    const c = colors.get(name) ?? 0;
    if (c === 1) {
      ctx.error(lines.get(name) ?? 1, `'${name}' : recursion is not allowed`);
      return true;
    }
    if (c === 2) return false;
    colors.set(name, 1);
    const callees = adj.get(name);
    if (callees !== undefined) {
      for (const callee of callees) {
        if (visit(callee)) return true;
      }
    }
    colors.set(name, 2);
    return false;
  };
  for (const name of adj.keys()) {
    if (visit(name)) return; // one cycle error is enough
  }
}

/* ------------------------------------------------------------------ */
/* Entry point                                                         */
/* ------------------------------------------------------------------ */

/**
 * Run the CORE semantic analysis over a parsed shader: register builtins,
 * pre-pass all global declarations (structs, variables, function signatures,
 * interface blocks), analyze every function body, and detect recursion.
 * Errors are collected on `ctx`; expression/statement nodes are annotated in
 * place (resolvedType/constValue/lvalue; TypeSpec.resolved).
 */
export function analyzeProgram(ast: TranslationUnit, ctx: SemContext): void {
  const global = new Scope(null);
  registerBuiltins(global, ctx);

  const defToSig = new Map<FunctionDefinition, FnSymbol>();
  for (const d of ast.declarations) {
    switch (d.kind) {
      case 'struct-decl':
        registerStructDecl(d, global, ctx);
        break;
      case 'global-var-decl':
        analyzeGlobalVarDecl(d, global, ctx);
        break;
      case 'function-prototype':
        registerPrototype(d, global, ctx);
        break;
      case 'function-definition': {
        const sig = registerDefinition(d, global, ctx);
        if (sig !== null) defToSig.set(d, sig);
        break;
      }
      case 'interface-block':
        registerInterfaceBlock(d, global, ctx);
        break;
      default:
        break; // precision-decl / extension-decl / invariant-decl: follow-up rules
    }
  }

  for (const d of ast.declarations) {
    if (d.kind !== 'function-definition') continue;
    const sig = defToSig.get(d);
    if (sig !== null && sig !== undefined) analyzeFunctionBody(d, sig, global, ctx);
  }

  detectRecursion(ctx);
}

/** Re-export helper needed by statement/declaration analysis (and follow-up). */
export { typeEquals as _typeEquals };
