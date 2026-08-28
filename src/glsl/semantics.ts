/**
 * semantics.ts — semantic analysis (the "S" in the GLSL ES 1.00 / 3.00
 * compile pipeline: preprocess → lexer → parse → SEMANTICS → link).
 *
 * Responsibilities (CORE, Phase 2a — the follow-up task adds declaration
 * qualifier/stage/precision rules + ShaderInfo assembly on top of this):
 * - symbol tables (`Scope`): ONE namespace for variables/functions/structs
 *   (GLSL ES §4.2), GLSL shadowing rules (same-scope redefinition is an
 *   error; CROSS-scope shadowing is legal — builtin variables/constants stay
 *   shadow-proof), builtin registration (functions, variables, gl_Max*
 *   constants — gated by the shader's enabled extensions).
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
 * here. Core entry point for a whole shader: `analyzeProgram(ast, ctx)`.
 * The full-shader entry `analyze(ast, opts)` (declaration rules + ShaderInfo
 * assembly + ShaderUses) lives in semantics-decl.ts and is re-exported here.
 */
import type {
  ExternalDecl, Expr, FunctionDefinition, FunctionPrototype, GlobalVarDecl,
  InterfaceBlockDecl, ParamDecl, StructDecl, StructDefinition, TranslationUnit,
  TypeName, TypeSpec, VarDeclarator,
} from './ast.js';
import type { GLSLType, Precision, SamplerKind, StorageClass, StructMember } from './types.js';
import { isFloat, typeEquals, typeName } from './types.js';
import { analyzeExpr, convertible } from './semantics-expr.js';
import { evalConstExpr, validateGlobalInit } from './semantics-const.js';
import { analyzeStatement } from './semantics-stmt.js';
import {
  builtinConstants, builtinSignatures, builtinVariables,
  extensionConstants, extensionFunctions, extensionVariables, matches,
} from './builtins/index.js';
import type { BuiltinSignature, BuiltinVariable } from './builtins/index.js';

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

  /**
   * Default precision per base-type key ('float' | 'int' | sampler kind),
   * GLSL ES §4.5.3. `precision` statements update it from their point onward
   * (last statement wins); the fragment float rule (§4.5.4 — every float
   * declaration needs an explicit precision or a default) is checked against
   * it. Function bodies see the state at their DEFINITION point (snapshot
   * taken by analyzeProgram's pre-pass).
   */
  defaultPrecisions: Map<string, Precision> = new Map();

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
    this.initDefaultPrecisions();
  }

  /**
   * Reset to the stage's PRE-DECLARED default precisions (§4.5.3): vertex
   * defaults float to highp, int to mediump and the sampler kinds to lowp
   * (1.00) / highp (3.00); fragment defaults ONLY int to mediump — float and
   * samplers have no fragment default (float declarations then require a
   * `precision` statement; sampler declarations are LENIENT by design — the
   * WebGL specs do not require sampler precision and real-world shaders omit
   * it).
   */
  initDefaultPrecisions(): void {
    const m = new Map<string, Precision>();
    if (this.stage === 'VERTEX') {
      m.set('float', 'highp');
      m.set('int', 'mediump');
      if (this.version === 100) {
        m.set('sampler2D', 'lowp');
        m.set('samplerCube', 'lowp');
      } else {
        for (const s of SAMPLER_300) m.set(s, 'highp');
      }
    } else {
      m.set('int', 'mediump');
    }
    this.defaultPrecisions = m;
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
  /**
   * const variables ONLY (never const-qualified params): the FULLY folded
   * initializer as flat components (column-major matrices, struct members
   * flattened in declaration order, arrays element-major, scalar = [v]).
   * Non-scalar consts have no scalar constValue — this array is their value.
   */
  constData?: (number | boolean)[];
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
  /**
   * user function SIGNATURES this function's body calls (recursion detection).
   * Edges are keyed by the RESOLVED callee FnSymbol, never the bare name: a
   * call to a DIFFERENT overload of the same name (`process(S1)` calling
   * `process(S2)`) is not a self-edge, while same-signature self-recursion and
   * mutual-recursion cycles still form cycles in this graph.
   */
  calls: Set<FnSymbol>;
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
 *   a name already declared in THIS scope is an error (same-scope
 *   redefinition — single namespace) and nothing is registered; a name that
 *   only exists in an ENCLOSING scope may be shadowed (CTS: shader-struct-
 *   scope, struct-nesting-of-variable-names, local-variable-shadowing-outer-
 *   function, conditional scoping, ogles build*). Builtin VARIABLES (gl_*)
 *   and gl_Max* constants stay shadow-proof from any scope. Returns false on
 *   failure.
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
   * Declare `sym`. Same-scope redefinition is an error (`'name' :
   * redefinition`) and the symbol is NOT registered; a name that exists only
   * in an ENCLOSING scope MAY be shadowed (cross-scope shadowing is legal
   * GLSL ES — CTS shader-struct-scope, struct-nesting-of-variable-names,
   * local-variable-shadowing-outer-function, in-parameter-passed-as-inout-
   * argument-and-global, ogles build CorrectFull_vert/CorrectPreprocess5),
   * with ONE exception: builtin VARIABLES (gl_*) and gl_Max* constants
   * (kind 'builtin-var'/'builtin-const') remain shadow-proof from any scope
   * (GLSL ES: builtin names cannot be redeclared). Function overloads are
   * handled by registerPrototype/registerDefinition (lookupLocal), never
   * here. Returns true on success.
   *
   * Exception (GLSL ES single namespace): a builtin FUNCTION name that no
   * user code has claimed yet (the pre-pass placeholder FnSymbol with no user
   * overloads attached) does not reserve the name in THIS scope — user
   * variables/structs may reuse it. The placeholder is replaced in this
   * scope's map (a later function decl with the same name then hits the
   * variable and errors 'redefinition', which is the correct GLSL behavior).
   * Once a user function overload claims the name, or the existing symbol is
   * a builtin VARIABLE (gl_*) or gl_Max* constant, the name is NOT free.
   */
  declare(sym: Symbol, ctx: SemContext, line: number): boolean {
    const existing = this.lookupLocal(sym.name);
    if (existing !== undefined && !isFreeBuiltinFnName(existing)) {
      ctx.error(line, `'${sym.name}' : redefinition`);
      return false;
    }
    // Cross-scope shadowing is legal for every symbol kind, except builtin
    // variables/constants (kept shadow-proof — see header comment).
    for (let s: Scope | null = this.parent; s !== null; s = s.parent) {
      const outer = s.lookupLocal(sym.name);
      if (outer !== undefined && (outer.kind === 'builtin-var' || outer.kind === 'builtin-const')) {
        ctx.error(line, `'${sym.name}' : redefinition`);
        return false;
      }
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

/**
 * True when `sym` is the builtin FUNCTION placeholder with NO user overloads
 * attached (its siblings are all builtin placeholders — i.e. just itself): a
 * user variable/struct may still claim the name (GLSL ES single namespace).
 * Once a user function overload exists, the name is claimed by user code →
 * a later var/struct decl with the same name is a redefinition.
 */
function isFreeBuiltinFnName(sym: Symbol): boolean {
  return sym.kind === 'fn' && sym.builtin && sym.siblings.every((s) => s.builtin);
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
    if (m.type.qualifiers.precision === undefined) checkFloatPrecision(ctx, m.loc.line, mt, m.name === '' ? null : m.name);
    // Member arrays (`float a[3]`): the dims are NOT part of the TypeSpec —
    // wrap here and cache the FULL member type on the TypeSpec so later
    // consumers (analyzeInterfaceBlock, codegen member walks) see it.
    const full = m.arrayDims.length > 0 ? wrapArrayDims(mt, m.arrayDims, scope, ctx, false, m.loc.line) : mt;
    m.type.resolved = full;
    members.push({ name: m.name, type: full });
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
 * GLSL ES §4.5.4: in FRAGMENT shaders, any float-typed declaration
 * (variable, struct/block member, parameter, function return) with NO
 * explicit precision qualifier requires a default float precision to have
 * been declared earlier in the shader; the error is reported at the
 * declaration's line. Vertex shaders default float to highp. Sampler
 * declarations are deliberately NOT checked (lenient — see
 * initDefaultPrecisions).
 */
function checkFloatPrecision(ctx: SemContext, line: number, t: GLSLType, name: string | null): void {
  if (ctx.stage !== 'FRAGMENT' || !isFloat(t)) return;
  if (ctx.defaultPrecisions.get('float') === undefined) {
    ctx.error(line, name === null ? 'No precision specified for (float)' : `'${name}' : No precision specified for (float)`);
  }
}

/**
 * Analyze an initializer + register each declarator as a VarSymbol.
 * Shared by global declarations and local decl-statements. Initializers are
 * analyzed as expressions and must convert (implicitly) to the declared
 * type; `const` variables require a CONSTANT initializer — `evalConstExpr`
 * (semantics-const.ts) folds the whole expression (scalar/vector/matrix
 * constructors, struct/array constant initializers, comma/ternary, binary
 * ops, member/index reads of consts); the folded flat components are stored
 * on the symbol as `constData` (scalars additionally mirror `constValue`).
 * Array dims are validated via wrapArrayDims.
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
    if (spec.qualifiers.precision === undefined) checkFloatPrecision(ctx, d.loc.line, type, d.name === '' ? null : d.name);
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
    let constData: (number | boolean)[] | undefined;
    if (spec.qualifiers.storage === 'const') {
      if (d.init === null) {
        ctx.error(d.loc.line, `'${d.name}' : const variable must be initialized`);
      } else {
        // evalConstExpr covers every constant-expression form: literals,
        // const reads, all constructor families (incl. the CTS boilerplate
        // `const vec4 green = vec4(0.0, 1.0, 0.0, 1.0);`), comma/ternary,
        // binary/unary ops and member/index reads of consts. Undefined =
        // a non-const leaf → not a constant expression.
        constData = evalConstExpr(d.init, scope, ctx);
        if (constData === undefined) {
          ctx.error(d.init.loc.line, `'${d.name}' : initializer of const variable must be a constant expression`);
        } else if (type.kind === 'scalar') {
          constValue = constData[0];
        }
      }
    } else if (scope.parent === null && d.init !== null) {
      // GLOBAL (non-const) initializers: ANGLE ValidateGlobalInitializer
      // parity (WebGL CTS global-variable-init.html) — uniforms and other
      // globals are allowed in WebGL1 (legacy compatibility), but texture
      // lookups, attributes/varyings/builtin non-constants, user function
      // calls and lvalue operations are compile errors. Const globals are
      // already covered by the evalConstExpr check above.
      validateGlobalInit(d.init, scope, ctx);
    }
    scope.declare({ kind: 'var', name: d.name, type, storage: spec.qualifiers.storage, constValue, constData }, ctx, d.loc.line);
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
  for (const v of builtinVariables(ctx.version)) {
    // Core-table entries may carry an `extension` tag (gl_DrawID →
    // GL_ANGLE_multi_draw); they are only visible when the shader enables
    // the extension, exactly like extensionVariables entries.
    if (v.extension !== undefined && !ctx.enabledExtensions.has(v.extension)) continue;
    vars.set(v.name, v);
  }
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
    if (p.type.qualifiers.precision === undefined) checkFloatPrecision(ctx, p.loc.line, t, p.name === '' ? null : p.name);
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

/**
 * The builtin signatures visible for `name` in this shader: the version's
 * core table plus extension-gated signatures whose extension is enabled —
 * the SAME sources the builtin pre-pass uses to reserve names.
 */
function enabledBuiltinSigs(name: string, ctx: SemContext): BuiltinSignature[] {
  const out = matches(name, builtinSignatures(ctx.version));
  for (const s of extensionFunctions) {
    if (s.name === name && s.extension !== undefined && ctx.enabledExtensions.has(s.extension)) out.push(s);
  }
  return out;
}

/**
 * Does the user function signature (name is already known to match) collide
 * with one builtin signature row? GLSL identifies a function by its name and
 * PARAMETER types — the return type cannot be overloaded, so a param-list
 * match is a same-signature redefinition regardless of the return type.
 */
function builtinSigExact(s: BuiltinSignature, params: ParamInfo[]): boolean {
  if (s.params.length !== params.length) return false;
  for (let i = 0; i < s.params.length; i++) {
    if (!sameParamType(s.params[i], params[i].type)) return false;
  }
  return true;
}

/** Register a function prototype (no body). Duplicate prototypes are OK; new
 * signatures become overloads; colliding builtin names error. */
function registerPrototype(d: FunctionPrototype, scope: Scope, ctx: SemContext): void {
  const retType = resolveTypeSpec(d.returnType, scope, ctx);
  if (retType === null) return;
  if (d.returnType.qualifiers.precision === undefined) checkFloatPrecision(ctx, d.loc.line, retType, d.name);
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
    // GLSL ES: a user function may reuse a builtin name when its signature
    // differs from EVERY visible builtin signature; a same-signature match is
    // a redefinition error. Compare against the builtin TABLES — never the
    // placeholder itself (void return, no params).
    if (enabledBuiltinSigs(d.name, ctx).some((s) => builtinSigExact(s, params))) {
      ctx.error(d.loc.line, `'${d.name}' : redefinition of built-in function`);
      return;
    }
  }
  for (const sig of existing.siblings) {
    if (sig.builtin) continue; // the placeholder carries no real signature
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
  if (proto.returnType.qualifiers.precision === undefined) checkFloatPrecision(ctx, d.loc.line, retType, proto.name);
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
    // User function with a builtin name: allowed only with a signature that
    // differs from every visible builtin signature (compare against the
    // builtin TABLES, never the void/[] placeholder itself).
    if (enabledBuiltinSigs(proto.name, ctx).some((s) => builtinSigExact(s, params))) {
      ctx.error(d.loc.line, `'${proto.name}' : redefinition of built-in function`);
      return null;
    }
  }
  for (const sig of existing.siblings) {
    if (sig.builtin) continue; // the placeholder carries no real signature
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
    if (mt !== null && mt.kind !== 'void') {
      if (m.type.qualifiers.precision === undefined) checkFloatPrecision(ctx, m.loc.line, mt, m.name === '' ? null : m.name);
      // Member arrays (same wrap as resolveStructDef — cache the full type).
      const full = m.arrayDims.length > 0 ? wrapArrayDims(mt, m.arrayDims, scope, ctx, false, m.loc.line) : mt;
      m.type.resolved = full;
      members.push({ name: m.name, type: full });
    }
  }
  const blockType: GLSLType = { kind: 'struct', name: d.blockName, members };
  scope.declare({ kind: 'struct', name: d.blockName, type: blockType }, ctx, d.loc.line);
  // The block instance (and bare members of instance-less blocks) carries the
  // block's OWN storage qualifier (GLSL ES 3.00 §4.3.9): uniform blocks →
  // 'uniform'; `out` (vertex) / `in` (fragment) varying blocks → 'out'/'in'.
  // varIsWritable then allows writes to vertex `out` block members and
  // rejects writes to fragment `in` (global input) and uniform block
  // members. Absent storage is treated as uniform (matches
  // analyzeInterfaceBlock's UBO path). Block members carry no storage of
  // their own (parser rejects member storage qualifiers); member-access
  // writability inherits from this instance symbol via analyzeMember.
  const storage = d.qualifiers.storage === undefined ? 'uniform' : d.qualifiers.storage;
  if (d.instanceName !== null && d.instanceName !== '') {
    const t = wrapArrayDims(blockType, d.arrayDims, scope, ctx, false, d.loc.line);
    scope.declare({ kind: 'var', name: d.instanceName, type: t, storage }, ctx, d.loc.line);
  } else {
    // Instance-less block: members are accessed by BARE name (GLSL ES 3.00
    // §4.3.7) — register them as global variables carrying the block's own
    // storage qualifier ('uniform' read-only for uniform blocks; 'out'/'in'
    // for instance-less varying blocks, which the linker supports). The
    // linker keeps UBO members OUT of the default block.
    for (const m of members) {
      scope.declare({ kind: 'var', name: m.name, type: m.type, storage }, ctx, d.loc.line);
    }
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
 * DFS over the user-function call graph. Nodes are the RESOLVED callee
 * signatures recorded during body analysis (`FnSymbol.calls`) — overloads are
 * DISTINCT nodes, so `process(S1)` calling `process(S2)` is legal (no cycle),
 * while a function signature reaching itself — directly (same-signature
 * self-call) or through a chain of other functions (mutual recursion, incl.
 * cycles across overloads) — is a cycle → one error per detected cycle:
 * "'name' : recursion is not allowed" (both versions).
 */
function detectRecursion(ctx: SemContext): void {
  const adj = new Map<FnSymbol, Set<FnSymbol>>();
  for (const f of ctx.userFns) {
    let set = adj.get(f);
    if (set === undefined) {
      set = new Set();
      adj.set(f, set);
    }
    for (const callee of f.calls) set.add(callee);
  }
  const colors = new Map<FnSymbol, number>(); // 0 white, 1 gray, 2 black
  const visit = (f: FnSymbol): boolean => {
    const c = colors.get(f) ?? 0;
    if (c === 1) {
      ctx.error(f.line, `'${f.name}' : recursion is not allowed`);
      return true;
    }
    if (c === 2) return false;
    colors.set(f, 1);
    const callees = adj.get(f);
    if (callees !== undefined) {
      for (const callee of callees) {
        if (visit(callee)) return true;
      }
    }
    colors.set(f, 2);
    return false;
  };
  for (const f of adj.keys()) {
    if (visit(f)) return; // one cycle error is enough
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
  // Precision-default state at each function definition site: function bodies
  // are analyzed in a second pass, so the snapshot taken HERE (in source
  // order with the precision statements) is restored per body.
  const defPrecisions = new Map<FunctionDefinition, Map<string, Precision>>();
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
        defPrecisions.set(d, new Map(ctx.defaultPrecisions));
        break;
      }
      case 'interface-block':
        registerInterfaceBlock(d, global, ctx);
        break;
      case 'precision-decl':
        // Default precision statements take effect from their point onward.
        ctx.defaultPrecisions.set(d.base, d.precision);
        break;
      default:
        break; // extension-decl / invariant-decl: no scope effect
    }
  }

  for (const d of ast.declarations) {
    if (d.kind !== 'function-definition') continue;
    const sig = defToSig.get(d);
    if (sig !== null && sig !== undefined) {
      const saved = ctx.defaultPrecisions;
      const snap = defPrecisions.get(d);
      if (snap !== undefined) ctx.defaultPrecisions = snap;
      analyzeFunctionBody(d, sig, global, ctx);
      ctx.defaultPrecisions = saved;
    }
  }

  detectRecursion(ctx);
}

/** Re-export helper needed by statement/declaration analysis (and follow-up). */
export { typeEquals as _typeEquals };

/**
 * The full-shader entry point (the public contract compileShader wires):
 * runs the CORE pass (analyzeProgram) then the declaration rules + ShaderInfo
 * assembly + ShaderUses scan (semantics-decl.ts). Returns the resolved
 * declaration summaries on success or the collected 1-based errors.
 */
export { analyze } from './semantics-decl.js';
