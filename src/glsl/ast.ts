/**
 * ast.ts — Abstract syntax tree definitions for GLSL ES 1.00 and 3.00.
 *
 * Type-only module (no logic): the parser produces these nodes, the semantics
 * pass annotates them in place (`resolvedType` on expressions, `resolved` on
 * type specs), and the linker walks the annotated tree to build the Program.
 *
 * Version notes baked into the grammar:
 * - `switch`/`case`/`default`, bitwise operators, `uint`, non-square matrices
 *   and `flat`/`centroid`/`layout(...)` are GLSL ES 3.00 only.
 * - `attribute`/`varying` are GLSL ES 1.00 only; `^^` (logical xor) is 1.00
 *   only and was removed in 3.00.
 * The parser enforces these via the version passed to it.
 */
import type { GLSLType, Precision, SamplerKind, TypeQualifiers } from './types.js';

/** Source position: 1-based line, 0-based column (compile errors use line). */
export interface Loc {
  line: number;
  column: number;
}

export interface Node {
  loc: Loc;
}

/* ------------------------------------------------------------------ */
/* Translation unit                                                    */
/* ------------------------------------------------------------------ */

export interface TranslationUnit extends Node {
  kind: 'translation-unit';
  /** Declared `#version` (default 100 when no directive present). */
  version: 100 | 300;
  declarations: ExternalDecl[];
}

export type ExternalDecl =
  | FunctionDefinition
  | FunctionPrototype
  | GlobalVarDecl
  | InterfaceBlockDecl
  | StructDecl
  | PrecisionDecl
  | ExtensionDecl
  | InvariantDecl;

/* ------------------------------------------------------------------ */
/* Type specs                                                          */
/* ------------------------------------------------------------------ */

/**
 * A declared type: qualifiers + base type (+ optional inline struct
 * definition). `resolved` is filled by semantics with the final GLSLType.
 */
export interface TypeSpec extends Node {
  kind: 'type-spec';
  qualifiers: TypeQualifiers;
  base: TypeName | StructDefinition;
  /** Filled by semantics: the resolved GLSL type. */
  resolved?: GLSLType;
}

export interface TypeName extends Node {
  kind: 'type-name';
  name: string;
}

/** Inline struct definition (used by TypeSpec.base and bare `struct S {...}`). */
export interface StructDefinition extends Node {
  kind: 'struct-definition';
  name: string | null;
  members: StructMemberDecl[];
}

export interface StructMemberDecl extends Node {
  kind: 'struct-member';
  name: string;
  type: TypeSpec;
  /** Array dimensions (struct/interface-block member arrays); `[null]` = unsized. */
  arrayDims: Expr[];
}

/* ------------------------------------------------------------------ */
/* Declarations                                                        */
/* ------------------------------------------------------------------ */

export interface VarDeclarator extends Node {
  kind: 'var-declarator';
  name: string;
  /** Array dimensions; `[null]` = unsized (function parameters, ES 3.00). */
  arrayDims: Expr[];
  init: Expr | null;
}

/** Global variable declaration (`attribute/varying/uniform/const/in/out ...`). */
export interface GlobalVarDecl extends Node {
  kind: 'global-var-decl';
  type: TypeSpec;
  declarators: VarDeclarator[];
}

export interface ParamDecl extends Node {
  kind: 'param-decl';
  name: string;
  /** May carry in/out/inout, const and precision qualifiers. */
  type: TypeSpec;
  arrayDims: Expr[];
}

export interface FunctionPrototype extends Node {
  kind: 'function-prototype';
  name: string;
  returnType: TypeSpec;
  params: ParamDecl[];
}

export interface FunctionDefinition extends Node {
  kind: 'function-definition';
  prototype: FunctionPrototype;
  body: CompoundStmt;
}

/**
 * Uniform block (ES 3.00): `uniform BlockName { ... } instance[2];`
 * `blockName` = 'BlockName', `instanceName` = 'instance' (null when the block
 * has no instance name — members are then accessed by bare name).
 */
export interface InterfaceBlockDecl extends Node {
  kind: 'interface-block';
  qualifiers: TypeQualifiers; // 'uniform', layout(binding=)
  blockName: string;
  instanceName: string | null;
  members: StructMemberDecl[];
  arrayDims: Expr[];
}

/** Bare struct definition: `struct S { ... };` (type declaration only). */
export interface StructDecl extends Node {
  kind: 'struct-decl';
  name: string;
  members: StructMemberDecl[];
}

/** `precision highp float;` — base is restricted to float/int/sampler types. */
export interface PrecisionDecl extends Node {
  kind: 'precision-decl';
  precision: Precision;
  base: 'float' | 'int' | SamplerKind;
}

/** `#extension name : behavior` (preprocessor produces this). */
export interface ExtensionDecl extends Node {
  kind: 'extension-decl';
  name: string;
  behavior: 'require' | 'enable' | 'warn' | 'disable';
}

/** `invariant <name>;` (gl_Position, a varying, or an out variable). */
export interface InvariantDecl extends Node {
  kind: 'invariant-decl';
  name: string;
}

/* ------------------------------------------------------------------ */
/* Statements                                                          */
/* ------------------------------------------------------------------ */

export type Stmt =
  | CompoundStmt
  | DeclStmt
  | ExprStmt
  | IfStmt
  | ForStmt
  | WhileStmt
  | DoWhileStmt
  | SwitchStmt
  | CaseLabelStmt
  | BreakStmt
  | ContinueStmt
  | ReturnStmt
  | DiscardStmt
  | EmptyStmt;

export interface CompoundStmt extends Node {
  kind: 'compound';
  body: Stmt[];
}

/** Local variable declaration (may declare multiple declarators). */
export interface DeclStmt extends Node {
  kind: 'decl-stmt';
  type: TypeSpec;
  declarators: VarDeclarator[];
}

export interface ExprStmt extends Node {
  kind: 'expr-stmt';
  expr: Expr | null; // null = `;`
}

export interface IfStmt extends Node {
  kind: 'if';
  cond: Expr;
  then: Stmt;
  else: Stmt | null;
}

export interface ForStmt extends Node {
  kind: 'for';
  init: Stmt | null;
  cond: Expr | null;
  update: Expr | null;
  body: Stmt;
}

export interface WhileStmt extends Node {
  kind: 'while';
  cond: Expr;
  body: Stmt;
}

export interface DoWhileStmt extends Node {
  kind: 'do-while';
  body: Stmt;
  cond: Expr;
}

/** ES 3.00 only. Body must be a compound of case labels + statements. */
export interface SwitchStmt extends Node {
  kind: 'switch';
  expr: Expr;
  body: Stmt;
}

/** `case <const-expr>:` / `default:` — value null means default. */
export interface CaseLabelStmt extends Node {
  kind: 'case';
  value: Expr | null;
}

export interface BreakStmt extends Node {
  kind: 'break';
}

export interface ContinueStmt extends Node {
  kind: 'continue';
}

export interface ReturnStmt extends Node {
  kind: 'return';
  value: Expr | null;
}

/** Fragment only. Codegen lowers to `ctx.discarded = true; return;`. */
export interface DiscardStmt extends Node {
  kind: 'discard';
}

export interface EmptyStmt extends Node {
  kind: 'empty';
}

/* ------------------------------------------------------------------ */
/* Expressions                                                         */
/* ------------------------------------------------------------------ */

/** Common annotation fields filled by semantics on every expression node. */
export interface ExprBase extends Node {
  /** Filled by semantics: resolved GLSL type. */
  resolvedType?: GLSLType;
  /** Filled by semantics: constant-folded value for compile-time constants. */
  constValue?: number | boolean;
  /** Filled by semantics: true when this expression may have side effects. */
  lvalue?: boolean;
}

export type Expr =
  | LiteralExpr
  | IdentifierExpr
  | UnaryExpr
  | BinaryExpr
  | AssignExpr
  | TernaryExpr
  | CallExpr
  | IndexExpr
  | MemberExpr
  | CommaExpr;

export interface LiteralExpr extends ExprBase {
  kind: 'literal';
  /** Numeric value (uint literals > 2^31 stored as plain JS numbers). */
  value: number | boolean;
  /** 'int' | 'uint' | 'float' | 'bool' — the literal's own type. */
  literalType: 'int' | 'uint' | 'float' | 'bool';
}

export interface IdentifierExpr extends ExprBase {
  kind: 'identifier';
  name: string;
}

export type UnaryOp = '+' | '-' | '!' | '~' | '++' | '--';

export interface UnaryExpr extends ExprBase {
  kind: 'unary';
  op: UnaryOp;
  operand: Expr;
  /** True for POSTFIX `x++` / `x--` (expression result = OLD value of the
   *  operand); absent/false = prefix (`++x`, result = NEW value). */
  postfix?: boolean;
}

export type BinaryOp =
  | '+' | '-' | '*' | '/' | '%'
  | '<<' | '>>'
  | '<' | '>' | '<=' | '>='
  | '==' | '!='
  | '&' | '^' | '|'
  | '&&' | '||' | '^^';

export interface BinaryExpr extends ExprBase {
  kind: 'binary';
  op: BinaryOp;
  left: Expr;
  right: Expr;
}

export type AssignOp = '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '<<=' | '>>=' | '&=' | '^=' | '|=';

export interface AssignExpr extends ExprBase {
  kind: 'assign';
  op: AssignOp;
  target: Expr;
  value: Expr;
}

export interface TernaryExpr extends ExprBase {
  kind: 'ternary';
  cond: Expr;
  whenTrue: Expr;
  whenFalse: Expr;
}

/**
 * Function or constructor call. `callee` is an identifier (user function,
 * builtin, or type name for constructors) or a member expression
 * (`instance.method` is not valid GLSL — reserved for future extension;
 * semantics rejects non-identifier callees).
 */
export interface CallExpr extends ExprBase {
  kind: 'call';
  callee: Expr;
  args: Expr[];
}

/** `obj[i]` — array indexing or vector/matrix component indexing. */
export interface IndexExpr extends ExprBase {
  kind: 'index';
  object: Expr;
  index: Expr;
}

/**
 * `obj.name` — struct member access or vector SWIZZLE (semantics resolves:
 * swizzle names are xyzw/rgba/stpq subsets of the vector size).
 */
export interface MemberExpr extends ExprBase {
  kind: 'member';
  object: Expr;
  name: string;
}

/** `a, b` — sequence operator (ES 1.00 allows it in limited contexts). */
export interface CommaExpr extends ExprBase {
  kind: 'comma';
  exprs: Expr[];
}
