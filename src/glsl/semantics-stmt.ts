/**
 * semantics-stmt.ts — statement semantic analysis for GLSL ES 1.00 / 3.00.
 *
 * `analyzeStatement` walks one statement; expression sub-nodes are delegated
 * to `analyzeExpr` (which annotates them in place). Handles: compound
 * statements (fresh scope), local declarations (shared `declareVariables`
 * path), if/for/while/do-while (boolean conditions, loop-depth tracking for
 * break/continue legality), switch (ES 3.00 — integer selector, constant
 * integer case labels), break/continue/return (return type checked against
 * the enclosing function's return type via implicit conversions), discard
 * (accepted — stage/qualifier rules are the follow-up's declaration work),
 * and empty statements.
 *
 * WEBGL 1.0 LOOP RESTRICTIONS (WebGL 1.0 spec §6.26, GLSL ES 1.00 Appendix A):
 * in ESSL 1.00 shaders (declared version 100) `while` and `do-while` are
 * disallowed and `for` conditions must be of the form `index op
 * constant-expression` (a comparison whose RHS folds to a compile-time
 * constant). ESSL 3.00 shaders (WebGL 2.0) are unrestricted.
 *
 * The `fn` argument carries the enclosing function's return type. Depth
 * counters live on SemContext (loopDepth / breakableDepth / switchDepth) so
 * nested functions (none in GLSL) would not interfere; they are reset per
 * function body by the caller.
 */
import type { CompoundStmt, DeclStmt, ExprStmt, ForStmt, IfStmt, Stmt } from './ast.js';
import type { GLSLType } from './types.js';
import { typeName } from './types.js';
import type { Scope, SemContext } from './semantics.js';
import { declareVariables, resolveTypeSpec } from './semantics.js';
import { analyzeExpr, convertible } from './semantics-expr.js';

/** True for bool scalars/vectors (`bool`, `bvec2..4`). */
function isBool(t: GLSLType): boolean {
  return (t.kind === 'scalar' || t.kind === 'vector') && t.base === 'bool';
}

/**
 * Struct-nesting depth of a type: 0 for non-structs (arrays unwrap to their
 * element), 1 + max(member depth) for structs. Memoized by struct name
 * (recursive structs are rejected at semantics, so the memo is safe; a
 * name-guard still prevents pathological loops).
 */
function structDepth(t: GLSLType, memo: Map<string, number>): number {
  if (t.kind === 'array') return structDepth(t.element, memo);
  if (t.kind !== 'struct') return 0;
  const cached = memo.get(t.name);
  if (cached !== undefined) return cached;
  memo.set(t.name, 0); // in-progress guard (defensive)
  let inner = 0;
  for (const m of t.members) {
    const md = structDepth(m.type, memo);
    if (md > inner) inner = md;
  }
  const depth = 1 + inner;
  memo.set(t.name, depth);
  return depth;
}

/** Report an error when `e`'s resolved type is not a boolean scalar. */
function requireBool(e: { loc: { line: number }; resolvedType?: GLSLType }, ctx: SemContext, msg: string): void {
  const t = e.resolvedType;
  if (t === undefined) return;
  if (!(t.kind === 'scalar' && t.base === 'bool')) ctx.error(e.loc.line, msg);
}

export function analyzeStatement(s: Stmt, scope: Scope, ctx: SemContext, fn: { returnType: GLSLType }): void {
  switch (s.kind) {
    case 'compound':
      analyzeCompound(s, scope, ctx, fn);
      return;
    case 'decl-stmt': {
      const baseType = resolveTypeSpec(s.type, scope, ctx);
      if (baseType !== null) {
        // GLSL ES §4.3.3/§4.3.5/§4.3.6: attribute/varying/uniform declarations
        // are GLOBAL-only. `attribute`/`varying` are 1.00 keywords (locals with
        // them never parse in 3.00); `uniform` inside a function body is
        // illegal in both versions. (ogles build attribute_frag/attribute_vert/
        // uniform_frag/varying_frag.)
        const storage = s.type.qualifiers.storage;
        if (storage === 'attribute' || storage === 'varying' || storage === 'uniform') {
          ctx.error(s.loc.line, `'${storage}' : storage qualifiers are only allowed on global declarations`);
        }
        // GLSL ES §4.5.3: precision qualifiers apply to float/int/sampler
        // types only — `mediump bool` / `mediump bvecN` are illegal
        // (conformance/glsl/misc/boolean_precision.html).
        if (s.type.qualifiers.precision !== undefined && isBool(baseType)) {
          ctx.error(s.loc.line, `'${s.type.qualifiers.precision}' : precision qualifiers are not allowed on bool types`);
        }
        if (s.type.base.kind === 'struct-definition' && baseType.kind === 'struct') {
          if (s.type.base.members.length === 0) {
            ctx.error(s.loc.line, `'struct' : structure must have at least one member`);
          }
          // `struct S {...};` / `struct S {...} v;` inside a function body:
          // register the struct TYPE in the CURRENT scope so later statements
          // in the same scope (declarations, constructors, member-typed
          // declarations of nested structs) resolve it — mirrors the global
          // pre-pass registration (analyzeGlobalVarDecl/registerStructDecl).
          // Registered BEFORE the declarators so `struct S {...} s = S(...)`
          // initializers can use the constructor.
          scope.declare({ kind: 'struct', name: baseType.name, type: baseType }, ctx, s.loc.line);
        }
        declareVariables(baseType, s.type, s.declarators, scope, ctx, false);
        // GLSL ES 1.00 Appendix A §5: arrays of arrays are 3.00-only (ogles
        // build array1). Array sizes must be CONSTANT INTEGRAL expressions —
        // a float const (`const float i = 3.0; float f[i];`) is illegal even
        // when its value is integral (ogles build array6). Runs AFTER
        // declareVariables (wrapArrayDims analyzes the dim expressions).
        if (ctx.version === 100) {
          for (const d of s.declarators) {
            if (d.arrayDims.length > 1) {
              ctx.error(d.loc.line, `'[' : arrays of arrays are not allowed in GLSL ES 1.00`);
            }
            for (const dim of d.arrayDims) {
              const dt = dim.resolvedType;
              if (dt !== undefined && dt.kind === 'scalar' && dt.base === 'float') {
                ctx.error(dim.loc.line, 'array size must be a constant integer expression');
              }
            }
          }
        }
        // WebGL 1.0 limit: at most 4 levels of struct nesting
        // (conformance/glsl/misc/struct-nesting-exceeds-maximum.html).
        if (ctx.version === 100 && baseType.kind === 'struct' && structDepth(baseType, new Map()) > 4) {
          ctx.error(s.loc.line, `'struct' : structure nesting exceeds the maximum of 4 levels`);
        }
      }
      return;
    }
    case 'expr-stmt':
      if (s.expr !== null) analyzeExpr(s.expr, scope, ctx);
      return;
    case 'if':
      analyzeIf(s, scope, ctx, fn);
      return;
    case 'for': {
      const inner = scope.push();
      if (s.init !== null) analyzeStatement(s.init, inner, ctx, fn);
      ctx.loopDepth++;
      ctx.breakableDepth++;
      if (s.cond !== null) {
        analyzeExpr(s.cond, inner, ctx);
        requireBool(s.cond, ctx, `'for' : condition must be a boolean expression`);
        if (ctx.version === 100) {
          // WebGL 1.0 §6.26: for loops must conform to the GLSL ES 1.00
          // Appendix A structural constraints — the condition must be of the
          // form `index op constant-expression`. Const-folding (analyzeExpr
          // above) annotates `constValue` on the RHS when it is a literal, a
          // const-qualified variable, or a foldable constant expression, so
          // `i < 5 + 5` / `i < constVar` pass while `i < u_numIterations`
          // (uniform) and `i < nonConstVar` (non-const local) fail. ESSL 3.00
          // (WebGL 2.0) shaders are unrestricted.
          const c = s.cond;
          const constCond =
            c.kind === 'binary' &&
            (c.op === '<' || c.op === '>' || c.op === '<=' || c.op === '>=' || c.op === '==' || c.op === '!=') &&
            c.right.constValue !== undefined;
          if (!constCond) {
            ctx.error(c.loc.line, `'for' : loop condition must be a comparison with a constant expression (WebGL 1.0)`);
          }
        }
      }
      if (s.update !== null) analyzeExpr(s.update, inner, ctx);
      analyzeStatement(s.body, inner, ctx, fn);
      ctx.loopDepth--;
      ctx.breakableDepth--;
      return;
    }
    case 'while': {
      // WebGL 1.0 §6.26: while loops are disallowed in ESSL 1.00 (they are
      // only OPTIONAL in GLSL ES 1.00 Appendix A). ESSL 3.00 keeps them.
      if (ctx.version === 100) ctx.error(s.loc.line, `'while' : not supported in WebGL 1.0`);
      analyzeExpr(s.cond, scope, ctx);
      requireBool(s.cond, ctx, `'while' : condition must be a boolean expression`);
      ctx.loopDepth++;
      ctx.breakableDepth++;
      analyzeStatement(s.body, scope, ctx, fn);
      ctx.loopDepth--;
      ctx.breakableDepth--;
      return;
    }
    case 'do-while': {
      // WebGL 1.0 §6.26: do-while loops are disallowed in ESSL 1.00 (only
      // optional in GLSL ES 1.00 Appendix A). ESSL 3.00 keeps them.
      if (ctx.version === 100) ctx.error(s.loc.line, `'do-while' : not supported in WebGL 1.0`);
      ctx.loopDepth++;
      ctx.breakableDepth++;
      analyzeStatement(s.body, scope, ctx, fn);
      ctx.loopDepth--;
      ctx.breakableDepth--;
      analyzeExpr(s.cond, scope, ctx);
      requireBool(s.cond, ctx, `'do-while' : condition must be a boolean expression`);
      return;
    }
    case 'switch': {
      analyzeExpr(s.expr, scope, ctx);
      const t = s.expr.resolvedType;
      if (t === undefined) return;
      if (!(t.kind === 'scalar' && (t.base === 'int' || t.base === 'uint'))) {
        ctx.error(s.expr.loc.line, `'switch' : expression must be an integer scalar`);
        return;
      }
      ctx.breakableDepth++;
      ctx.switchDepth++;
      analyzeStatement(s.body, scope, ctx, fn);
      ctx.breakableDepth--;
      ctx.switchDepth--;
      return;
    }
    case 'case': {
      if (ctx.switchDepth <= 0) {
        ctx.error(s.loc.line, `'case' : case label not inside a switch statement`);
        return;
      }
      if (s.value === null) return; // `default:`
      analyzeExpr(s.value, scope, ctx);
      const t = s.value.resolvedType;
      if (t === undefined) return;
      if (!(t.kind === 'scalar' && (t.base === 'int' || t.base === 'uint'))) {
        ctx.error(s.value.loc.line, `'case' : case label must be an integer`);
        return;
      }
      const cv = s.value.constValue;
      if (typeof cv !== 'number' || !Number.isInteger(cv)) {
        ctx.error(s.value.loc.line, `'case' : case label must be a constant integer expression`);
      }
      return;
    }
    case 'break':
      if (ctx.breakableDepth <= 0) ctx.error(s.loc.line, `'break' : not inside a loop or switch`);
      return;
    case 'continue':
      if (ctx.loopDepth <= 0) ctx.error(s.loc.line, `'continue' : not inside a loop`);
      return;
    case 'return': {
      if (s.value === null) {
        if (fn.returnType.kind !== 'void') {
          ctx.error(s.loc.line, `'return' : non-void function must return a value`);
        }
        return;
      }
      analyzeExpr(s.value, scope, ctx);
      const vt = s.value.resolvedType;
      if (vt === undefined) return;
      if (fn.returnType.kind === 'void') {
        ctx.error(s.loc.line, `'return' : void function cannot return a value`);
      } else if (vt.kind === 'void') {
        ctx.error(s.value.loc.line, `'return' : cannot return a void expression`);
      } else if (!convertible(vt, fn.returnType, ctx.version)) {
        ctx.error(s.value.loc.line, `cannot convert from '${typeName(vt)}' to '${typeName(fn.returnType)}'`);
      }
      return;
    }
    case 'discard':
      // GLSL ES: discard is a fragment-stage statement only.
      if (ctx.stage !== 'FRAGMENT') {
        ctx.error(s.loc.line, "'discard' : only allowed in fragment shaders");
      }
      return;
    case 'empty':
      return;
  }
}

function analyzeCompound(s: CompoundStmt, scope: Scope, ctx: SemContext, fn: { returnType: GLSLType }): void {
  const inner = scope.push();
  for (const st of s.body) analyzeStatement(st, inner, ctx, fn);
}

function analyzeIf(s: IfStmt, scope: Scope, ctx: SemContext, fn: { returnType: GLSLType }): void {
  analyzeExpr(s.cond, scope, ctx);
  requireBool(s.cond, ctx, `'if' : condition must be a boolean expression`);
  // GLSL ES Appendix A: selection-statement SUBSTATEMENTS each get their own
  // scope — `if (true) int g = 4;` scopes g to the substatement, so a second
  // `if (true) int g = 4;` does not collide and references after the
  // statement cannot see g/q (CTS shader-with-conditional-scoping.html and
  // its -negative twin). Braced substatements push one more scope inside
  // (analyzeCompound) — harmless nesting.
  const thenScope = scope.push();
  analyzeStatement(s.then, thenScope, ctx, fn);
  if (s.else !== null) {
    const elseScope = scope.push();
    analyzeStatement(s.else, elseScope, ctx, fn);
  }
}

// Type-only re-export so statement/expression analysis share one entry point.
export type { DeclStmt, ExprStmt, ForStmt };
