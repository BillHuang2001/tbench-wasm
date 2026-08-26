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
      if (baseType !== null) declareVariables(baseType, s.type, s.declarators, scope, ctx, false);
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
      }
      if (s.update !== null) analyzeExpr(s.update, inner, ctx);
      analyzeStatement(s.body, inner, ctx, fn);
      ctx.loopDepth--;
      ctx.breakableDepth--;
      return;
    }
    case 'while': {
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
      // Stage legality (fragment-only) is the follow-up's declaration work.
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
  analyzeStatement(s.then, scope, ctx, fn);
  if (s.else !== null) analyzeStatement(s.else, scope, ctx, fn);
}

// Type-only re-export so statement/expression analysis share one entry point.
export type { DeclStmt, ExprStmt, ForStmt };
