/**
 * functions.ts — the USER-FUNCTION INLINER (cross-module contract §1,
 * Design Decision §1: ALL user functions are inlined; zero per-invocation
 * allocation).
 *
 * installUserFunctions(ast, env) walks the TranslationUnit's function
 * definitions and installs env.emitUserCall. Every user call compiles to an
 * IIFE-wrapped inline, one per call site:
 *
 *   (function() {
 *     <arg materialization lines — left-to-right, GLSL evaluation order>
 *     <retTemps = 0 defensive init for non-void>
 *     EP_<n>: {
 *       var <param bindings>;
 *       <inlined body — returns compile to `retTemps[c] = ...; break EP_<n>;`>
 *     }
 *     <out/inout write-backs>
 *   })()
 *
 * WHY AN IIFE (deviation from "never wrap in a function wrapper" — see the
 * statements.ts header): the inlined lines must reach the statement stream
 * through the Value.pre seam, whose consumers fold pre entries as COMMA TERMS
 * (`foldPre`: `(p1, p2, v)` — used by every binary op, conditions, ternary,
 * comma and materialize). A bare labeled block is not an expression, so it
 * cannot be folded there (calls inside `f(x) * 2.0`, `if (f(x) > 0.5)`, or
 * nested argument lists would produce invalid JS). The IIFE is a single
 * valid expression in every consumer: statement emitters hoist it as a line
 * `(function(){...})();`, expression contexts fold it as
 * `((function(){...})(), r0)`. Inside the IIFE, every loop of the inlined
 * body is enclosed, so plain break/continue bind exactly as in GLSL, and
 * `break EP_<n>` (returns) plus the epilogue write-backs all live inside the
 * IIFE. The ONLY divergence is `discard` (`ctx.discarded = true; return;`):
 * the bare return exits the IIFE instead of the stage function — but the
 * rasterizer commits nothing when ctx.discarded is set, so the observable
 * behavior is identical (subsequent statements may run but their outputs
 * are discarded).
 *
 * PARAMETERS are not plain locals_: each call site pushes a ParamFrame
 * (env.pushParamFrame) holding per-call-site LocalVars (unique JS names
 * `<name>$c<N>` — `$` is not a GLSL identifier char, so they can never
 * collide with GLSL-derived names or temps) plus the function's local names
 * (pre-scanned from the body). env.resolveLocal consults the frames
 * top-down, so same-named params/locals of nested calls never collide
 * (locals_ alone can hold only ONE entry per GLSL name).
 *
 * KNOWN LIMITATIONS (reported):
 * - Same-named LOCALS of different types in different functions: statements.ts
 *   reuses the first registration (locals_ is keyed by GLSL name); a wider
 *   re-declaration throws a clear link-time error. Same-typed reuse is safe
 *   (each inline has its own IIFE var scope).
 * - Nested same-named ARRAY locals share one scratch region (silent aliasing).
 * - Call-site frames are NOT visible to statements.ts's decl-reuse path
 *   (lookupLocal stays locals_-only by design — see env.resolveLocal).
 */
import type {
  CompoundStmt,
  Expr,
  FunctionDefinition,
  ParamDecl,
  Stmt,
  TranslationUnit,
} from '../ast.js';
import type { GLSLType } from '../types.js';
import {
  CodegenEnv,
  LocalVar,
  convertScalar,
  flatComponents,
  foldPre,
  scalarBaseOf,
} from './env.js';
import { emitLValue } from './expressions.js';
import { emitStatements } from './statements.js';
import type { Value } from './index.js';

/**
 * Register all user functions from the TranslationUnit into env.emitUserCall.
 * Call BEFORE emitting main (main's body emission triggers the inlines).
 * Prototypes without definitions are left out (calling one falls through to
 * the builtin lookup, which reports it as an unknown builtin).
 */
export function installUserFunctions(ast: TranslationUnit, env: CodegenEnv): void {
  const fns = new Map<string, FunctionDefinition>();
  const fnLocalNames = new Map<string, Set<string>>();
  for (const d of ast.declarations) {
    if (d.kind === 'function-definition') {
      fns.set(d.prototype.name, d);
      fnLocalNames.set(d.prototype.name, collectLocalNames(d.body));
    }
  }
  const stack: string[] = [];
  let nextLabel = 0;
  let nextSuffix = 0;

  env.emitUserCall = (name, args, argTypes, rawArgs) => {
    const fn = fns.get(name);
    if (!fn) return null; // not a user function → builtin/ctor path
    if (name === 'main') {
      throw new Error("codegen: user function 'main' cannot be called (semantics should reject it)");
    }
    if (stack.includes(name)) {
      throw new Error(`codegen: recursive call to '${name}' (semantics should reject recursion)`);
    }
    stack.push(name);
    try {
      return inlineCall(fn, args, argTypes, rawArgs ?? [], env, fnLocalNames.get(name) ?? new Set(), {
        label: () => `EP_${nextLabel++}`,
        suffix: () => `$c${nextSuffix++}`,
      });
    } finally {
      stack.pop();
    }
  };
}

interface InlineCtx {
  /** Unique epilogue label for the call site (shared monotonic counter). */
  label(): string;
  /** Unique JS-name suffix for the call site's param locals. */
  suffix(): string;
}

interface OutArg {
  paramIndex: number;
  targets: string[];
  copyBack?: string;
}

/** Inline ONE call: args (left-to-right) → param bindings → body → write-backs. */
function inlineCall(
  fn: FunctionDefinition,
  args: Value[][],
  argTypes: GLSLType[],
  rawArgs: Expr[],
  env: CodegenEnv,
  fnLocalNames: Set<string>,
  ctx: InlineCtx,
): Value[] {
  const params = fn.prototype.params;
  if (params.length !== rawArgs.length || params.length !== args.length) {
    throw new Error(
      `codegen: call to '${fn.prototype.name}' with ${rawArgs.length} args, expected ${params.length}`,
    );
  }
  const frame = env.pushParamFrame();
  try {
    for (const n of fnLocalNames) frame.localNames.add(n);

    /* ---------- 1. args left-to-right (GLSL evaluation order) ---------- */
    const lines: string[] = [];
    const argTemps: string[][] = []; // per param: one temp per flat component
    const outArgs: OutArg[] = [];
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      const storage = p.type.qualifiers.storage ?? 'in';
      const isOut = storage === 'out' || storage === 'inout';
      const paramType = paramTypeOf(p, rawArgs[i].resolvedType);
      const conv = convertArg(args[i], argTypes[i], paramType);
      if (isOut) {
        // Capture the caller lvalue NOW (its prelude runs at this arg position).
        const lv = emitLValue(rawArgs[i], env);
        if (lv.prelude) lines.push(lv.prelude);
        if (storage === 'inout') {
          const temps = conv.map((v) => toTemp(v, env));
          for (const t of temps) lines.push(`${t.pre![0]};`);
          argTemps.push(temps.map((t) => t.v));
        } else {
          argTemps.push([]); // out: no in-value
        }
        outArgs.push({ paramIndex: i, targets: lv.targets, copyBack: lv.copyBack });
      } else {
        // in (or const-in): read the value at this arg position.
        const temps = conv.map((v) => toTemp(v, env));
        for (const t of temps) lines.push(`${t.pre![0]};`);
        argTemps.push(temps.map((t) => t.v));
      }
    }

    /* ---------- 2. param registration + bindings ---------- */
    const paramLVs: LocalVar[] = [];
    const bindLines: string[] = [];
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      const storage = p.type.qualifiers.storage ?? 'in';
      const isOut = storage === 'out' || storage === 'inout';
      const paramType = paramTypeOf(p, rawArgs[i].resolvedType);
      const lv = env.makeParamLocal(p.name, paramType, ctx.suffix());
      frame.params.set(p.name, lv);
      paramLVs.push(lv);
      if (lv.kind === 'scratch') {
        // Array param: copy the arg's flat components into the param's region.
        const store = lv.int ? 'ctx.intScratch' : 'ctx.scratch';
        const temps = argTemps[i];
        for (let c = 0; c < temps.length; c++) {
          bindLines.push(`${store}[${lv.scratchBase} + ${c}] = ${temps[c]};`);
        }
        continue;
      }
      const names = lv.compNames!;
      if (storage === 'out') {
        // 'out': no in-value — declare uninitialized.
        bindLines.push(`var ${names.join(', ')};`);
      } else {
        // 'in' and 'inout' both carry an in-value temp from the arg phase.
        const temps = argTemps[i];
        bindLines.push(`var ${names.map((n, c) => `${n} = ${temps[c]}`).join(', ')};`);
      }
    }

    /* ---------- 3. return temps ---------- */
    const retType = fn.prototype.returnType.resolved ?? { kind: 'scalar', base: 'float' };
    const nRet = flatComponents(retType);
    const retTemps: string[] = [];
    for (let c = 0; c < nRet; c++) retTemps.push(env.allocTemp());

    /* ---------- 4. inlined body ---------- */
    const label = ctx.label();
    const bodyLines = emitStatements(fn.body.body, env, {
      retTemps,
      epilogueLabel: label,
      retType,
    });

    /* ---------- 5. write-backs (after the labeled block — a
     * `break EP_<n>` from a return lands right after `}` and still runs) ---- */
    const wbLines: string[] = [];
    for (const o of outArgs) {
      const lv = paramLVs[o.paramIndex];
      const n = Math.min(o.targets.length, flatComponents(lv.type));
      if (lv.kind === 'scratch') {
        const store = lv.int ? 'ctx.intScratch' : 'ctx.scratch';
        for (let c = 0; c < n; c++) {
          wbLines.push(`${o.targets[c]} = ${store}[${lv.scratchBase} + ${c}];`);
        }
      } else {
        for (let c = 0; c < n; c++) wbLines.push(`${o.targets[c]} = ${lv.compNames![c]};`);
      }
      if (o.copyBack) wbLines.push(o.copyBack);
    }

    /* ---------- 6. assemble the IIFE expression ---------- */
    const inner: string[] = [];
    // Arg materialization lines FIRST — they run at the exact GLSL arg
    // positions (left-to-right, interleaved with out/inout preludes).
    inner.push(...lines);
    if (nRet > 0) {
      // Defensive fall-off-the-end init (semantics should enforce coverage).
      for (const t of retTemps) inner.push(`${t} = 0;`);
    }
    inner.push(`${label}: {`);
    inner.push(...indent([...bindLines, ...bodyLines]));
    inner.push('}');
    inner.push(...wbLines);
    const iife = `(function() {\n${inner.map((l) => '  ' + l).join('\n')}\n})()`;

    if (nRet === 0) {
      // Void call: a dummy value whose pre carries the whole inline — the
      // statement emitters hoist it as a line, expression contexts fold it.
      return [{ v: '0', pre: [iife] }];
    }
    const pre = [iife];
    return retTemps.map((t) => ({ v: t, pre }));
  } finally {
    env.popParamFrame();
  }
}

/** The param's GLSL type: base + array dims; an UNSIZED outer dim (`float a[]`)
 *  takes its size from the ARG's resolved array type (GLSL ES 3.00). */
function paramTypeOf(p: ParamDecl, argType?: GLSLType): GLSLType {
  const base = p.type.resolved;
  if (!base) throw new Error('codegen: param type unresolved (semantics must run first)');
  if (p.arrayDims.length === 0) return base;
  let t = base;
  for (let i = p.arrayDims.length - 1; i >= 0; i--) {
    const dim = p.arrayDims[i] as Expr | null;
    const cv = dim ? dim.constValue : undefined;
    let size = typeof cv === 'number' && Number.isInteger(cv) && cv > 0 ? cv : 0;
    if (size <= 0 && argType && argType.kind === 'array') size = argType.size ?? 1;
    if (size <= 0) size = 1; // defensive (semantics guarantees a size)
    t = { kind: 'array', element: t, size };
  }
  return t;
}

/** Convert arg values to the param's scalar base, PRESERVING pre (env's
 *  convertValue drops pre on conversion — the inliner must keep nested-call
 *  IIFEs attached). */
function convertArg(vals: Value[], from: GLSLType, to: GLSLType): Value[] {
  const fb = scalarBaseOf(from);
  const tb = scalarBaseOf(to);
  if (fb === null || tb === null || fb === tb) return vals;
  return vals.map((v) => ({ v: convertScalar(v.v, fb, tb), pre: v.pre }));
}

/** Materialize ONE value into a fresh temp — ALWAYS, even without pre, so
 *  side-effectful arg expressions (assignments) run at their exact arg
 *  position (left-to-right interleaving with later args' preludes). */
function toTemp(v: Value, env: CodegenEnv): Value {
  const t = env.allocTemp();
  return { v: t, pre: [`${t} = ${v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v}`] };
}

/** All local variable names declared anywhere in a function body (params are
 *  separate). Stamped on the call's frame so resolveLocal prefers an inner
 *  function's local over an outer function's same-named param. */
function collectLocalNames(body: CompoundStmt): Set<string> {
  const names = new Set<string>();
  const walk = (stmts: Stmt[]): void => {
    for (const s of stmts) {
      switch (s.kind) {
        case 'compound':
          walk(s.body);
          break;
        case 'decl-stmt':
          for (const d of s.declarators) if (d.name !== '') names.add(d.name);
          break;
        case 'if':
          walk([s.then]);
          if (s.else) walk([s.else]);
          break;
        case 'for':
          if (s.init) walk([s.init]);
          walk([s.body]);
          break;
        case 'while':
        case 'do-while':
          walk([s.body]);
          break;
        case 'switch':
          walk(s.body.kind === 'compound' ? s.body.body : [s.body]);
          break;
        default:
          break;
      }
    }
  };
  walk(body.body);
  return names;
}

function indent(lines: string[]): string[] {
  return lines.map((l) => '  ' + l);
}
