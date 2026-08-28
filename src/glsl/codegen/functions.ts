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
 * (pre-scanned from the body). env.resolveLocal consults the CURRENT
 * function's frame (env.bodyDepth - 1), so same-named params/locals of
 * nested calls never collide (locals_ alone can hold only ONE entry per
 * GLSL name), while CALLER frames stay invisible to the inlined body:
 * a callee body sees only its own scope + globals, exactly as GLSL scoping
 * requires (in-parameter-passed-as-inout-argument-and-global — a free name
 * matching the caller's param must resolve to the global). Arg
 * materialization runs BEFORE the body (bodyDepth not yet incremented), so
 * the args of a nested call resolve in the CALLER's scope, where they
 * belong.
 *
 * The function's OWN body locals get the SAME per-call-site treatment
 * (env.frameLocal, consulted by statements.ts's emitDeclStmt BEFORE the
 * locals_ reuse path): the first declaration of each local inside the
 * inlined body materializes it through makeParamLocal with the call site's
 * `<name>$c<N>` suffix, so a callee local can never alias a caller's
 * same-named local (ogles functions pages — without this, the shared
 * locals_ registration made the callee's `var` shadow the caller's values
 * via IIFE hoisting) and nested same-named locals never share scratch.
 * Sibling-scope re-declarations inside one body reuse the frame's var.
 *
 * KNOWN LIMITATIONS (reported):
 * - A local declared in DIFFERENT sibling scopes of one inlined body with a
 *   WIDER type than the first declaration throws a clear link-time error
 *   (same rule as the locals_ path — the frame var keeps the first type).
 * - Call-site frames are NOT visible to statements.ts's decl-reuse path
 *   (lookupLocal stays locals_-only by design — see env.resolveLocal;
 *   emitDeclStmt checks env.frameLocal first).
 */
import type {
  CompoundStmt,
  Expr,
  FunctionDefinition,
  ParamDecl,
  Stmt,
  TranslationUnit,
  VarDeclarator,
} from '../ast.js';
import type { GLSLType } from '../types.js';
import { typeEquals } from '../types.js';
import { matches, builtinSignatures } from '../builtins/index.js';
import {
  CodegenEnv,
  LocalVar,
  convertScalar,
  flatComponents,
  flatFloatness,
  foldPre,
  scalarBaseOf,
} from './env.js';
import { emitExpr, emitLValue } from './expressions.js';
import { emitStatements } from './statements.js';
import type { Value } from './index.js';

/**
 * Register all user functions from the TranslationUnit into env.emitUserCall.
 * Call BEFORE emitting main (main's body emission triggers the inlines).
 * Prototypes without definitions are left out (calling one falls through to
 * the builtin lookup, which reports it as an unknown builtin).
 *
 * OVERLOADS (BUG A fix): the registry is keyed by SIGNATURE (name + declared
 * param type list — storage qualifiers like out/inout are not part of the
 * GLSL overload identity), never by name alone: a name-keyed registry let
 * overloads overwrite each other and every call inlined the LAST definition
 * (`Cannot read properties of undefined (reading 'v')` at bind-line
 * generation when the call's component counts differed, silent miscompiles
 * when they aligned). emitUserCall re-derives the overload resolution
 * (mirror of semantics' scoreSignature/pickBest over all DEFINITIONS of the
 * name) and inlines the chosen body with ITS OWN locals.
 */
export function installUserFunctions(ast: TranslationUnit, env: CodegenEnv): void {
  const fns = new Map<string, FunctionDefinition>();
  const fnLocalNames = new Map<string, Set<string>>();
  const byName = new Map<string, string[]>(); // name → signature keys (definition order)
  for (const d of ast.declarations) {
    if (d.kind !== 'function-definition') continue; // bare prototypes have no body
    const name = d.prototype.name;
    if (name === 'main') continue; // the entry point is never a callable
    const key = signatureKey(name, d.prototype.params.map(declaredParamType));
    fns.set(key, d);
    fnLocalNames.set(key, collectLocalNames(d.body));
    let keys = byName.get(name);
    if (!keys) byName.set(name, (keys = []));
    keys.push(key);
  }
  const stack: string[] = [];
  let nextLabel = 0;
  let nextSuffix = 0;

  env.emitUserCall = (name, args, argTypes, rawArgs) => {
    const keys = byName.get(name);
    if (!keys || keys.length === 0) return null; // not a user function → builtin/ctor path
    if (name === 'main') {
      throw new Error("codegen: user function 'main' cannot be called (semantics should reject it)");
    }
    const best = pickUserOverload(name, keys, fns, rawArgs ?? [], env.layout.version);
    if (best === null) return null; // no user overload matches → builtin/ctor path
    if (best.score > 0 && matches(name, builtinSignatures(env.layout.version)).length > 0) {
      // Hybrid guard (GLSL ES 1.00 §6.1): user overloads of builtin names
      // resolve across user AND builtin signatures; a NON-exact user match
      // against a name with builtin signatures falls through to the builtin
      // path (in reachable states semantics never lets the user win at
      // score > 0 while a builtin signature exists — ties are link errors).
      return null;
    }
    const key = best.key;
    const fn = fns.get(key)!;
    // Signature-aware recursion backstop: the stack holds the signature KEYS
    // of in-progress inlinings, so an overload call (`process(S1)` calling
    // `process(S2)` — different key) is NOT flagged; only a true same-signature
    // self-call or a call cycle through signatures is (semantics rejects those
    // at compile time — this guard is defensive).
    if (stack.includes(key)) {
      throw new Error(`codegen: recursive call to '${name}' (semantics should reject recursion)`);
    }
    stack.push(key);
    try {
      return inlineCall(fn, args, argTypes, rawArgs ?? [], env, fnLocalNames.get(key) ?? new Set(), {
        label: () => `EP_${nextLabel++}`,
        suffix: () => `$c${nextSuffix++}`,
      });
    } finally {
      stack.pop();
    }
  };
}

/** The declared GLSL type of a function param: base + array dims (an unsized
 *  `[]` dim stays size null — mirror of semantics' wrapArrayDims). */
function declaredParamType(p: ParamDecl): GLSLType {
  const base = p.type.resolved;
  if (!base) throw new Error('codegen: param type unresolved (semantics must run first)');
  if (p.arrayDims.length === 0) return base;
  let t = base;
  for (let i = p.arrayDims.length - 1; i >= 0; i--) {
    const dim = p.arrayDims[i] as Expr | null;
    const cv = dim ? dim.constValue : undefined;
    const size = typeof cv === 'number' && Number.isInteger(cv) && cv > 0 ? cv : dim === null ? null : 1;
    t = { kind: 'array', element: t, size };
  }
  return t;
}

/** Canonical signature key: name + declared param types. Storage qualifiers
 *  (out/inout) are ignored — GLSL overload identity is the param type list. */
function signatureKey(name: string, params: GLSLType[]): string {
  return `${name}(${params.map(typeKey).join(',')})`;
}

function typeKey(t: GLSLType): string {
  switch (t.kind) {
    case 'void':
      return 'v';
    case 'scalar':
      return t.base;
    case 'vector':
      return `${t.base}${t.size}`;
    case 'matrix':
      return `m${t.cols}x${t.rows}`;
    case 'sampler':
      return `s:${t.sampler}`;
    case 'struct':
      return `t:${t.name}`;
    case 'array':
      return `${typeKey(t.element)}[${t.size === null ? '' : t.size}]`;
  }
}

interface UserOverloadPick {
  key: string;
  /** 0 = every arg typeEquals the param; > 0 = implicit conversions used
   *  (unreachable at HEAD — semantics' convertible IS typeEquals). */
  score: number;
}

/**
 * Re-derive the overload resolution semantics performed at analyze time
 * (semantics-expr.ts scoreSignature/pickBest — READ-ONLY there, mirrored
 * here): per-arg 0 = exact typeEquals, 1 = convertible (semantics'
 * convertible(from, to) is typeEquals at HEAD — no implicit promotions),
 * else no match; the unique lowest-scoring user DEFINITION wins; a tie is a
 * link error in semantics (unreachable here — treat as no user match);
 * no match → null (builtin/ctor path). `rawArgs[i].resolvedType` is the same
 * data semantics scored, so the re-derivation is exact.
 */
function pickUserOverload(
  name: string,
  keys: string[],
  fns: Map<string, FunctionDefinition>,
  rawArgs: Expr[],
  version: 100 | 300,
): UserOverloadPick | null {
  void version;
  const argTypes = rawArgs.map((a) => a.resolvedType!);
  let bestKey: string | null = null;
  let bestScore = Infinity;
  for (const key of keys) {
    const params = fns.get(key)!.prototype.params.map(declaredParamType);
    if (params.length !== argTypes.length) continue;
    let score = 0;
    let ok = true;
    for (let i = 0; i < params.length; i++) {
      const at = argTypes[i];
      if (typeEquals(at, params[i])) continue;
      // semantics' convertible(from, to) === typeEquals(from, to) at HEAD
      // (semantics-expr.ts:83) — no implicit promotions exist; kept as a
      // distinct branch so a future convertible stays mirrored.
      if (typeEquals(at, params[i])) {
        score += 1;
        continue;
      }
      ok = false;
      break;
    }
    if (!ok) continue;
    if (score < bestScore) {
      bestScore = score;
      bestKey = key;
    } else if (score === bestScore) {
      // Ambiguous (semantics reports a link error) — never pick arbitrarily.
      return null;
    }
  }
  if (bestKey === null) return null;
  return { key: bestKey, score: bestScore };
}

interface InlineCtx {
  /** Unique epilogue label for the call site (shared monotonic counter). */
  label(): string;
  /** Unique JS-name suffix for the call site's param locals. */
  suffix(): string;
}

/** The declared type of one global declarator: base + array dims (mirror of
 *  semantics-decl's fullType — dims fold to positive ints, recovery size 1). */
function globalDeclType(base: GLSLType, dec: VarDeclarator): GLSLType {
  if (dec.arrayDims.length === 0) return base;
  let t = base;
  for (let i = dec.arrayDims.length - 1; i >= 0; i--) {
    const dim = dec.arrayDims[i];
    const cv = dim.constValue;
    t = { kind: 'array', element: t, size: typeof cv === 'number' && cv > 0 ? cv : 1 };
  }
  return t;
}

/**
 * Register non-const file-scope GLOBAL variables into the env's scratch
 * storage (BUG B fix): `float gray = 0.0;` at file scope is legal GLSL ES
 * 1.00/3.00 (constant initializer); semantics accepts it and the walker
 * resolves the identifier through env.resolveLocal — but codegen had NO
 * storage surface for non-const globals (`codegen: unknown identifier` at
 * expressions.ts:547).
 *
 * Each global becomes a SCRATCH-kind LocalVar in env.locals_ (declareLocal
 * with the array flag — scratch is REQUIRED: there is no `var` declaration
 * site for a global, so flat JS-var locals would be undeclared in the body).
 * Safe: semantics forbids locals shadowing globals (Scope.declare rejects
 * redefinition in any enclosing scope), so resolveLocal/walk/leafRead/
 * leafWrite/leafDual/ensureDynScratch all handle it via the existing
 * scratch machinery. `const` globals are skipped (they fold via constValue).
 *
 * Returns the initialization LINES to prepend to main's body (run once per
 * invocation, before main's statements): per component
 * `ctx.scratch[base + c] = <init>;` (ctx.intScratch for integral blocks),
 * where `<init>` = the declarator's const init emitted per component, or 0
 * when there is no initializer. Dual mode: the dx/dy planes are zeroed
 * (allocScratch charged 3× — v at base, dx at base+n, dy at base+2n).
 */
export function installUserGlobals(ast: TranslationUnit, env: CodegenEnv): string[] {
  const lines: string[] = [];
  for (const d of ast.declarations) {
    if (d.kind !== 'global-var-decl') continue;
    // attribute/varying/uniform/in/out/const globals live on other surfaces
    // (layout-driven storage / const folding) — plain variables only.
    if (d.type.qualifiers.storage !== undefined) continue;
    const base = d.type.resolved;
    if (!base) continue; // defensive (semantics resolves every declaration)
    for (const dec of d.declarators) {
      const type = globalDeclType(base, dec);
      env.declareLocal(dec.name, type, { array: true }); // scratch-backed
      const lv = env.lookupLocal(dec.name)!;
      const n = flatComponents(type);
      const int = lv.int === true;
      const store = int ? 'ctx.intScratch' : 'ctx.scratch';
      const baseOff = lv.scratchBase!;
      if (dec.init) {
        const vals = emitExpr(dec.init, env);
        for (let c = 0; c < n; c++) {
          const v = vals[c];
          lines.push(`${store}[${baseOff} + ${c}] = ${v.pre && v.pre.length ? foldPre(v.pre, v.v) : v.v};`);
        }
      } else {
        for (let c = 0; c < n; c++) lines.push(`${store}[${baseOff} + ${c}] = 0;`);
      }
      if (env.dual && !int) {
        for (let c = 0; c < n; c++) {
          lines.push(`${store}[${baseOff} + ${n} + ${c}] = 0;`);
          lines.push(`${store}[${baseOff} + ${2 * n} + ${c}] = 0;`);
        }
      }
    }
  }
  return lines;
}

interface OutArg {
  paramIndex: number;
  targets: string[];
  /** Caller-lvalue dual slots per component (null = no dual planes). */
  dualTargets: ([string, string] | null)[] | null;
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
  // Defensive: the inlined body's param SHAPES must match the call's value
  // counts — a wrong-overload inline would otherwise crash on undefined-'.v'
  // (short temp arrays) or silently miscompile (aligned counts, wrong body).
  for (let i = 0; i < params.length; i++) {
    const n = flatComponents(paramTypeOf(params[i], rawArgs[i].resolvedType));
    if (args[i].length !== n) {
      throw new Error(
        `codegen: overload mismatch for '${fn.prototype.name}': param ${i} has ${n} components, call passes ${args[i].length}`,
      );
    }
  }
  const frame = env.pushParamFrame(ctx.suffix);
  try {
    for (const n of fnLocalNames) frame.localNames.add(n);

    /* ---------- 1. args left-to-right (GLSL evaluation order) ---------- */
    const lines: string[] = [];
    const argTemps: Value[][] = []; // per param: one temp Value per flat component
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
          // in-value: read the captured lvalue's targets NOW (after its
          // prelude ran). The emitExpr'd arg value must NOT be used — its v
          // string re-embeds the index expression, so a side-effectful index
          // (`foo(v[funcWithSideEffects()])`) would run its effects twice
          // (once in the value, once in the lvalue prelude — CTS
          // vector-dynamic-indexing "inout parameter ... with an index with
          // side effects"). The prelude already evaluated the index once.
          const readVals = lv.targets.map((tgt, c) => {
            const d = lv.dualTargets?.[c];
            return d ? { v: tgt, dx: d[0], dy: d[1] } : { v: tgt };
          });
          argTemps.push(materializeArg(convertArg(readVals, argTypes[i], paramType), env, lines));
        } else {
          argTemps.push([]); // out: no in-value
        }
        outArgs.push({ paramIndex: i, targets: lv.targets, dualTargets: lv.dualTargets ?? null, copyBack: lv.copyBack });
      } else {
        // in (or const-in): read the value at this arg position.
        argTemps.push(materializeArg(conv, env, lines));
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
          bindLines.push(`${store}[${lv.scratchBase} + ${c}] = ${temps[c].v};`);
        }
        continue;
      }
      const names = lv.compNames!;
      if (storage === 'out') {
        // 'out': no in-value — declare uninitialized. Dual mode: declare the
        // float params' `_dx`/`_dy` names too (the body's assignments write
        // the full triple; write-back reads them).
        if (env.dual && lv.dxNames) {
          const all: string[] = [];
          names.forEach((nm, c) => {
            all.push(nm);
            if (lv.dxNames![c]) all.push(lv.dxNames![c], `${nm}_dy`);
          });
          bindLines.push(`var ${all.join(', ')};`);
        } else {
          bindLines.push(`var ${names.join(', ')};`);
        }
      } else {
        // 'in' and 'inout' both carry an in-value temp from the arg phase.
        // Dual mode: bind the float params' dx/dy from the arg temp triples.
        const temps = argTemps[i];
        if (env.dual && lv.dxNames) {
          const parts: string[] = [];
          names.forEach((nm, c) => {
            parts.push(`${nm} = ${temps[c].v}`);
            if (lv.dxNames![c]) {
              parts.push(`${lv.dxNames![c]} = ${temps[c].dx ?? '0'}`);
              parts.push(`${nm}_dy = ${temps[c].dy ?? '0'}`);
            }
          });
          bindLines.push(`var ${parts.join(', ')};`);
        } else {
          bindLines.push(`var ${names.map((n, c) => `${n} = ${temps[c].v}`).join(', ')};`);
        }
      }
    }

    /* ---------- 3. return temps ---------- */
    const retType = fn.prototype.returnType.resolved ?? { kind: 'scalar', base: 'float' };
    const nRet = flatComponents(retType);
    // Dual mode: float return components get a [dx, dy] temp pair so the
    // returned triple survives the IIFE boundary.
    const retFloatness = env.dual ? flatFloatness(retType) : null;
    const retTemps: string[] = [];
    const retDualTemps: ([string, string] | null)[] = [];
    for (let c = 0; c < nRet; c++) {
      retTemps.push(env.allocTemp());
      retDualTemps.push(retFloatness && retFloatness[c] ? [env.allocTemp(), env.allocTemp()] : null);
    }

    /* ---------- 4. inlined body ---------- */
    const label = ctx.label();
    // Callee-body mode: while emitting THIS function's body, free-name
    // resolution must see only this function's frame + globals — never the
    // caller's frames (GLSL scoping; env.resolveLocal uses bodyDepth - 1 as
    // the current-function frame index). Arg materialization above stays in
    // the CALLER's scope (bodyDepth unchanged).
    env.bodyDepth++;
    let bodyLines: string[];
    try {
      bodyLines = emitStatements(fn.body.body, env, {
        retTemps,
        retDualTemps,
        epilogueLabel: label,
        retType,
      });
    } finally {
      env.bodyDepth--;
    }

    /* ---------- 5. write-backs (after the labeled block — a
     * `break EP_<n>` from a return lands right after `}` and still runs) ---- */
    const wbLines: string[] = [];
    for (const o of outArgs) {
      const lv = paramLVs[o.paramIndex];
      const n = Math.min(o.targets.length, flatComponents(lv.type));
      if (lv.kind === 'scratch') {
        const store = lv.int ? 'ctx.intScratch' : 'ctx.scratch';
        const bs = flatComponents(lv.type);
        for (let c = 0; c < n; c++) {
          // Dual mode: read the param's three scratch planes back into the
          // caller lvalue's three planes.
          if (env.dual && !lv.int && o.dualTargets?.[c]) {
            wbLines.push(
              `${env.dualWrite(o.targets[c], o.dualTargets[c], {
                v: `${store}[${lv.scratchBase} + ${c}]`,
                dx: `${store}[${lv.scratchBase} + ${bs} + ${c}]`,
                dy: `${store}[${lv.scratchBase} + ${2 * bs} + ${c}]`,
              })};`,
            );
          } else {
            wbLines.push(`${o.targets[c]} = ${store}[${lv.scratchBase} + ${c}];`);
          }
        }
      } else {
        for (let c = 0; c < n; c++) {
          // Dual mode: write the param local's full triple back to the
          // caller lvalue (dualWrite skips the planes when the caller side
          // has none — e.g. an output argument).
          if (env.dual && o.dualTargets?.[c] && lv.dxNames?.[c]) {
            wbLines.push(
              `${env.dualWrite(o.targets[c], o.dualTargets[c], {
                v: lv.compNames![c],
                dx: `${lv.compNames![c]}_dx`,
                dy: `${lv.compNames![c]}_dy`,
              })};`,
            );
          } else {
            wbLines.push(`${o.targets[c]} = ${lv.compNames![c]};`);
          }
        }
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
      for (let c = 0; c < retTemps.length; c++) {
        inner.push(`${retTemps[c]} = 0;`);
        const d = retDualTemps[c];
        if (d) {
          inner.push(`${d[0]} = 0;`);
          inner.push(`${d[1]} = 0;`);
        }
      }
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
    return retTemps.map((t, c) => {
      const d = retDualTemps[c];
      return d ? { v: t, dx: d[0], dy: d[1], pre } : { v: t, pre };
    });
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

/**
 * Materialize ONE arg's component values into temps, pushing the assignments
 * into `lines` at the exact arg position (GLSL left-to-right evaluation).
 * EVERY component is temped — even without pre — so side-effectful v-strings
 * (assignments, ++/--) run HERE, before later args (binding them at the
 * param-bind lines instead would reorder effects across args). Pres that are
 * SHARED by identity across components (a nested array-returning call's ONE
 * [iife] array on every component) run exactly ONCE: the first component
 * folds them into its temp assignment, later components read the call's
 * retTemps directly — folding a shared pre into every temp would re-run the
 * callee once per component. Dual mode: float values temp all THREE planes
 * (3 temps; the pre runs in the v line, the dx/dy strings may reference its
 * temps). The returned Values carry the temp names (no pre — the lines are
 * already in `lines`).
 */
function materializeArg(vals: Value[], env: CodegenEnv, lines: string[]): Value[] {
  const seen = new Set<string[]>();
  return vals.map((v) => {
    const p = v.pre;
    const first = !(p && p.length > 0) || !seen.has(p);
    if (p && p.length > 0) seen.add(p);
    const t = env.allocTemp();
    const vv = p && p.length > 0 ? (first ? foldPre(p, v.v) : v.v) : v.v;
    if (env.dual && v.dx !== undefined && v.dy !== undefined) {
      const td = env.allocTemp();
      const td2 = env.allocTemp();
      lines.push(`${t} = ${vv};`, `${td} = ${v.dx};`, `${td2} = ${v.dy};`);
      return { v: t, dx: td, dy: td2 };
    }
    lines.push(`${t} = ${vv};`);
    return { v: t };
  });
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
