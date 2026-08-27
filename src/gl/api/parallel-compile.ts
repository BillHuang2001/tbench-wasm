/**
 * src/gl/api/parallel-compile.ts — KHR_parallel_shader_compile async engine.
 *
 * Generic chunked-queue scheduler for DEFERRED shader compilation and program
 * linking. api/programs.ts builds the per-item step closures (capturing its
 * module-private state: shaderResults, programModels, linkGen, ...) and
 * enqueues them here. This module runs EXACTLY ONE step per setTimeout(0)
 * macrotask — never collapsing a multi-step item into a single task — so the
 * browser renders animation frames between chunks. The CTS page
 * (conformance/extensions/khr-parallel-shader-compile.html) depends on this:
 * its poll loop must observe MORE THAN 6 animation frames of
 * COMPLETION_STATUS_KHR == false before it flips true (a shader compile is 4
 * chunks, a link 1 chunk → 9 macrotask boundaries per program), and its
 * 256-program drain loop relies on the queue progressing on its own.
 *
 * Contract with api/programs.ts:
 *  - enqueueShaderCompile / enqueueProgramLink: register a pending item.
 *    Enqueueing for a key that already has a pending item REPLACES it (latest
 *    wins — the old item is removed from the queue and its intermediates are
 *    dropped). Stale items are also detected at pop time (safety net).
 *  - isShaderPending / isProgramPending: the COMPLETION_STATUS_KHR answer
 *    (true == no pending work).
 *  - cancelPendingShader: called by shaderSource (a new source invalidates the
 *    in-flight compile).
 *  - ensureShaderCompiled / ensureProgramLinked: SYNCHRONOUS triggers — run
 *    the item's remaining steps to completion and finalize. Every
 *    non-COMPLETION_STATUS_KHR query, draw, uniform write and debug-shaders
 *    read goes through these so the page observes real results.
 *
 * Runner guards: a popped item whose context is lost or whose object was
 * invalidated by context loss is dropped WITHOUT executing (KHR spec: status
 * reads return true while lost — the pending work is moot). Items are NOT
 * dropped on _deleted (shader deletion is deferred while attached; the
 * compile/link simply completes).
 *
 * Cycle constraint (IMPORTANT): this module imports NOTHING from extensions/
 * or api/programs.ts — programs.ts imports from here and extensions/misc.ts
 * imports from here. Steps and error finalizers are supplied by the caller;
 * glsl stage functions are imported by programs.ts, not here.
 */

import type { WebGLRenderingContext } from '../webgl1';
import type { WebGLShader, WebGLProgram } from '../objects';

/** One deferred unit of work: `steps` run in order, one per macrotask. */
export interface PendingItem {
  /** The WebGLShader or WebGLProgram this item produces state for. */
  key: object;
  /** 'shader' | 'program' — selects the pending map (no class imports needed). */
  kind: 'shader' | 'program';
  /** Step closures; the LAST step (or an early `complete`) finalizes. */
  steps: ReadonlyArray<() => void>;
  /** Index of the next step to run. */
  idx: number;
  /** Set by a step that finalizes early (e.g. a stage failed); runner removes the item. */
  complete: boolean;
  /** Exception finalizer: mirrors the sync paths' `internal compiler error:` handling. */
  onError?: (e: unknown) => void;
}

/** Per-context queue + scheduler state. */
interface QueueState {
  items: PendingItem[];
  scheduled: boolean;
}

const queueStates = new WeakMap<WebGLRenderingContext, QueueState>();
const pendingShaders = new WeakMap<WebGLShader, PendingItem>();
const pendingPrograms = new WeakMap<WebGLProgram, PendingItem>();

function stateFor(ctx: WebGLRenderingContext): QueueState {
  let st = queueStates.get(ctx);
  if (st === undefined) {
    st = { items: [], scheduled: false };
    queueStates.set(ctx, st);
  }
  return st;
}

/** Schedule the next chunk as a FRESH setTimeout(0) macrotask (chain). */
function scheduleRun(ctx: WebGLRenderingContext): void {
  const st = stateFor(ctx);
  if (st.scheduled) return;
  st.scheduled = true;
  setTimeout(() => runChunk(ctx), 0);
}

/** Remove an item from the queue and (if still current) from its pending map. */
function removeItem(ctx: WebGLRenderingContext, item: PendingItem): void {
  const st = queueStates.get(ctx);
  if (st !== undefined) {
    const i = st.items.indexOf(item);
    if (i >= 0) st.items.splice(i, 1);
  }
  if (item.kind === 'shader') {
    if (pendingShaders.get(item.key as WebGLShader) === item) pendingShaders.delete(item.key as WebGLShader);
  } else {
    if (pendingPrograms.get(item.key as WebGLProgram) === item) pendingPrograms.delete(item.key as WebGLProgram);
  }
}

/** Execute ONE step (guarded: a glsl stage exception finalizes as a failure). */
function runStep(ctx: WebGLRenderingContext, item: PendingItem): void {
  try {
    item.steps[item.idx]();
  } catch (e) {
    // A glsl stage (or finalize) threw — never allowed to reach the page.
    // Mirror the synchronous paths' try/catch: finalize as a compile/link
    // failure with a diagnostic info log.
    if (item.onError !== undefined) {
      try {
        item.onError(e);
      } catch {
        /* the finalizer itself must not throw */
      }
    }
    item.complete = true;
  }
  item.idx++;
}

/** The scheduler: pop the FIRST item, run ONE step, then chain the next chunk. */
function runChunk(ctx: WebGLRenderingContext): void {
  const st = queueStates.get(ctx);
  if (st === undefined) return;
  st.scheduled = false;
  const item = st.items[0];
  if (item === undefined) return;
  // Guard: context lost or object invalidated by loss → drop without executing.
  if (ctx._isLost || (item.key as { _invalidated?: boolean })._invalidated) {
    removeItem(ctx, item);
    scheduleRun(ctx);
    return;
  }
  // Stale item: a newer compile/link replaced this one (latest wins).
  const current = item.kind === 'shader'
    ? pendingShaders.get(item.key as WebGLShader)
    : pendingPrograms.get(item.key as WebGLProgram);
  if (current !== item) {
    removeItem(ctx, item);
    scheduleRun(ctx);
    return;
  }
  runStep(ctx, item);
  if (item.complete || item.idx >= item.steps.length) removeItem(ctx, item);
  scheduleRun(ctx);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Register a deferred shader compile (replaces any pending compile of the same shader). */
export function enqueueShaderCompile(
  ctx: WebGLRenderingContext,
  shader: WebGLShader,
  steps: ReadonlyArray<() => void>,
  onError?: (e: unknown) => void,
): PendingItem {
  const item: PendingItem = { key: shader, kind: 'shader', steps, idx: 0, complete: false, onError };
  const prev = pendingShaders.get(shader);
  if (prev !== undefined) removeItem(ctx, prev);
  pendingShaders.set(shader, item);
  stateFor(ctx).items.push(item);
  scheduleRun(ctx);
  return item;
}

/** Register a deferred program link (replaces any pending link of the same program). */
export function enqueueProgramLink(
  ctx: WebGLRenderingContext,
  program: WebGLProgram,
  steps: ReadonlyArray<() => void>,
  onError?: (e: unknown) => void,
): PendingItem {
  const item: PendingItem = { key: program, kind: 'program', steps, idx: 0, complete: false, onError };
  const prev = pendingPrograms.get(program);
  if (prev !== undefined) removeItem(ctx, prev);
  pendingPrograms.set(program, item);
  stateFor(ctx).items.push(item);
  scheduleRun(ctx);
  return item;
}

/** COMPLETION_STATUS_KHR for a shader: true iff no deferred compile is in flight. */
export function isShaderPending(shader: WebGLShader): boolean {
  return pendingShaders.has(shader);
}

/** COMPLETION_STATUS_KHR for a program: true iff no deferred link is in flight. */
export function isProgramPending(program: WebGLProgram): boolean {
  return pendingPrograms.has(program);
}

/** Drop the in-flight deferred compile of a shader (shaderSource). */
export function cancelPendingShader(ctx: WebGLRenderingContext, shader: WebGLShader): void {
  const item = pendingShaders.get(shader);
  if (item !== undefined) removeItem(ctx, item);
}

/** Drop the in-flight deferred link of a program (not used by the API today; kept symmetric). */
export function cancelPendingProgram(ctx: WebGLRenderingContext, program: WebGLProgram): void {
  const item = pendingPrograms.get(program);
  if (item !== undefined) removeItem(ctx, item);
}

/**
 * SYNCHRONOUS trigger: finish the deferred compile of `shader` now, exactly as
 * the async chunk path would (same steps, same finalize). No-op when the
 * shader has no pending compile (already completed / never deferred).
 */
export function ensureShaderCompiled(ctx: WebGLRenderingContext, shader: WebGLShader): void {
  const item = pendingShaders.get(shader);
  if (item === undefined) return;
  while (item.idx < item.steps.length && !item.complete) {
    runStep(ctx, item);
  }
  removeItem(ctx, item);
}

/**
 * SYNCHRONOUS trigger: finish the deferred link of `program` now. The link
 * step itself first forces the snapshot shaders' pending compiles to
 * completion (ordering guarantee: program-true implies shader-true), then
 * links. No-op when the program has no pending link.
 */
export function ensureProgramLinked(ctx: WebGLRenderingContext, program: WebGLProgram): void {
  const item = pendingPrograms.get(program);
  if (item === undefined) return;
  while (item.idx < item.steps.length && !item.complete) {
    runStep(ctx, item);
  }
  removeItem(ctx, item);
}
