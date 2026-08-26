/**
 * index.ts — public API of src/util, the shared low-level foundation.
 *
 * util is the BOTTOM of the dependency DAG: no module under src/ may import
 * anything except util (and sibling modules), and util imports nothing from
 * src/. All other modules import from '../util' (this barrel).
 */

export * from './math';
export * from './typedarray';
export * from './misc';
export * from './log';
