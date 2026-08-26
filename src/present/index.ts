/**
 * index.ts — present/ public API.
 *
 * Re-exports the canvas adapters (CanvasSurface) and image-source decoding
 * used by gl/ (drawing-buffer presentation, texImage2D DOM sources) and
 * entry.ts (canvas → surface wiring). See ./CONTEXT.md.
 */
export * from './canvas';
export * from './image';
