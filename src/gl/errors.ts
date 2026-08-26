/**
 * src/gl/errors.ts — GL error queue.
 *
 * WebGL spec: errors are queued per context; `getError()` returns and REMOVES the
 * oldest queued error, or NO_ERROR (0) when the queue is empty. Internal exceptions
 * must never propagate to the page — API entry points catch and convert to GL errors
 * (see CONTEXT.md Constraints). This queue is the single error sink for the context.
 */

import type { GLenum } from './types';

/** Hard cap on queued errors to bound memory (spec has no cap; browsers bound ~16). */
const MAX_QUEUED_ERRORS = 256;

export class ErrorQueue {
  private queue: GLenum[] = [];

  /** Push an error if the queue isn't full. */
  push(error: GLenum): void {
    if (this.queue.length < MAX_QUEUED_ERRORS) this.queue.push(error);
  }

  /** Pop and return the oldest error, or NO_ERROR (0) if empty. */
  get(): GLenum {
    return this.queue.length > 0 ? this.queue.shift()! : 0; // 0 = NO_ERROR
  }

  /** Return the oldest error WITHOUT removing it, or NO_ERROR. */
  peek(): GLenum {
    return this.queue.length > 0 ? this.queue[0] : 0;
  }

  /** True when at least one error is queued. */
  hasErrors(): boolean {
    return this.queue.length > 0;
  }

  /** Drop all queued errors (used on context restore). */
  clear(): void {
    this.queue.length = 0;
  }
}
