/**
 * src/gl/errors.ts — GL error queue.
 *
 * WebGL spec: errors are reported via error flags per context; `getError()`
 * returns and CLEARS one flag, or NO_ERROR (0) when none are set. Browsers
 * (Chromium — verified empirically against the CTS) keep ONE flag per distinct
 * error code: repeated identical errors coalesce into a single entry, and
 * getError returns the set flags in ascending code order (INVALID_ENUM,
 * INVALID_VALUE, INVALID_OPERATION, OUT_OF_MEMORY, INVALID_FRAMEBUFFER_OPERATION).
 * This queue is the single error sink for the context.
 */

import type { GLenum } from './types';

export class ErrorQueue {
  /** Distinct queued codes in ascending order (one flag per error code). */
  private queue: GLenum[] = [];

  /** Push an error unless the same code is already queued (flag model). */
  push(error: GLenum): void {
    if (this.queue.includes(error)) return;
    // Insert keeping ascending code order (Chromium's getError return order).
    let i = 0;
    while (i < this.queue.length && this.queue[i] < error) i++;
    this.queue.splice(i, 0, error);
  }

  /** Pop and return the oldest flag, or NO_ERROR (0) if none are set. */
  get(): GLenum {
    return this.queue.length > 0 ? this.queue.shift()! : 0; // 0 = NO_ERROR
  }

  /** Return the oldest flag WITHOUT removing it, or NO_ERROR. */
  peek(): GLenum {
    return this.queue.length > 0 ? this.queue[0] : 0;
  }

  /** True when at least one error flag is set. */
  hasErrors(): boolean {
    return this.queue.length > 0;
  }

  /** Drop all queued errors (used on context restore). */
  clear(): void {
    this.queue.length = 0;
  }
}
