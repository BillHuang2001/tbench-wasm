/**
 * log.ts — leveled debug logging for the renderer.
 *
 * The ONLY module in util with mutable state (the global log level).
 * All other modules create namespaced loggers via `createLogger`; every
 * logger consults the CURRENT global level at call time, so `setLogLevel`
 * takes effect immediately for all existing loggers.
 *
 * Works in Node and browsers (uses `console`). When a level is disabled the
 * logger methods are no-ops. Default level: 'warn' (errors and warnings
 * visible; debug/info silent).
 */

/** Named log levels, weakest to strongest. */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'silent';

/**
 * Numeric rank per level: a message of level L is emitted iff
 * rank(L) >= rank(current level). 'silent' disables everything.
 */
export const LOG_LEVEL_RANK: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
  silent: 99,
};

/** Default level used until `setLogLevel` is called. */
export const DEFAULT_LOG_LEVEL: LogLevel = 'warn';

/** A namespaced logger bound to the global level. */
export interface Logger {
  /** Debug-level message (only when level <= 'debug'). */
  debug(...args: unknown[]): void;
  /** Info-level message (only when level <= 'info'). */
  info(...args: unknown[]): void;
  /** Warning message (only when level <= 'warn'). */
  warn(...args: unknown[]): void;
  /** Error message (only when level <= 'error'). */
  error(...args: unknown[]): void;
}

/** Current global level; consulted at call time by every logger. */
let currentLevel: LogLevel = DEFAULT_LOG_LEVEL;

/**
 * Sets the global log level. Affects all existing and future loggers.
 */
export function setLogLevel(level: LogLevel): void {
  currentLevel = level;
}

/**
 * Returns the current global log level.
 */
export function getLogLevel(): LogLevel {
  return currentLevel;
}

/**
 * True iff messages at `level` are currently emitted.
 */
export function isLogEnabled(level: LogLevel): boolean {
  return LOG_LEVEL_RANK[level] >= LOG_LEVEL_RANK[currentLevel];
}

/**
 * Creates a namespaced logger that prefixes messages with `[namespace]`.
 * The logger's methods no-op when their level is below the current global
 * level. Cheap to call; create once per module at module scope.
 */
export function createLogger(namespace: string): Logger {
  const emit = (method: 'debug' | 'info' | 'warn' | 'error', level: LogLevel, args: unknown[]): void => {
    // Read `currentLevel` at call time so setLogLevel affects existing loggers.
    if (LOG_LEVEL_RANK[level] >= LOG_LEVEL_RANK[currentLevel]) {
      (console[method] as (...a: unknown[]) => void)(`[${namespace}]`, ...args);
    }
  };
  return {
    debug: (...args: unknown[]) => emit('debug', 'debug', args),
    info: (...args: unknown[]) => emit('info', 'info', args),
    warn: (...args: unknown[]) => emit('warn', 'warn', args),
    error: (...args: unknown[]) => emit('error', 'error', args),
  };
}
