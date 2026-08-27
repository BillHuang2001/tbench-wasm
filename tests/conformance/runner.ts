/**
 * Orchestrator: parses suites, asserts counts, starts the CTS server(s),
 * launches headless Chromium, drives a worker pool of pages through the test
 * queue, and writes a JSON report + human summary. Exit code is non-zero
 * when any test failed, timed out, or errored.
 *
 * Concurrency model: one browser, one browser context per worker, one page
 * per test. The renderer is CPU-bound, so default concurrency is 4 pages.
 * Results are collected in-process (never scraped from the DOM at volume).
 */

import fs from "node:fs";
import path from "node:path";
import { chromium, type Browser, type BrowserContext } from "playwright";
import {
  DEFAULT_SUITE_VERSION,
  assertSuiteCount,
  loadSuite,
  type TestEntry,
} from "./list.js";
import { startCtsServer, type CtsServer } from "./server.js";
import { HARNESS_SHIM_SCRIPT, runTestPage, type TestRunResult } from "./harness.js";
import { buildInterceptScript } from "../../src/context-intercept";

export interface SuiteSpec {
  /** Key into list.ts EXPECTED_COUNTS, e.g. "conformance". */
  name: string;
  /** Manifest relative to sdk/tests, e.g. "conformance/00_test_list.txt". */
  manifest: string;
  /** webglVersion query param for every test of this suite. */
  webglVersion: 1 | 2;
}

export interface RunOptions {
  suites: SuiteSpec[];
  /** CTS repository root (directory containing sdk/tests). */
  ctsDir: string;
  host: string;
  port: number;
  workers: number;
  timeoutMs: number;
  filter?: RegExp;
  smoke: boolean;
  reportPath: string;
  /** Renderer bundle path; overrides WEBGL_SOFTWARE_RENDERER. */
  rendererPath?: string;
}

export interface TestResult extends TestRunResult {
  url: string;
  suite: string;
}

export interface SuiteSummary {
  parsed: number;
  ran: number;
  pass: number;
  fail: number;
  timeout: number;
  error: number;
}

export interface RunSummary {
  total: number;
  pass: number;
  fail: number;
  timeout: number;
  error: number;
  skippedSubtests: number;
  durationMs: number;
  perSuite: Record<string, SuiteSummary>;
}

export interface RunReport {
  meta: Record<string, unknown>;
  summary: RunSummary;
  tests: TestResult[];
}

interface Job {
  url: string;
  suite: string;
  webglVersion: 1 | 2;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export function makeTestUrl(baseUrl: string, job: Job): string {
  return `${baseUrl}sdk/tests/${job.url}?webglVersion=${job.webglVersion}`;
}

function errorResult(message: string, job?: Job): TestResult {
  return {
    status: "error",
    pass: 0,
    fail: 0,
    skip: 0,
    messages: [message],
    timeMs: 0,
    rendererMissing: false,
    rendererActive: false,
    url: job ? job.url : "",
    suite: job ? job.suite : "",
  };
}

export async function runConformance(opts: RunOptions): Promise<{ exitCode: number; report: RunReport }> {
  const startedAt = new Date();
  const startMs = Date.now();

  // ---- 1. Parse suites + assert counts (fail loudly on drift) -------------
  const parsedSuites: { spec: SuiteSpec; entries: TestEntry[] }[] = [];
  for (const spec of opts.suites) {
    const entries = loadSuite(spec.manifest, opts.ctsDir, { version: DEFAULT_SUITE_VERSION });
    assertSuiteCount(spec.name, entries.length);
    parsedSuites.push({ spec, entries });
  }

  // ---- 2. Build queue (filter, smoke) -------------------------------------
  const queue: Job[] = [];
  for (const { spec, entries } of parsedSuites) {
    for (const entry of entries) {
      if (opts.filter && !opts.filter.test(entry.url)) continue;
      queue.push({ url: entry.url, suite: spec.name, webglVersion: spec.webglVersion });
    }
  }
  if (opts.filter && queue.length === 0) {
    throw new Error(`--filter '${opts.filter}' matched no tests in the selected suites`);
  }
  if (opts.smoke) queue.splice(10);

  const server: CtsServer = await startCtsServer({ root: opts.ctsDir, host: opts.host, port: opts.port });

  // Renderer path: --renderer wins, else WEBGL_SOFTWARE_RENDERER, else the
  // context-intercept default (./renderer.js -> RENDERER_NOT_FOUND stub).
  if (opts.rendererPath !== undefined) {
    process.env.WEBGL_SOFTWARE_RENDERER = opts.rendererPath;
  }
  const interceptScript = buildInterceptScript();

  let browser: Browser | null = null;
  try {
    browser = await chromium.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    });

    const results: TestResult[] = new Array<TestResult>(queue.length);
    let nextIndex = 0;
    let browserDead = false;

    const worker = async (): Promise<void> => {
      let context: BrowserContext | null = null;
      try {
        context = await browser!.newContext();
        await context.addInitScript(interceptScript);
        await context.addInitScript(HARNESS_SHIM_SCRIPT);
        for (;;) {
          if (browserDead) return;
          const i = nextIndex++;
          if (i >= queue.length) return;
          const job = queue[i];
          try {
            const r = await runTestPage(context, makeTestUrl(server.baseUrl, job), {
              timeoutMs: opts.timeoutMs,
            });
            results[i] = { ...r, url: job.url, suite: job.suite };
          } catch (err) {
            results[i] = errorResult(`worker error: ${err instanceof Error ? err.message : String(err)}`, job);
          }
        }
      } catch (err) {
        // Browser/context failure: mark everything not yet completed as error.
        browserDead = true;
        const msg = err instanceof Error ? err.message : String(err);
        for (let i = 0; i < queue.length; i++) {
          if (!results[i]) results[i] = errorResult(`browser failure: ${msg}`, queue[i]);
        }
      } finally {
        await context?.close().catch(() => {});
      }
    };

    await Promise.all(Array.from({ length: Math.min(opts.workers, queue.length) }, () => worker()));

    const report = buildReport(opts, parsedSuites, queue, results, server, startedAt, Date.now() - startMs);
    printSummary(report, opts, server);
    fs.mkdirSync(path.dirname(path.resolve(opts.reportPath)), { recursive: true });
    fs.writeFileSync(opts.reportPath, JSON.stringify(report, null, 2) + "\n");
    console.log(`Report: ${opts.reportPath}`);

    const s = report.summary;
    return { exitCode: s.fail + s.timeout + s.error > 0 ? 1 : 0, report };
  } finally {
    await browser?.close().catch(() => {});
    await server.close().catch(() => {});
  }
}

function buildReport(
  opts: RunOptions,
  parsedSuites: { spec: SuiteSpec; entries: TestEntry[] }[],
  queue: Job[],
  results: TestResult[],
  server: CtsServer,
  startedAt: Date,
  durationMs: number,
): RunReport {
  const bySuite: Record<string, SuiteSummary> = {};
  for (const { spec } of parsedSuites) {
    bySuite[spec.name] = { parsed: spec.name === "more" ? 0 : 0, ran: 0, pass: 0, fail: 0, timeout: 0, error: 0 };
  }
  // parsed counts reflect the full parse (before filter); queue may be filtered/sliced.
  for (const { spec, entries } of parsedSuites) {
    bySuite[spec.name].parsed = entries.length;
  }
  for (const r of results) {
    const s = bySuite[r.suite];
    if (s) {
      s.ran++;
      s[r.status]++;
    }
  }

  const summary: RunSummary = {
    total: results.length,
    pass: results.filter((r) => r.status === "pass").length,
    fail: results.filter((r) => r.status === "fail").length,
    timeout: results.filter((r) => r.status === "timeout").length,
    error: results.filter((r) => r.status === "error").length,
    skippedSubtests: results.reduce((acc, r) => acc + r.skip, 0),
    durationMs,
    perSuite: bySuite,
  };

  const tests = [...results].sort((a, b) => a.url.localeCompare(b.url));

  const renderer = process.env.WEBGL_SOFTWARE_RENDERER ?? "./renderer.js";
  const report: RunReport = {
    meta: {
      startedAt: startedAt.toISOString(),
      endedAt: new Date().toISOString(),
      ctsDir: opts.ctsDir,
      server: server.baseUrl,
      host: opts.host,
      port: server.port,
      suiteVersion: DEFAULT_SUITE_VERSION,
      renderer,
      workers: opts.workers,
      timeoutMs: opts.timeoutMs,
      filter: opts.filter ? opts.filter.source : null,
      smoke: opts.smoke,
      reportPath: opts.reportPath,
    },
    summary,
    tests,
  };
  return report;
}

function printSummary(report: RunReport, opts: RunOptions, server: CtsServer): void {
  const s = report.summary;
  const suiteNames = opts.suites.map((sp) => sp.name).join(", ");
  console.log("\n=== WebGL CTS run ===");
  console.log(`Suites: ${suiteNames}`);
  for (const [name, ss] of Object.entries(s.perSuite)) {
    console.log(
      `  ${name}: parsed ${ss.parsed}, ran ${ss.ran} ` +
        `(pass ${ss.pass}, fail ${ss.fail}, timeout ${ss.timeout}, error ${ss.error})`,
    );
  }
  console.log(`Server: ${server.baseUrl} (cross-origin logo via localhost on same port)`);
  console.log(`Renderer: ${process.env.WEBGL_SOFTWARE_RENDERER ?? "./renderer.js"}`);
  console.log(`Suite version: ${report.meta.suiteVersion as string}`);
  console.log(`Workers: ${opts.workers} | Timeout: ${opts.timeoutMs}ms | Filter: ${opts.filter?.source ?? "none"} | Smoke: ${opts.smoke}`);
  console.log(`\nTotal: ${s.total} tests in ${(s.durationMs / 1000).toFixed(1)}s`);
  console.log(`  PASS    ${s.pass}`);
  console.log(`  FAIL    ${s.fail}`);
  console.log(`  TIMEOUT ${s.timeout}`);
  console.log(`  ERROR   ${s.error}`);
  console.log(`  skipped subtests: ${s.skippedSubtests}`);

  // Renderer-active gate: error paths are navigation/driver failures, not
  // "ran without the software renderer" — only non-error results count.
  const inactiveCount = report.tests.filter((t) => !t.rendererActive && t.status !== "error").length;
  if (inactiveCount > 0) {
    console.log(`WARNING: ${inactiveCount} pages ran without the software renderer (bundle likely dead)`);
  }

  const bad = report.tests.filter((t) => t.status !== "pass");
  if (bad.length > 0) {
    console.log(`\nFailures (first 20 of ${bad.length}):`);
    for (const t of bad.slice(0, 20)) {
      const detail = t.messages[0] ? ` — ${t.messages[0]}` : "";
      console.log(`  [${t.status}] ${t.url}${detail}`);
    }
  }
}