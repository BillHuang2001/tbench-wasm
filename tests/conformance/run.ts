#!/usr/bin/env node
/**
 * CLI entry point for the CTS runner.
 *
 * Usage:
 *   tsx tests/conformance/run.ts [options]
 *
 * Options:
 *   --suite <webgl1|webgl2|all|deqp>  Suites to run (default: all)
 *   --deqp                            Append the optional deqp suite (885-page
 *                                     WebGL2 suite; 793 pages in the manifest)
 *   --include-more                    Append conformance/more (53 tests) to
 *                                     webgl1/all
 *   --filter <regex>                  Run only tests whose path matches
 *   --workers <n>                     Parallel pages (default: 4)
 *   --timeout <ms>                    Idle timeout per test (default: 60000)
 *   --smoke                           Run the first 10 tests to validate plumbing
 *   --report <path>                   JSON report path (default: tests/reports/conformance.json)
 *   --renderer <path>                 Renderer bundle; overrides WEBGL_SOFTWARE_RENDERER
 *   --cts <path>                      CTS repo root (default: /testsuites/WebGL)
 *   --host <host>                     Server bind host (default: 127.0.0.1)
 *   --port <n>                        Server port, 0 = auto (default: 0)
 *   --help                            Show this help
 *
 * Exit codes: 0 = all pass, 1 = failures/timeouts/errors, 2 = runner error.
 */

import { runConformance, type SuiteSpec } from "./runner.js";

const SUITES: Record<string, SuiteSpec> = {
  webgl1: { name: "conformance", manifest: "conformance/00_test_list.txt", webglVersion: 1 },
  webgl2: { name: "conformance2", manifest: "conformance2/00_test_list.txt", webglVersion: 2 },
  deqp: { name: "deqp", manifest: "deqp/00_test_list.txt", webglVersion: 2 },
};

const MORE: SuiteSpec = { name: "more", manifest: "conformance/more/00_test_list.txt", webglVersion: 1 };

interface CliOptions {
  suite: string;
  deqp: boolean;
  includeMore: boolean;
  filter?: RegExp;
  workers: number;
  timeoutMs: number;
  smoke: boolean;
  reportPath: string;
  rendererPath?: string;
  ctsDir: string;
  host: string;
  port: number;
  help: boolean;
}

const USAGE = `Usage: tsx tests/conformance/run.ts [options]

Options:
  --suite <webgl1|webgl2|all|deqp>  Suites to run (default: all)
  --deqp                            Append the optional deqp suite
  --include-more                    Append conformance/more to webgl1/all
  --filter <regex>                  Run only tests whose path matches
  --workers <n>                     Parallel pages (default: 4)
  --timeout <ms>                    Idle timeout per test (default: 60000)
  --smoke                           Run the first 10 tests to validate plumbing
  --report <path>                   JSON report path (default: tests/reports/conformance.json)
  --renderer <path>                 Renderer bundle; overrides WEBGL_SOFTWARE_RENDERER
  --cts <path>                      CTS repo root (default: /testsuites/WebGL)
  --host <host>                     Server bind host (default: 127.0.0.1)
  --port <n>                        Server port, 0 = auto (default: 0)
  --help                            Show this help

Exit codes: 0 = all pass, 1 = failures/timeouts/errors, 2 = runner error.`;

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    suite: "all",
    deqp: false,
    includeMore: false,
    workers: 4,
    timeoutMs: 60000,
    smoke: false,
    reportPath: "tests/reports/conformance.json",
    ctsDir: process.env.CTS_DIR ?? "/testsuites/WebGL",
    host: "127.0.0.1",
    port: 0,
    help: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = (name: string): string => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`--${name} requires a value`);
      return v;
    };
    switch (arg) {
      case "--suite":
        opts.suite = next("suite");
        break;
      case "--deqp":
        opts.deqp = true;
        break;
      case "--include-more":
        opts.includeMore = true;
        break;
      case "--filter":
        opts.filter = new RegExp(next("filter"));
        break;
      case "--workers":
        opts.workers = parseInt(next("workers"), 10);
        break;
      case "--timeout":
        opts.timeoutMs = parseInt(next("timeout"), 10);
        break;
      case "--smoke":
        opts.smoke = true;
        break;
      case "--report":
        opts.reportPath = next("report");
        break;
      case "--renderer":
        opts.rendererPath = next("renderer");
        break;
      case "--cts":
        opts.ctsDir = next("cts");
        break;
      case "--host":
        opts.host = next("host");
        break;
      case "--port":
        opts.port = parseInt(next("port"), 10);
        break;
      case "--help":
      case "-h":
        opts.help = true;
        break;
      default:
        throw new Error(`unknown option: ${arg}\n\n${USAGE}`);
    }
  }

  if (!["webgl1", "webgl2", "all", "deqp"].includes(opts.suite)) {
    throw new Error(`invalid --suite '${opts.suite}' (expected webgl1|webgl2|all|deqp)`);
  }
  if (!Number.isInteger(opts.workers) || opts.workers < 1) {
    throw new Error(`invalid --workers '${opts.workers}' (expected integer >= 1)`);
  }
  if (!Number.isInteger(opts.timeoutMs) || opts.timeoutMs < 1000) {
    throw new Error(`invalid --timeout '${opts.timeoutMs}' (expected ms >= 1000)`);
  }
  if (!Number.isInteger(opts.port) || opts.port < 0 || opts.port > 65535) {
    throw new Error(`invalid --port '${opts.port}'`);
  }
  return opts;
}

function resolveSuites(opts: CliOptions): SuiteSpec[] {
  const suites: SuiteSpec[] = [];
  if (opts.suite === "all") {
    suites.push(SUITES.webgl1, SUITES.webgl2);
  } else {
    suites.push(SUITES[opts.suite]);
  }
  if (opts.includeMore && opts.suite !== "deqp") {
    suites.push(MORE);
  }
  if (opts.deqp) {
    suites.push(SUITES.deqp);
  }
  return suites;
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) {
    console.log(USAGE);
    return;
  }
  const suites = resolveSuites(opts);
  const { exitCode, report } = await runConformance({
    suites,
    ctsDir: opts.ctsDir,
    host: opts.host,
    port: opts.port,
    workers: opts.workers,
    timeoutMs: opts.timeoutMs,
    filter: opts.filter,
    smoke: opts.smoke,
    reportPath: opts.reportPath,
    rendererPath: opts.rendererPath,
  });
  process.exitCode = exitCode;
  void report;
}

main().catch((err: unknown) => {
  console.error(`conformance runner failed: ${err instanceof Error ? err.message : String(err)}`);
  if (err instanceof Error && err.stack) console.error(err.stack);
  process.exitCode = 2;
});
