#!/usr/bin/env node
/**
 * Standalone diagnostic: reruns specific CTS pages against the software
 * renderer and dumps in-page failure messages for root-cause extraction.
 *
 * The graded runner (run.ts/runner.ts/harness.ts/server.ts/list.ts) stays
 * pristine — this tool only imports its exports (HARNESS_SHIM_SCRIPT,
 * startCtsServer, buildInterceptScript) and mirrors runner.ts wiring: env
 * WEBGL_SOFTWARE_RENDERER set before buildInterceptScript(), one browser
 * context per worker, intercept + harness shim added per context, activity-
 * based idle timeout with a 250ms in-page snapshot poll (REAL function, never
 * a string). Exit code is always 0 (diagnostic tool; failures are printed).
 */

import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { startCtsServer, type CtsServer } from "./server.js";
import { HARNESS_SHIM_SCRIPT } from "./harness.js";
import { buildInterceptScript } from "../../src/context-intercept";

interface Options {
  urls: string[];
  rendererPath: string;
  repeat: number;
  workers: number;
  timeoutMs: number;
}

const USAGE = `Usage: tsx tests/conformance/debug-page.ts <url> [<url> ...] [options]

Reruns exact relative CTS paths (e.g. conformance2/extensions/webgl-clip-cull-distance.html)
against the software renderer and dumps in-page failure messages.

Options:
  --renderer <path>   Renderer bundle (default: ./renderer.js)
  --repeat <n>        Runs per URL (default: 1)
  --workers <n>       Parallel pages (default: 4)
  --timeout-ms <ms>   Idle timeout per run (default: 180000)
  --help              Show this help`;

function parseArgs(argv: string[]): Options {
  const opts: Options = { urls: [], rendererPath: "./renderer.js", repeat: 1, workers: 4, timeoutMs: 180000 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = (name: string): string => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`--${name} requires a value`);
      return v;
    };
    switch (arg) {
      case "--renderer":
        opts.rendererPath = next("renderer");
        break;
      case "--repeat":
        opts.repeat = parseInt(next("repeat"), 10);
        break;
      case "--workers":
        opts.workers = parseInt(next("workers"), 10);
        break;
      case "--timeout-ms":
        opts.timeoutMs = parseInt(next("timeout-ms"), 10);
        break;
      case "--help":
      case "-h":
        console.log(USAGE);
        process.exit(0);
        break;
      default:
        if (arg.startsWith("-")) throw new Error(`unknown option: ${arg}\n\n${USAGE}`);
        opts.urls.push(arg);
    }
  }
  if (opts.urls.length === 0) throw new Error(`no URLs given\n\n${USAGE}`);
  if (opts.repeat < 1 || opts.workers < 1 || opts.timeoutMs < 1000) {
    throw new Error("invalid option value (repeat >= 1, workers >= 1, timeout-ms >= 1000)");
  }
  return opts;
}

interface Snapshot {
  pass: number;
  fail: number;
  skip: number;
  finished: boolean;
  title: string;
  description: string;
  rendererActive: boolean;
}

/** Must be a real function (never a string) so Playwright serializes and calls it. */
const pollSnapshot = (): Snapshot => {
  const s = (window as unknown as { __softglCts?: { pass?: number; fail?: number; skip?: number; finished?: boolean } }).__softglCts;
  const d = document.getElementById("description");
  return {
    pass: s && typeof s.pass === "number" ? s.pass : 0,
    fail: s && typeof s.fail === "number" ? s.fail : 0,
    skip: s && typeof s.skip === "number" ? s.skip : 0,
    finished: s ? s.finished === true : false,
    title: document.title || "",
    description: d && d.textContent ? d.textContent.trim() : "",
    rendererActive:
      typeof (window as unknown as { __createSoftwareWebGLContext?: unknown }).__createSoftwareWebGLContext === "function",
  };
};

interface Dump {
  messages: string[];
  description: string;
  rendererActive: boolean;
  title: string;
}

const dumpState = (): Dump => {
  const s = (window as unknown as { __softglCts?: { messages?: string[] } }).__softglCts;
  const d = document.getElementById("description");
  return {
    messages: s && Array.isArray(s.messages) ? s.messages : [],
    description: d && d.innerText ? d.innerText : "",
    rendererActive:
      typeof (window as unknown as { __createSoftwareWebGLContext?: unknown }).__createSoftwareWebGLContext === "function",
    title: document.title || "",
  };
};

const COMPLETE_RE = /TEST COMPLETE:\s*\d+\s*PASS,\s*\d+\s*FAIL/;
const POLL_INTERVAL_MS = 250;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** FAIL lines from #description innerText: deduped, no TEST COMPLETE line, cap 30, each <= 300 chars. */
function extractFailLines(description: string): string[] {
  const seen = new Set<string>();
  const lines: string[] = [];
  for (const raw of description.split("\n")) {
    const line = raw.trim();
    if (!line.includes("FAIL") || COMPLETE_RE.test(line) || seen.has(line)) continue;
    seen.add(line);
    lines.push(line.slice(0, 300));
    if (lines.length >= 30) break;
  }
  return lines;
}

async function runPage(
  context: BrowserContext,
  url: string,
  k: number,
  repeat: number,
  baseUrl: string,
  timeoutMs: number,
): Promise<void> {
  const page: Page = await context.newPage();
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  page.on("pageerror", (err) => {
    const stack = err instanceof Error && err.stack ? `\n${err.stack.split("\n").slice(0, 12).join("\n")}` : "";
    pageErrors.push(`${errMsg(err)}${stack}`);
  });
  page.on("console", (msg) => {
    if (msg.type() === "error" && consoleErrors.length < 10) consoleErrors.push(msg.text().slice(0, 300));
  });

  let timedOut = false;
  try {
    await page.goto(`${baseUrl}sdk/tests/${url}?webglVersion=2`, { waitUntil: "domcontentloaded", timeout: timeoutMs });

    let lastActivity = Date.now();
    let lastKey = "";
    let snap: Snapshot = { pass: 0, fail: 0, skip: 0, finished: false, title: "", description: "", rendererActive: false };
    for (;;) {
      snap = await page.evaluate(pollSnapshot);
      const now = Date.now();
      if (snap.finished || COMPLETE_RE.test(snap.description) || snap.title === "RENDERER_NOT_FOUND") break;
      const key = `${snap.pass},${snap.fail},${snap.skip},${snap.description}`;
      if (key !== lastKey) {
        lastActivity = now;
        lastKey = key;
      }
      if (now - lastActivity > timeoutMs) {
        timedOut = true;
        break;
      }
      await sleep(POLL_INTERVAL_MS);
    }

    const dump = await page.evaluate(dumpState);
    console.log(`=== ${url} (run ${k}/${repeat}) ===`);
    console.log(`P/F/S: ${snap.pass}P/${snap.fail}F/${snap.skip}S rendererActive=${snap.rendererActive}`);
    console.log(`title: ${dump.title}`);
    console.log("shim messages:");
    const msgs = dump.messages.map((m) => m.slice(0, 500));
    if (msgs.length === 0) console.log("  (none)");
    else for (const m of msgs) console.log(`  - ${m}`);
    console.log("FAIL lines:");
    const failLines = extractFailLines(dump.description);
    if (failLines.length === 0) console.log("  (none)");
    else for (const l of failLines) console.log(`  - ${l}`);
    if (pageErrors.length > 0) console.log(`pageerror: ${pageErrors[0]}`);
    if (consoleErrors.length > 0) console.log(`console errors: ${consoleErrors.join(" | ")}`);
    if (timedOut) console.log("TIMEOUT");
  } catch (err) {
    console.log(`=== ${url} (run ${k}/${repeat}) ===`);
    console.log(`run error: ${errMsg(err)}`);
    if (pageErrors.length > 0) console.log(`pageerror: ${pageErrors[0]}`);
    if (consoleErrors.length > 0) console.log(`console errors: ${consoleErrors.join(" | ")}`);
  } finally {
    await page.close().catch(() => {});
  }
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));

  // Renderer path: --renderer wins, else ./renderer.js. Set env BEFORE
  // buildInterceptScript() — it reads WEBGL_SOFTWARE_RENDERER at call time.
  process.env.WEBGL_SOFTWARE_RENDERER = opts.rendererPath;
  const interceptScript = buildInterceptScript();

  const server: CtsServer = await startCtsServer({
    root: process.env.CTS_DIR || "/testsuites/WebGL",
    host: "127.0.0.1",
    port: 0,
  });

  const jobs: { url: string; k: number }[] = [];
  for (const url of opts.urls) {
    for (let k = 1; k <= opts.repeat; k++) jobs.push({ url, k });
  }

  const browser: Browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
  });

  try {
    let nextIndex = 0;
    const worker = async (): Promise<void> => {
      const context: BrowserContext = await browser.newContext();
      await context.addInitScript(interceptScript);
      await context.addInitScript(HARNESS_SHIM_SCRIPT);
      try {
        for (;;) {
          const i = nextIndex++;
          if (i >= jobs.length) return;
          await runPage(context, jobs[i].url, jobs[i].k, opts.repeat, server.baseUrl, opts.timeoutMs);
        }
      } finally {
        await context.close().catch(() => {});
      }
    };
    await Promise.all(Array.from({ length: Math.min(opts.workers, jobs.length) }, () => worker()));
  } finally {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  }
}

main().catch((err: unknown) => {
  console.error(`debug-page failed: ${errMsg(err)}`);
  if (err instanceof Error && err.stack) console.error(err.stack);
  // Diagnostic tool: exit 0 regardless of failures.
  process.exitCode = 0;
});
