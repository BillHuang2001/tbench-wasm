#!/usr/bin/env node
/**
 * scripts/diag-page.ts — diagnostic CLI: open ONE CTS test page with the
 * software renderer injected and dump the page's full failure detail.
 *
 * Key output: the #description DOM node (js-test-post.js appends the
 * "TEST COMPLETE: N PASS, M FAIL" summary there) plus the #console div, which
 * js-test-pre.js fills with every per-subtest PASS/FAIL line (including
 * "FAIL ... (expected: ...)" messages and shader info logs). Console errors
 * and page errors are collected alongside.
 *
 * Usage:
 *   npx tsx scripts/diag-page.ts <test-path> [options]
 *
 *   <test-path>    Test page relative to sdk/tests, e.g.
 *                  conformance/glsl/bugs/angle-ambiguous-function-call.html
 *                  (use conformance2/... for WebGL2 pages)
 *
 * Options:
 *   --webglVersion <1|2>  Context version query param (default: 1)
 *   --renderer <path>     Renderer bundle (default: ./renderer.js relative to
 *                         the repo root). The WEBGL_SOFTWARE_RENDERER env var
 *                         is NEVER consulted (it may be poisoned); the chosen
 *                         path is set explicitly before injection.
 *   --cts <path>          CTS repository root (default: /testsuites/WebGL)
 *   --timeout <ms>        Max wall-clock time for the page (default: 60000)
 *   --full                Print the complete #description text (default:
 *                         truncated to 8000 chars)
 *   --help                Show this help
 *
 * Exit codes: 0 = PASS, 1 = FAIL/TIMEOUT/ERROR, 2 = usage error.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { startCtsServer, type CtsServer } from "../tests/conformance/server.js";
import { parseDomComplete } from "../tests/conformance/harness.js";
import { buildInterceptScript } from "../src/context-intercept";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const USAGE = `Usage: npx tsx scripts/diag-page.ts <test-path> [options]

Opens one CTS test page with the software renderer injected and prints the
full failure detail from #description (the div js-test-pre.js fills with
per-subtest failure lines), plus console errors and page errors.

Arguments:
  <test-path>      Test page relative to sdk/tests, e.g.
                   conformance/glsl/bugs/angle-ambiguous-function-call.html
                   (use conformance2/... for WebGL2 pages)

Options:
  --webglVersion <1|2>  Context version query param (default: 1)
  --renderer <path>     Renderer bundle (default: ./renderer.js relative to
                        the repo root; WEBGL_SOFTWARE_RENDERER is NEVER used)
  --cts <path>          CTS repository root (default: /testsuites/WebGL)
  --timeout <ms>        Max wall-clock time for the page (default: 60000)
  --full                Print the complete #description text (default:
                        truncated to 8000 chars)
  --help                Show this help

Exit codes: 0 = PASS, 1 = FAIL/TIMEOUT/ERROR, 2 = usage error.`;

interface DiagOptions {
  testPath: string;
  webglVersion: 1 | 2;
  /** Absolute path to the renderer bundle. */
  rendererPath: string;
  /** Absolute path to the CTS repository root. */
  ctsDir: string;
  timeoutMs: number;
  full: boolean;
}

function parseArgs(argv: string[]): DiagOptions | { help: true } {
  let testPath: string | null = null;
  let webglVersion: 1 | 2 = 1;
  let rendererPath = path.resolve(REPO_ROOT, "renderer.js");
  let ctsDir = "/testsuites/WebGL";
  let timeoutMs = 60000;
  let full = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") return { help: true };
    const eq = arg.startsWith("--") ? arg.indexOf("=") : -1;
    const name = eq !== -1 ? arg.slice(0, eq) : arg;
    const inline = eq !== -1 ? arg.slice(eq + 1) : undefined;
    const value = (optName: string): string => {
      if (inline !== undefined) return inline;
      const v = argv[++i];
      if (v === undefined) throw new Error(`--${optName} requires a value\n\n${USAGE}`);
      return v;
    };
    switch (name) {
      case "--webglVersion": {
        const v = value("webglVersion");
        if (v !== "1" && v !== "2") throw new Error(`invalid --webglVersion '${v}' (expected 1 or 2)`);
        webglVersion = v === "2" ? 2 : 1;
        break;
      }
      case "--renderer":
        rendererPath = path.resolve(value("renderer"));
        break;
      case "--cts":
        ctsDir = path.resolve(value("cts"));
        break;
      case "--timeout": {
        const n = parseInt(value("timeout"), 10);
        if (!Number.isInteger(n) || n < 1000) {
          throw new Error(`invalid --timeout '${argv[i]}' (expected ms >= 1000)`);
        }
        timeoutMs = n;
        break;
      }
      case "--full":
        full = true;
        break;
      default:
        if (arg.startsWith("-")) throw new Error(`unknown option: ${arg}\n\n${USAGE}`);
        if (testPath !== null) throw new Error(`unexpected extra argument: ${arg}\n\n${USAGE}`);
        testPath = arg;
        break;
    }
  }

  if (testPath === null) throw new Error(USAGE);
  if (testPath.includes("..")) throw new Error(`invalid test path (no '..' allowed): ${testPath}`);
  return { testPath, webglVersion, rendererPath, ctsDir, timeoutMs, full };
}

/**
 * In-page harness shim. Injected via addInitScript BEFORE page scripts:
 * js-test-pre.js caches `window.parent.webglTestHarness` at load time
 * (standalone page: window.parent === window), so the shim must exist before
 * the page's scripts run. Mirrors tests/conformance/harness.ts's
 * HARNESS_SHIM_SCRIPT (same pass/fail/skip aggregation), and additionally
 * sets `_finished` on the harness object itself — the completion flag this
 * tool polls for.
 */
const DIAG_SHIM_SCRIPT: string = `
(function() {
  if (window.__softglDiag) { return; }
  var state = { pass: 0, fail: 0, skip: 0, finished: false };
  window.__softglDiag = state;
  window.webglTestHarness = {
    reportResults: function(url, success, msg, skipped) {
      if (skipped) {
        state.skip += 1;
      } else if (success) {
        state.pass += 1;
      } else {
        state.fail += 1;
      }
    },
    notifyFinished: function(url) {
      state.finished = true;
      this._finished = true;
      window.webglTestHarness._finished = true;
    }
  };
})();
`;

interface DiagPageState {
  pass: number;
  fail: number;
  skip: number;
  finished: boolean;
}

interface DiagSnapshot {
  pass: number;
  fail: number;
  skip: number;
  finished: boolean;
  title: string;
  description: string;
  /** #console text with <br> rendered as newlines (per-subtest PASS/FAIL/SKIP lines). */
  consoleText: string;
  rendererActive: boolean;
}

/**
 * Must be a real function (not a string) so Playwright serializes and calls it.
 * IMPORTANT: the body must contain NO inner function definitions — tsx/esbuild
 * (keepNames) emits `__name(...)` helper statements into the enclosing scope
 * for any inner function, and that helper does not exist when Playwright
 * re-evaluates the serialized source in the page. Everything is inlined.
 */
const pollSnapshot = (): DiagSnapshot => {
  const s = (window as unknown as { __softglDiag?: DiagPageState }).__softglDiag;
  const h = (window as unknown as { webglTestHarness?: { _finished?: boolean } }).webglTestHarness;
  const d = document.getElementById("description");
  const c = document.getElementById("console");
  // #console text with <br> rendered as "\n" (iterative DOM walk; the per-
  // subtest PASS/FAIL/SKIP lines live in #console via js-test-pre _addSpan).
  let consoleText = "";
  if (c) {
    const stack: Node[] = [c];
    while (stack.length > 0) {
      const node = stack.pop() as Node;
      if (node.nodeType === Node.TEXT_NODE) {
        consoleText += node.nodeValue ?? "";
      } else if ((node as Element).nodeName === "BR") {
        consoleText += "\n";
      } else {
        let child = node.lastChild;
        while (child) {
          stack.push(child);
          child = child.previousSibling;
        }
      }
    }
  }
  return {
    pass: s ? s.pass : 0,
    fail: s ? s.fail : 0,
    skip: s ? s.skip : 0,
    finished: (s ? s.finished : false) || Boolean(h && h._finished),
    title: document.title || "",
    description: d && d.textContent ? d.textContent : "",
    consoleText,
    rendererActive:
      typeof (window as unknown as { __createSoftwareWebGLContext?: unknown }).__createSoftwareWebGLContext ===
      "function",
  };
};

type DiagStatus = "PASS" | "FAIL" | "TIMEOUT" | "ERROR";

interface DiagResult {
  status: DiagStatus;
  url: string;
  rendererActive: boolean;
  pass: number;
  fail: number;
  skip: number;
  description: string;
  /** Per-subtest PASS/FAIL/SKIP lines from the #console div. */
  consoleText: string;
  consoleErrors: string[];
  pageErrors: string[];
  timeMs: number;
  note?: string;
}

const MAX_COLLECTED = 30;
const MAX_DESCRIPTION_DEFAULT = 8000;
const POLL_INTERVAL_MS = 500;

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function diagnose(opts: DiagOptions): Promise<DiagResult> {
  const server: CtsServer = await startCtsServer({ root: opts.ctsDir, host: "127.0.0.1", port: 0 });
  const url = `${server.baseUrl}sdk/tests/${opts.testPath}?webglVersion=${opts.webglVersion}`;
  const started = Date.now();

  // Explicit override: the renderer path is the CLI value (or the repo-root
  // default ./renderer.js). WEBGL_SOFTWARE_RENDERER is never consulted.
  process.env.WEBGL_SOFTWARE_RENDERER = opts.rendererPath;
  const interceptScript = buildInterceptScript();

  let browser: Browser | null = null;
  let context: BrowserContext | null = null;
  try {
    browser = await chromium.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    });
    context = await browser.newContext();
    await context.addInitScript(interceptScript);
    await context.addInitScript(DIAG_SHIM_SCRIPT);

    const page: Page = await context.newPage();
    const consoleErrors: string[] = [];
    const pageErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error" && consoleErrors.length < MAX_COLLECTED) {
        consoleErrors.push(msg.text());
      }
    });
    page.on("pageerror", (err) => {
      if (pageErrors.length >= MAX_COLLECTED) return;
      const stack = err instanceof Error && err.stack ? `\n${err.stack.split("\n").slice(0, 12).join("\n")}` : "";
      pageErrors.push(`${errMsg(err)}${stack}`);
    });

    try {
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: opts.timeoutMs });
    } catch (err) {
      return {
        status: "ERROR",
        url,
        rendererActive: false,
        pass: 0,
        fail: 0,
        skip: 0,
        description: "",
        consoleText: "",
        consoleErrors,
        pageErrors,
        timeMs: Date.now() - started,
        note: `navigation failed: ${errMsg(err)}`,
      };
    }

    let snapshot: DiagSnapshot | null = null;
    let timedOut = false;
    let driverError: string | null = null;
    const deadline = Date.now() + opts.timeoutMs;
    for (;;) {
      try {
        snapshot = await page.evaluate(pollSnapshot);
      } catch (err) {
        driverError = `page driver error: ${errMsg(err)}`;
        break;
      }
      // Completion: harness shim flag OR the js-test-post.js DOM fallback
      // ("TEST COMPLETE: N PASS, M FAIL" appended to #description).
      if (snapshot.finished || parseDomComplete(snapshot.description)) break;
      if (Date.now() >= deadline) {
        timedOut = true;
        break;
      }
      await sleep(POLL_INTERVAL_MS);
    }

    if (driverError) {
      return {
        status: "ERROR",
        url,
        rendererActive: false,
        pass: 0,
        fail: 0,
        skip: 0,
        description: "",
        consoleText: "",
        consoleErrors,
        pageErrors,
        timeMs: Date.now() - started,
        note: driverError,
      };
    }
    if (timedOut || !snapshot) {
      return {
        status: "TIMEOUT",
        url,
        rendererActive: snapshot ? snapshot.rendererActive : false,
        pass: snapshot ? snapshot.pass : 0,
        fail: snapshot ? snapshot.fail : 0,
        skip: snapshot ? snapshot.skip : 0,
        description: snapshot ? snapshot.description : "",
        consoleText: snapshot ? snapshot.consoleText : "",
        consoleErrors,
        pageErrors,
        timeMs: Date.now() - started,
        note: `timed out after ${opts.timeoutMs}ms; partial results below`,
      };
    }

    if (snapshot.title === "RENDERER_NOT_FOUND") {
      return {
        status: "FAIL",
        url,
        rendererActive: false,
        pass: snapshot.pass,
        fail: snapshot.fail + 1,
        skip: snapshot.skip,
        description: snapshot.description,
        consoleText: snapshot.consoleText,
        consoleErrors,
        pageErrors,
        timeMs: Date.now() - started,
        note: "RENDERER_NOT_FOUND: the injected renderer bundle failed to load",
      };
    }

    // Completed: prefer the harness shim counts; fall back to the DOM summary.
    const dom = parseDomComplete(snapshot.description);
    const pass = dom ? dom.pass : snapshot.pass;
    const fail = dom ? dom.fail : snapshot.fail;
    const rendererInactive = !snapshot.rendererActive;
    const note = rendererInactive
      ? "renderer NOT active (window.__createSoftwareWebGLContext missing) — page ran WITHOUT the software renderer"
      : dom
        ? "completion via DOM fallback (#description 'TEST COMPLETE' line)"
        : "completion via harness shim (reportResults/notifyFinished)";
    return {
      status: fail > 0 || rendererInactive ? "FAIL" : "PASS",
      url,
      rendererActive: snapshot.rendererActive,
      pass,
      fail,
      skip: snapshot.skip,
      description: snapshot.description,
      consoleText: snapshot.consoleText,
      consoleErrors,
      pageErrors,
      timeMs: Date.now() - started,
      note,
    };
  } finally {
    await context?.close().catch(() => {});
    await browser?.close().catch(() => {});
    await server.close().catch(() => {});
  }
}

function printReport(r: DiagResult, full: boolean): void {
  console.log("=== CTS page diagnostic ===");
  console.log(`URL:      ${r.url}`);
  console.log(`Status:   ${r.status}${r.note ? ` — ${r.note}` : ""}`);
  console.log(`Renderer: ${r.rendererActive ? "active (window.__createSoftwareWebGLContext present)" : "NOT active"}`);
  console.log(`Subtests: pass ${r.pass}, fail ${r.fail}, skip ${r.skip}`);
  console.log(`Time:     ${r.timeMs}ms`);
  console.log("");
  if (r.description.length === 0) {
    console.log("--- #description: (empty) ---");
  } else if (full || r.description.length <= MAX_DESCRIPTION_DEFAULT) {
    console.log(`--- #description (${r.description.length} chars) ---`);
    console.log(r.description);
  } else {
    console.log(
      `--- #description (${r.description.length} chars; first ${MAX_DESCRIPTION_DEFAULT} shown — use --full for the complete text) ---`,
    );
    console.log(r.description.slice(0, MAX_DESCRIPTION_DEFAULT));
  }
  console.log("");
  if (r.consoleText.length === 0) {
    console.log("--- #console (per-subtest detail): (empty) ---");
  } else if (full || r.consoleText.length <= MAX_DESCRIPTION_DEFAULT) {
    console.log(`--- #console (per-subtest detail, ${r.consoleText.length} chars) ---`);
    console.log(r.consoleText);
  } else {
    console.log(
      `--- #console (per-subtest detail, ${r.consoleText.length} chars; first ${MAX_DESCRIPTION_DEFAULT} shown — use --full for the complete text) ---`,
    );
    console.log(r.consoleText.slice(0, MAX_DESCRIPTION_DEFAULT));
  }
  console.log("");
  console.log(`--- console errors (${r.consoleErrors.length}/${MAX_COLLECTED}) ---`);
  for (const m of r.consoleErrors) console.log(`  console: ${m}`);
  if (r.consoleErrors.length === 0) console.log("  (none)");
  console.log("");
  console.log(`--- pageerrors (${r.pageErrors.length}/${MAX_COLLECTED}) ---`);
  for (const m of r.pageErrors) console.log(`  pageerror: ${m}`);
  if (r.pageErrors.length === 0) console.log("  (none)");
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  if ("help" in opts) {
    console.log(USAGE);
    return;
  }

  const testFile = path.join(opts.ctsDir, "sdk", "tests", opts.testPath);
  if (!fs.existsSync(testFile)) {
    throw new Error(`test page not found: ${testFile} (pass --cts <CTS root> if needed)`);
  }
  if (!fs.existsSync(opts.rendererPath)) {
    throw new Error(
      `renderer bundle not found: ${opts.rendererPath} (build it with 'npm run build' or pass --renderer <path>)`,
    );
  }

  const result = await diagnose(opts);
  printReport(result, opts.full);
  process.exitCode = result.status === "PASS" ? 0 : 1;
}

main().catch((err: unknown) => {
  console.error(`diag-page: ${errMsg(err)}`);
  if (err instanceof Error && err.stack) console.error(err.stack);
  process.exitCode = 2;
});
