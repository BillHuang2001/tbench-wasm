/**
 * Page driver for a single CTS test page.
 *
 * Result collection: js-test-pre.js captures `window.parent.webglTestHarness`
 * at script-load time, so a shim implementing the harness's two-method API is
 * injected via addInitScript BEFORE any page script runs:
 *   reportResults(url, success, msg[, skipped])
 *   notifyFinished(url)
 * The shim aggregates counts in-page (pass/fail/skip) — required because the
 * deqp pages emit tens of millions of subtests; nothing is shipped
 * page->Node per subtest. The Node side polls a small serializable snapshot.
 *
 * Fallback: pages that bypass the harness write "TEST COMPLETE: N PASS,
 * M FAIL" into #description (js-test-post.js); the poll also scrapes that.
 *
 * Timeout: activity-based (any observed progress extends the deadline),
 * mirroring the official harness's per-reportResults watchdog bump — a busy
 * deqp page must never idle-timeout.
 */

import type { BrowserContext, Page } from "playwright";

export interface TestRunResult {
  status: "pass" | "fail" | "timeout" | "error";
  pass: number;
  fail: number;
  skip: number;
  messages: string[];
  timeMs: number;
  rendererMissing: boolean;
}

export interface RunPageOptions {
  /** Idle timeout: no observed progress for this long -> timeout-fail. */
  timeoutMs: number;
}

/** Poll interval for the in-page snapshot. */
const POLL_INTERVAL_MS = 250;

const MAX_MESSAGES = 20;
const MAX_MESSAGE_LENGTH = 500;

/**
 * In-page harness shim. Must be injected via context.addInitScript BEFORE
 * page scripts (js-test-pre.js caches window.parent.webglTestHarness at load;
 * in a standalone page window.parent === window, and same-origin iframes
 * created by tests reach it through their own window.parent).
 */
export const HARNESS_SHIM_SCRIPT: string = `
(function() {
  if (window.__softglCts) { return; }
  var state = { pass: 0, fail: 0, skip: 0, finished: false, messages: [], url: null };
  window.__softglCts = state;
  window.webglTestHarness = {
    reportResults: function(url, success, msg, skipped) {
      if (skipped) {
        state.skip += 1;
      } else if (success) {
        state.pass += 1;
      } else {
        state.fail += 1;
      }
      if (!state.url && url) { state.url = String(url); }
      if (msg && !success && state.messages.length < 20) {
        state.messages.push(String(msg).slice(0, 500));
      }
    },
    notifyFinished: function(url) {
      state.finished = true;
    }
  };
})();
`;

interface CtsPageState {
  pass: number;
  fail: number;
  skip: number;
  finished: boolean;
}

interface PollSnapshot {
  pass: number;
  fail: number;
  skip: number;
  finished: boolean;
  title: string;
  description: string;
}

/** Must be a real function (not a string) so Playwright serializes and calls it. */
const pollSnapshot = (): PollSnapshot => {
  const s = (window as unknown as { __softglCts?: CtsPageState }).__softglCts;
  const d = document.getElementById("description");
  return {
    pass: s ? s.pass : 0,
    fail: s ? s.fail : 0,
    skip: s ? s.skip : 0,
    finished: s ? s.finished : false,
    title: document.title || "",
    description: d && d.textContent ? d.textContent : "",
  };
};

/** Parses the js-test-post.js summary line, e.g. "TEST COMPLETE: 12 PASS, 0 FAIL". */
export function parseDomComplete(text: string): { pass: number; fail: number } | null {
  const m = /TEST COMPLETE:\s*(\d+)\s*PASS,\s*(\d+)\s*FAIL/.exec(text);
  return m ? { pass: parseInt(m[1], 10), fail: parseInt(m[2], 10) } : null;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Loads one CTS test page and drives it to completion/timeout. */
export async function runTestPage(
  context: BrowserContext,
  url: string,
  opts: RunPageOptions,
): Promise<TestRunResult> {
  const page: Page = await context.newPage();
  const started = Date.now();
  const messages: string[] = [];
  const pushMessage = (m: string) => {
    if (messages.length < MAX_MESSAGES) messages.push(m.slice(0, MAX_MESSAGE_LENGTH));
  };

  page.on("pageerror", (err) => pushMessage(`pageerror: ${errMsg(err)}`));
  page.on("console", (msg) => {
    if (msg.type() === "error") pushMessage(`console: ${msg.text()}`);
  });

  let lastActivity = Date.now();
  let lastKey = "";

  try {
    let response = null;
    try {
      response = await page.goto(url, { waitUntil: "domcontentloaded", timeout: opts.timeoutMs });
    } catch (err) {
      return {
        status: "error",
        pass: 0,
        fail: 0,
        skip: 0,
        messages: [...messages, `navigation failed: ${errMsg(err)}`],
        timeMs: Date.now() - started,
        rendererMissing: false,
      };
    }
    if (response && response.status() >= 400) {
      return {
        status: "error",
        pass: 0,
        fail: 0,
        skip: 0,
        messages: [...messages, `HTTP ${response.status()} for ${url}`],
        timeMs: Date.now() - started,
        rendererMissing: false,
      };
    }

    for (;;) {
      const snap = await page.evaluate(pollSnapshot);
      const now = Date.now();

      if (snap.finished) {
        return {
          status: snap.fail > 0 ? "fail" : "pass",
          pass: snap.pass,
          fail: snap.fail,
          skip: snap.skip,
          messages,
          timeMs: now - started,
          rendererMissing: false,
        };
      }

      // DOM fallback: pages that bypass the harness shim report via the
      // "TEST COMPLETE: N PASS, M FAIL" line appended to #description.
      const dom = parseDomComplete(snap.description);
      if (dom) {
        return {
          status: dom.fail > 0 ? "fail" : "pass",
          pass: dom.pass,
          fail: dom.fail,
          skip: snap.skip,
          messages,
          timeMs: now - started,
          rendererMissing: false,
        };
      }

      // Stub-renderer fast-fail: src/context-intercept.ts's stub sets
      // document.title = 'RENDERER_NOT_FOUND' before throwing from
      // getContext, so a missing/broken renderer fails fast with a clear
      // message instead of burning the full timeout.
      if (snap.title === "RENDERER_NOT_FOUND") {
        return {
          status: "fail",
          pass: snap.pass,
          fail: snap.fail + 1,
          skip: snap.skip,
          messages: [
            ...messages,
            "renderer not found: the software renderer bundle is missing or failed to load (WEBGL_SOFTWARE_RENDERER)",
          ],
          timeMs: now - started,
          rendererMissing: true,
        };
      }

      const key = `${snap.pass},${snap.fail},${snap.skip},${snap.title}`;
      if (key !== lastKey) {
        lastActivity = now;
        lastKey = key;
      }
      if (now - lastActivity > opts.timeoutMs) {
        return {
          status: "timeout",
          pass: snap.pass,
          fail: snap.fail,
          skip: snap.skip,
          messages: [...messages, `timed out after ${opts.timeoutMs}ms idle (no progress)`],
          timeMs: now - started,
          rendererMissing: false,
        };
      }

      await sleep(POLL_INTERVAL_MS);
    }
  } catch (err) {
    return {
      status: "error",
      pass: 0,
      fail: 0,
      skip: 0,
      messages: [...messages, `page driver error: ${errMsg(err)}`],
      timeMs: Date.now() - started,
      rendererMissing: false,
    };
  } finally {
    await page.close().catch(() => {});
  }
}
