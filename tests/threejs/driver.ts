import { readFileSync, existsSync, createReadStream, statSync } from "node:fs";
import { mkdir, copyFile, writeFile } from "node:fs/promises";
import { createServer as createHttpServer, type ServerResponse } from "node:http";
import { join, resolve, normalize, extname, sep } from "node:path";
import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { buildInterceptScript } from "../../src/context-intercept";
import { hasGolden, goldenPath, pageHtmlPath } from "./list";
import {
  readImage,
  downscale2x,
  compareImages,
  makeDiffImage,
  writePng,
  type DiffStats,
} from "./compare";

export type PageStatus = "pass" | "fail" | "timeout" | "error" | "skipped";

export interface PageResult {
  name: string;
  status: PageStatus;
  skipReason?: string;
  /** attempts actually executed */
  attempts: number;
  /** total across attempts */
  elapsedMs: number;
  /** best (min diffPercent) attempt */
  diff?: DiffStats;
  consoleErrors: string[];
  pageErrors: string[];
  artifacts: { actual: string; expected: string; diff: string } | null;
}

export interface RunOptions {
  repo: string;
  filter?: string;
  full?: boolean;
  workers: number;
  attempts: number;
  timeoutSec: number;
  reportDir: string;
  saveAll: boolean;
}

export interface Report {
  timestamp: string;
  repoPath: string;
  rendererPath: string;
  pages: PageResult[];
  summary: { total: number; passed: number; failed: number; skipped: number };
}

/* ------------------------------------------------------------------ */
/* Deterministic injection (verbatim port of three.js e2e)            */
/* ------------------------------------------------------------------ */

const DETERMINISTIC_SOURCE = `( function () {

	/* Deterministic random */

	window.Math._random = window.Math.random;

	let seed = Math.PI / 4;
	window.Math.random = function () {

		const x = Math.sin( seed ++ ) * 10000;
		return x - Math.floor( x );

	};

	/* Deterministic timer */

	window.performance._now = performance.now;

	const now = () => 0; // frameId * 16;
	window.Date.now = now;
	window.Date.prototype.getTime = now;
	window.performance.now = now;

	/* Deterministic RAF */

	window._renderStarted = false;
	window._renderFinished = false;

	window.requestAnimationFrame = function ( cb ) {

		if ( window._renderFinished === true ) return;

		const intervalId = setInterval( function () {

			if ( window._renderStarted === true ) {

				clearInterval( intervalId );
				window._renderFinished = true;
				cb( now() );

			}

		}, 100 );

	};

	/* Semi-deterministic video */

	const play = HTMLVideoElement.prototype.play;

	HTMLVideoElement.prototype.play = async function () {

		play.call( this );
		this.addEventListener( 'timeupdate', () => this.pause() );

		function renew() {

			this.load();
			play.call( this );
			RAF( renew ); // eslint-disable-line no-undef

		}

		RAF( renew ); // eslint-disable-line no-undef

	};

	/* Additional variable for ~5 examples */

	window.TESTING = true;

}() );
`;

/**
 * Verbatim port of /testsuites/three.js/test/e2e/deterministic-injection.js.
 * NOTE: the undefined `RAF` identifier inside the video override is INTENTIONAL
 * (upstream quirk — the resulting unhandled ReferenceError is why pageerror
 * is warn-only, never a page failure).
 */
export function buildDeterministicScript(): string {
  return DETERMINISTIC_SOURCE;
}

/* Verbatim port of /testsuites/three.js/test/e2e/clean-page.js */
const CLEAN_PAGE_SOURCE = `( function () {

	/* Remove start screen (or press some button ) */

	const button = document.getElementById( 'startButton' );
	if ( button ) button.click();

	/* Remove gui and fonts */

	const style = document.createElement( 'style' );
	style.innerHTML = '#info, #camera-info, .three-inspector, button, input, body > div.lil-gui, body > div.lbl { display: none !important; }';

	document.querySelector( 'head' ).appendChild( style );

	/* Remove Stats.js */

	for ( const element of document.querySelectorAll( 'div' ) ) {

		if ( getComputedStyle( element ).zIndex === '10000' ) {

			element.remove();
			break;

		}

	}

}() );
`;

/* ------------------------------------------------------------------ */
/* Static server with build patching + Range support                  */
/* ------------------------------------------------------------------ */

const BUILD_PATHS = [
  "/build/three.module.js",
  "/build/three.core.js",
  "/build/three.webgpu.js",
];

const MIME_TYPES: Record<string, string> = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".json": "application/json",
  ".glb": "model/gltf-binary",
  ".gltf": "model/gltf+json",
  ".hdr": "image/vnd.radiance",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".mp4": "video/mp4",
  ".ogv": "video/ogg",
  ".webm": "video/webm",
  ".ttf": "font/ttf",
  ".obj": "text/plain",
  ".mtl": "text/plain",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".bin": "application/octet-stream",
  ".exr": "image/x-exr",
  ".dds": "image/vnd-ms.dds",
  ".wav": "audio/wav",
  ".css": "text/css",
};

/**
 * Read + patch the three.js build files once at startup.
 * Returns a map of exact request path -> patched bytes, or null when a file
 * is unavailable (that URL then 404s). The `Math._random` replacement keeps
 * UUID entropy real; the trackTimestamp disable means goldens are never
 * written by the WebGPU path.
 */
function readBuildFiles(root: string): Map<string, Buffer | null> {
  const map = new Map<string, Buffer | null>();
  for (const rel of BUILD_PATHS) {
    try {
      const content = readFileSync(join(root, rel), "utf-8");
      const patched = content
        .replace(/Math\.random\(\) \* 0xffffffff/g, "Math._random() * 0xffffffff")
        .replace(
          /this\.trackTimestamp\s*=\s*\(\s*parameters\.trackTimestamp\s*===\s*true\s*\);/g,
          "Object.defineProperty(this, 'trackTimestamp', { get: () => false, set: () => {} });"
        );
      map.set(rel, Buffer.from(patched, "utf-8"));
    } catch {
      map.set(rel, null);
    }
  }
  return map;
}

/** Parse a single `Range: bytes=start-end` header (also `start-` and `-suffix`). */
function parseRange(header: string, total: number): { start: number; end: number } | null {
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!m || total <= 0) return null;
  const [, startStr, endStr] = m;
  if (startStr === "" && endStr === "") return null;
  let start: number;
  let end: number;
  if (startStr === "") {
    const suffix = Number(endStr);
    if (!Number.isFinite(suffix) || suffix <= 0) return null;
    start = Math.max(total - suffix, 0);
    end = total - 1;
  } else {
    start = Number(startStr);
    if (!Number.isInteger(start) || start < 0 || start >= total) return null;
    end = endStr === "" ? total - 1 : Number(endStr);
    if (!Number.isInteger(end)) return null;
    end = Math.min(end, total - 1);
    if (end < start) return null;
  }
  return { start, end };
}

function serveBuffer(
  res: ServerResponse,
  buf: Buffer,
  contentType: string,
  rangeHeader: string | undefined
): void {
  const total = buf.length;
  const base = {
    "Content-Type": contentType,
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-store",
  };
  const range = rangeHeader ? parseRange(rangeHeader, total) : null;
  if (range) {
    res.writeHead(206, {
      ...base,
      "Content-Range": `bytes ${range.start}-${range.end}/${total}`,
      "Content-Length": range.end - range.start + 1,
    });
    res.end(buf.subarray(range.start, range.end + 1));
    return;
  }
  res.writeHead(200, { ...base, "Content-Length": total });
  res.end(buf);
}

function serveFile(
  res: ServerResponse,
  filePath: string,
  contentType: string,
  rangeHeader: string | undefined
): void {
  const total = statSync(filePath).size;
  const base = {
    "Content-Type": contentType,
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-store",
  };
  const range = rangeHeader ? parseRange(rangeHeader, total) : null;
  if (range) {
    res.writeHead(206, {
      ...base,
      "Content-Range": `bytes ${range.start}-${range.end}/${total}`,
      "Content-Length": range.end - range.start + 1,
    });
    createReadStream(filePath, { start: range.start, end: range.end }).pipe(res);
    return;
  }
  res.writeHead(200, { ...base, "Content-Length": total });
  createReadStream(filePath).pipe(res);
}

export async function createServer(
  repo: string
): Promise<{ port: number; close(): Promise<void> }> {
  const root = resolve(repo);
  const buildFiles = readBuildFiles(root);

  const server = createHttpServer((req, res) => {
    const notFound = (): void => {
      res.writeHead(404, { "Content-Type": "text/plain", "Cache-Control": "no-store" });
      res.end("Not found");
    };

    let pathname: string;
    try {
      pathname = decodeURIComponent((req.url ?? "/").split("?")[0]);
    } catch {
      notFound();
      return;
    }
    const rel = normalize(pathname).replace(/^[/\\]+/, "");
    if (rel === "" || rel.split(/[\\/]/).includes("..")) {
      notFound();
      return;
    }
    const abs = resolve(root, rel);
    if (!abs.startsWith(root + sep)) {
      notFound();
      return;
    }

    const patched = buildFiles.get("/" + rel);
    if (patched !== undefined) {
      if (patched === null) {
        notFound();
        return;
      }
      serveBuffer(res, patched, "application/javascript", req.headers.range);
      return;
    }

    if (!existsSync(abs) || !statSync(abs).isFile()) {
      notFound();
      return;
    }
    const type = MIME_TYPES[extname(abs).toLowerCase()] ?? "text/plain";
    serveFile(res, abs, type, req.headers.range);
  });

  await new Promise<void>((resolveListen) =>
    server.listen(0, "127.0.0.1", resolveListen)
  );
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("Failed to determine server address");
  }
  return {
    port: address.port,
    close: () =>
      new Promise<void>((resolveClose, reject) => {
        server.closeAllConnections?.();
        server.closeIdleConnections?.();
        server.close((err) => (err ? reject(err) : resolveClose()));
      }),
  };
}

/* ------------------------------------------------------------------ */
/* Per-page run + worker pool                                         */
/* ------------------------------------------------------------------ */

const RENDER_WAIT_SOURCE = `(async () => {
  window._renderStarted = true;
  await new Promise((resolve) => {
    const t0 = performance._now();
    const iv = setInterval(() => {
      if (window._renderFinished) {
        clearInterval(iv);
        resolve();
      } else if (performance._now() - t0 > 5000) {
        clearInterval(iv);
        resolve();
      }
    }, 100);
  });
})()`;

/**
 * Normalize a console error message: strip the leading `[.WebGL-...]` prefix,
 * map the literal `JSHandle@error` to 'Unknown error'. Returns null for
 * messages that must be ignored ('Timestamp tracking is disabled').
 */
function normalizeConsoleMessage(text: string): string | null {
  let msg = text.replace(/^\[\.WebGL-[^\]]*\]/, "");
  if (msg === "JSHandle@error") msg = "Unknown error";
  if (msg === "Timestamp tracking is disabled") return null;
  return msg;
}

async function runPage(
  context: BrowserContext,
  name: string,
  opts: RunOptions,
  port: number
): Promise<PageResult> {
  await mkdir(opts.reportDir, { recursive: true });

  if (!existsSync(pageHtmlPath(opts.repo, name)) || !hasGolden(opts.repo, name)) {
    return {
      name,
      status: "skipped",
      skipReason: "missing html or golden",
      attempts: 0,
      elapsedMs: 0,
      consoleErrors: [],
      pageErrors: [],
      artifacts: null,
    };
  }

  const attempts = Math.max(1, opts.attempts);
  const consoleErrors = new Set<string>();
  const pageErrors: string[] = [];
  let bestDiff: DiffStats | null = null;
  let attemptsExecuted = 0;
  let allGotoTimeout = true;
  let elapsedMs = 0;
  let artifacts: PageResult["artifacts"] = null;

  for (let attempt = 0; attempt < attempts; attempt++) {
    const attemptStart = Date.now();
    const attemptPageErrors: string[] = [];
    let gotoTimeout = false;
    let diff: DiffStats | null = null;
    let screenshotOk = false;
    const actualPath = join(opts.reportDir, name + "-actual.png");

    const page: Page = await context.newPage();
    attemptsExecuted++;
    try {
      page.on("console", (msg) => {
        if (msg.type() === "error") {
          const normalized = normalizeConsoleMessage(msg.text());
          if (normalized !== null) {
            consoleErrors.add(normalized);
          }
        } else if (msg.type() === "warning") {
          process.stderr.write(msg.text() + "\n");
        }
      });
      page.on("pageerror", (err) => {
        const s = String(err);
        attemptPageErrors.push(s);
        pageErrors.push(s);
        process.stderr.write("PAGE ERROR: " + s + "\n");
      });

      // Byte tracking (per spec; diagnostics for the load).
      let bytes = 0;
      page.on("response", (res) => {
        if (res.status() === 200) {
          const cl = Number(res.headers()["content-length"] ?? 0);
          if (cl > 0) bytes += cl;
        }
      });

      try {
        await page.goto("http://127.0.0.1:" + port + "/examples/" + name + ".html", {
          // Playwright has no 'networkidle0' (Puppeteer name); 'networkidle' is
          // the equivalent ("no network connections for 500 ms").
          waitUntil: "networkidle",
          timeout: opts.timeoutSec * 1000,
        });
      } catch {
        gotoTimeout = true;
      }
      if (!gotoTimeout) allGotoTimeout = false;

      // On goto timeout we STILL attempt the screenshot below, then break.
      try {
        await page.evaluate(CLEAN_PAGE_SOURCE);
        await page.waitForLoadState("networkidle");
        await page.waitForTimeout(2000);
        await page.evaluate(RENDER_WAIT_SOURCE);
      } catch (e) {
        attemptPageErrors.push("page setup failed: " + String(e));
      }

      try {
        await page.screenshot({ path: actualPath });
        screenshotOk = true;
      } catch (e) {
        attemptPageErrors.push("screenshot failed: " + String(e));
      }
    } finally {
      await page.close().catch(() => {});
    }

    if (screenshotOk) {
      try {
        const actual = await readImage(actualPath);
        const downscaled = downscale2x(actual);
        const golden = await readImage(goldenPath(opts.repo, name));
        diff = compareImages(downscaled, golden);
        if (bestDiff === null || diff.diffPercent < bestDiff.diffPercent) {
          bestDiff = diff;
        }
        // Artifacts (diff/expected) are written for failed attempts only,
        // unless saveAll is set. Each attempt overwrites the same paths.
        if (!diff.pass || opts.saveAll) {
          const diffPath = join(opts.reportDir, name + "-diff.png");
          const expectedPath = join(opts.reportDir, name + "-expected.jpg");
          await writePng(diffPath, makeDiffImage(downscaled, golden));
          await copyFile(goldenPath(opts.repo, name), expectedPath);
          artifacts = { actual: actualPath, expected: expectedPath, diff: diffPath };
        }
      } catch (e) {
        attemptPageErrors.push("image diff failed: " + String(e));
      }
    }

    elapsedMs += Date.now() - attemptStart;

    // Surface per-attempt failures (page setup / screenshot / diff errors) at
    // page level, deduped against pageerror entries already pushed above.
    for (const e of attemptPageErrors) {
      if (!pageErrors.includes(e)) pageErrors.push(e);
    }

    if (gotoTimeout) break;
    if (diff !== null && diff.pass) break;
  }

  let status: PageStatus;
  if (attemptsExecuted === 0) {
    status = "error";
  } else if (allGotoTimeout) {
    status = "timeout";
  } else if (consoleErrors.size > 0) {
    status = "error";
  } else if (bestDiff !== null && bestDiff.pass) {
    status = "pass";
  } else {
    status = "fail";
  }

  if (status === "pass" && !opts.saveAll) {
    artifacts = null;
  }

  return {
    name,
    status,
    attempts: attemptsExecuted,
    elapsedMs,
    diff: bestDiff ?? undefined,
    consoleErrors: [...consoleErrors],
    pageErrors,
    artifacts,
  };
}

export async function runSuite(pages: string[], opts: RunOptions): Promise<Report> {
  const server = await createServer(opts.repo);
  let browser: Browser | null = null;
  try {
    await mkdir(opts.reportDir, { recursive: true });

    browser = await chromium.launch({
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        "--disable-gpu-driver-bug-workarounds",
        "--disable-gpu-watchdog",
        "--ignore-gpu-blocklist",
        "--autoplay-policy=no-user-gesture-required",
      ],
    });
    const b = browser;

    const results: (PageResult | null)[] = new Array(pages.length).fill(null);
    let next = 0;
    const workerCount = Math.max(1, opts.workers);

    const workers = Array.from({ length: workerCount }, async () => {
      const context = await b.newContext({
        viewport: { width: 400, height: 250 },
        deviceScaleFactor: 2,
      });
      await context.addInitScript(buildInterceptScript());
      await context.addInitScript(buildDeterministicScript());
      try {
        for (;;) {
          const idx = next++;
          if (idx >= pages.length) break;
          const name = pages[idx];
          results[idx] = await runPage(context, name, opts, server.port).catch((err) => {
            const result: PageResult = {
              name,
              status: "error",
              attempts: 0,
              elapsedMs: 0,
              consoleErrors: [],
              pageErrors: [String(err)],
              artifacts: null,
            };
            return result;
          });
        }
      } finally {
        await context.close().catch(() => {});
      }
    });

    await Promise.all(workers);

    const report = buildReport(pages, results, opts);
    printSummary(report);
    return report;
  } finally {
    await browser?.close().catch(() => {});
    await server.close();
  }
}

function buildReport(
  pages: string[],
  results: (PageResult | null)[],
  opts: RunOptions
): Report {
  const pageResults = pages.map((_, i) => {
    const r = results[i];
    if (r !== null) return r;
    return {
      name: pages[i],
      status: "error" as PageStatus,
      attempts: 0,
      elapsedMs: 0,
      consoleErrors: [] as string[],
      pageErrors: ["worker produced no result"],
      artifacts: null,
    };
  });
  const passed = pageResults.filter((r) => r.status === "pass").length;
  const failed = pageResults.filter(
    (r) => r.status === "fail" || r.status === "timeout" || r.status === "error"
  ).length;
  const skipped = pageResults.filter((r) => r.status === "skipped").length;
  return {
    timestamp: new Date().toISOString(),
    repoPath: opts.repo,
    rendererPath: process.env.WEBGL_SOFTWARE_RENDERER ?? "./renderer.js",
    pages: pageResults,
    summary: { total: pages.length, passed, failed, skipped },
  };
}

function printSummary(report: Report): void {
  const labels: Record<PageStatus, string> = {
    pass: "PASS",
    fail: "FAIL",
    timeout: "TIMEOUT",
    error: "ERROR",
    skipped: "SKIP",
  };
  for (const r of report.pages) {
    if (r.status === "skipped") {
      console.log(`${labels[r.status]} ${r.name} (${r.skipReason ?? "skipped"})`);
    } else {
      const diffPct = r.diff !== undefined ? r.diff.diffPercent.toFixed(2) : "n/a";
      console.log(
        `${labels[r.status]} ${r.name} (diff ${diffPct}%, ${(r.elapsedMs / 1000).toFixed(1)}s)`
      );
    }
  }
}

/* ------------------------------------------------------------------ */
/* Report writing                                                     */
/* ------------------------------------------------------------------ */

export async function writeReport(report: Report, reportDir: string): Promise<void> {
  await mkdir(reportDir, { recursive: true });
  const json = JSON.stringify(report, null, 2);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  await writeFile(join(reportDir, `report-${stamp}.json`), json);
  await writeFile(join(reportDir, "latest.json"), json);
}
