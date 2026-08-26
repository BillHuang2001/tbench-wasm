import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium, type Browser } from "playwright";
import { runScene, type SceneResult } from "./driver";
import { loadConfig, selectScenes, type SceneEntry } from "./scenes";
import { createServer } from "./server";
import { compareScreenshots, goldenPath, sanitizeReferenceName } from "./compare";

/**
 * Babylon.js visual regression driver — CLI entry.
 *
 * Runs curated scenes from the Babylon.js visualization config against the
 * software WebGL renderer in headless Chromium, screenshots each scene, and
 * compares against Babylon's own golden references.
 *
 * Works end-to-end even in stub mode (no renderer.js): every scene fails
 * with a RENDERER_NOT_FOUND error, the report is still written, and the
 * exit code is 1.
 */

const USAGE = `Usage: npx tsx tests/babylon/run.ts [options]
  --filter REGEX     only run scenes whose title matches REGEX
  --full             include scenes excluded from automatic testing upstream
  --workers N        number of parallel browser workers (default 2)
  --renderer PATH    software renderer bundle (default: WEBGL_SOFTWARE_RENDERER or ./renderer.js)
  --list             list the selected scenes and exit (no server, no browser)
  --limit N          cap the number of scenes that run
  --no-cache         disable the on-disk CDN proxy cache
  --timeout SECONDS  per-scene timeout (default 120)
  --out PATH         JSON report path (default: tests/reports/babylon-report.json)
  --cdn URL          Babylon CDN base (default: https://cdn.babylonjs.com)
  --root PATH        Babylon.js repo root (default: $BABYLON_ROOT or /testsuites/Babylon.js)`;

class UsageError extends Error {}

type CliArgs = {
  filter?: string;
  full: boolean;
  workers: number;
  renderer?: string;
  list: boolean;
  limit?: number;
  noCache: boolean;
  timeout: number;
  out: string;
  cdn: string;
  root: string;
};

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    full: false,
    list: false,
    noCache: false,
    workers: 2,
    timeout: 120,
    out: "tests/reports/babylon-report.json",
    cdn: "https://cdn.babylonjs.com",
    root: process.env.BABYLON_ROOT ?? "/testsuites/Babylon.js",
  };

  const valueFlags: Record<string, keyof CliArgs> = {
    "--filter": "filter",
    "--workers": "workers",
    "--renderer": "renderer",
    "--limit": "limit",
    "--timeout": "timeout",
    "--out": "out",
    "--cdn": "cdn",
    "--root": "root",
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--full") {
      args.full = true;
    } else if (arg === "--list") {
      args.list = true;
    } else if (arg === "--no-cache") {
      args.noCache = true;
    } else if (arg in valueFlags) {
      const value = argv[++i];
      if (value === undefined || value.startsWith("--")) {
        throw new UsageError(`missing value for ${arg}\n${USAGE}`);
      }
      (args as Record<string, unknown>)[valueFlags[arg]] = value;
    } else {
      throw new UsageError(`unknown option: ${arg}\n${USAGE}`);
    }
  }

  const num = (v: string | number | undefined, flag: string): number => {
    const n = typeof v === "number" ? v : Number(v);
    if (Number.isNaN(n)) throw new UsageError(`invalid number for ${flag}: ${String(v)}`);
    return n;
  };
  args.workers = num(args.workers, "--workers");
  args.timeout = num(args.timeout, "--timeout");
  if (args.limit !== undefined) args.limit = num(args.limit, "--limit");
  if (args.workers < 1) throw new UsageError("--workers must be >= 1\n" + USAGE);
  if (args.timeout < 1) throw new UsageError("--timeout must be >= 1\n" + USAGE);
  if (args.limit !== undefined && args.limit < 0) {
    throw new UsageError("--limit must be >= 0\n" + USAGE);
  }
  if (args.filter) {
    try {
      new RegExp(args.filter);
    } catch {
      throw new UsageError(`invalid --filter regex: ${args.filter}\n${USAGE}`);
    }
  }
  return args;
}

type SceneReport = {
  title: string;
  reference: string;
  kind: "file" | "script";
  ok: boolean;
  ready: boolean;
  pass: boolean;
  timeout?: boolean;
  error?: string;
  reason?: string;
  elapsedMs: number;
  diffPixels?: number;
  diffRatio?: number;
  maxDiffPixelRatio?: number;
};

function printList(scenes: SceneEntry[], babylonRoot: string): void {
  console.log("\ntitle | kind | reference | golden");
  for (const entry of scenes) {
    const reference = entry.referenceImage ?? entry.title;
    const golden = goldenPath(babylonRoot, entry);
    console.log(
      `${entry.title} | ${entry.kind} | ${reference} | ${existsSync(golden) ? "yes" : "MISSING"}`
    );
  }
}

async function main(): Promise<number> {
  let args: CliArgs;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (e) {
    if (e instanceof UsageError) {
      console.error(String(e));
      return 1;
    }
    throw e;
  }

  const rendererPath = args.renderer ?? process.env.WEBGL_SOFTWARE_RENDERER ?? "./renderer.js";
  process.env.WEBGL_SOFTWARE_RENDERER = rendererPath;

  const config = loadConfig(args.root);
  const goldenDir = path.join(
    args.root,
    "packages/tools/tests/test/visualization/ReferenceImages"
  );
  const selected = selectScenes({
    tests: config.tests,
    full: args.full,
    filter: args.filter,
    goldenDir,
  });

  console.log(`babylon root: ${args.root}`);
  console.log(`config root: ${config.root}`);
  console.log(
    `config entries: ${config.tests.length} (${config.skippedSnippets} playground snippets skipped)`
  );
  console.log(`selected scenes: ${selected.length}`);
  console.log(`renderer: ${rendererPath}`);

  if (args.list) {
    printList(selected, args.root);
    console.log(`\n${config.skippedSnippets} playground snippets out of scope`);
    return 0;
  }

  const scenes = args.limit !== undefined ? selected.slice(0, args.limit) : selected;
  if (scenes.length === 0) {
    console.log("no scenes selected — nothing to run");
    return 0;
  }
  console.log(`running: ${scenes.length} scenes (${args.workers} workers, ${args.timeout}s timeout each)`);

  if (process.env.BABYLON_PORT) {
    console.log(
      `note: BABYLON_PORT=${process.env.BABYLON_PORT} is ignored — the test server binds an ephemeral port`
    );
  }

  const reportDir = path.join("tests", "reports", "babylon");
  mkdirSync(path.dirname(args.out), { recursive: true });
  mkdirSync(reportDir, { recursive: true });

  const server = await createServer({
    cdn: args.cdn,
    cacheDir: path.join("tests", "reports", "babylon-cache"),
    noCache: args.noCache,
  });
  const serverUrl = server.url;

  let browser: Browser | undefined;
  try {
    browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
  } catch (e) {
    console.error(`failed to launch Chromium: ${String(e)}`);
    return 1;
  }
  const launchedBrowser = browser;

  let sigintHandled = false;
  const shutdown = async () => {
    try {
      await launchedBrowser.close();
    } catch {
      /* already closed */
    }
    try {
      await server.close();
    } catch {
      /* already closed */
    }
  };
  process.on("SIGINT", () => {
    if (sigintHandled) process.exit(130);
    sigintHandled = true;
    console.log("\nreceived SIGINT — shutting down");
    void shutdown().then(() => process.exit(130));
  });

  const threshold = process.env.SCREENSHOT_THRESHOLD ? +process.env.SCREENSHOT_THRESHOLD : 0.035;

  const runOne = async (entry: SceneEntry): Promise<SceneReport> => {
    const sceneResult: SceneResult = await runScene(launchedBrowser, serverUrl, entry, {
      sceneTimeoutMs: args.timeout * 1000,
      rendererPath,
    });
    const base = {
      title: sceneResult.title,
      reference: sceneResult.reference,
      kind: sceneResult.kind,
      ok: sceneResult.ok,
      ready: sceneResult.ready,
      timeout: sceneResult.timeout,
      error: sceneResult.error,
      elapsedMs: sceneResult.elapsedMs,
    };
    if (!sceneResult.screenshot) {
      return { ...base, pass: false, reason: "no screenshot" };
    }
    const maxDiffPixelRatio =
      (entry.errorRatio ??
        (process.env.SCREENSHOT_MAX_PIXEL ? +process.env.SCREENSHOT_MAX_PIXEL : 1.1)) /
      100;
    const cmp = await compareScreenshots({
      actualPng: sceneResult.screenshot,
      goldenPath: goldenPath(args.root, entry),
      threshold,
      maxDiffPixelRatio,
    });
    const pass = cmp.ok && sceneResult.ok;
    if (!pass) {
      const baseName = sanitizeReferenceName(sceneResult.reference);
      writeFileSync(path.join(reportDir, baseName + ".png"), sceneResult.screenshot);
      if (cmp.diffPng) {
        writeFileSync(path.join(reportDir, baseName + ".diff.png"), cmp.diffPng);
      }
    }
    return {
      ...base,
      pass,
      reason: cmp.reason,
      diffPixels: cmp.diffPixels,
      diffRatio: cmp.diffRatio,
      maxDiffPixelRatio: cmp.maxDiffPixelRatio,
    };
  };

  // Simple worker pool: N concurrent runners over a shared index.
  const reports: (SceneReport | undefined)[] = new Array(scenes.length);
  let next = 0;
  const runner = async () => {
    for (;;) {
      const i = next++;
      if (i >= scenes.length) return;
      reports[i] = await runOne(scenes[i]);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(args.workers, scenes.length) }, () => runner())
  );
  const report = reports as SceneReport[];

  writeFileSync(args.out, JSON.stringify(report, null, 2) + "\n");
  console.log(`\nreport written to ${args.out}`);

  const passed = report.filter((r) => r.pass).length;
  const failed = report.length - passed;

  console.log("\n=== Babylon visual regression summary ===");
  for (const r of report) {
    const status = r.pass ? "PASS" : "FAIL";
    const diff = r.diffRatio !== undefined ? r.diffRatio.toFixed(4) : "-";
    const extra = [r.error, r.reason].filter(Boolean).join(" | ");
    console.log(
      `${status.padEnd(4)} ${r.title}  diff=${diff}  maxDiff=${r.maxDiffPixelRatio?.toFixed(4) ?? "-"}  ${r.elapsedMs}ms${extra ? "  " + extra : ""}`
    );
  }
  console.log(`\n${passed} passed, ${failed} failed, ${report.length} total`);

  await shutdown();
  return failed > 0 ? 1 : 0;
}

const isMain =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMain) {
  main()
    .then((code) => {
      process.exitCode = code;
    })
    .catch((e) => {
      console.error(e);
      process.exitCode = 1;
    });
}
