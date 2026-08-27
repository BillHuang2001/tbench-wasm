import path from 'node:path';
import { repoPath, selectPages } from './list';
import { runSuite, writeReport, type Report } from './driver';

interface CliOptions {
  filter?: string;
  full: boolean;
  workers: number;
  attempts: number;
  timeoutSec: number;
  list: boolean;
  reportDir: string;
  saveAll: boolean;
}

const USAGE = `Usage: tsx tests/threejs/run.ts [options]

Options:
  --filter <s>     Comma-separated substrings; a page is selected if it contains any (OR)
  --full           Scan all eligible pages in the three.js repo (default: curated subset)
  --workers <n>    Parallel pages (default 3, min 1)
  --renderer <p>   Path to the renderer bundle (sets WEBGL_SOFTWARE_RENDERER before injection)
  --attempts <n>   Rerun attempts per page (default 3)
  --timeout <sec>  Per-attempt goto timeout in seconds (default 120)
  --list           Print selected page names and exit without running
  --out <dir>      Report output directory (default tests/reports/threejs)
  --save-all       Save artifacts for passing pages too`;

const VALUE_FLAGS = new Set(['--filter', '--workers', '--renderer', '--attempts', '--timeout', '--out']);
const BOOL_FLAGS = new Set(['--full', '--list', '--save-all']);

function fail(message: string): never {
  console.error(`run.ts: ${message}`);
  console.error(USAGE);
  process.exit(2);
}

function parseIntOr(raw: string, fallback: number): number {
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) ? n : fallback;
}

function parseArgs(argv: string[]): CliOptions {
  const opts: CliOptions = {
    full: false,
    workers: 3,
    attempts: 3,
    timeoutSec: 120,
    list: false,
    reportDir: 'tests/reports/threejs',
    saveAll: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('-')) continue; // stray positional args are ignored

    if (BOOL_FLAGS.has(arg)) {
      if (arg === '--full') opts.full = true;
      else if (arg === '--list') opts.list = true;
      else opts.saveAll = true;
      continue;
    }

    if (VALUE_FLAGS.has(arg)) {
      const value = argv[i + 1];
      if (value === undefined) fail(`missing value for ${arg}`);
      i++;
      switch (arg) {
        case '--filter':
          opts.filter = value;
          break;
        case '--workers':
          opts.workers = Math.max(1, parseIntOr(value, 3));
          break;
        case '--renderer':
          process.env.WEBGL_SOFTWARE_RENDERER = path.resolve(value);
          break;
        case '--attempts':
          opts.attempts = parseIntOr(value, 3);
          break;
        case '--timeout':
          opts.timeoutSec = parseIntOr(value, 120);
          break;
        case '--out':
          opts.reportDir = value;
          break;
      }
      continue;
    }

    fail(`unknown flag "${arg}"`);
  }

  return opts;
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));

  const repo = repoPath();
  const pages = selectPages(repo, { full: opts.full, filter: opts.filter });

  if (opts.list) {
    for (const name of pages) console.log(name);
    console.log(`(${pages.length} pages)`);
    process.exit(0);
  }

  const startedAt = Date.now();
  const report: Report = await runSuite(pages, {
    repo,
    filter: opts.filter,
    full: opts.full,
    workers: opts.workers,
    attempts: opts.attempts,
    timeoutSec: opts.timeoutSec,
    reportDir: opts.reportDir,
    saveAll: opts.saveAll,
  });
  await writeReport(report, opts.reportDir);

  const wallSec = (Date.now() - startedAt) / 1000;
  const passed = report.pages.filter((r) => r.status === 'pass').length;
  const failed = report.pages.filter(
    (r) =>
      r.status === 'fail' ||
      r.status === 'timeout' ||
      r.status === 'error' ||
      r.status === 'renderer-inactive'
  ).length;
  const skipped = report.pages.filter((r) => r.status === 'skipped').length;
  console.log(`Passed ${passed}/${report.pages.length} (failed ${failed}, skipped ${skipped}) in ${wallSec.toFixed(1)} s`);

  process.exit(failed);
}

try {
  await main();
} catch (err) {
  console.error(err instanceof Error ? err.stack : String(err));
  process.exit(1);
}
