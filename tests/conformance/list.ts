/**
 * CTS test-list parser — a faithful TypeScript port of `getFileList` /
 * `getFileListImpl` from /testsuites/WebGL/sdk/tests/js/webgl-test-harness.js
 * (the official harness module; the foreign repo is read-only, so the
 * semantics are duplicated here and must stay in sync with the source).
 *
 * Semantics ported EXACTLY (verified by running a verbatim Node port of the
 * official parser against the tree):
 * - Each line is trimmed; only lines with length > 4 whose first char is not
 *   '#' or ';' and which do not start with '//' are parsed. Full-line
 *   comments only — inline comments are NOT stripped.
 * - Parsed lines are split on /\s+/; a token starting with '-' must start
 *   with '--' (single dash throws); '--slow' takes no argument;
 *   '--min-version'/'--max-version' consume the next token; unknown options
 *   throw.
 * - All non-option tokens are joined with a single space to form the test
 *   URL, relative to the directory of the containing .txt file.
 * - Directives apply to their own line only, but a directive on a line whose
 *   URL ends in '.txt' becomes a DEFAULT inherited by every line inside that
 *   sublist (root defaults: defaultVersion "1.0", defaultMaxVersion null).
 *   A version-gated .txt subtree is recursed and its leaves filtered
 *   individually (which is how gated subtrees end up excluded).
 * - Leaf URLs are version-filtered: included iff
 *     suiteVersion >= minVersion  AND  (no maxVersion OR maxVersion >= suiteVersion)
 *   using greaterThanOrEqualToVersion: numeric dotted compare with
 *   "2.0.1 (beta)" -> "2.0.1" and missing components counting as 0.
 */

import fs from "node:fs";
import path from "node:path";

/** Official default suite version (webgl-conformance-tests.html: DEFAULT_CONFORMANCE_TEST_VERSION). */
export const DEFAULT_SUITE_VERSION = "2.0.1 (beta)";

export interface TestEntry {
  /** URL relative to the CTS sdk/tests/ root, e.g. "conformance/context/constants.html". */
  url: string;
  slow?: boolean;
  minVersion?: string;
  maxVersion?: string;
}

export interface ListOptions {
  /** Suite version used for min/max version filtering. */
  version: string;
  /** Official ?fast=1 mode: skip tests marked --slow. */
  fast?: boolean;
  minVersion?: string;
  maxVersion?: string;
}

interface HierarchicalOptions {
  defaultVersion: string;
  defaultMaxVersion: string | null;
  defaultSlow?: boolean;
}

interface LineOptions {
  minVersion?: string;
  maxVersion?: string;
  slow?: boolean;
}

/**
 * Port of greaterThanOrEqualToVersion(have, want): numeric dotted compare,
 * returns have >= want. "2.0.1 (beta)" splits on space ("2.0.1"), then on
 * '.'; each component is parsed as int; a missing 'have' component is 0.
 */
export function compareVersion(have: string, want: string): boolean {
  const haveParts = have.split(" ")[0].split(".");
  const wantParts = want.split(" ")[0].split(".");
  for (let ii = 0; ii < wantParts.length; ii++) {
    const wantNum = parseInt(wantParts[ii], 10);
    const haveNum = haveParts[ii] ? parseInt(haveParts[ii], 10) : 0;
    if (haveNum > wantNum) return true;
    if (haveNum < wantNum) return false;
  }
  return true;
}

function parseLine(
  prefix: string,
  line: string,
  lineNum: number,
  hierarchical: HierarchicalOptions,
  options: ListOptions,
  out: TestEntry[],
  readSublist: (relPath: string, childHier: HierarchicalOptions) => void,
): void {
  const args = line.split(/\s+/);
  const nonOptions: string[] = [];
  const testOptions: LineOptions = {};

  for (let jj = 0; jj < args.length; jj++) {
    const arg = args[jj];
    if (arg[0] === "-") {
      if (arg[1] !== "-") {
        throw new Error(`bad option in list at line ${lineNum}: '${arg}'`);
      }
      const option = arg.substring(2);
      switch (option) {
        case "slow":
          testOptions.slow = true;
          break;
        case "min-version":
        case "max-version":
          jj++;
          if (option === "min-version") testOptions.minVersion = args[jj];
          else testOptions.maxVersion = args[jj];
          break;
        default:
          throw new Error(`bad unknown option '${option}' in list at line ${lineNum}`);
      }
    } else {
      nonOptions.push(arg);
    }
  }

  const url = prefix + nonOptions.join(" ");
  if (url.endsWith(".txt")) {
    // Directives on a .txt line become defaults inherited by its subtree.
    const child: HierarchicalOptions = { ...hierarchical };
    if (testOptions.minVersion) child.defaultVersion = testOptions.minVersion;
    if (testOptions.maxVersion) child.defaultMaxVersion = testOptions.maxVersion;
    if (testOptions.slow) child.defaultSlow = true;
    readSublist(url, child);
    return;
  }

  const minVersion = testOptions.minVersion ?? hierarchical.defaultVersion;
  const maxVersion = testOptions.maxVersion ?? hierarchical.defaultMaxVersion;
  const slow = testOptions.slow ?? hierarchical.defaultSlow;

  let useTest = true;
  if (options.fast && slow) {
    useTest = false;
  } else if (options.minVersion) {
    useTest = compareVersion(minVersion, options.minVersion);
  } else if (options.maxVersion && maxVersion) {
    useTest = compareVersion(options.maxVersion, maxVersion);
  } else {
    useTest = compareVersion(options.version, minVersion);
    if (maxVersion) {
      useTest = useTest && compareVersion(maxVersion, options.version);
    }
  }

  if (useTest) {
    out.push({ url, slow: slow || undefined, minVersion: testOptions.minVersion, maxVersion: testOptions.maxVersion });
  }
}

function parseSublist(
  relPath: string,
  hierarchical: HierarchicalOptions,
  options: ListOptions,
  out: TestEntry[],
  readFile: (relPath: string) => string,
): void {
  const text = readFile(relPath);
  const prefix = relPath.substring(0, relPath.lastIndexOf("/") + 1);
  const lines = text.split("\n");
  for (let ii = 0; ii < lines.length; ii++) {
    const str = lines[ii].replace(/^\s\s*/, "").replace(/\s\s*$/, "");
    if (str.length > 4 && str[0] !== "#" && str[0] !== ";" && str.substr(0, 2) !== "//") {
      parseLine(prefix, str, ii + 1, hierarchical, options, out, (sub, childHier) =>
        parseSublist(sub, childHier, options, out, readFile),
      );
    }
  }
}

/** Reads a file below <ctsDir>/sdk/tests/, refusing paths that escape it. */
function readCtsFile(ctsDir: string, relPath: string): string {
  const base = path.resolve(ctsDir, "sdk", "tests");
  const full = path.resolve(base, relPath);
  if (full !== base && !full.startsWith(base + path.sep)) {
    throw new Error(`list entry escapes CTS root: ${relPath}`);
  }
  return fs.readFileSync(full, "utf-8");
}

/**
 * Parses a manifest (e.g. "conformance/00_test_list.txt") and returns the
 * version-filtered leaf tests. `ctsDir` is the CTS repository root (the
 * directory containing sdk/tests). Throws if the manifest or CTS root is
 * missing — a broken environment must fail loudly, not silently pass 0 tests.
 */
export function loadSuite(
  manifestPath: string,
  ctsDir: string,
  options: Partial<ListOptions> = {},
): TestEntry[] {
  const base = path.resolve(ctsDir, "sdk", "tests");
  if (!fs.existsSync(base)) {
    throw new Error(`CTS root not found: ${ctsDir} (expected ${base}). Pass --cts <repo-root> or set CTS_DIR.`);
  }
  const out: TestEntry[] = [];
  const opts: ListOptions = {
    version: options.version ?? DEFAULT_SUITE_VERSION,
    fast: options.fast,
    minVersion: options.minVersion,
    maxVersion: options.maxVersion,
  };
  parseSublist(
    manifestPath,
    { defaultVersion: "1.0", defaultMaxVersion: null },
    opts,
    out,
    (rel) => readCtsFile(ctsDir, rel),
  );
  return out;
}

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const n = parseInt(raw, 10);
  if (Number.isNaN(n)) throw new Error(`invalid ${name} env value: '${raw}'`);
  return n;
}

/**
 * Expected leaf-test counts at suite version 2.0.1, verified against the
 * current tree (/testsuites/WebGL) with a verbatim port of the official
 * parser. NOTE: the widely-quoted official 2.0.1 figure "887 conformance"
 * is STALE — the current tree yields 835 for conformance/ and 53 for
 * conformance/more/ (888 together). conformance2 is exactly 1184; deqp is
 * 793 (885 .html files exist on disk; 92 are excluded by the manifest:
 * 33 gles2-shader pages gated --max-version 1.9.9, 55 builtinprecision
 * pages commented out, 4 unlisted leftovers). Override via env
 * CTS_EXPECTED_CONFORMANCE / CTS_EXPECTED_MORE / CTS_EXPECTED_CONFORMANCE2 /
 * CTS_EXPECTED_DEQP if the tree legitimately changes.
 */
export const EXPECTED_COUNTS: Readonly<Record<string, number>> = {
  conformance: envInt("CTS_EXPECTED_CONFORMANCE", 835),
  more: envInt("CTS_EXPECTED_MORE", 53),
  conformance2: envInt("CTS_EXPECTED_CONFORMANCE2", 1184),
  deqp: envInt("CTS_EXPECTED_DEQP", 793),
};

/** Fails loudly when the parser's leaf count for a suite drifts from expectation. */
export function assertSuiteCount(name: string, actual: number): void {
  const expected = EXPECTED_COUNTS[name];
  if (expected === undefined) return;
  if (actual !== expected) {
    throw new Error(
      `Suite "${name}" parse count mismatch: expected ${expected} tests at suite version ` +
        `"${DEFAULT_SUITE_VERSION}", parser produced ${actual}. If the CTS tree legitimately changed, ` +
        `update EXPECTED_COUNTS in tests/conformance/list.ts or set CTS_EXPECTED_${name.toUpperCase()}.`,
    );
  }
}
