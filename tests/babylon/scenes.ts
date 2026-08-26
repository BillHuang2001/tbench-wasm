import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

// compare.ts is implemented in a sibling worktree; in the integrated tree it
// exports sanitizeReferenceName. While it is absent here (empty placeholder),
// fall back to the local mirror below — same rule, so behavior is identical.
// @ts-ignore -- sanitizeReferenceName is exported by compare.ts
import { sanitizeReferenceName as compareSanitizeReferenceName } from "./compare";

export type SceneEntry = {
  kind: "file" | "script";
  title: string;
  referenceImage?: string;
  sceneFolder?: string;
  sceneFilename?: string;
  scriptToRun?: string;
  functionToCall?: string;
  replace?: string;
  replaceUrl?: string;
  rootPath?: string;
  specificRoot?: string;
  errorRatio?: number;
  renderCount?: number;
  useLargeWorldRendering?: boolean;
  excludeFromAutomaticTesting?: boolean;
};

/**
 * Local mirror of compare.ts's `sanitizeReferenceName`: strip the extension,
 * collapse runs of characters outside [A-Za-z0-9._-] to "-", append ".png".
 * Used only until compare.ts lands; compare's implementation takes precedence.
 */
function sanitizeReferenceNameFallback(name: string): string {
  return name.replace(/\.[^/.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "-") + ".png";
}

const sanitizeReferenceName: (name: string) => string =
  typeof compareSanitizeReferenceName === "function"
    ? compareSanitizeReferenceName
    : sanitizeReferenceNameFallback;

/** Default (non-full) run: the 16 scenes from config.json with local goldens. */
export const CURATED_TITLES: string[] = [
  "Sponza",
  "Windows cafe",
  "Espilit",
  "The car",
  "Viper",
  "Retail",
  "Hill Valley",
  "Heart",
  "SpaceDeK",
  "Flat2009",
  "Fog",
  "Polygon",
  "Lines",
  "Lens",
  "Self shadowing",
  "GUI",
];

/** Only added with --full (both `excludeFromAutomaticTesting: true` upstream). */
export const FULL_ONLY_TITLES: string[] = ["Mansion", "Procedural textures"];

const CONFIG_RELATIVE_PATH = join("packages", "tools", "tests", "test", "visualization", "config.json");

/**
 * Parse the Babylon visualization config.json (UTF-8 BOM tolerated).
 * Entries with `sceneFolder`+`sceneFilename` become kind "file", entries with
 * `scriptToRun`+`functionToCall` become kind "script"; everything else (the
 * playgroundId snippet entries) is skipped and counted.
 */
export function loadConfig(babylonRoot: string): { root: string; tests: SceneEntry[]; skippedSnippets: number } {
  const configPath = join(babylonRoot, CONFIG_RELATIVE_PATH);
  let raw = readFileSync(configPath, "utf8");
  if (raw.charCodeAt(0) === 0xfeff) {
    raw = raw.slice(1);
  }

  let config: { root?: string; tests?: SceneEntry[] };
  try {
    config = JSON.parse(raw) as { root?: string; tests?: SceneEntry[] };
  } catch (error) {
    throw new Error(`[scenes] Failed to parse ${configPath}: ${(error as Error).message}`);
  }
  if (!config || !Array.isArray(config.tests)) {
    throw new Error(`[scenes] Invalid config.json at ${configPath}: expected { root, tests: [...] }`);
  }

  const tests: SceneEntry[] = [];
  let skippedSnippets = 0;
  for (const entry of config.tests) {
    if (entry.sceneFolder != null && entry.sceneFilename != null) {
      tests.push({ ...entry, kind: "file" });
    } else if (entry.scriptToRun != null && entry.functionToCall != null) {
      tests.push({ ...entry, kind: "script" });
    } else {
      skippedSnippets += 1;
    }
  }
  return { root: config.root ?? "", tests, skippedSnippets };
}

/**
 * Select runnable scenes: curated titles (plus full-only titles when `full`),
 * matched by exact title against `tests` keeping config order. Applies an
 * optional case-insensitive substring `filter`, then drops entries whose golden
 * reference file is missing from `goldenDir` (warnings for both skips).
 */
export function selectScenes(opts: { tests: SceneEntry[]; full: boolean; filter?: string; goldenDir: string }): SceneEntry[] {
  const wanted = new Set<string>(opts.full ? [...CURATED_TITLES, ...FULL_ONLY_TITLES] : CURATED_TITLES);

  for (const title of CURATED_TITLES) {
    if (!opts.tests.some((entry) => entry.title === title)) {
      console.warn(`[scenes] Curated title "${title}" not found in config.json — it will not be tested`);
    }
  }
  if (opts.full) {
    for (const title of FULL_ONLY_TITLES) {
      if (!opts.tests.some((entry) => entry.title === title)) {
        console.warn(`[scenes] Full-only title "${title}" not found in config.json — it will not be tested`);
      }
    }
  }

  let selected = opts.tests.filter((entry) => wanted.has(entry.title));

  if (opts.filter !== undefined) {
    let pattern: RegExp;
    try {
      pattern = new RegExp(opts.filter, "i");
    } catch (error) {
      throw new Error(`[scenes] Invalid --filter regex "${opts.filter}": ${(error as Error).message}`);
    }
    selected = selected.filter((entry) => pattern.test(entry.title));
  }

  return selected.filter((entry) => {
    const goldenName = sanitizeReferenceName(entry.referenceImage ?? entry.title);
    if (!existsSync(join(opts.goldenDir, goldenName))) {
      console.warn(`[scenes] Skipping "${entry.title}": golden file "${goldenName}" not found in ${opts.goldenDir}`);
      return false;
    }
    return true;
  });
}
