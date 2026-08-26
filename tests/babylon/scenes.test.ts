import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CURATED_TITLES, FULL_ONLY_TITLES, loadConfig, selectScenes } from "./scenes";
import type { SceneEntry } from "./scenes";

/**
 * Local copy of compare.ts's sanitizeReferenceName rule (compare.ts is a
 * placeholder in this worktree): strip extension, collapse runs of characters
 * outside [A-Za-z0-9._-] to "-", append ".png".
 */
const sanitizeReferenceName = (name: string): string =>
  name.replace(/\.[^/.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "-") + ".png";

/** File-kind curated titles (mirrors the real config: 10 of the 16). */
const FILE_TITLES = [
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
];

const CONFIG_SUBDIR = join("packages", "tools", "tests", "test", "visualization");

let tmpDirs: string[] = [];

function makeBabylonRoot(): string {
  const dir = mkdtempSync(join(tmpdir(), "babylon-scenes-test-"));
  tmpDirs.push(dir);
  return dir;
}

function fileEntry(title: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { title, sceneFolder: `/Scenes/${title}/`, sceneFilename: `${title}.babylon`, ...extra };
}

function scriptEntry(title: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { title, scriptToRun: `/Demos/${title}/${title}.js`, functionToCall: `Create${title}Scene`, ...extra };
}

function entryFor(title: string): Record<string, unknown> {
  return FILE_TITLES.includes(title) || title === "Mansion" ? fileEntry(title) : scriptEntry(title);
}

/** All 18 curated titles (16 default + 2 full-only). */
function allCuratedTitles(): string[] {
  return [...CURATED_TITLES, ...FULL_ONLY_TITLES];
}

function writeConfig(root: string, tests: unknown[]): string {
  const configPath = join(root, CONFIG_SUBDIR, "config.json");
  mkdirSync(join(root, CONFIG_SUBDIR), { recursive: true });
  writeFileSync(configPath, "\uFEFF" + JSON.stringify({ root: "https://cdn.babylonjs.com", tests }));
  return configPath;
}

function writeGoldens(root: string, names: string[]): string {
  const goldenDir = join(root, CONFIG_SUBDIR, "ReferenceImages");
  mkdirSync(goldenDir, { recursive: true });
  for (const name of names) {
    writeFileSync(join(goldenDir, name), "dummy golden");
  }
  return goldenDir;
}

afterEach(() => {
  for (const dir of tmpDirs) {
    rmSync(dir, { recursive: true, force: true });
  }
  tmpDirs = [];
  vi.restoreAllMocks();
});

describe("loadConfig", () => {
  it("strips the UTF-8 BOM, detects kinds, counts skipped snippets, keeps fields verbatim", () => {
    const root = makeBabylonRoot();
    writeConfig(root, [
      {
        title: "Sponza",
        sceneFolder: "/Scenes/Sponza/",
        sceneFilename: "Sponza.babylon",
        referenceImage: "Sponza.png",
        errorRatio: 5,
        dependsOn: ["XR"],
      },
      {
        title: "Fog",
        scriptToRun: "/Demos/Fog/fog.js",
        functionToCall: "CreateFogScene",
        referenceImage: "fog.png",
        replace: "a, b",
      },
      { title: "Gaussian Splatting GLTF", playgroundId: "#WSAFDA#0", referenceImage: "gsplat-splat-gltf.png", renderCount: 15 },
    ]);

    const { root: configRoot, tests, skippedSnippets } = loadConfig(root);

    expect(configRoot).toBe("https://cdn.babylonjs.com");
    expect(skippedSnippets).toBe(1);
    expect(tests).toHaveLength(2);

    expect(tests[0]).toMatchObject({
      kind: "file",
      title: "Sponza",
      sceneFolder: "/Scenes/Sponza/",
      sceneFilename: "Sponza.babylon",
      referenceImage: "Sponza.png",
      errorRatio: 5,
    });
    // Extra fields beyond SceneEntry must survive loadConfig verbatim.
    expect((tests[0] as SceneEntry & { dependsOn?: string[] }).dependsOn).toEqual(["XR"]);

    expect(tests[1]).toMatchObject({
      kind: "script",
      title: "Fog",
      scriptToRun: "/Demos/Fog/fog.js",
      functionToCall: "CreateFogScene",
      referenceImage: "fog.png",
      replace: "a, b",
    });
  });

  it("throws a clear error for a malformed config", () => {
    const root = makeBabylonRoot();
    writeConfig(root, [{ title: "Sponza" }]);
    // Corrupt the file after writing: replace the BOM+JSON with garbage.
    writeFileSync(join(root, CONFIG_SUBDIR, "config.json"), "{ not json");
    expect(() => loadConfig(root)).toThrow(/Failed to parse/);
  });
});

describe("selectScenes", () => {
  it("picks the 16 curated entries, keeping config order", () => {
    const root = makeBabylonRoot();
    // Reverse order in the config to prove config order (not CURATED_TITLES order) wins.
    const titles = [...CURATED_TITLES].reverse();
    writeConfig(root, titles.map((title) => entryFor(title)));
    const goldenDir = writeGoldens(root, titles.map((title) => sanitizeReferenceName(title)));

    const selected = selectScenes({ tests: loadConfig(root).tests, full: false, goldenDir });

    expect(selected).toHaveLength(CURATED_TITLES.length);
    expect(selected.map((e) => e.title)).toEqual(titles);
    expect(new Set(selected.map((e) => e.title))).toEqual(new Set(CURATED_TITLES));
  });

  it("with full selects all 18 curated + full-only entries; without full selects 16", () => {
    const root = makeBabylonRoot();
    const titles = allCuratedTitles();
    writeConfig(root, titles.map((title) => entryFor(title)));
    const goldenDir = writeGoldens(root, titles.map((title) => sanitizeReferenceName(title)));
    const tests = loadConfig(root).tests;

    const full = selectScenes({ tests, full: true, goldenDir });
    expect(full).toHaveLength(18);
    expect(full.map((e) => e.title)).toEqual(titles);

    const curated = selectScenes({ tests, full: false, goldenDir });
    expect(curated).toHaveLength(16);
    expect(curated.map((e) => e.title)).toEqual(CURATED_TITLES);
  });

  it("filter applies a case-insensitive substring regex", () => {
    const root = makeBabylonRoot();
    const titles = allCuratedTitles();
    writeConfig(root, titles.map((title) => entryFor(title)));
    const goldenDir = writeGoldens(root, titles.map((title) => sanitizeReferenceName(title)));
    const tests = loadConfig(root).tests;

    expect(selectScenes({ tests, full: true, goldenDir, filter: "sponza" }).map((e) => e.title)).toEqual(["Sponza"]);
    expect(selectScenes({ tests, full: true, goldenDir, filter: "WINDOWS" }).map((e) => e.title)).toEqual(["Windows cafe"]);
    expect(selectScenes({ tests, full: true, goldenDir, filter: "self shadowing" }).map((e) => e.title)).toEqual([
      "Self shadowing",
    ]);
    expect(selectScenes({ tests, full: true, goldenDir, filter: "zzz-no-match" })).toEqual([]);
  });

  it("filter throws a clear error on an invalid regex", () => {
    const root = makeBabylonRoot();
    writeConfig(root, allCuratedTitles().map((title) => entryFor(title)));
    const goldenDir = writeGoldens(root, allCuratedTitles().map((title) => sanitizeReferenceName(title)));

    expect(() =>
      selectScenes({ tests: loadConfig(root).tests, full: true, goldenDir, filter: "[" }),
    ).toThrow(/Invalid --filter regex "\["/);
  });

  it("skips entries whose golden file is missing, with a warning", () => {
    const root = makeBabylonRoot();
    writeConfig(root, [fileEntry("Sponza"), scriptEntry("Fog"), fileEntry("Windows cafe")]);
    // Only Sponza.png and Fog.png exist — Windows-cafe.png is missing.
    const goldenDir = writeGoldens(root, [sanitizeReferenceName("Sponza"), sanitizeReferenceName("Fog")]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const selected = selectScenes({ tests: loadConfig(root).tests, full: false, goldenDir });

    expect(selected.map((e) => e.title)).toEqual(["Sponza", "Fog"]);
    const messages = warn.mock.calls.map((call) => String(call[0])).join("\n");
    expect(messages).toContain("Windows cafe");
    expect(messages).toContain("Windows-cafe.png");
    expect(messages).not.toContain("Sponza");
  });

  it("uses referenceImage (when present) for the golden file name", () => {
    const root = makeBabylonRoot();
    writeConfig(root, [fileEntry("Sponza", { referenceImage: "renamed-sponza.png" })]);
    // Golden exists under the referenceImage-derived name, not the title-derived one.
    const goldenDir = writeGoldens(root, [sanitizeReferenceName("renamed-sponza.png")]);

    const selected = selectScenes({ tests: loadConfig(root).tests, full: false, goldenDir });

    expect(selected.map((e) => e.title)).toEqual(["Sponza"]);
  });

  it("warns when a curated title is missing from config and returns only what matches", () => {
    const root = makeBabylonRoot();
    const withoutGui = CURATED_TITLES.filter((title) => title !== "GUI");
    writeConfig(root, withoutGui.map((title) => entryFor(title)));
    const goldenDir = writeGoldens(root, withoutGui.map((title) => sanitizeReferenceName(title)));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const selected = selectScenes({ tests: loadConfig(root).tests, full: false, goldenDir });

    expect(selected.map((e) => e.title)).toEqual(withoutGui);
    expect(selected).toHaveLength(CURATED_TITLES.length - 1);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('"GUI"'));
  });

  it("returns an empty list when no curated title is present in config, warning for each", () => {
    const root = makeBabylonRoot();
    writeConfig(root, [{ title: "Unrelated", playgroundId: "#ABC#0" }]);
    const goldenDir = writeGoldens(root, []);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const selected = selectScenes({ tests: loadConfig(root).tests, full: false, goldenDir });

    expect(selected).toEqual([]);
    expect(warn).toHaveBeenCalledTimes(CURATED_TITLES.length);
  });

  it("warns when a full-only title is missing from config in full mode", () => {
    const root = makeBabylonRoot();
    writeConfig(root, CURATED_TITLES.map((title) => entryFor(title)));
    const goldenDir = writeGoldens(root, CURATED_TITLES.map((title) => sanitizeReferenceName(title)));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const selected = selectScenes({ tests: loadConfig(root).tests, full: true, goldenDir });

    expect(selected).toHaveLength(16);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('"Mansion"'));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('"Procedural textures"'));
  });
});

// ---- Real-config sanity checks (skipped automatically when the foreign repo is absent) ----
const REAL_BABYLON_ROOT = "/testsuites/Babylon.js";
const realConfigExists = existsSync(join(REAL_BABYLON_ROOT, "packages", "tools", "tests", "test", "visualization", "config.json"));

describe.runIf(realConfigExists)("real config.json", () => {
  it("loads 18 runnable tests (11 file + 7 script) and skips 835 snippets", () => {
    const { root, tests, skippedSnippets } = loadConfig(REAL_BABYLON_ROOT);

    expect(root).toBe("https://cdn.babylonjs.com");
    expect(skippedSnippets).toBe(835);
    expect(tests).toHaveLength(18);
    expect(tests.filter((t) => t.kind === "file")).toHaveLength(11);
    expect(tests.filter((t) => t.kind === "script")).toHaveLength(7);
  });

  it("contains every curated and full-only title verbatim", () => {
    const { tests } = loadConfig(REAL_BABYLON_ROOT);
    const titles = tests.map((t) => t.title);

    for (const title of [...CURATED_TITLES, ...FULL_ONLY_TITLES]) {
      expect(titles).toContain(title);
    }
  });

  it("has a matching golden file on disk for every curated entry", () => {
    const { tests } = loadConfig(REAL_BABYLON_ROOT);
    const goldenDir = join(REAL_BABYLON_ROOT, "packages", "tools", "tests", "test", "visualization", "ReferenceImages");
    const missing = tests.filter((t) => !existsSync(join(goldenDir, sanitizeReferenceName(t.referenceImage ?? t.title))));

    expect(missing.map((t) => t.title)).toEqual([]);
  });
});
