/**
 * Tests for src/context-intercept.ts — the test-harness helper that injects
 * the software renderer into browser pages. This is the only unit test file
 * that depends on a module which exists TODAY (src/context-intercept.ts is
 * real); everything else in tests/unit is written against final contracts and
 * fails until the corresponding src/ modules land.
 *
 * The "renderer missing" cases use an explicit non-existent path rather than
 * the default ./renderer.js so they keep passing once the real renderer.js is
 * built and committed.
 */
import { describe, it, expect, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildInterceptScript,
  getRendererPath,
  assertRendererExists,
} from "../../src/context-intercept";

const MISSING = "./definitely-missing-renderer.js";

afterEach(() => {
  delete process.env.WEBGL_SOFTWARE_RENDERER;
});

/** Runs `body` with WEBGL_SOFTWARE_RENDERER pointing at a real temp file. */
function withTempRenderer(body: (path: string) => void): void {
  const dir = mkdtempSync(join(tmpdir(), "swr-intercept-"));
  const path = join(dir, "renderer.js");
  writeFileSync(path, "window.__createSoftwareWebGLContext = function () {};\n");
  try {
    body(path);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

describe("getRendererPath", () => {
  it("defaults to ./renderer.js when WEBGL_SOFTWARE_RENDERER is unset", () => {
    // The outer test harness may preset this env var — never depend on it.
    delete process.env.WEBGL_SOFTWARE_RENDERER;
    expect(getRendererPath()).toBe("./renderer.js");
  });

  it("honors WEBGL_SOFTWARE_RENDERER", () => {
    process.env.WEBGL_SOFTWARE_RENDERER = "./some-renderer.js";
    expect(getRendererPath()).toBe("./some-renderer.js");
  });
});

describe("buildInterceptScript with a missing renderer", () => {
  it("produces a RENDERER_NOT_FOUND stub that throws for webgl context types", () => {
    process.env.WEBGL_SOFTWARE_RENDERER = MISSING;
    const script = buildInterceptScript();

    expect(script).toContain("RENDERER_NOT_FOUND");
    expect(script).toContain("HTMLCanvasElement.prototype.getContext");
    // The error message names the missing path so failures are diagnosable.
    expect(script).toContain(MISSING);
    // All webgl context types are intercepted…
    expect(script).toContain("'webgl'");
    expect(script).toContain("'webgl2'");
    expect(script).toContain("'experimental-webgl'");
    // …and non-webgl types ('2d') fall through to the original.
    expect(script).toContain("_orig.call(this, type, attrs)");
    // The stub must NOT pretend a renderer exists.
    expect(script).not.toContain("__createSoftwareWebGLContext");
  });

  it("patches OffscreenCanvas.getContext when the global exists", () => {
    process.env.WEBGL_SOFTWARE_RENDERER = MISSING;
    const script = buildInterceptScript();

    expect(script).toContain("OffscreenCanvas.prototype.getContext");
    // The patch is guarded so pages without OffscreenCanvas stay intact.
    expect(script).toContain("typeof OffscreenCanvas !== 'undefined'");
  });
});

describe("buildInterceptScript with a renderer present", () => {
  it("embeds the renderer source and routes webgl types through __createSoftwareWebGLContext", () => {
    withTempRenderer((path) => {
      process.env.WEBGL_SOFTWARE_RENDERER = path;
      const script = buildInterceptScript();

      expect(script).toContain("__createSoftwareWebGLContext");
      expect(script).toContain(
        "window.__createSoftwareWebGLContext(this, attrs, type)",
      );
      expect(script).not.toContain("RENDERER_NOT_FOUND");
      // Non-webgl types still fall through to the native implementation
      // (the renderer itself uses '2d' for presentation blits).
      expect(script).toContain("_orig.call(this, type, attrs)");
    });
  });

  it("patches OffscreenCanvas.getContext when the global exists", () => {
    withTempRenderer((path) => {
      process.env.WEBGL_SOFTWARE_RENDERER = path;
      const script = buildInterceptScript();

      expect(script).toContain("OffscreenCanvas.prototype.getContext");
      // The patch is guarded so pages without OffscreenCanvas stay intact.
      expect(script).toContain("typeof OffscreenCanvas !== 'undefined'");
    });
  });
});

describe("assertRendererExists", () => {
  it("throws a clear message when the renderer file is missing", () => {
    process.env.WEBGL_SOFTWARE_RENDERER = MISSING;
    expect(() => assertRendererExists()).toThrow(/Software renderer not found/);
  });

  it("passes when the renderer file exists", () => {
    withTempRenderer((path) => {
      process.env.WEBGL_SOFTWARE_RENDERER = path;
      expect(() => assertRendererExists()).not.toThrow();
    });
  });
});
