import { buildInterceptScript } from "../../src/context-intercept";
import type { SceneEntry } from "./scenes";

// BABYLON is a UMD global injected by empty.html (vendor/babylon.js).
declare const BABYLON: any;

/**
 * Per-scene result of the Babylon visual regression driver.
 *
 * `screenshot` is the raw PNG (Buffer) of the final page state and is ALWAYS
 * attempted — even when the scene failed, a blank page is a valid diff input.
 */
export type SceneResult = {
  title: string;
  reference: string;
  kind: "file" | "script";
  ok: boolean;
  ready: boolean;
  error?: string;
  timeout?: boolean;
  elapsedMs: number;
  screenshot?: Buffer;
};

/**
 * Runs one Babylon scene in a fresh browser context:
 *   1. injects the software renderer (via buildInterceptScript)
 *   2. loads empty.html and waits for the BABYLON global
 *   3. creates the engine (options mirror the upstream harness, minus
 *      failIfMajorPerformanceCaveat — a software renderer would trip it)
 *   4. prepares the scene (file kind: SceneLoader.LoadAsync; script kind:
 *      fetch + asset rewrites + eval, ported verbatim from upstream)
 *   5. renders `renderCount` frames until the scene is ready
 *   6. screenshots the page
 *
 * The whole flow races against `opts.sceneTimeoutMs`; the context is always
 * closed (try/finally) and the timeout timer is always cleaned up. In stub
 * mode (no renderer.js) the injected getContext throws inside
 * `new BABYLON.Engine(...)`; the error is recorded with a RENDERER_NOT_FOUND
 * prefix so failures are easy to grep.
 */
export function runScene(
  browser: import("playwright").Browser,
  serverUrl: string,
  entry: SceneEntry,
  opts: { sceneTimeoutMs: number; rendererPath: string }
): Promise<SceneResult> {
  const startedAt = Date.now();

  return new Promise<SceneResult>((resolve) => {
    let timer: NodeJS.Timeout | undefined;
    let context: import("playwright").BrowserContext | undefined;
    let contextClosed = false;
    let settled = false;

    const closeContext = async () => {
      if (context && !contextClosed) {
        contextClosed = true;
        try {
          await context.close();
        } catch {
          // Context may already be gone (browser closed, timeout path) — nothing to do.
        }
      }
    };

    type SettleInput = {
      ok: boolean;
      ready: boolean;
      error?: string;
      timeout?: boolean;
      screenshot?: Buffer;
    };
    const settle = (result: SettleInput) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      resolve({
        title: entry.title,
        reference: entry.referenceImage ?? entry.title,
        kind: entry.kind,
        elapsedMs: Date.now() - startedAt,
        ...result,
      });
    };

    // Hard timeout: race the whole flow; on fire, grab a last screenshot if
    // the page is still alive, settle the timeout result, then close the
    // context. Settling BEFORE closing is important: closing the context
    // makes any in-flight page call reject, and the timeout result must win
    // that race.
    timer = setTimeout(() => {
      void (async () => {
        let screenshot: Buffer | undefined;
        if (context && !contextClosed) {
          try {
            const pages = context.pages();
            if (pages.length > 0) {
              screenshot = await pages[0].screenshot({ type: "png", timeout: 5_000 });
            }
          } catch {
            // Page is dead — nothing to capture.
          }
        }
        settle({
          ok: false,
          ready: false,
          timeout: true,
          error: `timeout after ${Math.round(opts.sceneTimeoutMs / 1000)}s`,
          screenshot,
        });
        await closeContext();
      })();
    }, opts.sceneTimeoutMs);

    void (async () => {
      try {
        // buildInterceptScript reads WEBGL_SOFTWARE_RENDERER at call time —
        // set it from opts so the CLI --renderer flag takes effect.
        process.env.WEBGL_SOFTWARE_RENDERER = opts.rendererPath;
        const interceptScript = buildInterceptScript();

        context = await browser.newContext({ viewport: { width: 600, height: 400 } });
        const page = await context.newPage();
        await page.addInitScript(interceptScript);
        await page.goto(serverUrl + "/empty.html", { timeout: 30_000, waitUntil: "load" });
        await page.waitForSelector("#babylon-canvas");
        await page.waitForFunction(() => (window as any).BABYLON !== undefined);

        let error: string | undefined;
        let ready = false;

        // 1) Engine init. MUST catch everything in-page — never throw across
        // the bridge. In stub mode getContext throws inside the constructor;
        // prefix the recorded error with RENDERER_NOT_FOUND.
        const engineInit = await page.evaluate(
          (useLargeWorldRendering: boolean) => {
            try {
              const canvas = document.getElementById("babylon-canvas") as HTMLCanvasElement;
              const w = window as any;
              w.engine = new BABYLON.Engine(canvas, false, {
                useHighPrecisionFloats: true,
                disableWebGL2Support: false,
                forceSRGBBufferSupportState: true,
                powerPreference: "high-performance",
                useLargeWorldRendering: useLargeWorldRendering,
                // DEVIATION (documented): failIfMajorPerformanceCaveat is
                // omitted — a software renderer would trip it.
              });
              w.engine.enableOfflineSupport = false;
              w.engine.setDitheringState(false);
              w.engine.renderEvenInBackground = true;
              w.engine.getCaps().parallelShaderCompile = undefined;
              BABYLON.SceneLoader.ShowLoadingScreen = false;
              BABYLON.SceneLoader.ForceFullSceneLoadingForIncremental = true;
              return { ok: true };
            } catch (e) {
              const msg = String(e);
              const rendererMissing =
                document.title === "RENDERER_NOT_FOUND" ||
                /renderer not found/i.test(msg);
              return {
                ok: false,
                error: (rendererMissing ? "RENDERER_NOT_FOUND: " : "") + msg,
              };
            }
          },
          entry.useLargeWorldRendering ?? false
        );

        if (!engineInit.ok) {
          error = engineInit.error;
        } else {
          // 2) Scene prep. The deterministic seed is set HERE — not during
          // engine init — because Babylon may call Math.random while
          // creating the engine.
          const scenePrep = await page.evaluate(
            async (args: { root: string; entry: SceneEntry }) => {
              const { root, entry } = args;
              try {
                const w = window as any;
                w.seed = 1;
                w.Math.random = function () {
                  const x = Math.sin(w.seed++) * 10000;
                  return x - Math.floor(x);
                };
                if (entry.kind === "file") {
                  w.scene = await BABYLON.SceneLoader.LoadAsync(
                    root + entry.sceneFolder,
                    entry.sceneFilename,
                    w.engine
                  );
                } else {
                  if (entry.specificRoot) {
                    BABYLON.Tools.BaseUrl = root + entry.specificRoot;
                  }
                  let code = await (await fetch(root + entry.scriptToRun)).text();
                  code = code.replace(/..\/..\/assets\//g, root + "/Assets/");
                  code = code.replace(/..\/..\/Assets\//g, root + "/Assets/");
                  code = code.replace(/\/assets\//g, root + "/Assets/");
                  if (entry.replace) {
                    const split = entry.replace.split(",");
                    for (let i = 0; i < split.length; i += 2) {
                      const source = split[i].trim();
                      const destination = split[i + 1].trim();
                      code = code.replace(source, destination);
                    }
                  }
                  if (entry.replaceUrl) {
                    const split = entry.replaceUrl.split(",");
                    for (let i = 0; i < split.length; i++) {
                      const source = split[i].trim();
                      code = code.replace(
                        new RegExp(source, "g"),
                        root + entry.rootPath + source
                      );
                    }
                  }
                  // Note: `engine` resolves to window.engine (the page's
                  // scripts are sloppy mode) — matches upstream exactly.
                  w.scene = eval(code + "\n" + entry.functionToCall + "(engine)");
                }
                return { ok: true };
              } catch (e) {
                return { ok: false, error: String(e) };
              }
            },
            { root: serverUrl + "/cdn", entry }
          );

          if (!scenePrep.ok) {
            error = scenePrep.error;
          } else {
            // 3) Render heuristic — exact port of upstream
            // evaluateRenderSceneForVisualization.
            ready = await page.evaluate(
              (args: { renderCount: number; continueRenderingOnDone: boolean }) => {
                let { renderCount, continueRenderingOnDone } = args;
                return new Promise<boolean>((resolve) => {
                  const w = window as any;
                  if (!w.scene || !w.engine) return resolve(false);
                  w.scene.useConstantAnimationDeltaTime = true;
                  w.scene.executeWhenReady(function () {
                    if (!w.scene || !w.engine) return resolve(false);
                    if (
                      w.scene.activeCamera &&
                      w.scene.activeCamera.useAutoRotationBehavior
                    ) {
                      w.scene.activeCamera.useAutoRotationBehavior = false;
                    }
                    w.engine.runRenderLoop(function () {
                      try {
                        if (renderCount <= 0) {
                          const sceneReady = w.scene.isReady();
                          if (!sceneReady || !continueRenderingOnDone) {
                            w.engine.stopRenderLoop();
                          }
                          if (sceneReady) {
                            if (continueRenderingOnDone) {
                              w.scene.render();
                            }
                          } else {
                            console.error(
                              "Scene is not ready after rendering is done"
                            );
                          }
                          return resolve(sceneReady);
                        } else {
                          w.scene.render();
                          renderCount--;
                        }
                      } catch (e) {
                        w.engine.stopRenderLoop();
                        console.error(e);
                        return resolve(false);
                      }
                    });
                  }, true);
                });
              },
              { renderCount: entry.renderCount ?? 1, continueRenderingOnDone: false }
            );
            if (!ready) {
              error = error ?? "scene is not ready after rendering";
            }
          }
        }

        // 4) Screenshot — ALWAYS attempted, even after failures above (a
        // blank page is a valid diff input).
        let screenshot: Buffer | undefined;
        try {
          screenshot = await page.screenshot({ type: "png" });
        } catch (e) {
          error = (error ? error + "; " : "") + "screenshot failed: " + String(e);
        }

        await closeContext();
        settle({
          ok: !error && ready,
          ready,
          error,
          screenshot,
        });
      } catch (e) {
        // Flow-level failure (goto / waitForFunction / evaluate bridge...).
        settle({ ok: false, ready: false, error: String(e) });
      } finally {
        await closeContext();
        if (!settled) {
          settle({ ok: false, ready: false, error: "scene flow did not complete" });
        }
      }
    })();
  });
}
