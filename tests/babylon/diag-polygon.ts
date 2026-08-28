/**
 * TEMPORARY diagnostic (delete after use): runs the Polygon scene like
 * driver.ts but captures page console + pageerror output, to explain why
 * scene.isReady() never becomes true (harness-side black-screen investigation).
 */
import { chromium } from "playwright";
import sharp from "sharp";
import { buildInterceptScript } from "../../src/context-intercept";
import { loadConfig, selectScenes } from "./scenes";
import { createServer } from "./server";

const ROOT = "/testsuites/Babylon.js";

async function main() {
  process.env.WEBGL_SOFTWARE_RENDERER = "./renderer.js";
  const interceptScript = buildInterceptScript();

  const config = loadConfig(ROOT);
  const goldenDir = ROOT + "/packages/tools/tests/test/visualization/ReferenceImages";
  const selected = selectScenes({ tests: config.tests, full: false, filter: "Polygon", goldenDir });
  const entry = selected[0];
  console.log("entry:", entry.title, entry.kind, entry.scriptToRun, entry.functionToCall);

  const server = await createServer({
    cdn: "https://cdn.babylonjs.com",
    cacheDir: "tests/reports/babylon-cache",
  });
  const serverUrl = server.url;

  const browser = await chromium.launch({ headless: true, args: ["--disable-gpu"] });
  const context = await browser.newContext({ viewport: { width: 600, height: 400 } });
  const page = await context.newPage();

  const consoleLines: string[] = [];
  const pageErrors: string[] = [];
  page.on("console", (msg) => {
    const t = msg.type();
    if (t === "error" || t === "warning" || t === "log") {
      consoleLines.push(`[${t}] ${msg.text().slice(0, 300)}`);
    }
  });
  page.on("pageerror", (err) => pageErrors.push(String(err).slice(0, 300)));

  await page.addInitScript(interceptScript);
  await page.goto(serverUrl + "/empty.html", { timeout: 30_000, waitUntil: "load" });
  await page.waitForFunction(() => (window as any).BABYLON !== undefined);
  const factory = await page.evaluate(
    () => typeof (window as any).__createSoftwareWebGLContext === "function"
  );
  console.log("rendererActive:", factory);

  const engineInit = await page.evaluate(() => {
    try {
      const w = window as any;
      w.canvas = document.getElementById("babylon-canvas");
      w.engine = new BABYLON.Engine(w.canvas, false, {
        useHighPrecisionFloats: true,
        disableWebGL2Support: false,
        forceSRGBBufferSupportState: true,
        powerPreference: "high-performance",
        preserveDrawingBuffer: true,
      });
      w.engine.enableOfflineSupport = false;
      w.engine.setDitheringState(false);
      w.engine.renderEvenInBackground = true;
      w.engine.getCaps().parallelShaderCompile = undefined;
      BABYLON.SceneLoader.ShowLoadingScreen = false;
      BABYLON.SceneLoader.ForceFullSceneLoadingForIncremental = true;
      return { ok: true };
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  });
  console.log("engineInit:", JSON.stringify(engineInit));

  const scenePrep = await page.evaluate(
    async (args: { root: string; scriptToRun?: string; functionToCall?: string }) => {
      const { root, scriptToRun, functionToCall } = args;
      try {
        const w = window as any;
        w.seed = 1;
        w.Math.random = function () {
          const x = Math.sin(w.seed++) * 10000;
          return x - Math.floor(x);
        };
        let code = await (await fetch(root + scriptToRun)).text();
        code = code.replace(/..\/..\/assets\//g, root + "/Assets/");
        code = code.replace(/..\/..\/Assets\//g, root + "/Assets/");
        code = code.replace(/\/assets\//g, root + "/Assets/");
        w.scene = eval(code + "\n" + functionToCall + "(engine)");
        return { ok: true };
      } catch (e) {
        return { ok: false, error: String(e) };
      }
    },
    { root: serverUrl + "/cdn", scriptToRun: entry.scriptToRun, functionToCall: entry.functionToCall }
  );
  console.log("scenePrep:", JSON.stringify(scenePrep));

  if (scenePrep.ok) {
    const render = await page.evaluate(
      () =>
        new Promise<{ ready: boolean; isReady: boolean; meshes: number; activeMeshes: number }>((resolve) => {
          const w = window as any;
          const t0 = Date.now();
          const timer = setTimeout(() => {
            console.error("DIAG: executeWhenReady never fired after 30s");
            const s = w.scene;
            resolve({
              ready: false,
              isReady: s ? s.isReady() : false,
              meshes: s ? s.meshes.length : -1,
              activeMeshes: s && s.activeCamera ? s.activeCamera.getActiveMeshes().length : -1,
            });
          }, 30_000);
          w.scene.useConstantAnimationDeltaTime = true;
          w.scene.executeWhenReady(function () {
            clearTimeout(timer);
            console.log("DIAG: executeWhenReady fired after " + (Date.now() - t0) + "ms");
            w.engine.runRenderLoop(function () {
              try {
                w.scene.render();
                w.engine.stopRenderLoop();
                resolve({
                  ready: true,
                  isReady: w.scene.isReady(),
                  meshes: w.scene.meshes.length,
                  activeMeshes: w.scene.activeCamera
                    ? w.scene.activeCamera.getActiveMeshes().length
                    : -1,
                });
              } catch (e) {
                w.engine.stopRenderLoop();
                console.error("DIAG: render threw: " + e);
                resolve({
                  ready: false,
                  isReady: w.scene.isReady(),
                  meshes: w.scene.meshes.length,
                  activeMeshes: -1,
                });
              }
            });
          }, true);
        })
    );
    console.log("render:", JSON.stringify(render));
  }

  const screenshot = await page.screenshot({ type: "png" });
  const img = sharp(screenshot).removeAlpha().ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const { data, info } = await img;
  let nonBlack = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > 8 || data[i + 1] > 8 || data[i + 2] > 8) nonBlack++;
  }
  console.log(`screenshot: ${info.width}x${info.height}, nonBlackPixels=${nonBlack}`);

  console.log("\n=== console (last 60) ===");
  for (const l of consoleLines.slice(-60)) console.log(l);
  console.log("\n=== pageerrors ===");
  for (const e of pageErrors) console.log(e);

  await context.close();
  await browser.close();
  await server.close();
}

main().catch((e) => {
  console.error("diag failed:", e);
  process.exit(1);
});
