// TMP DIAGNOSTIC (delete before finishing): capture the FULL Babylon shader
// source + compile log for a failing standard-material scene, verbatim.
// Usage: SCENE=Polygon npx tsx src/glsl/tmp-babylon-diag.ts
import { chromium } from "playwright";
import { createServer } from "../../tests/babylon/server.ts";
import { loadConfig } from "../../tests/babylon/scenes.ts";
import { buildInterceptScript } from "../context-intercept.ts";
import { writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const TITLE = process.env.SCENE ?? "Polygon";
const cdn = process.env.BABYLON_CDN ?? "https://cdn.babylonjs.com";
// NOTE: force-override — WEBGL_SOFTWARE_RENDERER may be poisoned in the
// environment (documented gotcha); a non-existent path silently yields the
// RENDERER_NOT_FOUND stub.
process.env.WEBGL_SOFTWARE_RENDERER = process.env.DIAG_RENDERER ?? "./renderer.js";
const babylonRoot = process.env.BABYLON_ROOT ?? "/testsuites/Babylon.js";

const { tests } = loadConfig(babylonRoot);
const entry = tests.find((t) => t.title === TITLE);
if (!entry) throw new Error(`scene not found: ${TITLE}`);
console.log("scene:", TITLE, "kind:", entry.kind);

const server = await createServer({
  cdn,
  cacheDir: path.join(os.tmpdir(), "babylon-cdn-cache"),
  noCache: false,
});
const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 600, height: 400 } });
const page = await context.newPage();

page.on("console", (msg) => {
  console.log(`PAGE [${msg.type()}]: ${msg.text()}`);
});

const intercept = buildInterceptScript();
await page.addInitScript(intercept);
await page.goto(server.url + "/empty.html", { timeout: 30_000, waitUntil: "load" });
await page.waitForFunction(() => (window as any).BABYLON !== undefined);

const out = await page.evaluate(
  async (args: { root: string; entry: typeof entry }) => {
    const { root, entry } = args;
    const w = window as any;
    w.canvas = document.getElementById("babylon-canvas");
    w.engine = new BABYLON.Engine(w.canvas, false, {
      useHighPrecisionFloats: true,
      disableWebGL2Support: false,
      forceSRGBBufferSupportState: true,
      powerPreference: "high-performance",
      useLargeWorldRendering: entry.useLargeWorldRendering ?? false,
      preserveDrawingBuffer: true,
    });
    w.engine.enableOfflineSupport = false;
    w.engine.setDitheringState(false);
    w.engine.renderEvenInBackground = true;
    w.engine.getCaps().parallelShaderCompile = undefined;
    BABYLON.SceneLoader.ShowLoadingScreen = false;
    BABYLON.SceneLoader.ForceFullSceneLoadingForIncremental = true;

    // Hook the software GL context: capture every shaderSource + info log.
    const gl = w.engine._gl;
    const srcs = new Map<any, string>();
    const logs = new Map<any, string>();
    if (gl) {
      const origSS = gl.shaderSource.bind(gl);
      gl.shaderSource = (shader: any, src: string) => {
        srcs.set(shader, String(src));
        return origSS(shader, src);
      };
      const origGSIL = gl.getShaderInfoLog.bind(gl);
      gl.getShaderInfoLog = (shader: any) => {
        const l = origGSIL(shader);
        if (l) logs.set(shader, l);
        return l;
      };
    }

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
          code = code.replace(new RegExp(source, "g"), root + entry.rootPath + source);
        }
      }
      w.scene = eval(code + "\n" + entry.functionToCall + "(engine)");
    }

    w.scene.useConstantAnimationDeltaTime = true;
    // Mirror the driver's render heuristic (executeWhenReady + runRenderLoop
    // polling) — direct scene.render() calls don't wait for asset loads.
    const ready = await new Promise<boolean>((resolve) => {
      const timer = setTimeout(() => resolve(false), 60_000);
      w.scene.executeWhenReady(function () {
        if (!w.scene || !w.engine) return resolve(false);
        if (w.scene.activeCamera && w.scene.activeCamera.useAutoRotationBehavior) {
          w.scene.activeCamera.useAutoRotationBehavior = false;
        }
        w.engine.runRenderLoop(function () {
          try {
            const sceneReady = w.scene.isReady();
            if (!sceneReady) return;
            w.engine.stopRenderLoop();
            clearTimeout(timer);
            resolve(true);
          } catch (e) {
            w.engine.stopRenderLoop();
            clearTimeout(timer);
            console.error(e);
            resolve(false);
          }
        });
      }, true);
    });

    const failed: { source: string; log: string }[] = [];
    for (const [shader, src] of srcs) {
      const log = logs.get(shader);
      if (log) failed.push({ source: src, log });
    }
    return { ready, failed };
  },
  { root: server.url + "/cdn", entry }
);

writeFileSync("/tmp/babylon-diag.json", JSON.stringify(out, null, 2));
console.log("READY:", out.ready, "failed shaders:", out.failed.length);
out.failed.forEach((f, i) => {
  console.log(`=== FAILED SHADER ${i} ===`);
  console.log("--- SOURCE ---");
  console.log(f.source);
  console.log("--- LOG ---");
  console.log(f.log);
});
await browser.close();
await server.close();
