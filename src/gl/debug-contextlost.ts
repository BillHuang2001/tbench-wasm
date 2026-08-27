/**
 * TEMP diagnostic: load the context-lost CTS pages and dump the failing
 * assertions (the harness shim records reportResults failure messages in
 * window.__softglCts.messages; the runner's JSON report does not surface them).
 * Usage: npx tsx src/gl/debug-contextlost.ts [url...]
 */
import { chromium } from "playwright";
import { startCtsServer } from "../../tests/conformance/server";
import { HARNESS_SHIM_SCRIPT } from "../../tests/conformance/harness";
import { buildInterceptScript } from "../../src/context-intercept";

const ctsDir = "/testsuites/WebGL";
const urls = process.argv.slice(2).length
  ? process.argv.slice(2)
  : [
      "conformance/context/context-lost.html",
      "conformance/context/context-lost-restored.html",
      "conformance/offscreencanvas/context-lost.html",
      "conformance/offscreencanvas/context-lost-restored.html",
    ];

const server = await startCtsServer({ root: ctsDir, host: "127.0.0.1", port: 0 });
process.env.WEBGL_SOFTWARE_RENDERER ??= "./renderer.js";
const intercept = buildInterceptScript();

const browser = await chromium.launch({
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
});
const context = await browser.newContext();
await context.addInitScript(intercept);
await context.addInitScript(HARNESS_SHIM_SCRIPT);

for (const url of urls) {
  const page = await context.newPage();
  const full = `${server.baseUrl}sdk/tests/${url}?webglVersion=1`;
  const msgs: string[] = [];
  const logs: string[] = [];
  page.on("pageerror", (err) => msgs.push(`pageerror: ${err.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") msgs.push(`console: ${m.text()}`);
    else if (m.type() === "log" || m.type() === "warning") logs.push(`${m.type()}: ${m.text()}`);
  });
  await page.goto(full, { waitUntil: "domcontentloaded", timeout: 30000 });
  // Poll for finish (or 30s).
  const deadline = Date.now() + 30000;
  let state: { finished: boolean } | null = null;
  for (;;) {
    state = await page.evaluate(() => (window as any).__softglCts ?? null);
    if (state?.finished || Date.now() > deadline) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  const snap = await page.evaluate(() => {
    const s = (window as any).__softglCts;
    const d = document.getElementById("description");
    return {
      pass: s?.pass ?? 0,
      fail: s?.fail ?? 0,
      skip: s?.skip ?? 0,
      finished: s?.finished ?? false,
      messages: s?.messages ?? [],
      description: d?.textContent ?? "",
    };
  });
  console.log(`\n===== ${url} (${snap.pass} pass, ${snap.fail} fail, skip ${snap.skip}, finished=${snap.finished}) =====`);
  for (const m of snap.messages) console.log(`  FAILMSG: ${m}`);
  for (const m of msgs.slice(0, 8)) console.log(`  ${m}`);
  for (const l of logs.filter((l) => l.includes("FAIL") || l.includes("expected") || l.includes("error") || l.includes("Should")).slice(0, 40)) console.log(`  ${l}`);
  if (snap.messages.length === 0 && snap.fail > 0) {
    console.log(`  (no shim messages; description tail: ${snap.description.slice(-800)})`);
  }
  await page.close();
}
await browser.close();
await server.close();
