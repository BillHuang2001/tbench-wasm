# scripts/ — Build & Tooling

## Intent
Build and maintenance tooling for the repository.

## API Surface
- `scripts/build.mjs` — builds the renderer bundle: `esbuild src/entry.ts --bundle --format=iife --target=es2019 --outfile=renderer.js` (plus `--minify` optional via flag). Run via `npm run build`. Must succeed from a clean checkout (`npm install` then `npm run build`). Output is committed.
- `scripts/fetch-babylon-cdn.sh` — (existing) downloads Babylon.js UMD bundles into `vendor/babylon-cdn/` (run once; bundles are committed).
- `scripts/diag-page.ts` — single-page CTS diagnostic: serves `sdk/tests/` from the WebGL CTS repo, injects the renderer shim, runs ONE test page, and prints status, `#description` "TEST COMPLETE: N PASS, M FAIL", the per-subtest `#console` lines, and pageerrors/console errors with stacks. Usage: `NODE_PATH=/app/node_modules npx tsx scripts/diag-page.ts <conformance2/.../page.html> --webglVersion 2 --renderer ./renderer.js --timeout 150000`. Exit 0 = page PASS, 1 = FAIL. Fast for early-failing pages (~1-2 s), up to the timeout for slow ones. Console text is truncated to 8000 chars unless `--full`.

## Constraints
- Node ≥ 22; no runtime deps beyond esbuild (devDependency).
- Build must be fast (< 30s) and deterministic; no sourcemaps required (keep bundle readable for debugging — do NOT minify by default).
- The bundle must define `window.__createSoftwareWebGLContext` and run in both browser (IIFE) and Node (also assign `module.exports` when present).

## Routing Table
- No child directories. `package.json` scripts at repo root reference `scripts/build.mjs` (build) and the tests/ runners.
