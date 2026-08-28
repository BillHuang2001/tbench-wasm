/**
 * server.ts — static file server + /cdn proxy for the Babylon.js visual
 * regression driver (tests/babylon/).
 *
 * Routes:
 *   GET /empty.html            → tests/babylon/empty.html (text/html)
 *   GET /vendor/<file>         → vendor/babylon-cdn/<file> (whitelisted bundles only)
 *   GET /cdn/<path>            → proxy to the Babylon CDN, cached on disk
 *   anything else              → 404 text/plain
 *
 * The /cdn proxy buffers upstream bodies (cache writes need the full body
 * anyway), dedupes concurrent in-flight fetches per URL via a Map of
 * promises, and — unless `noCache` — persists every successful (200)
 * response to `cacheDir/<sha1(url)>.bin` + `cacheDir/<sha1(url)>.meta`
 * (meta = the upstream content-type string) so repeat runs are
 * offline-capable. Upstream non-200 responses are forwarded as-is but never
 * cached; network failures produce a 502 whose message includes the URL and
 * the underlying error. Every response carries no-cache headers.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import {
  createServer as createHttpServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export type ServerHandle = { port: number; url: string; close: () => Promise<void> };

/** Whitelisted vendor bundles (babylonjs.serializers.min.js on disk is NOT whitelisted). */
const VENDOR_WHITELIST = new Set([
  "earcut.min.js",
  "babylon.js",
  "babylon.gui.min.js",
  "babylonjs.materials.min.js",
  "babylonjs.loaders.min.js",
]);

const HOST = "127.0.0.1";

/** Applied to every response so nothing in the chain caches test assets. */
const NO_CACHE_HEADERS = {
  "Cache-Control": "no-cache, no-store, must-revalidate",
  Pragma: "no-cache",
} as const;

const EMPTY_HTML_PATH = fileURLToPath(new URL("empty.html", import.meta.url));
const VENDOR_DIR = fileURLToPath(new URL("../../vendor/babylon-cdn/", import.meta.url));

/** A response from the upstream CDN (or reconstructed from the on-disk cache). */
type UpstreamEntry = { status: number; body: Buffer; contentType: string };

function sha1(value: string): string {
  return createHash("sha1").update(value).digest("hex");
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function send(res: ServerResponse, status: number, body: string | Buffer, contentType: string): void {
  const buf = typeof body === "string" ? Buffer.from(body) : body;
  res.writeHead(status, {
    "Content-Type": contentType,
    "Content-Length": buf.length,
    ...NO_CACHE_HEADERS,
  });
  res.end(buf);
}

export function createServer(opts: { cdn: string; cacheDir: string; noCache?: boolean }): Promise<ServerHandle> {
  const cdnBase = opts.cdn.replace(/\/+$/, "");
  const cacheDir = opts.cacheDir;
  const noCache = opts.noCache === true;

  // Per-URL dedupe of in-flight upstream fetches: concurrent requests for the
  // same URL await the same promise; the entry is dropped once it settles.
  const inflight = new Map<string, Promise<UpstreamEntry>>();

  async function fetchUpstream(url: string): Promise<UpstreamEntry> {
    let res: Response;
    try {
      res = await fetch(url);
    } catch (err) {
      // undici wraps the real error (e.g. ECONNREFUSED) in `cause`; surface it.
      const cause = (err as { cause?: unknown }).cause;
      throw new Error(cause ? `fetch failed: ${errorMessage(cause)}` : `fetch failed: ${errorMessage(err)}`);
    }
    const contentType = res.headers.get("content-type") ?? "application/octet-stream";
    const body = Buffer.from(await res.arrayBuffer());
    if (res.status === 200 && !noCache) {
      await writeCache(url, body, contentType);
    }
    return { status: res.status, body, contentType };
  }

  function getOrFetch(url: string): Promise<UpstreamEntry> {
    const existing = inflight.get(url);
    if (existing) return existing;
    const promise = fetchUpstream(url).finally(() => inflight.delete(url));
    inflight.set(url, promise);
    return promise;
  }

  async function tryReadCache(url: string): Promise<UpstreamEntry | null> {
    const base = path.join(cacheDir, sha1(url));
    try {
      const [contentType, body] = await Promise.all([
        readFile(base + ".meta", "utf8"),
        readFile(base + ".bin"),
      ]);
      return { status: 200, body, contentType };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return null; // ordinary miss
      throw err; // genuine IO error: surface it (→ 500)
    }
  }

  async function writeCache(url: string, body: Buffer, contentType: string): Promise<void> {
    const base = path.join(cacheDir, sha1(url));
    // .bin first, .meta last: the meta file acts as the commit marker, so a
    // crash mid-write leaves an incomplete pair that reads as a cache miss.
    await writeFile(base + ".bin", body);
    await writeFile(base + ".meta", contentType, "utf8");
  }

  async function handleCdn(pathname: string, res: ServerResponse): Promise<void> {
    const pathPart = pathname.slice("/cdn/".length);
    if (pathPart.length === 0) {
      send(res, 404, "Not found\n", "text/plain; charset=utf-8");
      return;
    }
    const upstreamUrl = `${cdnBase}/${pathPart}`;
    if (!noCache) {
      const hit = await tryReadCache(upstreamUrl);
      if (hit) {
        send(res, 200, hit.body, hit.contentType);
        return;
      }
    }
    let entry: UpstreamEntry;
    try {
      entry = await getOrFetch(upstreamUrl);
    } catch (err) {
      send(res, 502, `CDN proxy error for ${upstreamUrl}: ${errorMessage(err)}\n`, "text/plain; charset=utf-8");
      return;
    }
    // Upstream non-200 (e.g. 404) is forwarded as-is; it is never cached.
    send(res, entry.status, entry.body, entry.contentType);
  }

  async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (!req.url || req.method !== "GET") {
      send(res, 404, "Not found\n", "text/plain; charset=utf-8");
      return;
    }
    let pathname: string;
    try {
      pathname = new URL(req.url, "http://localhost").pathname;
    } catch {
      send(res, 400, "Bad request\n", "text/plain; charset=utf-8");
      return;
    }

    if (pathname === "/empty.html") {
      const body = await readFile(EMPTY_HTML_PATH);
      send(res, 200, body, "text/html; charset=utf-8");
    } else if (pathname.startsWith("/vendor/")) {
      const name = pathname.slice("/vendor/".length);
      if (!VENDOR_WHITELIST.has(name)) {
        send(res, 404, `Not found: /vendor/${name}\n`, "text/plain; charset=utf-8");
        return;
      }
      const body = await readFile(path.join(VENDOR_DIR, name));
      send(res, 200, body, "application/javascript");
    } else if (pathname.startsWith("/cdn/")) {
      await handleCdn(pathname, res);
    } else {
      send(res, 404, "Not found\n", "text/plain; charset=utf-8");
    }
  }

  const server = createHttpServer((req, res) => {
    handleRequest(req, res).catch((err) => {
      if (!res.headersSent) {
        send(res, 500, `Internal server error: ${errorMessage(err)}\n`, "text/plain; charset=utf-8");
      } else {
        res.destroy();
      }
    });
  });

  return (async () => {
    if (!noCache) await mkdir(cacheDir, { recursive: true });

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      server.on("error", (err) => {
        if (settled) {
          // Post-listen errors (e.g. accept failures) must not crash the run.
          console.error(`[tests/babylon server] error after listen: ${errorMessage(err)}`);
          return;
        }
        settled = true;
        reject(err); // listen failure → reject createServer
      });
      server.listen(0, HOST, () => {
        if (settled) return;
        settled = true;
        resolve();
      });
    });

    const addr = server.address() as AddressInfo | null;
    if (!addr) throw new Error("server.address() returned null after listen");
    const port = addr.port;
    const url = `http://${HOST}:${port}`;

    let closePromise: Promise<void> | null = null;
    const close = (): Promise<void> => {
      closePromise ??= new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
        // Don't wait on idle keep-alive sockets from fetch/Playwright.
        server.closeIdleConnections();
      });
      return closePromise;
    };

    return { port, url, close };
  })();
}
