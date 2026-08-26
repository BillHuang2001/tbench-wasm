/**
 * No-cache static HTTP server for the CTS, mirroring serve_localhost.py:
 * - Cache-Control: no-cache, no-store, must-revalidate + Pragma: no-cache +
 *   Expires: 0 on every response.
 * - Root is the CTS repository root; test URLs are /sdk/tests/<path>.
 *
 * CROSS-ORIGIN NOTE (verified in js/webgl-test-utils.js): the six
 * origin-clean CTS tests load the conformance-resources logo from
 * getLocalCrossOrigin(), which flips the loopback hostname (127.0.0.1 <-> 
 * localhost) on the SAME port — NOT a second port, and NOT the imgUrl
 * override (imgUrl is only consulted when NOT running on localhost). So this
 * server binds BOTH 127.0.0.1 and ::1 on the same port; Chromium resolving
 * "localhost" (IPv6-first) and "127.0.0.1" both reach it.
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const NO_CACHE_HEADERS: Readonly<Record<string, string>> = {
  "Cache-Control": "no-cache, no-store, must-revalidate",
  Pragma: "no-cache",
  Expires: "0",
};

/** Extension -> MIME map covering every extension present under sdk/tests. */
const MIME_TYPES: Readonly<Record<string, string>> = {
  html: "text/html",
  htm: "text/html",
  js: "text/javascript",
  mjs: "text/javascript",
  css: "text/css",
  txt: "text/plain",
  json: "application/json",
  xml: "text/xml",
  md: "text/markdown",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  svg: "image/svg+xml",
  ico: "image/x-icon",
  webp: "image/webp",
  bmp: "image/bmp",
  psd: "image/vnd.adobe.photoshop",
  hdr: "image/vnd.radiance",
  mp4: "video/mp4",
  webm: "video/webm",
  ogg: "video/ogg",
  wav: "audio/wav",
  mp3: "audio/mpeg",
  wasm: "application/wasm",
  // Shader sources and misc text assets (fetched via XHR as text):
  vert: "text/plain",
  frag: "text/plain",
  vs: "text/plain",
  glsl: "text/plain",
  test: "text/plain",
  template: "text/plain",
  data: "application/octet-stream",
};

function mimeFor(filePath: string): string {
  const ext = path.extname(filePath).slice(1).toLowerCase();
  return MIME_TYPES[ext] ?? "application/octet-stream";
}

export interface CtsServer {
  baseUrl: string;
  port: number;
  close(): Promise<void>;
}

export interface ServerOptions {
  /** CTS repository root (directory containing sdk/tests). */
  root: string;
  /** Primary bind host; defaults to 127.0.0.1. */
  host: string;
  /** Port to bind; 0 = OS-assigned. */
  port: number;
}

function serveFile(res: http.ServerResponse, filePath: string, pathname: string): void {
  fs.stat(filePath, (statErr, st) => {
    if (statErr || !st.isFile()) {
      res.writeHead(404, NO_CACHE_HEADERS);
      res.end(`Not found: ${pathname}`);
      return;
    }
    res.writeHead(200, { ...NO_CACHE_HEADERS, "Content-Type": mimeFor(filePath) });
    if (res.req.method === "HEAD") {
      res.end();
      return;
    }
    const stream = fs.createReadStream(filePath);
    stream.on("error", () => res.destroy());
    stream.pipe(res);
  });
}

function handleRequest(req: http.IncomingMessage, res: http.ServerResponse, root: string): void {
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405, NO_CACHE_HEADERS);
    res.end("Method not allowed");
    return;
  }
  let pathname: string;
  try {
    pathname = decodeURIComponent(new URL(req.url ?? "/", "http://localhost").pathname);
  } catch {
    res.writeHead(400, NO_CACHE_HEADERS);
    res.end("Bad request");
    return;
  }
  const filePath = path.resolve(root, pathname.replace(/^\/+/, ""));
  if (filePath !== root && !filePath.startsWith(root + path.sep)) {
    res.writeHead(403, NO_CACHE_HEADERS);
    res.end("Forbidden");
    return;
  }
  serveFile(res, filePath, pathname);
}

/**
 * Starts the CTS server. When `host` is a loopback address, also binds the
 * complementary loopback (127.0.0.1 <-> ::1) so the origin-clean tests'
 * cross-origin logo fetch (getLocalCrossOrigin: same port, flipped hostname)
 * succeeds regardless of how Chromium resolves "localhost". If the IPv6 bind
 * fails (no IPv6), the IPv4 bind alone is used — Chromium falls back from
 * ::1 to 127.0.0.1 when the former is refused.
 */
export async function startCtsServer(opts: ServerOptions): Promise<CtsServer> {
  const root = path.resolve(opts.root);
  if (!fs.existsSync(path.join(root, "sdk", "tests"))) {
    throw new Error(`CTS root invalid: ${root} (expected ${path.join(root, "sdk", "tests")} to exist)`);
  }

  const server = http.createServer((req, res) => handleRequest(req, res, root));

  const listen = (port: number, host: string): Promise<number> =>
    new Promise<number>((resolve, reject) => {
      const onError = (err: Error) => {
        server.removeListener("listening", onListening);
        reject(err);
      };
      const onListening = () => {
        server.removeListener("error", onError);
        const addr = server.address();
        resolve(typeof addr === "object" && addr ? addr.port : port);
      };
      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(port, host);
    });

  const port = await listen(opts.port, opts.host);
  const loopbackAliases: Record<string, string | null> = { "127.0.0.1": "::1", "::1": "127.0.0.1" };
  const alias = loopbackAliases[opts.host];
  if (alias) {
    try {
      await listen(port, alias);
    } catch {
      // No IPv6 (or no IPv4) — the primary binding suffices; Chromium retries
      // the other loopback address on connect failure.
    }
  }

  const close = () =>
    new Promise<void>((resolve) => {
      server.close(() => resolve());
    });

  return { baseUrl: `http://${opts.host}:${port}/`, port, close };
}
