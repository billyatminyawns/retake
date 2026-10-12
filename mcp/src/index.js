#!/usr/bin/env node
// Retake MCP server — entry point.
//
// Transports:
//   (default)  stdio            — Claude Desktop custom connector / Claude Code
//   --http     Streamable HTTP  — remote/hosted connector (POST /mcp)
//   --doctor   config + connectivity check against the Retake API, then exit
//
// In stdio mode stdout is the JSON-RPC channel: diagnostics go to stderr only.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

import { createServer, PKG_VERSION, TOOL_NAMES } from "./server.js";
import { RetakeClient } from "./retake.js";

// .env: this folder first, then the retake-cloud project root (so a local API
// and this MCP share RETAKE_API_KEYS). Values already in the environment win.
for (const rel of ["../.env", "../../.env"]) loadEnv(fileURLToPath(new URL(rel, import.meta.url)));

function loadEnv(path) {
  let raw;
  try {
    raw = readFileSync(path, "utf8");
  } catch (err) {
    if (err?.code !== "ENOENT") console.error(`[retake] could not read ${path}: ${err?.message ?? err}`);
    return;
  }
  for (const line of raw.split(/\r?\n/)) {
    const m = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(line);
    if (!m) continue;
    let val = m[2].trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) val = val.slice(1, -1);
    if (process.env[m[1]] === undefined) process.env[m[1]] = val;
  }
}

const args = process.argv.slice(2);
const useHttp = args.includes("--http") || process.env.MCP_TRANSPORT === "http";
const doDoctor = args.includes("--doctor");

async function runDoctor() {
  const client = new RetakeClient();
  console.log(`Retake MCP ${PKG_VERSION} — doctor\n`);
  console.log(`  API base:   ${client.baseUrl}`);
  console.log(`  API key:    ${client.hasApiKey() ? "set ✓" : "not set (fine when the API runs in open mode)"}`);
  console.log(`  Node:       ${process.version}`);
  console.log(`  Tools:      ${TOOL_NAMES.join(", ")}`);
  process.stdout.write("\n  GET /v1/health ... ");
  try {
    const h = await client.health();
    console.log(`OK — version ${h.version}, auth ${h.auth}, whisper ${h.models?.whisper}, phonemes ${h.models?.phonemes}, wellsaid ${h.wellsaid ? "yes" : "no"}, anthropic ${h.anthropic ? "yes" : "no"}`);
    if (h.auth === "key" && !client.hasApiKey()) {
      console.log("\n  The API requires a key: set RETAKE_API_KEY (one of the server's RETAKE_API_KEYS).");
      process.exit(1);
    }
    console.log("\nAll good.");
    process.exit(0);
  } catch (err) {
    console.log("FAILED");
    console.log(`  ${err?.message ?? err}`);
    process.exit(1);
  }
}

async function runStdio() {
  const server = createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`[retake] MCP server ${PKG_VERSION} running on stdio → ${process.env.RETAKE_API_BASE || "http://localhost:8080"}`);
}

async function runHttp() {
  const { StreamableHTTPServerTransport } = await import("@modelcontextprotocol/sdk/server/streamableHttp.js");
  const http = await import("node:http");
  const port = Number(process.env.PORT || 8788);

  const httpServer = http.createServer(async (req, res) => {
    if (req.url !== "/mcp") {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not found. POST JSON-RPC to /mcp");
      return;
    }
    // Stateless: a fresh server + transport per request (simple + safe for a
    // demo/remote connector).
    try {
      const body = await readJson(req);
      const server = createServer();
      const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
      res.on("close", () => {
        transport.close();
        server.close();
      });
      await server.connect(transport);
      await transport.handleRequest(req, res, body);
    } catch (err) {
      const status = err?.statusCode || 500;
      console.error("[retake] http error:", err?.message || err);
      if (!res.headersSent) {
        res.writeHead(status, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ jsonrpc: "2.0", error: { code: -32603, message: String(err?.message || err) }, id: null }));
      }
    }
  });
  httpServer.on("error", (err) => {
    console.error("[retake] HTTP transport error:", err?.message || err);
    process.exit(1);
  });
  httpServer.listen(port, () => {
    console.error(`[retake] MCP server running on http://localhost:${port}/mcp (Streamable HTTP)`);
  });
}

const MAX_HTTP_BODY = 4 * 1024 * 1024;

function readJson(req) {
  return new Promise((resolve, reject) => {
    const declared = Number(req.headers["content-length"]);
    if (declared && declared > MAX_HTTP_BODY) {
      const e = new Error("Payload too large");
      e.statusCode = 413;
      return reject(e);
    }
    const chunks = [];
    let received = 0;
    req.on("data", (c) => {
      received += c.length;
      if (received > MAX_HTTP_BODY) {
        const e = new Error("Payload too large");
        e.statusCode = 413;
        req.destroy();
        return reject(e);
      }
      chunks.push(c);
    });
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw) return resolve(undefined);
      try {
        resolve(JSON.parse(raw));
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

if (doDoctor) {
  runDoctor();
} else if (useHttp) {
  runHttp().catch((err) => {
    console.error("[retake] failed to start HTTP transport:", err);
    process.exit(1);
  });
} else {
  runStdio().catch((err) => {
    console.error("[retake] failed to start:", err);
    process.exit(1);
  });
}
