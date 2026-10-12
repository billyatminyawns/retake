// Real-process smoke test: spawn the MCP server exactly as an MCP client would
// (node src/index.js over stdio), pointed at an API base nothing listens on:
//   - the process starts and completes the MCP handshake
//   - tools/list returns the retake_* tools
//   - a tool call returns a clean, actionable error (API unreachable), not a crash

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import { TOOL_NAMES } from "../src/server.js";

const entry = fileURLToPath(new URL("../src/index.js", import.meta.url));

const transport = new StdioClientTransport({
  command: process.execPath,
  args: [entry],
  env: { ...process.env, RETAKE_API_BASE: "http://127.0.0.1:1", RETAKE_API_KEY: "", MCP_TRANSPORT: "stdio" },
  stderr: "pipe",
  cwd: path.dirname(entry),
});

const client = new Client({ name: "smoke", version: "0.0.0" });
await client.connect(transport);
console.log("  ✓ handshake completed (server initialized over stdio)");

const { tools } = await client.listTools();
assert.equal(tools.length, TOOL_NAMES.length);
console.log(`  ✓ tools/list returned ${tools.length} tools`);

const r = await client.callTool({ name: "retake_health", arguments: {} });
assert.ok(r.isError, "expected an error result when the API is unreachable");
assert.match(r.content[0].text, /Cannot reach the Retake API/);
console.log("  ✓ unreachable API returns a clean, actionable error");

await client.close();
console.log("\nstdio smoke test passed.");
