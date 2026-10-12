// LIVE run against a real Retake API (not part of `npm test`).
//
//   node test/live.smoke.js check <media_path> [script_file]
//   node test/live.smoke.js fix <job_id> <speaker_id> <preview|caruso> [occurrence_id ...]
//   node test/live.smoke.js tool <tool_name> '<json args>'
//
// Spawns the MCP server over stdio exactly like Claude would, calls one tool,
// prints the tool's text result. Env: RETAKE_API_BASE, RETAKE_API_KEY.
// The `fix` subcommand renders through WellSaid and SPENDS credits.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const entry = fileURLToPath(new URL("../src/index.js", import.meta.url));
const [cmd, ...rest] = process.argv.slice(2);

let name, args;
if (cmd === "check") {
  const [media, scriptFile] = rest;
  if (!media) usage();
  name = "retake_check_pronunciation";
  args = { media_path: path.resolve(media), only_flagged: false };
  if (scriptFile) args.script = readFileSync(scriptFile, "utf8");
} else if (cmd === "fix") {
  const [jobId, speaker, model, ...ids] = rest;
  if (!jobId || !speaker || !model) usage();
  name = "retake_fix_pronunciation";
  args = { job_id: jobId, speaker_id: Number(speaker), model };
  if (ids.length) args.occurrence_ids = ids;
} else if (cmd === "tool") {
  const [tool, json] = rest;
  if (!tool) usage();
  name = tool;
  args = json ? JSON.parse(json) : {};
} else {
  usage();
}

function usage() {
  console.error("usage: node test/live.smoke.js check <media> [script] | fix <job_id> <speaker_id> <model> [occ ...] | tool <name> '<json>'");
  process.exit(2);
}

const transport = new StdioClientTransport({
  command: process.execPath,
  args: [entry],
  env: { ...process.env, MCP_TRANSPORT: "stdio" },
  stderr: "inherit",
  cwd: path.dirname(entry),
});
const client = new Client({ name: "live", version: "0.0.0" });
await client.connect(transport);
const t0 = Date.now();
console.log(`→ ${name} ${JSON.stringify({ ...args, script: args.script ? `(${args.script.length} chars)` : undefined })}`);
const r = await client.callTool({ name, arguments: args }, undefined, {
  timeout: 60 * 60 * 1000,
  resetTimeoutOnProgress: true,
  onprogress: (p) => console.error(`  … ${p.message || ""} ${p.progress ?? ""}/${p.total ?? ""}`),
});
console.log(`← ${r.isError ? "ERROR" : "ok"} in ${((Date.now() - t0) / 1000).toFixed(1)}s\n`);
for (const c of r.content) if (c.type === "text") console.log(c.text);
await client.close();
process.exit(r.isError ? 1 : 0);
