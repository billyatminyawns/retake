// In-process test: an MCP Client talks to the Retake MCP server over an
// in-memory transport, while the server talks to a tiny mock of the Retake
// HTTP API (node http). No models, no network, no WellSaid credits.

import assert from "node:assert/strict";
import http from "node:http";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import { createServer, TOOL_NAMES } from "../src/server.js";
import { RetakeClient } from "../src/retake.js";

// ------------------------------------------------------------- mock API

const T = "?t=tok123";
const OCC = [
  { id: "o1", word: "use", t_start: 13.12, t_end: 13.4, sentence: "Says to use the bookmark.", intended: "use_v",
    intended_respell: "YOOZ", intended_desc: "to employ (YOOZ)", detected: "use_v", detected_desc: "to employ (YOOZ)",
    margin: 0.378, free_decode: "sæztijuːz", flag: false, confidence: "med", reason: "ok", intent_engine: "claude",
    clip_wav: "/v1/jobs/j1/files/work/clips/o1_use_wide.wav" + T, sentence_wav: "/v1/jobs/j1/files/work/clips/o1_sentence.wav" + T },
  { id: "o3", word: "re-records", t_start: 38.46, t_end: 39.1, sentence: "It re-records just that line.", intended: "record_v",
    intended_respell: "ree-ruh-KORDZ", intended_desc: "to capture audio (rih-KORD)", detected: "record_n",
    detected_desc: "stored information (REH-kurd)", margin: 1.4126, free_decode: "ɪtɹiːɹɛkɚdz", flag: true, confidence: "med",
    reason: "best candidate is record_n (ree-REH-kurdz), not intended record_v (ree-ruh-KORDZ)", intent_engine: "claude",
    clip_wav: "/v1/jobs/j1/files/work/clips/o3_re_records_wide.wav" + T, sentence_wav: "/v1/jobs/j1/files/work/clips/o3_sentence.wav" + T },
];
const state = { jobPolls: 0, fixPolls: 0, calls: [], lastJobBody: null, lastFixBody: null, lastHeaders: null };

function jobDoc(status) {
  const base = { id: "j1", status, stage: status === "done" ? "done" : "listening", progress: status === "done" ? 1 : 0.6,
    created: "2026-10-12T06:00:00Z", mode: "upload", filename: "film.mp4", duration: 65.9, occurrences: 2, flags: 1, error: null,
    params: {}, files: {}, fixes: [] };
  if (status !== "done") return base;
  return { ...base,
    files: { media: "/v1/jobs/j1/files/media.mp4" + T, report_json: "/v1/jobs/j1/files/report.json" + T, report_html: "/v1/jobs/j1/files/report.html" + T },
    result: { duration: 65.9, flags: 1, verdict: "issues", engine: "claude", occurrences: OCC, summary: ["2 heteronym occurrences, 1 flagged"] } };
}
function fixDoc(status) {
  const base = { fix_id: "f1", job_id: "j1", status, stage: status === "done" ? "done" : "rendering", progress: status === "done" ? 1 : 0.3,
    created: "2026-10-12T06:01:00Z", speaker_id: 169, model: "preview", occurrence_ids: ["o3"], results: [], summary: {}, error: null };
  if (status !== "done") return base;
  const t1 = { n: 1, strategy: "spelling", respell: "ree-recordz", text: "It ree-recordz just that line.", verified: true, margin: 0.27,
    detected: "record_v", heard_as: "ɪtɹiːɹᵻkɔːɹdz", take_mp3: "/v1/jobs/j1/files/fixes/f1/o3/try1.mp3" + T, word_wav: "/v1/jobs/j1/files/fixes/f1/o3/try1_word.wav" + T };
  const t2 = { ...t1, n: 2, respell: "ree-ruh-kordz", text: "It ree-ruh-kordz just that line.", margin: 0.88,
    take_mp3: "/v1/jobs/j1/files/fixes/f1/o3/try2.mp3" + T, word_wav: "/v1/jobs/j1/files/fixes/f1/o3/try2_word.wav" + T };
  return { ...base, results: [{ occurrence_id: "o3", word: "re-records", intended: "record_v", intended_respell: "ree-ruh-KORDZ",
    strategy: "spelling", tries: [t1, t2], best: t2, verified: true, strong: true, summary: "re-records fixed on try 2/3",
    before: { detected: "record_n", detected_desc: "stored information (REH-kurd)", free_decode: "ɪtɹiːɹɛkɚdz", clip_wav: OCC[1].clip_wav } }],
    summary: { requested: 1, done: 1, verified: 1, unverified: 0, errors: 0 } };
}

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
  });
}

const mock = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  const body = await readBody(req);
  state.calls.push({ method: req.method, path: url.pathname });
  state.lastHeaders = req.headers;
  const send = (code, obj) => {
    res.writeHead(code, { "Content-Type": "application/json" });
    res.end(JSON.stringify(obj));
  };
  if (req.headers["x-api-key"] !== "rtk_test") return send(401, { error: { code: "unauthorized", message: "Missing X-Api-Key header." } });
  if (req.method === "GET" && url.pathname === "/v1/health") return send(200, { ok: true, version: "1.0.0", auth: "key", models: { whisper: "loaded", phonemes: "cold" }, wellsaid: true, anthropic: true, queue: 0 });
  if (req.method === "GET" && url.pathname === "/v1/voices") return send(200, [{ speaker_id: 26, name: "Wade C.", style: "Promo" }, { speaker_id: 169, name: "Jennie V.", style: "Narration", accent: "English (United States)", gender: "female" }]);
  if (req.method === "POST" && url.pathname === "/v1/check-text") {
    const j = JSON.parse(body.toString());
    return send(200, { text: j.text, count: 1, engine: "spacy", occurrences: [{ id: "o1", word: "record", base: "record", sentence: j.text, char_start: 0, char_end: 6, intended: "record_v", intended_respell: "ruh-KORD", intended_desc: "to capture (rih-KORD)", intent_confidence: 0.75, respell_candidates: ["ruh-KORD", "REH-kurd"] }], summary: [] });
  }
  if (req.method === "POST" && url.pathname === "/v1/jobs") {
    const ct = req.headers["content-type"] || "";
    state.lastJobBody = ct.includes("multipart/form-data") ? { multipart: true, hasMedia: body.includes('name="media"'), hasScript: body.includes('name="script"'), size: body.length } : JSON.parse(body.toString());
    return send(202, { job_id: "j1", status: "queued", status_url: "/v1/jobs/j1" });
  }
  if (req.method === "GET" && url.pathname === "/v1/jobs") return send(200, { count: 1, jobs: [jobDoc("done")] });
  if (req.method === "GET" && url.pathname === "/v1/jobs/j1") {
    state.jobPolls += 1;
    return send(200, jobDoc(state.jobPolls < 2 ? "running" : "done"));
  }
  if (req.method === "POST" && url.pathname === "/v1/jobs/j1/fix") {
    state.lastFixBody = JSON.parse(body.toString());
    return send(202, { fix_id: "f1", job_id: "j1", status: "queued", status_url: "/v1/jobs/j1/fixes/f1" });
  }
  if (req.method === "GET" && url.pathname === "/v1/jobs/j1/fixes/f1") {
    state.fixPolls += 1;
    return send(200, fixDoc(state.fixPolls < 2 ? "running" : "done"));
  }
  if (req.method === "GET" && url.pathname === "/v1/jobs/nope") return send(404, { error: { code: "not_found", message: "Job nope does not exist." } });
  send(404, { error: { code: "not_found", message: "no route " + url.pathname } });
});
await new Promise((r) => mock.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${mock.address().port}`;

// ------------------------------------------------------------- MCP wiring

const client = new RetakeClient({ baseUrl: base, apiKey: "rtk_test" });
const server = createServer(client);
const [ct, st] = InMemoryTransport.createLinkedPair();
const mcp = new Client({ name: "test", version: "0.0.0" });
await server.connect(st);
await mcp.connect(ct);

let passed = 0;
const ok = (l) => {
  passed += 1;
  console.log(`  ✓ ${l}`);
};
const textOf = (r) => r.content.find((c) => c.type === "text").text;
const jsonOf = (r) => JSON.parse(textOf(r).slice(textOf(r).indexOf("\n\n{") + 2));

// 1) tools
const { tools } = await mcp.listTools();
assert.deepEqual(tools.map((t) => t.name).sort(), [...TOOL_NAMES].sort());
ok(`lists ${tools.length} retake_* tools`);

// 2) health
let r = await mcp.callTool({ name: "retake_health", arguments: {} });
assert.ok(!r.isError, textOf(r));
assert.match(textOf(r), /auth key \(key set\)/);
assert.match(textOf(r), /whisper loaded, phonemes cold/);
ok("health renders auth + model state");

// 3) voices with filter
r = await mcp.callTool({ name: "retake_list_voices", arguments: { query: "jennie" } });
assert.match(textOf(r), /1 voice matching "jennie"/);
assert.match(textOf(r), /speaker_id 169 — Jennie V\. · Narration/);
ok("list_voices filters by query");

// 4) check-text
r = await mcp.callTool({ name: "retake_check_text", arguments: { text: "Record the session." } });
assert.match(textOf(r), /1 heteronym occurrence/);
assert.match(textOf(r), /"record" in "Record the session\." → ruh-KORD/);
ok("check_text summarises intended pronunciations");

// 5) check_pronunciation needs exactly one source
r = await mcp.callTool({ name: "retake_check_pronunciation", arguments: {} });
assert.ok(r.isError);
assert.match(textOf(r), /exactly one of media_path, media_url, or text/);
r = await mcp.callTool({ name: "retake_check_pronunciation", arguments: { text: "Record it." } });
assert.ok(r.isError);
assert.match(textOf(r), /needs speaker_id/);
ok("check_pronunciation validates its source arguments");

// 6) media_path route: uploads multipart, polls to done, formats flagged
const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "retake-mcp-"));
const media = path.join(tmp, "film.mp4");
await fs.writeFile(media, Buffer.alloc(2048, 1));
r = await mcp.callTool({ name: "retake_check_pronunciation", arguments: { media_path: media, script: "It re-records just that line." } });
assert.ok(!r.isError, textOf(r));
assert.equal(state.lastJobBody.multipart, true);
assert.ok(state.lastJobBody.hasMedia && state.lastJobBody.hasScript, "multipart carries media + script");
assert.ok(state.jobPolls >= 2, "polled until done");
let txt = textOf(r);
assert.match(txt, /job j1 \(done\): 2 heteronyms, 1 mispronounced in 65\.9 s/);
assert.match(txt, /✗ \[38\.5s\] re-records — meant ree-ruh-KORDZ/);
assert.match(txt, /heard REH-kurd/);
assert.match(txt, /margin 1\.41 · confidence med · id o3/);
assert.doesNotMatch(txt, /✓ \[13\.1s\] use/, "only flagged by default");
let data = jsonOf(r);
assert.equal(data.job_id, "j1");
assert.equal(data.counts.flagged, 1);
assert.equal(data.flagged[0].clip_url, `${base}/v1/jobs/j1/files/work/clips/o3_re_records_wide.wav${T}`);
assert.equal(data.report_html_url, `${base}/v1/jobs/j1/files/report.html${T}`);
assert.equal(state.lastHeaders["x-api-key"], "rtk_test");
ok("check_pronunciation(media_path) uploads, waits, and formats the flagged occurrence with absolute URLs");

// 7) text route sends JSON
r = await mcp.callTool({ name: "retake_check_pronunciation", arguments: { text: "Record it.", speaker_id: 169, model: "preview", wait: false } });
assert.ok(!r.isError, textOf(r));
assert.deepEqual(state.lastJobBody, { text: "Record it.", speaker_id: 169, model: "preview" });
assert.match(textOf(r), /job j1 queued/);
ok("check_pronunciation(text) renders first; wait=false returns the job id");

// 8) get_job with all occurrences
r = await mcp.callTool({ name: "retake_get_job", arguments: { job_id: "j1", only_flagged: false } });
assert.match(textOf(r), /✓ \[13\.1s\] use — YOOZ as intended/);
ok("get_job(only_flagged=false) lists clean occurrences too");

// 9) fix: posts body, polls, formats tries + best
r = await mcp.callTool({ name: "retake_fix_pronunciation", arguments: { job_id: "j1", occurrence_ids: ["o3"], speaker_id: 169, model: "preview", max_tries: 3 } });
assert.ok(!r.isError, textOf(r));
assert.deepEqual(state.lastFixBody, { speaker_id: 169, model: "preview", occurrence_ids: ["o3"], max_tries: 3 });
txt = textOf(r);
assert.match(txt, /fix f1 on job j1 \(done\): 1 requested · 1 verified · 0 unverified · 0 errors · model preview · voice 169/);
assert.match(txt, /✓ re-records \(o3\): VERIFIED on try 2\/2 — spelling "ree-ruh-kordz" · heard \/ɪtɹiːɹᵻkɔːɹdz\/ · margin 0\.88/);
assert.match(txt, /try 1 spelling "ree-recordz": verified · heard/);
data = jsonOf(r);
assert.equal(data.results[0].best.take_url, `${base}/v1/jobs/j1/files/fixes/f1/o3/try2.mp3${T}`);
ok("fix_pronunciation posts, waits, and formats tries/best with take URLs");

// 10) API errors come back clean
r = await mcp.callTool({ name: "retake_get_job", arguments: { job_id: "nope" } });
assert.ok(r.isError);
assert.match(textOf(r), /404: Job nope does not exist/);
ok("API errors are clean tool errors");

// 11) wrong key -> actionable 401
const bad = new RetakeClient({ baseUrl: base, apiKey: "wrong" });
await assert.rejects(() => bad.health(), /401.*RETAKE_API_KEY/);
ok("401 explains RETAKE_API_KEY");

await mcp.close();
mock.close();
await fs.rm(tmp, { recursive: true, force: true });
console.log(`\nmock-api test passed (${passed} checks).`);
