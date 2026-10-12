// Builds the MCP server and registers the Retake tools.
//
// Transport-agnostic: index.js wires this server up to stdio or Streamable
// HTTP. Every tool is a thin call to the Retake HTTP API (src/retake.js); the
// heavy lifting (whisper, phoneme CTC scoring, WellSaid renders) happens in
// the API process. Tools are namespaced `retake_*`.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

import { RetakeClient, RetakeError } from "./retake.js";

export const PKG_VERSION = "1.0.0";

const DEFAULT_TIMEOUT_S = 20 * 60;

function text(value) {
  const s = typeof value === "string" ? value : JSON.stringify(value ?? null, null, 2);
  return { content: [{ type: "text", text: s ?? "null" }] };
}
function errorResult(msg) {
  return { content: [{ type: "text", text: msg }], isError: true };
}
function guard(fn) {
  return async (args, extra) => {
    try {
      return await fn(args, extra);
    } catch (err) {
      if (err instanceof RetakeError) {
        let msg = `Retake error: ${err.message}`;
        if (err.partial) msg += `\n\n${JSON.stringify(compactPartial(err.partial), null, 2)}`;
        return errorResult(msg);
      }
      return errorResult(`Unexpected error: ${err?.stack || err?.message || String(err)}`);
    }
  };
}

function compactPartial(p) {
  return { id: p.id || p.fix_id, job_id: p.job_id, status: p.status, stage: p.stage, progress: p.progress };
}

/** Send MCP progress notifications when the client asked for them. */
function progressReporter(extra, label) {
  const token = extra?._meta?.progressToken;
  if (token === undefined || typeof extra?.sendNotification !== "function") return () => {};
  let last = -1;
  return (obj) => {
    const pct = Math.round((obj.progress || 0) * 100);
    if (pct === last) return;
    last = pct;
    extra
      .sendNotification({
        method: "notifications/progress",
        params: { progressToken: token, progress: pct, total: 100, message: `${label}: ${obj.stage || obj.status} ${pct}%` },
      })
      .catch(() => {});
  };
}

const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(Number(x)) ? "n/a" : Number(x).toFixed(d));
const hint = (desc) => {
  const m = /\(([^()]*)\)\s*$/.exec(desc || "");
  return m ? m[1].trim() : "";
};

// ------------------------------------------------------------------ formatting

function compactOccurrence(o, client) {
  return {
    id: o.id,
    word: o.word,
    t_start: o.t_start,
    t_end: o.t_end,
    sentence: o.sentence,
    flag: Boolean(o.flag),
    confidence: o.confidence,
    intended: o.intended,
    intended_respell: o.intended_respell,
    intended_desc: o.intended_desc,
    detected: o.detected,
    detected_respell: hint(o.detected_desc) || null,
    detected_desc: o.detected_desc,
    free_decode: o.free_decode,
    margin: o.margin,
    reason: o.reason,
    intent_engine: o.intent_engine,
    clip_url: client.absolute(o.clip_wav),
    sentence_url: client.absolute(o.sentence_wav),
  };
}

function occurrenceLine(o) {
  const mark = o.flag ? "✗" : "✓";
  const t = o.t_start === undefined || o.t_start === null ? "" : `[${num(o.t_start, 1)}s] `;
  if (o.flag) {
    return (
      `  ${mark} ${t}${o.word} — meant ${o.intended_respell || o.intended} (${o.intended_desc || o.intended}), ` +
      `heard ${o.detected_respell || o.detected || "?"}${o.detected_desc ? ` (${o.detected_desc})` : ""}` +
      ` · phonemes /${o.free_decode || "?"}/ · margin ${num(o.margin)} · confidence ${o.confidence || "?"} · id ${o.id}`
    );
  }
  return `  ${mark} ${t}${o.word} — ${o.intended_respell || o.intended} as intended (margin ${num(o.margin)}) · id ${o.id}`;
}

function formatJob(job, client, { onlyFlagged = true } = {}) {
  const r = job.result || {};
  const occs = (r.occurrences || []).map((o) => compactOccurrence(o, client));
  const flagged = occs.filter((o) => o.flag);
  const files = job.files || {};
  const out = {
    job_id: job.id,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    mode: job.mode,
    filename: job.filename,
    duration_s: r.duration ?? job.duration,
    verdict: r.verdict,
    intent_engine: r.engine || (r.intent && r.intent.engine),
    counts: { heteronyms: occs.length, flagged: flagged.length },
    flagged,
    ...(onlyFlagged ? {} : { all: occs }),
    report_html_url: client.absolute(files.report_html),
    report_json_url: client.absolute(files.report_json),
    media_url: client.absolute(files.media),
    fixes: job.fixes || [],
    error: job.error || null,
  };
  const lines = [];
  if (job.status !== "done") {
    lines.push(`Retake job ${job.id}: ${job.status} (${job.stage || "?"}, ${Math.round((job.progress || 0) * 100)}%)`);
    if (job.error) lines.push(`  error: ${job.error.message || JSON.stringify(job.error)}`);
    lines.push(`  Poll with retake_get_job {"job_id": "${job.id}"}.`);
    return { summary: lines.join("\n"), data: out };
  }
  lines.push(
    `Retake check — job ${job.id} (done): ${occs.length} heteronym${occs.length === 1 ? "" : "s"}, ` +
      `${flagged.length} mispronounced in ${num(out.duration_s, 1)} s of audio · verdict: ${r.verdict || "n/a"}` +
      (out.intent_engine ? ` · intent: ${out.intent_engine}` : "")
  );
  const shown = onlyFlagged ? flagged : occs;
  for (const o of shown) lines.push(occurrenceLine(o));
  if (onlyFlagged && !flagged.length && occs.length) lines.push("  every heteronym was pronounced as intended");
  if (!occs.length) lines.push("  no heteronyms from the lexicon occur in this audio");
  if (out.report_html_url) lines.push(`Report: ${out.report_html_url}`);
  if (flagged.length) {
    lines.push(
      `Next: retake_fix_pronunciation {"job_id": "${job.id}", "speaker_id": <WellSaid voice>, "model": "preview"|"caruso"}` +
        ` re-renders those lines and verifies the takes.`
    );
  }
  return { summary: lines.join("\n"), data: out };
}

function compactTry(t, client) {
  return {
    n: t.n,
    strategy: t.strategy,
    value: t.respell,
    text: t.text,
    verified: Boolean(t.verified),
    margin: t.margin,
    detected: t.detected,
    heard_phonemes: t.heard_as,
    reason: t.reason,
    error: t.error,
    take_url: client.absolute(t.take_mp3),
    word_clip_url: client.absolute(t.word_wav),
  };
}

function formatFix(fix, client) {
  const results = (fix.results || []).map((res) => {
    const tries = (res.tries || []).map((t) => compactTry(t, client));
    const best = res.best ? compactTry(res.best, client) : null;
    return {
      occurrence_id: res.occurrence_id,
      word: res.word,
      intended: res.intended,
      intended_respell: res.intended_respell,
      strategy: res.strategy,
      verified: Boolean(res.verified),
      strong: res.strong,
      best,
      tries,
      summary: res.summary,
      error: res.error || null,
      before: res.before
        ? {
            detected: res.before.detected,
            detected_desc: res.before.detected_desc,
            free_decode: res.before.free_decode,
            clip_url: client.absolute(res.before.clip_wav),
          }
        : null,
    };
  });
  const s = fix.summary || {};
  const out = {
    fix_id: fix.fix_id,
    job_id: fix.job_id,
    status: fix.status,
    stage: fix.stage,
    progress: fix.progress,
    model: fix.model,
    speaker_id: fix.speaker_id,
    summary: s,
    results,
    error: fix.error || null,
  };
  const lines = [];
  if (fix.status !== "done") {
    lines.push(`Retake fix ${fix.fix_id} on job ${fix.job_id}: ${fix.status} (${fix.stage || "?"}, ${Math.round((fix.progress || 0) * 100)}%)`);
    if (fix.error) lines.push(`  error: ${fix.error.message || JSON.stringify(fix.error)}`);
    lines.push(`  Poll with retake_get_fix {"job_id": "${fix.job_id}", "fix_id": "${fix.fix_id}"}.`);
    return { summary: lines.join("\n"), data: out };
  }
  lines.push(
    `Retake fix ${fix.fix_id} on job ${fix.job_id} (done): ${s.requested ?? results.length} requested · ` +
      `${s.verified ?? 0} verified · ${s.unverified ?? 0} unverified · ${s.errors ?? 0} errors · model ${fix.model || "?"} · voice ${fix.speaker_id ?? "?"}`
  );
  for (const r of results) {
    const b = r.best;
    if (r.verified && b) {
      lines.push(
        `  ✓ ${r.word} (${r.occurrence_id}): VERIFIED on try ${b.n}/${r.tries.length} — ${b.strategy} "${b.value}" · heard /${b.heard_phonemes || "?"}/ · ` +
          `margin ${num(b.margin)}${r.strong === false ? " (weak)" : ""}${b.take_url ? ` · take: ${b.take_url}` : ""}`
      );
    } else if (r.error) {
      lines.push(`  ✗ ${r.word} (${r.occurrence_id}): error — ${r.error.message || r.error}`);
    } else {
      lines.push(`  ✗ ${r.word} (${r.occurrence_id}): not verified after ${r.tries.length} tr${r.tries.length === 1 ? "y" : "ies"}${b?.take_url ? ` · closest take: ${b.take_url}` : ""}`);
    }
    for (const t of r.tries) {
      if (b && t.n === b.n) continue;
      lines.push(
        `      try ${t.n} ${t.strategy} "${t.value}": ${t.error ? `error ${t.error}` : `${t.verified ? "verified" : "not verified"} · heard /${t.heard_phonemes || "?"}/ · margin ${num(t.margin)}`}`
      );
    }
  }
  return { summary: lines.join("\n"), data: out };
}

function result(summary, data) {
  return text(`${summary}\n\n${JSON.stringify(data, null, 2)}`);
}

// ------------------------------------------------------------------ server

/**
 * Create and configure the MCP server.
 * @param {RetakeClient} [client]
 */
export function createServer(client = new RetakeClient()) {
  const server = new McpServer(
    { name: "retake", version: PKG_VERSION },
    {
      instructions:
        "Retake — pronunciation QA for AI voiceover. It finds heteronyms (record/record, live/live, " +
        "read/read…) in audio or video, decides from the script which pronunciation was meant, scores what " +
        "the voice actually said at the phoneme level, and fixes flagged lines by re-rendering them through " +
        "WellSaid until a take verifies. Typical flow: retake_check_pronunciation (media_path, media_url, or " +
        "text+speaker_id to render first) → read the flagged occurrences → retake_fix_pronunciation with a " +
        "WellSaid speaker_id and model ('preview' for the newest voices, 'caruso' for AI Director voices). " +
        "Checks are free; fixes render through WellSaid and spend credits (≤ 3 renders per flagged line). " +
        "Use retake_check_text to pre-flight a script before rendering it.",
    }
  );

  const engineSchema = z
    .enum(["claude", "spacy", "both"])
    .optional()
    .describe("Intent engine for 'which sense did the script mean': claude (default when the API has a key), spacy, or both.");
  const modelSchema = z.enum(["preview", "caruso"]).describe("WellSaid model: 'preview' (newest voices; fixes spell the word phonetically) or 'caruso' (AI Director; fixes use an inline <respell> tag).");

  // ---------------------------------------------------------------- health
  server.registerTool(
    "retake_health",
    {
      title: "Retake API health",
      description: "Check the Retake API: version, auth mode, whether the whisper + phoneme models are warm, and whether WellSaid/Anthropic keys are configured.",
      inputSchema: {},
    },
    guard(async () => {
      const h = await client.health();
      const lines = [
        `Retake API ${h.version} at ${client.baseUrl}: ${h.ok ? "ok" : "NOT ok"} · auth ${h.auth}${h.auth === "key" ? (client.hasApiKey() ? " (key set)" : " (NO RETAKE_API_KEY set — calls will fail)") : ""}`,
        `  models: whisper ${h.models?.whisper}, phonemes ${h.models?.phonemes} · wellsaid key ${h.wellsaid ? "yes" : "no"} · anthropic key ${h.anthropic ? "yes (Claude intent)" : "no (spaCy intent)"} · queue ${h.queue ?? 0}`,
      ];
      return result(lines.join("\n"), h);
    })
  );

  // ---------------------------------------------------------------- voices
  server.registerTool(
    "retake_list_voices",
    {
      title: "List WellSaid voices",
      description: "List WellSaid voices (speaker_id, name, style, accent, gender) available for fixes and script renders. Filter with `query` (case-insensitive substring over name/style/accent).",
      inputSchema: {
        query: z.string().optional().describe("Substring filter, e.g. 'Jennie' or 'Narration'"),
        limit: z.number().int().min(1).max(500).optional().describe("Max rows (default 60)"),
      },
    },
    guard(async ({ query, limit }) => {
      const all = await client.voices();
      const list = Array.isArray(all) ? all : all?.voices || [];
      const q = (query || "").toLowerCase();
      const rows = list.filter((v) => !q || [v.name, v.style, v.accent, v.gender].some((x) => String(x || "").toLowerCase().includes(q)));
      const shown = rows.slice(0, limit || 60);
      const lines = shown.map((v) => `  speaker_id ${v.speaker_id} — ${v.name || "?"}${v.style ? ` · ${v.style}` : ""}${v.accent ? ` · ${v.accent}` : ""}${v.gender ? ` · ${v.gender}` : ""}`);
      const head = `${rows.length} voice${rows.length === 1 ? "" : "s"}${q ? ` matching "${query}"` : ""}${shown.length < rows.length ? ` (showing ${shown.length})` : ""}`;
      return text([head, ...lines].join("\n"));
    })
  );

  // ---------------------------------------------------------------- check-text
  server.registerTool(
    "retake_check_text",
    {
      title: "Pre-flight a script for heteronyms",
      description: "Scan script text (no audio) for heteronyms and report which pronunciation each one should get in context, with the WellSaid respelling to use. Free; good before rendering a script.",
      inputSchema: {
        text: z.string().min(1).max(200000).describe("The script text"),
        engine: engineSchema,
      },
    },
    guard(async ({ text: t, engine }) => {
      const r = await client.checkText(t, engine);
      const occs = r.occurrences || [];
      const lines = [`${occs.length} heteronym occurrence${occs.length === 1 ? "" : "s"} in the text · intent engine: ${r.engine || "?"}`];
      for (const o of occs) {
        lines.push(
          `  "${o.word}" in "${(o.sentence || "").slice(0, 90)}" → ${o.intended_respell || o.intended} (${o.intended_desc || o.intended})` +
            (o.intent_confidence !== undefined && o.intent_confidence !== null ? ` [${num(o.intent_confidence)}]` : "")
        );
      }
      const data = {
        count: occs.length,
        engine: r.engine,
        occurrences: occs.map((o) => ({
          id: o.id,
          word: o.word,
          sentence: o.sentence,
          text_start: o.text_start,
          text_end: o.text_end,
          intended: o.intended,
          intended_respell: o.intended_respell,
          intended_desc: o.intended_desc,
          intent_confidence: o.intent_confidence,
          respell_candidates: o.respell_candidates,
        })),
      };
      return result(lines.join("\n"), data);
    })
  );

  // ---------------------------------------------------------------- check
  server.registerTool(
    "retake_check_pronunciation",
    {
      title: "Check a voiceover for mispronounced heteronyms",
      description:
        "Run a pronunciation check. Give ONE source: `media_path` (audio/video file on the API machine — the MCP uploads it), `media_url` (downloaded then uploaded), or `text` + `speaker_id` + `model` (rendered through WellSaid first, then checked). " +
        "Optionally pass the intended `script` so intent is decided from the real text. By default waits for the job to finish (up to timeout_s) and returns the flagged occurrences with evidence (intended vs heard pronunciation, phonemes heard, CTC margin, clip URLs) plus report URLs. " +
        "Set wait=false to get the job_id immediately and poll with retake_get_job. Checking is free; the text route spends WellSaid credits for the render.",
      inputSchema: {
        media_path: z.string().optional().describe("Path to an audio or video file"),
        media_url: z.string().url().optional().describe("http(s) URL of an audio or video file"),
        text: z.string().optional().describe("Script to render through WellSaid, then check"),
        speaker_id: z.number().int().min(0).optional().describe("WellSaid voice for the text route"),
        model: modelSchema.optional(),
        script: z.string().optional().describe("The intended script for a media file (improves intent decisions; occurrences are aligned to the transcript)"),
        engine: engineSchema,
        wait: z.boolean().optional().describe("Wait for the result (default true)"),
        timeout_s: z.number().int().min(10).max(7200).optional().describe(`Max seconds to wait (default ${DEFAULT_TIMEOUT_S})`),
        only_flagged: z.boolean().optional().describe("Return only flagged occurrences (default true); false also lists the ones pronounced as intended"),
      },
    },
    guard(async (args, extra) => {
      const sources = ["media_path", "media_url", "text"].filter((k) => args[k]);
      if (sources.length !== 1) {
        return errorResult("Give exactly one of media_path, media_url, or text (with speaker_id and model).");
      }
      let created;
      if (args.media_path) created = await client.createJobFromFile(args.media_path, { script: args.script, engine: args.engine });
      else if (args.media_url) created = await client.createJobFromUrl(args.media_url, { script: args.script, engine: args.engine });
      else {
        if (args.speaker_id === undefined) return errorResult("The text route needs speaker_id (use retake_list_voices) and model.");
        created = await client.createJobFromText(args.text, args.speaker_id, args.model || "caruso", args.engine);
      }
      const jobId = created.job_id;
      if (args.wait === false) {
        return result(`Retake job ${jobId} queued. Poll with retake_get_job {"job_id": "${jobId}"}.`, { job_id: jobId, status: created.status || "queued" });
      }
      const onTick = progressReporter(extra, `job ${jobId}`);
      const job = await client.waitJob(jobId, { timeoutMs: (args.timeout_s || DEFAULT_TIMEOUT_S) * 1000, onTick });
      const { summary, data } = formatJob(job, client, { onlyFlagged: args.only_flagged !== false });
      return result(summary, data);
    })
  );

  server.registerTool(
    "retake_get_job",
    {
      title: "Get a check job",
      description: "Status and results of a pronunciation check job (same compact shape as retake_check_pronunciation).",
      inputSchema: {
        job_id: z.string().min(1),
        only_flagged: z.boolean().optional().describe("Return only flagged occurrences (default true)"),
      },
    },
    guard(async ({ job_id, only_flagged }) => {
      const job = await client.getJob(job_id);
      const { summary, data } = formatJob(job, client, { onlyFlagged: only_flagged !== false });
      return result(summary, data);
    })
  );

  server.registerTool(
    "retake_list_jobs",
    {
      title: "List recent check jobs",
      description: "The newest pronunciation check jobs on the API (id, status, file, flags).",
      inputSchema: { limit: z.number().int().min(1).max(50).optional().describe("Max rows (default 20)") },
    },
    guard(async ({ limit }) => {
      const r = await client.listJobs();
      const jobs = (r.jobs || []).slice(0, limit || 20);
      const lines = jobs.map((j) => `  ${j.id} · ${j.status}${j.stage && j.status !== "done" ? ` (${j.stage})` : ""} · ${j.filename || j.mode} · ${j.occurrences ?? "?"} heteronyms, ${j.flags ?? "?"} flagged · ${j.created}`);
      return text([`${r.count ?? jobs.length} job${(r.count ?? jobs.length) === 1 ? "" : "s"}`, ...lines].join("\n"));
    })
  );

  // ---------------------------------------------------------------- fix
  server.registerTool(
    "retake_fix_pronunciation",
    {
      title: "Fix mispronounced lines through WellSaid",
      description:
        "Re-render the flagged lines of a finished check job through WellSaid with the intended pronunciation and verify every take by ear (phoneme scoring). " +
        "'caruso' uses an inline <respell> tag; 'preview' spells the word the way it should sound (preview ignores respell tags). Up to max_tries renders per line; the loop stops at the first convincingly verified take. " +
        "Returns per-line tries (what was tried, verified yes/no, heard phonemes, CTC margin, take URL) and the best take. SPENDS WellSaid credits. Default: all flagged occurrences; pass occurrence_ids to limit.",
      inputSchema: {
        job_id: z.string().min(1),
        occurrence_ids: z.array(z.string()).optional().describe("Occurrence ids from the check (default: every flagged one)"),
        speaker_id: z.number().int().min(0).describe("WellSaid voice to re-render with (ideally the original voice)"),
        model: modelSchema,
        max_tries: z.number().int().min(1).max(6).optional().describe("Renders per line (default 3)"),
        wait: z.boolean().optional().describe("Wait for the fix to finish (default true)"),
        timeout_s: z.number().int().min(10).max(7200).optional().describe(`Max seconds to wait (default ${DEFAULT_TIMEOUT_S})`),
      },
    },
    guard(async (args, extra) => {
      const created = await client.createFix(args.job_id, {
        occurrenceIds: args.occurrence_ids,
        speakerId: args.speaker_id,
        model: args.model,
        maxTries: args.max_tries,
      });
      const fixId = created.fix_id;
      if (args.wait === false) {
        return result(`Retake fix ${fixId} queued on job ${args.job_id}. Poll with retake_get_fix.`, { job_id: args.job_id, fix_id: fixId, status: "queued" });
      }
      const onTick = progressReporter(extra, `fix ${fixId}`);
      const fix = await client.waitFix(args.job_id, fixId, { timeoutMs: (args.timeout_s || DEFAULT_TIMEOUT_S) * 1000, onTick });
      const { summary, data } = formatFix(fix, client);
      return result(summary, data);
    })
  );

  server.registerTool(
    "retake_get_fix",
    {
      title: "Get a fix",
      description: "Status and results of a fix (same compact shape as retake_fix_pronunciation).",
      inputSchema: { job_id: z.string().min(1), fix_id: z.string().min(1) },
    },
    guard(async ({ job_id, fix_id }) => {
      const fix = await client.getFix(job_id, fix_id);
      const { summary, data } = formatFix(fix, client);
      return result(summary, data);
    })
  );

  return server;
}

export const TOOL_NAMES = [
  "retake_health",
  "retake_list_voices",
  "retake_check_text",
  "retake_check_pronunciation",
  "retake_get_job",
  "retake_list_jobs",
  "retake_fix_pronunciation",
  "retake_get_fix",
];
