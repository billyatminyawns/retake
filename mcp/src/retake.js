// Minimal client for the Retake API (v1) used by the MCP tools.
//
// Config (env):
//   RETAKE_API_BASE   e.g. http://localhost:8080  (default)
//   RETAKE_API_KEY    sent as X-Api-Key; falls back to the first entry of
//                     RETAKE_API_KEYS so a local server and this MCP can share
//                     one .env. Omitted when the server runs in open mode.
//
// Never logs the key. Uses the global fetch/FormData/Blob of Node >= 18.

import { readFile, stat } from "node:fs/promises";
import path from "node:path";

export class RetakeError extends Error {
  /** @param {string} message @param {number} [status] @param {string} [code] */
  constructor(message, status, code) {
    super(message);
    this.name = "RetakeError";
    this.status = status;
    this.code = code;
  }
}

const DEFAULT_BASE = "http://localhost:8080";
const MAX_MEDIA_BYTES = 500 * 1024 * 1024;

function resolveKey(explicit) {
  if (explicit !== undefined) return explicit || "";
  const k = (process.env.RETAKE_API_KEY || "").trim();
  if (k) return k;
  const list = (process.env.RETAKE_API_KEYS || "").split(",").map((s) => s.trim()).filter(Boolean);
  return list[0] || "";
}

export class RetakeClient {
  constructor({ baseUrl, apiKey, fetchImpl } = {}) {
    this.baseUrl = (baseUrl || process.env.RETAKE_API_BASE || DEFAULT_BASE).replace(/\/+$/, "");
    this.apiKey = resolveKey(apiKey);
    this.fetch = fetchImpl || globalThis.fetch;
    if (typeof this.fetch !== "function") {
      throw new RetakeError("global fetch is unavailable — use Node >= 18.17");
    }
  }

  hasApiKey() {
    return Boolean(this.apiKey);
  }

  /** Absolute URL for a path the API returned (e.g. /v1/jobs/x/files/y?t=…). */
  absolute(p) {
    if (!p) return p;
    if (/^https?:\/\//i.test(p)) return p;
    return this.baseUrl + (p.startsWith("/") ? p : "/" + p);
  }

  headers(extra = {}) {
    const h = { Accept: "application/json", "Bypass-Tunnel-Reminder": "true", ...extra };
    if (this.apiKey) h["X-Api-Key"] = this.apiKey;
    return h;
  }

  /**
   * One request. `json` sends a JSON body, `form` a FormData body.
   * Returns the parsed JSON (or text). Throws RetakeError with the API's
   * {error:{code,message}} when the status is not 2xx.
   */
  async request(method, p, { json, form, query, timeoutMs = 120000 } = {}) {
    let url = this.absolute(p);
    if (query) {
      const q = new URLSearchParams();
      for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== null) q.set(k, String(v));
      const qs = q.toString();
      if (qs) url += (url.includes("?") ? "&" : "?") + qs;
    }
    const init = { method, headers: this.headers() };
    if (json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(json);
    } else if (form !== undefined) {
      init.body = form; // fetch sets the multipart boundary
    }
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    init.signal = ctrl.signal;
    let res;
    try {
      res = await this.fetch(url, init);
    } catch (err) {
      const why = err?.name === "AbortError" ? `timed out after ${Math.round(timeoutMs / 1000)}s` : err?.message || String(err);
      throw new RetakeError(`Cannot reach the Retake API at ${this.baseUrl} (${why}). Is the API running? Set RETAKE_API_BASE.`);
    } finally {
      clearTimeout(timer);
    }
    const text = await res.text();
    let body = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = text;
    }
    if (!res.ok) {
      const e = body && body.error ? body.error : null;
      const msg = e ? `${e.message || e.code || "request failed"}` : `HTTP ${res.status} ${res.statusText}`;
      if (res.status === 401) {
        throw new RetakeError(`Retake API rejected the request (401): ${msg}. Set RETAKE_API_KEY to one of the server's RETAKE_API_KEYS.`, 401, e?.code);
      }
      throw new RetakeError(`Retake API ${method} ${p} → ${res.status}: ${msg}`, res.status, e?.code);
    }
    return body;
  }

  // ---------------------------------------------------------------- reads
  health() {
    return this.request("GET", "/v1/health", { timeoutMs: 15000 });
  }
  voices() {
    return this.request("GET", "/v1/voices", { timeoutMs: 60000 });
  }
  lexicon() {
    return this.request("GET", "/v1/lexicon", { timeoutMs: 30000 });
  }
  checkText(text, engine) {
    const body = { text };
    if (engine) body.engine = engine;
    return this.request("POST", "/v1/check-text", { json: body, timeoutMs: 180000 });
  }
  getJob(jobId) {
    return this.request("GET", `/v1/jobs/${encodeURIComponent(jobId)}`, { timeoutMs: 30000 });
  }
  listJobs() {
    return this.request("GET", "/v1/jobs", { timeoutMs: 30000 });
  }
  getFix(jobId, fixId) {
    return this.request("GET", `/v1/jobs/${encodeURIComponent(jobId)}/fixes/${encodeURIComponent(fixId)}`, { timeoutMs: 30000 });
  }

  // ---------------------------------------------------------------- writes
  /** Upload a local media file (multipart). */
  async createJobFromFile(mediaPath, { script, engine } = {}) {
    const abs = path.resolve(mediaPath);
    let st;
    try {
      st = await stat(abs);
    } catch {
      throw new RetakeError(`media_path not found: ${abs}`);
    }
    if (!st.isFile()) throw new RetakeError(`media_path is not a file: ${abs}`);
    if (st.size > MAX_MEDIA_BYTES) throw new RetakeError(`media_path is ${Math.round(st.size / 1048576)} MB; the API accepts up to 500 MB`);
    const bytes = await readFile(abs);
    return this.#postMedia(new Blob([bytes]), path.basename(abs), { script, engine });
  }

  /** Download a media URL, then upload it. */
  async createJobFromUrl(mediaUrl, { script, engine } = {}) {
    let res;
    try {
      res = await this.fetch(mediaUrl);
    } catch (err) {
      throw new RetakeError(`could not download media_url (${err?.message || err})`);
    }
    if (!res.ok) throw new RetakeError(`could not download media_url: HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length > MAX_MEDIA_BYTES) throw new RetakeError(`media_url is ${Math.round(buf.length / 1048576)} MB; the API accepts up to 500 MB`);
    let name = "media";
    try {
      name = path.basename(new URL(mediaUrl).pathname) || "media";
    } catch {
      /* keep default */
    }
    return this.#postMedia(new Blob([buf]), name, { script, engine });
  }

  async #postMedia(blob, filename, { script, engine }) {
    const form = new FormData();
    form.append("media", blob, filename);
    if (script) form.append("script", script);
    if (engine) form.append("engine", engine);
    return this.request("POST", "/v1/jobs", { form, timeoutMs: 600000 });
  }

  /** Render text through WellSaid first, then check the render. */
  createJobFromText(text, speakerId, model, engine) {
    const body = { text, speaker_id: Number(speakerId), model: model || "caruso" };
    if (engine) body.engine = engine;
    return this.request("POST", "/v1/jobs", { json: body, timeoutMs: 60000 });
  }

  createFix(jobId, { occurrenceIds, speakerId, model, maxTries }) {
    const body = { speaker_id: Number(speakerId), model: model || "caruso" };
    if (occurrenceIds && occurrenceIds.length) body.occurrence_ids = occurrenceIds;
    if (maxTries) body.max_tries = Number(maxTries);
    return this.request("POST", `/v1/jobs/${encodeURIComponent(jobId)}/fix`, { json: body, timeoutMs: 60000 });
  }

  // ---------------------------------------------------------------- polling
  /** Poll until status is done|error. onTick(job) is called on every poll. */
  async waitJob(jobId, { timeoutMs = 20 * 60 * 1000, pollMs = 2000, onTick } = {}) {
    return this.#wait(() => this.getJob(jobId), { timeoutMs, pollMs, onTick, what: `job ${jobId}` });
  }

  async waitFix(jobId, fixId, { timeoutMs = 20 * 60 * 1000, pollMs = 2000, onTick } = {}) {
    return this.#wait(() => this.getFix(jobId, fixId), { timeoutMs, pollMs, onTick, what: `fix ${fixId}` });
  }

  async #wait(getter, { timeoutMs, pollMs, onTick, what }) {
    const t0 = Date.now();
    let last = null;
    for (;;) {
      last = await getter();
      if (onTick) onTick(last);
      if (last.status === "done" || last.status === "error") return last;
      if (Date.now() - t0 > timeoutMs) {
        const e = new RetakeError(`${what} is still ${last.status} (${last.stage || "?"}, ${Math.round((last.progress || 0) * 100)}%) after ${Math.round(timeoutMs / 1000)}s — poll it later with the get tool.`);
        e.partial = last;
        throw e;
      }
      await new Promise((r) => setTimeout(r, pollMs));
    }
  }
}
