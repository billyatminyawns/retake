"""Job store (SQLite) + the single background worker.

Layout (SPEC): `RETAKE_DATA/jobs.db` and `RETAKE_DATA/jobs/<id>/` holding the
uploaded (or rendered) media, the engine work dir (clips, 16 kHz wav, whisper
cache), fixes/<fix_id>/ takes, report.json and report.html.

One daemon worker thread consumes a FIFO queue: the engine's models are not
re-entrant, so checks and fixes run strictly one at a time. Progress
(stage + fraction) is written to the DB as the engine reports it; clients
poll GET /v1/jobs/{id}.

Results are stored RAW (absolute paths) and rewritten to URLs under
/v1/jobs/{id}/files/... on the way out (`rewrite_paths`). In key mode the URLs
carry the job's capability token (`?t=`) so <audio src> works in a browser.
"""
import json
import logging
import os
import queue
import secrets
import shutil
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import engine, settings
from .report import write_reports

log = logging.getLogger("retake.jobs")

STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR = "queued", "running", "done", "error"
FILE_KEYS = ("clip_wav", "sentence_wav", "take_mp3", "media", "clip", "wav", "audio", "mp3",
             "take", "before", "after", "path", "file", "report_json", "report_html")


def utcnow():
    # type: () -> str
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_id(nbytes=8):
    # type: (int) -> str
    return secrets.token_hex(nbytes)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, token TEXT NOT NULL, status TEXT NOT NULL, stage TEXT, progress REAL DEFAULT 0,
  created TEXT NOT NULL, updated TEXT NOT NULL, mode TEXT NOT NULL, filename TEXT,
  params TEXT, result TEXT, error TEXT, duration REAL, occurrences INTEGER, flags INTEGER
);
CREATE TABLE IF NOT EXISTS fixes (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL, status TEXT NOT NULL, stage TEXT, progress REAL DEFAULT 0,
  created TEXT NOT NULL, updated TEXT NOT NULL, params TEXT, results TEXT, summary TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS fixes_job ON fixes(job_id);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created);
"""


def _j(obj):
    return json.dumps(obj, ensure_ascii=False) if obj is not None else None


def _p(text, default=None):
    if text is None:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


class JobStore(object):
    """Thread-safe SQLite store (one connection, serialized by a lock)."""

    def __init__(self, path=None):
        self.path = str(path or settings.db_path())
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        os.makedirs(str(settings.jobs_dir()), exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(_SCHEMA)

    # -- helpers -----------------------------------------------------------

    def job_dir(self, job_id):
        # type: (str) -> str
        return os.path.join(str(settings.jobs_dir()), job_id)

    def _exec(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, args)

    def close(self):
        with self._lock:
            try:
                self._db.close()
            except Exception:
                pass

    # -- jobs --------------------------------------------------------------

    def create_job(self, mode, filename, params):
        # type: (str, Optional[str], Dict[str, Any]) -> Dict[str, Any]
        job_id = new_id(8)
        now = utcnow()
        os.makedirs(self.job_dir(job_id), exist_ok=True)
        self._exec("INSERT INTO jobs (id, token, status, stage, progress, created, updated, mode, filename, params)"
                   " VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (job_id, secrets.token_urlsafe(18), STATUS_QUEUED, "queued", 0.0, now, now, mode,
                    filename, _j(params)))
        return self.get_job(job_id)

    def get_job(self, job_id):
        # type: (str) -> Optional[Dict[str, Any]]
        row = self._exec("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_row(row) if row else None

    def _job_row(self, row):
        d = dict(row)
        d["params"] = _p(d.get("params"), {}) or {}
        d["result"] = _p(d.get("result"))
        d["error"] = _p(d.get("error"))
        d["progress"] = float(d.get("progress") or 0.0)
        return d

    def list_jobs(self, limit=50):
        # type: (int) -> List[Dict[str, Any]]
        rows = self._exec("SELECT id, status, stage, progress, created, updated, mode, filename, duration,"
                          " occurrences, flags, error FROM jobs ORDER BY created DESC, rowid DESC LIMIT ?",
                          (int(limit),)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["error"] = _p(d.get("error"))
            d["progress"] = float(d.get("progress") or 0.0)
            out.append(d)
        return out

    def update_job(self, job_id, **fields):
        if not fields:
            return
        fields["updated"] = utcnow()
        for k in ("result", "error", "params"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = _j(fields[k])
        cols = ", ".join("%s=?" % k for k in fields)
        self._exec("UPDATE jobs SET %s WHERE id=?" % cols, tuple(fields.values()) + (job_id,))

    def delete_job(self, job_id):
        # type: (str) -> bool
        row = self._exec("SELECT id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return False
        self._exec("DELETE FROM fixes WHERE job_id=?", (job_id,))
        self._exec("DELETE FROM jobs WHERE id=?", (job_id,))
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True

    def jobs_with_status(self, status):
        rows = self._exec("SELECT id FROM jobs WHERE status=? ORDER BY created", (status,)).fetchall()
        return [r["id"] for r in rows]

    # -- fixes -------------------------------------------------------------

    def create_fix(self, job_id, params):
        fix_id = new_id(4)
        now = utcnow()
        self._exec("INSERT INTO fixes (id, job_id, status, stage, progress, created, updated, params)"
                   " VALUES (?,?,?,?,?,?,?,?)",
                   (fix_id, job_id, STATUS_QUEUED, "queued", 0.0, now, now, _j(params)))
        return self.get_fix(job_id, fix_id)

    def get_fix(self, job_id, fix_id):
        row = self._exec("SELECT * FROM fixes WHERE id=? AND job_id=?", (fix_id, job_id)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = _p(d.get("params"), {}) or {}
        d["results"] = _p(d.get("results"), []) or []
        d["summary"] = _p(d.get("summary"), {}) or {}
        d["error"] = _p(d.get("error"))
        d["progress"] = float(d.get("progress") or 0.0)
        return d

    def list_fix_ids(self, job_id):
        rows = self._exec("SELECT id FROM fixes WHERE job_id=? ORDER BY created", (job_id,)).fetchall()
        return [r["id"] for r in rows]

    def update_fix(self, fix_id, **fields):
        if not fields:
            return
        fields["updated"] = utcnow()
        for k in ("results", "summary", "error", "params"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = _j(fields[k])
        cols = ", ".join("%s=?" % k for k in fields)
        self._exec("UPDATE fixes SET %s WHERE id=?" % cols, tuple(fields.values()) + (fix_id,))

    def fixes_with_status(self, status):
        rows = self._exec("SELECT id, job_id FROM fixes WHERE status=? ORDER BY created", (status,)).fetchall()
        return [(r["job_id"], r["id"]) for r in rows]


# ---------------------------------------------------------------------------
# path -> URL rewriting
# ---------------------------------------------------------------------------

def file_url(job_id, relpath, token=None):
    # type: (str, str, Optional[str]) -> str
    url = "%s/jobs/%s/files/%s" % (settings.API_PREFIX, job_id, relpath.replace(os.sep, "/"))
    if token and settings.auth_mode() == "key":
        url += "?t=" + token
    return url


def _is_path_key(key):
    k = key.lower()
    return k in FILE_KEYS or k.endswith("_wav") or k.endswith("_mp3") or k.endswith("_path") or k.endswith("_file")


def rewrite_paths(obj, job_id, job_dir, token=None):
    """Deep-copy `obj` replacing absolute file paths with /v1/jobs/{id}/files URLs.

    A path under the job dir maps directly. A path elsewhere (an engine cache,
    say) is copied into <job>/ext/ first so the URL keeps working after the
    cache is cleaned. Only string values whose key looks like a file key, or
    that are absolute paths to existing files, are touched."""
    job_dir = os.path.abspath(job_dir)
    ext_dir = os.path.join(job_dir, "ext")

    def conv(val, key=None):
        if isinstance(val, dict):
            return {k: conv(v, k) for k, v in val.items()}
        if isinstance(val, list):
            return [conv(v, key) for v in val]
        if isinstance(val, str) and val.startswith("/") and (key is None or _is_path_key(key)
                                                            or os.path.isfile(val)):
            if val.startswith(settings.API_PREFIX + "/"):
                return val
            ap = os.path.abspath(val)
            if ap.startswith(job_dir + os.sep):
                return file_url(job_id, os.path.relpath(ap, job_dir), token)
            if os.path.isfile(ap):
                os.makedirs(ext_dir, exist_ok=True)
                dst = os.path.join(ext_dir, os.path.basename(ap))
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(ap, dst)
                    except OSError:
                        return val
                return file_url(job_id, os.path.relpath(dst, job_dir), token)
        return val

    return conv(obj)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

class Worker(object):
    """Single consumer thread over a FIFO queue of ("check", job_id) and
    ("fix", job_id, fix_id) tasks."""

    def __init__(self, store):
        # type: (JobStore) -> None
        self.store = store
        self.q = queue.Queue()  # type: queue.Queue
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="retake-worker", daemon=True)
        self.current = None  # type: Optional[tuple]

    def start(self):
        self._recover()
        self._thread.start()
        log.info("worker started", extra={"event": "worker_start", "queued": self.q.qsize()})

    def stop(self):
        self._stop.set()
        self.q.put(None)

    def pending(self):
        # type: () -> int
        return self.q.qsize() + (1 if self.current else 0)

    def enqueue_check(self, job_id):
        self.q.put(("check", job_id))

    def enqueue_fix(self, job_id, fix_id):
        self.q.put(("fix", job_id, fix_id))

    def _recover(self):
        """After a restart: running tasks are lost (mark error), queued ones re-queue."""
        err = {"code": "interrupted", "message": "The API restarted while this task was running. Submit it again."}
        for jid in self.store.jobs_with_status(STATUS_RUNNING):
            self.store.update_job(jid, status=STATUS_ERROR, stage="error", error=err)
        for jid, fid in self.store.fixes_with_status(STATUS_RUNNING):
            self.store.update_fix(fid, status=STATUS_ERROR, stage="error", error=err)
        for jid in self.store.jobs_with_status(STATUS_QUEUED):
            self.enqueue_check(jid)
        for jid, fid in self.store.fixes_with_status(STATUS_QUEUED):
            self.enqueue_fix(jid, fid)

    def _run(self):
        while not self._stop.is_set():
            task = self.q.get()
            if task is None:
                break
            self.current = task
            t0 = time.time()
            try:
                if task[0] == "check":
                    self._do_check(task[1])
                elif task[0] == "fix":
                    self._do_fix(task[1], task[2])
            except Exception:  # never let the worker die
                log.error("worker task crashed", extra={"event": "worker_crash", "task": task[0],
                                                        "trace": traceback.format_exc()[-2000:]})
            finally:
                log.info("task finished", extra={"event": "task_done", "task": task[0], "job_id": task[1],
                                                 "seconds": round(time.time() - t0, 2)})
                self.current = None
                self.q.task_done()

    # -- check --------------------------------------------------------------

    def _progress_cb(self, update, floor=0.0, ceil=1.0):
        state = {"last": 0.0, "stage": None}

        def cb(stage, frac):
            try:
                frac = float(frac)
            except (TypeError, ValueError):
                frac = 0.0
            frac = max(0.0, min(1.0, frac))
            scaled = floor + (ceil - floor) * frac
            now = time.time()
            if stage != state["stage"] or now - state["last"] > 0.4 or frac >= 1.0:
                state["last"], state["stage"] = now, stage
                update(stage=str(stage), progress=round(scaled, 4))
        return cb

    def _do_check(self, job_id):
        store = self.store
        job = store.get_job(job_id)
        if not job or job["status"] != STATUS_QUEUED:
            return
        jdir = store.job_dir(job_id)
        params = job["params"]
        store.update_job(job_id, status=STATUS_RUNNING, stage="preparing", progress=0.0)

        def update(**f):
            store.update_job(job_id, **f)

        try:
            media_path = os.path.join(jdir, params.get("media_name") or "media")
            script = params.get("script")
            if job["mode"] == "script":
                media_path = os.path.join(jdir, "media.mp3")
                update(stage="rendering", progress=0.0)
                engine.render_script(params["text"], params["speaker_id"], params.get("model") or "caruso",
                                     media_path, progress=self._progress_cb(update, 0.0, 0.15))
                script = params["text"]
                store.update_job(job_id, filename="media.mp3")
                floor = 0.15
            else:
                floor = 0.0
            if not os.path.isfile(media_path):
                raise FileNotFoundError("media file missing: %s" % os.path.basename(media_path))
            work_dir = os.path.join(jdir, "work")
            os.makedirs(work_dir, exist_ok=True)
            update(stage="transcribing", progress=floor)
            result = engine.check_media(media_path, script, params.get("engine") or settings.default_engine(),
                                        work_dir, self._progress_cb(update, floor, 0.98))
            if not isinstance(result, dict):
                raise RuntimeError("engine returned %s instead of a result dict" % type(result).__name__)
            result.setdefault("media", media_path)
            occs = result.get("occurrences") or []
            if not isinstance(result.get("flags"), int):
                result["flags"] = sum(1 for o in occs if isinstance(o, dict) and o.get("flag"))
            for i, o in enumerate(occs, 1):
                if isinstance(o, dict) and not o.get("id"):
                    o["id"] = "o%d" % i
            update(stage="writing report", progress=0.99)
            job = store.get_job(job_id)
            try:
                write_reports(job, result, jdir)
            except Exception as e:
                log.warning("report writing failed: %s", e, extra={"event": "report_fail", "job_id": job_id})
            store.update_job(job_id, status=STATUS_DONE, stage="done", progress=1.0, result=result,
                             duration=result.get("duration"), occurrences=len(occs),
                             flags=int(result.get("flags") or 0), error=None)
        except engine.CoreUnavailable as e:
            self._fail_job(job_id, "core_unavailable", str(e))
        except engine.UpstreamError as e:
            self._fail_job(job_id, "upstream_error", str(e))
        except Exception as e:
            log.error("check failed", extra={"event": "check_fail", "job_id": job_id,
                                             "trace": traceback.format_exc()[-2000:]})
            self._fail_job(job_id, "processing_error", "%s: %s" % (type(e).__name__, engine._safe_err(e)))

    def _fail_job(self, job_id, code, message):
        self.store.update_job(job_id, status=STATUS_ERROR, stage="error",
                              error={"code": code, "message": message})

    # -- fix ----------------------------------------------------------------

    def _do_fix(self, job_id, fix_id):
        store = self.store
        fix = store.get_fix(job_id, fix_id)
        job = store.get_job(job_id)
        if not fix or not job or fix["status"] != STATUS_QUEUED:
            return
        params = fix["params"]
        ids = list(params.get("occurrence_ids") or [])
        occs = {o["id"]: o for o in (job.get("result") or {}).get("occurrences", []) if isinstance(o, dict)}
        store.update_fix(fix_id, status=STATUS_RUNNING, stage="starting", progress=0.0)
        results = []
        summary = {"requested": len(ids), "done": 0, "verified": 0, "unverified": 0, "errors": 0}
        fdir = os.path.join(store.job_dir(job_id), "fixes", fix_id)
        os.makedirs(fdir, exist_ok=True)
        fatal = None
        for n, oid in enumerate(ids):
            occ = occs.get(oid)
            lo, hi = float(n) / len(ids), float(n + 1) / len(ids)

            def update(**f):
                store.update_fix(fix_id, **f)
            cb = self._progress_cb(update, lo, hi)
            cb("fixing %s (%d/%d)" % (occ.get("word") if occ else oid, n + 1, len(ids)), 0.0)
            try:
                res = engine.fix_occurrence(occ, params["speaker_id"], params.get("model") or "caruso",
                                            os.path.join(fdir, oid), params.get("max_tries", 3), cb)
                if not isinstance(res, dict):
                    raise RuntimeError("engine returned %s" % type(res).__name__)
                res.setdefault("occurrence_id", oid)
                res["word"] = occ.get("word")
                res["before"] = {"clip_wav": occ.get("clip_wav"), "sentence_wav": occ.get("sentence_wav"),
                                 "detected": occ.get("detected"), "detected_desc": occ.get("detected_desc"),
                                 "free_decode": occ.get("free_decode")}
                res["intended"] = occ.get("intended")
                res["intended_desc"] = occ.get("intended_desc")
                results.append(res)
                summary["done"] += 1
                if res.get("verified"):
                    summary["verified"] += 1
                else:
                    summary["unverified"] += 1
            except engine.CoreUnavailable as e:
                fatal = ("core_unavailable", str(e))
                break
            except Exception as e:
                code = "upstream_error" if isinstance(e, engine.UpstreamError) else "processing_error"
                log.error("fix failed", extra={"event": "fix_fail", "job_id": job_id, "fix_id": fix_id,
                                               "occurrence": oid, "trace": traceback.format_exc()[-1500:]})
                results.append({"occurrence_id": oid, "word": occ.get("word") if occ else None,
                                "tries": [], "best": None, "verified": False,
                                "error": {"code": code, "message": engine._safe_err(e)}})
                summary["errors"] += 1
            store.update_fix(fix_id, results=results, summary=summary)
        if fatal:
            store.update_fix(fix_id, status=STATUS_ERROR, stage="error", results=results, summary=summary,
                             error={"code": fatal[0], "message": fatal[1]})
        else:
            store.update_fix(fix_id, status=STATUS_DONE, stage="done", progress=1.0, results=results,
                             summary=summary, error=None)
