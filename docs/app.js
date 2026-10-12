/* =====================================================================
   RETAKE · pronunciation QA — UI logic. Plain ES5-ish JS, zero deps, no build.
   Talks to the Retake API (/v1/*) at window.RETAKE_API_BASE (config.js) or the
   base saved in Settings. ?mock=1 loads ui/mock.json and never touches the network.
   ===================================================================== */
'use strict';

(function () {

  /* ------------------------------ constants ------------------------------ */

  var MOCK = new URLSearchParams(window.location.search).get('mock') === '1';
  var POLL_MS = 1500;
  var LS = { base: 'retake.apiBase', key: 'retake.apiKey', voice: 'retake.voice', model: 'retake.model' };

  // Pipeline stages in order. `match` = substrings of the API's stage string.
  var STAGES = [
    { key: 'uploading',    label: 'Uploading',           match: ['upload', 'render', 'queued', 'extract', 'convert', 'probe'] },
    { key: 'transcribing', label: 'Transcribing',        match: ['transcri', 'whisper'] },
    { key: 'finding',      label: 'Finding heteronyms',  match: ['find', 'heteronym', 'detect', 'occurren'] },
    { key: 'intent',       label: 'Deciding intent',     match: ['intent', 'decid', 'claude', 'spacy'] },
    { key: 'listening',    label: 'Listening',           match: ['listen', 'scor', 'verif', 'ctc', 'phoneme', 'clip'] },
    { key: 'done',         label: 'Done',                match: ['done', 'complete', 'finish'] }
  ];

  var SENSE_POS = { _n: 'noun', _v: 'verb', _adj: 'adjective', _adv: 'adverb', _present: 'verb · present', _past: 'verb · past' };

  var state = {
    route: { name: 'home' },
    voices: [],
    health: null,
    jobs: [],
    job: null,          // current GET /v1/jobs/{id} payload
    fix: null,          // current fix payload for the job
    fixId: null,
    pollTimer: null,
    fixTimer: null,
    filter: 'all',
    file: null,
    mock: null,         // mock.json fixtures
    players: [],
    blobCache: {},
    keyRevealed: false
  };

  /* ------------------------------ helpers ------------------------------ */

  function $(sel, el) { return (el || document).querySelector(sel); }
  function $all(sel, el) { return Array.prototype.slice.call((el || document).querySelectorAll(sel)); }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function h(tag, attrs, children) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v == null || v === false) return;
        if (k === 'class') el.className = v;
        else if (k === 'html') el.innerHTML = v;
        else if (k === 'text') el.textContent = v;
        else if (k.indexOf('on') === 0 && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else if (k === 'dataset') Object.keys(v).forEach(function (d) { el.dataset[d] = v[d]; });
        else el.setAttribute(k, v === true ? '' : v);
      });
    }
    (children || []).forEach(function (c) {
      if (c == null || c === false) return;
      el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return el;
  }
  function fmtTime(sec) {
    if (sec == null || isNaN(sec)) return '–';
    sec = Math.max(0, sec);
    var m = Math.floor(sec / 60), s = sec - m * 60;
    return m > 0 ? m + ':' + (s < 10 ? '0' : '') + s.toFixed(1) : s.toFixed(1) + 's';
  }
  function fmtClock(sec) {
    if (sec == null || isNaN(sec)) return '0:00';
    if (sec < 10) return sec.toFixed(1) + 's';
    var m = Math.floor(sec / 60), s = Math.floor(sec - m * 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }
  function fmtWhen(v) {
    if (v == null) return '';
    var d = typeof v === 'number' ? new Date(v < 1e12 ? v * 1000 : v) : new Date(v);
    if (isNaN(d.getTime())) return String(v);
    var diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return 'just now';
    if (diff < 3600) return Math.floor(diff / 60) + ' min ago';
    if (diff < 86400) return Math.floor(diff / 3600) + ' h ago';
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  }
  function basename(p) { if (!p) return ''; var s = String(p).split('?')[0]; return decodeURIComponent(s.substring(s.lastIndexOf('/') + 1)); }
  function posOf(senseId) {
    var id = String(senseId || '');
    var keys = Object.keys(SENSE_POS);
    for (var i = 0; i < keys.length; i++) if (id.endsWith(keys[i])) return SENSE_POS[keys[i]];
    return '';
  }
  function senseRespell(desc) { var m = /\(([^()]*)\)\s*$/.exec(desc || ''); return m ? m[1] : ''; }
  function senseText(desc) { return String(desc || '').replace(/\s*\([^()]*\)\s*$/, ''); }
  function lsGet(k) { try { return localStorage.getItem(k) || ''; } catch (e) { return ''; } }
  function lsSet(k, v) { try { if (v) localStorage.setItem(k, v); else localStorage.removeItem(k); } catch (e) { /* private mode */ } }
  function wordCount(s) { return (String(s || '').trim().match(/\S+/g) || []).length; }
  function copyText(text, label) {
    var done = function () { toast((label || 'Copied') + ' to clipboard', 'ok'); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
    } else { fallbackCopy(text); done(); }
  }
  function fallbackCopy(text) {
    var ta = h('textarea', { style: 'position:fixed;opacity:0;left:-9999px' }); ta.value = text;
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }
  function downloadBlob(blob, name) {
    var url = URL.createObjectURL(blob);
    var a = h('a', { href: url, download: name }); document.body.appendChild(a); a.click();
    setTimeout(function () { document.body.removeChild(a); URL.revokeObjectURL(url); }, 1500);
  }

  /* ------------------------------ toasts ------------------------------ */

  function toast(msg, kind, code) {
    var box = $('#toasts');
    var el = h('div', { class: 'toast t-' + (kind || 'info'), role: kind === 'error' ? 'alert' : 'status' });
    el.appendChild(document.createTextNode(msg));
    if (code) el.appendChild(h('div', { html: '<code>' + esc(code) + '</code>' }));
    var kill = function () { el.classList.add('out'); setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 320); };
    el.addEventListener('click', kill);
    box.appendChild(el);
    setTimeout(kill, kind === 'error' ? 9000 : 4200);
  }

  /* ------------------------------ API client ------------------------------ */

  function apiBase() {
    var b = lsGet(LS.base) || window.RETAKE_API_BASE || '';
    return String(b).replace(/\/+$/, '');
  }
  function apiKey() { return lsGet(LS.key); }
  function authHeaders(extra) {
    var hd = extra || {};
    var k = apiKey();
    if (k) hd['X-Api-Key'] = k;
    hd['Bypass-Tunnel-Reminder'] = 'true';  // localtunnel demo hosting; harmless elsewhere
    return hd;
  }
  function ApiError(code, message, status) { this.code = code; this.message = message; this.status = status; }
  ApiError.prototype = Object.create(Error.prototype);

  function api(method, path, opts) {
    opts = opts || {};
    if (MOCK) return mockApi(method, path, opts);
    var url = apiBase() + path;
    var init = { method: method, headers: authHeaders({}) };
    if (opts.json !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(opts.json); }
    else if (opts.form) { init.body = opts.form; }
    return fetch(url, init).then(function (res) {
      var ct = res.headers.get('content-type') || '';
      var parse = ct.indexOf('json') >= 0 ? res.json() : res.text().then(function (t) { return t ? { text: t } : {}; });
      return parse.catch(function () { return {}; }).then(function (body) {
        if (!res.ok) {
          var err = (body && body.error) || {};
          var msg = err.message || (body && body.detail && (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))) || (res.status + ' ' + res.statusText);
          throw new ApiError(err.code || ('http_' + res.status), msg, res.status);
        }
        return body;
      });
    }, function (e) {
      throw new ApiError('network', "Can't reach the API at " + (apiBase() || window.location.origin) + ' (' + (e && e.message ? e.message : 'network error') + ')', 0);
    });
  }

  function reportError(e, context) {
    var msg = (e && e.message) || String(e);
    if (context) msg = context + ': ' + msg;
    toast(msg, 'error', e && e.code ? e.code : null);
    if (window.console && e && !(e instanceof ApiError)) console.warn(e);
  }

  // Resolve a result path/URL (already rewritten by the API to /v1/jobs/{id}/files/…) to a fetchable URL.
  function fileUrl(u) {
    if (!u) return '';
    if (/^(https?:|blob:|data:)/.test(u)) return u;
    if (u.charAt(0) === '/') return apiBase() + u;
    return apiBase() + '/' + u;
  }
  // Media src for <audio>: with an API key the files endpoint needs the header, so fetch to a blob.
  function mediaSrc(u) {
    if (MOCK) return Promise.resolve(mockTone(u));
    var url = fileUrl(u);
    if (!apiKey()) return Promise.resolve(url);
    if (state.blobCache[url]) return Promise.resolve(state.blobCache[url]);
    return fetch(url, { headers: authHeaders({}) }).then(function (res) {
      if (!res.ok) throw new ApiError('http_' + res.status, 'Could not load ' + basename(url) + ' (' + res.status + ')', res.status);
      return res.blob();
    }).then(function (b) { var o = URL.createObjectURL(b); state.blobCache[url] = o; return o; });
  }
  function fetchFile(u) {
    if (MOCK) return Promise.resolve(mockFile(u));
    var url = fileUrl(u);
    return fetch(url, { headers: authHeaders({}) }).then(function (res) {
      if (!res.ok) throw new ApiError('http_' + res.status, 'Could not download ' + basename(url) + ' (' + res.status + ')', res.status);
      return res.blob();
    });
  }

  /* ------------------------------ mock backend ------------------------------ */

  function loadMock() {
    if (state.mock) return Promise.resolve(state.mock);
    return fetch('mock.json', { cache: 'no-store' }).then(function (r) { return r.json(); }).then(function (m) { state.mock = m; return m; });
  }
  var mockFixPolls = 0, mockJobStart = 0, mockJobMode = 'upload';
  function mockApi(method, path, opts) {
    return loadMock().then(function (m) {
      var clone = function (o) { return JSON.parse(JSON.stringify(o)); };
      var delay = function (v, ms) { return new Promise(function (r) { setTimeout(function () { r(v); }, ms || 120); }); };
      var mm;
      if (method === 'GET' && path === '/v1/health') return delay(clone(m.health));
      if (method === 'GET' && path === '/v1/voices') return delay(clone(m.voices));
      if (method === 'GET' && path === '/v1/jobs') return delay(clone(m.jobs));
      if (method === 'POST' && path === '/v1/jobs') { mockJobStart = Date.now(); mockJobMode = opts.mode || 'upload'; return delay({ job_id: 'j_live', status_url: '/v1/jobs/j_live' }, 400); }
      if ((mm = /^\/v1\/jobs\/([^/]+)$/.exec(path)) && method === 'GET') {
        if (mm[1] === 'j_live') {                 // simulated pipeline: ~9 s through the stages
          var el = (Date.now() - mockJobStart) / 1000, frac = Math.min(1, el / 9);
          var names = ['uploading', 'transcribing', 'finding heteronyms', 'deciding intent', 'listening', 'done'];
          var idx = Math.min(5, Math.floor(frac * 5.999));
          if (frac >= 1) { var j = clone(m.job); j.id = 'j_live'; j.mode = mockJobMode; j.created = new Date(mockJobStart).toISOString(); return delay(j); }
          return delay({ id: 'j_live', status: 'running', stage: names[idx], progress: frac, created: new Date(mockJobStart).toISOString(), mode: mockJobMode, result: null, error: null });
        }
        var hit = null;
        if (mm[1] === m.job.id) hit = clone(m.job);
        else (m.jobs || []).forEach(function (jj) { if (jj.id === mm[1]) hit = clone(jj); });
        if (!hit) return Promise.reject(new ApiError('not_found', 'No job ' + mm[1] + ' in the sample data', 404));
        return delay(hit);
      }
      if ((mm = /^\/v1\/jobs\/([^/]+)$/.exec(path)) && method === 'DELETE') { m.jobs = (m.jobs || []).filter(function (j) { return j.id !== mm[1]; }); return delay({ ok: true, id: mm[1] }); }
      if ((mm = /^\/v1\/jobs\/([^/]+)\/fix$/.exec(path)) && method === 'POST') { mockFixPolls = 0; return delay({ fix_id: m.fix.fix_id }, 300); }
      if ((mm = /^\/v1\/jobs\/([^/]+)\/fixes\/([^/]+)$/.exec(path)) && method === 'GET') {
        mockFixPolls++;
        var f = clone(m.fix);
        if (mockFixPolls < 4) { f.status = 'running'; f.progress = mockFixPolls / 4; f.results = f.results.slice(0, mockFixPolls - 1); f.summary = null; }
        return delay(f);
      }
      if (method === 'POST' && path === '/v1/check-text') {
        var text = (opts.json && opts.json.text) || '';
        var words = ['record', 'records', 'live', 'read', 'reads', 'present', 'content', 'project', 'produce', 'object', 'lead', 'minute', 'wind', 'tear', 'bow', 'close', 'use', 'desert', 'permit', 'contract', 'increase', 'subject', 'progress', 'conduct', 'refuse', 'rebel', 'suspect', 'address', 'combine', 'perfect', 'separate', 'estimate', 'graduate', 'moderate', 'alternate', 'attribute', 'invalid', 'convert', 'export', 'import', 'insert', 'impact', 'update', 'upgrade', 'survey', 'transfer', 'compound', 'digest', 'escort', 'extract', 'protest', 'console', 'entrance', 'resume', 'frequent', 'house', 'excuse', 'abuse', 'sow', 'does', 'polish', 'putting', 'axes', 'bass', 'buffet', 'dove', 'invalid', 'minute', 'number', 'row', 'wound', 'learned', 'intimate', 'delegate', 'advocate', 'associate', 'duplicate', 'elaborate', 'articulate', 'appropriate', 'approximate', 'coordinate', 'deliberate', 'affect', 'refund', 'rerun', 'relay', 'recall', 'reject', 'present'];
        var occ = []; var re = /[A-Za-z][A-Za-z'\-]*/g; var mt;
        while ((mt = re.exec(text))) {
          var w = mt[0].toLowerCase(), stripped = w.replace(/^(re|un|pre|mis)-?/, '');
          if (words.indexOf(w) >= 0 || (stripped !== w && words.indexOf(stripped) >= 0)) occ.push({ word: mt[0], char_start: mt.index });
        }
        return delay({ occurrences: occ });
      }
      return Promise.reject(new ApiError('mock', 'Mock mode has no handler for ' + method + ' ' + path, 404));
    });
  }
  // Synthesized clip so the sample job's players actually play: a soft two-tone "voice" blip, 16 kHz PCM wav.
  var toneCache = {};
  function mockTone(key) {
    key = String(key || 'x');
    if (toneCache[key]) return toneCache[key];
    var seed = 0; for (var i = 0; i < key.length; i++) seed = (seed * 31 + key.charCodeAt(i)) >>> 0;
    var sr = 16000, isSentence = /sentence|media|autopilot/.test(key), dur = isSentence ? 2.6 : 0.7;
    var n = Math.floor(sr * dur), f0 = 110 + (seed % 60), buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
    var w = function (o, s) { for (var j = 0; j < s.length; j++) v.setUint8(o + j, s.charCodeAt(j)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVE'); w(12, 'fmt '); v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, n * 2, true);
    for (var k = 0; k < n; k++) {
      var t = k / sr, syl = Math.sin(Math.PI * t * (isSentence ? 4.2 : 3.1)); syl = Math.max(0, syl);
      var env = Math.min(1, t / 0.03) * Math.min(1, (dur - t) / 0.08) * (0.35 + 0.65 * syl);
      var s = Math.sin(2 * Math.PI * f0 * t) * 0.55 + Math.sin(2 * Math.PI * f0 * 2.01 * t) * 0.25 + Math.sin(2 * Math.PI * f0 * 3.02 * t) * 0.12;
      v.setInt16(44 + k * 2, Math.round(s * env * 0.5 * 32767), true);
    }
    var url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
    toneCache[key] = url; return url;
  }
  function mockFile(u) {
    var name = basename(u);
    if (/report\.json$/.test(name)) return new Blob([JSON.stringify(state.job, null, 2)], { type: 'application/json' });
    if (/report\.html$/.test(name)) return new Blob([buildReportHtml(state.job)], { type: 'text/html' });
    return fetch(mockTone(u)).then(function (r) { return r.blob(); });
  }

  /* ------------------------------ audio player ------------------------------ */

  var currentPlayer = null;
  function stopAllPlayers() { state.players.forEach(function (p) { p.stop(); }); state.players = []; }
  // Compact custom player: needs no <audio controls>, so auth headers and blob URLs work the same way.
  function player(src, opts) {
    opts = opts || {};
    var bars = [], nb = opts.bars || 28;
    var barsEl = h('div', { class: 'player-bars', 'aria-hidden': 'true' });
    var seed = 0; var sk = String(src || ''); for (var i = 0; i < sk.length; i++) seed = (seed * 31 + sk.charCodeAt(i)) >>> 0;
    for (var b = 0; b < nb; b++) {
      seed = (seed * 1103515245 + 12345) >>> 0;
      var hgt = 25 + (seed % 70);
      var bar = h('i', { style: 'height:' + hgt + '%' }); bars.push(bar); barsEl.appendChild(bar);
    }
    var btn = h('button', { class: 'play', type: 'button', 'aria-label': 'Play ' + (opts.label || 'clip'), html: iconPlay() });
    var time = h('span', { class: 'player-time', text: opts.duration != null ? fmtClock(opts.duration) : '' });
    var track = h('div', { class: 'player-track' }, [barsEl]);
    var body = h('div', { class: 'player-body' }, [opts.label ? h('div', { class: 'player-label', text: opts.label }) : null, track]);
    var root = h('div', { class: 'player' + (opts.kind ? ' ' + opts.kind : '') }, [btn, body, time]);
    var audio = null, dur = opts.duration || 0;
    function paint(frac) { var on = Math.round(frac * nb); bars.forEach(function (bb, j) { bb.classList.toggle('on', j < on); }); }
    function ensure() {
      if (audio) return Promise.resolve(audio);
      root.classList.add('loading');
      return mediaSrc(src).then(function (url) {
        audio = new Audio(); audio.preload = 'auto'; audio.src = url;
        audio.addEventListener('timeupdate', function () { var d = audio.duration || dur; if (d) { paint(audio.currentTime / d); time.textContent = fmtClock(audio.currentTime); } });
        audio.addEventListener('loadedmetadata', function () { if (isFinite(audio.duration)) { dur = audio.duration; if (!audio.currentTime) time.textContent = fmtClock(dur); } });
        audio.addEventListener('ended', function () { root.classList.remove('playing'); btn.innerHTML = iconPlay(); paint(1); time.textContent = fmtClock(dur); setTimeout(function () { paint(0); }, 400); });
        audio.addEventListener('error', function () { root.classList.remove('playing', 'loading'); root.classList.add('failed'); toast('Could not play ' + (opts.label || basename(src)), 'error'); });
        root.classList.remove('loading');
        return audio;
      }, function (e) { root.classList.remove('loading'); root.classList.add('failed'); reportError(e); throw e; });
    }
    var self = {
      el: root,
      stop: function () { if (audio && !audio.paused) audio.pause(); root.classList.remove('playing'); btn.innerHTML = iconPlay(); },
      toggle: function () {
        if (audio && !audio.paused) { self.stop(); return; }
        ensure().then(function (a) {
          if (currentPlayer && currentPlayer !== self) currentPlayer.stop();
          currentPlayer = self;
          root.classList.add('playing'); btn.innerHTML = iconPause();
          var p = a.play(); if (p && p.catch) p.catch(function () { self.stop(); });
        }, function () { /* already toasted */ });
      }
    };
    btn.addEventListener('click', self.toggle);
    track.addEventListener('click', function (ev) {
      var r = track.getBoundingClientRect(); var frac = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
      ensure().then(function (a) { var d = a.duration || dur; if (d) { a.currentTime = frac * d; paint(frac); } if (a.paused) self.toggle(); }, function () {});
    });
    state.players.push(self);
    return root;
  }
  function iconPlay() { return '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.5v9l8-4.5z"/></svg>'; }
  function iconPause() { return '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 1.5h3v9H2zM7 1.5h3v9H7z"/></svg>'; }

  /* ------------------------------ router ------------------------------ */

  function parseRoute() {
    var hsh = window.location.hash.replace(/^#\/?/, '');
    var parts = hsh.split('/').filter(Boolean);
    if (!parts.length) return { name: 'home' };
    if (parts[0] === 'api') return { name: 'api' };
    if (parts[0] === 'jobs' && parts[1]) return { name: 'job', id: decodeURIComponent(parts[1]) };
    return { name: 'home' };
  }
  function navigate(hash) { if (window.location.hash === hash) route(); else window.location.hash = hash; }
  function showView(name) {
    ['home', 'job', 'results', 'api'].forEach(function (v) { $('#view-' + v).classList.toggle('hidden', v !== name); });
    $all('.nav a').forEach(function (a) { a.classList.toggle('active', a.dataset.nav === (name === 'api' ? 'api' : 'home')); });
    window.scrollTo(0, 0);
  }
  function route() {
    stopPolling(); stopAllPlayers();
    var r = parseRoute(); state.route = r;
    if (r.name === 'home') { showView('home'); loadJobs(); }
    else if (r.name === 'api') { showView('api'); renderApiPage(); }
    else if (r.name === 'job') { openJob(r.id); }
  }
  function stopPolling() {
    if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
    if (state.fixTimer) { clearTimeout(state.fixTimer); state.fixTimer = null; }
  }

  /* ------------------------------ health + voices ------------------------------ */

  function refreshHealth(silent) {
    var dot = $('#api-dot'), txt = $('#api-status-text');
    return api('GET', '/v1/health').then(function (hh) {
      state.health = hh;
      var ok = !!hh.ok;
      dot.className = 'dot ' + (ok ? 'ok' : 'warn');
      txt.textContent = ok ? ('API ' + (hh.auth === 'open' ? 'open' : 'keyed')) : 'API degraded';
      if (hh.auth === 'key' && !apiKey() && !MOCK && !silent) toast('This API requires a key — add it in Settings.', 'info');
      return hh;
    }, function (e) {
      state.health = null;
      dot.className = 'dot bad'; txt.textContent = 'API offline';
      if (!silent) reportError(e);
      throw e;
    });
  }
  function loadVoices() {
    return api('GET', '/v1/voices').then(function (v) {
      var list = Array.isArray(v) ? v : (v.voices || v.items || []);
      state.voices = list;
      $all('.voice-select').forEach(fillVoiceSelect);
      return list;
    }, function (e) {
      $all('.voice-select').forEach(function (sel) { sel.innerHTML = '<option value="">voices unavailable</option>'; });
      if (!(e && e.code === 'network')) reportError(e, 'Voices');
    });
  }
  function fillVoiceSelect(sel) {
    var saved = lsGet(LS.voice) || sel.value;
    sel.innerHTML = '';
    if (!state.voices.length) { sel.appendChild(h('option', { value: '', text: 'no voices' })); return; }
    state.voices.forEach(function (v) {
      var label = v.name + (v.style ? ' · ' + v.style : '') + (v.accent ? ' · ' + v.accent : '');
      sel.appendChild(h('option', { value: String(v.speaker_id), text: label + '  (#' + v.speaker_id + ')' }));
    });
    if (saved && $all('option', sel).some(function (o) { return o.value === saved; })) sel.value = saved;
    sel.addEventListener('change', function () { lsSet(LS.voice, sel.value); });
  }
  function fillModelSelect(sel) {
    var saved = lsGet(LS.model); if (saved) sel.value = saved;
    sel.addEventListener('change', function () { lsSet(LS.model, sel.value); });
  }
  function voiceName(id) {
    var v = state.voices.filter(function (x) { return String(x.speaker_id) === String(id); })[0];
    return v ? v.name + (v.style ? ' · ' + v.style : '') : ('#' + id);
  }

  /* ------------------------------ home view ------------------------------ */

  function initHome() {
    // tabs
    $all('#intake-tabs .tab').forEach(function (t) {
      t.addEventListener('click', function () {
        $all('#intake-tabs .tab').forEach(function (x) { x.classList.toggle('active', x === t); x.setAttribute('aria-selected', x === t ? 'true' : 'false'); });
        $('#tab-upload').classList.toggle('hidden', t.dataset.tab !== 'upload');
        $('#tab-script').classList.toggle('hidden', t.dataset.tab !== 'script');
      });
    });
    // dropzone
    var dz = $('#dropzone'), fi = $('#file-input');
    dz.addEventListener('click', function () { fi.click(); });
    dz.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });
    ['dragenter', 'dragover'].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add('over'); }); });
    ['dragleave', 'drop'].forEach(function (ev) { dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove('over'); }); });
    dz.addEventListener('drop', function (e) { if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
    fi.addEventListener('change', function () { if (fi.files && fi.files[0]) setFile(fi.files[0]); });
    $('#btn-check-upload').addEventListener('click', submitUpload);
    // script
    var st = $('#script-text');
    var hetTimer = null;
    st.addEventListener('input', function () {
      $('#script-count').textContent = wordCount(st.value) + ' words';
      if (hetTimer) clearTimeout(hetTimer);
      hetTimer = setTimeout(preflightScript, 600);
    });
    $('#btn-render-check').addEventListener('click', submitScript);
    fillModelSelect($('#model-script'));
    $('#btn-refresh-jobs').addEventListener('click', function () { loadJobs(true); });
    if (MOCK) { $('#demo-link').textContent = 'Sample job →'; $('#demo-link').setAttribute('href', '#/jobs/' + ((state.mock && state.mock.job.id) || 'j_7f3a2c')); }
  }
  function setFile(f) {
    state.file = f;
    var dz = $('#dropzone'), lbl = $('#dz-file');
    dz.classList.add('has-file');
    lbl.classList.remove('hidden');
    lbl.innerHTML = '<b>' + esc(f.name) + '</b> ' + (f.size / 1048576).toFixed(1) + ' MB';
    $('#btn-check-upload').disabled = false;
    if (f.size > 500 * 1048576) toast('That file is over the 500 MB limit.', 'error');
  }
  var preflightSeq = 0;
  function preflightScript() {
    var text = $('#script-text').value.trim(), out = $('#script-hets');
    if (wordCount(text) < 2) { out.textContent = ''; return; }
    var seq = ++preflightSeq;
    api('POST', '/v1/check-text', { json: { text: text } }).then(function (r) {
      if (seq !== preflightSeq) return;
      var occ = (r && (r.occurrences || r.items)) || (Array.isArray(r) ? r : []);
      var words = occ.map(function (o) { return o.word; }).filter(Boolean);
      var uniq = words.filter(function (w, i) { return words.indexOf(w) === i; });
      out.textContent = occ.length ? (occ.length + ' heteronym' + (occ.length === 1 ? '' : 's') + ' to check: ' + uniq.slice(0, 8).join(', ') + (uniq.length > 8 ? '…' : '')) : 'no heteronyms in this script';
    }, function () { out.textContent = ''; });
  }
  function submitUpload() {
    if (!state.file) { toast('Pick an audio or video file first.', 'info'); return; }
    var fd = new FormData();
    fd.append('media', state.file, state.file.name);
    var script = $('#upload-script').value.trim(); if (script) fd.append('script', script);
    fd.append('engine', $('#engine-select').value);
    var btn = $('#btn-check-upload'); btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Uploading…';
    api('POST', '/v1/jobs', { form: fd, mode: 'upload' }).then(function (r) {
      btn.disabled = false; btn.textContent = 'Check pronunciation';
      if (!r || !r.job_id) throw new ApiError('bad_response', 'The API did not return a job_id');
      toast('Job ' + r.job_id + ' queued', 'ok');
      navigate('#/jobs/' + encodeURIComponent(r.job_id));
    }, function (e) { btn.disabled = false; btn.textContent = 'Check pronunciation'; reportError(e, 'Upload'); });
  }
  function submitScript() {
    var text = $('#script-text').value.trim();
    if (wordCount(text) < 2) { toast('Paste a script first.', 'info'); return; }
    var voice = $('#voice-script').value, model = $('#model-script').value;
    if (!voice) { toast('Pick a WellSaid voice.', 'info'); return; }
    var btn = $('#btn-render-check'); btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Submitting…';
    api('POST', '/v1/jobs', { json: { text: text, speaker_id: Number(voice), model: model }, mode: 'script' }).then(function (r) {
      btn.disabled = false; btn.textContent = 'Render & check';
      if (!r || !r.job_id) throw new ApiError('bad_response', 'The API did not return a job_id');
      toast('Rendering with ' + voiceName(voice) + ' on ' + model, 'ok');
      navigate('#/jobs/' + encodeURIComponent(r.job_id));
    }, function (e) { btn.disabled = false; btn.textContent = 'Render & check'; reportError(e, 'Render'); });
  }
  function loadJobs(manual) {
    var box = $('#jobs-list');
    api('GET', '/v1/jobs').then(function (r) {
      var list = Array.isArray(r) ? r : (r.jobs || r.items || []);
      state.jobs = list;
      box.innerHTML = '';
      if (!list.length) { box.appendChild(h('div', { class: 'empty', text: 'No jobs yet — drop a file above.' })); return; }
      list.forEach(function (j) { box.appendChild(jobRow(j)); });
      if (manual) toast('Jobs refreshed', 'info');
    }, function (e) {
      box.innerHTML = '';
      box.appendChild(h('div', { class: 'empty', text: e.code === 'network' ? 'API not reachable — set the base URL in Settings.' : ('Could not list jobs: ' + e.message) }));
      if (manual) reportError(e, 'Jobs');
    });
  }
  function jobRow(j) {
    var res = j.result || {};
    var name = j.filename || j.name || (res.media ? basename(res.media) : null) || (j.mode === 'script' ? 'script render' : j.id);
    var flags = j.flags != null ? j.flags : (res.flags != null ? res.flags : null);
    var n = j.occurrences != null ? j.occurrences : (res.occurrences ? res.occurrences.length : null);
    var chipClass = j.status, chipText = j.status;
    if (j.status === 'done') { if (flags) { chipClass = 'flagged'; chipText = flags + ' flagged'; } else { chipText = n != null ? 'clean · ' + n + ' checked' : 'done'; } }
    if (j.status === 'running') chipText = (j.stage || 'running') + (j.progress != null ? ' ' + Math.round(j.progress * 100) + '%' : '');
    var row = h('a', { class: 'job-row', href: '#/jobs/' + encodeURIComponent(j.id) }, [
      h('span', { class: 'job-id', text: j.id }),
      h('span', { class: 'job-name', text: name, title: name }),
      h('span', { class: 'status-chip ' + chipClass, text: chipText }),
      h('span', { class: 'job-when', text: fmtWhen(j.created) })
    ]);
    var del = h('button', { class: 'btn btn-mini btn-danger job-del', type: 'button', text: 'Delete', title: 'Delete job', onclick: function (e) { e.preventDefault(); e.stopPropagation(); deleteJob(j.id, row); } });
    row.appendChild(del);
    return row;
  }
  function deleteJob(id, rowEl) {
    if (!window.confirm('Delete job ' + id + ' and its files?')) return;
    api('DELETE', '/v1/jobs/' + encodeURIComponent(id)).then(function () {
      toast('Deleted ' + id, 'ok');
      if (rowEl && rowEl.parentNode) rowEl.parentNode.removeChild(rowEl);
      if (state.route.name === 'job' && state.route.id === id) navigate('#/');
      else if (state.route.name === 'home' && !$('#jobs-list').children.length) loadJobs();
    }, function (e) { reportError(e, 'Delete'); });
  }

  /* ------------------------------ job view (progress) ------------------------------ */

  function openJob(id) {
    state.job = null; state.fix = null; state.fixId = null;
    showView('job');
    $('#job-title').textContent = 'Opening job…';
    $('#job-sub').textContent = id;
    $('#job-error').classList.add('hidden');
    $('.progress-card').classList.remove('is-error');
    $('#job-cancel').classList.add('hidden');
    $('#job-cancel').onclick = function () { deleteJob(id); };
    $('#job-back').onclick = function () { navigate('#/'); };
    renderStages(-1);
    pollJob(id);
  }
  function stageIndex(stage, status) {
    var s = String(stage || '').toLowerCase();
    if (status === 'done') return STAGES.length - 1;
    if (status === 'queued') return -1;
    for (var i = 0; i < STAGES.length; i++) for (var j = 0; j < STAGES[i].match.length; j++) if (s.indexOf(STAGES[i].match[j]) >= 0) return i;
    return status === 'running' ? 0 : -1;
  }
  function renderStages(active) {
    var ol = $('#stages'); ol.innerHTML = '';
    STAGES.forEach(function (st, i) {
      ol.appendChild(h('li', { class: 'stage' + (i < active ? ' done' : i === active ? ' active' : '') }, [h('i'), st.label]));
    });
  }
  function pollJob(id) {
    api('GET', '/v1/jobs/' + encodeURIComponent(id)).then(function (j) {
      if (state.route.name !== 'job' || state.route.id !== id) return;
      state.job = j;
      if (j.status === 'done' && j.result) { renderResults(j); return; }
      var idx = stageIndex(j.stage, j.status);
      renderStages(idx);
      var label = j.status === 'queued' ? 'Queued — waiting for the worker' : (idx >= 0 ? STAGES[idx].label : 'Working');
      if (j.stage && /render/i.test(j.stage)) label = 'Rendering with WellSaid';
      $('#job-title').textContent = label + '…';
      var pct = j.progress != null ? Math.round(Math.min(1, Math.max(0, j.progress)) * 100) : (idx >= 0 ? Math.round((idx / (STAGES.length - 1)) * 100) : 0);
      $('#job-bar').style.width = pct + '%';
      $('#job-pct').textContent = pct + '%' + (j.stage ? ' · ' + j.stage : '') + (j.mode ? ' · ' + j.mode : '');
      $('#job-sub').textContent = j.id + (j.created ? ' · ' + fmtWhen(j.created) : '');
      if (j.status === 'error') {
        $('.progress-card').classList.add('is-error');
        $('#job-title').textContent = 'This job failed';
        var err = j.error; if (err && typeof err === 'object') err = err.message || JSON.stringify(err);
        $('#job-error').textContent = err || 'Unknown error'; $('#job-error').classList.remove('hidden');
        $('#job-cancel').classList.remove('hidden');
        return;
      }
      if (j.status === 'done' && !j.result) { $('#job-error').textContent = 'Job is done but has no result payload.'; $('#job-error').classList.remove('hidden'); return; }
      state.pollTimer = setTimeout(function () { pollJob(id); }, POLL_MS);
    }, function (e) {
      if (state.route.name !== 'job' || state.route.id !== id) return;
      $('.progress-card').classList.add('is-error');
      $('#job-title').textContent = e.status === 404 ? 'Job not found' : "Can't reach the job";
      $('#job-error').textContent = e.message; $('#job-error').classList.remove('hidden');
      if (e.code === 'network') state.pollTimer = setTimeout(function () { pollJob(id); }, POLL_MS * 3);
      else reportError(e);
    });
  }

  /* ------------------------------ results view ------------------------------ */

  function occsOf(job) { return (job && job.result && job.result.occurrences) || []; }
  function flaggedOf(job) { return occsOf(job).filter(function (o) { return o.flag; }); }
  function fixResultFor(occId) {
    if (!state.fix || !state.fix.results) return null;
    return state.fix.results.filter(function (r) { return r.occurrence_id === occId; })[0] || null;
  }

  function renderResults(job) {
    showView('results');
    stopAllPlayers();
    state.filter = 'all';
    renderHead(job);
    renderTranscript(job);
    renderOccurrences(job);
    renderFixPanel(job);
    $all('.filter').forEach(function (f) {
      f.classList.toggle('active', f.dataset.filter === state.filter);
      f.onclick = function () { state.filter = f.dataset.filter; $all('.filter').forEach(function (x) { x.classList.toggle('active', x === f); }); applyFilter(); };
    });
    applyFilter();
  }
  function applyFilter() {
    var shown = 0;
    $all('.occ').forEach(function (el) { var show = state.filter === 'all' || el.classList.contains('flag'); el.classList.toggle('hidden', !show); if (show) shown++; });
    var total = occsOf(state.job).length;
    $('#occ-count').textContent = state.filter === 'all' ? total : shown + ' of ' + total;
  }

  function renderHead(job) {
    var res = job.result, occs = occsOf(job), flagged = flaggedOf(job);
    var fixedN = flagged.filter(function (o) { var r = fixResultFor(o.id); return r && r.verified; }).length;
    var verdictClass = flagged.length === 0 ? 'clean' : (fixedN === flagged.length ? 'fixed' : 'flag');
    var verdictText = flagged.length === 0
      ? (occs.length ? 'Clean — every heteronym read as intended' : 'Clean — no heteronyms to check')
      : (fixedN === flagged.length ? 'All ' + flagged.length + ' flagged line' + (flagged.length === 1 ? '' : 's') + ' fixed and verified'
        : fixedN ? (flagged.length - fixedN) + ' still to fix · ' + fixedN + ' fixed' : flagged.length + ' mispronunciation' + (flagged.length === 1 ? '' : 's') + ' to fix');
    var verdictIcon = verdictClass === 'flag'
      ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 9v4M12 17h.01M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20h15.4a2 2 0 0 0 1.7-2.8L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>'
      : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
    var name = basename(res.media) || job.id;
    var head = $('#results-head'); head.innerHTML = '';
    head.appendChild(h('div', { class: 'head-top' }, [
      h('div', { class: 'head-file' }, [
        h('h2', { text: name }),
        h('div', { class: 'meta' }, [
          h('span', { text: job.id }),
          h('span', { text: fmtTime(res.duration) + ' long' }),
          job.mode ? h('span', { text: job.mode }) : null,
          job.created ? h('span', { text: fmtWhen(job.created) }) : null,
          occs[0] && occs[0].intent_engine ? h('span', { text: 'intent: ' + occs[0].intent_engine }) : null
        ])
      ]),
      h('div', { class: 'verdict ' + verdictClass, html: verdictIcon + '<span>' + esc(verdictText) + '</span>' })
    ]));
    head.appendChild(h('div', { class: 'head-stats' }, [
      stat(fmtTime(res.duration), 'duration'),
      stat(String(occs.length), 'heteronyms'),
      stat(String(flagged.length), 'flagged', flagged.length ? 'flagged' : 'clean'),
      stat(String((res.transcript && res.transcript.words && res.transcript.words.length) || wordCount(res.transcript && res.transcript.text)), 'words'),
      state.fix && state.fix.results ? stat(fixedN + '/' + flagged.length, 'verified fixes', fixedN === flagged.length ? 'clean' : 'flagged') : null
    ]));
    var actions = h('div', { class: 'head-actions' }, [
      flagged.length ? h('button', { class: 'btn btn-primary', type: 'button', text: 'Fix ' + flagged.length + ' flagged', onclick: function () { $('#fix-panel').scrollIntoView({ behavior: 'smooth', block: 'start' }); } }) : null,
      h('button', { class: 'btn btn-ghost', type: 'button', text: 'Download report (JSON)', onclick: function () { downloadReport('json'); } }),
      h('button', { class: 'btn btn-ghost', type: 'button', text: 'Report (HTML)', onclick: function () { downloadReport('html'); } }),
      h('button', { class: 'btn btn-ghost', type: 'button', text: 'Copy curl', title: 'curl for this job', onclick: function () { copyText(curlForJob(job.id), 'curl'); } }),
      h('a', { class: 'btn btn-ghost', href: '#/api', text: 'API' }),
      h('button', { class: 'btn btn-ghost', type: 'button', text: 'New check', onclick: function () { navigate('#/'); } })
    ]);
    head.appendChild(actions);
    if (res.media) head.appendChild(h('div', { class: 'head-player' }, [player(res.media, { label: 'Full recording · ' + name, duration: res.duration, bars: 64 })]));
  }
  function stat(num, lbl, cls) {
    return h('div', { class: 'hs' }, [h('div', { class: 'hs-num' + (cls ? ' ' + cls : ''), text: num }), h('div', { class: 'hs-lbl', text: lbl })]);
  }

  // Locate every occurrence in the transcript text (sentence-then-word scan, in time order) and mark it.
  function renderTranscript(job) {
    var res = job.result, text = (res.transcript && res.transcript.text) || '';
    var occs = occsOf(job).slice().sort(function (a, b) { return (a.t_start || 0) - (b.t_start || 0); });
    var p = $('#transcript'); p.innerHTML = '';
    if (!text) { p.appendChild(h('span', { class: 'muted', text: 'No transcript.' })); return; }
    var marks = []; // {start,end,occ}
    var cursor = 0, lastSentPos = -1, wordCursor = 0;
    occs.forEach(function (o) {
      var sPos = o.sentence ? text.indexOf(o.sentence, Math.max(0, cursor - 1)) : -1;
      if (sPos < 0 && o.sentence) sPos = text.indexOf(o.sentence);
      var from, to;
      if (sPos >= 0) { if (sPos !== lastSentPos) { lastSentPos = sPos; wordCursor = sPos; } from = wordCursor; to = sPos + o.sentence.length; }
      else { from = cursor; to = text.length; }
      var re = new RegExp('(^|[^A-Za-z\\-\'’])(' + o.word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')(?![A-Za-z\\-])', 'i');
      var slice = text.substring(from, to), m = re.exec(slice);
      if (!m) { re = new RegExp('(^|[^A-Za-z])(' + o.word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'i'); m = re.exec(text.substring(cursor)); if (m) { from = cursor; } }
      if (!m) return;
      var start = from + m.index + m[1].length, end = start + m[2].length;
      marks.push({ start: start, end: end, occ: o });
      wordCursor = end; cursor = Math.max(cursor, end);
    });
    marks.sort(function (a, b) { return a.start - b.start; });
    var pos = 0;
    marks.forEach(function (mk) {
      if (mk.start < pos) return;
      if (mk.start > pos) p.appendChild(document.createTextNode(text.substring(pos, mk.start)));
      var r = fixResultFor(mk.occ.id);
      var cls = 'het ' + (mk.occ.flag ? 'flag' : 'clean') + (r && r.verified ? ' fixed' : '');
      var el = h('mark', { class: cls, text: text.substring(mk.start, mk.end), tabindex: '0', role: 'button', dataset: { occ: mk.occ.id },
        title: (mk.occ.flag ? 'Flagged · ' : 'Clean · ') + fmtTime(mk.occ.t_start) + ' · meant ' + (mk.occ.intended_respell || mk.occ.intended) });
      var go = function () { jumpToOcc(mk.occ.id); };
      el.addEventListener('click', go);
      el.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
      p.appendChild(el);
      pos = mk.end;
    });
    if (pos < text.length) p.appendChild(document.createTextNode(text.substring(pos)));
  }
  function jumpToOcc(id) {
    var card = $('#occ-' + id);
    if (!card) return;
    if (card.classList.contains('hidden')) { state.filter = 'all'; $all('.filter').forEach(function (x) { x.classList.toggle('active', x.dataset.filter === 'all'); }); applyFilter(); }
    card.scrollIntoView({ behavior: document.hidden ? 'auto' : 'smooth', block: 'start' });
    card.classList.remove('flash'); void card.offsetWidth; card.classList.add('flash');
  }

  function renderOccurrences(job) {
    var box = $('#occurrences'); box.innerHTML = '';
    var occs = occsOf(job);
    if (!occs.length) { box.appendChild(h('div', { class: 'card empty', text: 'No heteronyms in this recording — nothing to second-guess.' })); return; }
    occs.forEach(function (o) { box.appendChild(occCard(o)); });
  }
  function occCard(o) {
    var r = fixResultFor(o.id);
    var fixed = !!(r && r.verified);
    var cls = 'occ ' + (o.flag ? 'flag' : 'clean') + (fixed ? ' fixed' : '');
    var compound = o.base && o.word && o.word.toLowerCase().replace(/[-'’]/g, '') !== o.base.toLowerCase();
    var sentenceHtml = highlightWord(o.sentence, o.word);
    var intendedIpa = o.intended_ipa || (o.candidates && o.candidates[o.intended]) || '';
    var detectedIpa = (o.candidates && o.detected && o.candidates[o.detected]) || '';
    var undecidable = o.detected == null;
    var conf = (o.confidence || 'low').toLowerCase().replace('medium', 'med');

    var card = h('article', { class: cls, id: 'occ-' + o.id, dataset: { occ: o.id } }, [
      h('div', { class: 'occ-head' }, [
        h('span', { class: 'occ-word', html: esc(o.word) + (compound ? '<small>' + esc(o.base) + ' + affix</small>' : '') }),
        h('span', { class: 'occ-time', text: fmtTime(o.t_start) + ' – ' + fmtTime(o.t_end) }),
        h('span', { class: 'chip ' + (fixed ? 'fixed' : o.flag ? 'flag' : 'clean'), text: fixed ? 'fixed ✓' : o.flag ? 'flagged' : 'clean' }),
        h('span', { class: 'chip conf-' + conf, title: 'Confidence of the intent decision + phoneme margin', text: 'confidence ' + conf }),
        h('span', { class: 'spacer' }),
        o.intent_engine ? h('span', { class: 'chip engine', text: 'intent · ' + o.intent_engine }) : null,
        h('span', { class: 'occ-id', text: o.id })
      ]),
      h('p', { class: 'occ-sentence', html: sentenceHtml }),
      h('div', { class: 'meant-grid' }, [
        h('div', { class: 'meant you' }, [
          h('div', { class: 'meant-h', text: 'You meant' }),
          h('div', { class: 'meant-sense', html: '<span class="pos">' + esc(posOf(o.intended)) + '</span>' + esc(senseText(o.intended_desc) || o.intended) }),
          h('div', { class: 'meant-row' }, [
            o.intended_respell ? h('span', { class: 'respell', title: 'WellSaid respelling', text: o.intended_respell }) : null,
            h('span', { class: 'ipa', text: '/' + intendedIpa + '/' })
          ])
        ]),
        h('div', { class: 'meant it' }, [
          h('div', { class: 'meant-h', text: 'It said' }),
          h('div', { class: 'meant-sense', html: undecidable ? '<span class="muted">undecidable — clip too short to score</span>' : '<span class="pos">' + esc(posOf(o.detected)) + '</span>' + esc(senseText(o.detected_desc) || o.detected) }),
          h('div', { class: 'meant-row' }, [
            o.detected_desc && senseRespell(o.detected_desc) ? h('span', { class: 'respell', text: senseRespell(o.detected_desc) }) : null,
            detectedIpa ? h('span', { class: 'ipa', text: '/' + detectedIpa + '/' }) : null,
            o.free_decode ? h('span', { class: 'ipa-lbl', text: 'heard' }) : null,
            o.free_decode ? h('span', { class: 'ipa heard', title: 'Free phoneme decode (greedy CTC)', text: '/' + o.free_decode + '/' }) : null
          ])
        ])
      ]),
      scoresBlock(o),
      h('div', { class: 'occ-players' }, [
        o.clip_wav ? player(o.clip_wav, { label: 'Word · ' + o.word, duration: (o.t_end - o.t_start) || null, kind: o.flag ? 'before' : '' }) : null,
        o.sentence_wav ? player(o.sentence_wav, { label: 'Sentence', duration: (o.sentence_t_end != null && o.sentence_t_start != null) ? (o.sentence_t_end - o.sentence_t_start) : null, bars: 40 }) : null
      ]),
      o.reason && o.reason !== 'ok' ? h('div', { class: 'occ-reason', html: '<b>why:</b> ' + esc(o.reason) }) : null
    ]);
    return card;
  }
  function highlightWord(sentence, word) {
    if (!sentence) return '';
    var re = new RegExp('(^|[^A-Za-z\\-\'’])(' + String(word || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')(?![A-Za-z\\-])', 'i');
    var m = re.exec(sentence);
    if (!m) return esc(sentence);
    var s = m.index + m[1].length;
    return esc(sentence.substring(0, s)) + '<mark>' + esc(m[2]) + '</mark>' + esc(sentence.substring(s + m[2].length));
  }
  function scoresBlock(o) {
    var sc = o.scores;
    if (!sc || typeof sc !== 'object') return null;
    var ids = Object.keys(sc);
    if (!ids.length) return null;
    var vals = ids.map(function (k) { return sc[k]; }).filter(function (v) { return typeof v === 'number' && isFinite(v); });
    if (!vals.length) return null;
    var best = Math.max.apply(null, vals), worst = Math.min.apply(null, vals);
    var span = Math.max(0.5, best - worst) * 1.35;
    var rows = ids.map(function (id) {
      var v = sc[id]; var ok = typeof v === 'number' && isFinite(v);
      var frac = ok ? Math.max(0.06, 1 - (best - v) / span) : 0;
      var cls = 'score' + (id === o.intended ? ' intended' : '') + (id === o.detected ? ' detected' + (o.detected !== o.intended ? ' wrong' : '') : '');
      return h('div', { class: cls }, [
        h('span', { class: 'sid', text: id + (id === o.intended ? ' · meant' : id === o.detected ? ' · heard' : '') }),
        h('div', { class: 'bar' }, [h('i', { style: 'width:' + Math.round(frac * 100) + '%' })]),
        h('span', { class: 'val', text: ok ? v.toFixed(2) : '–' })
      ]);
    });
    var box = h('div', { class: 'scores', title: 'CTC log-likelihood per phoneme for each candidate pronunciation' }, rows);
    if (o.margin != null) box.appendChild(h('div', { class: 'margin-note', text: 'margin ' + Number(o.margin).toFixed(2) + ' between the top two candidates' }));
    return box;
  }

  /* ------------------------------ fix panel ------------------------------ */

  function renderFixPanel(job) {
    var panel = $('#fix-panel'); panel.innerHTML = '';
    var flagged = flaggedOf(job);
    if (!flagged.length) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    var voiceSel = h('select', { class: 'voice-select', id: 'fix-voice', 'aria-label': 'Voice' });
    fillVoiceSelect(voiceSel);
    if (!state.voices.length) loadVoices().then(function () { fillVoiceSelect(voiceSel); });
    var modelSel = h('select', { class: 'model-select', id: 'fix-model', 'aria-label': 'Model' }, [
      h('option', { value: 'preview', text: 'Preview (13.3) · spelling-as-pronunciation' }),
      h('option', { value: 'caruso', text: 'Caruso · inline <respell>' })
    ]);
    fillModelSelect(modelSel);
    var running = state.fix && state.fix.status && state.fix.status !== 'done' && state.fix.status !== 'error';
    var btn = h('button', { class: 'btn btn-primary', type: 'button', disabled: running, html: running ? '<span class="spin"></span> Fixing…' : 'Re-render ' + flagged.length + ' flagged line' + (flagged.length === 1 ? '' : 's') });
    btn.addEventListener('click', function () { startFix(job, voiceSel.value, modelSel.value); });
    panel.appendChild(h('div', { class: 'fix-head' }, [h('h2', { text: 'Fix flagged lines' }), h('span', { class: 'chip flag', text: flagged.length + ' flagged' })]));
    panel.appendChild(h('p', { class: 'fix-sub', text: 'Retake re-renders each flagged sentence through WellSaid with the model-appropriate pronunciation mechanism, then listens again. A take only counts as fixed when the phoneme scorer agrees.' }));
    panel.appendChild(h('div', { class: 'fix-controls' }, [
      h('label', { for: 'fix-voice', class: 'muted small', text: 'Voice' }), voiceSel,
      h('label', { for: 'fix-model', class: 'muted small', text: 'Model' }), modelSel,
      h('span', { class: 'spacer' }), btn
    ]));
    var prog = h('div', { class: 'fix-progress hidden', id: 'fix-progress' }, [h('span', { class: 'spin' }), h('div', { class: 'progress' }, [h('div', { class: 'progress-fill', id: 'fix-bar', style: 'width:0%' })]), h('span', { id: 'fix-status', text: 'queued' })]);
    panel.appendChild(prog);
    panel.appendChild(h('div', { class: 'fix-results', id: 'fix-results' }));
    if (state.fix) paintFix(state.fix);
  }
  function startFix(job, voice, model) {
    if (!voice) { toast('Pick a WellSaid voice first.', 'info'); return; }
    var ids = flaggedOf(job).map(function (o) { return o.id; });
    var btn = $('#fix-panel .btn-primary'); btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Submitting…';
    api('POST', '/v1/jobs/' + encodeURIComponent(job.id) + '/fix', { json: { occurrence_ids: ids, speaker_id: Number(voice), model: model } }).then(function (r) {
      if (!r || !r.fix_id) throw new ApiError('bad_response', 'The API did not return a fix_id');
      state.fixId = r.fix_id; state.fix = { status: 'queued', results: [], speaker_id: Number(voice), model: model };
      $('#fix-progress').classList.remove('hidden');
      btn.innerHTML = '<span class="spin"></span> Fixing…';
      toast('Fixing ' + ids.length + ' line' + (ids.length === 1 ? '' : 's') + ' with ' + voiceName(voice) + ' on ' + model, 'ok');
      pollFix(job.id, r.fix_id);
    }, function (e) { btn.disabled = false; btn.textContent = 'Re-render ' + ids.length + ' flagged line' + (ids.length === 1 ? '' : 's'); reportError(e, 'Fix'); });
  }
  function pollFix(jobId, fixId) {
    api('GET', '/v1/jobs/' + encodeURIComponent(jobId) + '/fixes/' + encodeURIComponent(fixId)).then(function (f) {
      if (!state.job || state.job.id !== jobId || state.route.name !== 'job') return;
      f.speaker_id = f.speaker_id != null ? f.speaker_id : state.fix && state.fix.speaker_id;
      f.model = f.model || (state.fix && state.fix.model);
      state.fix = f;
      paintFix(f);
      if (f.status === 'done' || f.status === 'error') {
        var ok = (f.results || []).filter(function (r) { return r.verified; }).length;
        if (f.status === 'error') toast('Fix failed: ' + (f.error && (f.error.message || f.error) || 'unknown error'), 'error');
        else toast(ok + ' of ' + (f.results || []).length + ' verified fixed', ok === (f.results || []).length ? 'ok' : 'info');
        renderHead(state.job); renderTranscript(state.job);
        // refresh the occurrence cards' fixed state without losing scroll position
        (f.results || []).forEach(function (r) { var c = $('#occ-' + r.occurrence_id); if (c) { var fresh = occCard(occsOf(state.job).filter(function (o) { return o.id === r.occurrence_id; })[0]); c.parentNode.replaceChild(fresh, c); } });
        applyFilter();
        return;
      }
      state.fixTimer = setTimeout(function () { pollFix(jobId, fixId); }, POLL_MS);
    }, function (e) {
      reportError(e, 'Fix status');
      if (e.code === 'network') state.fixTimer = setTimeout(function () { pollFix(jobId, fixId); }, POLL_MS * 3);
    });
  }
  function paintFix(f) {
    var box = $('#fix-results'), prog = $('#fix-progress'), btn = $('#fix-panel .btn-primary');
    if (!box) return;
    var flagged = flaggedOf(state.job), done = f.status === 'done' || f.status === 'error';
    var results = f.results || [];
    if (!done) {
      prog.classList.remove('hidden');
      var frac = f.progress != null ? f.progress : (flagged.length ? results.length / flagged.length : 0);
      $('#fix-bar').style.width = Math.round(frac * 100) + '%';
      $('#fix-status').textContent = (f.stage || f.status || 'running') + ' · ' + results.length + '/' + flagged.length;
    } else {
      prog.classList.add('hidden');
      btn.disabled = false; btn.textContent = 'Re-render again';
    }
    box.innerHTML = '';
    if (done && f.summary) {
      var summary = Array.isArray(f.summary) ? f.summary.join(' · ') : (typeof f.summary === 'object' ? JSON.stringify(f.summary) : f.summary);
      box.appendChild(h('div', { class: 'fix-summary', text: summary }));
    }
    results.forEach(function (r) {
      var o = occsOf(state.job).filter(function (x) { return x.id === r.occurrence_id; })[0];
      if (!o) return;
      box.appendChild(fixRow(o, r, f));
    });
    if (done && !results.length) box.appendChild(h('div', { class: 'empty', text: 'No takes came back.' }));
  }
  function fixRow(o, r, f) {
    var tries = r.tries || [], best = r.best || (tries.length ? tries[tries.length - 1] : null);
    var row = h('div', { class: 'fix-row ' + (r.verified ? 'ok' : 'bad') });
    row.appendChild(h('div', { class: 'fix-row-head' }, [
      h('span', { class: 'fix-word', text: o.word }),
      h('span', { class: 'occ-time', text: fmtTime(o.t_start) }),
      h('span', { class: 'fix-arrow', text: '→' }),
      h('span', { class: 'respell', text: o.intended_respell || o.intended }),
      h('span', { class: 'spacer' }),
      h('span', { class: 'vmark ' + (r.verified ? 'ok' : 'bad'), text: r.verified ? '✓ verified' : '✗ not verified' }),
      h('span', { class: 'occ-id', text: tries.length + ' tr' + (tries.length === 1 ? 'y' : 'ies') + (f.model ? ' · ' + f.model : '') + (f.speaker_id != null ? ' · ' + voiceName(f.speaker_id) : '') })
    ]));
    row.appendChild(h('div', { class: 'fix-ab' }, [
      player(o.sentence_wav || o.clip_wav, { label: 'Before · original', kind: 'before', bars: 36, duration: (o.sentence_t_end != null && o.sentence_t_start != null) ? (o.sentence_t_end - o.sentence_t_start) : null }),
      best && best.take_mp3 ? player(best.take_mp3, { label: 'After · ' + (r.verified ? 'verified take' : 'best attempt'), kind: 'after', bars: 36 }) : h('div', { class: 'player failed' }, [h('div', { class: 'player-body' }, [h('div', { class: 'player-label', text: 'After' }), h('div', { class: 'muted small', text: 'no take rendered' })])])
    ]));
    if (best) {
      row.appendChild(h('div', { class: 'fix-meta' }, [
        h('span', { text: 'heard as' }), h('span', { class: 'ipa ' + (r.verified ? '' : 'heard'), text: '/' + (best.heard_as || '?') + '/' }),
        best.detected ? h('span', { text: 'detected ' + best.detected }) : null,
        best.margin != null ? h('span', { text: 'margin ' + Number(best.margin).toFixed(2) }) : null,
        h('span', { class: 'spacer' }),
        best.take_mp3 ? h('button', { class: 'btn btn-mini', type: 'button', text: 'Download take', onclick: function () { downloadTake(best, o); } }) : null,
        best.text ? h('button', { class: 'btn btn-mini', type: 'button', text: 'Copy text', title: 'The exact text sent to WellSaid', onclick: function () { copyText(best.text, 'Rendered text'); } }) : null
      ]));
    }
    if (tries.length) {
      var det = h('details', { class: 'tries' }, [h('summary', { text: 'All ' + tries.length + ' tr' + (tries.length === 1 ? 'y' : 'ies') })]);
      tries.forEach(function (t, i) {
        det.appendChild(h('div', { class: 'try' }, [
          h('span', { class: 'try-n', text: String(i + 1) }),
          h('div', { class: 'try-text', html: diffText(o.sentence, t.text) }),
          h('div', { class: 'try-side' }, [
            h('span', { class: 'vmark ' + (t.verified ? 'ok' : 'bad'), text: t.verified ? '✓' : '✗' }),
            h('span', { class: 'ipa ' + (t.verified ? '' : 'heard'), text: '/' + (t.heard_as || '?') + '/' }),
            t.take_mp3 ? player(t.take_mp3, { label: 'take ' + (i + 1), kind: t.verified ? 'after' : '', bars: 18 }) : null
          ])
        ]));
      });
      row.appendChild(det);
    }
    return row;
  }
  // Show what changed in the rendered text vs the original sentence (word-level), marking the substituted span.
  function diffText(orig, text) {
    if (!text) return '';
    if (!orig) return esc(text);
    var a = orig.split(/\s+/), b = text.split(/\s+/);
    var pre = 0; while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
    var suf = 0; while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf++;
    var mid = b.slice(pre, b.length - suf);
    if (!mid.length) return esc(text);
    return esc(b.slice(0, pre).join(' ')) + (pre ? ' ' : '') + '<mark>' + esc(mid.join(' ')) + '</mark>' + (suf ? ' ' : '') + esc(b.slice(b.length - suf).join(' '));
  }
  function downloadTake(t, o) {
    fetchFile(t.take_mp3).then(function (b) { downloadBlob(b, (state.job.id + '_' + o.id + '_' + (o.word || 'take')).replace(/[^A-Za-z0-9_\-]+/g, '_') + '.' + (/\.wav$/i.test(t.take_mp3) ? 'wav' : 'mp3')); }, function (e) { reportError(e, 'Download'); });
  }

  /* ------------------------------ downloads ------------------------------ */

  function downloadReport(kind) {
    var job = state.job; if (!job) return;
    var base = '/v1/jobs/' + encodeURIComponent(job.id) + '/files/';
    var name = 'retake_' + job.id + '_report.' + kind;
    fetchFile(base + 'report.' + kind).then(function (b) { downloadBlob(b, name); toast('Downloaded ' + name, 'ok'); }, function (e) {
      if (kind === 'json') { downloadBlob(new Blob([JSON.stringify(job, null, 2)], { type: 'application/json' }), name); toast('Server report unavailable — saved the job payload instead', 'info'); }
      else if (kind === 'html') { downloadBlob(new Blob([buildReportHtml(job)], { type: 'text/html' }), name); toast('Server report unavailable — built one from this page instead', 'info'); }
      else reportError(e, 'Download');
    });
  }
  function buildReportHtml(job) {
    var res = job.result || {}, occs = occsOf(job), flagged = flaggedOf(job);
    var rows = occs.map(function (o) {
      var r = fixResultFor(o.id);
      return '<tr class="' + (o.flag ? 'flag' : 'ok') + '"><td>' + esc(fmtTime(o.t_start)) + '</td><td><b>' + esc(o.word) + '</b></td><td>' + esc(o.intended_desc) + '<br><code>' + esc(o.intended_respell || '') + '</code> /' + esc(o.intended_ipa || '') + '/</td><td>' + esc(o.detected_desc || 'undecidable') + '<br>heard /' + esc(o.free_decode || '') + '/</td><td>' + (o.flag ? 'FLAG' : 'ok') + (r ? (r.verified ? ' → fixed ✓' : ' → fix unverified ✗') : '') + '</td><td>' + esc(o.reason || '') + '</td></tr>';
    }).join('');
    return '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Retake report · ' + esc(job.id) + '</title><style>body{font-family:Assistant,"DM Sans",system-ui,-apple-system,"Segoe UI",sans-serif;background:#111111;color:#fcfcfc;padding:32px;max-width:1100px;margin:auto}h1{font-family:"DM Sans",system-ui,sans-serif;letter-spacing:normal}h1 span{color:#7fed9b}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #2c2c2c;padding:8px;vertical-align:top;text-align:left}tr.flag td:first-child{border-left:3px solid #f46c95}tr.ok td:first-child{border-left:3px solid #7fed9b}code{color:#c99fff}p.t{line-height:1.7;background:#1e1e1e;padding:16px;border-radius:4px}.m{color:#c7c7c7}</style></head><body><h1>WellSaid – <span>Retake</span></h1><p class="m">' + esc(basename(res.media) || job.id) + ' · ' + esc(fmtTime(res.duration)) + ' · ' + occs.length + ' heteronyms · ' + flagged.length + ' flagged · ' + esc(job.created || '') + '</p><p class="t">' + esc((res.transcript && res.transcript.text) || '') + '</p><table><thead><tr><th>t</th><th>word</th><th>you meant</th><th>it said</th><th>verdict</th><th>why</th></tr></thead><tbody>' + rows + '</tbody></table>' + (res.summary && res.summary.length ? '<h3>Summary</h3><ul>' + res.summary.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ul>' : '') + '</body></html>';
  }

  /* ------------------------------ API page ------------------------------ */

  function displayBase() { return apiBase() || window.location.origin; }
  function curlForJob(id) {
    return 'curl -s -H "X-Api-Key: $RETAKE_API_KEY" ' + displayBase() + '/v1/jobs/' + id;
  }
  function renderApiPage() {
    var base = displayBase();
    $('#api-base-label').textContent = base;
    $('#docs-link').setAttribute('href', base + '/docs');
    $('#redoc-link').setAttribute('href', base + '/redoc');
    paintKey();
    $('#btn-key-reveal').onclick = function () { state.keyRevealed = !state.keyRevealed; paintKey(); };
    $('#btn-key-edit').onclick = function () { openSettings(); $('#set-key').focus(); };
    var grid = $('#health-grid'); grid.innerHTML = '';
    grid.appendChild(h('div', { class: 'hcell' }, [h('div', { class: 'k', text: 'health' }), h('div', { class: 'v', html: '<span class="spin sm"></span> checking' })]));
    refreshHealth(true).then(function (hh) { paintHealth(hh); }, function (e) {
      grid.innerHTML = ''; grid.appendChild(h('div', { class: 'hcell' }, [h('div', { class: 'k', text: 'health' }), h('div', { class: 'v', html: '<span class="dot bad"></span> offline' }), h('div', { class: 'muted small', text: e.message })]));
    });
    var ex = $('#api-examples'); ex.innerHTML = '';
    examples().forEach(function (e) { ex.appendChild(exampleCard(e)); });
  }
  function paintKey() {
    var k = apiKey(), el = $('#key-masked');
    if (!k) { el.textContent = 'not set' + (state.health && state.health.auth === 'open' ? ' (API is open)' : ''); $('#btn-key-reveal').classList.add('hidden'); return; }
    $('#btn-key-reveal').classList.remove('hidden');
    $('#btn-key-reveal').textContent = state.keyRevealed ? 'Hide' : 'Reveal';
    el.textContent = state.keyRevealed ? k : (k.length > 6 ? k.slice(0, 3) + '•'.repeat(Math.min(12, k.length - 5)) + k.slice(-2) : '••••••');
  }
  function paintHealth(hh) {
    var grid = $('#health-grid'); grid.innerHTML = '';
    var cell = function (k, v, dotCls) { grid.appendChild(h('div', { class: 'hcell' }, [h('div', { class: 'k', text: k }), h('div', { class: 'v', html: (dotCls ? '<span class="dot ' + dotCls + '"></span>' : '') + esc(v) })])); };
    cell('status', hh.ok ? 'ok' : 'degraded', hh.ok ? 'ok' : 'warn');
    cell('version', hh.version || '–');
    cell('auth', hh.auth === 'open' ? 'open (dev mode)' : 'X-Api-Key', hh.auth === 'open' ? 'warn' : 'ok');
    var m = hh.models || {};
    cell('whisper', m.whisper || '–', m.whisper === 'loaded' ? 'ok' : 'warn');
    cell('phonemes', m.phonemes || '–', m.phonemes === 'loaded' ? 'ok' : 'warn');
    cell('wellsaid', hh.wellsaid ? 'configured' : 'no key', hh.wellsaid ? 'ok' : 'bad');
    cell('anthropic', hh.anthropic ? 'configured' : 'spaCy fallback', hh.anthropic ? 'ok' : 'warn');
  }
  function examples() {
    var base = displayBase();
    var jid = state.job ? state.job.id : 'JOB_ID';
    var sh = function (s) { return s; };
    return [
      { method: 'GET', path: '/v1/health', title: 'Health',
        curl: 'curl -s ' + base + '/v1/health',
        py: 'import requests\n\nBASE = "' + base + '"\nprint(requests.get(f"{BASE}/v1/health", timeout=10).json())' },
      { method: 'POST', path: '/v1/jobs', title: 'Check a recording (multipart)',
        curl: sh('curl -s -X POST ' + base + '/v1/jobs \\\n  -H "X-Api-Key: $RETAKE_API_KEY" \\\n  -F "media=@voiceover.mp4" \\\n  -F "script=@script.txt" \\\n  -F "engine=both"\n# → 202 {"job_id": "j_…", "status_url": "/v1/jobs/j_…"}'),
        py: 'import os, requests\n\nBASE = "' + base + '"\nH = {"X-Api-Key": os.environ["RETAKE_API_KEY"]}\n\nwith open("voiceover.mp4", "rb") as f:\n    r = requests.post(f"{BASE}/v1/jobs", headers=H,\n                      files={"media": ("voiceover.mp4", f)},\n                      data={"script": open("script.txt").read(), "engine": "both"},\n                      timeout=600)\nr.raise_for_status()\njob_id = r.json()["job_id"]' },
      { method: 'POST', path: '/v1/jobs', title: 'Render a script through WellSaid, then check it (JSON)',
        curl: 'curl -s -X POST ' + base + '/v1/jobs \\\n  -H "X-Api-Key: $RETAKE_API_KEY" -H "Content-Type: application/json" \\\n  -d \'{"text": "Nothing goes live until a reviewer reads the diff.", "speaker_id": 3, "model": "preview"}\'',
        py: 'r = requests.post(f"{BASE}/v1/jobs", headers=H, json={\n    "text": "Nothing goes live until a reviewer reads the diff.",\n    "speaker_id": 3,\n    "model": "preview",   # or "caruso"\n}, timeout=60)\njob_id = r.json()["job_id"]' },
      { method: 'GET', path: '/v1/jobs/{id}', title: 'Poll until done',
        curl: 'curl -s -H "X-Api-Key: $RETAKE_API_KEY" ' + base + '/v1/jobs/' + jid + '\n# status: queued | running | done | error · stage · progress · result.occurrences[]',
        py: 'import time\n\nwhile True:\n    job = requests.get(f"{BASE}/v1/jobs/{job_id}", headers=H, timeout=30).json()\n    print(job["status"], job.get("stage"), round(job.get("progress") or 0, 2))\n    if job["status"] in ("done", "error"):\n        break\n    time.sleep(1.5)\n\nif job["status"] == "error":\n    raise SystemExit(job["error"])\n\nres = job["result"]\nflagged = [o for o in res["occurrences"] if o["flag"]]\nfor o in flagged:\n    print(f\'{o["t_start"]:6.2f}s {o["word"]:14} meant {o["intended_respell"]:14} heard /{o["free_decode"]}/\')' },
      { method: 'POST', path: '/v1/jobs/{id}/fix', title: 'Fix the flagged lines',
        curl: 'curl -s -X POST ' + base + '/v1/jobs/' + jid + '/fix \\\n  -H "X-Api-Key: $RETAKE_API_KEY" -H "Content-Type: application/json" \\\n  -d \'{"speaker_id": 3, "model": "preview"}\'\n# → 202 {"fix_id": "f_…"}  · omit occurrence_ids to fix every flagged occurrence\n\ncurl -s -H "X-Api-Key: $RETAKE_API_KEY" ' + base + '/v1/jobs/' + jid + '/fixes/FIX_ID\n# → {status, results: [{occurrence_id, tries: [{text, take_mp3, heard_as, detected, margin, verified}], best, verified}], summary}',
        py: 'fix_id = requests.post(f"{BASE}/v1/jobs/{job_id}/fix", headers=H, json={\n    "occurrence_ids": [o["id"] for o in flagged],   # optional, defaults to all flagged\n    "speaker_id": 3,\n    "model": "preview",\n}, timeout=30).json()["fix_id"]\n\nwhile True:\n    fix = requests.get(f"{BASE}/v1/jobs/{job_id}/fixes/{fix_id}", headers=H, timeout=30).json()\n    if fix["status"] in ("done", "error"):\n        break\n    time.sleep(1.5)\n\nfor r in fix["results"]:\n    take = r["best"]\n    print(r["occurrence_id"], "verified" if r["verified"] else "NOT verified",\n          take and f"{BASE}{take[\'take_mp3\']}")' },
      { method: 'GET', path: '/v1/jobs/{id}/files/{name}', title: 'Download clips, takes and reports',
        curl: 'curl -s -H "X-Api-Key: $RETAKE_API_KEY" -o report.json ' + base + '/v1/jobs/' + jid + '/files/report.json\ncurl -s -H "X-Api-Key: $RETAKE_API_KEY" -o report.html ' + base + '/v1/jobs/' + jid + '/files/report.html',
        py: 'for name in ("report.json", "report.html"):\n    b = requests.get(f"{BASE}/v1/jobs/{job_id}/files/{name}", headers=H, timeout=60).content\n    open(name, "wb").write(b)' },
      { method: 'POST', path: '/v1/check-text', title: 'Script pre-flight (no audio)',
        curl: 'curl -s -X POST ' + base + '/v1/check-text \\\n  -H "X-Api-Key: $RETAKE_API_KEY" -H "Content-Type: application/json" \\\n  -d \'{"text": "Autopilot re-records the affected lines overnight."}\'',
        py: 'r = requests.post(f"{BASE}/v1/check-text", headers=H,\n                  json={"text": "Autopilot re-records the affected lines overnight."}, timeout=60)\nfor o in r.json()["occurrences"]:\n    print(o["word"], "→", o.get("intended"), o.get("intended_desc"))' },
      { method: 'DELETE', path: '/v1/jobs/{id}', title: 'Delete a job',
        curl: 'curl -s -X DELETE -H "X-Api-Key: $RETAKE_API_KEY" ' + base + '/v1/jobs/' + jid,
        py: 'requests.delete(f"{BASE}/v1/jobs/{job_id}", headers=H, timeout=30)' }
    ];
  }
  function exampleCard(e) {
    var pre = h('pre', { class: 'code' }); pre.innerHTML = colorize(e.curl);
    var which = 'curl';
    var tabs = h('div', { class: 'ex-tabs tabs', style: 'margin:0' }, ['curl', 'python'].map(function (t) {
      return h('button', { class: 'tab' + (t === 'curl' ? ' active' : ''), type: 'button', text: t, onclick: function (ev) {
        which = t; $all('.tab', tabs).forEach(function (x) { x.classList.toggle('active', x === ev.currentTarget); });
        pre.innerHTML = colorize(t === 'curl' ? e.curl : e.py);
      } });
    }));
    return h('div', { class: 'card ex' }, [
      h('div', { class: 'ex-head' }, [
        h('h3', { html: '<span class="method ' + e.method.toLowerCase() + '">' + e.method + '</span>' + esc(e.title) }),
        h('span', { class: 'path', text: e.path }),
        tabs
      ]),
      pre,
      h('div', { class: 'ex-foot' }, [h('button', { class: 'btn btn-mini', type: 'button', text: 'Copy', onclick: function () { copyText(which === 'curl' ? e.curl : e.py, 'Snippet'); } })])
    ]);
  }
  function colorize(code) {
    return esc(code).split('\n').map(function (line) {
      if (/^\s*#/.test(line)) return '<span class="c">' + line + '</span>';
      return line.replace(/(&quot;[^&]*?&quot;|&#39;[^&]*?&#39;)/g, '<span class="s">$1</span>').replace(/\$RETAKE_API_KEY/g, '<span class="k">$RETAKE_API_KEY</span>');
    }).join('\n');
  }

  /* ------------------------------ settings drawer ------------------------------ */

  function openSettings() {
    $('#set-base').value = lsGet(LS.base);
    $('#set-base').placeholder = window.RETAKE_API_BASE ? window.RETAKE_API_BASE : 'same origin (' + window.location.origin + ')';
    $('#set-key').value = apiKey();
    $('#set-result').textContent = ''; $('#set-result').className = 'set-result';
    $('#settings').classList.remove('hidden'); $('#settings').setAttribute('aria-hidden', 'false');
    $('#drawer-backdrop').classList.remove('hidden');
    setTimeout(function () { if (!$('#settings').classList.contains('hidden')) $('#set-base').focus(); }, 30);
  }
  function closeSettings() {
    var drawer = $('#settings');
    if (drawer.contains(document.activeElement)) { try { $('#btn-settings').focus(); } catch (e) { document.activeElement.blur(); } }
    drawer.classList.add('hidden'); drawer.setAttribute('aria-hidden', 'true');
    $('#drawer-backdrop').classList.add('hidden');
  }
  function initSettings() {
    $('#btn-settings').addEventListener('click', openSettings);
    $('#api-status').addEventListener('click', openSettings);
    $('#btn-settings-close').addEventListener('click', closeSettings);
    $('#drawer-backdrop').addEventListener('click', closeSettings);
    $('#settings-api-link').addEventListener('click', closeSettings);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !$('#settings').classList.contains('hidden')) closeSettings(); });
    $('#btn-set-save').addEventListener('click', function () {
      var base = $('#set-base').value.trim().replace(/\/+$/, '');
      if (base && !/^https?:\/\//.test(base)) { toast('API base must start with http:// or https://', 'error'); return; }
      lsSet(LS.base, base); lsSet(LS.key, $('#set-key').value.trim());
      state.blobCache = {};
      toast('Settings saved', 'ok');
      closeSettings();
      refreshHealth(true).catch(function () {}); loadVoices();
      if (state.route.name === 'api') renderApiPage();
      if (state.route.name === 'home') loadJobs();
    });
    $('#btn-set-test').addEventListener('click', function () {
      var out = $('#set-result'); out.className = 'set-result'; out.innerHTML = '<span class="spin sm"></span> testing…';
      var base = $('#set-base').value.trim().replace(/\/+$/, '');
      var key = $('#set-key').value.trim();
      if (MOCK) { out.className = 'set-result ok'; out.textContent = 'Mock mode: no network calls. The sample API reports ok · auth key · models loaded.'; return; }
      var hd = {}; if (key) hd['X-Api-Key'] = key;
      fetch((base || apiBase()) + '/v1/health', { headers: hd }).then(function (r) { return r.json().then(function (b) { return { ok: r.ok, status: r.status, body: b }; }); }).then(function (x) {
        if (!x.ok) { out.className = 'set-result bad'; out.textContent = 'HTTP ' + x.status + ': ' + ((x.body.error && x.body.error.message) || JSON.stringify(x.body)); return; }
        var b = x.body;
        out.className = 'set-result ok';
        out.textContent = 'ok · v' + (b.version || '?') + ' · auth ' + (b.auth || '?') + ' · whisper ' + ((b.models || {}).whisper || '?') + ' · phonemes ' + ((b.models || {}).phonemes || '?') + ' · wellsaid ' + (b.wellsaid ? 'yes' : 'no') + ' · anthropic ' + (b.anthropic ? 'yes' : 'no') + (b.auth === 'key' && !key ? ' — this API needs a key' : '');
      }, function (e) { out.className = 'set-result bad'; out.textContent = "Can't reach " + (base || displayBase()) + ' — ' + (e && e.message ? e.message : 'network error') + '. Is the API running and CORS allowed for this origin?'; });
    });
  }

  /* ------------------------------ boot ------------------------------ */

  function init() {
    if (MOCK) $('#mock-chip').classList.remove('hidden');
    initSettings();
    initHome();
    window.addEventListener('hashchange', route);
    var start = function () {
      refreshHealth(true).catch(function () {});
      loadVoices();
      if (MOCK && !window.location.hash) { window.location.hash = '#/jobs/' + state.mock.job.id; return; }
      route();
    };
    if (MOCK) loadMock().then(start, function (e) { toast('Could not load mock.json: ' + e.message, 'error'); route(); });
    else start();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

})();
