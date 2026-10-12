"""Static report writers: <job>/report.json and <job>/report.html.

report.json is the rewritten result (URLs relative to the job's files
endpoint, so it stays valid wherever the API is hosted). report.html is a
self-contained page in the WellSaid Studio look (see ui/brand/skin.css and
SPEC.md "BRANDING OVERRIDE"): dark Studio theme by default, Studio light when
the viewer prefers it, mint for clean/verified, red-300/400 for flagged,
sunflower for warnings, inline CSS + inline WellSaid mark, no font files
and no external requests — handy to attach to a ticket or share with a
voice vendor. Audio players point at the same relative file URLs as
report.json.
"""
import html
import json
import os
import re
from typing import Any, Dict, List

from . import settings


def _rel(val, job_dir):
    """Absolute path under the job dir -> path relative to the files endpoint."""
    if isinstance(val, dict):
        return {k: _rel(v, job_dir) for k, v in val.items()}
    if isinstance(val, list):
        return [_rel(v, job_dir) for v in val]
    if isinstance(val, str) and val.startswith("/"):
        ap = os.path.abspath(val)
        if ap.startswith(job_dir + os.sep):
            return os.path.relpath(ap, job_dir).replace(os.sep, "/")
    return val


def _fmt_t(t):
    try:
        t = float(t)
    except (TypeError, ValueError):
        return "–"
    m, s = divmod(t, 60)
    return "%d:%05.2f" % (int(m), s)


def _audio(src):
    if not src:
        return ""
    return '<audio controls preload="none" src="%s"></audio>' % html.escape(src, quote=True)


def write_reports(job, result, job_dir):
    # type: (Dict[str, Any], Dict[str, Any], str) -> None
    job_dir = os.path.abspath(job_dir)
    rel = _rel(result, job_dir)
    payload = {"retake": settings.VERSION, "job_id": job["id"], "created": job["created"],
               "mode": job["mode"], "filename": job.get("filename"), "params": {
                   k: v for k, v in (job.get("params") or {}).items() if k not in ("text", "script")},
               "result": rel}
    with open(os.path.join(job_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(job_dir, "report.html"), "w", encoding="utf-8") as fh:
        fh.write(render_html(job, rel))


# ── WellSaid Studio tokens (ui/brand/skin.css + colors.css), inlined ─────────
# Dark = Studio default: cards #1e1e1e on #111, text #fcfcfc, mint-300 accent.
_DARK_TOKENS = (
    "color-scheme:dark;"
    "--page:#111111;--bar:#111111;--card:#1e1e1e;--raised:#232323;--hover:#2c2c2c;"
    "--line:#2c2c2c;--line-2:#333333;--line-input:#646464;"
    "--text:#fcfcfc;--text-2:#c7c7c7;--text-3:#999999;"
    "--accent:#7fed9b;--accent-text:#7fed9b;--on-accent:#1e1e1e;--focus:#ceffdc;--link:#7ab6ff;"
    "--chip:#1e1e1e;--tag-bg:#4a3310;--tag-fg:#ffbf5f;"
    "--flag:#f46c95;--flag-soft:rgba(244,108,149,.14);--flag-line:rgba(244,108,149,.5);"
    "--clean:#7fed9b;--clean-soft:rgba(127,237,155,.12);--clean-line:rgba(127,237,155,.5);"
    "--warn:#ffbf5f;--warn-soft:rgba(255,191,95,.14);--warn-line:rgba(255,191,95,.5);"
    "--lav:#c99fff;--sel:rgba(127,237,155,.28)"
)
# Light = Studio light on sand neutrals: gray-1000 text, mint-500 accent, red-400 flags.
_LIGHT_TOKENS = (
    "color-scheme:light;"
    "--page:#fbfaf9;--bar:#f4f3f1;--card:#ffffff;--raised:#fbfaf9;--hover:#f2f0ec;"
    "--line:#e9e4dc;--line-2:#dad6cf;--line-input:#999999;"
    "--text:#1e1e1e;--text-2:#444444;--text-3:#646464;"
    "--accent:#3c8455;--accent-text:#3c8455;--on-accent:#fcfcfc;--focus:#2c2c2c;--link:#264974;"
    "--chip:#f0f0f0;--tag-bg:#fbe0b7;--tag-fg:#7d561b;"
    "--flag:#e13465;--flag-soft:rgba(225,52,101,.09);--flag-line:rgba(225,52,101,.4);"
    "--clean:#3c8455;--clean-soft:rgba(97,207,124,.14);--clean-line:rgba(97,207,124,.55);"
    "--warn:#cd871d;--warn-soft:rgba(243,167,53,.14);--warn-line:rgba(243,167,53,.5);"
    "--lav:#63477e;--sel:rgba(97,207,124,.3)"
)


def _light_rules(selector):
    return ("%s{%s}%s .i-sun{display:block}%s .i-moon{display:none}"
            % (selector, _LIGHT_TOKENS, selector, selector))


_CSS = (
    ":root{" + _DARK_TOKENS + ";"
    '--font:"DM Sans","Assistant",-apple-system,sans-serif;'
    '--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;--r:4px;--r-lg:8px}'
    + "@media (prefers-color-scheme:light){" + _light_rules(":root:not([data-theme=dark])") + "}"
    + _light_rules(":root[data-theme=light]")
    + """
*{box-sizing:border-box}
html,body{margin:0;background:var(--page);color:var(--text)}
body{font:400 15px/1.55 var(--font);-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
::selection{background:var(--sel)}
a{color:var(--link)}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.i-sun{display:none}
/* Studio top bar */
.bar{position:sticky;top:0;z-index:5;height:44px;background:var(--bar);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px;padding:0 16px}
.bar .mark{display:flex;align-items:center;flex:0 0 auto}.bar .mark svg{width:24px;height:24px;display:block}
.bar .mod{display:flex;align-items:center;gap:8px;min-width:0;white-space:nowrap;overflow:hidden;font:400 14px/21px var(--font);color:var(--text-2)}
.bar .mod b{font-weight:700;color:var(--text)}.bar .mod .sep{color:var(--text-3)}
.bar .right{margin-left:auto;display:flex;align-items:center;gap:8px}
.bar .tag{font:600 12px/16px var(--font);color:var(--text-2);background:var(--chip);border:1px solid var(--line-2);border-radius:var(--r);padding:2px 8px;white-space:nowrap}
.ibtn{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:var(--r);background:var(--raised);border:1px solid var(--line-2);color:var(--text);cursor:pointer;padding:0;font:inherit}
.ibtn:hover{background:var(--hover)}.ibtn svg{width:14px;height:14px;display:block}
/* page */
.wrap{max-width:960px;margin:0 auto;padding:28px 16px 56px}
.eyebrow{font:400 14px/18px var(--font);color:var(--text-2);margin-bottom:6px}
h1{font:700 24px/31px var(--font);margin:0 0 6px;letter-spacing:normal;word-break:break-word}
h2{font:700 16px/22px var(--font);margin:28px 0 12px}
h2 small{font-weight:400;color:var(--text-3);margin-left:8px}
.meta{font-size:12px;color:var(--text-3);display:flex;gap:6px 14px;flex-wrap:wrap}
.meta code{font-family:var(--mono);font-size:11px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);padding:16px}
.head{padding:20px 24px}
.head-top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px 24px;flex-wrap:wrap}
.verdict{display:inline-flex;align-items:center;gap:10px;border-radius:var(--r);padding:8px 14px;font:700 14px/20px var(--font);border:1px solid}
.verdict.clean{color:var(--clean);background:var(--clean-soft);border-color:var(--clean-line)}
.verdict.flag{color:var(--flag);background:var(--flag-soft);border-color:var(--flag-line)}
.verdict svg{width:18px;height:18px;flex:0 0 auto}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}
.stat{flex:1 1 120px;min-width:120px;background:var(--raised);border:1px solid var(--line);border-radius:var(--r);padding:10px 14px}
.stat b{display:block;font:700 24px/30px var(--font);letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.stat i{font-style:normal;font-size:12px;color:var(--text-2)}
.stat.flag b{color:var(--flag)}.stat.clean b{color:var(--clean)}
.k{font:600 12px/16px var(--font);color:var(--text-2);margin-bottom:4px}
.src{margin-top:12px}
/* transcript */
.tx{line-height:1.9;font-size:15px}
.tx a{text-decoration:none;color:inherit}
mark{background:var(--clean-soft);color:inherit;padding:0 3px;border-radius:2px;border-bottom:2px solid var(--clean)}
mark.flag{background:var(--flag-soft);border-bottom-color:var(--flag)}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text-3);margin-top:10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:var(--clean)}
.legend i.flag{background:var(--flag)}
/* occurrences */
.occ{margin:12px 0;border-left:3px solid var(--clean)}
.occ.flag{border-left-color:var(--flag)}
.occ-head{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}
.word{font:700 18px/24px var(--font);margin-right:4px}
.time{font-family:var(--mono);font-size:11px;color:var(--text-2);background:var(--chip);border:1px solid var(--line);border-radius:var(--r);padding:2px 7px;font-variant-numeric:tabular-nums}
.chip{font:600 12px/16px var(--font);border-radius:var(--r);padding:2px 8px;border:1px solid transparent;white-space:nowrap}
.chip.flag{color:var(--flag);background:var(--flag-soft)}
.chip.clean{color:var(--clean);background:var(--clean-soft)}
.chip.warn{color:var(--warn);background:var(--warn-soft)}
.chip.conf-high{color:var(--clean);border-color:var(--clean-line)}
.chip.conf-med{color:var(--warn);border-color:var(--warn-line)}
.chip.conf-low{color:var(--text-2);border-color:var(--line-2)}
.chip.conf{color:var(--text-2);border-color:var(--line-2)}
.sent{margin:10px 0 0;color:var(--text-2)}
.sent em{font-style:normal;font-weight:700;color:var(--text)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:640px){.grid{grid-template-columns:1fr}}
.pane{background:var(--raised);border:1px solid var(--line);border-radius:var(--r);padding:10px 12px}
.pane .k{color:var(--clean)}.pane.said .k{color:var(--text-2)}.occ.flag .pane.said .k{color:var(--flag)}
.desc{color:var(--text)}
.pron{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin-top:6px}
.rs{font:700 14px/20px var(--font);background:var(--chip);border:1px solid var(--line-2);border-radius:var(--r);padding:1px 8px}
.occ.flag .pane.said .rs{border-color:var(--flag-line)}
.ipa{font-family:var(--mono);font-size:13px;color:var(--lav)}
.pos{font-size:12px;color:var(--text-3)}
audio{width:100%;height:36px;display:block;margin-top:4px}
.reason{margin:12px 0 0;color:var(--text-2);font-size:13px}
.note{margin:4px 0 0;color:var(--text-3);font-size:12px}
.note code{font-family:var(--mono);font-size:12px}
.summary{margin:0;padding-left:20px;color:var(--text-2)}.summary li{margin:4px 0}
footer{color:var(--text-3);font-size:12px;margin-top:36px;display:flex;gap:8px 16px;flex-wrap:wrap;align-items:center}
footer svg{width:14px;height:14px;vertical-align:-2px;margin-right:4px}
@media print{""" + _light_rules(":root") + """
.bar{position:static}.ibtn{display:none}audio{display:none}.occ{break-inside:avoid}}
"""
)

# WellSaid mark (ui/brand/wellsaid-mark.svg), inlined so the report stays self-contained.
_MARK_PATH = (
    "M21.0572 7.95701C19.8484 6.72301 17.7368 6.72301 16.5281 7.95701L15.3973 9.11299L14.2645 7.95701C13.9703 "
    "7.65349 13.618 7.41232 13.2286 7.24791C12.8392 7.08349 12.4207 6.99918 11.998 7.00001C11.5757 6.99956 11.1576 "
    "7.08404 10.7686 7.24844C10.3795 7.41285 10.0275 7.65382 9.73347 7.95701L8.59871 9.11299L7.46594 7.95701C7.17164 "
    "7.65358 6.81932 7.41246 6.42993 7.24805C6.04054 7.08364 5.62206 6.99929 5.1994 7.00001C4.77708 6.99956 4.35897 "
    "7.08404 3.96995 7.24844C3.58093 7.41285 3.22892 7.65382 2.93486 7.95701C2.33103 8.57408 1.99509 9.40457 2.00005 "
    "10.268C1.99616 11.132 2.33357 11.9626 2.93886 12.579L6.33817 16.044C6.63232 16.3459 6.98381 16.586 7.372 "
    "16.7501C7.76018 16.9143 8.17725 16.9993 8.59871 17C9.02085 16.9999 9.43871 16.9153 9.82764 16.7511C10.2166 "
    "16.5869 10.5687 16.3465 10.8632 16.044L11.998 14.889L13.1308 16.044C13.4256 16.3467 13.7781 16.5872 14.1673 "
    "16.7514C14.5566 16.9156 14.9748 17.0001 15.3973 17C15.8194 16.9998 16.2373 16.9151 16.6262 16.7509C17.0151 "
    "16.5867 17.3672 16.3464 17.6619 16.044L21.0612 12.579C21.6665 11.9626 22.0039 11.132 22 10.268C22.0028 9.40343 "
    "21.6639 8.57279 21.0572 7.95701ZM3.59973 10.268C3.5974 9.83617 3.76597 9.42095 4.06863 9.11299C4.21546 8.96151 "
    "4.3912 8.84107 4.58544 8.75879C4.77968 8.67651 4.98846 8.63409 5.1994 8.63401C5.41054 8.63372 5.61958 8.67598 "
    "5.81403 8.75828C6.00848 8.84058 6.18436 8.96121 6.33117 9.11299L7.46494 10.268L6.33117 11.423C6.10707 11.6422 "
    "5.82417 11.7917 5.51679 11.8533C5.20941 11.9149 4.89077 11.886 4.59952 11.77C4.57653 11.761 4.55553 11.746 "
    "4.53354 11.736C4.46053 11.7047 4.39001 11.668 4.32258 11.626C4.28604 11.5996 4.25099 11.5712 4.2176 "
    "11.541C4.1677 11.5045 4.11997 11.4651 4.07463 11.423C3.76995 11.1159 3.59921 10.7006 3.59973 10.268ZM9.73147 "
    "14.889C9.58411 15.0402 9.40796 15.1605 9.21341 15.2425C9.01886 15.3246 8.80985 15.3669 8.59871 15.3669C8.38756 "
    "15.3669 8.17855 15.3246 7.984 15.2425C7.78945 15.1605 7.6133 15.0402 7.46594 14.889L6.03123 13.425C6.16769 "
    "13.3891 6.30135 13.3433 6.43115 13.288C6.60126 13.2138 6.76452 13.1248 6.91905 13.022C6.95804 12.997 6.99603 "
    "12.971 7.03403 12.943C7.18931 12.8346 7.33484 12.7128 7.46894 12.579L9.73447 10.268L10.8662 9.11299C11.1685 "
    "8.81605 11.5753 8.64969 11.999 8.64969C12.4227 8.64969 12.8295 8.81605 13.1318 9.11299L14.2645 10.268L10.8652 "
    "13.734L9.73147 14.889ZM16.5301 14.889C16.3827 15.0402 16.2066 15.1605 16.012 15.2425C15.8175 15.3246 15.6085 "
    "15.3669 15.3973 15.3669C15.1862 15.3669 14.9772 15.3246 14.7826 15.2425C14.5881 15.1605 14.4119 15.0402 "
    "14.2645 14.889L13.1328 13.734L14.6425 12.193L15.3973 11.423L16.5301 12.579C16.6642 12.7128 16.8097 12.8346 "
    "16.965 12.943C17.003 12.971 17.041 12.997 17.08 13.022C17.2345 13.1248 17.3978 13.2138 17.5679 13.288C17.6977 "
    "13.3433 17.8313 13.3891 17.9678 13.425L16.5301 14.889ZM17.6629 11.423L16.5301 10.268L17.6629 9.11299C17.9649 "
    "8.81588 18.3715 8.64937 18.7951 8.64937C19.2187 8.64937 19.6254 8.81588 19.9274 9.11299C20.2301 9.42095 "
    "20.3986 9.83617 20.3963 10.268C20.3984 10.6998 20.2299 11.1149 19.9274 11.423C19.625 11.7202 19.2181 11.8868 "
    "18.7941 11.8868C18.3702 11.8868 17.9632 11.7202 17.6609 11.423H17.6629Z"
)
_MARK_SVG = ('<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
             '<path d="%s" fill="#6DED97"/></svg>' % _MARK_PATH)

_ICON_OK = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>')
_ICON_FLAG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
              'stroke-linejoin="round" aria-hidden="true"><path d="M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 '
              '0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>')
_ICON_MOON = ('<svg class="i-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 '
              '7 7 0 0 0 21 12.79Z"/></svg>')
_ICON_SUN = ('<svg class="i-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
             '<path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 '
             '1.41M19.07 4.93l-1.41 1.41"/></svg>')

# Theme toggle: Studio's dark by default, the OS preference otherwise, a click
# flips it and remembers the choice under the same key the Retake UI uses.
_JS = ("(function(){var d=document.documentElement,b=document.getElementById('theme');"
       "try{var s=localStorage.getItem('ws-theme');if(s==='light'||s==='dark')d.setAttribute('data-theme',s)}catch(e){}"
       "if(!b)return;b.addEventListener('click',function(){var t=d.getAttribute('data-theme');"
       "if(!t)t=(window.matchMedia&&window.matchMedia('(prefers-color-scheme: light)').matches)?'light':'dark';"
       "var n=t==='light'?'dark':'light';d.setAttribute('data-theme',n);"
       "try{localStorage.setItem('ws-theme',n)}catch(e){}})})();")


def _esc(val):
    return html.escape("" if val is None else str(val))


def _slug(val):
    return re.sub(r"[^a-z0-9_-]+", "-", str(val or "").lower()).strip("-")


def _highlight_transcript(text, occs):
    # type: (str, List[Dict[str, Any]]) -> str
    """Mark every occurrence in the transcript (by sentence + char offsets when
    available, else by word search), flagged=red, clean=mint; each mark links
    to its occurrence card."""
    if not text:
        return ""
    spans = []
    cursor = 0
    low = text.lower()
    for o in occs:
        w = (o.get("word") or "").lower()
        if not w:
            continue
        i = low.find(w, cursor)
        if i < 0:
            i = low.find(w)
            if i < 0:
                continue
        spans.append((i, i + len(w), bool(o.get("flag")), o))
        cursor = i + len(w)
    spans.sort(key=lambda s: (s[0], s[1]))
    out, pos = [], 0
    for a, b, flag, o in spans:
        if a < pos:
            continue
        out.append(html.escape(text[pos:a]))
        if flag:
            tip = "Flagged: meant %s, said %s" % (o.get("intended_respell") or o.get("intended") or "?",
                                                 o.get("detected_respell") or o.get("detected") or "?")
        else:
            tip = "Clean: %s" % (o.get("intended_respell") or o.get("intended") or "as intended")
        mark = '<mark class="%s" title="%s">%s</mark>' % ("flag" if flag else "", html.escape(tip, quote=True),
                                                         html.escape(text[a:b]))
        if o.get("id"):
            mark = '<a href="#occ-%s">%s</a>' % (html.escape(_slug(o["id"]), quote=True), mark)
        out.append(mark)
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _conf_chip(conf):
    if not conf:
        return ""
    s = _slug(conf)
    cls = "conf-" + s if s in ("high", "med", "medium", "low") else "conf"
    if cls == "conf-medium":
        cls = "conf-med"
    return '<span class="chip %s">%s confidence</span>' % (cls, _esc(str(conf).capitalize()))


def _sentence(o):
    """The sentence with the heteronym emphasised (char offsets when they line up)."""
    sent = str(o.get("sentence") or "")
    word = str(o.get("word") or "")
    a, b = o.get("char_start"), o.get("char_end")
    if isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(sent) and sent[a:b].lower() == word.lower():
        return "%s<em>%s</em>%s" % (_esc(sent[:a]), _esc(sent[a:b]), _esc(sent[b:]))
    if word:
        i = sent.lower().find(word.lower())
        if i >= 0:
            return "%s<em>%s</em>%s" % (_esc(sent[:i]), _esc(sent[i:i + len(word)]), _esc(sent[i + len(word):]))
    return _esc(sent)


def _pane(cls, label, desc, respell, ipa, pos):
    bits = ['<div class="pane %s"><div class="k">%s</div><div class="desc">%s</div>' % (cls, label, _esc(desc))]
    pron = []
    if respell:
        pron.append('<b class="rs">%s</b>' % _esc(respell))
    if ipa:
        pron.append('<code class="ipa">/%s/</code>' % _esc(ipa))
    if pos:
        pron.append('<span class="pos">%s</span>' % _esc(pos))
    if pron:
        bits.append('<div class="pron">%s</div>' % "".join(pron))
    bits.append("</div>")
    return "".join(bits)


def _occurrence(o):
    flag = bool(o.get("flag"))
    oid = _slug(o.get("id") or "")
    head = ['<span class="word">%s</span>' % _esc(o.get("word") or ""),
            '<span class="time">%s – %s</span>' % (_fmt_t(o.get("t_start")), _fmt_t(o.get("t_end"))),
            '<span class="chip %s">%s</span>' % ("flag" if flag else "clean", "Flagged" if flag else "Clean")]
    if o.get("undecidable"):
        head.append('<span class="chip warn">Undecided</span>')
    head.append(_conf_chip(o.get("confidence")))
    parts = ['<article class="card occ %s"%s><div class="occ-head">%s</div>'
             % ("flag" if flag else "", (' id="occ-%s"' % html.escape(oid, quote=True)) if oid else "", "".join(head))]
    if o.get("sentence"):
        parts.append('<p class="sent">%s</p>' % _sentence(o))
    parts.append('<div class="grid">%s%s</div>' % (
        _pane("meant", "You meant", o.get("intended_desc") or o.get("intended") or "–",
              o.get("intended_respell"), o.get("intended_ipa"), o.get("intended_pos")),
        _pane("said", "It said", o.get("detected_desc") or o.get("detected") or "undecided",
              o.get("detected_respell"), o.get("detected_ipa"), o.get("detected_pos"))))
    clips = []
    if o.get("clip_wav"):
        clips.append('<div><div class="k">Word clip</div>%s</div>' % _audio(o.get("clip_wav")))
    if o.get("sentence_wav"):
        clips.append('<div><div class="k">Sentence clip</div>%s</div>' % _audio(o.get("sentence_wav")))
    if clips:
        parts.append('<div class="grid">%s</div>' % "".join(clips))
    if o.get("reason") and str(o.get("reason")) != "ok":
        parts.append('<p class="reason">%s</p>' % _esc(o["reason"]))
    notes = []
    if o.get("confidence_reason"):
        notes.append("Confidence: %s" % _esc(o["confidence_reason"]))
    if o.get("free_decode"):
        notes.append("Free decode: <code>/%s/</code>" % _esc(o["free_decode"]))
    if o.get("margin") is not None:
        try:
            notes.append("CTC margin %.2f" % float(o["margin"]))
        except (TypeError, ValueError):
            pass
    if notes:
        parts.append('<p class="note">%s</p>' % " · ".join(notes))
    parts.append("</article>")
    return "".join(parts)


def render_html(job, rel):
    # type: (Dict[str, Any], Dict[str, Any]) -> str
    occs = [o for o in rel.get("occurrences") or [] if isinstance(o, dict)]
    flags = sum(1 for o in occs if o.get("flag"))
    clean = len(occs) - flags
    name = job.get("filename") or ("rendered script" if job.get("mode") == "script" else "media")
    dur = rel.get("duration")
    if occs and flags:
        verdict = ('<div class="verdict flag">%s %d of %d heteronym%s need%s a retake</div>'
                   % (_ICON_FLAG, flags, len(occs), "" if len(occs) == 1 else "s", "s" if flags == 1 else ""))
        eyebrow = "Pronunciation QA · retake needed"
    elif occs:
        verdict = '<div class="verdict clean">%s Every heteronym was pronounced as intended</div>' % _ICON_OK
        eyebrow = "Pronunciation QA · verified"
    else:
        verdict = '<div class="verdict clean">%s No heteronyms found — nothing to check</div>' % _ICON_OK
        eyebrow = "Pronunciation QA · nothing to check"
    meta = ['<span>job <code>%s</code></span>' % _esc(job.get("id") or "–"),
            "<span>%s</span>" % _esc(job.get("created") or "–"),
            "<span>%s</span>" % _esc(job.get("mode") or "")]
    if rel.get("engine"):
        meta.append("<span>engine %s</span>" % _esc(rel.get("engine")))
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<meta name="color-scheme" content="dark light"><meta name="theme-color" content="#111111">',
             "<title>WellSaid – Retake report</title><style>%s</style></head><body>" % _CSS,
             '<header class="bar"><span class="mark">%s</span>' % _MARK_SVG,
             '<div class="mod"><b>Retake</b><span class="sep">·</span><span>Pronunciation QA</span></div>',
             '<div class="right"><span class="tag">Report</span>',
             '<button type="button" class="ibtn" id="theme" aria-label="Switch theme" title="Switch theme">%s%s</button>'
             % (_ICON_MOON, _ICON_SUN),
             '</div></header><main class="wrap">',
             '<section class="card head"><div class="head-top"><div>',
             '<div class="eyebrow">%s</div><h1>%s</h1><div class="meta">%s</div></div>%s</div>'
             % (_esc(eyebrow), _esc(name), "".join(meta), verdict),
             '<div class="stats"><div class="stat"><b>%s</b><i>Duration</i></div>' % (
                 _fmt_t(dur) if dur is not None else "–"),
             '<div class="stat"><b>%d</b><i>Heteronyms</i></div>' % len(occs),
             '<div class="stat%s"><b>%d</b><i>Flagged</i></div>' % (" flag" if flags else "", flags),
             '<div class="stat%s"><b>%d</b><i>Clean</i></div></div>' % (" clean" if clean else "", clean)]
    media = rel.get("media")
    if isinstance(media, str) and not media.startswith("/"):
        parts.append('<div class="src"><div class="k">Source audio</div>%s</div>' % _audio(media))
    parts.append("</section>")
    tx = (rel.get("transcript") or {}).get("text") or ""
    if tx:
        parts.append('<h2>Transcript</h2><section class="card"><div class="tx">%s</div>'
                     '<div class="legend"><span><i></i>pronounced as intended</span>'
                     '<span><i class="flag"></i>flagged for a retake</span></div></section>'
                     % _highlight_transcript(tx, occs))
    if occs:
        parts.append("<h2>Occurrences<small>%d heteronym%s · %d flagged</small></h2>"
                     % (len(occs), "" if len(occs) == 1 else "s", flags))
        parts.extend(_occurrence(o) for o in occs)
    summary = rel.get("summary") or []
    if summary:
        parts.append('<h2>Summary</h2><section class="card"><ul class="summary">%s</ul></section>'
                     % "".join("<li>%s</li>" % _esc(s) for s in summary))
    parts.append('<footer><span>%sWellSaid Labs</span><span>Generated by Retake %s</span>'
                 '<span>intent engine %s</span></footer></main><script>%s</script></body></html>'
                 % (_MARK_SVG, _esc(settings.VERSION),
                    _esc((occs[0].get("intent_engine") if occs else None) or "–"), _JS))
    return "".join(parts)
