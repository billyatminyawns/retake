# -*- coding: utf-8 -*-
"""ffmpeg helpers: locate the binary, probe media, extract 16 kHz wav, cut
clips, concatenate takes and splice a replacement into a track.

ffmpeg resolution order (cached per process):
  1. env FFMPEG_BIN (explicit path)
  2. `ffmpeg` on PATH (Docker: apt-get install ffmpeg)
  3. `import imageio_ffmpeg` in this interpreter
  4. `/usr/bin/python3 -c "import imageio_ffmpeg; ..."` (this Mac)

Stdlib-only at import time; numpy/soundfile are imported lazily by load_wav.
"""
import os
import re
import shutil
import subprocess
import wave
from typing import Any, Dict, List, Optional

SYSTEM_PY = "/usr/bin/python3"
FFMPEG_TIMEOUT = 600

_FFMPEG_PATH = None  # type: Optional[str]


class MediaError(RuntimeError):
    """ffmpeg missing or an ffmpeg invocation failed."""


# ---------------------------------------------------------------------------
# binary
# ---------------------------------------------------------------------------

def ffmpeg_path():
    # type: () -> str
    """Absolute path to an ffmpeg binary (see module docstring). Cached."""
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    candidates = []  # type: List[Optional[str]]
    env_bin = os.environ.get("FFMPEG_BIN", "").strip()
    if env_bin:
        candidates.append(env_bin)
    candidates.append(shutil.which("ffmpeg"))
    for c in candidates:
        if c and os.path.exists(c):
            _FFMPEG_PATH = c
            return c
    try:
        import imageio_ffmpeg  # type: ignore
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            _FFMPEG_PATH = p
            return p
    except Exception:
        pass
    if os.path.exists(SYSTEM_PY):
        try:
            out = subprocess.check_output(
                [SYSTEM_PY, "-c",
                 "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                stderr=subprocess.DEVNULL, timeout=60)
            p = out.decode("utf-8", "replace").strip()
            if p and os.path.exists(p):
                _FFMPEG_PATH = p
                return p
        except Exception:
            pass
    raise MediaError(
        "ffmpeg not found: set FFMPEG_BIN, put ffmpeg on PATH, or install "
        "imageio-ffmpeg (pip install imageio-ffmpeg)")


def run_ffmpeg(args, what, timeout=FFMPEG_TIMEOUT):
    # type: (List[str], str, int) -> str
    """Run ffmpeg quietly (-y, errors only). Returns stderr text. Raises
    MediaError with a stderr tail on a non-zero exit."""
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y"] + list(args)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        raise MediaError("ffmpeg timed out (%s)" % what)
    err = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise MediaError("ffmpeg failed (%s): %s" % (what, err.strip()[-600:] or "no stderr"))
    return err


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?:\s*(Audio|Video|Subtitle|Data|Attachment):\s*([^,\s]+)")
_INPUT_RE = re.compile(r"Input #0,\s*([^,]+),")


def probe(path):
    # type: (str) -> Dict[str, Any]
    """Inspect a media file with `ffmpeg -i` (imageio-ffmpeg ships no ffprobe).

    Returns {"ok": bool, "duration": float|None, "has_audio": bool,
             "has_video": bool, "format": str|None, "audio_codec": str|None,
             "video_codec": str|None, "error": str|None}.
    `ok` is True only when ffmpeg recognised the container AND it has an
    audio stream — use it to reject non-media uploads.
    """
    info = {"ok": False, "duration": None, "has_audio": False, "has_video": False,
            "format": None, "audio_codec": None, "video_codec": None, "error": None}
    if not os.path.isfile(path):
        info["error"] = "file not found"
        return info
    cmd = [ffmpeg_path(), "-hide_banner", "-i", path, "-f", "null", "-t", "0", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=120)
    except subprocess.TimeoutExpired:
        info["error"] = "ffmpeg probe timed out"
        return info
    except MediaError as e:
        info["error"] = str(e)
        return info
    err = proc.stderr.decode("utf-8", "replace")
    m = _INPUT_RE.search(err)
    if m:
        info["format"] = m.group(1).strip()
    m = _DUR_RE.search(err)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration"] = h * 3600 + mi * 60 + s
    for m in _STREAM_RE.finditer(err):
        kind, codec = m.group(1), m.group(2)
        if kind == "Audio":
            info["has_audio"] = True
            info["audio_codec"] = info["audio_codec"] or codec
        elif kind == "Video":
            # mjpeg/png "video" streams are cover art, not a film
            if codec.lower() not in ("mjpeg", "png", "bmp", "gif"):
                info["has_video"] = True
            info["video_codec"] = info["video_codec"] or codec
    if info["format"] is None:
        tail = err.strip().splitlines()
        info["error"] = (tail[-1] if tail else "not a media file").strip()[:300]
        return info
    if not info["has_audio"]:
        info["error"] = "no audio stream"
        return info
    info["ok"] = True
    return info


# ---------------------------------------------------------------------------
# extract / cut
# ---------------------------------------------------------------------------

def extract_wav16k(media, dst):
    # type: (str, str) -> str
    """Any audio/video container -> 16 kHz mono pcm_s16le wav (idempotent)."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    if not os.path.isfile(media):
        raise MediaError("source does not exist: %s" % media)
    d = os.path.dirname(dst)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = dst + ".part.wav"
    run_ffmpeg(["-i", media, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp],
               "extract_wav16k %s" % os.path.basename(media))
    os.replace(tmp, dst)
    return dst


ensure_wav16k = extract_wav16k


def wav_duration(path):
    # type: (str) -> float
    """Duration in seconds of a wav file (stdlib wave, soundfile fallback)."""
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            if rate <= 0:
                raise wave.Error("bad frame rate")
            return wf.getnframes() / float(rate)
    except (wave.Error, EOFError):
        import soundfile as sf  # lazy
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)


def cut_clip(src_wav, start, end, pad, dst_wav):
    # type: (str, float, float, float, str) -> str
    """Cut [start-pad, end+pad] (clamped to [0, duration]) out of src_wav.

    -ss/-to go AFTER -i for sample-accurate cutting. Keeps pcm_s16le.
    """
    if not os.path.isfile(src_wav):
        raise MediaError("cut_clip: source does not exist: %s" % src_wav)
    dur = wav_duration(src_wav)
    if dur <= 0:
        raise MediaError("cut_clip: %s has zero duration" % src_wav)
    s = max(0.0, float(start) - float(pad))
    e = min(dur, float(end) + float(pad))
    if e - s < 0.01:  # degenerate window: widen around its centre
        mid = (s + e) / 2.0
        s = max(0.0, mid - 0.05)
        e = min(dur, mid + 0.05)
    if e <= s:
        raise MediaError("cut_clip: empty window [%.3f, %.3f] within %.3fs file %s"
                         % (s, e, dur, src_wav))
    d = os.path.dirname(dst_wav)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = dst_wav + ".part.wav"
    run_ffmpeg(["-i", src_wav, "-ss", "%.6f" % s, "-to", "%.6f" % e, "-c:a", "pcm_s16le", tmp],
               "cut_clip %s" % os.path.basename(dst_wav))
    os.replace(tmp, dst_wav)
    return dst_wav


def load_wav(path):
    """Load an audio file as a mono float32 numpy array (soundfile)."""
    import numpy as np  # lazy
    import soundfile as sf  # lazy
    data, _sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32)


def write_bytes(path, data):
    # type: (str, bytes) -> str
    """Write bytes atomically (tmp + rename), creating parent dirs."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# stitch
# ---------------------------------------------------------------------------

def concat_audio(parts, dst, sample_rate=None):
    # type: (List[str], str, Optional[int]) -> str
    """Concatenate audio files (any formats) into dst, re-encoded by extension
    (.wav -> pcm_s16le, .mp3 -> libmp3lame)."""
    parts = [p for p in parts if p]
    if not parts:
        raise MediaError("concat_audio: no input files")
    for p in parts:
        if not os.path.isfile(p):
            raise MediaError("concat_audio: missing input %s" % p)
    args = []  # type: List[str]
    for p in parts:
        args += ["-i", p]
    chains = "".join("[%d:a]" % i for i in range(len(parts)))
    filt = "%sconcat=n=%d:v=0:a=1[out]" % (chains, len(parts))
    args += ["-filter_complex", filt, "-map", "[out]"]
    if sample_rate:
        args += ["-ar", str(int(sample_rate))]
    args += _codec_for(dst) + [dst + ".part" + os.path.splitext(dst)[1]]
    d = os.path.dirname(dst)
    if d:
        os.makedirs(d, exist_ok=True)
    run_ffmpeg(args, "concat_audio %s" % os.path.basename(dst))
    os.replace(dst + ".part" + os.path.splitext(dst)[1], dst)
    return dst


def replace_segment(src, t_start, t_end, replacement, dst):
    # type: (str, float, float, str, str) -> str
    """Splice `replacement` audio over [t_start, t_end] of `src` (audio only):
    head of src + replacement + tail of src -> dst. Lengths may differ, so the
    result is as long as head + replacement + tail."""
    for p in (src, replacement):
        if not os.path.isfile(p):
            raise MediaError("replace_segment: missing input %s" % p)
    t_start = max(0.0, float(t_start))
    t_end = max(t_start, float(t_end))
    filt = ("[0:a]atrim=0:%.6f,asetpts=PTS-STARTPTS[head];"
            "[0:a]atrim=start=%.6f,asetpts=PTS-STARTPTS[tail];"
            "[1:a]aformat=sample_rates=16000:channel_layouts=mono,asetpts=PTS-STARTPTS[mid];"
            "[head]aformat=sample_rates=16000:channel_layouts=mono[h];"
            "[tail]aformat=sample_rates=16000:channel_layouts=mono[t];"
            "[h][mid][t]concat=n=3:v=0:a=1[out]" % (t_start, t_end))
    ext = os.path.splitext(dst)[1]
    tmp = dst + ".part" + ext
    d = os.path.dirname(dst)
    if d:
        os.makedirs(d, exist_ok=True)
    run_ffmpeg(["-i", src, "-i", replacement, "-filter_complex", filt, "-map", "[out]"]
               + _codec_for(dst) + [tmp], "replace_segment %s" % os.path.basename(dst))
    os.replace(tmp, dst)
    return dst


def _codec_for(path):
    # type: (str) -> List[str]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2"]
    if ext in (".m4a", ".aac"):
        return ["-c:a", "aac"]
    return ["-c:a", "pcm_s16le"]
