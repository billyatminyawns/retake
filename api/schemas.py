"""Pydantic models — the public API contract (SPEC "HTTP API").

Occurrence / fix payloads mirror the `core` dicts, with every on-disk path
rewritten to a URL under /v1/jobs/{id}/files/... by the jobs router. Models
allow extra keys so new core fields flow through without an API change.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# error envelope
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["not_found"])
    message: str = Field(..., examples=["Job 7f3a… does not exist."])
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Every non-2xx response has this shape."""
    error: ErrorDetail


# ---------------------------------------------------------------------------
# health / voices / lexicon
# ---------------------------------------------------------------------------

class ModelsState(BaseModel):
    whisper: str = Field("cold", description='"loaded" or "cold"', examples=["cold"])
    phonemes: str = Field("cold", description='"loaded" or "cold"', examples=["loaded"])


class HealthResponse(BaseModel):
    ok: bool = True
    version: str = Field(..., examples=["1.0.0"])
    auth: str = Field(..., description='"open" (dev, no key required) or "key"', examples=["key"])
    models: ModelsState
    wellsaid: bool = Field(..., description="A WellSaid API key is configured")
    anthropic: bool = Field(..., description="ANTHROPIC_API_KEY is configured (Claude intent engine)")
    core: bool = Field(True, description="The analysis engine package imported cleanly")
    queue: int = Field(0, description="Jobs waiting for the worker")
    uptime_s: float = Field(0.0, description="Seconds since the API started")


class Voice(BaseModel):
    model_config = ConfigDict(extra="allow")
    speaker_id: int = Field(..., examples=[26])
    name: Optional[str] = Field(None, examples=["Wade C."])
    style: Optional[str] = Field(None, examples=["Narration"])
    accent: Optional[str] = Field(None, examples=["United States"])
    gender: Optional[str] = Field(None, examples=["Male"])


class LexiconSense(BaseModel):
    id: str = Field(..., examples=["record_n"])
    pos: Optional[str] = Field(None, examples=["NOUN"])
    desc: str = Field("", examples=["a stored account of something (REH-kurd)"])
    ipa: str = Field("", description="IPA string (tokens joined)", examples=["ɹɛkɚd"])
    ipa_tokens: List[str] = Field(default_factory=list, examples=[["ɹ", "ɛ", "k", "ɚ", "d"]])
    respell: Optional[str] = Field(None, description="Pronunciation hint from the sense description",
                                   examples=["REH-kurd"])


class LexiconEntry(BaseModel):
    word: str = Field(..., examples=["record"])
    senses: List[LexiconSense]
    respell_candidates: List[str] = Field(default_factory=list,
                                          description="WellSaid-format respellings for this word",
                                          examples=[["REH-kurd", "ree-KORD"]])


class LexiconResponse(BaseModel):
    count: int = Field(..., examples=[100])
    words: List[LexiconEntry]


# ---------------------------------------------------------------------------
# check-text (script pre-flight)
# ---------------------------------------------------------------------------

class CheckTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200000,
                      description="Script text to pre-flight for heteronyms.",
                      examples=["Please record the record. We present the present live, live from the studio."])
    engine: Optional[str] = Field(None, pattern="^(claude|spacy|both)$",
                                  description='Intent engine: "claude", "spacy", or "both" (default; Claude with spaCy fallback).',
                                  examples=["both"])


class TextOccurrence(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(..., examples=["o1"])
    word: str = Field(..., description="Surface form as written", examples=["record"])
    base: str = Field(..., description="Lexicon key", examples=["record"])
    sentence: str
    char_start: int = Field(..., description="Offset of the word inside `sentence`")
    char_end: int
    text_start: Optional[int] = Field(None, description="Offset of the word inside the full text")
    text_end: Optional[int] = None
    intended: Optional[str] = Field(None, examples=["record_v"])
    intended_desc: Optional[str] = Field(None, examples=["to capture sound or data (rih-KORD)"])
    intended_ipa: Optional[str] = Field(None, examples=["ɹɪkɔːɹd"])
    intended_respell: Optional[str] = Field(None, description="Readable hint from the lexicon", examples=["rih-KORD"])
    respell_candidates: List[str] = Field(default_factory=list, description="WellSaid-format respellings")
    intent_engine: Optional[str] = Field(None, examples=["claude"])
    intent_confidence: Optional[float] = Field(None, examples=[0.95])
    candidates: Dict[str, str] = Field(default_factory=dict, description="sense id -> IPA")
    senses: List[LexiconSense] = Field(default_factory=list)


class CheckTextResponse(BaseModel):
    text: str
    count: int = Field(..., examples=[4])
    occurrences: List[TextOccurrence]
    engine: str = Field(..., description="Engine actually used", examples=["claude"])
    summary: List[str] = Field(default_factory=list,
                               examples=[['"record" -> record_v (to capture sound or data) [claude 0.95]']])


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

class JobTextRequest(BaseModel):
    """JSON body for POST /v1/jobs: render the text through WellSaid, then check it."""
    text: str = Field(..., min_length=1, examples=["Please record the record live, live from the studio."])
    speaker_id: int = Field(..., ge=0, examples=[26])
    model: str = Field("caruso", pattern="^(preview|caruso)$", examples=["caruso"])
    engine: Optional[str] = Field(None, pattern="^(claude|spacy|both)$")


class JobCreated(BaseModel):
    job_id: str = Field(..., examples=["7f3a1c2e9b0d4e51"])
    status: str = Field("queued", examples=["queued"])
    status_url: str = Field(..., examples=["/v1/jobs/7f3a1c2e9b0d4e51"])


class JobFiles(BaseModel):
    media: Optional[str] = Field(None, examples=["/v1/jobs/7f3a1c2e9b0d4e51/files/media.mp4"])
    report_json: Optional[str] = Field(None, examples=["/v1/jobs/7f3a1c2e9b0d4e51/files/report.json"])
    report_html: Optional[str] = Field(None, examples=["/v1/jobs/7f3a1c2e9b0d4e51/files/report.html"])


class JobParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    engine: Optional[str] = None
    speaker_id: Optional[int] = None
    model: Optional[str] = None
    has_script: bool = False


class JobSummary(BaseModel):
    """Row in GET /v1/jobs."""
    model_config = ConfigDict(extra="allow")
    id: str
    status: str = Field(..., description="queued | running | done | error")
    stage: Optional[str] = None
    progress: float = 0.0
    created: str = Field(..., description="ISO-8601 UTC")
    updated: Optional[str] = None
    mode: str = Field(..., description='"upload" (media file) or "script" (rendered via WellSaid)')
    filename: Optional[str] = None
    duration: Optional[float] = None
    occurrences: Optional[int] = None
    flags: Optional[int] = None
    error: Optional[ErrorDetail] = None


class JobResponse(JobSummary):
    params: JobParams = Field(default_factory=JobParams)
    files: JobFiles = Field(default_factory=JobFiles)
    result: Optional[Dict[str, Any]] = Field(
        None, description="check_media result with every path rewritten to /v1/jobs/{id}/files/... URLs")
    fixes: List[str] = Field(default_factory=list, description="Fix ids for this job")


class JobListResponse(BaseModel):
    count: int
    jobs: List[JobSummary]


class DeleteResponse(BaseModel):
    deleted: str
    ok: bool = True


# ---------------------------------------------------------------------------
# fixes
# ---------------------------------------------------------------------------

class FixRequest(BaseModel):
    occurrence_ids: Optional[List[str]] = Field(
        None, description="Occurrences to fix (default: every flagged one)", examples=[["o3", "o5"]])
    speaker_id: int = Field(..., ge=0, examples=[26])
    model: str = Field("caruso", pattern="^(preview|caruso)$", examples=["caruso"])
    max_tries: int = Field(3, ge=1, le=6)


class FixCreated(BaseModel):
    fix_id: str = Field(..., examples=["c41d0b7e"])
    job_id: str
    status: str = "queued"
    status_url: str = Field(..., examples=["/v1/jobs/7f3a1c2e9b0d4e51/fixes/c41d0b7e"])


class FixSummary(BaseModel):
    requested: int = 0
    done: int = 0
    verified: int = 0
    unverified: int = 0
    errors: int = 0


class FixResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    fix_id: str
    job_id: str
    status: str = Field(..., description="queued | running | done | error")
    stage: Optional[str] = None
    progress: float = 0.0
    created: str
    updated: Optional[str] = None
    speaker_id: Optional[int] = None
    model: Optional[str] = None
    occurrence_ids: List[str] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list,
                                          description="fix_occurrence results with take URLs")
    summary: FixSummary = Field(default_factory=FixSummary)
    error: Optional[ErrorDetail] = None
