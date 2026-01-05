from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional


# ---------------------------------------------------------------------------
# Canonical company status
# ---------------------------------------------------------------------------

CompanyStatus = Literal[
    "pending",
    "markdown_not_done",
    "markdown_done",
    "llm_not_done",
    "llm_done",
    "terminal_done",
]

COMPANY_STATUS_PENDING: CompanyStatus = "pending"
COMPANY_STATUS_MD_NOT_DONE: CompanyStatus = "markdown_not_done"
COMPANY_STATUS_MD_DONE: CompanyStatus = "markdown_done"
COMPANY_STATUS_LLM_NOT_DONE: CompanyStatus = "llm_not_done"
COMPANY_STATUS_LLM_DONE: CompanyStatus = "llm_done"
COMPANY_STATUS_TERMINAL_DONE: CompanyStatus = "terminal_done"

_KNOWN_COMPANY_STATUSES = {
    COMPANY_STATUS_PENDING,
    COMPANY_STATUS_MD_NOT_DONE,
    COMPANY_STATUS_MD_DONE,
    COMPANY_STATUS_LLM_NOT_DONE,
    COMPANY_STATUS_LLM_DONE,
    COMPANY_STATUS_TERMINAL_DONE,
}

_COMPANY_STATUS_RANK: Dict[str, int] = {
    COMPANY_STATUS_PENDING: 0,
    COMPANY_STATUS_MD_NOT_DONE: 1,
    COMPANY_STATUS_MD_DONE: 2,
    COMPANY_STATUS_LLM_NOT_DONE: 3,
    COMPANY_STATUS_LLM_DONE: 4,
    COMPANY_STATUS_TERMINAL_DONE: 5,
}


def _normalize_company_status(st: Optional[str]) -> CompanyStatus:
    s = (st or "").strip()
    if not s:
        return COMPANY_STATUS_PENDING
    if s in _KNOWN_COMPANY_STATUSES:
        return s  # type: ignore[return-value]
    return COMPANY_STATUS_PENDING


def _status_rank(st: Optional[str]) -> int:
    return _COMPANY_STATUS_RANK.get(_normalize_company_status(st), -1)


def _prefer_higher_status(
    current: Optional[str], derived: Optional[str]
) -> CompanyStatus:
    c = _normalize_company_status(current)
    d = _normalize_company_status(derived)
    return c if _status_rank(c) >= _status_rank(d) else d


def _to_int_or_none(v: Any) -> Optional[int]:
    """
    Int parser that tolerates pandas/JSON artifacts like "0.0" / 0.0.
    Raises ValueError for truly invalid values (keeps errors actionable).
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)

    s = str(v).strip()
    if not s:
        return None

    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError as e:
            raise ValueError(f"invalid literal for int(): {v!r}") from e


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    return float(s)


def _to_str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    return s if s.strip() else None


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, int):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return True
    if s in ("0", "false", "no", "n", "f", ""):
        return False
    return False


def _safe_json_obj(v: Any) -> Optional[Dict[str, Any]]:
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    # NOTE: Keep strict: only dict is accepted at model level.
    return None


# ---------------------------------------------------------------------------
# URL Index models (url_index.json)
# ---------------------------------------------------------------------------

URL_INDEX_META_KEY: str = "__meta__"

# NOTE: status is intentionally not strict; pipelines may add more statuses.
UrlIndexEntryStatus = str


@dataclass(slots=True)
class UrlIndexEntry:
    """
    Canonical per-URL entry payload stored under url_index.json[url].

    This mirrors what crawl.runner.py writes today, plus adds company_id
    so a single entry remains self-identifying when merged/exported.
    """

    # identity
    company_id: str
    url: str
    requested_url: Optional[str] = None

    # network / crawl signals
    status_code: Optional[int] = None
    error: Any = None
    depth: Optional[int] = None

    # pipeline counters
    presence: int = 0
    extracted: int = 0

    # markdown gating
    gating_accept: bool = False
    gating_action: Optional[str] = None
    gating_reason: Optional[str] = None
    md_total_words: Optional[float] = None

    # status + timestamps
    status: UrlIndexEntryStatus = ""
    updated_at: Optional[str] = None
    created_at: Optional[str] = None  # recommended to be set once at first write

    # artifacts
    markdown_path: Optional[str] = None
    html_path: Optional[str] = None

    # retry markers
    scheduled_retry: Optional[bool] = None
    timeout_page_exceeded: Optional[bool] = None
    memory_pressure: Optional[bool] = None

    # stage-done hints (crawl.state.py may infer from these)
    markdown_done: Optional[bool] = None
    llm_done: Optional[bool] = None

    # llm artifacts (crawl.state.py checks these keys today)
    json_path: Optional[str] = None
    extraction_path: Optional[str] = None
    product_path: Optional[str] = None
    products_path: Optional[str] = None

    # forward-compat: keep unknown fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "UrlIndexEntry":
        cid = (self.company_id or "").strip()
        u = (self.url or "").strip()

        return UrlIndexEntry(
            company_id=cid,
            url=u,
            requested_url=_to_str_or_none(self.requested_url),
            status_code=_to_int_or_none(self.status_code),
            error=self.error,
            depth=_to_int_or_none(self.depth),
            presence=int(self.presence or 0),
            extracted=int(self.extracted or 0),
            gating_accept=_to_bool(self.gating_accept),
            gating_action=_to_str_or_none(self.gating_action),
            gating_reason=_to_str_or_none(self.gating_reason),
            md_total_words=_to_float_or_none(self.md_total_words),
            status=str(self.status or ""),
            updated_at=_to_str_or_none(self.updated_at),
            created_at=_to_str_or_none(self.created_at),
            markdown_path=_to_str_or_none(self.markdown_path),
            html_path=_to_str_or_none(self.html_path),
            scheduled_retry=(
                None if self.scheduled_retry is None else _to_bool(self.scheduled_retry)
            ),
            timeout_page_exceeded=(
                None
                if self.timeout_page_exceeded is None
                else _to_bool(self.timeout_page_exceeded)
            ),
            memory_pressure=(
                None if self.memory_pressure is None else _to_bool(self.memory_pressure)
            ),
            markdown_done=(
                None if self.markdown_done is None else _to_bool(self.markdown_done)
            ),
            llm_done=(None if self.llm_done is None else _to_bool(self.llm_done)),
            json_path=_to_str_or_none(self.json_path),
            extraction_path=_to_str_or_none(self.extraction_path),
            product_path=_to_str_or_none(self.product_path),
            products_path=_to_str_or_none(self.products_path),
            extra=dict(self.extra) if isinstance(self.extra, dict) else {},
        )

    @staticmethod
    def from_dict(
        d: Mapping[str, Any], *, company_id: str, url: str
    ) -> "UrlIndexEntry":
        src = dict(d) if isinstance(d, Mapping) else {}
        known = UrlIndexEntry(
            company_id=_to_str_or_none(src.pop("company_id")) or company_id,
            url=_to_str_or_none(src.pop("url")) or url,
            requested_url=_to_str_or_none(src.pop("requested_url")),
            status_code=_to_int_or_none(src.pop("status_code", None)),
            error=src.pop("error", None),
            depth=_to_int_or_none(src.pop("depth", None)),
            presence=int(src.pop("presence", 0) or 0),
            extracted=int(src.pop("extracted", 0) or 0),
            gating_accept=_to_bool(src.pop("gating_accept", False)),
            gating_action=_to_str_or_none(src.pop("gating_action", None)),
            gating_reason=_to_str_or_none(src.pop("gating_reason", None)),
            md_total_words=_to_float_or_none(src.pop("md_total_words", None)),
            status=str(src.pop("status", "") or ""),
            updated_at=_to_str_or_none(src.pop("updated_at", None)),
            created_at=_to_str_or_none(src.pop("created_at", None)),
            markdown_path=_to_str_or_none(src.pop("markdown_path", None)),
            html_path=_to_str_or_none(src.pop("html_path", None)),
            scheduled_retry=(
                None
                if "scheduled_retry" not in src
                else _to_bool(src.pop("scheduled_retry"))
            ),
            timeout_page_exceeded=(
                None
                if "timeout_page_exceeded" not in src
                else _to_bool(src.pop("timeout_page_exceeded"))
            ),
            memory_pressure=(
                None
                if "memory_pressure" not in src
                else _to_bool(src.pop("memory_pressure"))
            ),
            markdown_done=(
                None
                if "markdown_done" not in src
                else _to_bool(src.pop("markdown_done"))
            ),
            llm_done=None if "llm_done" not in src else _to_bool(src.pop("llm_done")),
            json_path=_to_str_or_none(src.pop("json_path", None)),
            extraction_path=_to_str_or_none(src.pop("extraction_path", None)),
            product_path=_to_str_or_none(src.pop("product_path", None)),
            products_path=_to_str_or_none(src.pop("products_path", None)),
            extra=src,  # leftover fields
        )
        return known.normalized()

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "company_id": self.company_id,
            "url": self.url,
            "requested_url": self.requested_url,
            "status_code": self.status_code,
            "error": self.error,
            "depth": self.depth,
            "presence": int(self.presence),
            "extracted": int(self.extracted),
            "gating_accept": bool(self.gating_accept),
            "gating_action": self.gating_action,
            "gating_reason": self.gating_reason,
            "md_total_words": self.md_total_words,
            "status": self.status,
            "updated_at": self.updated_at,
            "created_at": self.created_at,
            "markdown_path": self.markdown_path,
            "html_path": self.html_path,
            "scheduled_retry": self.scheduled_retry,
            "timeout_page_exceeded": self.timeout_page_exceeded,
            "memory_pressure": self.memory_pressure,
            "markdown_done": self.markdown_done,
            "llm_done": self.llm_done,
            "json_path": self.json_path,
            "extraction_path": self.extraction_path,
            "product_path": self.product_path,
            "products_path": self.products_path,
        }
        if isinstance(self.extra, dict) and self.extra:
            base.update(self.extra)
        return {k: v for k, v in base.items() if v is not None}


@dataclass(slots=True)
class UrlIndexMeta:
    """
    Canonical __meta__ payload stored at url_index.json["__meta__"].

    crawl.runner.py currently writes:
      - crawl_finished, crawl_finished_at, crawl_reason
      - total_pages, markdown_saved, markdown_suppressed, timeout_pages, memory_pressure_pages
      - pages_seen, hard_max_pages, hard_max_pages_hit
    """

    company_id: str

    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    crawl_finished: Optional[bool] = None
    crawl_finished_at: Optional[str] = None
    crawl_reason: Optional[str] = None

    total_pages: Optional[int] = None
    markdown_saved: Optional[int] = None
    markdown_suppressed: Optional[int] = None
    timeout_pages: Optional[int] = None
    memory_pressure_pages: Optional[int] = None

    pages_seen: Optional[int] = None
    hard_max_pages: Optional[int] = None
    hard_max_pages_hit: Optional[bool] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "UrlIndexMeta":
        return UrlIndexMeta(
            company_id=(self.company_id or "").strip(),
            created_at=_to_str_or_none(self.created_at),
            updated_at=_to_str_or_none(self.updated_at),
            crawl_finished=(
                None if self.crawl_finished is None else _to_bool(self.crawl_finished)
            ),
            crawl_finished_at=_to_str_or_none(self.crawl_finished_at),
            crawl_reason=_to_str_or_none(self.crawl_reason),
            total_pages=_to_int_or_none(self.total_pages),
            markdown_saved=_to_int_or_none(self.markdown_saved),
            markdown_suppressed=_to_int_or_none(self.markdown_suppressed),
            timeout_pages=_to_int_or_none(self.timeout_pages),
            memory_pressure_pages=_to_int_or_none(self.memory_pressure_pages),
            pages_seen=_to_int_or_none(self.pages_seen),
            hard_max_pages=_to_int_or_none(self.hard_max_pages),
            hard_max_pages_hit=(
                None
                if self.hard_max_pages_hit is None
                else _to_bool(self.hard_max_pages_hit)
            ),
            extra=dict(self.extra) if isinstance(self.extra, dict) else {},
        )

    @staticmethod
    def from_dict(d: Mapping[str, Any], *, company_id: str) -> "UrlIndexMeta":
        src = dict(d) if isinstance(d, Mapping) else {}
        known = UrlIndexMeta(
            company_id=_to_str_or_none(src.pop("company_id")) or company_id,
            created_at=_to_str_or_none(src.pop("created_at", None)),
            updated_at=_to_str_or_none(src.pop("updated_at", None)),
            crawl_finished=(
                None
                if "crawl_finished" not in src
                else _to_bool(src.pop("crawl_finished"))
            ),
            crawl_finished_at=_to_str_or_none(src.pop("crawl_finished_at", None)),
            crawl_reason=_to_str_or_none(src.pop("crawl_reason", None)),
            total_pages=_to_int_or_none(src.pop("total_pages", None)),
            markdown_saved=_to_int_or_none(src.pop("markdown_saved", None)),
            markdown_suppressed=_to_int_or_none(src.pop("markdown_suppressed", None)),
            timeout_pages=_to_int_or_none(src.pop("timeout_pages", None)),
            memory_pressure_pages=_to_int_or_none(
                src.pop("memory_pressure_pages", None)
            ),
            pages_seen=_to_int_or_none(src.pop("pages_seen", None)),
            hard_max_pages=_to_int_or_none(src.pop("hard_max_pages", None)),
            hard_max_pages_hit=(
                None
                if "hard_max_pages_hit" not in src
                else _to_bool(src.pop("hard_max_pages_hit"))
            ),
            extra=src,
        )
        return known.normalized()

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "company_id": self.company_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "crawl_finished": self.crawl_finished,
            "crawl_finished_at": self.crawl_finished_at,
            "crawl_reason": self.crawl_reason,
            "total_pages": self.total_pages,
            "markdown_saved": self.markdown_saved,
            "markdown_suppressed": self.markdown_suppressed,
            "timeout_pages": self.timeout_pages,
            "memory_pressure_pages": self.memory_pressure_pages,
            "pages_seen": self.pages_seen,
            "hard_max_pages": self.hard_max_pages,
            "hard_max_pages": self.hard_max_pages,
            "hard_max_pages_hit": self.hard_max_pages_hit,
        }
        if isinstance(self.extra, dict) and self.extra:
            base.update(self.extra)
        return {k: v for k, v in base.items() if v is not None}


# ---------------------------------------------------------------------------
# Unified Company (single canonical class)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Company:
    # -----------------------------------------------------------------------
    # Identity / input (run.py + load_source.py + crawl.state snapshot identity)
    # -----------------------------------------------------------------------
    company_id: str  # canonical id (bvdid / hojin_id)
    root_url: str  # canonical root url (domain_url / url / root_url)
    name: Optional[str] = None

    # input metadata (load_source CompanyInput.metadata)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Classification (run.py + crawl.state snapshot)
    # -----------------------------------------------------------------------
    industry: Optional[int] = None
    nace: Optional[int] = None
    industry_label: Optional[str] = None
    industry_label_source: Optional[str] = None  # crawl_meta key: industry_label_source

    # -----------------------------------------------------------------------
    # Crawl progress + terminal info (crawl.state snapshot)
    # -----------------------------------------------------------------------
    status: CompanyStatus = COMPANY_STATUS_PENDING
    crawl_finished: bool = False

    urls_total: int = 0
    urls_markdown_done: int = 0
    urls_llm_done: int = 0

    last_error: Optional[str] = None
    done_reason: Optional[str] = None
    done_details: Optional[Dict[str, Any]] = None
    done_at: Optional[str] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_crawled_at: Optional[str] = None

    # optional run-level hint stored in crawl_meta.json in crawl.state.py
    max_pages: Optional[int] = None

    # -----------------------------------------------------------------------
    # Retry state (schedule.retry::CompanyRetryState) - flattened
    # -----------------------------------------------------------------------
    retry_cls: str = "net"
    retry_attempts: int = 0
    retry_next_eligible_at: float = 0.0
    retry_updated_at: float = 0.0
    retry_last_error: str = ""
    retry_last_stage: str = ""

    retry_net_attempts: int = 0
    retry_stall_attempts: int = 0
    retry_mem_attempts: int = 0
    retry_other_attempts: int = 0

    retry_mem_hits: int = 0
    retry_last_stall_kind: str = "unknown"

    retry_last_progress_md_done: int = 0
    retry_last_seen_md_done: int = 0

    retry_last_error_sig: str = ""
    retry_same_error_streak: int = 0
    retry_last_error_sig_updated_at: float = 0.0

    # -----------------------------------------------------------------------
    # Analyze-only fields (analyze_output.py CompanyRow + CompanyProfileStats)
    # Keep for schema; leave unfilled until analyze script populates them.
    # -----------------------------------------------------------------------

    # URL index aggregates (analyze_output CompanyRow)
    url_count: Optional[int] = None
    url_status_ok: Optional[int] = None
    url_status_redirect: Optional[int] = None
    url_status_client_error: Optional[int] = None
    url_status_server_error: Optional[int] = None
    url_status_other: Optional[int] = None
    url_error_count: Optional[int] = None
    gating_accept_true: Optional[int] = None
    gating_accept_false: Optional[int] = None
    presence_positive: Optional[int] = None
    presence_zero: Optional[int] = None
    extracted_positive: Optional[int] = None
    extracted_zero: Optional[int] = None
    markdown_saved: Optional[int] = None
    markdown_suppressed: Optional[int] = None
    markdown_other_status: Optional[int] = None
    md_words_files: Optional[int] = None
    md_words_total: Optional[int] = None
    md_words_mean_per_file: Optional[float] = None
    md_words_median_per_file: Optional[float] = None

    # Token accounting
    md_tokens_all: Optional[int] = None
    llm_input_tokens_done: Optional[int] = None
    llm_output_tokens_done: Optional[int] = None
    llm_done_pages: Optional[int] = None
    llm_pending_pages: Optional[int] = None
    product_files_total: Optional[int] = None
    product_files_used_done: Optional[int] = None

    # Cost
    cost_input_usd_expected: Optional[float] = None
    cost_input_usd_all_hit: Optional[float] = None
    cost_input_usd_all_miss: Optional[float] = None
    cost_output_usd: Optional[float] = None
    cost_total_usd_expected: Optional[float] = None

    # Company profile stats (analyze_output CompanyProfileStats, flattened with prefix)
    profile_present: Optional[bool] = None
    profile_pipeline_version: str = ""  # analyze fills from extensions.utils.versioning
    profile_offerings_total: Optional[int] = None
    profile_offerings_products: Optional[int] = None
    profile_offerings_services: Optional[int] = None
    profile_sources_total: Optional[int] = None
    profile_desc_sentences_total: Optional[int] = None
    profile_alias_total: Optional[int] = None

    profile_embedding_model: str = ""
    profile_embedding_dim: Optional[int] = None
    profile_embedding_offerings_vecs: Optional[int] = None

    profile_json_tokens: Optional[int] = None
    profile_json_bytes: Optional[int] = None
    profile_md_tokens: Optional[int] = None
    profile_md_bytes: Optional[int] = None

    # Debug paths (analyze_output CompanyRow)
    _company_dir: Optional[str] = None
    _meta_path: Optional[str] = None
    _url_index_path: Optional[str] = None
    _db_path: Optional[str] = None

    # -----------------------------------------------------------------------
    # Aliases (NO extra storage fields; these are properties only)
    # -----------------------------------------------------------------------

    @property
    def bvdid(self) -> str:
        return self.company_id

    @property
    def domain_url(self) -> str:
        return self.root_url

    @property
    def industry_source(self) -> Optional[str]:
        # crawl.state snapshot uses industry_source but it maps to crawl_meta key industry_label_source
        return self.industry_label_source

    # -----------------------------------------------------------------------
    # Metadata helpers (from CompanyInput ergonomics)
    # -----------------------------------------------------------------------

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        v = self.metadata.get(key)
        if v is None:
            return default
        s = str(v)
        return s if s.strip() else default

    def get_str(self, key: str, default: str = "") -> str:
        v = self.metadata.get(key)
        if v is None:
            return default
        return str(v)

    def get_int(self, key: str) -> Optional[int]:
        return _to_int_or_none(self.metadata.get(key))

    # -----------------------------------------------------------------------
    # Normalization (must match crawl.state.py CompanySnapshot.normalized)
    # -----------------------------------------------------------------------

    def normalized(self) -> "Company":
        st = _normalize_company_status(self.status)

        total = int(self.urls_total or 0)
        md = int(self.urls_markdown_done or 0)
        llm = int(self.urls_llm_done or 0)

        total = max(total, md, llm)
        md = min(md, total)
        llm = min(llm, total)

        # llm implies markdown
        if llm > 0:
            md = max(md, llm)

        # llm_done implies all done
        if st == COMPANY_STATUS_LLM_DONE:
            md = total
            llm = total

        # terminal_done is treated as markdown-finished for accounting,
        # but must NOT be treated as llm_done.
        if st == COMPANY_STATUS_TERMINAL_DONE:
            md = total
            llm = min(llm, total)

        le = self.last_error
        if le is not None:
            le = (le or "")[:4000] or None

        dr = self.done_reason
        if dr is not None:
            dr = (dr or "")[:256] or None

        return Company(
            company_id=self.company_id,
            root_url=self.root_url,
            name=self.name,
            metadata=dict(self.metadata),
            industry=self.industry,
            nace=self.nace,
            industry_label=self.industry_label,
            industry_label_source=self.industry_label_source,
            status=st,
            crawl_finished=bool(self.crawl_finished),
            urls_total=total,
            urls_markdown_done=md,
            urls_llm_done=llm,
            last_error=le,
            done_reason=dr,
            done_details=self.done_details
            if isinstance(self.done_details, dict)
            else None,
            done_at=self.done_at,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_crawled_at=self.last_crawled_at,
            max_pages=self.max_pages,
            # retry (copy-through)
            retry_cls=self.retry_cls,
            retry_attempts=int(self.retry_attempts),
            retry_next_eligible_at=float(self.retry_next_eligible_at),
            retry_updated_at=float(self.retry_updated_at),
            retry_last_error=str(self.retry_last_error or ""),
            retry_last_stage=str(self.retry_last_stage or ""),
            retry_net_attempts=int(self.retry_net_attempts),
            retry_stall_attempts=int(self.retry_stall_attempts),
            retry_mem_attempts=int(self.retry_mem_attempts),
            retry_other_attempts=int(self.retry_other_attempts),
            retry_mem_hits=int(self.retry_mem_hits),
            retry_last_stall_kind=str(self.retry_last_stall_kind or "unknown"),
            retry_last_progress_md_done=int(self.retry_last_progress_md_done),
            retry_last_seen_md_done=int(self.retry_last_seen_md_done),
            retry_last_error_sig=str(self.retry_last_error_sig or ""),
            retry_same_error_streak=int(self.retry_same_error_streak),
            retry_last_error_sig_updated_at=float(self.retry_last_error_sig_updated_at),
            # analyze-only fields: keep current values (do not touch)
            url_count=self.url_count,
            url_status_ok=self.url_status_ok,
            url_status_redirect=self.url_status_redirect,
            url_status_client_error=self.url_status_client_error,
            url_status_server_error=self.url_status_server_error,
            url_status_other=self.url_status_other,
            url_error_count=self.url_error_count,
            gating_accept_true=self.gating_accept_true,
            gating_accept_false=self.gating_accept_false,
            presence_positive=self.presence_positive,
            presence_zero=self.presence_zero,
            extracted_positive=self.extracted_positive,
            extracted_zero=self.extracted_zero,
            markdown_saved=self.markdown_saved,
            markdown_suppressed=self.markdown_suppressed,
            markdown_other_status=self.markdown_other_status,
            md_words_files=self.md_words_files,
            md_words_total=self.md_words_total,
            md_words_mean_per_file=self.md_words_mean_per_file,
            md_words_median_per_file=self.md_words_median_per_file,
            md_tokens_all=self.md_tokens_all,
            llm_input_tokens_done=self.llm_input_tokens_done,
            llm_output_tokens_done=self.llm_output_tokens_done,
            llm_done_pages=self.llm_done_pages,
            llm_pending_pages=self.llm_pending_pages,
            product_files_total=self.product_files_total,
            product_files_used_done=self.product_files_used_done,
            cost_input_usd_expected=self.cost_input_usd_expected,
            cost_input_usd_all_hit=self.cost_input_usd_all_hit,
            cost_input_usd_all_miss=self.cost_input_usd_all_miss,
            cost_output_usd=self.cost_output_usd,
            cost_total_usd_expected=self.cost_total_usd_expected,
            profile_present=self.profile_present,
            profile_pipeline_version=self.profile_pipeline_version,
            profile_offerings_total=self.profile_offerings_total,
            profile_offerings_products=self.profile_offerings_products,
            profile_offerings_services=self.profile_offerings_services,
            profile_sources_total=self.profile_sources_total,
            profile_desc_sentences_total=self.profile_desc_sentences_total,
            profile_alias_total=self.profile_alias_total,
            profile_embedding_model=self.profile_embedding_model,
            profile_embedding_dim=self.profile_embedding_dim,
            profile_embedding_offerings_vecs=self.profile_embedding_offerings_vecs,
            profile_json_tokens=self.profile_json_tokens,
            profile_json_bytes=self.profile_json_bytes,
            profile_md_tokens=self.profile_md_tokens,
            profile_md_bytes=self.profile_md_bytes,
            _company_dir=self._company_dir,
            _meta_path=self._meta_path,
            _url_index_path=self._url_index_path,
            _db_path=self._db_path,
        )

    # -----------------------------------------------------------------------
    # Conversions (strict; no imports from extensions.* to avoid cycles)
    # -----------------------------------------------------------------------

    @staticmethod
    def from_input(
        *,
        company_id: str,
        root_url: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Company":
        md = dict(metadata) if isinstance(metadata, dict) else {}
        c = Company(company_id=company_id, root_url=root_url, name=name, metadata=md)

        # If enrichment wrote into metadata, pull it into canonical fields.
        c.industry = _to_int_or_none(md.get("industry"))
        c.nace = _to_int_or_none(md.get("nace"))
        c.industry_label = _to_str_or_none(md.get("industry_label"))
        c.industry_label_source = _to_str_or_none(md.get("industry_label_source"))

        return c.normalized()

    def apply_snapshot_dict(self, snap: Dict[str, Any]) -> None:
        """
        Apply a crawl.state-like snapshot dict into this Company instance.
        This is for in-process refresh without reconstructing the object.
        """
        self.name = _to_str_or_none(snap.get("name")) or self.name
        self.root_url = _to_str_or_none(snap.get("root_url")) or self.root_url

        self.industry = _to_int_or_none(snap.get("industry"))
        self.nace = _to_int_or_none(snap.get("nace"))
        self.industry_label = _to_str_or_none(snap.get("industry_label"))
        self.industry_label_source = _to_str_or_none(
            snap.get("industry_label_source")
            if "industry_label_source" in snap
            else snap.get("industry_source")
        )

        self.status = _normalize_company_status(_to_str_or_none(snap.get("status")))
        self.crawl_finished = _to_bool(snap.get("crawl_finished"))

        self.urls_total = _to_int_or_none(snap.get("urls_total")) or 0
        self.urls_markdown_done = _to_int_or_none(snap.get("urls_markdown_done")) or 0
        self.urls_llm_done = _to_int_or_none(snap.get("urls_llm_done")) or 0

        self.last_error = _to_str_or_none(snap.get("last_error"))
        self.done_reason = _to_str_or_none(snap.get("done_reason"))
        self.done_details = _safe_json_obj(snap.get("done_details"))
        self.done_at = _to_str_or_none(snap.get("done_at"))

        self.created_at = _to_str_or_none(snap.get("created_at"))
        self.updated_at = _to_str_or_none(snap.get("updated_at"))
        self.last_crawled_at = _to_str_or_none(snap.get("last_crawled_at"))

        self.max_pages = _to_int_or_none(snap.get("max_pages"))

        # normalize counters/status invariants
        n = self.normalized()
        self.status = n.status
        self.urls_total = n.urls_total
        self.urls_markdown_done = n.urls_markdown_done
        self.urls_llm_done = n.urls_llm_done
        self.last_error = n.last_error
        self.done_reason = n.done_reason

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "Company",
    "CompanyStatus",
    "COMPANY_STATUS_PENDING",
    "COMPANY_STATUS_MD_NOT_DONE",
    "COMPANY_STATUS_MD_DONE",
    "COMPANY_STATUS_LLM_NOT_DONE",
    "COMPANY_STATUS_LLM_DONE",
    "COMPANY_STATUS_TERMINAL_DONE",
    # url index models
    "URL_INDEX_META_KEY",
    "UrlIndexEntry",
    "UrlIndexEntryStatus",
    "UrlIndexMeta",
]
