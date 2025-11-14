from __future__ import annotations

import os
import logging
from dataclasses import dataclass, asdict
from typing import Optional

# --------------------------------------------------------------------
# Module logger
# --------------------------------------------------------------------
_logger = logging.getLogger("slot_allocator")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())

# -----------------------------
# Config & snapshots
# -----------------------------
@dataclass(frozen=True)
class SlotConfig:
    """
    Global policy knobs.

    - max_slots:             The CLI --max-slots for this run (your 'budget' knob).
    - per_company_cap:       Hard ceiling per company (prevents 400-per-company blowups). Default 16.
    - per_company_min:       Floor per company. Default 2 (keeps some parallelism for very large runs).
    - tail_start_fraction:   When (finished / total) passes this, we begin tail boosting. Default 0.75.
    - tail_boost_cap:        Optional higher cap only in the tail (kept = cap if None).
    """
    max_slots: int
    per_company_cap: int = int(os.getenv("SLOT_PER_COMPANY_CAP", "16"))
    per_company_min: int = int(os.getenv("SLOT_PER_COMPANY_MIN", "2"))
    tail_start_fraction: float = float(os.getenv("SLOT_TAIL_FRAC", "0.75"))
    tail_boost_cap: Optional[int] = None  # e.g., 24 or 32; if None, reuse per_company_cap


@dataclass(frozen=True)
class SessionSnapshot:
    """
    A lightweight view of the session's progress.
    Provide what you have; fields are conservative if unknown.
    """
    total_companies: int
    finished_companies: int
    running_companies: int  # currently executing companies (tasks not yet finished)


@dataclass(frozen=True)
class CompanySnapshot:
    """
    Optional per-company signal to bias slot distribution in the tail.
    All fields are optional and safely defaulted by the allocator.

    - urls_total / urls_done:  Remaining backlog heuristic.
    - timeout_rate:            0.0 ~ 1.0 recent timeout share (transport + soft timeouts).
    """
    urls_total: Optional[int] = None
    urls_done: Optional[int] = None
    timeout_rate: Optional[float] = None


# -----------------------------
# Helpers
# -----------------------------
def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _baseline_per_company(max_slots: int, running_companies: int, lo: int, hi: int) -> int:
    """
    Keep the spirit of your old formula (global_slots / running),
    but strictly clamp into [lo, hi] so single-company runs never exceed the cap.
    """
    if running_companies <= 0:
        running_companies = 1
    raw = max_slots // min(running_companies, max_slots) if max_slots > 0 else lo
    out = _clamp(raw, lo, hi)
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            "[slot] baseline_per_company: max_slots=%d running=%d lo=%d hi=%d → raw=%d out=%d",
            max_slots, running_companies, lo, hi, raw, out,
        )
    return out


# -----------------------------
# Public API
# -----------------------------
class SlotAllocator:
    """
    Compute safe initial slots and provide dynamic reallocation guidance for 'tail' scenarios.

    This class is deliberately side-effect free; you call it to get integers you then pass
    into your per-company MemoryAdaptiveDispatcher (via make_dispatcher(...)).
    """

    def __init__(self, cfg: SlotConfig):
        self.cfg = cfg
        if _logger.isEnabledFor(logging.INFO):
            _logger.info("[slot] init: %s", asdict(cfg))

    # ---- 1) initial cap / safety: solve "1 company * 400 slots" problem ----
    def initial_per_company(self, total_companies: int) -> int:
        """
        Use this at session start to compute the per-company slots you pass to make_dispatcher()
        for each new company. It respects the hard cap.
        """
        running = max(1, min(total_companies, self.cfg.max_slots))
        out = _baseline_per_company(
            self.cfg.max_slots,
            running,
            self.cfg.per_company_min,
            self.cfg.per_company_cap,
        )
        if _logger.isEnabledFor(logging.INFO):
            _logger.info(
                "[slot] initial_per_company: total_companies=%d running=%d → per_company=%d (cfg: min=%d cap=%d max_slots=%d)",
                total_companies, running, out, self.cfg.per_company_min, self.cfg.per_company_cap, self.cfg.max_slots,
            )
        return out

    # ---- 2) dynamic tail: solve "stragglers take forever" problem ----
    def recommend_for_company(
        self,
        bvdid: str,
        session: SessionSnapshot,
        company: Optional[CompanySnapshot] = None,
    ) -> int:
        """
        Return the recommended *current* per-company slot count for a given company,
        given present session state. You can poll this periodically (e.g., per pass/loop)
        and update that company's dispatcher max permits if your dispatcher supports it.

        The recommendation:
          - Pre-tail: even share, clamped to [min, cap].
          - Tail (>= tail_start_fraction finished): boost up to tail cap (default: same as cap).
          - Within tail, bias *slightly* by remaining backlog & timeout rate — but never above cap.
        """
        cfg = self.cfg

        # 2a) Baseline from global budget & current running count
        base = _baseline_per_company(
            cfg.max_slots,
            max(1, session.running_companies),
            cfg.per_company_min,
            cfg.per_company_cap,
        )

        # 2b) Are we in the tail?
        finished = session.finished_companies
        total = max(1, session.total_companies)
        frac_done = finished / total
        in_tail = frac_done >= cfg.tail_start_fraction

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "[slot] session: bvdid=%s finished=%d total=%d frac=%.4f running=%d tail_start=%.2f in_tail=%s",
                bvdid, finished, total, frac_done, session.running_companies, cfg.tail_start_fraction, in_tail,
            )

        if not in_tail:
            if _logger.isEnabledFor(logging.INFO):
                _logger.info("[slot] recommend (pre-tail): bvdid=%s → %d", bvdid, base)
            return base

        # 2c) Tail: allow a larger headroom (often equal to cap; can be a separate tail cap)
        tail_cap = cfg.tail_boost_cap if (cfg.tail_boost_cap and cfg.tail_boost_cap > 0) else cfg.per_company_cap

        # If few runners remain, the baseline naturally grows (max_slots / runners), but still clamp to tail_cap.
        boosted = _clamp(
            cfg.max_slots // max(1, session.running_companies),
            cfg.per_company_min,
            tail_cap,
        )

        if company is None:
            if _logger.isEnabledFor(logging.INFO):
                _logger.info(
                    "[slot] recommend (tail, no company stats): bvdid=%s base=%d boosted=%d tail_cap=%d → %d",
                    bvdid, base, boosted, tail_cap, boosted,
                )
            return boosted

        # 2d) Light biasing by company backlog & timeout profile (never exceed tail_cap)
        rem = None
        if company.urls_total is not None and company.urls_done is not None:
            rem = max(0, int(company.urls_total) - int(company.urls_done))

        if rem is None:
            backlog_weight = 0.7
        else:
            backlog_weight = min(1.0, (rem / 100.0) ** 0.5)

        to_rate = max(0.0, min(1.0, float(company.timeout_rate or 0.0)))
        timeout_bonus = 1 if to_rate >= 0.20 else 0

        target = int(round(base + backlog_weight * (boosted - base))) + timeout_bonus
        out = _clamp(target, base, tail_cap)

        if _logger.isEnabledFor(logging.INFO):
            _logger.info(
                "[slot] recommend (tail): bvdid=%s base=%d boosted=%d tail_cap=%d rem=%s weight=%.3f "
                "timeout_rate=%.2f bonus=%d → target=%d out=%d",
                bvdid, base, boosted, tail_cap, "unknown" if rem is None else rem, backlog_weight,
                to_rate, timeout_bonus, target, out,
            )
        return out


# -----------------------------
# Convenience constructors
# -----------------------------
def default_slot_allocator(max_slots: int) -> SlotAllocator:
    """
    Build a SlotAllocator using env-var defaults for caps, mins, and tail fraction.
    """
    alloc = SlotAllocator(SlotConfig(max_slots=max_slots))
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug("[slot] default_slot_allocator created: max_slots=%d", max_slots)
    return alloc


# -----------------------------
# Backwards-compat convenience
# -----------------------------
def per_company_slots_capped(n_companies: int, max_slots: int, *, cap: int = 16, min_slots: int = 2) -> int:
    """
    A simple function form mirroring your old helper, but with a hard cap.
    Useful if you want a single-line swap without adopting the full SlotAllocator.
    """
    cfg = SlotConfig(max_slots=max_slots, per_company_cap=cap, per_company_min=min_slots)
    alloc = SlotAllocator(cfg)
    out = alloc.initial_per_company(n_companies)
    if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug(
            "[slot] per_company_slots_capped: n_companies=%d max_slots=%d cap=%d min=%d → %d",
            n_companies, max_slots, cap, min_slots, out,
        )
    return out


__all__ = [
    "SlotConfig",
    "SessionSnapshot",
    "CompanySnapshot",
    "SlotAllocator",
    "default_slot_allocator",
    "per_company_slots_capped",
]