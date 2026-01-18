from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from configs.browser import default_browser_factory
from configs.crawler import default_crawler_factory
from configs.js_injection import (
    PageInteractionFactory,
    PageInteractionPolicy,
    default_page_interaction_factory,
)
from configs.language import default_language_factory
from configs.llm import (
    IndustryAwareStrategyCache,
    LLMExtractionFactory,
    provider_strategy_from_llm_model_selector,
)
from configs.md import default_md_factory

from extensions.crawl.runner_constants import CACHE_MODE_MAP, build_accept_language
from extensions.crawl.state import CrawlState, get_crawl_state
from extensions.filter import md_gating
from extensions.guard.connectivity import ConnectivityGuard
from extensions.io import output_paths
from extensions.schedule.adaptive import AdaptiveSchedulingConfig
from extensions.schedule.crawler_pool import CrawlerPool
from extensions.utils.logging import LoggingExtension
from extensions.utils.resource_monitor import ResourceMonitor, ResourceMonitorConfig


@dataclass(frozen=True, slots=True)
class RunnerContext:
    repo_root: Path
    out_dir: Path
    logging_ext: LoggingExtension
    resource_monitor: Optional[ResourceMonitor]
    guard: ConnectivityGuard
    state: CrawlState
    crawler_base_cfg: Any
    page_policy: PageInteractionPolicy
    page_interaction_factory: PageInteractionFactory
    industry_llm_cache: Optional[IndustryAwareStrategyCache]
    crawler_pool: CrawlerPool
    scheduler_cfg: AdaptiveSchedulingConfig


async def build_runner_context(args: argparse.Namespace) -> RunnerContext:
    out_dir = output_paths.ensure_output_root(args.out_dir)

    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    repo_root = Path(str(args.repo_root)).expanduser().resolve()

    logging_ext = LoggingExtension(
        global_level=log_level,
        per_company_level=log_level,
        max_open_company_logs=128,
        enable_session_log=bool(args.enable_session_log),
        session_log_path=output_paths.global_path_obj("session.log")
        if args.enable_session_log
        else None,
    )

    resource_monitor: Optional[ResourceMonitor] = None
    if bool(args.enable_resource_monitor):
        resource_monitor = ResourceMonitor(
            output_path=output_paths.global_path_obj("resource_usage.json"),
            config=ResourceMonitorConfig(),
        )
        resource_monitor.start()

    # Language must be initialized once, early, and used everywhere through the factory.
    default_language_factory.set_language(args.lang)
    effective_langs = default_language_factory.effective_langs()
    logging.getLogger(__name__).info(
        "language_ready lang_target=%s lang_effective=%s",
        str(args.lang),
        ",".join(effective_langs),
    )

    cache_mode = CACHE_MODE_MAP[str(args.crawl4ai_cache_mode)]
    cache_dir = args.crawl4ai_cache_dir

    industry_llm_cache: Optional[IndustryAwareStrategyCache] = None
    if str(args.llm_mode) != "none":
        provider_strategy = provider_strategy_from_llm_model_selector(args.llm_model)
        llm_factory = LLMExtractionFactory(provider_strategy=provider_strategy)
        industry_llm_cache = IndustryAwareStrategyCache(
            factory=llm_factory,
            schema=None,
            extraction_type="schema",
            input_format=None,
            extra_args=None,
            verbose=False,
        )

    guard = ConnectivityGuard()
    await guard.start()

    # md_gating must not cache language regex at import-time; build config after set_language().
    gating_cfg = md_gating.build_gating_config()
    markdown_generator = default_md_factory.create(
        min_meaningful_words=gating_cfg.min_meaningful_words,
        interstitial_max_share=gating_cfg.interstitial_max_share,
        interstitial_min_hits=gating_cfg.interstitial_min_hits,
        cookie_max_fraction=gating_cfg.cookie_max_fraction,
        require_structure=gating_cfg.require_structure,
    )

    crawler_base_cfg = default_crawler_factory.create(
        markdown_generator=markdown_generator,
        page_timeout=int(args.page_timeout_ms) if args.page_timeout_ms else None,
        cache_mode=cache_mode,
        cache_base_dir=cache_dir,
    )

    page_policy = PageInteractionPolicy(wait_timeout_ms=int(args.page_timeout_ms))
    page_interaction_factory = default_page_interaction_factory

    state = get_crawl_state()

    # HTTP Accept-Language must reflect "target + English" deterministically.
    accept_language = build_accept_language(
        target=str(args.lang), effective=effective_langs
    )
    browser_cfg = default_browser_factory.create(lang=accept_language, headless=True)

    crawler_pool = CrawlerPool(
        browser_cfg=browser_cfg,
        size=int(args.crawler_pool_size),
        recycle_after_companies=int(args.crawler_recycle_after),
    )
    await crawler_pool.start()

    scheduler_cfg = AdaptiveSchedulingConfig(
        retry_base_dir=(out_dir / "_retry").resolve(),
        max_start_per_tick=int(args.max_start_per_tick),
        crawler_capacity_multiplier=int(args.crawler_capacity_multiplier),
        idle_recycle_interval_sec=float(args.idle_recycle_interval_sec),
        idle_recycle_raw_frac=float(args.idle_recycle_raw_frac),
        idle_recycle_eff_frac=float(args.idle_recycle_eff_frac),
    )

    llm_mode = str(getattr(args, "llm_mode", "none") or "none")
    llm_enabled = llm_mode != "none"

    # Policy:
    # When LLM is enabled, quarantine must NOT block scheduler admission/eligibility.
    # (Quarantine remains recorded, but scheduler ignores it.)
    if llm_enabled:
        scheduler_cfg.quarantine_enabled = False
        scheduler_cfg.last_one_standing_quarantine_enabled = False

    return RunnerContext(
        repo_root=repo_root,
        out_dir=out_dir,
        logging_ext=logging_ext,
        resource_monitor=resource_monitor,
        guard=guard,
        state=state,
        crawler_base_cfg=crawler_base_cfg,
        page_policy=page_policy,
        page_interaction_factory=page_interaction_factory,
        industry_llm_cache=industry_llm_cache,
        crawler_pool=crawler_pool,
        scheduler_cfg=scheduler_cfg,
    )
