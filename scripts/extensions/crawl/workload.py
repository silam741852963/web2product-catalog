from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from configs.deep_crawl import (
    DFSDeepCrawlStrategyProvider,
    DeepCrawlStrategyFactory,
    DeepCrawlStrategyFactory as DeepCrawlStrategyFactoryType,
)
from configs.models import Company

from extensions.crawl.recrawl_policy import apply_recrawl_policy
from extensions.crawl.state import CrawlState
from extensions.filter.dataset_external import build_dataset_externals
from extensions.filter.dual_bm25 import (
    DualBM25Filter,
    DualBM25Scorer,
    build_dual_bm25_components,
)
from extensions.io.load_source import (
    IndustryEnrichmentConfig,
    load_companies_from_source_with_industry,
)


def load_companies(args: argparse.Namespace) -> List[Company]:
    if args.company_file:
        industry_enabled = bool(args.industry_enrichment) and (
            not bool(args.no_industry_enrichment)
        )
        cfg = IndustryEnrichmentConfig(
            enabled=industry_enabled,
            nace_path=Path(args.industry_nace_path)
            if args.industry_nace_path
            else None,
            fallback_path=Path(args.industry_fallback_path)
            if args.industry_fallback_path
            else None,
        )
        return load_companies_from_source_with_industry(
            Path(args.company_file),
            industry_config=cfg,
            encoding=str(args.source_encoding),
            limit=args.source_limit,
            aggregate_same_url=not bool(args.source_no_aggregate_same_url),
            interleave_domains=not bool(args.source_no_interleave_domains),
        )

    url = args.url
    assert url is not None
    cid = args.company_id
    if not cid:
        parsed = urlparse(url)
        cid = (parsed.netloc or parsed.path or "company").replace(":", "_")
    return [Company.from_input(company_id=cid, root_url=url, name=None, metadata={})]


async def prepare_run_and_state(
    *,
    state: CrawlState,
    args: argparse.Namespace,
    companies: List[Company],
    cache_dir: Optional[str],
    run_name: str = "deep_crawl",
) -> Tuple[str, List[Company], Dict[str, Company], List[str], set[str]]:
    prev_run_max_pages = await state.get_latest_run_max_pages()
    current_max_pages = int(args.max_pages)

    run_id = await state.start_run(
        run_name,
        version=None,
        args_hash=f"max_pages={current_max_pages}",
        crawl4ai_cache_base_dir=str(Path(cache_dir).expanduser().resolve())
        if cache_dir
        else None,
        crawl4ai_cache_mode=str(args.crawl4ai_cache_mode),
    )

    for c in companies:
        await state.upsert_company(
            c.company_id,
            name=c.name,
            root_url=c.domain_url,
            industry_label=c.industry_label,
            industry=c.industry,
            nace=c.nace,
            industry_source=c.industry_source,
            write_meta=True,
        )

    _ = int(
        await apply_recrawl_policy(
            state=state,
            companies=companies,
            current_max_pages=current_max_pages,
            prev_run_max_pages=prev_run_max_pages,
            force_full_recrawl=bool(args.force_recrawl),
        )
    )

    await state.recompute_all_in_progress(concurrency=32)

    if bool(args.finalize_in_progress_md):
        inprog = set(await state.get_in_progress_company_ids(limit=1_000_000))
        companies = [c for c in companies if c.company_id in inprog]

    companies_by_id: Dict[str, Company] = {c.company_id: c for c in companies}
    company_ids_all: List[str] = [c.company_id for c in companies]
    company_id_set: set[str] = set(company_ids_all)

    return run_id, companies, companies_by_id, company_ids_all, company_id_set


def build_dataset_externals_for_companies(
    *, args: argparse.Namespace, companies: List[Company]
) -> frozenset[str]:
    return build_dataset_externals(args=args, companies=companies)


def build_scoring_and_strategy(
    *, args: argparse.Namespace
) -> Tuple[
    Optional[DualBM25Scorer],
    Optional[DualBM25Filter],
    Optional[DeepCrawlStrategyFactoryType],
]:
    bm25 = build_dual_bm25_components()
    url_scorer: Optional[DualBM25Scorer] = bm25["url_scorer"]
    bm25_filter: Optional[DualBM25Filter] = bm25["url_filter"]

    dfs_factory: Optional[DeepCrawlStrategyFactoryType] = None
    if args.strategy == "dfs":
        dfs_factory = DeepCrawlStrategyFactory(
            provider=DFSDeepCrawlStrategyProvider(default_max_depth=3)
        )

    return url_scorer, bm25_filter, dfs_factory
