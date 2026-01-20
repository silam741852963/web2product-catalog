from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from extensions.crawl.state import CrawlState
from extensions.io.output_paths import ensure_output_root
from extensions.utils.versioning import safe_version_metadata

from .crawl_meta import patch_crawl_meta_file
from .db_migration import rebuild_db_to_current_schema
from .sampling import pick_sample_company_id, sample
from .source_enrichment import filter_source_map_to_db, load_source_company_map
from .types import CalibrationReport, CalibrationSample
from .url_index import normalize_url_index_file


async def calibrate_async(
    *,
    out_dir: Path,
    db_path: Path,
    sample_company_id: Optional[str] = None,
    write_global_state: bool = True,
    concurrency: int = 32,
    dataset_file: Optional[Path] = None,
    company_file: Optional[Path] = None,
    industry_nace_path: Optional[Path] = None,
    industry_fallback_path: Optional[Path] = None,
) -> CalibrationReport:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    rebuild_db_to_current_schema(actual_db_path)

    version_meta = safe_version_metadata(
        component="state_calibration", start_path=Path(__file__)
    )
    if not isinstance(version_meta, dict):
        version_meta = {
            "component": "state_calibration",
            "available": False,
            "reason": "unavailable",
        }

    state = CrawlState(db_path=actual_db_path)
    try:
        rows = await state._query_all("SELECT company_id FROM companies", tuple())
        db_ids = [str(r["company_id"]) for r in rows]

        src_map, src_loaded_rows = load_source_company_map(
            dataset_file=dataset_file,
            company_file=company_file,
            industry_nace_path=industry_nace_path,
            industry_fallback_path=industry_fallback_path,
        )
        src_map = filter_source_map_to_db(db_company_ids=db_ids, src_map=src_map)
        src_used = int(len(src_map))

        company_id = pick_sample_company_id(state, sample_company_id)
        sample_before = await sample(out_dir, state, company_id)

        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def _one(cid: str) -> None:
            async with sem:
                src = src_map.get(cid)

                if src is not None:
                    await state.upsert_company(
                        cid,
                        root_url=src.root_url,
                        name=src.name,
                        metadata={},
                        industry=src.industry,
                        nace=src.nace,
                        industry_label=src.industry_label,
                        industry_label_source=src.industry_label_source,
                    )

                db_snap = await state.get_company_snapshot(cid, recompute=False)

                await asyncio.to_thread(
                    normalize_url_index_file,
                    out_dir,
                    cid,
                    version_meta=version_meta,
                )
                await asyncio.to_thread(
                    patch_crawl_meta_file,
                    out_dir,
                    cid,
                    db_snap=db_snap,
                    version_meta=version_meta,
                )

        batch = max(64, int(concurrency) * 8)
        for i in range(0, len(db_ids), batch):
            await asyncio.gather(*(_one(cid) for cid in db_ids[i : i + batch]))

        if write_global_state:
            await state.write_global_state_from_db_only(pretty=False)

        sample_after = await sample(out_dir, state, company_id)

        return CalibrationReport(
            out_dir=str(out_dir),
            db_path=str(state.db_path),
            touched_companies=int(len(db_ids)),
            wrote_global_state=bool(write_global_state),
            source_companies_loaded=int(src_loaded_rows),
            source_companies_used=int(src_used),
            sample_before=sample_before,
            sample_after=sample_after,
        )
    finally:
        state.close()


async def check_async(
    *,
    out_dir: Path,
    db_path: Path,
    sample_company_id: Optional[str] = None,
) -> CalibrationSample:
    out_dir = ensure_output_root(str(out_dir))
    actual_db_path = Path(db_path).expanduser().resolve()
    rebuild_db_to_current_schema(actual_db_path)

    state = CrawlState(db_path=actual_db_path)
    try:
        cid = pick_sample_company_id(state, sample_company_id)
        return await sample(out_dir, state, cid)
    finally:
        state.close()


def calibrate(
    *,
    out_dir: Path,
    db_path: Path,
    sample_company_id: Optional[str] = None,
    write_global_state: bool = True,
    concurrency: int = 32,
    dataset_file: Optional[Path] = None,
    company_file: Optional[Path] = None,
    industry_nace_path: Optional[Path] = None,
    industry_fallback_path: Optional[Path] = None,
) -> CalibrationReport:
    return asyncio.run(
        calibrate_async(
            out_dir=out_dir,
            db_path=db_path,
            sample_company_id=sample_company_id,
            write_global_state=write_global_state,
            concurrency=concurrency,
            dataset_file=dataset_file,
            company_file=company_file,
            industry_nace_path=industry_nace_path,
            industry_fallback_path=industry_fallback_path,
        )
    )


def check(
    *,
    out_dir: Path,
    db_path: Path,
    sample_company_id: Optional[str] = None,
) -> CalibrationSample:
    return asyncio.run(
        check_async(
            out_dir=out_dir,
            db_path=db_path,
            sample_company_id=sample_company_id,
        )
    )
