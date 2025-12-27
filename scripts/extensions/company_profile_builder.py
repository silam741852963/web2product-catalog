from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from extensions.embedding_backend import Embedder, cosine
from extensions.text_norm import (
    descriptiveness_score,
    evidence_probe,
    is_denylisted,
    norm_desc,
    norm_name,
    norm_type,
    split_sentences,
)

logger = logging.getLogger("company_profile_builder")

PIPELINE_VERSION = "profile-v1"
EMBED_MODEL = "BAAI/bge-m3"


@dataclass
class PageRecord:
    url: str
    product_json_path: Path
    markdown_path: Optional[Path]
    root_url: Optional[str] = None
    company_domain: Optional[str] = None


@dataclass
class Mention:
    type_raw: str
    raw_name: str
    raw_description: str
    source_url: str
    product_json_path: Path
    markdown_path: Optional[Path]

    type_norm: str = ""
    name_norm: str = ""
    desc_norm: str = ""


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        return True


def _setup_company_logger(company_dir: Path) -> None:
    company_dir.mkdir(parents=True, exist_ok=True)
    log_dir = company_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "company_profile_build.log"

    lg = logging.getLogger("company_profile_builder")
    lg.setLevel(logging.DEBUG)

    for h in list(lg.handlers):
        lg.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    lg.addHandler(fh)
    lg.addHandler(ch)
    lg.propagate = False

    lg.info("pipeline=%s start", PIPELINE_VERSION)
    lg.info("log_path=%s", str(log_path))


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _domain_from_root(root_url: Optional[str]) -> Optional[str]:
    if not root_url:
        return None
    s = root_url.replace("https://", "").replace("http://", "")
    s = s.split("/")[0]
    return s or None


def _resolve_maybe_relative(base: Path, p: str) -> Path:
    """
    url_index.json often stores:
      - absolute paths (older runs)
      - or relative paths like "product/xxx.json" / "markdown/xxx.md"
    We accept both.
    """
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def _load_inputs(
    outputs_dir: Path, company_id: str
) -> Tuple[Path, List[PageRecord], Dict[str, Any]]:
    company_dir = outputs_dir / company_id
    meta_dir = company_dir / "metadata"
    prod_dir = company_dir / "product"

    crawl_meta_path = meta_dir / "crawl_meta.json"
    url_index_path = meta_dir / "url_index.json"

    crawl_meta = _load_json(crawl_meta_path) if crawl_meta_path.exists() else None
    url_index = _load_json(url_index_path) if url_index_path.exists() else None

    crawl_meta = crawl_meta if isinstance(crawl_meta, dict) else {}
    url_index = url_index if isinstance(url_index, dict) else {}

    root_url = crawl_meta.get("root_url")
    domain = _domain_from_root(root_url)

    product_files = sorted(prod_dir.glob("*.json"))
    logger.info("step1: product_files=%d", len(product_files))
    logger.info(
        "step1: crawl_meta_loaded=%s url_index_loaded=%s",
        bool(crawl_meta),
        bool(url_index),
    )

    # Primary linking: via url_index[*].product_path (absolute or relative)
    prod_map: Dict[str, Tuple[str, Optional[str]]] = {}
    for url, ent in url_index.items():
        if not isinstance(ent, dict):
            continue
        pp = ent.get("product_path")
        mp = ent.get("markdown_path")
        if isinstance(pp, str):
            resolved_pp = _resolve_maybe_relative(company_dir, pp)
            prod_map[resolved_pp.as_posix()] = (
                str(url),
                mp if isinstance(mp, str) else None,
            )

    records: List[PageRecord] = []
    unlinked: List[Path] = []

    for pf in product_files:
        url: Optional[str] = None
        md: Optional[str] = None
        key = pf.resolve().as_posix()

        if key in prod_map:
            url, md = prod_map[key]
        else:
            # Fallback: stem match .json <-> .md
            stem = pf.stem
            md_guess = company_dir / "markdown" / f"{stem}.md"
            if md_guess.exists():
                md = md_guess.as_posix()

            for u, ent in url_index.items():
                if isinstance(ent, dict):
                    pp = ent.get("product_path")
                    if isinstance(pp, str):
                        # Compare by filename too, in case stored as "product/x.json" etc
                        try:
                            if Path(pp).name == pf.name:
                                url = str(u)
                                if md is None:
                                    mp = ent.get("markdown_path")
                                    if isinstance(mp, str):
                                        md = mp
                                break
                        except Exception:
                            pass

        if url is None:
            unlinked.append(pf)
            continue

        md_path: Optional[Path] = None
        if md:
            md_path = _resolve_maybe_relative(company_dir, md)
            if not md_path.exists():
                logger.warning(
                    "step1: markdown missing for url=%s md_path=%s", url, md_path
                )

        records.append(
            PageRecord(
                url=url,
                product_json_path=pf.resolve(),
                markdown_path=md_path,
                root_url=str(root_url) if root_url else None,
                company_domain=domain,
            )
        )

    logger.info("step1: linked_records=%d", len(records))
    if unlinked:
        logger.warning(
            "step1: unlinked_product_files=%d (showing up to 20)", len(unlinked)
        )
        for x in unlinked[:20]:
            logger.warning("step1: unlinked=%s", x)

    return company_dir, records, crawl_meta


def _flatten_mentions(records: List[PageRecord]) -> List[Mention]:
    mentions: List[Mention] = []
    bad = 0
    pages_with = 0

    for r in records:
        obj = _load_json(r.product_json_path)
        if not isinstance(obj, dict):
            bad += 1
            continue
        offs = obj.get("offerings")
        if not isinstance(offs, list):
            continue

        count = 0
        for o in offs:
            if not isinstance(o, dict):
                continue
            mentions.append(
                Mention(
                    type_raw=str(o.get("type") or ""),
                    raw_name=str(o.get("name") or ""),
                    raw_description=str(o.get("description") or ""),
                    source_url=r.url,
                    product_json_path=r.product_json_path,
                    markdown_path=r.markdown_path,
                )
            )
            count += 1

        if count:
            pages_with += 1
            logger.debug("step2: page=%s mentions=%d", r.url, count)

    logger.info(
        "step2: mentions=%d pages_with_mentions=%d malformed_product_json=%d",
        len(mentions),
        pages_with,
        bad,
    )
    return mentions


def _load_markdown_cached(md_path: Optional[Path], cache: Dict[str, str]) -> str:
    if md_path is None:
        return ""
    k = md_path.as_posix()
    if k in cache:
        return cache[k]
    try:
        txt = md_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = ""
    cache[k] = txt
    return txt


def _normalize_filter_mentions(
    mentions: List[Mention], *, min_name_len: int = 3
) -> List[Mention]:
    md_cache: Dict[str, str] = {}

    dropped_empty = 0
    dropped_short = 0
    dropped_deny = 0
    dropped_unsupported = 0

    clean: List[Mention] = []
    for m in mentions:
        m.type_norm = norm_type(m.type_raw)
        m.name_norm = norm_name(m.raw_name)
        m.desc_norm = norm_desc(m.raw_description)

        if not m.name_norm:
            dropped_empty += 1
            continue
        if len(m.name_norm) < min_name_len:
            dropped_short += 1
            continue
        if is_denylisted(m.raw_name) or is_denylisted(m.raw_description):
            dropped_deny += 1
            continue

        md_body = _load_markdown_cached(m.markdown_path, md_cache)
        if not evidence_probe(md_body, m.raw_name, m.raw_description):
            dropped_unsupported += 1
            continue

        clean.append(m)

    total = len(mentions)
    logger.info(
        "step3: in=%d out=%d dropped_empty=%d dropped_short=%d dropped_deny=%d dropped_unsupported=%d supported_rate=%.1f%%",
        total,
        len(clean),
        dropped_empty,
        dropped_short,
        dropped_deny,
        dropped_unsupported,
        (100.0 * len(clean) / max(total, 1)),
    )
    return clean


def _cluster_mentions(
    mentions: List[Mention],
    *,
    embedder: Embedder,
    name_merge_threshold: float,
) -> Dict[str, List[List[Mention]]]:
    by_type: Dict[str, List[Mention]] = {"product": [], "service": []}
    for m in mentions:
        by_type.setdefault(m.type_norm, []).append(m)

    clusters_by_type: Dict[str, List[List[Mention]]] = {}

    for typ, items in by_type.items():
        if not items:
            clusters_by_type[typ] = []
            continue

        blocks: Dict[str, List[int]] = {}
        for i, m in enumerate(items):
            blocks.setdefault(m.name_norm, []).append(i)

        uf = UnionFind(len(items))

        for idxs in blocks.values():
            if len(idxs) >= 2:
                base = idxs[0]
                for j in idxs[1:]:
                    uf.union(base, j)

        unique_names = sorted(blocks.keys())
        er = embedder.embed_texts(unique_names, normalize=True)
        vecs = er.vectors

        logger.info(
            "step4[%s]: mentions=%d unique_names=%d dim=%d backend=%s elapsed_ms=%.1f threshold=%.3f",
            typ,
            len(items),
            len(unique_names),
            er.dim,
            er.backend,
            er.elapsed_ms,
            name_merge_threshold,
        )

        merges = 0
        comps = 0
        for i in range(len(unique_names)):
            vi = vecs[i]
            for j in range(i + 1, len(unique_names)):
                comps += 1
                if cosine(vi, vecs[j]) >= name_merge_threshold:
                    ni = unique_names[i]
                    nj = unique_names[j]
                    ai = blocks[ni][0]
                    aj = blocks[nj][0]
                    if uf.union(ai, aj):
                        merges += 1

        logger.info(
            "step4[%s]: name_cosine_comparisons=%d merges=%d", typ, comps, merges
        )

        group: Dict[int, List[Mention]] = {}
        for i, m in enumerate(items):
            root = uf.find(i)
            group.setdefault(root, []).append(m)

        clusters = list(group.values())
        clusters_by_type[typ] = clusters
        logger.debug(
            "step4[%s]: clusters=%d sizes_top10=%s",
            typ,
            len(clusters),
            sorted([len(c) for c in clusters], reverse=True)[:10],
        )

    return clusters_by_type


def _stable_offering_id(company_id: str, typ: str, best_name_norm: str) -> str:
    h = hashlib.sha1(
        f"{company_id}|{typ}|{best_name_norm}".encode("utf-8", errors="ignore")
    ).hexdigest()
    return f"off_{h[:16]}"


def _pick_best_name(names: List[str]) -> Tuple[str, List[str]]:
    uniq: Dict[str, str] = {}
    for n in names:
        key = norm_name(n)
        if key and key not in uniq:
            uniq[key] = n.strip()

    variants = list(uniq.values())
    variants.sort(key=lambda x: descriptiveness_score(x), reverse=True)

    best = variants[0] if variants else ""
    others = [x for x in variants[1:] if x != best]
    return best, others


def _dedup_descriptions(
    descs: List[str],
    *,
    embedder: Embedder,
    sent_merge_threshold: float,
    max_desc_sentences: int,
) -> List[str]:
    sents: List[str] = []
    for d in descs:
        sents.extend(split_sentences(d))

    normed: List[str] = []
    seen = set()
    for s in sents:
        ss = norm_desc(s)
        if not ss:
            continue
        k = ss.lower()
        if k not in seen:
            seen.add(k)
            normed.append(ss)

    if not normed:
        return []
    if len(normed) == 1:
        return normed

    er = embedder.embed_texts(normed, normalize=True)
    vecs = er.vectors

    kept: List[str] = []
    kept_vecs: List[List[float]] = []
    for s, v in zip(normed, vecs):
        redundant = False
        for kv in kept_vecs:
            if cosine(v, kv) >= sent_merge_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(s)
            kept_vecs.append(v)
        if len(kept) >= max_desc_sentences:
            break

    return kept


def _merge_clusters_to_offerings(
    company_id: str,
    clusters_by_type: Dict[str, List[List[Mention]]],
    *,
    embedder: Embedder,
    sent_merge_threshold: float,
    max_desc_sentences: int,
) -> List[Dict[str, Any]]:
    offerings: List[Dict[str, Any]] = []

    for typ in ("product", "service"):
        for ci, cluster in enumerate(clusters_by_type.get(typ, [])):
            names = [m.raw_name for m in cluster if m.raw_name]
            descs = [m.raw_description for m in cluster if m.raw_description]

            best, others = _pick_best_name(names)
            best_norm = norm_name(best)
            oid = _stable_offering_id(company_id, typ, best_norm)

            sources_map: Dict[str, Dict[str, str]] = {}
            for m in cluster:
                key = f"{m.source_url}|{m.product_json_path.as_posix()}"
                sources_map[key] = {
                    "url": m.source_url,
                    "product_json_path": m.product_json_path.as_posix(),
                    "markdown_path": m.markdown_path.as_posix()
                    if m.markdown_path
                    else "",
                }

            descriptions = _dedup_descriptions(
                descs,
                embedder=embedder,
                sent_merge_threshold=sent_merge_threshold,
                max_desc_sentences=max_desc_sentences,
            )

            offerings.append(
                {
                    "offering_id": oid,
                    "type": typ,
                    "name": [best, others],
                    "description": descriptions,
                    "sources": list(sources_map.values()),
                }
            )

            logger.debug(
                "step5: typ=%s idx=%d mentions=%d name_variants=%d desc_sents=%d sources=%d offering_id=%s",
                typ,
                ci,
                len(cluster),
                1 + len(others),
                len(descriptions),
                len(sources_map),
                oid,
            )

    logger.info("step5: canonical_offerings=%d", len(offerings))
    return offerings


def _write_company_profile_md(
    company_dir: Path,
    company_id: str,
    root_url: Optional[str],
    offerings: List[Dict[str, Any]],
) -> Path:
    meta_dir = company_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    md_path = meta_dir / "company_profile.md"

    def fmt_offering(o: Dict[str, Any]) -> str:
        best = (o.get("name") or ["", []])[0]
        aliases = (o.get("name") or ["", []])[1] or []
        descs = o.get("description") or []
        sources = o.get("sources") or []
        lines: List[str] = []
        lines.append(f"### {best}")
        if aliases:
            lines.append("")
            lines.append("Aliases: " + "; ".join(aliases))
        if descs:
            lines.append("")
            lines.append("Description:")
            for d in descs:
                lines.append(f"- {d}")
        if sources:
            lines.append("")
            lines.append("Sources:")
            for s in sources[:3]:
                u = s.get("url") or ""
                if u:
                    lines.append(f"- {u}")
        lines.append("")
        return "\n".join(lines)

    products = [o for o in offerings if o.get("type") == "product"]
    services = [o for o in offerings if o.get("type") == "service"]
    products.sort(key=lambda o: (o.get("name") or ["", []])[0].lower())
    services.sort(key=lambda o: (o.get("name") or ["", []])[0].lower())

    parts: List[str] = []
    parts.append(f"# Company Profile — {company_id}")
    if root_url:
        parts.append("")
        parts.append(f"Root URL: {root_url}")
    parts.append("")
    parts.append("## Offerings")

    if products:
        parts.append("")
        parts.append("## Products")
        parts.append("")
        for o in products:
            parts.append(fmt_offering(o))

    if services:
        parts.append("")
        parts.append("## Services")
        parts.append("")
        for o in services:
            parts.append(fmt_offering(o))

    text = "\n".join(parts).strip() + "\n"
    md_path.write_text(text, encoding="utf-8")
    logger.info(
        "step6: wrote_md=%s chars=%d products=%d services=%d",
        md_path,
        len(text),
        len(products),
        len(services),
    )
    return md_path


def _offering_embed_text(o: Dict[str, Any]) -> str:
    best = (o.get("name") or ["", []])[0]
    aliases = (o.get("name") or ["", []])[1] or []
    typ = o.get("type") or ""
    descs = o.get("description") or []
    lines: List[str] = []
    lines.append(f"Type: {typ}")
    lines.append(f"Name: {best}")
    if aliases:
        lines.append("Aliases: " + "; ".join(aliases))
    lines.append("")
    lines.append("Description:")
    for d in descs:
        lines.append(f"- {d}")
    return "\n".join(lines).strip()


def _embed_all(
    company_dir: Path,
    embedder: Embedder,
    md_path: Path,
    offerings: List[Dict[str, Any]],
) -> Tuple[List[float], Dict[str, List[float]]]:
    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    er1 = embedder.embed_texts([md_text], normalize=True)
    company_vec = er1.vectors[0]
    logger.info(
        "step7: embedded_company_profile dim=%d backend=%s elapsed_ms=%.1f",
        er1.dim,
        er1.backend,
        er1.elapsed_ms,
    )

    texts: List[str] = []
    ids: List[str] = []
    for o in offerings:
        oid = str(o.get("offering_id") or "")
        if oid:
            ids.append(oid)
            texts.append(_offering_embed_text(o))

    offering_vecs: Dict[str, List[float]] = {}
    if texts:
        er2 = embedder.embed_texts(texts, normalize=True)
        for oid, vec in zip(ids, er2.vectors):
            offering_vecs[oid] = vec
        logger.info(
            "step7: embedded_offerings=%d dim=%d elapsed_ms=%.1f",
            len(offering_vecs),
            er2.dim,
            er2.elapsed_ms,
        )
    else:
        logger.info("step7: embedded_offerings=0")

    cache_path = company_dir / "metadata" / "embed_cache.jsonl"
    try:
        embedder.persist_cache_jsonl(cache_path)
        logger.info("step7: wrote_embed_cache=%s", cache_path)
    except Exception as e:
        logger.warning("step7: embed_cache_write_failed err=%s", e)

    return company_vec, offering_vecs


def _write_company_profile_json(
    company_dir: Path,
    company_id: str,
    crawl_meta: Dict[str, Any],
    offerings: List[Dict[str, Any]],
    *,
    embedder: Embedder,
    company_vec: List[float],
    offering_vecs: Dict[str, List[float]],
) -> Path:
    out_path = company_dir / "company_profile.json"
    payload: Dict[str, Any] = {
        "company_id": company_id,
        "pipeline_version": PIPELINE_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_url": crawl_meta.get("root_url"),
        "crawl_meta": crawl_meta,
        "offerings": offerings,
        "embeddings": {
            "model": EMBED_MODEL,
            "dim": embedder.dim,
            "embedding_company_profile": company_vec,
            "embedding_offerings": offering_vecs,
        },
    }

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    logger.info(
        "step8: wrote_json=%s offerings=%d size_bytes=%d",
        out_path,
        len(offerings),
        out_path.stat().st_size,
    )
    return out_path


def build_company_profile_for_company(
    *,
    outputs_dir: Path,
    company_id: str,
    embed_device: str = "cpu",
    name_merge_threshold: float = 0.94,
    sent_merge_threshold: float = 0.90,
    max_desc_sentences: int = 8,
) -> bool:
    company_dir = outputs_dir / company_id
    _setup_company_logger(company_dir)

    try:
        logger.info(
            "company_id=%s outputs_dir=%s embed_device=%s",
            company_id,
            outputs_dir,
            embed_device,
        )
        logger.info(
            "step0: detected_paths metadata=%s product=%s markdown=%s",
            company_dir / "metadata",
            company_dir / "product",
            company_dir / "markdown",
        )

        company_dir, records, crawl_meta = _load_inputs(outputs_dir, company_id)
        if not records:
            logger.warning("no linked product records; writing empty profile")
            empty_offerings: List[Dict[str, Any]] = []
            md_path = _write_company_profile_md(
                company_dir, company_id, crawl_meta.get("root_url"), empty_offerings
            )

            embedder = Embedder(EMBED_MODEL, device=embed_device)
            company_vec, offering_vecs = _embed_all(
                company_dir, embedder, md_path, empty_offerings
            )
            _write_company_profile_json(
                company_dir,
                company_id,
                crawl_meta,
                empty_offerings,
                embedder=embedder,
                company_vec=company_vec,
                offering_vecs=offering_vecs,
            )
            return True

        mentions = _flatten_mentions(records)

        clean = _normalize_filter_mentions(mentions)
        if not clean:
            logger.warning(
                "all mentions dropped after filtering; writing empty profile"
            )
            empty_offerings = []
            md_path = _write_company_profile_md(
                company_dir, company_id, crawl_meta.get("root_url"), empty_offerings
            )

            embedder = Embedder(EMBED_MODEL, device=embed_device)
            company_vec, offering_vecs = _embed_all(
                company_dir, embedder, md_path, empty_offerings
            )
            _write_company_profile_json(
                company_dir,
                company_id,
                crawl_meta,
                empty_offerings,
                embedder=embedder,
                company_vec=company_vec,
                offering_vecs=offering_vecs,
            )
            return True

        embedder = Embedder(EMBED_MODEL, device=embed_device)
        logger.info(
            "embedder_ready model=%s backend=%s dim=%d device=%s",
            EMBED_MODEL,
            embedder.backend,
            embedder.dim,
            embed_device,
        )

        clusters_by_type = _cluster_mentions(
            clean, embedder=embedder, name_merge_threshold=name_merge_threshold
        )

        offerings = _merge_clusters_to_offerings(
            company_id,
            clusters_by_type,
            embedder=embedder,
            sent_merge_threshold=sent_merge_threshold,
            max_desc_sentences=max_desc_sentences,
        )

        md_path = _write_company_profile_md(
            company_dir, company_id, crawl_meta.get("root_url"), offerings
        )

        company_vec, offering_vecs = _embed_all(
            company_dir, embedder, md_path, offerings
        )

        _write_company_profile_json(
            company_dir,
            company_id,
            crawl_meta,
            offerings,
            embedder=embedder,
            company_vec=company_vec,
            offering_vecs=offering_vecs,
        )

        logger.info("done company_id=%s", company_id)
        return True

    except Exception as e:
        logger.exception("fatal error company_id=%s err=%s", company_id, e)
        return False
