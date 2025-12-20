from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("embedding_backend")


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / ((na**0.5) * (nb**0.5)))


@dataclass
class EmbedResult:
    vectors: List[List[float]]
    dim: int
    model: str
    backend: str
    elapsed_ms: float


class Embedder:
    """
    Multilingual embedder with per-run cache.
    Prefers FlagEmbedding BGEM3FlagModel (best for BAAI/bge-m3),
    falls back to sentence-transformers.

    device: "cpu" | "cuda" | "mps"
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = (device or "cpu").lower()
        self._cache: Dict[str, List[float]] = {}
        self._backend: str = "unknown"
        self._model = None
        self._dim: int = 0
        self._init_backend()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dim(self) -> int:
        return self._dim

    def _init_backend(self) -> None:
        # 1) FlagEmbedding (best for bge-m3)
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            self._backend = "FlagEmbedding"
            use_fp16 = self.device == "cuda"
            logger.info(
                "[embed] backend=%s model=%s device=%s fp16=%s",
                self._backend,
                self.model_name,
                self.device,
                use_fp16,
            )

            self._model = BGEM3FlagModel(
                self.model_name,
                device=self.device,
                use_fp16=use_fp16,
            )

            test = self._model.encode(["test"], batch_size=1, max_length=128)  # type: ignore
            vec = test["dense_vecs"][0]
            self._dim = len(vec)
            return
        except Exception as e:
            logger.debug("[embed] FlagEmbedding init failed: %s", e)

        # 2) sentence-transformers fallback
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._backend = "sentence-transformers"
            logger.info(
                "[embed] backend=%s model=%s device=%s",
                self._backend,
                self.model_name,
                self.device,
            )

            self._model = SentenceTransformer(self.model_name, device=self.device)  # type: ignore
            vec = self._model.encode(["test"], normalize_embeddings=False)[0]  # type: ignore
            self._dim = len(vec)
            return
        except Exception as e:
            logger.debug("[embed] sentence-transformers init failed: %s", e)

        raise RuntimeError(
            "No embedding backend available. Install one:\n"
            "  pip install FlagEmbedding\n"
            "or\n"
            "  pip install sentence-transformers\n"
            f"Model requested: {self.model_name}"
        )

    def embed_texts(self, texts: List[str], *, normalize: bool = True) -> EmbedResult:
        t0 = time.perf_counter()

        # Fill from cache / track misses
        out: List[List[float]] = []
        miss_texts: List[str] = []
        miss_positions: List[int] = []

        for i, tx in enumerate(texts):
            tx = tx or ""
            if tx in self._cache:
                out.append(self._cache[tx])
            else:
                out.append([])
                miss_texts.append(tx)
                miss_positions.append(i)

        if not miss_texts:
            return EmbedResult(
                vectors=out,
                dim=self._dim,
                model=self.model_name,
                backend=self._backend,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Encode misses
        if self._backend == "FlagEmbedding":
            enc = self._model.encode(miss_texts, batch_size=16, max_length=8192)  # type: ignore
            vecs = [list(map(float, v)) for v in enc["dense_vecs"]]
        else:
            vecs = self._model.encode(miss_texts, normalize_embeddings=False)  # type: ignore
            vecs = [list(map(float, v)) for v in vecs]

        if normalize:
            vecs = [self._l2norm(v) for v in vecs]

        # Fill cache + output
        for pos, tx, v in zip(miss_positions, miss_texts, vecs):
            self._cache[tx] = v
            out[pos] = v

        return EmbedResult(
            vectors=out,
            dim=self._dim,
            model=self.model_name,
            backend=self._backend,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _l2norm(self, v: List[float]) -> List[float]:
        s = 0.0
        for x in v:
            s += x * x
        if s <= 0.0:
            return v
        inv = 1.0 / (s**0.5)
        return [x * inv for x in v]

    def persist_cache_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for k, v in self._cache.items():
                f.write(json.dumps({"text": k, "vec": v}, ensure_ascii=False) + "\n")
