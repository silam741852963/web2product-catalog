from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional
import psutil

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass(frozen=True, slots=True)
class OOMGuardConfig:
    soft_frac: float = 0.90
    hard_frac: float = 0.95
    check_interval_sec: float = 2.0
    soft_pause_sec: float = 20.0


class OOMGuard:
    """
    Periodically reads psutil.virtual_memory().percent.

    - On soft threshold: extends `pause_until_ts` (caller can read it).
    - On hard threshold: calls on_hard() once and returns (task ends).
    """

    def __init__(
        self,
        *,
        cfg: OOMGuardConfig,
        on_hard: Callable[[float], None],
    ) -> None:
        self._cfg = cfg
        self._on_hard = on_hard
        self._task: Optional[asyncio.Task] = None
        self.pause_until_ts: float = 0.0

    @property
    def enabled(self) -> bool:
        return psutil is not None

    def start(self, stop_event: asyncio.Event) -> None:
        if psutil is None:
            logger.info("[OOMGuard] psutil not available -> disabled")
            return
        if self._task is not None:
            return

        async def _run() -> None:
            try:
                while not stop_event.is_set():
                    used_frac = psutil.virtual_memory().percent / 100.0

                    if used_frac >= self._cfg.hard_frac:
                        logger.critical(
                            "[OOMGuard] used_frac_raw=%.3f >= hard=%.3f -> triggering on_hard",
                            used_frac,
                            self._cfg.hard_frac,
                        )
                        self._on_hard(used_frac)
                        return

                    if used_frac >= self._cfg.soft_frac:
                        self.pause_until_ts = max(
                            self.pause_until_ts,
                            time.time() + float(self._cfg.soft_pause_sec),
                        )

                    await asyncio.sleep(max(0.5, float(self._cfg.check_interval_sec)))
            except asyncio.CancelledError:
                # Normal shutdown path
                return

        self._task = asyncio.create_task(_run(), name="oom_guard")

    async def stop(self) -> None:
        t = self._task
        if t is None:
            return
        self._task = None

        if not t.done():
            t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        except BaseException:
            # Never let OOMGuard shutdown crash the runner
            logger.debug("[OOMGuard] stop(): swallowed task error", exc_info=True)
