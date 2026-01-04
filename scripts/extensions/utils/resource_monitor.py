from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import psutil  # type: ignore

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass
class ResourceMonitorConfig:
    """
    Configuration for ResourceMonitor.

    interval_sec: sampling interval in seconds.
    """

    interval_sec: float = 2.0


class ResourceMonitor:
    """
    Lightweight resource monitor for the entire run.

    - Samples system & process CPU usage.
    - Samples system & process memory usage.
    - Tracks global network I/O deltas.
    - Tracks process disk I/O deltas.
    - Writes a single JSON summary at the end.
    """

    def __init__(
        self,
        output_path: Path,
        config: Optional[ResourceMonitorConfig] = None,
    ) -> None:
        self.output_path = output_path
        self.config = config or ResourceMonitorConfig()
        self.interval_sec = max(float(self.config.interval_sec), 0.5)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._started = False

        self._process: Optional[Any] = None

        # Timing
        self._start_ts_iso: Optional[str] = None
        self._end_ts_iso: Optional[str] = None
        self._start_monotonic: Optional[float] = None
        self._end_monotonic: Optional[float] = None

        # Sampling stats
        self._sample_count: int = 0
        self._sample_errors: int = 0

        # CPU metrics (percent)
        self._cpu_system_avg: float = 0.0
        self._cpu_system_peak: float = 0.0
        self._cpu_process_avg: float = 0.0
        self._cpu_process_peak: float = 0.0
        self._cpu_per_cpu_avg: Optional[list[float]] = None
        self._cpu_per_cpu_peak: Optional[list[float]] = None

        # Memory metrics
        self._mem_system_used_percent_avg: float = 0.0
        self._mem_system_used_percent_peak: float = 0.0
        self._mem_proc_rss_avg: float = 0.0
        self._mem_proc_rss_peak: float = 0.0
        self._mem_proc_vms_avg: float = 0.0
        self._mem_proc_vms_peak: float = 0.0

        # Threads
        self._threads_avg: float = 0.0
        self._threads_peak: int = 0

        # Network (system-wide)
        self._net_start: Optional[Tuple[int, int]] = None  # (sent, recv)
        self._net_last: Optional[Tuple[int, int]] = None

        # IO (per-process, if available)
        self._io_start: Optional[Tuple[int, int]] = None  # (read_bytes, write_bytes)
        self._io_last: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """
        Start the background sampling thread.
        """
        if self._started:
            return

        self._started = True
        now = datetime.now(timezone.utc)
        self._start_ts_iso = now.isoformat()
        self._start_monotonic = time.monotonic()

        try:
            self._process = psutil.Process(os.getpid())  # type: ignore[union-attr]
            # Warm up CPU percent metrics to avoid weird initial 0.0
            self._process.cpu_percent(interval=None)  # type: ignore[union-attr]
            psutil.cpu_percent(interval=None, percpu=True)  # type: ignore[union-attr]
        except Exception as e:  # pragma: no cover
            logger.warning(
                "[ResourceMonitor] Failed to initialize psutil Process: %s", e
            )
            self._process = None

        try:
            net = psutil.net_io_counters()  # type: ignore[union-attr]
            self._net_start = (net.bytes_sent, net.bytes_recv)
            self._net_last = self._net_start
        except Exception:
            self._net_start = None
            self._net_last = None

        try:
            if self._process is not None:
                io = self._process.io_counters()  # type: ignore[union-attr]
                self._io_start = (io.read_bytes, io.write_bytes)
                self._io_last = self._io_start
        except Exception:
            self._io_start = None
            self._io_last = None

        # Background thread (runs even if psutil is missing; samples will be no-ops)
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ResourceMonitorThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[ResourceMonitor] started (interval=%.2fs)",
            self.interval_sec,
        )

    def stop(self) -> None:
        """
        Stop the background sampling thread and write the JSON summary.
        """
        if not self._started:
            return

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        self._end_monotonic = time.monotonic()
        now = datetime.now(timezone.utc)
        self._end_ts_iso = now.isoformat()

        # Take one last sample if psutil is available (best-effort)
        try:
            self._take_sample()
        except Exception:
            pass

        try:
            summary = self._build_summary()
            self._write_summary(summary)
        except Exception:
            logger.exception("[ResourceMonitor] Failed to write summary JSON")

        logger.info(
            "[ResourceMonitor] stopped (samples=%d, errors=%d)",
            self._sample_count,
            self._sample_errors,
        )

    # ------------------------------------------------------------------ #
    # Internal loop
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """
        Background sampling loop. Uses Event.wait(interval) so stop() is
        reactive without busy-waiting.
        """

        # Main sampling loop
        while not self._stop_event.wait(self.interval_sec):
            try:
                self._take_sample()
            except Exception as e:  # pragma: no cover
                self._sample_errors += 1
                logger.debug(
                    "[ResourceMonitor] sampling error: %s",
                    e,
                    exc_info=True,
                )

    def _take_sample(self) -> None:
        # If we have no psutil process, sampling is effectively disabled
        if self._process is None:
            return

        with self._lock:
            self._sample_count += 1
            n = self._sample_count

            # CPU (system & per-CPU)
            try:
                per_cpu = psutil.cpu_percent(  # type: ignore[union-attr]
                    interval=None,
                    percpu=True,
                )
            except Exception:
                per_cpu = []

            if per_cpu:
                cpu_system = float(sum(per_cpu) / len(per_cpu))
                self._cpu_system_avg += (cpu_system - self._cpu_system_avg) / n
                if cpu_system > self._cpu_system_peak:
                    self._cpu_system_peak = cpu_system

                if self._cpu_per_cpu_avg is None:
                    self._cpu_per_cpu_avg = [0.0] * len(per_cpu)
                    self._cpu_per_cpu_peak = [0.0] * len(per_cpu)

                for i, val in enumerate(per_cpu):
                    if i >= len(self._cpu_per_cpu_avg):
                        # Defensive – resize if CPU count changed (unlikely)
                        self._cpu_per_cpu_avg.append(0.0)
                        self._cpu_per_cpu_peak.append(0.0)
                    avg_i = self._cpu_per_cpu_avg[i]
                    self._cpu_per_cpu_avg[i] = avg_i + (val - avg_i) / n
                    if val > self._cpu_per_cpu_peak[i]:
                        self._cpu_per_cpu_peak[i] = val

            # CPU (process)
            try:
                proc_cpu = float(self._process.cpu_percent(interval=None))  # type: ignore[union-attr]
            except Exception:
                proc_cpu = 0.0
            self._cpu_process_avg += (proc_cpu - self._cpu_process_avg) / n
            if proc_cpu > self._cpu_process_peak:
                self._cpu_process_peak = proc_cpu

            # Memory (system)
            try:
                vm = psutil.virtual_memory()  # type: ignore[union-attr]
                used_pct = float(vm.percent)
                self._mem_system_used_percent_avg += (
                    used_pct - self._mem_system_used_percent_avg
                ) / n
                if used_pct > self._mem_system_used_percent_peak:
                    self._mem_system_used_percent_peak = used_pct
            except Exception:
                pass

            # Memory (process)
            try:
                pmem = self._process.memory_info()  # type: ignore[union-attr]
                rss = float(pmem.rss)
                vms = float(pmem.vms)
                self._mem_proc_rss_avg += (rss - self._mem_proc_rss_avg) / n
                if rss > self._mem_proc_rss_peak:
                    self._mem_proc_rss_peak = rss
                self._mem_proc_vms_avg += (vms - self._mem_proc_vms_avg) / n
                if vms > self._mem_proc_vms_peak:
                    self._mem_proc_vms_peak = vms
            except Exception:
                pass

            # Threads
            try:
                threads = int(self._process.num_threads())  # type: ignore[union-attr]
                self._threads_avg += (threads - self._threads_avg) / n
                if threads > self._threads_peak:
                    self._threads_peak = threads
            except Exception:
                pass

            try:
                net = psutil.net_io_counters()  # type: ignore[union-attr]
                self._net_last = (net.bytes_sent, net.bytes_recv)
            except Exception:
                pass

            # Process IO
            try:
                io = self._process.io_counters()  # type: ignore[union-attr]
                self._io_last = (io.read_bytes, io.write_bytes)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Summary building & writing
    # ------------------------------------------------------------------ #

    def _build_host_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            },
            "python": {
                "implementation": platform.python_implementation(),
                "version": sys.version.split()[0],
            },
        }

        try:
            info["cpu"] = {
                "logical_cores": psutil.cpu_count(logical=True),  # type: ignore[union-attr]
                "physical_cores": psutil.cpu_count(logical=False),  # type: ignore[union-attr]
            }
            try:
                freq = psutil.cpu_freq()  # type: ignore[union-attr]
            except Exception:
                freq = None
            if freq:
                info["cpu"]["max_freq_mhz"] = freq.max
                info["cpu"]["min_freq_mhz"] = freq.min
        except Exception:
            pass

        try:
            vm = psutil.virtual_memory()  # type: ignore[union-attr]
            info["memory"] = {
                "total_gb": round(vm.total / (1024**3), 2),
            }
        except Exception:
            pass

        # Load average (if available)
        try:
            load1, load5, load15 = psutil.getloadavg()  # type: ignore[attr-defined]
            info["load_average"] = {
                "1m": load1,
                "5m": load5,
                "15m": load15,
            }
        except Exception:
            pass

        return info

    def _build_process_info(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "pid": os.getpid(),
            "cmdline": sys.argv,
        }
        if self._process is not None:
            try:
                d["name"] = self._process.name()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                d["exe"] = self._process.exe()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                ct = self._process.create_time()  # type: ignore[union-attr]
                d["create_time"] = datetime.fromtimestamp(
                    ct, tz=timezone.utc
                ).isoformat()
            except Exception:
                pass
        return d

    def _build_summary(self) -> Dict[str, Any]:
        duration_sec: Optional[float] = None
        if self._start_monotonic is not None and self._end_monotonic is not None:
            duration_sec = max(0.0, self._end_monotonic - self._start_monotonic)

        samples = self._sample_count or 0

        # CPU section
        cpu: Dict[str, Any] = {}
        if samples > 0:
            cpu["samples"] = samples
            cpu["system_percent_avg"] = round(self._cpu_system_avg, 2)
            cpu["system_percent_peak"] = round(self._cpu_system_peak, 2)
            cpu["process_percent_avg"] = round(self._cpu_process_avg, 2)
            cpu["process_percent_peak"] = round(self._cpu_process_peak, 2)
            if self._cpu_per_cpu_avg is not None:
                cpu["per_cpu_percent_avg"] = [
                    round(v, 2) for v in self._cpu_per_cpu_avg
                ]
            if self._cpu_per_cpu_peak is not None:
                cpu["per_cpu_percent_peak"] = [
                    round(v, 2) for v in self._cpu_per_cpu_peak
                ]

        # Memory section
        memory: Dict[str, Any] = {}
        if samples > 0:
            memory["samples"] = samples
            memory["system_used_percent_avg"] = round(
                self._mem_system_used_percent_avg, 2
            )
            memory["system_used_percent_peak"] = round(
                self._mem_system_used_percent_peak, 2
            )
            memory["process_rss_avg_mb"] = round(self._mem_proc_rss_avg / (1024**2), 2)
            memory["process_rss_peak_mb"] = round(
                self._mem_proc_rss_peak / (1024**2), 2
            )
            memory["process_vms_avg_mb"] = round(self._mem_proc_vms_avg / (1024**2), 2)
            memory["process_vms_peak_mb"] = round(
                self._mem_proc_vms_peak / (1024**2), 2
            )

        # Threads section
        threads: Dict[str, Any] = {}
        if samples > 0:
            threads["samples"] = samples
            threads["process_threads_avg"] = round(self._threads_avg, 2)
            threads["process_threads_peak"] = self._threads_peak

        # Network section (system-wide)
        network: Dict[str, Any] = {}
        if self._net_start is not None and self._net_last is not None:
            sent_start, recv_start = self._net_start
            sent_end, recv_end = self._net_last
            delta_sent = max(0, sent_end - sent_start)
            delta_recv = max(0, recv_end - recv_start)
            network = {
                "bytes_sent_delta": delta_sent,
                "bytes_recv_delta": delta_recv,
                "mb_sent_delta": round(delta_sent / (1024**2), 2),
                "mb_recv_delta": round(delta_recv / (1024**2), 2),
            }

        # IO section (per-process)
        io: Dict[str, Any] = {}
        if self._io_start is not None and self._io_last is not None:
            r_start, w_start = self._io_start
            r_end, w_end = self._io_last
            delta_r = max(0, r_end - r_start)
            delta_w = max(0, w_end - w_start)
            io = {
                "read_bytes_delta": delta_r,
                "write_bytes_delta": delta_w,
                "read_mb_delta": round(delta_r / (1024**2), 2),
                "write_mb_delta": round(delta_w / (1024**2), 2),
            }

        summary: Dict[str, Any] = {
            "started_at": self._start_ts_iso,
            "ended_at": self._end_ts_iso,
            "duration_sec": duration_sec,
            "sampling_interval_sec": self.interval_sec,
            "samples": samples,
            "sample_errors": self._sample_errors,
            "host": self._build_host_info(),
            "process": self._build_process_info(),
            "cpu": cpu,
            "memory": memory,
            "threads": threads,
            "network": network,
            "io": io,
        }
        return summary

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            try:
                f.flush()
            except Exception:
                pass
        tmp_path.replace(self.output_path)


__all__ = [
    "ResourceMonitor",
    "ResourceMonitorConfig",
]

