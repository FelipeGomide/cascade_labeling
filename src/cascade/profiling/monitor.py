"""Resource profiling for cascade stages: wall time, CPU time/RAM, GPU util/VRAM/energy.

Usage:
    with ResourceMonitor("bm25") as mon:
        ... run stage ...
    stats = mon.stats  # dict, merge into resources.json
"""

import threading
import time

import psutil

try:
    import pynvml

    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ResourceMonitor:
    """Context manager that samples GPU power/util in a background thread while
    a stage runs, and reports CPU/RAM/GPU/time deltas on exit."""

    def __init__(self, stage_name: str, gpu_index: int = 0, sample_interval_s: float = 0.05):
        self.stage_name = stage_name
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_s
        self.stats: dict = {}
        self._stop_flag = False
        self._samples: list[tuple[float, float]] = []  # (timestamp, power_watts)
        self._thread: threading.Thread | None = None
        self._nvml_handle = None

    def __enter__(self):
        self._process = psutil.Process()
        self._cpu_times_start = self._process.cpu_times()
        self._rss_start = self._process.memory_info().rss
        self._wall_start = time.perf_counter()

        if _HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(self.gpu_index)
            except RuntimeError:
                pass

        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                self._stop_flag = False
                self._thread = threading.Thread(target=self._sample_loop, daemon=True)
                self._thread.start()
            except pynvml.NVMLError:
                self._nvml_handle = None

        return self

    def _sample_loop(self):
        while not self._stop_flag:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                self._samples.append((time.perf_counter(), power_mw / 1000.0, util.gpu))
            except pynvml.NVMLError:
                pass
            time.sleep(self.sample_interval_s)

    def __exit__(self, exc_type, exc_val, exc_tb):
        wall_time_s = time.perf_counter() - self._wall_start

        self._stop_flag = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        cpu_times_end = self._process.cpu_times()
        cpu_time_s = (cpu_times_end.user - self._cpu_times_start.user) + (
            cpu_times_end.system - self._cpu_times_start.system
        )
        rss_end = self._process.memory_info().rss

        peak_vram_mb = 0.0
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                peak_vram_mb = torch.cuda.max_memory_allocated(self.gpu_index) / (1024 * 1024)
            except RuntimeError:
                pass

        mean_gpu_util, energy_j = 0.0, 0.0
        if self._samples:
            powers = [p for _, p, _ in self._samples]
            utils = [u for _, _, u in self._samples]
            mean_gpu_util = sum(utils) / len(utils)
            # energy = sum(power_i * dt_i), trapezoid over consecutive samples
            for (t0, p0, _), (t1, p1, _) in zip(self._samples, self._samples[1:]):
                energy_j += 0.5 * (p0 + p1) * (t1 - t0)

        self.stats = {
            "stage": self.stage_name,
            "wall_time_s": wall_time_s,
            "cpu_time_s": cpu_time_s,
            "peak_rss_mb": rss_end / (1024 * 1024),
            "peak_vram_mb": peak_vram_mb,
            "mean_gpu_util_pct": mean_gpu_util,
            "energy_j": energy_j,
        }
        return False
