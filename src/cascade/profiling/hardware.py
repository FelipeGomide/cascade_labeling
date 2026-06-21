"""Static hardware info recorded once per experiment run, for context alongside
the per-stage resource measurements in resources.json."""

import platform

import psutil


def get_hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "gpu_name": None,
        "gpu_vram_gb": None,
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info["gpu_vram_gb"] = round(mem.total / (1024**3), 1)
    except Exception:
        pass
    return info
