"""GPU device selection and status (advisory helper for shared servers)."""

import torch

# Threshold: used memory below this (MiB) is "likely free". Advisory only.
LIKELY_FREE_THRESHOLD_MIB = 500


def get_gpu_memory_info(device_id: int) -> tuple[int, int] | None:
    """
    Get (free_bytes, total_bytes) for a GPU.
    Returns None if device is invalid or not available.
    """
    if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
        return None
    try:
        free, total = torch.cuda.mem_get_info(device_id)
        return (free, total)
    except (AttributeError, RuntimeError):
        return None


def select_device(gpu_id: int | None = None) -> str:
    """
    Select device for model inference.
    If gpu_id is provided and valid, use that GPU.
    Otherwise, select the GPU with the most free memory.
    Falls back to CPU if no CUDA GPUs available.
    """
    if not torch.cuda.is_available():
        return "cpu"

    if gpu_id is not None and 0 <= gpu_id < torch.cuda.device_count():
        return f"cuda:{gpu_id}"

    n = torch.cuda.device_count()
    if n == 0:
        return "cpu"
    if n == 1:
        return "cuda:0"

    best_id = 0
    best_free = 0
    for i in range(n):
        info = get_gpu_memory_info(i)
        if info:
            free, _ = info
            if free > best_free:
                best_free = free
                best_id = i
    return f"cuda:{best_id}"


def get_gpu_status() -> list[dict]:
    """
    Get status for all GPUs. Returns list of dicts with:
      id, name, used_mib, total_mib, free_mib, likely_free
    """
    if not torch.cuda.is_available():
        return []

    result = []
    for i in range(torch.cuda.device_count()):
        info = get_gpu_memory_info(i)
        if info:
            free, total = info
            used = total - free
            used_mib = used // (1024 * 1024)
            total_mib = total // (1024 * 1024)
            likely_free = used_mib < LIKELY_FREE_THRESHOLD_MIB and total_mib > 0
        else:
            used_mib = total_mib = 0
            likely_free = False

        try:
            name = torch.cuda.get_device_name(i)
        except Exception:
            name = "Unknown"

        result.append({
            "id": i,
            "name": name,
            "used_mib": used_mib,
            "total_mib": total_mib,
            "likely_free": likely_free,
        })
    return result


def suggest_gpu() -> int | None:
    """
    Suggest the GPU with most free memory.
    Advisory only—"likely free" does not mean guaranteed free on a shared server.
    Returns None if no GPUs available.
    """
    status = get_gpu_status()
    if not status:
        return None

    best_id = 0
    best_free = 0
    for s in status:
        free = s["total_mib"] - s["used_mib"]
        if free > best_free:
            best_free = free
            best_id = s["id"]
    return best_id


def print_gpu_status() -> None:
    """Print GPU status. Kept for backward compatibility; prefer check_gpus.py."""
    _print_advisory_status()


def _print_advisory_status() -> None:
    """
    Print advisory GPU status for shared servers.
    Shows used/total memory and suggests a GPU. User should manually run:
      CUDA_VISIBLE_DEVICES=<id> python scripts/...
    """
    status = get_gpu_status()
    if not status:
        print("No CUDA GPUs available.")
        return

    suggested = suggest_gpu()
    print("GPU status (advisory—check nvidia-smi; 'likely free' does not guarantee availability):\n")

    for s in status:
        label = "  -> likely free" if s["likely_free"] else "  -> in use"
        print(f"GPU {s['id']}: {s['used_mib']} MiB used / {s['total_mib']} MiB total{label}")
        print(f"  {s['name']}")

    if suggested is not None:
        print(f"\nSuggested GPU: {suggested}")
        print(f"To use: CUDA_VISIBLE_DEVICES={suggested} python scripts/run_mert_retrieval.py ...")
