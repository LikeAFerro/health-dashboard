#!/usr/bin/env python3
"""Print CPU usage, RAM stats, and NVIDIA GPU temperature (via nvidia-smi)."""

from __future__ import annotations

import csv
import io
import subprocess
import sys

import psutil


def format_gib(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.2f} GiB"


def print_system_stats() -> None:
    # One sample window so the percentage reflects recent utilization.
    cpu_pct = psutil.cpu_percent(interval=1.0)
    mem = psutil.virtual_memory()

    print("=== System ===")
    print(f"CPU usage:     {cpu_pct:.1f}%")
    print(
        f"RAM:          {mem.percent:.1f}% used "
        f"({format_gib(mem.used)} / {format_gib(mem.total)})"
    )
    print(f"RAM available: {format_gib(mem.available)}")


def print_gpu_stats() -> None:
    print("\n=== NVIDIA GPU ===")
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except FileNotFoundError:
        print(
            "nvidia-smi not found. Install NVIDIA drivers / CUDA tools, "
            "or run on a machine with an NVIDIA GPU.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi failed: {e.stderr or e.stdout or e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("nvidia-smi timed out.", file=sys.stderr)
        sys.exit(1)

    reader = csv.reader(io.StringIO(proc.stdout.strip()))
    rows = list(reader)
    if not rows:
        print("No GPU data returned.")
        return

    header, *data = rows
    if len(header) >= 3 and header[0].lower() == "index":
        pass
    else:
        data = rows

    for row in data:
        if len(row) < 3:
            print(f"  (unexpected row: {row})")
            continue
        idx, name, temp = row[0].strip(), row[1].strip(), row[2].strip()
        print(f"  GPU {idx}: {name} — {temp} °C")


def main() -> None:
    print_system_stats()
    print_gpu_stats()


if __name__ == "__main__":
    main()
