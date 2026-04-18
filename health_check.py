#!/usr/bin/env python3
"""Print CPU usage, RAM stats, and NVIDIA GPU temperature (via nvidia-smi)."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import time
from datetime import datetime

import psutil

_RED = "\033[31m"
_RESET = "\033[0m"

LOG_INTERVAL_DEFAULT_SEC = 300


def format_gib(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.2f} GiB"


def read_system_metrics() -> tuple[float, psutil._common.svmem]:
    cpu_pct = psutil.cpu_percent(interval=1.0)
    mem = psutil.virtual_memory()
    return cpu_pct, mem


def system_stats_lines(cpu_pct: float, mem: psutil._common.svmem, *, for_tty: bool) -> list[str]:
    lines = ["=== System ==="]
    cpu_line = f"CPU usage:     {cpu_pct:.1f}%"
    if for_tty and cpu_pct > 80.0 and sys.stdout.isatty():
        cpu_line = f"{_RED}{cpu_line}{_RESET}"
    lines.append(cpu_line)
    lines.append(
        f"RAM:          {mem.percent:.1f}% used "
        f"({format_gib(mem.used)} / {format_gib(mem.total)})"
    )
    lines.append(f"RAM available: {format_gib(mem.available)}")
    return lines


def gpu_stats_lines() -> tuple[list[str] | None, str | None]:
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
        return None, (
            "nvidia-smi not found. Install NVIDIA drivers / CUDA tools, "
            "or run on a machine with an NVIDIA GPU."
        )
    except subprocess.CalledProcessError as e:
        return None, f"nvidia-smi failed: {e.stderr or e.stdout or e!r}"
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi timed out."

    reader = csv.reader(io.StringIO(proc.stdout.strip()))
    rows = list(reader)
    if not rows:
        return [], None

    header, *data = rows
    if not (len(header) >= 3 and header[0].lower() == "index"):
        data = rows

    out: list[str] = []
    for row in data:
        if len(row) < 3:
            out.append(f"  (unexpected row: {row})")
            continue
        idx, name, temp = row[0].strip(), row[1].strip(), row[2].strip()
        out.append(f"  GPU {idx}: {name} — {temp} °C")
    return out, None


def snapshot_plain_text() -> str:
    cpu_pct, mem = read_system_metrics()
    ts = datetime.now().isoformat(timespec="seconds")
    sys_lines = system_stats_lines(cpu_pct, mem, for_tty=False)
    gpu_lines, gpu_err = gpu_stats_lines()

    block = [f"[{ts}]", *sys_lines, "", "=== NVIDIA GPU ==="]
    if gpu_err is not None:
        block.append(gpu_err)
    elif not gpu_lines:
        block.append("No GPU data returned.")
    else:
        block.extend(gpu_lines)
    return "\n".join(block) + "\n\n"


def print_system_stats() -> None:
    cpu_pct, mem = read_system_metrics()
    for line in system_stats_lines(cpu_pct, mem, for_tty=True):
        print(line)


def print_gpu_stats() -> None:
    print("\n=== NVIDIA GPU ===")
    lines, err = gpu_stats_lines()
    if err is not None:
        print(err, file=sys.stderr)
        sys.exit(1)
    if not lines:
        print("No GPU data returned.")
        return
    for line in lines:
        print(line)


def run_log_loop(path: str, interval_sec: int) -> None:
    print(
        f"Appending stats to {path!r} every {interval_sec}s (Ctrl+C to stop).",
        file=sys.stderr,
    )
    while True:
        text = snapshot_plain_text()
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
        time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help=(
            "Append a timestamped snapshot to this file every "
            f"{LOG_INTERVAL_DEFAULT_SEC // 60} minutes (runs until Ctrl+C)."
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=LOG_INTERVAL_DEFAULT_SEC,
        metavar="SEC",
        help=f"Seconds between log writes when using --log-file (default: {LOG_INTERVAL_DEFAULT_SEC}).",
    )
    args = parser.parse_args()

    if args.log_file:
        if args.interval < 1:
            print("--interval must be at least 1.", file=sys.stderr)
            sys.exit(2)
        try:
            run_log_loop(args.log_file, args.interval)
        except KeyboardInterrupt:
            print("\nStopped.", file=sys.stderr)
        return

    print_system_stats()
    print_gpu_stats()


if __name__ == "__main__":
    main()
