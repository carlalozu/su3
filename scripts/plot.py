#!/usr/bin/env python3
"""
plot_times.py
"""

import re
import sys
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt


THREADS_RE = re.compile(r"Running with\s+(\d+)\s+threads", re.IGNORECASE)
NO_OMP_RE = re.compile(r"OpenMP is not enabled", re.IGNORECASE)
VOLUME_RE = re.compile(r"Volume:\s*(\d+)", re.IGNORECASE)
AVX_RE    = re.compile(r"AVX vectorization is\s+(ON|OFF)", re.IGNORECASE)

# Matches lines like:
# AoS compute              total   0.070443 s | avg   ...
LINE_RE = re.compile(
    r"^(AoS|SoA|AoSoA)\s+(init|compute)\s+total\s+([0-9]*\.?[0-9]+)\s*s",
    re.IGNORECASE
)


def read_text(path: str | None) -> str:
    if path is None:
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def parse(text: str):
    """
    Returns:
      data[threads][phase][layout] = total_time_seconds
    Where threads can be:
      - integer thread count (1,2,4,...)
      - 0 meaning "No OpenMP" baseline
    """
    data = defaultdict(lambda: {"init": {}, "compute": {}})

    current_threads = None
    in_no_omp_block = False

    for raw in text.splitlines():
        line = raw.strip()

        mV = VOLUME_RE.search(line)
        if mV:
            volume = int(mV.group(1))
        
        mA = AVX_RE.search(line)
        if mA:
            avx_on = (mA.group(1).upper() == "ON")

        # detect "No OpenMP" mode (baseline)
        if NO_OMP_RE.search(line):
            current_threads = 0
            in_no_omp_block = True

        mT = THREADS_RE.search(line)
        if mT:
            current_threads = int(mT.group(1))
            in_no_omp_block = False

        m = LINE_RE.match(line)
        if m and current_threads is not None:
            layout = m.group(1)
            phase = m.group(2).lower()
            total = float(m.group(3))
            # normalize layout capitalization
            layout = {"aos": "AoS", "soa": "SoA", "aosoa": "AoSoA"}[layout.lower()]
            data[current_threads][phase][layout] = total

    # Convert to normal dict and sort threads (baseline 0 first, then ascending)
    threads_sorted = sorted(data.keys(), key=lambda t: (t != 0, t))
    return OrderedDict((t, data[t]) for t in threads_sorted), volume, avx_on

def plot_phase(data, phase: str, volume: int, avx_on:bool):
    markers = ["o", "*", "v"]
    threads = list(data.keys())[1:]
    layouts = ["AoS", "SoA", "AoSoA"]
    x = list(range(len(threads)))

    plt.figure()
    for i, layout in enumerate(layouts):
        y = []
        for t in threads:
            y.append(data[t][phase].get(layout, float("nan")))
        line, = plt.plot(x, y, marker=markers[i], label=layout)
        plt.hlines(data[0][phase].get(layout, float("nan")), x[0], x[-1], 
            label="No OMP", linestyle='-', color=line.get_color(),
            zorder=0, alpha=0.5)
        # add perfect scaling line

        # perfect scaling: linear speedup
        perfect_y = [y[0] / t for t in threads]
        plt.plot(x, perfect_y,
            linestyle=":", color=line.get_color(), alpha=0.5,)

    plt.xticks(x, [str(t) for t in threads])
    plt.xlabel("Threads")
    plt.ylim([10e-4, 10e-1])
    plt.yscale('log')
    plt.ylabel("Total time (s)")
    plt.title(f"Compute total time vs threads (volume {volume}, avx {avx_on})")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plot_{phase}_avx{avx_on}_log.pdf", dpi=200)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    text = read_text(path)
    data, volume, avx_on = parse(text)

    if not data:
        print("No timing data found. Did you paste the output correctly?", file=sys.stderr)
        sys.exit(1)

    # Show what we parsed (brief)
    print("Parsed thread blocks:", ", ".join("NoOMP" if t == 0 else str(t) for t in data.keys()))

    # Plot compute + init + speedup
    plot_phase(data, "compute", volume, avx_on)

if __name__ == "__main__":
    main()
