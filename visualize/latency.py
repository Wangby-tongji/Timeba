#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize latency/FLOPs scaling vs history length.

Run:
  python latency_vis.py

Outputs:
  latency_vs_history.png
  flops_vs_history.png
  throughput_vs_history.png
"""

import pandas as pd
import matplotlib.pyplot as plt

DATA = [
    # history, ms/batch, ms/sample, samples/s, GFLOPs per *batch forward*
    (8,  22.809, 1.426, 701.47, 170.178),
    (16, 26.288, 1.643, 608.63, 237.060),
    (24, 27.643, 1.728, 578.81, 271.001),
    (32, 30.609, 1.913, 522.72, 306.986),
]

def main():
    rows = []
    for h, ms_b, ms_s, sps, gflops_b in DATA:
        rows.append({
            "history": h,
            "ms_per_batch": ms_b,
            "ms_per_sample": ms_s,
            "samples_per_s": sps,
            "gflops_per_batch": gflops_b,
            "batch_size_est": ms_b / ms_s,
        })
    df = pd.DataFrame(rows)

    # batch size inferred from timing
    bs = int(round(df["batch_size_est"].median()))
    df["gflops_per_sample"] = df["gflops_per_batch"] / bs

    print("Inferred batch size (median round):", bs)
    print(df[["history","ms_per_sample","samples_per_s","gflops_per_batch","gflops_per_sample","batch_size_est"]].to_string(index=False))

    # latency
    plt.figure()
    plt.plot(df["history"], df["ms_per_sample"], marker="o")
    plt.xlabel("History length (frames)")
    plt.ylabel("Latency (ms / sample)")
    plt.title("Inference latency vs. history length")
    plt.savefig("latency_vs_history.png", dpi=200, bbox_inches="tight")
    plt.close()

    # FLOPs per sample
    plt.figure()
    plt.plot(df["history"], df["gflops_per_sample"], marker="o")
    plt.xlabel("History length (frames)")
    plt.ylabel("FLOPs (GFLOPs / sample)")
    plt.title(f"FLOPs per sample vs. history length")
    plt.savefig("flops_vs_history.png", dpi=200, bbox_inches="tight")
    plt.close()

    # throughput
    plt.figure()
    plt.plot(df["history"], df["samples_per_s"], marker="o")
    plt.xlabel("History length (frames)")
    plt.ylabel("Throughput (samples / s)")
    plt.title("Throughput vs. history length")
    plt.savefig("throughput_vs_history.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved: latency_vs_history.png, flops_vs_history.png, throughput_vs_history.png")

if __name__ == "__main__":
    main()
