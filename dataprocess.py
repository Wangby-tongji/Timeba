#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ngsim_one_shot.py (v74 split)

一键生成最终数据（只保留最终 NGSIM6），支持多进程加速 + 进度条。

更新点（按你的最新要求）：
- 数据划分：train=01..06，val=07，test=08（val/test 各 1 个原始文件）
- 时域长度：全部改为 74 帧（Total_Frames>=74、滑窗 window=74、id 出现次数>=74）
- 行数阈值：默认改为 2*74+1=149（与原来 109=2*54+1 保持同一逻辑）

默认假设当前工作目录下存在：
  ./data/01.csv ... ./data/08.csv

运行：
  python ngsim_one_shot_v74.py

输出：
  ./NGSIM6/train/*.csv
  ./NGSIM6/val/*.csv
  ./NGSIM6/test/*.csv
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, **kwargs):
        return x if x is not None else []


@dataclass(frozen=True)
class Cfg:
    raw_dir: Path
    out_dir: Path
    tmp_dir: Path

    files: List[int]
    train_ids: range
    val_ids: range
    test_ids: range

    # parameters
    min_total_frames: int = 74
    window_frames: int = 74
    stride_frames: int = 8
    min_id_count: int = 74
    min_rows_per_split: int = 149  # 2*74+1

    feet_to_m: float = 0.3048
    round_decimals: int = 4
    vel_cols: Tuple[str, ...] = ("v_Vel", "v_Acc")


def _ensure_empty_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{where}] 缺少列: {missing}. 当前列: {list(df.columns)}")


def _split_for_file_id(i: int, cfg: Cfg) -> str:
    if i in cfg.train_ids:
        return "train"
    if i in cfg.val_ids:
        return "val"
    if i in cfg.test_ids:
        return "test"
    return "train"


def convert_units(df: pd.DataFrame, cols: Iterable[str], factor: float, decimals: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (out[c] * factor).round(decimals)
    return out


def filter_total_frames(df: pd.DataFrame, col: str, min_frames: int) -> pd.DataFrame:
    _require_cols(df, [col], "filter_total_frames")
    out = df[df[col] >= min_frames].copy()
    out = out.drop(columns=[col])
    return out


def sort_by_frame_id(df: pd.DataFrame, frame_col: str = "frame", id_col: str = "id") -> pd.DataFrame:
    _require_cols(df, [frame_col, id_col], "sort_by_frame_id")
    out = df.copy()
    out[frame_col] = pd.to_numeric(out[frame_col], errors="coerce")
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce")
    out = out.dropna(subset=[frame_col, id_col])
    out[frame_col] = out[frame_col].astype(int)
    out[id_col] = out[id_col].astype(int)
    return out.sort_values(by=[frame_col, id_col])


def split_sliding_windows(df: pd.DataFrame, window: int, stride: int, frame_col: str = "frame"):
    min_frame = int(df[frame_col].min())
    max_frame = int(df[frame_col].max())
    for start in range(min_frame, max_frame, stride):
        end = start + window
        chunk = df[(df[frame_col] >= start) & (df[frame_col] < end)].copy()
        if not chunk.empty:
            yield start, chunk


def filter_ids_by_count(df: pd.DataFrame, id_col: str, min_count: int) -> pd.DataFrame:
    _require_cols(df, [id_col], "filter_ids_by_count")
    counts = df[id_col].value_counts()
    keep = counts[counts >= min_count].index
    return df[df[id_col].isin(keep)].copy()


def add_agent_type(df: pd.DataFrame, id_col: str = "id", type_col: str = "type") -> pd.DataFrame:
    _require_cols(df, [id_col], "add_agent_type")
    out = df.copy()
    agent_id = int(out[id_col].min())
    out[type_col] = "OTHERS"
    out.loc[out[id_col] == agent_id, type_col] = "AGENT"
    return out


def apply_prefix_padding(
    df: pd.DataFrame,
    prefix: str,
    frame_padding: str,
    id_padding: str,
    frame_col: str = "frame",
    id_col: str = "id",
) -> pd.DataFrame:
    _require_cols(df, [frame_col, id_col], "apply_prefix_padding")
    out = df.copy()
    out[frame_col] = pd.to_numeric(out[frame_col], errors="coerce").fillna(0).astype(int)
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce").fillna(0).astype(int)
    out[frame_col] = out[frame_col].map(lambda x: f"{prefix}{frame_padding}{x}")
    out[id_col] = out[id_col].map(lambda x: f"{prefix}{id_padding}{x}")
    return out


def _parse_tmp_name(stem: str) -> Tuple[int, int]:
    m = re.match(r"^(\d{2})_(\d+)$", stem)
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def _process_one_raw_file(i: int, cfg: Cfg) -> Tuple[str, int]:
    split = _split_for_file_id(i, cfg)
    src = cfg.raw_dir / f"{i:02d}.csv"
    if not src.exists():
        raise FileNotFoundError(f"找不到输入文件: {src}")

    df = pd.read_csv(src)
    df = convert_units(df, cfg.vel_cols, cfg.feet_to_m, cfg.round_decimals)
    df = filter_total_frames(df, "Total_Frames", cfg.min_total_frames)
    df = sort_by_frame_id(df, "frame", "id")

    tmp_split = cfg.tmp_dir / split
    _ensure_dir(tmp_split)

    produced = 0
    for start, chunk in split_sliding_windows(df, cfg.window_frames, cfg.stride_frames, "frame"):
        chunk = filter_ids_by_count(chunk, "id", cfg.min_id_count)
        if chunk.empty:
            continue
        chunk = add_agent_type(chunk, "id", "type")
        if len(chunk) < cfg.min_rows_per_split:
            continue

        out_name = f"{i:02d}_{start:010d}.csv"
        chunk.to_csv(tmp_split / out_name, index=False)
        produced += 1

    return split, produced


def finalize_split(cfg: Cfg, split: str) -> int:
    tmp_split = cfg.tmp_dir / split
    out_split = cfg.out_dir / split
    _ensure_empty_dir(out_split)

    files = sorted(list(tmp_split.glob("*.csv")), key=lambda p: _parse_tmp_name(p.stem))
    if not files:
        return 0

    # distinct paddings per split to avoid collisions
    if split == "train":
        frame_padding, id_padding = "000000", "000"
    elif split == "test":
        frame_padding, id_padding = "0000000", "0000"
    else:  # val
        frame_padding, id_padding = "000000000", "00000"

    for idx, p in enumerate(tqdm(files, desc=f"Finalizing {split}", unit="csv"), start=1):
        df = pd.read_csv(p)
        df = apply_prefix_padding(df, str(idx), frame_padding, id_padding, "frame", "id")
        df.to_csv(out_split / f"{idx}.csv", index=False)
        p.unlink(missing_ok=True)

    shutil.rmtree(tmp_split, ignore_errors=True)
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--raw_dir", type=str, default="data", help="输入目录，默认 ./data (包含 01.csv..08.csv)")
    ap.add_argument("--out_dir", type=str, default="NGSIM6", help="输出目录，默认 ./NGSIM6")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 8))),
                    help="并行进程数，默认 min(8, CPU核数)")
    ap.add_argument("--files", type=str, default="1-8", help="处理哪些原始文件，默认 1-8")

    ap.add_argument("--window_frames", type=int, default=74, help="滑窗长度（帧），默认 74")
    ap.add_argument("--stride_frames", type=int, default=8, help="滑窗步长（帧），默认 8")
    ap.add_argument("--min_total_frames", type=int, default=74, help="Total_Frames 下限，默认 74")
    ap.add_argument("--min_id_count", type=int, default=74, help="切片内每个 id 的最少出现次数，默认 74")
    ap.add_argument("--min_rows", type=int, default=149, help="切片行数下限，默认 149 (=2*74+1)")
    args = ap.parse_args()

    if "-" in args.files:
        a, b = args.files.split("-", 1)
        files = list(range(int(a), int(b) + 1))
    else:
        files = [int(x.strip()) for x in args.files.split(",") if x.strip()]

    raw_dir = Path(args.raw_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    tmp_dir = out_dir.parent / (out_dir.name + ".__tmp__")

    cfg = Cfg(
        raw_dir=raw_dir,
        out_dir=out_dir,
        tmp_dir=tmp_dir,
        files=files,
        train_ids=range(1, 7),   # 01..06
        val_ids=range(7, 8),     # 07
        test_ids=range(8, 9),    # 08
        min_total_frames=args.min_total_frames,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        min_id_count=args.min_id_count,
        min_rows_per_split=args.min_rows,
    )

    _ensure_empty_dir(cfg.out_dir)
    _ensure_empty_dir(cfg.tmp_dir)

    counts = {"train": 0, "val": 0, "test": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one_raw_file, i, cfg) for i in cfg.files]
        pbar = tqdm(total=len(futs), desc="Processing raw files", unit="file")
        try:
            for fut in as_completed(futs):
                split, produced = fut.result()
                counts[split] += produced
                pbar.update(1)
                pbar.set_postfix(train_windows=counts["train"], val_windows=counts["val"], test_windows=counts["test"])
        finally:
            try:
                pbar.close()
            except Exception:
                pass

    train_n = finalize_split(cfg, "train")
    val_n = finalize_split(cfg, "val")
    test_n = finalize_split(cfg, "test")

    shutil.rmtree(cfg.tmp_dir, ignore_errors=True)

    print(f"Done. Final output: {cfg.out_dir}")
    print(f"  train: {train_n} files")
    print(f"  val  : {val_n} files")
    print(f"  test : {test_n} files")


if __name__ == "__main__":
    main()
