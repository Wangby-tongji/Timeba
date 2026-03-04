# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import time
from itertools import chain
from data import ArgoTestDataset
from utils import Logger, load_pretrain


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


# define parser
parser = argparse.ArgumentParser(description="Argoverse Motion Forecasting in Pytorch")
parser.add_argument(
    "-m", "--model", default="angle90", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true", default=True)
parser.add_argument(
    "--split", type=str, default="val", help='data split, "val" or "test"'
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)

# ---------------------------------------------------------------------------
# Argoverse-style forecasting metrics (inlined; no external argoverse dependency)
# Faithfully reproduces the core logic of argoverse.evaluation.eval_forecasting{,2}.compute_forecasting_metrics:
# - minFDE chooses the best guess (among top-K), and minADE is ADE of that same chosen guess (NOT min over ADE).
# - MR (miss rate) uses the chosen minFDE guess with threshold.
# - Optional probabilistic variants (p-min*, brier-min*) are supported if probabilities are provided.
# NOTE: DAC (drivable area compliance) is not computed here because it requires Argoverse HD maps.
# ---------------------------------------------------------------------------

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


def _to_numpy_xy(traj):
    """Convert a single trajectory to numpy array of shape (T, 2)."""
    if traj is None:
        raise ValueError("Trajectory is None")

    if "torch" in globals() and hasattr(torch, "Tensor") and isinstance(traj, torch.Tensor):
        arr = traj.detach().cpu().numpy()
    else:
        arr = traj if isinstance(traj, np.ndarray) else np.asarray(traj)

    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError(f"Trajectory must have shape (T,2). Got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _normalize_guesses(guesses):
    """
    Normalize different prediction container formats to a python list of (T,2) numpy arrays.

    Supported:
      - (T,2) -> [ (T,2) ]
      - (K,T,2) -> [ (T,2) ] * K
      - list[ (T,2) ] -> list
      - torch tensors with same shapes are also supported
    """
    if guesses is None:
        raise ValueError("Pred guesses is None")

    if "torch" in globals() and hasattr(torch, "Tensor") and isinstance(guesses, torch.Tensor):
        guesses = guesses.detach().cpu().numpy()

    if isinstance(guesses, np.ndarray):
        if guesses.ndim == 2 and guesses.shape[-1] == 2:
            return [_to_numpy_xy(guesses)]
        if guesses.ndim == 3 and guesses.shape[-1] == 2:
            return [_to_numpy_xy(guesses[k]) for k in range(guesses.shape[0])]
        raise ValueError(f"Unsupported ndarray prediction shape {guesses.shape}")

    if isinstance(guesses, (list, tuple)):
        if len(guesses) == 0:
            return []
        g0 = np.asarray(guesses[0])
        # if first element looks like a point [x,y], treat as a single trajectory in list form
        if g0.ndim == 1 and g0.shape[0] == 2:
            return [_to_numpy_xy(np.asarray(guesses))]
        return [_to_numpy_xy(g) for g in guesses]

    return [_to_numpy_xy(np.asarray(guesses))]


def _get_ade(pred_xy, gt_xy):
    dif = pred_xy - gt_xy
    d = np.sqrt(np.sum(dif * dif, axis=1))
    return float(np.sum(d) / pred_xy.shape[0])


def _get_fde(pred_xy, gt_xy):
    dx = float(pred_xy[-1, 0] - gt_xy[-1, 0])
    dy = float(pred_xy[-1, 1] - gt_xy[-1, 1])
    return float(np.sqrt(dx * dx + dy * dy))


def _get_rmse(pred_xy, gt_xy):
    """Root Mean Squared Error over Euclidean displacement per timestep."""
    dif = pred_xy - gt_xy
    mse = float(np.mean(np.sum(dif * dif, axis=1)))
    return float(np.sqrt(mse))



# ------------------------
# Profiling helpers
# ------------------------
def _count_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _fmt_num(n):
    if n is None:
        return "N/A"
    n = float(n)
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n/div:.3f}{unit}"
    return f"{n:.0f}"


def _measure_inference_speed(model, sample_data, warmup=10, iters=50):
    """Measure forward-pass latency on the current device.
    Returns dict with ms/batch, ms/sample, samples/s.
    """
    # Best-effort batch size
    bs = None
    if isinstance(sample_data, dict):
        # common key: 'idx' (list) or tensor-like
        if "idx" in sample_data:
            try:
                bs = len(sample_data["idx"])
            except Exception:
                bs = None
        if bs is None:
            # try any tensor value
            for v in sample_data.values():
                if hasattr(v, "shape") and getattr(v, "shape", None) is not None:
                    try:
                        bs = int(v.shape[0])
                        break
                    except Exception:
                        pass

    device = None
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    model.eval()
    with torch.inference_mode():
        # warmup
        for _ in range(max(0, int(warmup))):
            _ = model(sample_data)
        _sync()

        # timed
        t0 = time.perf_counter()
        for _ in range(max(1, int(iters))):
            _ = model(sample_data)
        _sync()
        t1 = time.perf_counter()

    dt = (t1 - t0) / max(1, int(iters))
    ms_per_batch = dt * 1000.0
    if bs is None or bs <= 0:
        return {
            "batch_size": bs,
            "ms_per_batch": ms_per_batch,
            "ms_per_sample": None,
            "samples_per_s": None,
        }
    ms_per_sample = ms_per_batch / float(bs)
    samples_per_s = float(bs) / dt if dt > 0 else None
    return {
        "batch_size": bs,
        "ms_per_batch": ms_per_batch,
        "ms_per_sample": ms_per_sample,
        "samples_per_s": samples_per_s,
    }


def _estimate_flops_profiler(model, sample_data):
    """Estimate FLOPs for ONE forward pass on sample_data using torch.profiler.
    Note: this is best-effort; some custom ops may not report FLOPs.
    Returns (flops_int_or_None, err_or_None).
    """
    try:
        from torch.profiler import profile, ProfilerActivity
        activities = [ProfilerActivity.CPU]
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.inference_mode():
                _ = model(sample_data)

        total_flops = 0
        for evt in prof.key_averages():
            fl = getattr(evt, "flops", None)
            if fl is None:
                continue
            # torch sometimes returns 0 or None
            try:
                total_flops += int(fl)
            except Exception:
                pass
        if total_flops <= 0:
            return None, "Profiler returned 0 FLOPs (custom ops may not be counted)."
        return total_flops, None
    except Exception as e:
        return None, str(e)

def compute_forecasting_metrics(
    forecasted_trajectories,
    gt_trajectories,
    city_names,
    max_n_guesses,
    horizon,
    miss_threshold,
    forecasted_probabilities=None,
):
    min_ade_list = []
    min_fde_list = []
    miss_list = []
    rmse_list = []

    prob_min_ade_list = []
    prob_min_fde_list = []
    prob_miss_list = []
    prob_min_rmse_list = []
    brier_min_ade_list = []
    brier_min_fde_list = []
    brier_min_rmse_list = []

    for seq_id, gt_raw in gt_trajectories.items():
        if gt_raw is None:
            continue
        if seq_id not in forecasted_trajectories:
            raise KeyError(f"Missing forecast for seq_id={seq_id}")

        gt = _to_numpy_xy(gt_raw)[:horizon]
        guesses = _normalize_guesses(forecasted_trajectories[seq_id])
        if len(guesses) == 0:
            continue

        max_num_traj = min(int(max_n_guesses), len(guesses))

        if forecasted_probabilities is not None:
            probs_raw = forecasted_probabilities[seq_id]
            if len(probs_raw) != len(guesses):
                raise ValueError(
                    f"seq_id={seq_id}: probabilities length {len(probs_raw)} != trajectories length {len(guesses)}"
                )
            probs = np.asarray([float(p) for p in probs_raw], dtype=np.float64)
            sorted_idx = np.argsort(-probs, kind="stable")
            pruned_idx = sorted_idx[:max_num_traj]
            pruned_probs = probs[pruned_idx]
            psum = float(np.sum(pruned_probs))
            if not np.isfinite(psum) or psum <= 0.0:
                pruned_probs = np.ones_like(pruned_probs, dtype=np.float64) / max_num_traj
            else:
                pruned_probs = pruned_probs / psum
            pruned_guesses = [guesses[int(i)] for i in pruned_idx]
            pruned_probs = pruned_probs.tolist()
        else:
            pruned_guesses = guesses[:max_num_traj]
            pruned_probs = None

        curr_min_fde = float("inf")
        best_idx = 0
        for j, traj in enumerate(pruned_guesses):
            traj_h = traj[:horizon]
            fde = _get_fde(traj_h, gt)
            if fde < curr_min_fde:
                curr_min_fde = fde
                best_idx = j

        best_traj = pruned_guesses[best_idx][:horizon]
        curr_min_ade = _get_ade(best_traj, gt)
        curr_rmse = _get_rmse(best_traj, gt)

        min_ade_list.append(curr_min_ade)
        rmse_list.append(curr_rmse)
        min_fde_list.append(curr_min_fde)
        miss_list.append(1.0 if curr_min_fde > miss_threshold else 0.0)

        if pruned_probs is not None:
            p_best = float(pruned_probs[best_idx])
            prob_miss_list.append(1.0 if curr_min_fde > miss_threshold else (1.0 - p_best))

            nll_cap = min(-float(np.log(max(p_best, 1e-12))), -float(np.log(LOW_PROB_THRESHOLD_FOR_METRICS)))
            prob_min_ade_list.append(nll_cap + curr_min_ade)
            prob_min_fde_list.append(nll_cap + curr_min_fde)
            prob_min_rmse_list.append(nll_cap + curr_rmse)

            brier = (1.0 - p_best) ** 2
            brier_min_ade_list.append(brier + curr_min_ade)
            brier_min_fde_list.append(brier + curr_min_fde)
            brier_min_rmse_list.append(brier + curr_rmse)

    if len(min_ade_list) == 0:
        print("[WARN] No valid GT found; cannot compute forecasting metrics.")
        return {}

    metrics = {
        "minADE": float(np.mean(min_ade_list)),
        "minFDE": float(np.mean(min_fde_list)),
        "MR": float(np.mean(miss_list)),
        "RMSE": float(np.mean(rmse_list)),
    }
    if forecasted_probabilities is not None and len(prob_min_ade_list) > 0:
        metrics.update(
            {
                "p-minADE": float(np.mean(prob_min_ade_list)),
                "p-minFDE": float(np.mean(prob_min_fde_list)),
                "p-RMSE": float(np.mean(prob_min_rmse_list)),
                "p-MR": float(np.mean(prob_miss_list)),
                "brier-minADE": float(np.mean(brier_min_ade_list)),
                "brier-minFDE": float(np.mean(brier_min_fde_list)),
                "brier-RMSE": float(np.mean(brier_min_rmse_list)),
            }
        )

    print("------------------------------------------------")
    print(f"Prediction Horizon : {horizon}, Max #guesses (K): {max_n_guesses}")
    print("------------------------------------------------")
    print(metrics)
    print("------------------------------------------------")
    return metrics

def main():
    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, _, collate_fn, net, loss, post_process, opt = model.get_model()

    # load pretrain model
    ckpt_path = args.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    # Data loader for evaluation
    dataset = ArgoTestDataset(args.split, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # begin inference
    # ------------------------
    # Model profiling (params / FLOPs / speed)
    # ------------------------
    total_params, trainable_params = _count_parameters(net)

    # Take ONE batch as a representative sample for profiling
    _it = iter(data_loader)
    try:
        _sample = next(_it)
        _sample = dict(_sample)
    except StopIteration:
        _sample = None

    print("================================================")
    print(f"[PROFILE] Model: {args.model}")
    print(f"[PROFILE] Params: total={_fmt_num(total_params)} ({total_params}) | trainable={_fmt_num(trainable_params)} ({trainable_params})")

    if _sample is not None:
        # Speed
        sp = _measure_inference_speed(net, _sample, warmup=5, iters=10)
        if sp.get("ms_per_sample") is None:
            print(f"[PROFILE] Speed: {sp['ms_per_batch']:.3f} ms/batch (batch_size unknown)")
        else:
            print(f"[PROFILE] Speed: {sp['ms_per_batch']:.3f} ms/batch | {sp['ms_per_sample']:.3f} ms/sample | {sp['samples_per_s']:.2f} samples/s")

        # FLOPs
        flops, flops_err = _estimate_flops_profiler(net, _sample)
        if flops is None:
            print(f"[PROFILE] FLOPs: N/A ({flops_err})")
        else:
            print(f"[PROFILE] FLOPs: {flops/1e9:.3f} GFLOPs (one forward on sample batch)")
    else:
        print("[PROFILE] Skip speed/FLOPs: empty dataloader.")

    print("================================================")

    preds = {}
    gts = {}
    gts2 = {}
    feats = {}
    cities = {}
    # begin inference
    if _sample is not None:
        _stream = chain([_sample], _it)
        _total = len(data_loader)
    else:
        _stream = data_loader
        _total = len(data_loader)
    for ii, data in tqdm(enumerate(_stream), total=_total):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            # (Optional) print per-batch GPU memory usage (disabled by default)
            if os.environ.get("PRINT_GPU_MEM", "0") == "1" and torch.cuda.is_available():
                allocated_memory = torch.cuda.max_memory_allocated()
                print(f"Batch {ii + 1}/{len(data_loader)}, Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB")
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]

        for i, (argo_idx, pred_traj) in enumerate(zip(data["idx"], results)):
            preds[argo_idx] = pred_traj.squeeze()
            cities[argo_idx] = data["city"][i]
            feats[argo_idx] = data["past"][i][0] if "past" in data else None
            gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None
            gts2[argo_idx] = data["gt_preds"][i][1:] if "gt_preds" in data else None


    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        gts2 = gts2,
        feats = feats,
        cities = cities,
    )
    torch.save(res,f"{config['save_dir']}/results.pkl")
    # evaluate (val/test share the same logic; do NOT generate h5)
    # evaluate (val/test share the same logic; do NOT generate h5)
    valid_ids = [k for k, v in gts.items() if v is not None]
    if len(valid_ids) == 0:
        print(f"[WARN] Split={args.split}: dataset provides no GT (gt_preds missing). Skip metric computation.")
        return
    preds_eval = {k: preds[k] for k in valid_ids if k in preds}
    gts_eval = {k: gts[k] for k in valid_ids}
    cities_eval = {k: cities.get(k, "") for k in valid_ids}
    # Max #guesses (K): 6
    _ = compute_forecasting_metrics(preds_eval, gts_eval, cities_eval, 6, 42, 2)
    # Max #guesses (K): 1
    _ = compute_forecasting_metrics(preds_eval, gts_eval, cities_eval, 1, 42, 2)

    # import ipdb;ipdb.set_trace()
if __name__ == "__main__":
    main()
