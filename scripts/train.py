"""Official config-driven Timeba training CLI."""

import argparse
from dataclasses import replace
from pathlib import Path

from timeba.engine import build_pipeline, evaluate_batch, save_checkpoint, train_step
from timeba.engine.checkpoint import load_checkpoint

from scripts.common import (
    add_common_arguments,
    build_split_loader,
    experiment_dict,
    load_metrics_history,
    resolve_experiment,
    seed_everything,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def _mean_validation_metrics(components, loader, max_batches):
    totals = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        evaluated = evaluate_batch(components, batch)
        scene_count = evaluated["reg"].shape[0]
        count += scene_count
        for key, values in evaluated["metrics"].items():
            target = totals.setdefault(key, {"minADE": 0.0, "minFDE": 0.0, "MR": 0.0})
            for metric in target:
                target[metric] += values[metric] * scene_count
    if count == 0:
        raise ValueError("validation loader produced no batches")
    return {
        key: {metric: value / count for metric, value in values.items()}
        for key, values in totals.items()
    }


def main():
    args = parse_args()
    experiment = resolve_experiment(args)
    if args.epochs is not None:
        experiment = replace(
            experiment,
            training=replace(experiment.training, epochs=args.epochs),
        )
    seed_everything(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and args.resume is None:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}"
        )

    train_loader = build_split_loader(
        experiment,
        args.data_root,
        "train",
        train=True,
    )
    val_loader = build_split_loader(
        experiment,
        args.data_root,
        "val",
        train=False,
    )
    components = build_pipeline(experiment, device="cuda")

    start_epoch = 0
    metrics_log = []
    if args.resume is not None:
        payload, _ = load_checkpoint(
            args.resume,
            components.model,
            optimizer=components.optimizer,
            strict=True,
            map_location="cpu",
        )
        start_epoch = int(payload["epoch"])
        metrics_log = load_metrics_history(
            output_dir / "metrics.json",
            resume_epoch=start_epoch,
        )

    write_json(output_dir / "experiment.json", experiment_dict(experiment))
    for epoch_index in range(start_epoch, experiment.training.epochs):
        train_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
            progress = epoch_index + (batch_index + 1) / len(train_loader)
            result = train_step(components, batch, epoch=progress)
            train_loss += float(result["loss"].cpu())
            batches += 1
        if batches == 0:
            raise ValueError("training loader produced no batches")

        validation = _mean_validation_metrics(
            components,
            val_loader,
            args.max_val_batches,
        )
        record = {
            "epoch": epoch_index + 1,
            "train_loss": train_loss / batches,
            "validation": validation,
        }
        metrics_log.append(record)
        write_json(output_dir / "metrics.json", metrics_log)
        save_checkpoint(
            output_dir / f"epoch_{epoch_index + 1:03d}.ckpt",
            components,
            epoch=epoch_index + 1,
            extra={"synthetic_only": False},
        )
        print(record)


if __name__ == "__main__":
    main()
