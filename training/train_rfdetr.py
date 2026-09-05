"""Fine-tune RF-DETR Small on the forklift/person COCO dataset in ./dataset.

The dataset is a Roboflow COCO export (train/valid/test each holding images plus
an `_annotations.coco.json`), which is the layout RF-DETR's trainer reads directly.

Usage:
    .venv/bin/python train_rfdetr.py                    # defaults below
    .venv/bin/python train_rfdetr.py --epochs 100 --batch-size 4
    .venv/bin/python train_rfdetr.py --resume output/checkpoint.pth
"""

import argparse
import json
import os
from pathlib import Path

from rfdetr import RFDETRSmall

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Train RF-DETR Small on a COCO dataset")
    p.add_argument("--dataset-dir", default=str(ROOT / "dataset"))
    p.add_argument("--output-dir", default=str(ROOT / "output"))
    p.add_argument("--epochs", type=int, default=60)
    # Effective batch = batch_size * grad_accum_steps. RF-DETR is tuned around 16.
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    # Must be divisible by 32 for RFDETRSmall (patch_size 16 * num_windows 2).
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    p.add_argument("--no-test", action="store_true", help="skip evaluation on the test split")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--project", default="forklift-rfdetr")
    p.add_argument("--run", default=None)
    return p.parse_args()


def describe_dataset(dataset_dir: Path):
    """Sanity-check the splits and print what we are about to train on."""
    for split in ("train", "valid", "test"):
        ann = dataset_dir / split / "_annotations.coco.json"
        if not ann.exists():
            if split == "test":
                print(f"[dataset] no test split at {ann}, skipping")
                continue
            raise FileNotFoundError(f"missing required annotations: {ann}")
        with open(ann) as f:
            coco = json.load(f)
        names = {c["id"]: c["name"] for c in coco["categories"]}
        counts = {}
        for a in coco["annotations"]:
            counts[names[a["category_id"]]] = counts.get(names[a["category_id"]], 0) + 1
        print(
            f"[dataset] {split:5s} images={len(coco['images']):4d} "
            f"boxes={len(coco['annotations']):4d} per-class={counts}"
        )


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    describe_dataset(dataset_dir)

    if args.resolution % 32 != 0:
        raise ValueError("--resolution must be divisible by 32 for RF-DETR Small")

    model = RFDETRSmall()

    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        resolution=args.resolution,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        # Small dataset (295 images) -> keep EMA + multi-scale aug on to limit overfitting.
        use_ema=True,
        multi_scale=True,
        expanded_scales=True,
        tensorboard=True,
        wandb=args.wandb,
        project=args.project,
        run=args.run,
        run_test=not args.no_test,
        resume=args.resume,
    )

    print(f"\n[done] weights written to {output_dir}")
    for name in ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth"):
        p = output_dir / name
        if p.exists():
            print(f"  {name}: {p.stat().st_size / 1e6:.1f} MB")
    print(f"[tensorboard] tensorboard --logdir {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
