"""Fine-tune a pretrained backbone on the crops (models 2-5).

Examples:
    # ResNet-50 baseline
    python scripts/train.py --model resnet50.a1_in1k --run resnet50 --epochs 15
    # Main model: DINOv2 ViT-B/14 with layer-wise LR decay
    python scripts/train.py --model vit_base_patch14_dinov2.lvd142m --run dinov2_b14_ft --lr 5e-4 --layer-decay 0.7
    # Laptop smoke test
    python scripts/train.py --model resnet18.a1_in1k --manifest data/processed/manifest_mini.csv \
        --crops data/crops_mini --epochs 1 --batch-size 16 --run smoke

Recipe: AdamW, warmup + cosine schedule, label smoothing, RandAugment,
optional layer-wise LR decay (ViTs), weight EMA, mixed precision on GPU.
A checkpoint is written every epoch and --resume continues from it, so
interrupted sessions lose at most one epoch. The best epoch (validation
macro-F1) is evaluated on the unseen test reserves at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from timm.optim import create_optimizer_v2
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.classes import CLASSES  # noqa: E402
from faunavision.data.dataset import CLASS_INDEX, CLASS_NAMES, CropDataset, available_rows  # noqa: E402
from faunavision.metrics import fit_temperature, summarize  # noqa: E402

CLUSTERS = {c: CLASSES[c].cluster for c in CLASS_NAMES}


def build_transforms(model, img_size: int):
    cfg = timm.data.resolve_model_data_config(model)
    cfg.update(input_size=(3, img_size, img_size), crop_pct=1.0)
    train_tf = timm.data.create_transform(
        **cfg, is_training=True, scale=(0.6, 1.0), hflip=0.5,
        auto_augment="rand-m7-mstd0.5", re_prob=0.1,
    )
    eval_tf = timm.data.create_transform(**cfg, is_training=False)
    return train_tf, eval_tf


def balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Sample rare classes more often (inverse square-root frequency).

    Square root is a middle ground: full inverse frequency would show a
    19-image class as often as a 4,000-image one and overfit it.
    """
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return WeightedRandomSampler(torch.as_tensor(w[labels]), num_samples=len(labels), replacement=True)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    logits, labels = [], []
    for x, y, _ in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "manifest.csv")
    p.add_argument("--crops", type=Path, default=ROOT / "data" / "crops")
    p.add_argument("--model", default="resnet50.a1_in1k")
    p.add_argument("--run", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--layer-decay", type=float, default=None, help="layer-wise LR decay, e.g. 0.7 for ViTs")
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999, help="0 disables weight EMA")
    p.add_argument("--no-balance", action="store_true", help="plain shuffling instead of class-balanced sampling")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init", type=Path, default=None, help="start from this checkpoint's weights (e.g. 224 px run)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    run_dir = args.out / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"img_size": args.img_size} if "vit" in args.model else {}
    model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASS_NAMES), **kwargs).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device, weights_only=True)["model"]
        model.load_state_dict(state)
    if hasattr(model, "set_grad_checkpointing") and device == "cuda":
        model.set_grad_checkpointing(True)  # fits ViT-B at 448 px on a 16 GB GPU
    train_tf, eval_tf = build_transforms(model, args.img_size)

    rows = available_rows(pd.read_csv(args.manifest), args.crops)
    splits = {s: rows[rows.split == s] for s in ("train", "val", "test")}
    print(f"{args.model} on {device}: " + ", ".join(f"{s} {len(d):,}" for s, d in splits.items()))
    train_ds = CropDataset(splits["train"], args.crops, train_tf)
    train_labels = splits["train"].cls.map(CLASS_INDEX).to_numpy()
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=amp,
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, drop_last=True,
                              **({"shuffle": True} if args.no_balance else {"sampler": balanced_sampler(train_labels)}),
                              **loader_kw)
    eval_loaders = {s: DataLoader(CropDataset(splits[s], args.crops, eval_tf), **loader_kw) for s in ("val", "test")}

    opt = create_optimizer_v2(model, "adamw", lr=args.lr, weight_decay=args.weight_decay,
                              layer_decay=args.layer_decay)
    for g in opt.param_groups:
        g["base_lr"] = args.lr * g.get("lr_scale", 1.0)
    scaler = torch.amp.GradScaler(enabled=amp)
    # warmup: decay ramps up from ~0, so early EMA weights track training instead of the untrained init
    ema = ModelEmaV3(model, decay=args.ema, use_warmup=True) if args.ema > 0 else None

    start_epoch, best_f1, history = 0, -1.0, []
    ckpt_path = run_dir / "last.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.module.load_state_dict(ck["ema"])
        start_epoch, best_f1, history = ck["epoch"] + 1, ck["best_f1"], ck["history"]
        print(f"resumed at epoch {start_epoch}")

    steps_per_epoch = len(train_loader)
    total, warmup = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for i, (x, y, _) in enumerate(train_loader):
            step = epoch * steps_per_epoch + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, g["base_lr"])
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(x.to(device, non_blocking=True)), y.to(device),
                                       label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model, step=step)
            running += loss.item()

        eval_model = ema.module if ema else model
        lv, yv = predict(eval_model, eval_loaders["val"], device, amp)
        val = summarize(lv, yv, CLASS_NAMES, CLUSTERS)
        history.append({"epoch": epoch, "train_loss": running / steps_per_epoch,
                        "val_top1": val["top1"], "val_macro_f1": val["macro_f1"]})
        print(f"epoch {epoch + 1}/{args.epochs}  loss {running / steps_per_epoch:.3f}  "
              f"val top1 {val['top1']:.3f}  macro-F1 {val['macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save({"model": eval_model.state_dict(), "model_name": args.model,
                        "img_size": args.img_size, "classes": CLASS_NAMES}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.module.state_dict() if ema else None, "epoch": epoch,
                    "best_f1": best_f1, "history": history}, ckpt_path)

    # Final: best checkpoint, temperature fitted on val, scored on unseen reserves
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["model"])
    lv, yv = predict(model, eval_loaders["val"], device, amp)
    lt, yt = predict(model, eval_loaders["test"], device, amp)
    t = fit_temperature(lv, yv)
    results = {"model": args.model, "args": {k: str(v) for k, v in vars(args).items()}, "history": history,
               "val": summarize(lv, yv, CLASS_NAMES, CLUSTERS, t), "test": summarize(lt, yt, CLASS_NAMES, CLUSTERS, t)}
    out = ROOT / "reports" / "results" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    np.savez(out / "test_logits.npz", logits=lt, labels=yt)
    print(f"best val macro-F1 {best_f1:.3f} | test top1 {results['test']['top1']:.3f} "
          f"macro-F1 {results['test']['macro_f1']:.3f} | {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
