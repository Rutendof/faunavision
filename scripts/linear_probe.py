"""Model 1: linear probe and k-NN on frozen backbone features.

Example:
    python scripts/linear_probe.py --features data/features --run dinov2_vitb14_probe

Trains a linear classifier on the train features (weight decay chosen on
validation macro-F1), fits a temperature on validation, and reports
validation (held-out cameras) and test (unseen reserves) results.
Writes reports/results/<run>/{metrics.json, report.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faunavision.data.classes import CLASSES, COMMON_NAMES  # noqa: E402
from faunavision.data.dataset import CLASS_INDEX, CLASS_NAMES  # noqa: E402
from faunavision.metrics import fit_temperature, pair_accuracy, summarize  # noqa: E402

CLUSTERS = {c: CLASSES[c].cluster for c in CLASS_NAMES}
LOOKALIKE_PAIRS = [
    ("white_rhino", "black_rhino"),
    ("plains_zebra", "mountain_zebra"),
    ("blue_wildebeest", "black_wildebeest"),
    ("spotted_hyena", "brown_hyena"),
    ("bontebok", "blesbok"),
    ("kudu", "nyala"),
    ("common_duiker", "steenbok"),
    ("black_backed_jackal", "cape_fox"),
]


def load(path: Path):
    d = np.load(path)  # plain arrays only; pickle stays disabled
    return d["features"].astype(np.float32), d["labels"].astype(np.int64)


def train_linear(x, y, n_classes, weight_decay, epochs=100, lr=1e-3, balanced=True, device="cpu", seed=0):
    torch.manual_seed(seed)
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    head = torch.nn.Linear(x.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    counts = torch.bincount(y, minlength=n_classes).float()
    weight = (counts.sum() / (counts.clamp(min=1) * (counts > 0).sum())) if balanced else None
    if weight is not None:
        weight[counts == 0] = 0
    batch = 4096
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=device)
        for i in range(0, len(y), batch):
            b = perm[i : i + batch]
            loss = F.cross_entropy(head(x[b]), y[b], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return head


def logits_of(head, x, device):
    with torch.no_grad():
        return head(torch.from_numpy(x).to(device)).cpu().numpy()


def knn_logits(train_x, train_y, x, n_classes, k=20, tau=0.07, device="cpu"):
    """Cosine k-NN with temperature-weighted votes (DINO-style), as logits."""
    tx = F.normalize(torch.from_numpy(train_x).to(device), dim=1)
    ty = torch.from_numpy(train_y).to(device)
    out = []
    for i in range(0, len(x), 2048):
        q = F.normalize(torch.from_numpy(x[i : i + 2048]).to(device), dim=1)
        sim, idx = (q @ tx.T).topk(min(k, len(ty)), dim=1)
        votes = torch.zeros(len(q), n_classes, device=device)
        votes.scatter_add_(1, ty[idx], (sim / tau).exp())
        out.append(torch.log(votes + 1e-8).cpu().numpy())
    return np.concatenate(out)


def pair_table(logits, y):
    pred = logits.argmax(1)
    rows = {}
    for a, b in LOOKALIKE_PAIRS:
        acc, n = pair_accuracy(pred, y, CLASS_INDEX[a], CLASS_INDEX[b])
        rows[f"{a} vs {b}"] = {"accuracy": acc, "n": n}
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=ROOT / "data" / "features")
    p.add_argument("--run", default="dinov2_probe")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--weight-decays", default="1e-4,1e-3,1e-2,1e-1")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(CLASS_NAMES)
    xtr, ytr = load(args.features / "train.npz")
    xva, yva = load(args.features / "val.npz")
    xte, yte = load(args.features / "test.npz")
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr_s, xva_s, xte_s = (xtr - mu) / sd, (xva - mu) / sd, (xte - mu) / sd

    # Choose weight decay on validation macro-F1
    best = None
    for wd in map(float, args.weight_decays.split(",")):
        head = train_linear(xtr_s, ytr, n, wd, args.epochs, device=device)
        f1 = summarize(logits_of(head, xva_s, device), yva, CLASS_NAMES, CLUSTERS)["macro_f1"]
        print(f"weight_decay={wd:g}: val macro-F1 {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, wd, head)
    _, wd, head = best

    results = {"model": "linear probe", "weight_decay": wd, "n_train": int(len(ytr))}
    lv, lt = logits_of(head, xva_s, device), logits_of(head, xte_s, device)
    t = fit_temperature(lv, yva)
    results["linear"] = {
        "val": summarize(lv, yva, CLASS_NAMES, CLUSTERS, t) | {"pairs": pair_table(lv, yva)},
        "test": summarize(lt, yte, CLASS_NAMES, CLUSTERS, t) | {"pairs": pair_table(lt, yte)},
        "test_uncalibrated_ece": summarize(lt, yte, CLASS_NAMES, CLUSTERS)["ece"],
    }
    kv, kt = knn_logits(xtr, ytr, xva, n, device=device), knn_logits(xtr, ytr, xte, n, device=device)
    results["knn"] = {
        "val": summarize(kv, yva, CLASS_NAMES, CLUSTERS),
        "test": summarize(kt, yte, CLASS_NAMES, CLUSTERS) | {"pairs": pair_table(kt, yte)},
    }

    out = ROOT / "reports" / "results" / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render(args.run, results), encoding="utf-8")
    lin = results["linear"]
    print(f"linear: val top1 {lin['val']['top1']:.3f} | test top1 {lin['test']['top1']:.3f} "
          f"macro-F1 {lin['test']['macro_f1']:.3f} | report {out / 'report.md'}")


def render(run: str, r: dict) -> str:
    lin, knn = r["linear"], r["knn"]
    f = lambda v: "n/a" if v != v else f"{v:.1%}"  # noqa: E731  (NaN-safe)
    lines = [
        f"# Results: {run}",
        "",
        "Validation = held-out cameras in training reserves. Test = unseen reserves (Pilanesberg, Kruger, Camdeboo).",
        "",
        "| Head | Split | Top-1 | Top-5 | Macro-F1 | Balanced acc | ECE | Near-miss share of errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, res in (("Linear", lin), ("k-NN", knn)):
        for split in ("val", "test"):
            s = res[split]
            lines.append(f"| {name} | {split} | {f(s['top1'])} | {f(s['top5'])} | {f(s['macro_f1'])} | "
                         f"{f(s['balanced_accuracy'])} | {f(s['ece'])} | {f(s['near_miss_share_of_errors'])} |")
    lines += ["", f"Calibration: test ECE {f(lin['test_uncalibrated_ece'])} before, {f(lin['test']['ece'])} after "
              f"temperature scaling (T = {lin['test']['temperature']:.2f}).", "",
              "## Lookalike pairs (test, linear head)", "",
              "Among images of either species assigned to one of the two, the share given the right one.", "",
              "| Pair | Accuracy | n |", "|---|---|---|"]
    for pair, v in lin["test"]["pairs"].items():
        a, b = pair.split(" vs ")
        lines.append(f"| {COMMON_NAMES[a]} vs {COMMON_NAMES[b]} | {f(v['accuracy'])} | {v['n']} |")
    lines += ["", "## Per-class recall (test, linear head)", "", "| Class | Recall | F1 | Test images |", "|---|---|---|---|"]
    for c, v in sorted(lin["test"]["per_class"].items(), key=lambda kv: kv[1]["recall"]):
        lines.append(f"| {COMMON_NAMES[c]} | {f(v['recall'])} | {f(v['f1'])} | {v['support']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
