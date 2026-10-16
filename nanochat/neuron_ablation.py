"""Weak vs strong MLP activation ablation, after "Neural Networks Are Already Clean" (Goodfire, 2026).

Claim under test: zeroing the weakest ~75% of each neuron's activations barely changes the model
(flips are near-ties), while zeroing the strongest few percent wrecks it. If meaning were packed into
weak activations (superposition), the weak ablation should hurt much more than a random ablation.

Thresholds are per neuron, over its nonzero ReLU^2 activations, fit on the first half of the val
batches and evaluated on the second half. Example:
    python -m scripts.neuron_ablation --checkpoint .nanochat-cache/mlx-ablation-main/final-hybrid/checkpoint \
        --data .nanochat-cache/mlx-ablation-main/data
"""
import os
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import argparse
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from nanochat.mlx_backend.checkpoint import load_checkpoint
from nanochat.mlx_backend.model import MLP, cross_entropy

STATE = {"mode": None, "layers": None, "captured": {}, "lo": {}, "hi": {}, "drop": 0.}


class AblatedMLP(MLP):
    def __call__(self, x):
        h = mx.square(nn.relu(self.c_fc(x)))
        i, mode = self.layer, STATE["mode"]
        if mode == "capture":
            STATE["captured"].setdefault(i, []).append(np.array(h.reshape(-1, h.shape[-1]).astype(mx.float32)))
        elif STATE["layers"] is None or i in STATE["layers"]:
            if mode == "weak":
                h = mx.where(h < STATE["lo"][i], 0., h)
            elif mode == "strong":
                h = mx.where(h > STATE["hi"][i], 0., h)
            elif mode == "random":
                h = mx.where(mx.random.uniform(shape=h.shape) < STATE["drop"], 0., h)
        return self.c_proj(h)


class Calibration:
    """Per-neuron sorted activations, for nonzero-quantile thresholds and removed-mass accounting."""

    def __init__(self, acts):
        self.sorted = {i: np.sort(a, axis=0) for i, a in acts.items()}
        self.cumsum = {i: np.cumsum(s, axis=0, dtype=np.float64) for i, s in self.sorted.items()}
        self.nonzero = {i: (s > 0).sum(0) for i, s in self.sorted.items()}
        self.total = sum(c[-1].sum() for c in self.cumsum.values())

    def index(self, i, q):
        n, nz = self.sorted[i].shape[0], self.nonzero[i]
        return np.clip(n - nz + np.floor(q * nz).astype(int), 0, n - 1)

    def threshold(self, i, q):
        t = self.sorted[i][self.index(i, q), np.arange(self.sorted[i].shape[1])]
        return np.where(self.nonzero[i] > 0, t, np.inf).astype(np.float32)

    def mass_below(self, q, layers):
        out = 0.
        for i in layers:
            idx, c = self.index(i, q), self.cumsum[i]
            below = np.where(idx > 0, c[np.maximum(idx - 1, 0), np.arange(c.shape[1])], 0.)
            out += np.where(self.nonzero[i] > 0, below, 0.).sum()
        return out / self.total

    def mass_above(self, q, layers):
        out = 0.
        for i in layers:
            c = self.cumsum[i]
            out += np.where(self.nonzero[i] > 0, c[-1] - c[self.index(i, q), np.arange(c.shape[1])], 0.).sum()
        return out / self.total


def run(model, rows):
    out = {"loss": [], "pred": [], "share": [], "correct": []}
    for row in rows:
        x, y = mx.array(row[:, :-1]), mx.array(row[:, 1:])
        logits = model(x)
        top2 = mx.topk(logits, 2, axis=-1)
        gap = mx.abs(top2[..., 0] - top2[..., 1])
        pred = mx.argmax(logits, axis=-1)
        values = [cross_entropy(logits, y, "none"), pred, mx.sigmoid(gap), pred == y]
        mx.eval(values)
        for k, v in zip(out, values):
            out[k].append(np.array(v).ravel())
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(name, result, base, mass):
    changed = result["pred"] != base["pred"]
    return {
        "config": name,
        "val_loss": float(result["loss"].mean()),
        "d_loss": float(result["loss"].mean() - base["loss"].mean()),
        "top1_acc": float(result["correct"].mean()),
        "changed": float(changed.mean()),
        "flip_top2_share": float(np.median(base["share"][changed])) if changed.any() else None,
        "mass_removed": None if mass is None else float(mass),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="Prepared data directory containing val.npy")
    p.add_argument("--output", help="Write results JSON here")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    mx.random.seed(args.seed)
    model, meta = load_checkpoint(args.checkpoint)
    for i, block in enumerate(model.transformer["h"]):
        block.mlp.__class__ = AblatedMLP
        block.mlp.layer = i
    n_layer = meta["config"]["n_layer"]
    every = list(range(n_layer))
    val = np.load(Path(args.data) / "val.npy")
    calib_rows, eval_rows = val[:len(val) // 2], val[len(val) // 2:]

    STATE["mode"] = "capture"
    for row in calib_rows:
        mx.eval(model(mx.array(row[:, :-1])))
    cal = Calibration({i: np.concatenate(a) for i, a in STATE["captured"].items()})
    STATE["captured"].clear()
    zero_frac = 1 - sum(n.sum() for n in cal.nonzero.values()) / sum(s.size for s in cal.sorted.values())

    STATE["mode"] = None
    base = run(model, eval_rows)
    results = [summarize("baseline", base, base, 0.)]

    def ablate(name, mode, layers=None, q=None, drop=0., mass=None):
        STATE.update(mode=mode, layers=layers, drop=drop)
        for i in (layers or every):
            if mode == "weak":
                STATE["lo"][i] = mx.array(cal.threshold(i, q))
            if mode == "strong":
                STATE["hi"][i] = mx.array(cal.threshold(i, 1 - q))
        results.append(summarize(name, run(model, eval_rows), base, mass))
        STATE["mode"] = None

    for q in (.25, .5, .75, .9):
        ablate(f"weak {q:.0%} (all layers)", "weak", q=q, mass=cal.mass_below(q, every))
    for f in (.01, .07):
        ablate(f"strong {f:.0%} (all layers)", "strong", q=f, mass=cal.mass_above(1 - f, every))
    # Strongest slice carrying the same total activation mass as the weakest 75%.
    target = cal.mass_below(.75, every)
    grid = np.linspace(.0005, .5, 1000)
    f = grid[np.argmin([abs(cal.mass_above(1 - g, every) - target) for g in grid])]
    ablate(f"strong {f:.1%} (mass-matched to weak 75%)", "strong", q=f, mass=cal.mass_above(1 - f, every))
    ablate("random 75% of nonzero (count-matched)", "random", drop=.75)
    for i in every:
        ablate(f"layer {i}: weak 75%", "weak", layers=[i], q=.75, mass=cal.mass_below(.75, [i]))
        ablate(f"layer {i}: strong 7%", "strong", layers=[i], q=.07, mass=cal.mass_above(.93, [i]))

    report = {
        "checkpoint": args.checkpoint,
        "config": meta["config"],
        "calib_tokens": int(calib_rows[..., :-1].size),
        "eval_tokens": int(eval_rows[..., :-1].size),
        "exact_zero_fraction": float(zero_frac),
        "median_top2_share_all_tokens": float(np.median(base["share"])),
        "results": results,
    }
    print(f"{args.checkpoint}\n{report['config']}  calib {report['calib_tokens']} tok, eval {report['eval_tokens']} tok")
    print(f"exact-zero ReLU^2 activations: {zero_frac:.1%}; median top-2 share over all tokens: "
          f"{report['median_top2_share_all_tokens']:.2f}")
    print(f"{'config':44s} {'loss':>7s} {'dloss':>7s} {'acc':>6s} {'changed':>8s} {'flip-share':>10s} {'mass':>6s}")
    for r in results:
        share = "-" if r["flip_top2_share"] is None else f"{r['flip_top2_share']:.2f}"
        mass = "-" if r["mass_removed"] is None else f"{r['mass_removed']:.1%}"
        print(f"{r['config']:44s} {r['val_loss']:7.3f} {r['d_loss']:+7.3f} {r['top1_acc']:6.1%} "
              f"{r['changed']:8.1%} {share:>10s} {mass:>6s}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
