"""Weak vs strong SwiGLU activation ablation + gate/value sign split, on a cs336 transformer_lm checkpoint.

Replicates two claims from "Neural Networks Are Already Clean" (Goodfire, 2026):
  1. Zeroing each neuron's weakest ~75% of activations barely changes predictions (and the flips are
     near-ties), while zeroing its strongest few percent wrecks the model.
  2. In SwiGLU, h = silu(gate) * value. When gate < 0 and value < 0 the output is a small *positive*
     "double-negative echo". For a few % of neurons this echo dominates the top of the positive list,
     which makes a top-k dashboard mislabel the neuron.

Thresholds are per neuron over |h|, fit on calibration windows and evaluated on disjoint windows.
Reads the cs336 repo read-only. Run with the repo's venv:
    ../assignment1-basics/.venv/bin/python swiglu_ablation.py --run comic-cherry-166
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent / "assignment1-basics"
sys.path.insert(0, str(REPO))
from cs336_basics.gpt import transformer_lm  # noqa: E402

STATE = {"mode": None, "layers": None, "capture": {}, "lo": {}, "hi": {}, "drop": 0.}


def patched_ff(ff, layer):
    def forward(x):
        a, b = ff.w1(x), ff.w3(x)
        h = F.silu(a) * b
        mode = STATE["mode"]
        if mode == "capture":
            STATE["capture"].setdefault(layer, []).append((h.flatten(0, -2).float().cpu(), a.flatten(0, -2).float().cpu()))
        elif STATE["layers"] is None or layer in STATE["layers"]:
            if mode == "weak":
                h = torch.where(h.abs() < STATE["lo"][layer], 0., h)
            elif mode == "strong":
                h = torch.where(h.abs() > STATE["hi"][layer], 0., h)
            elif mode == "random":
                h = torch.where(torch.rand_like(h) < STATE["drop"], 0., h)
            elif mode == "double_negative":
                h = torch.where((a < 0) & (b < 0), 0., h)
            elif mode == "gate_closed":
                h = torch.where(a < 0, 0., h)
        return ff.w2(h)
    return forward


class Calibration:
    """Per-neuron sorted |h|, for quantile thresholds and removed-mass accounting."""

    def __init__(self, mags):
        self.sorted = {i: np.sort(m, axis=0) for i, m in mags.items()}
        self.cumsum = {i: np.cumsum(s, axis=0, dtype=np.float64) for i, s in self.sorted.items()}
        self.total = sum(c[-1].sum() for c in self.cumsum.values())

    def index(self, i, q):
        return min(int(q * self.sorted[i].shape[0]), self.sorted[i].shape[0] - 1)

    def threshold(self, i, q):
        return self.sorted[i][self.index(i, q)]

    def mass_below(self, q, layers):
        return sum(self.cumsum[i][self.index(i, q) - 1].sum() for i in layers) / self.total

    def mass_above(self, q, layers):
        return sum((self.cumsum[i][-1] - self.cumsum[i][self.index(i, q)]).sum() for i in layers) / self.total


@torch.no_grad()
def run(model, windows, cfg, device):
    out = {"loss": [], "pred": [], "share": [], "correct": []}
    for chunk in torch.split(windows, 16):
        x, y = chunk[:, :-1].to(device), chunk[:, 1:].to(device)
        logits = model(x, cfg["num_heads"], cfg["d_model"]).float()
        top2 = logits.topk(2, dim=-1).values
        pred = logits.argmax(-1)
        values = [F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none"),
                  pred.flatten(), torch.sigmoid(top2[..., 0] - top2[..., 1]).flatten(), (pred == y).flatten()]
        for k, v in zip(out, values):
            out[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


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
    p.add_argument("--run", default="comic-cherry-166", help="Checkpoint run name under checkpoints/")
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--val", default=str(REPO / "out" / "tinystories_valid.npy"))
    p.add_argument("--vocab", default=str(REPO / "out" / "tinystories_vocab.pkl"))
    p.add_argument("--num-heads", type=int, default=16, help="Not recoverable from weights; from wandb config")
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--windows", type=int, default=128, help="Windows each for calibration and eval")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--output", help="Write results JSON here")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    sd = torch.load(REPO / "checkpoints" / args.run / args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    vocab, d_model = sd["tok_embd.weight"].shape
    d_ff = sd["blocks.0.ff.w1.weight"].shape[0]
    cfg = dict(vocab_size=vocab, context_length=args.context, num_layers=n_layer, d_model=d_model,
               num_heads=args.num_heads, d_ff=d_ff, rope_theta=args.rope_theta)
    model = transformer_lm(**cfg)
    model.load_state_dict(sd)
    model.to(args.device).eval()
    for i, block in enumerate(model.blocks):
        block.ff.forward = patched_ff(block.ff, i)
    every = list(range(n_layer))

    val = np.load(args.val, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    stride = args.context + 1  # non-overlapping windows, so calibration and eval tokens are disjoint
    starts = rng.choice(len(val) // stride - 1, 2 * args.windows, replace=False) * stride
    windows = torch.from_numpy(np.stack([val[s:s + stride] for s in starts]).astype(np.int64))
    calib, evalw = windows[:args.windows], windows[args.windows:]

    STATE["mode"] = "capture"
    run(model, calib, cfg, args.device)
    STATE["mode"] = None
    acts = {i: (torch.cat([h for h, _ in v]).numpy(), torch.cat([a for _, a in v]).numpy()) for i, v in STATE["capture"].items()}
    STATE["capture"].clear()
    cal = Calibration({i: np.abs(h) for i, (h, _) in acts.items()})

    # Sign split: for each neuron, how many of its top-k positive activations come with the gate closed?
    calib_tokens = calib[:, :-1].reshape(-1).numpy()
    with open(args.vocab, "rb") as f:
        id_to_bytes = pickle.load(f)

    def context(t):
        lo = max(t - 12, (t // args.context) * args.context)  # stay inside the firing token's window
        text = b"".join(id_to_bytes[int(j)] for j in calib_tokens[lo:t]).decode("utf-8", "replace")
        tok = id_to_bytes[int(calib_tokens[t])].decode("utf-8", "replace")
        return (text + "[[" + tok + "]]").replace("\n", " ")

    sign_split = {"per_layer": [], "examples": []}
    for i, (h, a) in acts.items():
        top = np.argpartition(-h, args.topk, axis=0)[:args.topk]  # (k, d_ff) token indices
        closed = (np.take_along_axis(a, top, axis=0) < 0).mean(0)  # share of top-k with gate < 0
        echo_dominated = closed >= .5
        sign_split["per_layer"].append({
            "layer": i,
            "neurons": int(h.shape[1]),
            "echo_dominated_frac": float(echo_dominated.mean()),
            "mean_gate_closed_share_of_topk": float(closed.mean()),
            "share_of_all_pos_acts_from_double_negative": float(((a < 0) & (h > 0)).sum() / max((h > 0).sum(), 1)),
        })
        for n in np.argsort(-closed)[:3]:
            order = top[np.argsort(-h[top[:, n], n]), n][:5]
            gate_open = np.where(a[:, n] > 0, h[:, n], -np.inf)
            open_top = [t for t in np.argsort(-gate_open)[:3] if np.isfinite(gate_open[t])]
            neg_top = np.argsort(h[:, n])[:3]
            sign_split["examples"].append({
                "neuron": f"L{i}/N{n}", "gate_closed_share": float(closed[n]),
                "top_positive (what a top-k dashboard shows)": [(round(float(h[t, n]), 3), context(t)) for t in order],
                "top_positive_gate_open": [(round(float(h[t, n]), 3), context(t)) for t in open_top],
                "top_negative (gate open, value < 0)": [(round(float(h[t, n]), 3), context(t)) for t in neg_top],
            })
    del acts

    base = run(model, evalw, cfg, args.device)
    results = [summarize("baseline", base, base, 0.)]

    def ablate(name, mode, layers=None, q=None, drop=0., mass=None):
        STATE.update(mode=mode, layers=layers, drop=drop)
        for i in (layers or every):
            if mode == "weak":
                STATE["lo"][i] = torch.from_numpy(cal.threshold(i, q)).to(args.device)
            if mode == "strong":
                STATE["hi"][i] = torch.from_numpy(cal.threshold(i, 1 - q)).to(args.device)
        results.append(summarize(name, run(model, evalw, cfg, args.device), base, mass))
        STATE["mode"] = None

    for q in (.25, .5, .75, .9):
        ablate(f"weak {q:.0%} (all layers)", "weak", q=q, mass=cal.mass_below(q, every))
    for f in (.01, .07):
        ablate(f"strong {f:.0%} (all layers)", "strong", q=f, mass=cal.mass_above(1 - f, every))
    target = cal.mass_below(.75, every)
    grid = np.linspace(.0005, .5, 1000)
    f = grid[np.argmin([abs(cal.mass_above(1 - g, every) - target) for g in grid])]
    ablate(f"strong {f:.1%} (mass-matched to weak 75%)", "strong", q=f, mass=cal.mass_above(1 - f, every))
    ablate("random 75% (count-matched)", "random", drop=.75)
    ablate("double-negative regime zeroed", "double_negative")
    ablate("gate-closed (gate<0) zeroed", "gate_closed")
    for i in every:
        ablate(f"layer {i}: weak 75%", "weak", layers=[i], q=.75, mass=cal.mass_below(.75, [i]))
        ablate(f"layer {i}: strong 7%", "strong", layers=[i], q=.07, mass=cal.mass_above(.93, [i]))

    report = {"run": args.run, "ckpt": args.ckpt, "config": cfg, "tokens_per_split": int(calib[:, :-1].numel()),
              "median_top2_share_all_tokens": float(np.median(base["share"])), "results": results,
              "sign_split": sign_split}
    print(f"{args.run}/{args.ckpt}  {cfg}\n{report['tokens_per_split']} calibration tokens, same number eval tokens")
    print(f"median top-2 share over all eval tokens: {report['median_top2_share_all_tokens']:.2f}\n")
    print(f"{'config':44s} {'loss':>7s} {'dloss':>7s} {'acc':>6s} {'changed':>8s} {'flip-share':>10s} {'|h| mass':>8s}")
    for r in results:
        share = "-" if r["flip_top2_share"] is None else f"{r['flip_top2_share']:.2f}"
        mass = "-" if r["mass_removed"] is None else f"{r['mass_removed']:.1%}"
        print(f"{r['config']:44s} {r['val_loss']:7.3f} {r['d_loss']:+7.3f} {r['top1_acc']:6.1%} "
              f"{r['changed']:8.1%} {share:>10s} {mass:>8s}")
    print(f"\nSwiGLU sign split (top-{args.topk} positive activations per neuron):")
    for r in sign_split["per_layer"]:
        print(f"  layer {r['layer']}: {r['echo_dominated_frac']:.1%} of neurons have >=50% of top-k from gate<0 echo; "
              f"mean gate-closed share {r['mean_gate_closed_share_of_topk']:.1%}; "
              f"{r['share_of_all_pos_acts_from_double_negative']:.1%} of all positive acts are double-negative")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
