"""Watch SwiGLU neurons grow across training checkpoints: recursive generalization vs arbitrary packing.

From "Neural Networks Are Already Clean" (Goodfire, 2026): when a neuron picks up new top contexts over
training, generalization predicts they are *related* to what it already responded to (it branches
outward locally), while superposition / arbitrary packing predicts *unrelated* additions.

For each checkpoint we record every neuron's top-k contexts (by positive SwiGLU output) on a fixed set
of held-out windows. For each neuron and each consecutive pair of checkpoints we take the entrants (new
top-k contexts) and measure their similarity to the neuron's previous top-k set, using a model-free
TF-IDF bag of the surrounding tokens. Two nulls: the same entrants against a *different* neuron's
previous set, and random contexts against this neuron's previous set.

Reads the cs336 repo read-only:
    ../assignment1-basics/.venv/bin/python swiglu_growth.py --run bumbling-brook-124
"""
import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from swiglu_ablation import REPO, transformer_lm


def load(path, num_heads, context, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    vocab, d_model = sd["tok_embd.weight"].shape
    cfg = dict(vocab_size=vocab, context_length=context, d_model=d_model, num_heads=num_heads, rope_theta=10000.0,
               num_layers=1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")),
               d_ff=sd["blocks.0.ff.w1.weight"].shape[0])
    model = transformer_lm(**cfg)
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, ckpt["iteration"]


@torch.no_grad()
def top_contexts(model, cfg, windows, k, device):
    """Per layer: (k, d_ff) flat token indices of each neuron's strongest positive outputs, plus values."""
    best = {}

    def hook(layer, ff):
        def forward(x):
            h = F.silu(ff.w1(x)) * ff.w3(x)
            flat = h.flatten(0, -2).float()
            idx = torch.arange(flat.shape[0], device=flat.device)[:, None].expand_as(flat) + STEP["offset"]
            if layer in best:
                flat, idx = torch.cat([best[layer][0], flat]), torch.cat([best[layer][1], idx])
            vals, pos = flat.topk(k, dim=0)
            best[layer] = (vals, idx.gather(0, pos))
            return ff.w2(h)
        return forward

    STEP = {"offset": 0}
    for i, block in enumerate(model.blocks):
        block.ff.forward = hook(i, block.ff)
    for chunk in torch.split(windows[:, :-1], 32):
        model(chunk.to(device), cfg["num_heads"], cfg["d_model"])
        STEP["offset"] += chunk.numel()
    return {i: (v.cpu().numpy(), p.cpu().numpy()) for i, (v, p) in best.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="bumbling-brook-124")
    p.add_argument("--val", default=str(REPO / "out" / "tinystories_valid.npy"))
    p.add_argument("--vocab", default=str(REPO / "out" / "tinystories_vocab.pkl"))
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--windows", type=int, default=512)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--span", type=int, default=24, help="Tokens of context (ending at the firing token) for TF-IDF")
    p.add_argument("--sample", type=int, default=256, help="Neurons per layer to score")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--output", help="Write results JSON here")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    ckpt_dir = REPO / "checkpoints" / args.run
    paths = sorted(ckpt_dir.glob("ckpt_[0-9]*.pt"), key=lambda q: int(re.findall(r"\d+", q.name)[0]))
    val = np.load(args.val, mmap_mode="r")
    stride = args.context + 1
    starts = rng.choice(len(val) // stride - 1, args.windows, replace=False) * stride
    windows = torch.from_numpy(np.stack([val[s:s + stride] for s in starts]).astype(np.int64))
    tokens = windows[:, :-1].reshape(-1).numpy()
    T = args.context

    steps, tops = [], []
    for path in paths:
        model, cfg, step = load(path, args.num_heads, args.context, args.device)
        tops.append(top_contexts(model, cfg, windows, args.k, args.device))
        steps.append(step)
        print(f"step {step}: recorded top-{args.k} contexts", flush=True)
    n_layer, d_ff, vocab = cfg["num_layers"], cfg["d_ff"], cfg["vocab_size"]

    # Model-free context representation: TF-IDF over the span of tokens ending at the firing token
    # (within its own window), with the firing token counted 3x.
    df = np.zeros(vocab)
    for w in windows[:, :-1].numpy():
        df[np.unique(w)] += 1
    idf = np.log((1 + len(windows)) / (1 + df)) + 1

    def embed(positions):
        out = np.zeros((len(positions), vocab), dtype=np.float32)
        for r, t in enumerate(positions):
            lo = max(t - args.span + 1, (t // T) * T)
            np.add.at(out[r], tokens[lo:t + 1], 1.)
            out[r, tokens[t]] += 2.
        out *= idf
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    with open(args.vocab, "rb") as f:
        id_to_bytes = pickle.load(f)

    def text(t):
        lo = max(t - 14, (t // T) * T)
        before = b"".join(id_to_bytes[int(j)] for j in tokens[lo:t]).decode("utf-8", "replace")
        return (before + "[[" + id_to_bytes[int(tokens[t])].decode("utf-8", "replace") + "]]").replace("\n", " ")

    def best_match(a, a_pos, b, b_pos):
        """Max cosine of each row of a to b, ignoring pairs from the same story window."""
        sims = a @ b.T
        sims[np.asarray(a_pos)[:, None] // T == np.asarray(b_pos)[None, :] // T] = -np.inf
        best = sims.max(1)
        return best[np.isfinite(best)].tolist()

    random_pool = rng.choice(len(tokens), 4000, replace=False)
    random_emb = embed(random_pool)
    layers = []
    trajectories = []
    for i in range(n_layer):
        neurons = np.sort(rng.choice(d_ff, min(args.sample, d_ff), replace=False))
        partner = np.roll(neurons, 1)  # a different neuron, for the cross-neuron null
        own, cross, rand, jac_step, jac_total, drift = [], [], [], [], [], []
        for n, m in zip(neurons, partner):
            sets = [set(tops[s][i][1][:, n].tolist()) for s in range(len(steps))]
            other = [set(tops[s][i][1][:, m].tolist()) for s in range(len(steps))]
            jac_step += [len(a & b) / len(a | b) for a, b in zip(sets, sets[1:])]
            jac_total.append(len(sets[0] & sets[-1]) / len(sets[0] | sets[-1]))
            for s in range(1, len(steps)):
                entrants = sorted(sets[s] - sets[s - 1])
                if not entrants:
                    continue
                e = embed(entrants)
                prev_pos, other_pos = sorted(sets[s - 1]), sorted(other[s - 1])
                prev, prev_other = embed(prev_pos), embed(other_pos)
                pick = rng.choice(len(random_pool), len(entrants), replace=False)
                own += best_match(e, entrants, prev, prev_pos)
                cross += best_match(e, entrants, prev_other, other_pos)
                rand += best_match(random_emb[pick], random_pool[pick], prev, prev_pos)
            first_pos, last_pos = sorted(sets[0]), sorted(sets[-1])
            drift.append(float(np.mean(best_match(embed(last_pos), last_pos, embed(first_pos), first_pos) or [np.nan])))
        own, cross, rand = map(np.array, (own, cross, rand))
        null95 = np.quantile(cross, .95)
        layers.append({
            "layer": i, "neurons_scored": len(neurons),
            "jaccard_consecutive_mean": float(np.mean(jac_step)),
            "jaccard_first_last_mean": float(np.mean(jac_total)),
            "entrants": int(len(own)),
            "entrant_sim_to_own_prev_median": float(np.median(own)),
            "entrant_sim_to_other_neuron_prev_median": float(np.median(cross)),
            "random_ctx_sim_to_own_prev_median": float(np.median(rand)),
            "entrants_above_cross_null_p95": float((own > null95).mean()),
            "entrants_below_cross_null_median": float((own < np.median(cross)).mean()),
            "last_set_sim_to_first_set_mean": float(np.mean(drift)),
        })
        # The most-changed neurons are where unrelated packing would be easiest to see.
        moved = neurons[np.argsort([len(set(tops[0][i][1][:, n]) & set(tops[-1][i][1][:, n])) for n in neurons])[:3]]
        for n in moved:
            snaps = {}
            for s in (0, len(steps) // 2, len(steps) - 1):
                vals, pos = tops[s][i]
                snaps[f"step {steps[s]}"] = [(round(float(vals[r, n]), 2), text(int(pos[r, n]))) for r in range(5)]
            trajectories.append({"neuron": f"L{i}/N{n}", **snaps})

    report = {"run": args.run, "steps": steps, "config": cfg, "tokens": int(len(tokens)), "k": args.k,
              "layers": layers, "trajectories": trajectories}
    print(f"\n{args.run}: {len(steps)} checkpoints {steps[0]}..{steps[-1]}, {len(tokens)} tokens, top-{args.k}")
    print("Entrant similarity = max TF-IDF cosine of a new top-k context to the neuron's previous top-k set.\n")
    for r in layers:
        print(f"layer {r['layer']}: top-k overlap consecutive {r['jaccard_consecutive_mean']:.2f}, "
              f"first->last {r['jaccard_first_last_mean']:.2f}; {r['entrants']} entrants")
        print(f"   entrant sim  own prev {r['entrant_sim_to_own_prev_median']:.3f} | other neuron's prev "
              f"{r['entrant_sim_to_other_neuron_prev_median']:.3f} | random ctx vs own prev "
              f"{r['random_ctx_sim_to_own_prev_median']:.3f}")
        print(f"   {r['entrants_above_cross_null_p95']:.0%} of entrants beat the cross-neuron null's 95th pct; "
              f"{r['entrants_below_cross_null_median']:.0%} fall below its median")
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
