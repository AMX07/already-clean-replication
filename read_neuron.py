"""Read one SwiGLU neuron the way "Neural Networks Are Already Clean" (Goodfire, 2026) does.

Top-activating examples only give a first guess. This script then tries to break that guess:
  1. contexts   top contexts over >1M held-out tokens: gate-open positive pole, double-negative echo,
                negative pole, plus which tokens it fires on and predicts
  2. label      new sentences written from the label alone (should fire) and near misses (should not)
  3. ablation   zero the neuron: change in counting-prompt probabilities and val loss, vs random neurons
  4. steering   clamp the neuron high and generate greedily from neutral prompts
  5. circuit    upstream layer-(L-1) MLP neurons that drive it, and the tokens its output pushes
  6. match      the best-matching neuron in a second, independently trained model, run through 1-3

Reads the cs336 repo read-only:
    uv run --project ../assignment1-basics --no-sync python read_neuron.py --output read_L3N590
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from swiglu_ablation import REPO
from swiglu_growth import load
from cs336_basics.tokenizer import Tokenizer

HOOKS = []  # f(layer, gate, value, h) -> new h or None, applied inside every SwiGLU block


def install(model):
    for i, block in enumerate(model.blocks):
        def forward(x, ff=block.ff, i=i):
            a, b = ff.w1(x), ff.w3(x)
            h = F.silu(a) * b
            for f in HOOKS:
                out = f(i, a, b, h)
                if out is not None:
                    h = out
            return ff.w2(h)
        block.ff.forward = forward


class Reader:
    def __init__(self, model, cfg, tok, device):
        self.model, self.cfg, self.tok, self.device = model, cfg, tok, device
        install(model)

    @torch.no_grad()
    def logits(self, ids):
        return self.model(ids.to(self.device), self.cfg["num_heads"], self.cfg["d_model"]).float()

    @torch.no_grad()
    def neuron(self, inputs, layer, n, batch=32):
        """Gate, value and output of one neuron at every input position, flattened."""
        out = []

        def grab(i, a, b, h):
            if i == layer:
                out.append(torch.stack([a[..., n], b[..., n], h[..., n]], -1).flatten(0, -2).float().cpu())

        HOOKS.append(grab)
        for chunk in torch.split(inputs, batch):
            self.logits(chunk)
        HOOKS.remove(grab)
        a, b, h = torch.cat(out).numpy().T
        return a, b, h

    @torch.no_grad()
    def token_losses(self, inputs, targets, batch=32):
        return torch.cat([F.cross_entropy(self.logits(x).flatten(0, 1), y.to(self.device).flatten(), reduction="none").cpu()
                          for x, y in zip(torch.split(inputs, batch), torch.split(targets, batch))]).numpy()

    def encode(self, text):
        return torch.tensor([self.tok.encode("<|endoftext|>" + text)])

    def piece(self, i):
        return self.tok.decode([int(i)])

    def sentence_test(self, layer, n, peak, sentences):
        """Strongest response anywhere in each sentence, as a share of the corpus max."""
        rows = []
        for s in sentences:
            ids = self.encode(s)
            _, _, h = self.neuron(ids, layer, n)
            t = int(h.argmax())
            rows.append({"text": s, "share": float(h[t] / peak), "token": self.piece(ids[0, t])})
        return rows

    def prompt_probs(self, prompts, hook=None):
        if hook:
            HOOKS.append(hook)
        out = []
        for prompt, target in prompts:
            target_id = self.tok.encode(target)
            assert len(target_id) == 1, target
            out.append(float(self.logits(self.encode(prompt))[0, -1].softmax(-1)[target_id[0]]))
        if hook:
            HOOKS.remove(hook)
        return np.array(out)

    @torch.no_grad()
    def generate(self, prompt, steps=40):
        ids = self.encode(prompt)
        for _ in range(steps):
            ids = torch.cat([ids, self.logits(ids[:, -self.cfg["context_length"]:])[:, -1].argmax(-1, keepdim=True).cpu()], 1)
        return self.tok.decode(ids[0, len(self.encode(prompt)[0]):].tolist())


LABEL = ("Counting aloud: fires on the number word of a count in progress ('started to count, \"[One]', "
         "'One, [two]', 'one by [one]'), and predicts the comma or next item that follows")

# Written from the label alone, before measuring. Each should fire strongly (>= 30% of corpus max).
SHOULD_FIRE = [
    'Mia closed her eyes and began to count. "One',
    'Ben hid under the bed while Sue counted, "One, two',
    'The kids lined up and jumped into the pool one by one',
    'Grandpa pointed at the ducks and counted them: one, two, three',
    'Lily put the eggs in the basket one by one',
    'It was time for hide and seek. Anna faced the wall and counted, "One',
    'Max counted his cookies. One, two, three, four',
    'They took turns going down the slide, one by one',
    'The teacher clapped and the whole class counted together, "One, two, three',
    'The frog hopped across the stones one by one',
]
# Near misses: number words and counting talk that are not a count in progress. Should stay weak.
SHOULD_NOT = [
    'One day, a little girl named Mia went to the store',
    'Ben had one red ball and one blue ball',
    'Sue had two cats and three dogs',
    'Lily was three years old',
    'Tom wanted to count the stars in the sky',
    'She was the only one who could help',
    '"Which one do you want?" Mom asked',
    'Tim ate one apple and two pears',
    'The bird liked to fly high in the sky',
    'He found a coin with the number one on it',
]
# (prompt, next token) pairs: the next count word, or the comma after one.
COUNT_PROMPTS = [
    ('Spot closed his eyes and started to count. "One,', " two"),
    ('They counted together: one, two,', " three"),
    ('She counted her toys. One, two, three,', " four"),
    ('Anna faced the tree and started to count. "', "One"),
    ('Anna faced the tree and started to count. "One', ","),
    ('The kids went down the slide one by', " one"),
    ('He picked up the rocks one', " by"),
]
# Revised label, formed after sections 3-5 showed what the neuron writes, and tested with OUTPUT_PROMPTS.
REVISED_LABEL = ("A switch on the word just said: positive on a counting 'One'/'two'/'one by one', negative on a "
                 "story-opening 'One' -- its output suppresses time words (' day', ' night', ' evening') and "
                 "favours clause continuations (' because', ' where', ' but')")
# (prompt, next token): zeroing the neuron should raise ' day' after a counting "One", lower it after a
# story-opening "One" (where the neuron is negative), and lower the clause continuations.
OUTPUT_PROMPTS = [
    ('Anna faced the tree and started to count. "One', " day"),
    ('The bird lived in a big tree near a small house. One', " day"),
    ('They took turns cutting the paper with the scissors. They were very careful', " because"),
    ('Toto watched his friend play with the shiny new ball. Toto felt envious', " because"),
    ('Once upon a time, there was a purple cat. The cat loved to skip', " all"),
]
NEUTRAL_PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She liked to",
    "Tom and his dog went to the park. They",
    "The sun was shining and the birds were singing in the garden.",
]
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


def set_neurons(layer, idx, value=0.):
    """Hook that overwrites neurons idx of `layer` with a constant."""
    def f(i, a, b, h):
        if i == layer:
            h = h.clone()
            h[..., idx] = value
            return h
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="bumbling-brook-124")
    p.add_argument("--ckpt", default="ckpt_50000.pt")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--neuron", type=int, default=590)
    p.add_argument("--match-run", default="comic-cherry-166")
    p.add_argument("--match-ckpt", default="ckpt_final.pt")
    p.add_argument("--tokens", type=int, default=1_200_000, help="Held-out tokens for top contexts")
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--sections", default="contexts,label,ablation,steering,circuit,match")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--output", help="Write <output>.txt and <output>.json")
    args = p.parse_args()
    sections = set(args.sections.split(","))
    rng = np.random.default_rng(args.seed)
    lines, report = [], {"run": args.run, "ckpt": args.ckpt, "neuron": f"L{args.layer}/N{args.neuron}"}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    tok = Tokenizer.from_files(str(REPO / "out" / "tinystories_vocab.pkl"), str(REPO / "out" / "tinystories_merges.pkl"),
                               ["<|endoftext|>"])
    model, cfg, step = load(REPO / "checkpoints" / args.run / args.ckpt, args.num_heads, args.context, args.device)
    r = Reader(model, cfg, tok, args.device)
    T, L, N = args.context, args.layer, args.neuron
    val = np.load(REPO / "out" / "tinystories_valid.npy", mmap_mode="r")
    W = args.tokens // T
    stream = torch.from_numpy(np.asarray(val[:W * T + 1]).astype(np.int64))
    inputs, targets = stream[:-1].view(W, T), stream[1:].view(W, T)
    flat = inputs.flatten().numpy()

    def text(t, before=18, after=3):
        lo = max(t - before, (t // T) * T)
        s = tok.decode(flat[lo:t].tolist()) + "[[" + tok.decode([int(flat[t])]) + "]]"
        s += " → " + tok.decode(stream[t + 1:t + 1 + after].tolist())
        return s.replace("\n", " ")

    say(f"Reading L{L}/N{N} of {args.run} (step {step}), {cfg['num_layers']} layers x d_ff {cfg['d_ff']}")
    say(f"Held-out corpus: first {len(flat):,} TinyStories validation tokens\n")
    a, b, h = r.neuron(inputs, L, N)
    peak = float(h.max())
    report["corpus_max"] = peak

    # 1. Contexts ------------------------------------------------------------------------------------
    if "contexts" in sections:
        say("== 1. Top contexts ==")
        say(f"max {peak:.2f}; fires above 10% of max on {np.mean(h > .1 * peak):.3%} of tokens, "
            f"above 50% on {np.mean(h > .5 * peak):.4%}; min {h.min():.2f}")
        groups = {
            "positive, gate open (the neuron's job)": np.argsort(-np.where(a > 0, h, -np.inf))[:15],
            "positive, double-negative echo (gate<0, value<0)": np.argsort(-np.where((a < 0) & (b < 0), h, -np.inf))[:5],
            "negative pole": np.argsort(h)[:10],
        }
        report["contexts"] = {}
        for name, idx in groups.items():
            say(f"-- {name}")
            report["contexts"][name] = []
            for t in idx:
                say(f"  {h[t]:+7.2f} ({h[t] / peak:+.0%})  {text(t)}")
                report["contexts"][name].append([round(float(h[t]), 3), text(t)])
        strong = np.where(h > .3 * peak)[0]
        fire_tok = {}
        next_tok = {}
        for t in strong:
            fire_tok[r.piece(flat[t])] = fire_tok.get(r.piece(flat[t]), 0) + 1
            nxt = r.piece(stream[t + 1])
            next_tok[nxt] = next_tok.get(nxt, 0) + 1
        top = lambda d: sorted(d.items(), key=lambda kv: -kv[1])[:15]
        say(f"-- {len(strong)} tokens above 30% of max. Firing token: "
            + ", ".join(f"{k!r} {v}" for k, v in top(fire_tok)))
        say("   Next token: " + ", ".join(f"{k!r} {v}" for k, v in top(next_tok)))
        report["firing_tokens"], report["next_tokens"] = top(fire_tok), top(next_tok)
        # The essay's advice: strong-but-not-extreme activations are where a label is easiest to fool.
        report["mid_band"] = {}
        for word, _ in top(fire_tok)[:6]:
            hits = [t for t in strong if r.piece(flat[t]) == word]
            say(f"-- mid band, firing on {word!r} ({len(hits)} tokens), random examples:")
            report["mid_band"][word] = []
            for t in rng.choice(hits, min(4, len(hits)), replace=False):
                say(f"  {h[t]:+7.2f} ({h[t] / peak:+.0%})  {text(t)}")
                report["mid_band"][word].append([round(float(h[t]), 3), text(t)])
        say()

    def label_test(reader, layer, n, peak):
        out = {}
        for name, sentences in (("should fire", SHOULD_FIRE), ("near misses, should not fire", SHOULD_NOT)):
            rows = reader.sentence_test(layer, n, peak, sentences)
            say(f"-- {name} (strongest token, % of corpus max)")
            for row in rows:
                say(f"  {row['share']:+5.0%}  on {row['token']!r:10s} {row['text']}")
            out[name] = {"rows": rows, "strong_rate": float(np.mean([row["share"] >= .3 for row in rows]))}
        say(f"   strong (>= 30% of max): {out['should fire']['strong_rate']:.0%} of should-fire sentences, "
            f"{out['near misses, should not fire']['strong_rate']:.0%} of near misses")
        return out

    def prompt_ablation(reader, layer, n, d_ff, prompts, n_random=20):
        """P(next token) with the neuron zeroed, and the change in log-prob vs zeroing random neurons."""
        base = reader.prompt_probs(prompts)
        ablated = reader.prompt_probs(prompts, set_neurons(layer, [n]))
        others = rng.choice([m for m in range(d_ff) if m != n], n_random, replace=False)
        rand = np.stack([reader.prompt_probs(prompts, set_neurons(layer, [int(m)])) for m in others])
        d, d_rand = np.log(ablated / base), np.log(rand / base)
        say(f"  {'prompt (end)':44s} {'next':>10s} {'base':>6s} {'zeroed':>7s} {'Δ log p':>8s}   "
            f"{n_random} random neurons: mean |Δ| / max |Δ|")
        rows = []
        for k, (prompt, target) in enumerate(prompts):
            say(f"  {prompt[-44:]:44s} {target!r:>10s} {base[k]:6.1%} {ablated[k]:7.1%} {d[k]:+8.3f}   "
                f"{np.abs(d_rand[:, k]).mean():.3f} / {np.abs(d_rand[:, k]).max():.3f}")
            rows.append({"prompt": prompt, "target": target, "base": float(base[k]), "ablated": float(ablated[k]),
                         "dlogp": float(d[k]), "random_mean_abs_dlogp": float(np.abs(d_rand[:, k]).mean()),
                         "random_max_abs_dlogp": float(np.abs(d_rand[:, k]).max())})
        return rows, others

    # 2. Label ---------------------------------------------------------------------------------------
    if "label" in sections:
        say("== 2. Label, then sentences written from the label alone ==")
        say(f"Label: {LABEL}")
        report["label"] = LABEL
        report["label_test"] = label_test(r, L, N, peak)
        say()

    # 3. Ablation ------------------------------------------------------------------------------------
    if "ablation" in sections:
        say("== 3. Ablation: zero the neuron everywhere ==")
        say("-- counting prompts (does the model need it to count?)")
        rows, others = prompt_ablation(r, L, N, cfg["d_ff"], COUNT_PROMPTS)
        say(f"-- output-side prompts, for the revised label: {REVISED_LABEL}")
        out_rows, _ = prompt_ablation(r, L, N, cfg["d_ff"], OUTPUT_PROMPTS)
        sub_in, sub_tg = inputs[:400], targets[:400]
        base_loss = r.token_losses(sub_in, sub_tg)
        deltas = {}
        for m in [N] + [int(m) for m in others[:10]]:
            hook = set_neurons(L, [m])
            HOOKS.append(hook)
            loss = r.token_losses(sub_in, sub_tg)
            HOOKS.remove(hook)
            deltas[m] = loss - base_loss
        rand_mean = [float(deltas[m].mean()) for m in deltas if m != N]
        say(f"val loss on {sub_in.numel():,} tokens: baseline {base_loss.mean():.4f}; zeroing N{N} {deltas[N].mean():+.5f}; "
            f"zeroing one of 10 random L{L} neurons {np.mean(rand_mean):+.5f} on average (largest {max(rand_mean):+.5f})")
        say("tokens whose loss rises most when it is zeroed (the prediction of the token after the arrow):")
        worst = np.argsort(-deltas[N])[:8]
        for t in worst:
            say(f"  {deltas[N][t]:+6.2f} nats  {text(int(t), after=1)}")
        report["revised_label"] = REVISED_LABEL
        report["ablation"] = {"prompts": rows, "output_prompts": out_rows, "val_loss_base": float(base_loss.mean()), "val_loss_delta": float(deltas[N].mean()),
                              "random_neuron_deltas": rand_mean,
                              "most_hurt": [[float(deltas[N][t]), text(int(t), after=1)] for t in worst]}
        say()

    # 4. Steering ------------------------------------------------------------------------------------
    if "steering" in sections:
        say("== 4. Steering: clamp the neuron to a multiple of its corpus max at every position, greedy decode ==")
        report["steering"] = []
        for prompt in NEUTRAL_PROMPTS:
            say(f"-- {prompt}")
            for scale in (None, 1., 2., 3.):
                hook = None if scale is None else set_neurons(L, [N], scale * peak)
                if hook:
                    HOOKS.append(hook)
                out = r.generate(prompt)
                if hook:
                    HOOKS.remove(hook)
                numbers = sum(w in NUMBER_WORDS for w in re.findall(r"[a-z]+", out.lower()))
                name = "baseline" if scale is None else f"clamp {scale:g}x max"
                say(f"  {name:14s} [{numbers} number words] {out!r}")
                report["steering"].append({"prompt": prompt, "setting": name, "number_words": numbers, "text": out})
        say()

    # 5. Circuit -------------------------------------------------------------------------------------
    if "circuit" in sections:
        say("== 5. Circuit ==")
        blocks = model.blocks
        g_final, w_out = model.norm.weight.detach(), model.output_layer.weight.detach()
        logit_dir = (w_out @ (g_final * blocks[L].ff.w2.weight[:, N].detach())).cpu().numpy()
        up, down = np.argsort(-logit_dir)[:15], np.argsort(logit_dir)[:15]
        say("Output: tokens its w2 column pushes up / down through the final norm and unembedding")
        say("  up:   " + ", ".join(f"{r.piece(i)!r} {logit_dir[i]:+.2f}" for i in up))
        say("  down: " + ", ".join(f"{r.piece(i)!r} {logit_dir[i]:+.2f}" for i in down))
        report["circuit"] = {"pushes_up": [[r.piece(i), float(logit_dir[i])] for i in up],
                             "pushes_down": [[r.piece(i), float(logit_dir[i])] for i in down]}

        # Upstream: direct path from each layer-(L-1) MLP neuron through block L's rms2 into this neuron's gate
        # and value, attributed to its output h by first-order expansion, averaged over its top gate-open firings.
        tops = np.argsort(-np.where(a > 0, h, -np.inf))[:40]
        rows_idx = np.unique(tops // T)
        cap = {}
        grab = lambda i, a_, b_, h_: cap.__setitem__("h", h_.float().cpu()) if i == L - 1 else None
        handle = blocks[L].rms2.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0].float().cpu()))
        HOOKS.append(grab)
        r.logits(inputs[rows_idx])
        HOOKS.remove(grab)
        handle.remove()
        g2 = blocks[L].rms2.weight.detach().float().cpu()
        w2_up = blocks[L - 1].ff.w2.weight.detach().float().cpu()  # (d_model, d_ff)
        to_gate = w2_up.T @ (blocks[L].ff.w1.weight[N].detach().float().cpu() * g2)
        to_value = w2_up.T @ (blocks[L].ff.w3.weight[N].detach().float().cpu() * g2)
        attr, gate_share = [], []
        for t in tops:
            row, col = int(np.searchsorted(rows_idx, t // T)), int(t % T)
            hu, x = cap["h"][row, col], cap["x"][row, col]
            rms = torch.sqrt((x * x).mean() + 1e-5)
            ca, cb = hu * to_gate / rms, hu * to_value / rms
            sig = 1 / (1 + np.exp(-a[t]))
            attr.append((sig * (1 + a[t] * (1 - sig)) * b[t] * ca + a[t] * sig * cb).numpy())
            gate_share.append(float(ca.sum() / a[t]))
        attr = np.mean(attr, 0)
        # Label upstream neurons by their own top contexts on the first 300k tokens.
        best = {}

        def stream_top(i, a_, b_, h_):
            if i == L - 1:
                y = h_.flatten(0, -2).float()
                pos = torch.arange(y.shape[0], device=y.device)[:, None].expand_as(y) + best["offset"]
                if "v" in best:
                    y, pos = torch.cat([best["v"], y]), torch.cat([best["p"], pos])
                v, k = y.topk(3, dim=0)
                best["v"], best["p"] = v, pos.gather(0, k)
        best["offset"] = 0
        HOOKS.append(stream_top)
        for chunk in torch.split(inputs[:1172], 32):
            r.logits(chunk)
            best["offset"] += chunk.numel()
        HOOKS.remove(stream_top)
        top_pos = best["p"].cpu().numpy()
        say(f"Input: direct contributions of layer-{L - 1} MLP neurons, averaged over its top {len(tops)} gate-open firings")
        say(f"  (layer-{L - 1} MLP neurons supply {np.mean(gate_share):.0%} of the gate pre-activation on average; "
            f"the rest comes from attention, embeddings and earlier layers)")
        report["circuit"]["upstream"] = []
        for label, order in (("drives it up", np.argsort(-attr)[:8]), ("holds it down", np.argsort(attr)[:4])):
            say(f"  {label}:")
            for m in order:
                examples = [text(int(t), before=10, after=0) for t in top_pos[:2, m]]
                say(f"    L{L - 1}/N{m:<5d} {attr[m]:+6.2f}   e.g. {examples[0]!r} | {examples[1]!r}")
                report["circuit"]["upstream"].append({"neuron": f"L{L - 1}/N{m}", "effect": label,
                                                      "attribution": float(attr[m]), "examples": examples})
        say()

    # 6. Match in an independently trained model ----------------------------------------------------
    if "match" in sections:
        say(f"== 6. The same neuron in {args.match_run}? ==")
        model2, cfg2, step2 = load(REPO / "checkpoints" / args.match_run / args.match_ckpt, args.num_heads, args.context,
                                   args.device)
        r2 = Reader(model2, cfg2, tok, args.device)
        # Match by top contexts, which is scale-free: for every neuron of the second model, how many of this
        # neuron's top-100 contexts are among its own top-1000 on the same tokens. Also report the correlation
        # of the two neurons' outputs over the first 300k tokens.
        M = 1172 * T
        xt = torch.tensor((h[:M] - h[:M].mean()) / h[:M].std(), device=args.device, dtype=torch.float32)
        sums = {i: np.zeros((3, cfg2["d_ff"])) for i in range(cfg2["num_layers"])}
        best2 = {}
        state = {"offset": 0}

        def scan(i, a_, b_, h_):
            y = h_.flatten(0, -2).float()
            pos = torch.arange(y.shape[0], device=y.device)[:, None].expand_as(y) + state["offset"]
            if i in best2:
                y_all, pos = torch.cat([best2[i][0], y]), torch.cat([best2[i][1], pos])
            else:
                y_all = y
            v, k = y_all.topk(1000, dim=0)
            best2[i] = (v, pos.gather(0, k))
            if state["offset"] < M:
                y = y[:M - state["offset"]]
                xs = xt[state["offset"]:state["offset"] + y.shape[0], None]
                sums[i] += torch.stack([y.sum(0), (y * y).sum(0), (xs * y).sum(0)]).cpu().double().numpy()
            if i == cfg2["num_layers"] - 1:
                state["offset"] += y.shape[0]
        HOOKS.append(scan)
        for chunk in torch.split(inputs, 32):
            r2.logits(chunk)
        HOOKS.remove(scan)
        top100 = np.argsort(-h)[:100]
        scores = []
        for i, (sy, syy, sxy) in sums.items():
            pos = best2[i][1].cpu().numpy()  # (1000, d_ff)
            c = (sxy / M) / np.sqrt(syy / M - (sy / M) ** 2)
            in_top1000 = np.isin(pos, top100).sum(0)
            in_top100 = np.isin(pos[:100], top100).sum(0)
            scores += [(int(in_top1000[m]), int(in_top100[m]), float(c[m]), i, m) for m in range(len(c))]
        scores.sort(reverse=True)
        say(f"Every {args.match_run} neuron scored by how many of L{L}/N{N}'s top-100 contexts are in its own top-1000 "
            f"(and top-100), with the correlation of outputs over {M:,} tokens. Best:")
        for s1000, s100, c, i, m in scores[:5]:
            say(f"  L{i}/N{m}: {s1000}/100 in its top-1000, {s100}/100 in its top-100, r = {c:.3f}")
        _, _, _, layer2, n2 = scores[0]
        a2, b2, h2 = r2.neuron(inputs, layer2, n2)
        peak2 = float(h2.max())
        top_a, top_b = set(np.argsort(-h)[:100]), set(np.argsort(-h2)[:100])
        say(f"Best match L{layer2}/N{n2} (step {step2}): max {peak2:.2f}; shares {len(top_a & top_b)} of its top-100 "
            f"contexts with L{L}/N{N}. Its top gate-open contexts:")
        for t in np.argsort(-np.where(a2 > 0, h2, -np.inf))[:8]:
            say(f"  {h2[t]:+7.2f} ({h2[t] / peak2:+.0%})  {text(t)}")
        say("Same sentence test:")
        match_label = label_test(r2, layer2, n2, peak2)
        say("Same ablation, counting prompts:")
        match_rows, _ = prompt_ablation(r2, layer2, n2, cfg2["d_ff"], COUNT_PROMPTS)
        say("Same ablation, output-side prompts:")
        match_out, _ = prompt_ablation(r2, layer2, n2, cfg2["d_ff"], OUTPUT_PROMPTS)
        w_out2, g_final2 = model2.output_layer.weight.detach(), model2.norm.weight.detach()
        logit_dir2 = (w_out2 @ (g_final2 * model2.blocks[layer2].ff.w2.weight[:, n2].detach())).cpu().numpy()
        say("  pushes up:   " + ", ".join(repr(r2.piece(i)) for i in np.argsort(-logit_dir2)[:12])
            + ("" if layer2 == cfg2["num_layers"] - 1 else "  (direct path only; later layers also read it)"))
        say("  pushes down: " + ", ".join(repr(r2.piece(i)) for i in np.argsort(logit_dir2)[:12]))
        report["match"] = {"run": args.match_run, "candidates": [{"neuron": f"L{i}/N{m}", "in_top1000": s1000, "in_top100": s100,
                                                                      "r": c} for s1000, s100, c, i, m in scores[:5]],
                           "neuron": f"L{layer2}/N{n2}", "corpus_max": peak2, "top100_shared": len(top_a & top_b),
                           "label_test": match_label, "prompts": match_rows, "output_prompts": match_out,
                           "pushes_up": [r2.piece(i) for i in np.argsort(-logit_dir2)[:12]],
                           "pushes_down": [r2.piece(i) for i in np.argsort(logit_dir2)[:12]]}
        say()

    if args.output:
        Path(args.output + ".txt").write_text("\n".join(lines) + "\n")
        Path(args.output + ".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
