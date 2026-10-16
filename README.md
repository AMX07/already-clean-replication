# Are Neural Networks Already Clean? A small replication

Three experiments that test claims from Goodfire's essay
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com) (Cammarata, 2026) on small
language models I trained myself.

The essay argues that neurons which look "polysemantic" (doing several unrelated jobs) mostly aren't
evidence of **superposition** (unrelated concepts packed into the same neurons). Instead, it says they come
from weak tools, weak architectures and **recursive generalization**, where a neuron grows outward over
training into *related* territory. If that's right, a model's own neurons are readable building blocks,
and you don't have to untangle them with a sparse autoencoder first.

## Verdict

| Essay claim | Essay evidence | What we found | Held up? |
|---|---|---|---|
| Weak activations barely matter; strong ones carry the computation | Zero weakest 75%: acc 72% → 71%. Zero strongest 7%: 72% → 4% (ConvNeXt, ImageNet) | Zero weakest 75%: loss +0.037, acc 63.5% → 62.6%. Zero the strongest 3.2%, which removes the **same total activation**: loss +2.03, acc → 30.3% | **Yes** |
| Predictions that flip under weak ablation are near-ties | Shown qualitatively | Flipped tokens had a median top-2 probability share of 0.56 (0.5 = coin flip), vs 0.82 across all tokens | **Yes** |
| SwiGLU's double-negative "echo" mislabels ~4% of neurons | Qwen3-32B | Real but rare: dominates the top-20 list of 0.0–0.7% of neurons. Removing it slightly *hurt* (+0.068 loss) | **Partly** |
| New contexts a neuron picks up during training are related to old ones (generalization, not packing) | Neurons watched growing over training | New top contexts are 2–5× more similar to the neuron's own earlier contexts than to another neuron's. No neuron looked like an unrelated mix | **Yes, mostly** |
| Neurons *branch outward* (recursive generalization) | "iy" → "diy" → "do it yourself" | From step 5k on, most changing neurons **narrowed and sharpened** instead. Steps before 5k weren't covered | **Not observed** |

**Bottom line:** the core claim, that neurons are readable building blocks rather than tangled mixtures, fits
these models. They are small TinyStories-scale models, so treat this as supporting evidence, not proof.

## Models

| Name | Architecture | Data | Training | Val loss | Used for |
|---|---|---|---|---|---|
| `comic-cherry-166` | 4 layers, d_model 512, 16 heads, SwiGLU d_ff 1344 | TinyStories (10k BPE) | 2,000 steps × 640 × 256 tokens | 1.37 | Experiments 1 and 2 |
| `bumbling-brook-124` | same | TinyStories | 50,000 steps × 32 × 256 tokens, checkpoints every 5k | 1.38 | Experiment 3 |
| nanochat `final-hybrid` | 6 layers, d_model 384, ReLU² MLP | ClimbMix | 5,000 steps, 20M tokens | 4.60 | Experiment 1 (ReLU² comparison) |

The two TinyStories models come from my Stanford CS336 assignment 1 implementation. The nanochat model
comes from my MLX port of [nanochat](https://github.com/karpathy/nanochat).

---

## Experiment 1: weak vs strong activation ablation

**Question.** If meaning were packed into weak activations (superposition), removing them should hurt. Does it?

**Method** (`swiglu_ablation.py`, `nanochat/neuron_ablation.py`):

1. Run 128 non-overlapping 256-token calibration windows and record every MLP neuron's activation
   (|silu(gate) · value| for SwiGLU; nonzero ReLU² outputs for nanochat).
2. Fit **per-neuron** percentile thresholds on that calibration set.
3. On a disjoint set of eval windows, zero each neuron's weakest *q*% or strongest *f*% of activations.
   Then measure loss, next-token accuracy, the share of next-token predictions that change, and how close
   the flipped predictions were to a tie.
4. Controls:
   - a strong ablation whose cutoff is chosen to remove the **same total activation magnitude** as the
     weak-75% ablation (*mass-matched*);
   - a random ablation that drops 75% of activations (*count-matched*).

**Results: cs336 SwiGLU model** (`comic-cherry-166`, 32k eval tokens; full table in
[`results_comic-cherry-166.txt`](results_comic-cherry-166.txt))

| Ablation | Activation mass removed | Δ loss | Top-1 acc | Predictions changed | Top-2 share of flipped tokens |
|---|---|---|---|---|---|
| none | – | 1.368 | 63.5% | – | – |
| weakest 50% | 10.8% | +0.006 | 63.3% | 3.3% | 0.52 |
| **weakest 75%** | **29.8%** | **+0.037** | 62.6% | 8.7% | **0.56** |
| weakest 90% | 51.2% | +0.127 | 60.8% | 16.0% | 0.59 |
| strongest 1% | 15.7% | +1.005 | 44.3% | 44.5% | 0.72 |
| **strongest 3.2% (mass-matched)** | **29.7%** | **+2.030** | 30.3% | 63.4% | 0.77 |
| strongest 7% | 42.0% | +2.889 | 21.5% | 74.5% | 0.79 |
| random 75% (count-matched) | – | +2.389 | 26.8% | 68.3% | 0.77 |

Median top-2 share across all eval tokens is 0.82. Ablating one layer at a time, layer 0's strongest 7%
matters most (+1.41 loss), and no single layer's weakest 75% costs more than +0.02.

**Results: nanochat ReLU² model** ([`nanochat/results_final-hybrid.json`](nanochat/results_final-hybrid.json)).
91.5% of ReLU² activations are exactly zero, so the percentiles are taken over the nonzero ones.

| Ablation | Δ loss | Predictions changed |
|---|---|---|
| weakest 75% of nonzero | +0.080 | 13.0% |
| strongest 2.2% (mass-matched) | +0.394 | 34.3% |
| random 75% (count-matched) | +1.026 | 51.6% |

Same pattern, milder. This model is much weaker (loss 4.6), so it has less to lose.

## Experiment 2: SwiGLU sign split

**Question.** A SwiGLU neuron computes `silu(gate) · value`. When gate < 0 and value < 0, the output is a
small *positive* "double-negative echo". The essay says that for ~4% of Qwen3-32B neurons, this echo fills
the top of the positive list, so a top-k dashboard labels the neuron with the wrong job.

**Method.** For each neuron, take its top-20 positive activations on the calibration set and count how many
had the gate closed (< 0). Call a neuron *echo-dominated* if at least half did. Separately, zero the whole
double-negative regime across the network and measure loss.

**Results.**

- About 70% of all positive SwiGLU outputs are double-negative echoes, but they are tiny.
- Only 0.0% (layer 0) to 0.7% (layers 1–3) of neurons are echo-dominated. That is well below the essay's ~4%.
- Zeroing the double-negative regime: loss **+0.068**. Zeroing everything with the gate closed: +0.194.
  The essay says the double-negative ablation *improves* most benchmarks; we see a small cost instead
  (different metric, much smaller model).
- When a neuron is echo-dominated, the dashboard label really is wrong:

| Neuron | What a top-k dashboard shows (echo) | What the neuron actually does (gate open) |
|---|---|---|
| `L1/N1263` | +0.70 on "You **might**", +0.68 on "under a big **leaf**" | −13.7 on "a small **hole**" |
| `L2/N929` | +0.87 on "Once upon a time, in a cheerful forest**,**" | −9.0 on "ran as fast **as**" |
| `L3/N963` | +1.74 on "His mom said yes and **bought**" | gate never produces a positive output in the 32k calibration tokens; everything shown is echo |

## Experiment 3: watching neurons change over training

**Question.** When a neuron changes what it responds to during training, are the new contexts *related* to
the old ones (generalization) or *unrelated* (arbitrary packing)?

**Method** (`swiglu_growth.py`):

1. Run all 10 `bumbling-brook-124` checkpoints (steps 5k–50k) on the same 131k held-out tokens. Record each
   neuron's top-30 contexts by positive output. Neuron identity is stable across checkpoints of a single run.
2. An **entrant** is a context that newly enters a neuron's top 30 between consecutive checkpoints.
3. Measure each entrant's similarity to the neuron's *previous* top 30, using a model-free TF-IDF bag of the
   24 tokens ending at the firing token. Pairs from the same story window are excluded.
4. Nulls: the same entrants scored against a **different** neuron's previous top 30, and random contexts
   scored against this neuron's previous top 30.
5. 256 neurons sampled per layer. The 3 most-changed neurons per layer are printed for inspection.

**Results** ([`growth_bumbling-brook-124.txt`](growth_bumbling-brook-124.txt)):

| Layer | Top-30 overlap between consecutive checkpoints | Overlap, 5k vs 50k | Entrant vs own previous | Entrant vs other neuron | Random vs own | Entrants below other-neuron median |
|---|---|---|---|---|---|---|
| 0 | 0.64 | 0.23 | **0.38** | 0.07 | 0.10 | 24% |
| 1 | 0.51 | 0.10 | **0.27** | 0.08 | 0.10 | 23% |
| 2 | 0.43 | 0.07 | **0.22** | 0.11 | 0.12 | 20% |
| 3 | 0.40 | 0.05 | **0.20** | 0.12 | 0.13 | 23% |

If new contexts were unrelated, about 50% would fall below the other-neuron median; we see 20–24%.
Neurons also keep changing late in training, more so in deeper layers.

The most-changed neurons mostly **sharpened**:

- `L2/N1013`: "she saw [that]" → "saw [**what**]"
- `L3/N590`: mixed nouns → counting ("[**One**], [**two**]", "one by [**one**]")
- `L2/N948`: story endings → morals ("learned that [**sharing**]", "always [**tell**]")
- `L1/N952`: "[looked]" → the period that closes a pleasant clause

The exceptions are layer-0 neurons that switched token outright: `L0/N1338` went from "[there]" to
"[blanket]", and `L0/N793` from `said, "[Hi]` to a mix including "[purr]" and "[Mr]". These are
reassignments, not mixtures.

---

## Review

### What was implemented correctly

- **Held-out thresholds.** Per-neuron cutoffs are fit on calibration windows and evaluated on disjoint,
  non-overlapping windows.
- **The key control is in place.** The mass-matched strong ablation removes the same total activation as
  the weak ablation, which is the comparison the essay's argument rests on. A count-matched random
  ablation is included as well.
- **Near-tie test.** Flipped predictions are scored by the top-2 probability share from the *unablated*
  model, which tests the essay's "flips are near-ties" claim directly.
- **Model configs verified.** The head count isn't recoverable from the weights, so it was taken from the
  wandb config and checked against val loss: `comic-cherry-166` scores 1.368 here vs 1.369 logged, and
  `bumbling-brook-124` scores 1.384 on 16k tokens vs 1.467 logged on a different subset. A wrong head count
  would give far worse loss.
- **Exact sign split.** The regimes are read from the real gate and value pre-activations, so the
  double-negative case is identified exactly.
- **Valid growth tracking.** Neuron *n* is the same unit across checkpoints of one run (no permutation
  issue). Relatedness is measured without the model itself, against two nulls, with same-story pairs
  excluded so that sharing a story can't make contexts look related.
- **Assignment code untouched.** The scripts only import the CS336 model class and load checkpoints;
  nothing in the assignment repo was modified.

Two bugs were found and fixed during review, and Experiment 1 was rerun (numbers moved by < 0.01 loss):

- calibration and eval windows could overlap;
- example texts in the sign-split output could cross a window boundary into a different story.

### What can be improved

1. **Scale and statistics.** One seed, 33k–131k tokens, no confidence intervals. Add bootstrap CIs, more
   seeds and more tokens.
2. **Metric mismatch with the essay.** The essay measures classification answers. We measure next-token
   argmax at every position, which includes many low-stakes tokens. Evaluating on downstream completions
   would compare more directly.
3. **A random control that matches activation mass.** The random-75% control also removes strong
   activations. A random ablation that removes the same ~30% of activation mass would isolate "which
   activations" from "how much".
4. **A stronger ReLU² model.** The nanochat model is undertrained (loss 4.6, 20M tokens).
5. **Sign-split threshold.** The "≥ 50% of the top 20" cutoff is arbitrary. Report the full distribution.
   Also, the "gate open" example list can show negative outputs when a neuron has no positive gate-open
   activations (as with `L3/N963`).
6. **No early checkpoints.** Branching should happen early in training, and the first checkpoint is
   step 5k. A retrain that saves every 100–250 steps from step 0 is the obvious next run.
7. **Lexical relatedness only.** TF-IDF with the firing token weighted 3× mostly measures "same token or
   same words". It misses grammatical and semantic relations, and near-duplicate TinyStories stories in
   different windows can inflate it. Use a semantic embedding or an LLM judge, report the same-token rate
   separately, and dedupe near-duplicates.
8. **Boundary noise in top-k overlap.** Entrants near rank 30 are often just noise at the cutoff.
   Activation-weighted overlap would be more robust.
9. **Negative poles ignored.** The essay stresses that a neuron's negative side carries meaning; the growth
   analysis tracks only the positive side (and ReLU² has no negative side).
10. **No neuron labels.** We never labeled neurons, so the essay's "~90% have one main job" claim is
    untested. Next steps: automated labels from an LLM plus the essay's blind-maximum test.
11. **Narrow domain.** TinyStories has a tiny vocabulary of situations. Repeat on a larger OpenWebText model.

## Reproduce

The CS336 assignment code and checkpoints are **not** included: course policy for the code, size for the
checkpoints. The scripts expect a sibling checkout:

```
cs336/
├── assignment1-basics/          # provides cs336_basics.gpt.transformer_lm, checkpoints/, out/*.npy, out/*_vocab.pkl
└── neuron-ablation/             # this repo
```

```bash
../assignment1-basics/.venv/bin/python swiglu_ablation.py --run comic-cherry-166 --output results_comic-cherry-166.json
```

```bash
../assignment1-basics/.venv/bin/python swiglu_growth.py --run bumbling-brook-124 --output growth_bumbling-brook-124.json
```

`nanochat/neuron_ablation.py` belongs in `scripts/` of a nanochat checkout with the MLX backend
(`nanochat.mlx_backend`):

```bash
python -m scripts.neuron_ablation --checkpoint <mlx checkpoint dir> --data <prepared data dir with val.npy>
```

Each experiment runs in under a minute on an Apple-silicon Mac (MPS / MLX).

## Files

| File | What |
|---|---|
| `swiglu_ablation.py` | Experiments 1 and 2 on a CS336 SwiGLU checkpoint |
| `swiglu_growth.py` | Experiment 3 across checkpoints |
| `nanochat/neuron_ablation.py` | Experiment 1 on a nanochat ReLU² model (MLX) |
| `results_comic-cherry-166.{txt,json}` | Experiments 1 and 2 output, including sign-split example texts |
| `growth_bumbling-brook-124.{txt,json}` | Experiment 3 output, including most-changed neuron trajectories |
| `nanochat/results_final-hybrid.json` | nanochat ablation output |

## Reference

Cammarata, N., 2026. *Neural Networks Are Already Clean.* Goodfire. https://already-clean.goodfire.com
