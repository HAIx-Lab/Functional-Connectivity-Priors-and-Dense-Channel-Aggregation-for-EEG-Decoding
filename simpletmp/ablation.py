"""
Knockout ablation: how load-bearing is each component, per config?

Tests the hypothesis that the connectivity prior is functionally used in
proportion to how meaningful it is -- i.e. zeroing the map-dependent SPATIAL
components (RegionAttention, BrainEmbed) should hurt `hcp`/`cont` more than
`random`, while a TEMPORAL knockout (control) should hurt all configs similarly.

Knockout = zero a component's residual contribution at inference via a forward
hook (all targets are residual adds, so this cleanly removes them). This measures
RELIANCE on a component the model trained with; it is not a retrain-without.

Usage (notebook, repo root):
    from ablation import run_ablation, ABLATIONS
    df = run_ablation(RUNS[0])          # RUNS imported from run_attributions
    print(df)                            # config x ablation table with deltas
"""

import os
import numpy as np
import torch

from run_attributions import (
    DATASET_META, CONFIG, build_model, _make_params, _resolve_ckpts, _arch_for, _loader,
)
from eeg_attention_rollout import _get_backbone

# name -> module class names to zero. "*" suffix marks temporal controls.
ABLATIONS = {
    "intact":            [],
    "region_attn":       ["RegionAttention"],
    "brain_embed":       ["BrainEmbedEEGLayer"],
    "spatial_all":       ["RegionAttention", "BrainEmbedEEGLayer"],
    "temporal_attn*":    ["TemporalAttention"],     # control
    "tem_embed*":        ["TemEmbedEEGLayer"],      # control
    "temporal_all*":     ["TemporalAttention", "TemEmbedEEGLayer"],  # control
}

# primary metric per task mode (used for the delta column)
PRIMARY = {"bin": "auroc", "mul": "bal_acc", "reg": "pearson"}


class Knockout:
    def __init__(self, backbone, class_names):
        self.backbone = backbone
        self.class_names = set(class_names)
        self.handles = []

    def __enter__(self):
        def zero_hook(m, i, o):
            return torch.zeros_like(o) if torch.is_tensor(o) else o
        if self.class_names:
            for mod in self.backbone.modules():
                if type(mod).__name__ in self.class_names:
                    self.handles.append(mod.register_forward_hook(zero_hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def _metric(logits, y, mode):
    from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                                 accuracy_score, cohen_kappa_score, mean_squared_error)
    y = y.numpy().reshape(-1)
    if mode == "bin":
        # handle both head styles: single logit + sigmoid, or 2-class softmax
        if logits.shape[-1] == 1:
            prob = torch.sigmoid(logits).numpy().reshape(-1)
        else:
            prob = torch.softmax(logits, dim=-1)[:, 1].numpy().reshape(-1)
        pred = (prob >= 0.5).astype(int)
        try:
            auroc = roc_auc_score(y, prob)
        except Exception:
            auroc = float("nan")
        return {"auroc": auroc, "bal_acc": balanced_accuracy_score(y, pred)}
    if mode == "mul":
        pred = logits.argmax(-1).numpy()
        return {"acc": accuracy_score(y, pred),
                "bal_acc": balanced_accuracy_score(y, pred),
                "kappa": cohen_kappa_score(y, pred)}
    pred = logits.numpy().reshape(-1)
    return {"rmse": mean_squared_error(y, pred) ** 0.5,
            "pearson": float(np.corrcoef(y, pred)[0, 1])}


def _evaluate(model, batches, device, mode):
    logits, ys = [], []
    with torch.no_grad():
        for x, y in batches:
            logits.append(model(x.to(device)).cpu())
            ys.append(y.reshape(-1) if torch.is_tensor(y) else torch.as_tensor(y).reshape(-1))
    return _metric(torch.cat(logits), torch.cat(ys), mode)


def run_ablation(run, configs=None, ablations=None, split=None, max_batches=None, save=True):
    """Returns a tidy DataFrame: one row per (config, ablation) with metrics and
    the delta of the primary metric vs the intact model."""
    import pandas as pd
    name = run["dataset"]
    meta = DATASET_META[name]
    mode = run["mode"]
    primary = PRIMARY[mode]
    device = torch.device(f"cuda:{CONFIG['cuda']}" if torch.cuda.is_available() else "cpu")

    params = _make_params(run, meta)
    split = split or CONFIG["split"]
    loader = _loader(name)(params).get_data_loader()
    dl = loader.get(split) or loader.get("test") or loader.get("train")
    # cache batches so every config/ablation is scored on identical data
    batches = []
    for i, (x, y) in enumerate(dl):
        if max_batches is not None and i >= max_batches:
            break
        batches.append((x, y))
    print(f"[{name}] eval on '{split}' split, {len(batches)} batches, mode={mode}, primary={primary}")

    ckpts = _resolve_ckpts(run)
    ablations = ablations or ABLATIONS
    rows = []
    intact_by_cfg = {}
    for cfg in (configs or list(ckpts)):
        if not os.path.exists(ckpts[cfg]):
            print(f"  [{cfg}] checkpoint missing, skipping")
            continue
        model = build_model(meta, ckpts[cfg], device, arch=_arch_for(cfg)).eval()
        bb = _get_backbone(model)
        base = None
        for abl, classes in ablations.items():
            with Knockout(bb, classes):
                m = _evaluate(model, batches, device, mode)
            if abl == "intact":
                base = m[primary]
                intact_by_cfg[cfg] = base
            row = {"config": cfg, "ablation": abl, **{k: round(v, 4) for k, v in m.items()}}
            d = None if base is None else m[primary] - base
            row[f"d_{primary}"] = None if d is None else round(d, 4)
            # relative drop: fraction of the intact metric lost (baseline-normalized)
            row[f"rel_{primary}"] = None if (d is None or base in (0, None)) else round(d / base, 4)
            rows.append(row)
            print(f"  [{cfg:7s}] {abl:14s} {primary}={m[primary]:.4f}"
                  + (f"  Δ={d:+.4f}  ({100*d/base:+.1f}%)" if d is not None else "  (baseline)"))
    # flag baseline mismatch -- raw Δ comparisons across configs are only fair if
    # the intact metrics are close; otherwise use the rel_ column.
    if len(intact_by_cfg) > 1:
        lo, hi = min(intact_by_cfg.values()), max(intact_by_cfg.values())
        if hi - lo > 0.03:
            print(f"  [note] intact {primary} differs across configs by {hi-lo:.3f} "
                  f"({intact_by_cfg}); compare the rel_{primary} column, not raw Δ.")
    df = pd.DataFrame(rows)
    if save:
        out = os.path.join(os.path.expanduser(CONFIG["out_dir"]), f"ablation_{name}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print(f"saved -> {out}")
    return df


def compare_reliance(df, ref="random", use_relative=True):
    """Pivot the ablation table per (ablation x config) and add each config's GAP
    vs a reference. Uses the baseline-normalized relative drop by default (raw Δ is
    confounded when intact metrics differ across configs). The test of spatial
    specificity: compare the gap on spatial_all vs the temporal_all* control."""
    import pandas as pd
    col = f"rel_{[c for c in df.columns if c.startswith('d_')][0][2:]}" if use_relative \
        else [c for c in df.columns if c.startswith("d_")][0]
    if col not in df.columns:                       # fallback if rel_ not present
        col = [c for c in df.columns if c.startswith("d_")][0]
    piv = df[df["ablation"] != "intact"].pivot(index="ablation", columns="config", values=col)
    for c in piv.columns:
        if c != ref and ref in piv.columns:
            piv[f"{c}_minus_{ref}"] = (piv[c] - piv[ref]).round(4)
    piv.attrs["metric_column"] = col
    return piv


def plot_ablation(df, dataset="", save_path=None):
    """Grouped bars of Δprimary per ablation, colored by config. Spatial knockouts
    vs the temporal-control knockouts (marked with *) side by side."""
    import matplotlib.pyplot as plt
    dcol = [c for c in df.columns if c.startswith("d_")][0]
    abls = [a for a in df["ablation"].unique() if a != "intact"]
    cfgs = list(df["config"].unique())
    colors = {"random": "#888888", "hcp": "#1f77b4", "cont": "#d62728"}
    x = np.arange(len(abls)); w = 0.8 / max(len(cfgs), 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, cfg in enumerate(cfgs):
        sub = df[df["config"] == cfg].set_index("ablation")
        vals = [sub.loc[a, dcol] if a in sub.index else 0 for a in abls]
        ax.bar(x + i * w, vals, w, label=cfg, color=colors.get(cfg))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w * (len(cfgs) - 1) / 2)
    ax.set_xticklabels(abls, rotation=20, ha="right")
    ax.set_ylabel(f"Δ {dcol[2:]} vs intact  (negative = knockout hurts)")
    ax.set_title(f"{dataset}: component reliance by config  (* = temporal control)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(os.path.expanduser(save_path), dpi=130, bbox_inches="tight")
        print("saved", save_path)
    return fig
