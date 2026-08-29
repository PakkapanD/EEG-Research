"""Renders spec 2 section 10's four figures from the data files build_figure_data.py
and build_tradeoff_curve.py produce. Kept separate from data prep so a
styling tweak never requires re-scoring (some of that data cost real
compute - see build_tradeoff_curve.py's docstring).

Every figure saves an .svg next to the .csv/.png it was built from, so the
raw numbers behind a plotted line are always one directory away, per spec 2
section 11 ("output/07_figures/ - graphs above, both .svg and the raw data
behind them").
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from audit import OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "07_figures"

# Shared style: family = channel-selection strategy (the project's actual
# research question - CLAUDE.md section 1), criterion = clinical use case.
FAMILY_COLORS = {"Full-18": "#333333", "Glass-n": "#1f77b4", "Best-n": "#d95f02"}
CRITERION_STYLE = {
    "medication_titration": {"linestyle": "-", "marker": "o"},
    "realtime_alert": {"linestyle": "--", "marker": "s"},
}
CRITERION_LABEL = {
    "medication_titration": "Medication titration (Sens>=60%, FA<=5/day)",
    "realtime_alert": "Realtime alert (Sens>=80%, FA<=2/day)",
}
HARD_CASE_COLOR = "#d62728"

# Zanetti 2025's two reported operating points (CLAUDE.md section 13): the
# project's tier-1 benchmark. Ali 2024's equivalent operating point is not
# recorded anywhere in this repo (checked CLAUDE.md + docs/06-spec2...) -
# omitted rather than guessed; add if the project owner supplies it.
ZANETTI_2025_POINTS = [
    {"label": "Zanetti 2025, 18ch", "sens": 0.80, "fa": 1.81},
    {"label": "Zanetti 2025, 2ch", "sens": 0.64, "fa": 2.35},
]


def _family_of(config):
    if config == "Full-18":
        return "Full-18"
    return "Glass-n" if config.startswith("Glass") else "Best-n"


def fig1_channel_ladder():
    df = pd.read_csv(FIG_DIR / "data_channel_ladder.csv")

    fig, (ax_main, ax_auc) = plt.subplots(
        2, 1, figsize=(9, 8.5), sharex=True, height_ratios=[2.6, 1],
        gridspec_kw={"hspace": 0.08},
    )

    for criterion in ["medication_titration", "realtime_alert"]:
        cdf = df[df.criterion == criterion]
        style = CRITERION_STYLE[criterion]
        for family in ["Glass-n", "Best-n"]:
            fam = cdf[cdf.family.isin([family, "Full-18"])].sort_values("n_channels")
            ax_main.plot(
                fam.electrodes_mean, fam.pct_met_5fold,
                color=FAMILY_COLORS[family], linestyle=style["linestyle"],
                marker=style["marker"], markersize=7, linewidth=1.8,
                markerfacecolor=FAMILY_COLORS[family], markeredgecolor="white",
                markeredgewidth=0.8, alpha=0.9, zorder=3,
            )
        # Best-n electrode-count spread (data-driven, varies by fold)
        best = cdf[cdf.family == "Best-n"]
        ax_main.hlines(best.pct_met_5fold, best.electrodes_min, best.electrodes_max,
                        color=FAMILY_COLORS["Best-n"], alpha=0.35, linewidth=4, zorder=1)

        loo = cdf.dropna(subset=["pct_met_loo"])
        ax_main.scatter(
            loo.electrodes_mean, loo.pct_met_loo, marker="D", s=70,
            facecolor="none", edgecolor=[FAMILY_COLORS[_family_of(c)] for c in loo.config],
            linewidth=1.8, zorder=4,
        )

        anchor = cdf[cdf.config == "Full-18"]
        ax_main.scatter(anchor.electrodes_mean, anchor.pct_met_5fold, marker=style["marker"],
                         s=75, color=FAMILY_COLORS["Full-18"], edgecolor="white",
                         linewidth=0.8, zorder=5)

    ax_main.set_ylabel("% subjects meeting criterion\n(event-level, SzCORE)")
    ax_main.set_ylim(-3, 103)
    ax_main.grid(alpha=0.25)
    ax_main.set_title(
        "Channel-ladder: pass rate vs. actual electrode count\n"
        "solid=medication titration, dashed=realtime alert  |  "
        "filled circle/square=5-fold per-subject, open diamond=LOO"
    )

    for criterion in ["medication_titration", "realtime_alert"]:
        adf = df[df.criterion == criterion].drop_duplicates("config")
        for family in ["Glass-n", "Best-n"]:
            fam = adf[adf.family.isin([family, "Full-18"])].sort_values("n_channels")
            ax_auc.plot(fam.electrodes_mean, fam.auc_roc, color=FAMILY_COLORS[family],
                        linestyle="-", marker="o", markersize=4, linewidth=1.3, alpha=0.5)
        anchor = adf[adf.config == "Full-18"]
        ax_auc.scatter(anchor.electrodes_mean, anchor.auc_roc, color=FAMILY_COLORS["Full-18"],
                        s=30, zorder=5)
        break  # AUC-ROC doesn't depend on criterion - plot once, not twice
    ax_auc.set_ylabel("AUC-ROC\n(window-level,\npooled 5-fold test)")
    ax_auc.set_xlabel("Mean actual electrode count (bar = Best-n range across 5 folds)")
    ax_auc.grid(alpha=0.25)
    ax_auc.set_ylim(0.65, 0.85)
    ax_auc.annotate("Glass-7: Tier B (825k test windows,\ndifferent file/hour base than\nthe other 6 configs - PR-AUC\nespecially not directly comparable)",
                     xy=(8, df[(df.config == "Glass-7")].auc_roc.iloc[0]),
                     xytext=(9.5, 0.695), fontsize=7.5, color="#555",
                     arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

    legend_elems = [
        Line2D([0], [0], color=FAMILY_COLORS["Glass-n"], lw=2, label="Glass-n (anatomically constrained)"),
        Line2D([0], [0], color=FAMILY_COLORS["Best-n"], lw=2, label="Best-n (data-driven)"),
        Line2D([0], [0], marker="o", color="grey", lw=0, label="Full-18 (shared anchor)", markersize=7),
        Line2D([0], [0], linestyle="-", color="black", label=CRITERION_LABEL["medication_titration"], lw=1.5),
        Line2D([0], [0], linestyle="--", color="black", label=CRITERION_LABEL["realtime_alert"], lw=1.5),
        Line2D([0], [0], marker="D", color="black", lw=0, markerfacecolor="none", label="LOO (Full-18/Glass-7/Glass-2 only)", markersize=8),
    ]
    ax_main.legend(handles=legend_elems, loc="upper left", fontsize=8, framealpha=0.9)

    fig.savefig(FIG_DIR / "fig1_channel_ladder.svg", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig1_channel_ladder.svg")


def fig2_tradeoff():
    path = FIG_DIR / "data_tradeoff_curve.csv"
    if not path.exists():
        print("SKIP fig2_tradeoff: data_tradeoff_curve.csv not built yet")
        return
    df = pd.read_csv(path)
    ladder = pd.read_csv(FIG_DIR / "data_channel_ladder.csv")
    n_channels_by_config = ladder.drop_duplicates("config").set_index("config").n_channels

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.axhspan(0.60, 1.02, xmin=0, xmax=1, color="#2ca02c", alpha=0.05, zorder=0)
    ax.axvspan(1e-3, 5, ymin=0, ymax=1, color="#2ca02c", alpha=0.0)
    ax.fill_betweenx([0.60, 1.02], 1e-3, 5, color="#ffcc00", alpha=0.10,
                      label="Medication titration zone (Sens>=60%, FA<=5/day)", zorder=0)
    ax.fill_betweenx([0.80, 1.02], 1e-3, 2, color="#2ca02c", alpha=0.15,
                      label="Realtime alert zone (Sens>=80%, FA<=2/day)", zorder=0)
    ax.axhline(0.60, color="#b8960a", linestyle=":", linewidth=1)
    ax.axvline(5, color="#b8960a", linestyle=":", linewidth=1)
    ax.axhline(0.80, color="#1a7a1a", linestyle=":", linewidth=1)
    ax.axvline(2, color="#1a7a1a", linestyle=":", linewidth=1)

    cmap_glass = plt.get_cmap("Blues")
    cmap_best = plt.get_cmap("Oranges")
    for config in ["Full-18", "Best-7", "Glass-7", "Best-4", "Glass-4", "Best-2", "Glass-2"]:
        cdf = df[df.config == config].sort_values("threshold")
        nch = n_channels_by_config[config]
        if config == "Full-18":
            color = FAMILY_COLORS["Full-18"]
        elif config.startswith("Glass"):
            color = cmap_glass(0.35 + 0.55 * (nch / 18))
        else:
            color = cmap_best(0.35 + 0.55 * (nch / 18))
        ax.plot(cdf.fa_per_day_szcore.clip(lower=1e-3), cdf.sens_szcore,
                color=color, linewidth=2, label=f"{config} ({nch}ch)", zorder=3)

    for pt in ZANETTI_2025_POINTS:
        ax.scatter(pt["fa"], pt["sens"], marker="*", s=220, color="#e6194b",
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(pt["label"], (pt["fa"], pt["sens"]), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color="#e6194b")

    ax.set_xscale("log")
    ax.set_xlim(1e-2, 3e2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False alarms / day (micro, event-level SzCORE, pooled held-out test)")
    ax.set_ylabel("Sensitivity (micro, event-level SzCORE)")
    n_grid_points = df.groupby("config").threshold.nunique().max()
    ax.set_title(f"Sensitivity-vs-FA/day tradeoff by config\n"
                 f"(coarse {n_grid_points}-point threshold grid, visualization only - not the locked\n"
                 f"279-point grid used for actual operating-point selection in 06_results)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2, framealpha=0.9)

    fig.savefig(FIG_DIR / "fig2_tradeoff.svg", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig2_tradeoff.svg")


def fig3_per_patient_heatmap():
    df = pd.read_csv(FIG_DIR / "data_per_patient_heatmap.csv")
    config_order = ["Full-18", "Best-7", "Glass-7", "Best-4", "Glass-4", "Best-2", "Glass-2"]

    # sharey=False deliberately: each panel sorts subjects independently by
    # its OWN criterion's pass rate ("this criterion" in the ylabel below),
    # so the two panels' y-tick labels differ. sharey=True shares the tick
    # formatter across both axes, so the second set_yticklabels() call
    # silently overwrote the first panel's labels while leaving its cell
    # colors untouched - row labels and cell data went out of sync (bug
    # found by project owner 2026-08-17, comparing chb06 in the rendered
    # figure against data_per_patient_heatmap.csv).
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=False)
    for ax, criterion in zip(axes, ["medication_titration", "realtime_alert"]):
        cdf = df[df.criterion == criterion]
        subj_order = (
            cdf.groupby("subject").met.mean().sort_values(ascending=False).index.tolist()
        )
        pivot = cdf.pivot(index="subject", columns="config", values="met").reindex(
            index=subj_order, columns=config_order
        )
        mat = pivot.astype(float).values
        ax.imshow(mat, cmap=matplotlib.colors.ListedColormap(["#f2f2f2", "#2ca02c"]),
                  vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(config_order)))
        ax.set_xticklabels(config_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(subj_order)))
        ytick_labels = ax.set_yticklabels(subj_order, fontsize=7)
        for label, subj in zip(ytick_labels, subj_order):
            if subj in ("chb12", "chb14", "chb06", "chb13", "chb20"):
                label.set_color(HARD_CASE_COLOR)
                label.set_fontweight("bold")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, hatch="///",
                                                      fill=False, edgecolor="#bbb", linewidth=0))
        ax.set_title(CRITERION_LABEL[criterion], fontsize=10)

    for ax in axes:
        ax.set_ylabel("Subject (sorted by pass rate across configs, this criterion)")
    fig.suptitle("Per-patient pass/fail by config (5-fold-per-subject, SzCORE, RF)\n"
                 "red/bold = recurring hard-case subjects (chb12/14/06/13/20)  |  "
                 "hatched = no valid operating point for that fold", fontsize=10)
    legend_elems = [
        mpatches.Patch(color="#2ca02c", label="Met criterion"),
        mpatches.Patch(color="#f2f2f2", label="Did not meet"),
        mpatches.Patch(facecolor="white", edgecolor="#bbb", hatch="///", label="No valid threshold (fold)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(FIG_DIR / "fig3_per_patient_heatmap.svg", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig3_per_patient_heatmap.svg")


def fig4_szcore_vs_ali():
    df = pd.read_csv(FIG_DIR / "data_szcore_vs_ali.csv")
    df = df[df.scheme == "5fold"]
    config_order = ["Full-18", "Best-7", "Glass-7", "Best-4", "Glass-4", "Best-2", "Glass-2"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = np.arange(len(config_order))
    width = 0.36
    for ax, criterion in zip(axes, ["medication_titration", "realtime_alert"]):
        cdf = df[df.criterion == criterion].set_index("config")
        sz = cdf[cdf.rule == "szcore"].reindex(config_order).pct_subjects_meeting_criterion
        al = cdf[cdf.rule == "ali"].reindex(config_order).pct_subjects_meeting_criterion
        ax.bar(x - width / 2, sz.fillna(0), width, label="SzCORE (primary)", color="#1f77b4")
        ax.bar(x + width / 2, al.fillna(0), width, label="Ali 2024 (secondary)", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(config_order, rotation=45, ha="right", fontsize=8)
        ax.set_title(CRITERION_LABEL[criterion], fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("% subjects meeting criterion")
    axes[0].legend(fontsize=8)
    fig.suptitle("Scoring rule comparison: SzCORE vs. Ali 2024, same predictions, same config\n"
                 "(5-fold-per-subject, RF) - shows how much the evaluation rule itself moves the number", fontsize=10)

    fig.savefig(FIG_DIR / "fig4_szcore_vs_ali.svg", bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig4_szcore_vs_ali.svg")


def main():
    fig1_channel_ladder()
    fig2_tradeoff()
    fig3_per_patient_heatmap()
    fig4_szcore_vs_ali()


if __name__ == "__main__":
    main()
