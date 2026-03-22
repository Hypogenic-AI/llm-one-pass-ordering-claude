"""
Analysis and visualization of LLM ordering experiment results.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path("/workspaces/llm-one-pass-ordering-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load results
with open(RESULTS_DIR / "ordering_results.json") as f:
    all_results = json.load(f)

CATEGORY_LABELS = {
    "syntactic": "Syntactic\n(Alphabetical)",
    "factual_wellknown": "Well-Known\nFacts",
    "factual_knowledge": "General\nKnowledge",
    "factual_specific": "Specific\nFacts",
    "temporal": "Temporal\nOrdering",
}
CATEGORY_ORDER = ["syntactic", "factual_wellknown", "temporal", "factual_knowledge", "factual_specific"]

# ── 1. Aggregate by category ──
print("=" * 70)
print("RESULTS BY CATEGORY")
print("=" * 70)

for model, tasks in all_results.items():
    print(f"\n--- {model} ---")
    cat_taus = defaultdict(list)
    for t in tasks:
        if t["mean_kendall_tau"] is not None:
            cat_taus[t["category"]].append(t["mean_kendall_tau"])

    for cat in CATEGORY_ORDER:
        vals = cat_taus.get(cat, [])
        if vals:
            print(f"  {cat:25s}: tau = {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)} tasks)")

# ── 2. Per-task comparison table ──
print("\n" + "=" * 70)
print("PER-TASK RESULTS")
print("=" * 70)
print(f"{'Task ID':<30} {'Category':<20} {'GPT-4.1 tau':>12} {'GPT-4o-mini tau':>15} {'Diff':>8}")
print("-" * 85)

models = list(all_results.keys())
task_ids = [t["task_id"] for t in all_results[models[0]]]
task_lookup = {}
for model in models:
    for t in all_results[model]:
        task_lookup[(model, t["task_id"])] = t

for tid in task_ids:
    t0 = task_lookup.get((models[0], tid))
    t1 = task_lookup.get((models[1], tid))
    tau0 = t0["mean_kendall_tau"] if t0 and t0["mean_kendall_tau"] is not None else float('nan')
    tau1 = t1["mean_kendall_tau"] if t1 and t1["mean_kendall_tau"] is not None else float('nan')
    diff = tau0 - tau1 if not (np.isnan(tau0) or np.isnan(tau1)) else float('nan')
    cat = t0["category"] if t0 else "?"
    print(f"{tid:<30} {cat:<20} {tau0:>12.3f} {tau1:>15.3f} {diff:>+8.3f}")

# ── 3. Statistical tests ──
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

for model in models:
    print(f"\n--- {model}: Kruskal-Wallis test across categories ---")
    cat_taus = defaultdict(list)
    for t in all_results[model]:
        if t["mean_kendall_tau"] is not None:
            cat_taus[t["category"]].append(t["mean_kendall_tau"])

    groups = [cat_taus[c] for c in CATEGORY_ORDER if c in cat_taus]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        h_stat, p_val = stats.kruskal(*groups)
        print(f"  H = {h_stat:.3f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print("  => Significant difference across categories (p < 0.05)")
        else:
            print("  => No significant difference (p >= 0.05)")

    # Pairwise Mann-Whitney with Bonferroni
    print(f"\n  Pairwise Mann-Whitney U tests (Bonferroni corrected):")
    n_comparisons = len(CATEGORY_ORDER) * (len(CATEGORY_ORDER) - 1) // 2
    for i, c1 in enumerate(CATEGORY_ORDER):
        for j, c2 in enumerate(CATEGORY_ORDER):
            if j <= i:
                continue
            g1 = cat_taus.get(c1, [])
            g2 = cat_taus.get(c2, [])
            if len(g1) >= 2 and len(g2) >= 2:
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                p_corrected = min(p_val * n_comparisons, 1.0)
                sig = "*" if p_corrected < 0.05 else ""
                print(f"    {c1} vs {c2}: U={u_stat:.1f}, p_corrected={p_corrected:.4f} {sig}")

# ── 4. GPT-4.1 vs GPT-4o-mini comparison ──
print(f"\n--- Model comparison (paired) ---")
paired_taus_41 = []
paired_taus_mini = []
for tid in task_ids:
    t0 = task_lookup.get((models[0], tid))
    t1 = task_lookup.get((models[1], tid))
    if t0 and t1 and t0["mean_kendall_tau"] is not None and t1["mean_kendall_tau"] is not None:
        paired_taus_41.append(t0["mean_kendall_tau"])
        paired_taus_mini.append(t1["mean_kendall_tau"])

if paired_taus_41:
    t_stat, p_val = stats.wilcoxon(paired_taus_41, paired_taus_mini, alternative='two-sided')
    mean_diff = np.mean(np.array(paired_taus_41) - np.array(paired_taus_mini))
    print(f"  GPT-4.1 mean tau: {np.mean(paired_taus_41):.3f}")
    print(f"  GPT-4o-mini mean tau: {np.mean(paired_taus_mini):.3f}")
    print(f"  Mean difference: {mean_diff:+.3f}")
    print(f"  Wilcoxon signed-rank: W={t_stat:.1f}, p={p_val:.4f}")

# ── 5. Correlation between models (do they agree on which tasks are hard?) ──
rho, p_rho = stats.spearmanr(paired_taus_41, paired_taus_mini)
print(f"\n  Cross-model Spearman correlation: rho={rho:.3f}, p={p_rho:.4f}")
print(f"  => Models {'agree' if rho > 0.5 else 'disagree'} on which properties are easier/harder")

# ════════════════════════════════════════════════
# VISUALIZATIONS
# ════════════════════════════════════════════════

sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Plot 1: Category performance comparison (both models) ──
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(CATEGORY_ORDER))
width = 0.35
colors = ["#2196F3", "#FF9800"]

for mi, model in enumerate(models):
    cat_means = []
    cat_stds = []
    for cat in CATEGORY_ORDER:
        taus = [t["mean_kendall_tau"] for t in all_results[model]
                if t["category"] == cat and t["mean_kendall_tau"] is not None]
        cat_means.append(np.mean(taus) if taus else 0)
        cat_stds.append(np.std(taus) if taus else 0)
    offset = (mi - 0.5) * width
    bars = ax.bar(x + offset, cat_means, width, yerr=cat_stds,
                  label=model, color=colors[mi], alpha=0.85, capsize=4)

ax.set_xlabel("Property Category")
ax.set_ylabel("Mean Kendall's τ")
ax.set_title("LLM One-pass Ordering Accuracy by Property Category")
ax.set_xticks(x)
ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER])
ax.set_ylim(0, 1.15)
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_comparison.png", dpi=150)
plt.close()
print(f"\nSaved: {PLOTS_DIR / 'category_comparison.png'}")

# ── Plot 2: Per-task heatmap ──
fig, ax = plt.subplots(figsize=(8, 12))
task_labels = []
tau_matrix = []
for tid in task_ids:
    row = []
    for model in models:
        t = task_lookup.get((model, tid))
        row.append(t["mean_kendall_tau"] if t and t["mean_kendall_tau"] is not None else np.nan)
    tau_matrix.append(row)
    cat = task_lookup[(models[0], tid)]["category"]
    task_labels.append(f"{tid} [{cat}]")

tau_matrix = np.array(tau_matrix)
sns.heatmap(tau_matrix, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            xticklabels=[m.replace("gpt-", "GPT-") for m in models],
            yticklabels=task_labels, ax=ax, linewidths=0.5)
ax.set_title("Kendall's τ by Task and Model")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "task_heatmap.png", dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'task_heatmap.png'}")

# ── Plot 3: Scatter plot - model agreement ──
fig, ax = plt.subplots(figsize=(7, 7))
cat_colors = {"syntactic": "#4CAF50", "factual_wellknown": "#2196F3",
              "factual_knowledge": "#FF9800", "factual_specific": "#F44336",
              "temporal": "#9C27B0"}

for tid, tau41, tau_mini in zip(task_ids, paired_taus_41, paired_taus_mini):
    cat = task_lookup[(models[0], tid)]["category"]
    ax.scatter(tau41, tau_mini, c=cat_colors[cat], s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

# Add legend
for cat, color in cat_colors.items():
    ax.scatter([], [], c=color, label=cat.replace("_", " ").title(), s=80, edgecolors='black', linewidth=0.5)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel("GPT-4.1 Kendall's τ")
ax.set_ylabel("GPT-4o-mini Kendall's τ")
ax.set_title(f"Model Agreement on Task Difficulty\n(Spearman ρ = {rho:.2f})")
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_agreement.png", dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'model_agreement.png'}")

# ── Plot 4: Difficulty breakdown within categories ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for mi, model in enumerate(models):
    ax = axes[mi]
    diff_cat_data = defaultdict(lambda: defaultdict(list))
    for t in all_results[model]:
        if t["mean_kendall_tau"] is not None:
            diff_cat_data[t["category"]][t["difficulty"]].append(t["mean_kendall_tau"])

    cats = CATEGORY_ORDER
    diffs = ["easy", "medium", "hard"]
    diff_colors = {"easy": "#4CAF50", "medium": "#FF9800", "hard": "#F44336"}

    x = np.arange(len(cats))
    w = 0.25
    for di, diff in enumerate(diffs):
        vals = []
        for cat in cats:
            v = diff_cat_data[cat].get(diff, [])
            vals.append(np.mean(v) if v else 0)
        offset = (di - 1) * w
        ax.bar(x + offset, vals, w, label=diff, color=diff_colors[diff], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=9)
    ax.set_ylabel("Mean Kendall's τ")
    ax.set_title(model)
    ax.set_ylim(0, 1.15)
    ax.legend(title="Difficulty")

fig.suptitle("Ordering Accuracy by Category and Difficulty Level", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "difficulty_breakdown.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {PLOTS_DIR / 'difficulty_breakdown.png'}")

# ── Plot 5: Task ranking (best to worst) for GPT-4.1 ──
fig, ax = plt.subplots(figsize=(10, 8))
task_data = [(t["task_id"], t["mean_kendall_tau"], t["category"])
             for t in all_results[models[0]] if t["mean_kendall_tau"] is not None]
task_data.sort(key=lambda x: x[1], reverse=True)

y_pos = np.arange(len(task_data))
colors_list = [cat_colors[td[2]] for td in task_data]
bars = ax.barh(y_pos, [td[1] for td in task_data], color=colors_list, alpha=0.85, edgecolor='black', linewidth=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels([td[0] for td in task_data], fontsize=9)
ax.set_xlabel("Mean Kendall's τ")
ax.set_title(f"GPT-4.1: Task Ranking (Best to Worst)")
ax.set_xlim(0, 1.1)
ax.invert_yaxis()

# Add legend
for cat, color in cat_colors.items():
    ax.barh([], [], color=color, label=cat.replace("_", " ").title())
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "task_ranking.png", dpi=150)
plt.close()
print(f"Saved: {PLOTS_DIR / 'task_ranking.png'}")

# ── Summary statistics for report ──
print("\n" + "=" * 70)
print("SUMMARY FOR REPORT")
print("=" * 70)

for model in models:
    taus = [t["mean_kendall_tau"] for t in all_results[model] if t["mean_kendall_tau"] is not None]
    ems = [t["exact_match_rate"] for t in all_results[model]]
    print(f"\n{model}:")
    print(f"  Overall mean tau: {np.mean(taus):.3f} ± {np.std(taus):.3f}")
    print(f"  Overall exact match rate: {np.mean(ems):.1%}")
    print(f"  Perfect tasks (tau=1.0): {sum(1 for t in taus if t == 1.0)}/{len(taus)}")
    print(f"  Lowest tau: {min(taus):.3f}")

    # Hardest tasks
    sorted_tasks = sorted(all_results[model], key=lambda t: t["mean_kendall_tau"] if t["mean_kendall_tau"] is not None else 2)
    print(f"  Hardest 5 tasks:")
    for t in sorted_tasks[:5]:
        tau = t["mean_kendall_tau"]
        print(f"    {t['task_id']:30s} tau={tau:.3f}  [{t['category']}]")

print("\n\nAnalysis complete!")
