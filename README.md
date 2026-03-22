# LLM One-pass Orderings

**Research question**: If you give an LLM a list of items and ask it to sort by some property (alphabetical order, weight, date of birth, etc.), which orderings succeed and which fail? Can these patterns reveal which features LLMs have easier access to?

## Key Findings

- **Syntactic ordering (alphabetical) is trivially easy** — both GPT-4.1 and GPT-4o-mini achieve perfect τ = 1.000
- **Famous sequences are near-perfect** — planet order, periodic table, Mohs scale, presidents chronology all achieve τ ≈ 1.0 for GPT-4.1
- **Temporal ordering is surprisingly strong** — historical events, inventions, presidencies ordered almost perfectly
- **Continuous numeric facts are hardest** — city elevations (τ = 0.81), river lengths (τ = 0.88), building heights (τ = 0.87) require precise number recall
- **The real distinction is sequence vs. independent numeric facts** — not "well-known" vs. "obscure"
- **GPT-4.1 significantly outperforms GPT-4o-mini** (mean τ = 0.948 vs 0.867, p = 0.008), but both agree on which properties are hardest (ρ = 0.61)

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy scipy matplotlib seaborn pandas

# Run experiment (requires OPENAI_API_KEY)
python src/run_experiment.py

# Analyze results and generate plots
python src/analyze_results.py
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── datasets/
│   ├── ordering_tasks.json      # 22 ordering tasks (primary dataset)
│   ├── nba_heights_200.csv      # NBA player heights (reference)
│   └── population_by_country_2020.csv  # Country populations (reference)
├── src/
│   ├── run_experiment.py        # Main experiment script
│   └── analyze_results.py       # Analysis and visualization
├── results/
│   ├── ordering_results.json    # Raw experimental results
│   ├── config.json              # Experiment configuration
│   └── plots/                   # All visualizations
├── papers/                      # Related work PDFs
├── code/                        # Reference implementations
└── literature_review.md         # Literature review
```

See [REPORT.md](REPORT.md) for full details.
