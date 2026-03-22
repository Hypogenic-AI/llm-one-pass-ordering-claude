# Research Plan: LLM One-pass Orderings

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used for data processing tasks including sorting and ranking. Understanding which properties LLMs can reliably order by reveals what factual knowledge is easily accessible in their parametric memory vs. what requires computation or external tools. This has practical implications for building reliable LLM-based data pipelines and for understanding LLM knowledge representations.

### Gap in Existing Work
SortBench (2025) tests syntactic sorting (numbers, strings) but not factual/semantic sorting. Access Paths (2025) tests a few factual properties (NBA heights, country populations) but doesn't systematically compare across property types. **No existing work creates a taxonomy of property accessibility by testing LLMs on diverse ordering tasks spanning syntactic, temporal, and factual properties.**

### Our Novel Contribution
We systematically test LLM one-pass ordering ability across 22 tasks in 5 categories (syntactic, factual_wellknown, factual_knowledge, factual_specific, temporal) and 3 difficulty levels. By measuring ordering accuracy per category, we reveal which "features" LLMs have easiest access to, creating a property accessibility hierarchy.

### Experiment Justification
- **Experiment 1 (Main ordering test)**: Tests GPT-4.1 on all 22 tasks with 3 random shuffles each. Needed to establish the baseline ordering performance across all property types.
- **Experiment 2 (Model comparison)**: Tests GPT-4o-mini on same tasks. Needed to see if property accessibility patterns are model-dependent or universal.
- **Experiment 3 (Positional bias check)**: Multiple shuffles per task reveal whether failures are due to knowledge gaps or input order sensitivity.

## Research Question
Can we use LLM ordering accuracy across diverse property types (alphabetical, weight, dates, geographic facts, etc.) to identify which features LLMs have easier or harder access to in their parametric memory?

## Hypothesis Decomposition
1. **H1**: Syntactic properties (alphabetical order) are easiest to order correctly.
2. **H2**: Well-known factual properties (planet distances, continent sizes) are easier than obscure ones (city elevations, river lengths).
3. **H3**: Temporal ordering (historical events, birth dates) falls between syntactic and factual in difficulty.
4. **H4**: Performance varies by difficulty level within categories.
5. **H5**: Property accessibility patterns are consistent across different models.

## Proposed Methodology

### Approach
One-pass listwise ordering: present shuffled items to the LLM, ask it to sort by a specified property, parse the output, and compare to ground truth using Kendall's tau.

### Experimental Steps
1. For each of 22 tasks, create 3 random shuffles of the items
2. Send each shuffle to the LLM with a standardized prompt
3. Parse the LLM's ordered output
4. Compute Kendall's tau, Spearman's rho, and exact match against ground truth
5. Aggregate by category and difficulty level
6. Compare across models (GPT-4.1 vs GPT-4o-mini)

### Baselines
- Random ordering (expected Kendall's tau ≈ 0)
- Input order (no reordering - tests if model just echoes input)

### Evaluation Metrics
- **Kendall's tau** (primary): rank correlation, -1 to 1
- **Spearman's rho**: alternative rank correlation
- **Exact match**: binary - is the ordering perfectly correct?
- **Displacement score**: average positional displacement per item

### Statistical Analysis Plan
- Compare category means with one-way ANOVA or Kruskal-Wallis
- Pairwise comparisons with Bonferroni correction
- 95% confidence intervals for all metrics
- Effect sizes (Cohen's d) for key comparisons

## Expected Outcomes
- Syntactic ordering (alphabetical): tau > 0.9
- Well-known facts (planets, continents): tau > 0.8
- General knowledge (populations, weights): tau 0.5-0.8
- Specific facts (elevations, river lengths): tau 0.3-0.6
- Temporal ordering: tau 0.6-0.9 (varies by familiarity)

## Timeline
- Phase 1 (Planning): 10 min ✓
- Phase 2 (Setup): 10 min
- Phase 3 (Implementation): 30 min
- Phase 4 (Run experiments): 30 min
- Phase 5 (Analysis): 20 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- API rate limits → use exponential backoff
- Output parsing failures → robust regex + JSON parsing
- Ambiguous ground truths → verify with multiple sources
- Positional bias → mitigate with multiple shuffles

## Success Criteria
1. Successfully run ordering tests on all 22 tasks with at least 2 models
2. Identify statistically significant differences between property categories
3. Create a clear property accessibility hierarchy
