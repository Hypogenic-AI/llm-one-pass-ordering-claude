# LLM One-pass Orderings: Which Features Do LLMs Have Easiest Access To?

## 1. Executive Summary

We systematically tested how well large language models can sort lists of items by different properties in a single pass—alphabetical order, physical weight, historical dates, geographic facts, and more. By measuring ordering accuracy across 22 tasks in 5 property categories, we identify which types of knowledge LLMs access most reliably from parametric memory.

**Key finding**: LLMs order by syntactic properties (alphabetical) and well-known categorical facts (planet order, president chronology) nearly perfectly, but struggle with continuous numeric properties requiring precise recall (city elevations, river lengths, building heights). This suggests LLMs have excellent access to *categorical/sequential* knowledge but poorer access to *precise quantitative* knowledge. GPT-4.1 significantly outperforms GPT-4o-mini (mean τ = 0.948 vs 0.867, p = 0.008), but both models agree on which properties are hardest (Spearman ρ = 0.61, p = 0.003).

## 2. Goal

**Hypothesis**: The ability of LLMs to correctly order items by various properties varies systematically, and these patterns reveal which features LLMs have easier or harder access to in parametric memory.

**Why this matters**: LLMs are increasingly used for data processing including sorting and ranking. Understanding which properties they can reliably sort by informs (1) when to trust LLM-based data pipelines, (2) what knowledge representations LLMs have internalized, and (3) how to design better evaluation benchmarks.

**Gap filled**: Prior work (SortBench 2025) tested syntactic sorting; Access Paths (2025) tested a few factual properties. No work has systematically compared ordering accuracy across a taxonomy of property types to identify patterns of feature accessibility.

## 3. Data Construction

### Dataset Description
- **Primary dataset**: `ordering_tasks.json` — 22 hand-curated ordering tasks
- **5 categories**: syntactic (2), factual_wellknown (3), factual_knowledge (6), factual_specific (7), temporal (4)
- **3 difficulty levels**: easy (5), medium (8), hard (9)
- **Items per task**: 6–10 items
- **Ground truth**: Verified correct orderings for each property

### Example Samples

| Task ID | Property | Items (sample) | Category |
|---------|----------|----------------|----------|
| alpha_countries | Alphabetical order | France, Germany, Brazil, Japan, ... | Syntactic |
| planet_distance | Distance from Sun | Saturn, Earth, Mars, Mercury, ... | Well-known fact |
| city_elevation | Elevation above sea level | Denver, La Paz, Amsterdam, ... | Specific fact |
| celebrity_dob | Date of birth | Taylor Swift, Elvis Presley, ... | Temporal |

### Preprocessing
- Each task tested with 3 random shuffles of input order (to control for positional bias)
- Temperature = 0 for reproducibility
- Standardized prompt: "Sort the following items by [property]. Return ONLY the sorted list."

## 4. Experiment Description

### Methodology

#### High-Level Approach
One-pass listwise ordering: present a shuffled list to the LLM, ask it to sort by a specified property, parse the output, and compare to ground truth using rank correlation metrics.

#### Why This Method?
One-pass ordering (vs. pairwise or iterative) directly tests what knowledge the LLM can access from parametric memory without multi-step reasoning. This is the most revealing approach for our research question.

### Implementation Details

#### Tools and Libraries
- Python 3.12, OpenAI API (openai 1.x)
- scipy 1.15.x (Kendall's τ, Spearman's ρ, statistical tests)
- matplotlib 3.10.x, seaborn 0.13.x (visualization)
- numpy 2.2.x, pandas 2.2.x

#### Models Tested
| Model | Reasoning | Cost (per 1M tokens) |
|-------|-----------|---------------------|
| GPT-4.1 | Strong, current SOTA | ~$2/input, ~$8/output |
| GPT-4o-mini | Efficient, smaller | ~$0.15/input, ~$0.60/output |

#### Hyperparameters
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Temperature | 0.0 | Reproducibility |
| Shuffles per task | 3 | Control for positional bias |
| Max tokens | 1000 | Sufficient for 10-item lists |
| Random seed | 42 | Reproducibility |

### Evaluation Metrics
- **Kendall's τ** (primary): Rank correlation, -1 to 1. Measures fraction of concordant pairs.
- **Exact match**: Binary — is ordering perfectly correct?
- **Spearman's ρ**: Alternative rank correlation (more sensitive to displacement magnitude)

### Raw Results

#### Overall Performance

| Model | Mean τ | Std τ | Exact Match Rate | Perfect Tasks |
|-------|--------|-------|------------------|---------------|
| GPT-4.1 | 0.948 | 0.064 | 54.5% | 10/22 |
| GPT-4o-mini | 0.867 | 0.190 | 47.0% | 7/22 |

#### Performance by Category

| Category | GPT-4.1 τ | GPT-4o-mini τ |
|----------|-----------|---------------|
| Syntactic (Alphabetical) | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Well-Known Facts | 1.000 ± 0.000 | 0.884 ± 0.165 |
| Temporal Ordering | 0.974 ± 0.045 | 0.926 ± 0.054 |
| General Knowledge | 0.909 ± 0.049 | 0.843 ± 0.164 |
| Specific Facts | 0.930 ± 0.075 | 0.810 ± 0.256 |

#### Per-Task Results (GPT-4.1)

| Task | Category | τ | Exact Match |
|------|----------|---|-------------|
| alpha_countries | Syntactic | 1.000 | 100% |
| alpha_animals | Syntactic | 1.000 | 100% |
| planet_distance | Well-Known | 1.000 | 100% |
| planet_size | Well-Known | 1.000 | 100% |
| continent_area | Well-Known | 1.000 | 100% |
| us_president_chronological | Temporal | 1.000 | 100% |
| invention_date | Temporal | 1.000 | 100% |
| historical_event_date | Temporal | 1.000 | 100% |
| element_atomic_number | Specific | 1.000 | 100% |
| color_wavelength | Specific | 1.000 | 100% |
| mohs_hardness | Specific | 1.000 | 100% |
| element_weight | Specific | 0.970 | 67% |
| speed_of_animals | Knowledge | 0.956 | 0% |
| animal_weight | Knowledge | 0.956 | 0% |
| country_population | Knowledge | 0.941 | 0% |
| animal_lifespan | Knowledge | 0.911 | 33% |
| celebrity_dob | Temporal | 0.896 | 0% |
| river_length | Specific | 0.881 | 0% |
| building_height | Knowledge | 0.867 | 0% |
| city_latitude | Specific | 0.852 | 0% |
| country_area | Knowledge | 0.822 | 0% |
| city_elevation | Specific | 0.807 | 0% |

### Visualizations

All plots saved to `results/plots/`:
- `category_comparison.png` — Bar chart comparing models across categories
- `task_heatmap.png` — Heatmap of τ values for all task/model combinations
- `task_ranking.png` — Horizontal bar chart ranking tasks by difficulty
- `model_agreement.png` — Scatter plot showing cross-model correlation
- `difficulty_breakdown.png` — Performance by difficulty level within categories

## 5. Result Analysis

### Key Findings

**Finding 1: Syntactic ordering is trivially easy.**
Both models achieve τ = 1.000 on alphabetical sorting with 100% exact match. This is the easiest property to access—it's a mechanical operation on token representations.

**Finding 2: Well-known categorical sequences are near-perfect for strong models.**
GPT-4.1 achieves τ = 1.000 on planet order, planet size, and continent area. These are heavily represented in training data as fixed sequences. However, GPT-4o-mini drops to τ = 0.651 on continent area, showing this knowledge is less robustly encoded in smaller models.

**Finding 3: Temporal ordering is surprisingly strong.**
GPT-4.1 achieves τ = 1.000 on US presidents, inventions, and historical events (all medium difficulty). This suggests LLMs have excellent access to temporal/sequential knowledge for historically significant events. The exception is celebrity birth dates (τ = 0.896), where precise dates matter more than relative chronology.

**Finding 4: Continuous quantitative properties are hardest.**
The bottom 5 tasks for both models all involve ordering by continuous numeric values:
- City elevation (τ = 0.807/0.733)
- Country area (τ = 0.822/0.763)
- City latitude (τ = 0.852/0.659)
- Building height (τ = 0.867/0.511)
- River length (τ = 0.881/0.274)

These require recalling specific numbers (e.g., "Denver is at 1,609m, Nairobi is at 1,795m") rather than categorical knowledge.

**Finding 5: Some "specific" facts are actually well-known sequences.**
Element atomic numbers (τ = 1.000), Mohs hardness scale (τ = 1.000), and color wavelength order (τ = 1.000) are labeled "specific" but are actually well-known *fixed sequences* taught in education. The key distinction is not specificity but whether the ordering forms a commonly-taught sequence vs. requiring independent numeric recall.

**Finding 6: GPT-4.1 significantly outperforms GPT-4o-mini.**
Wilcoxon signed-rank test: W = 10.5, p = 0.008. The mean difference is +0.081 τ units. GPT-4.1 is more uniformly strong, while GPT-4o-mini has catastrophic failures on some tasks (river_length: τ = 0.274).

**Finding 7: Models agree on which properties are hard.**
Cross-model Spearman correlation: ρ = 0.611, p = 0.003. The property accessibility hierarchy is not model-specific—it reflects genuine differences in how well different types of knowledge are encoded.

### Property Accessibility Hierarchy

Based on our results, LLMs have access to features in this order (easiest to hardest):

1. **Algorithmic/syntactic** (τ ≈ 1.0): Alphabetical sorting — a mechanical operation
2. **Famous sequences** (τ ≈ 1.0): Planet order, periodic table, Mohs scale, color spectrum — memorized as fixed sequences
3. **Historical chronology** (τ ≈ 0.97): Presidents, inventions, events — strong temporal associations
4. **Approximate magnitudes** (τ ≈ 0.93): Animal weights, speeds, populations — rough size categories
5. **Precise numeric facts** (τ ≈ 0.83): City elevations, river lengths, building heights — requires exact number recall

### Hypothesis Testing Results

| Hypothesis | Supported? | Evidence |
|-----------|-----------|----------|
| H1: Syntactic easiest | **Yes** | τ = 1.000 for both models |
| H2: Well-known > obscure | **Partially** | Well-known facts perfect for GPT-4.1, but some "specific" facts (Mohs, elements) are also perfect because they're known sequences |
| H3: Temporal intermediate | **No** — temporal was near-perfect | τ = 0.974 for GPT-4.1, almost as good as syntactic |
| H4: Difficulty varies within categories | **Yes** | Large variance within factual_specific (0.807 to 1.000) |
| H5: Patterns consistent across models | **Yes** | ρ = 0.611, p = 0.003 |

### Statistical Tests

- **Kruskal-Wallis across categories (GPT-4.1)**: H = 7.79, p = 0.10. Not significant at α = 0.05, but trending. The high performance across the board compresses variance.
- **Kruskal-Wallis (GPT-4o-mini)**: H = 4.17, p = 0.38. Not significant, due to high within-category variance.
- **Model comparison (Wilcoxon)**: W = 10.5, p = 0.008. Significant — GPT-4.1 is reliably better.

### Error Analysis

The most common error pattern is **adjacent swaps** rather than gross misordering. For example:
- Building height: Big Ben and Statue of Liberty swapped (96m vs 93m — very close)
- Country area: Australia and Brazil swapped (similar sizes)
- City elevation: Fine ordering of mid-range cities confused

This suggests LLMs have rough magnitude knowledge but lose precision when values are close together, consistent with storing approximate rather than exact numeric facts.

### Surprises and Insights

1. **Mohs hardness scale**: Perfect performance despite being "factual_specific" — this is a well-known educational sequence, not a set of independent facts.
2. **River length near-failure for GPT-4o-mini** (τ = 0.274): This is close to random. The model appears to have very little knowledge of relative river lengths.
3. **Temporal ordering strength**: We expected this to be intermediate, but famous historical sequences are extremely well-encoded.
4. **The real distinction is sequence vs. independent facts**: Tasks where items form a commonly-taught sequence (planets, periodic table, presidents) are easy. Tasks where each item's value must be independently recalled from memory are hard.

### Limitations

1. **Small task set**: 22 tasks with 6-10 items each. Larger lists would likely show more degradation.
2. **Only 2 models tested**: Adding Claude, Gemini, and open models would strengthen generalizability claims.
3. **Ground truth ambiguity**: Some orderings depend on measurement methodology (e.g., "typical" animal lifespan).
4. **Only English**: Cross-lingual ordering might reveal different patterns.
5. **Temperature 0 only**: Higher temperatures might show different variance patterns.
6. **Category boundaries**: The distinction between "well-known" and "specific" facts is somewhat subjective.
7. **Statistical power**: With only 2-7 tasks per category, between-category tests lack power.

## 6. Conclusions

### Summary
LLMs can reliably order items by properties that are encoded as **well-known sequences** (alphabetical, planetary, chronological) or **categorical magnitude** (animal sizes, population rough order). They struggle with properties requiring **precise numeric recall** of independent facts (city elevations, river lengths, building heights). The key distinction is not whether knowledge is "well-known" vs. "obscure," but whether it's stored as a **sequence** vs. **independent numeric values**.

### Implications

**Practical**: When building LLM-based data processing pipelines, trust LLMs for ordering by well-known categorical or sequential properties. Use external tools (databases, APIs) for ordering by precise numeric facts.

**Theoretical**: LLMs appear to store factual knowledge in two forms: (1) sequential/associative chains (A comes before B) that enable reliable ordering, and (2) approximate numeric representations that enable rough magnitude comparison but not precise ordering. This aligns with the "Access Paths" paper's finding that pointwise ordering works well only when facts are in parametric memory.

**For LLM evaluation**: Ordering tasks provide a clean, scalable way to probe what knowledge LLMs have internalized. The gap between sequence-based and numeric ordering reveals the boundary between "knowing" and "precisely recalling."

### Confidence in Findings
**Moderate-high**. The patterns are clear and consistent across models, with statistically significant model differences. However, the task set is small and only 2 models were tested. The core finding—that sequences are easier than independent numeric facts—is robust.

## 7. Next Steps

### Immediate Follow-ups
1. **Scale up list length**: Test with 20, 50, 100 items to find degradation curves
2. **More models**: Test Claude 4.5, Gemini 2.5 Pro, Llama 3, Qwen 2.5 to confirm cross-model patterns
3. **More property types**: Add sensory properties (taste, smell descriptors), cultural knowledge, relative spatial relationships

### Alternative Approaches
- **Pointwise scoring**: Ask LLMs to assign numeric values, then sort externally — compare accuracy to direct ordering
- **Chain-of-thought**: Test if CoT reasoning improves ordering of hard properties
- **Fine-grained error analysis**: For each error, determine if the LLM's ordering reflects a plausible but incorrect value or a random guess

### Open Questions
1. Does ordering accuracy correlate with fact frequency in training data?
2. Do LLMs store ordering relationships (A > B) directly, or derive them from stored values?
3. Can we predict ordering difficulty from property characteristics without running experiments?
4. How does list length interact with property type — do "easy" properties degrade at different rates than "hard" ones?

## References

1. Herbold, S. (2025). SortBench: Benchmarking LLMs based on their ability to sort lists. arXiv:2504.08312.
2. Zhao, F. et al. (2025). Access Paths for Efficient Ordering with Large Language Models. arXiv:2509.00303.
3. Tang, R. et al. (2023). Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking. arXiv:2310.07712.
4. Patel, L. et al. (2025). Semantic Operators: A Declarative Model for Rich, AI-based Data Processing. arXiv:2407.11418.
5. Sun, W. et al. (2023). Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. arXiv:2304.09542.
