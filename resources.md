# Resources Catalog

## Summary
This document catalogs all resources gathered for the LLM One-pass Orderings research project.

## Papers
Total papers downloaded: 12

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| SortBench | Herbold | 2025 | papers/sortbench_benchmarking_llms_sorting.pdf | LLM sorting benchmark across data types |
| Access Paths for Efficient Ordering | Zhao et al. | 2025 | papers/access_paths_efficient_ordering_llm.pdf | Pointwise vs comparison-based LLM sorting |
| Semantic Operators (LOTUS) | Patel et al. | 2025 | papers/semantic_operators_declarative_model.pdf | Declarative semantic ordering operators |
| Found in the Middle | Tang et al. | 2023 | papers/found_in_middle_permutation_self_consistency.pdf | Positional bias mitigation in ranking |
| Measuring Inconsistency | Various | 2024 | papers/measuring_inconsistency_llm_ranking.pdf | Ranking consistency analysis |
| LLM-RankFusion | Zeng et al. | 2024 | papers/llm_rankfusion_inconsistency.pdf | Order/transitive inconsistency in LLM ranking |
| Implicit Ranking Unfairness | Various | 2023 | papers/implicit_ranking_unfairness_llm.pdf | Bias in LLM ranking |
| ChatGPT as Re-Ranking Agent | Sun et al. | 2023 | papers/chatgpt_reranking_agent.pdf | RankGPT listwise ranking approach |
| Challenging BIG-Bench Tasks | Suzgun et al. | 2022 | papers/challenging_bigbench_tasks_cot.pdf | BIG-Bench sorting/ordering tasks |
| Optimal Algorithms for LLM Sorting | Various | 2025 | papers/optimal_algorithms_sorting_llm_pairwise.pdf | Sorting with noisy LLM comparisons |
| CoLoTa | Various | 2025 | papers/colota_commonsense_reasoning.pdf | Entity commonsense reasoning |
| OrderProbe | Various | 2026 | papers/orderprobe_structural_reconstruction.pdf | LLM order sensitivity |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Ordering Tasks | Curated | 22 tasks, 6-10 items each | Order items by property | datasets/ordering_tasks.json | Primary dataset, 5 categories, 3 difficulty levels |
| NBA Heights | OrderByLLM/OpenIntro | 200 players | Order by height | datasets/nba_heights_200.csv | Factual ordering benchmark |
| World Population | OrderByLLM/Worldometer | 199 countries | Order by population | datasets/population_by_country_2020.csv | Factual ordering benchmark |

See datasets/README.md for detailed descriptions and loading instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| OrderByLLM | github.com/ZhaoFuheng/OrderByLLM | LLM sorting algorithms + factual ordering evaluation | code/OrderByLLM/ | Contains NBA/population datasets, sorting implementations |
| LOTUS | github.com/lotus-data/lotus | Semantic data processing with LLM operators | code/lotus/ | Provides sem_topk semantic ordering operator |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder with multiple queries covering LLM ordering, sorting, ranking, knowledge probing
2. Searched arXiv, Semantic Scholar, and web for specific papers on LLM factual ordering
3. Identified gap: no systematic study of property-dependent ordering ability
4. Found key datasets from the Access Paths paper (OrderByLLM repo)

### Selection Criteria
- Papers directly about LLM sorting/ordering ability (SortBench, Access Paths)
- Papers about LLM ranking consistency and bias (important confounds)
- Papers about systems for LLM-based data ordering (LOTUS)
- Datasets with factual properties that can be used for ordering experiments

### Challenges Encountered
- Most LLM ranking literature focuses on passage/document re-ranking (IR), not factual property ordering
- No existing dataset specifically designed for multi-property factual ordering
- SortBench code repository was blinded in the paper at review time

### Gaps and Workarounds
- **No multi-property ordering dataset**: Created our own (ordering_tasks.json) with 22 tasks across 5 categories
- **No systematic property taxonomy**: Proposed categories: syntactic, factual_wellknown, factual_knowledge, factual_specific, temporal

## Recommendations for Experiment Design

1. **Primary dataset**: ordering_tasks.json (22 tasks testing ordering by diverse properties)
2. **Baseline comparison**: NBA heights and World Population from OrderByLLM for validating against prior results
3. **Evaluation metrics**: Kendall's tau (primary), exact match, Spearman's rho
4. **Code to adapt/reuse**: OrderByLLM repository has sorting algorithm implementations and evaluation code
5. **Key insight from literature**: Pointwise (one-pass) ordering works well for factual data in LLM parametric memory. Our experiment should test *which* types of factual knowledge are most/least accessible.
