# Literature Review: LLM One-pass Orderings

## Research Area Overview

This research investigates how well large language models (LLMs) can order lists of items by various factual properties in a single pass (without multi-step reasoning or tool use). The hypothesis is that LLM ordering ability varies systematically across property types—some orderings consistently succeed, some fail, and some fail abysmally—and that these patterns reveal which features LLMs have easier or harder access to in their parametric memory.

This work sits at the intersection of three research areas: (1) LLM-based sorting/ranking, (2) factual knowledge probing in LLMs, and (3) LLM evaluation benchmarks.

## Key Papers

### Paper 1: SortBench: Benchmarking LLMs based on their ability to sort lists
- **Authors**: Steffen Herbold
- **Year**: 2025
- **Source**: arXiv:2504.08312
- **Key Contribution**: First comprehensive benchmark specifically for LLM sorting ability across different data types and difficulties.
- **Methodology**: Generates lists of various types (integers, floats, English words, random strings) at different lengths (2-256 items) and evaluates LLMs on sorting correctness, faithfulness (not changing items), and output validity.
- **Datasets Used**: Synthetically generated lists of integers, floats, words, random strings, number words.
- **Results**:
  - o3-mini dominates overall (0.977 ModelScore), followed by GPT-4o (0.914)
  - Claude-3.5-Sonnet excels at sorting correctness (0.962) but struggles with output validity
  - Key finding: **NumberWords task** (sorting written-out numbers like "three", "seven" lexicographically) fools reasoning models—they convert to numeric values instead of sorting lexicographically
  - All models struggle with faithfulness on long lists (drop/add items)
  - Test-time reasoning can cause *overthinking*, especially in DeepSeek-r1
- **Code Available**: Repository mentioned but blinded in paper
- **Relevance**: Directly related—tests syntactic sorting. Our research extends this to *semantic/factual* sorting (ordering by properties like weight, height, date of birth). SortBench's NumberWords finding about syntax vs. semantics confusion is particularly relevant.

### Paper 2: Access Paths for Efficient Ordering with Large Language Models
- **Authors**: Fuheng Zhao et al. (Snowflake, UChicago, UCLA, UCSB)
- **Year**: 2025
- **Source**: arXiv:2509.00303
- **Key Contribution**: Systematic study of different physical implementations for LLM-based semantic sorting (pointwise, pairwise, listwise, merge sort).
- **Methodology**: Compares value-based (pointwise) vs. comparison-based (pairwise) sorting across factual and semantic tasks.
- **Datasets Used**: NBA player heights (200 players), World Population (200 countries), TREC DL19/DL20 (passage ranking), TweetEval (sentiment), SembenchMovie (review sentiment).
- **Results**:
  - **Critical finding for our research**: For factual data (NBA heights, country populations), **pointwise methods achieve high accuracy with minimal cost** because LLMs directly recall memorized facts from training data
  - For non-factual tasks (passage relevance ranking), comparison-based methods excel
  - No single algorithm is universally optimal across all task types
  - The authors note pointwise success on factual tasks likely stems from **memorization of training corpora** (e.g., Wikipedia)
- **Code Available**: https://github.com/ZhaoFuheng/OrderByLLM
- **Relevance**: Highly relevant—demonstrates that LLM ordering success on factual properties depends on whether the knowledge is in parametric memory. Our research systematically tests which properties are accessible.

### Paper 3: Semantic Operators: A Declarative Model for Rich, AI-based Data Processing
- **Authors**: Liana Patel et al. (Stanford, UC Berkeley)
- **Year**: 2025
- **Source**: arXiv:2407.11418
- **Key Contribution**: Formalizes semantic operators including sem_topk for LLM-based ranking, implemented in the LOTUS system.
- **Methodology**: Proposes semantic operators (filter, join, sort, group-by, aggregate) with natural language specifications and accuracy guarantees.
- **Datasets Used**: FEVER (fact-checking), BioDEX (biomedical), TREC DL (search), ArXiv papers (topic analysis).
- **Results**: LOTUS programs match or exceed SOTA AI pipelines while being more concise and efficient.
- **Code Available**: https://github.com/lotus-data/lotus
- **Relevance**: Provides system infrastructure for semantic ordering experiments.

### Paper 4: Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking
- **Authors**: Raphael Tang et al.
- **Year**: 2023
- **Source**: arXiv:2310.07712
- **Key Contribution**: Addresses positional bias in LLM listwise ranking by shuffling input order and aggregating results.
- **Methodology**: Repeatedly shuffles list order in prompts and aggregates rankings to marginalize positional bias.
- **Results**: Improves scores by up to 34-52% for Mistral, 7-18% for GPT-3.5. Also includes **sorting experiments** (not just passage ranking).
- **Code Available**: https://github.com/castorini/perm-sc
- **Relevance**: Positional bias is a confound we must account for in our ordering experiments.

### Paper 5: Measuring the Inconsistency of LLMs in Preferential Ranking
- **Authors**: Various
- **Year**: 2024
- **Source**: arXiv:2410.08851
- **Key Contribution**: Studies how consistently LLMs rank items, identifying order and transitive inconsistencies.
- **Relevance**: Important for understanding whether ordering failures are due to lack of knowledge or inconsistent reasoning.

### Paper 6: LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking
- **Authors**: Yifan Zeng et al.
- **Year**: 2024
- **Source**: arXiv:2406.00231
- **Key Contribution**: Identifies order inconsistency and transitive inconsistency in LLM pairwise comparisons and proposes mitigation strategies.
- **Relevance**: These inconsistencies affect one-pass ordering—if the LLM can't make transitive comparisons, sorting will fail.

### Paper 7: Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
- **Authors**: Suzgun et al.
- **Year**: 2022
- **Source**: arXiv:2210.09261
- **Key Contribution**: Identifies 23 BIG-Bench tasks where LLMs lag behind humans, including tasks related to logical reasoning and knowledge.
- **Relevance**: BIG-Bench includes word_sorting and logical_sequence tasks relevant to ordering ability.

### Paper 8: Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent
- **Authors**: Weiwei Sun et al.
- **Year**: 2023
- **Source**: arXiv:2304.09542
- **Key Contribution**: Introduces RankGPT approach for using LLMs as passage re-rankers, showing competitive performance with supervised methods.
- **Relevance**: Establishes that LLMs can do listwise ranking effectively, forming a baseline approach for our ordering experiments.

## Common Methodologies

1. **Pointwise scoring**: Ask the LLM to assign a numeric score to each item, then sort by score. Works well for factual recall tasks (Access Paths paper).
2. **Pairwise comparison**: Ask the LLM to compare pairs, use sorting algorithm. More robust for semantic tasks but expensive.
3. **Listwise ranking**: Present the entire list and ask for a permutation. One-pass approach most relevant to our research.
4. **Permutation self-consistency**: Shuffle inputs multiple times and aggregate rankings to reduce positional bias.

## Standard Baselines

- **Pointwise sorting** (value-based): Used by BigQuery ML, many analytic systems
- **Quicksort with LLM comparator**: Used by LOTUS
- **RankGPT sliding window**: Listwise approach from Sun et al.
- **Random baseline**: Expected Kendall's tau ≈ 0

## Evaluation Metrics

- **Kendall's tau**: Rank correlation coefficient, -1 to 1 (used for full ranking)
- **nDCG@k**: Normalized discounted cumulative gain (used for top-k ranking)
- **Spearman's rho**: Alternative rank correlation
- **Exact match**: Whether ordering is perfectly correct
- **Unordered pairs (UP)**: Fraction of pairs in wrong order (SortBench)
- **Faithfulness**: Whether output items match input items (SortBench)

## Gaps and Opportunities

1. **No systematic study of property-dependent ordering**: SortBench tests syntactic sorting (numbers, strings). Access Paths tests a few factual properties (height, population). **No work systematically tests which factual properties LLMs can sort by and identifies patterns of success/failure.**

2. **Syntax vs. semantics interaction**: SortBench's NumberWords finding shows that semantic meaning can interfere with syntactic sorting. The reverse—whether syntactic properties (token frequency, string length) affect semantic ordering—is unexplored.

3. **One-pass ordering**: Most LLM sorting work uses multi-step approaches (pairwise, iterative). Testing one-pass listwise ordering reveals what knowledge is directly accessible vs. requiring computation.

4. **Property taxonomy**: No work categorizes orderable properties by how accessible they are to LLMs (e.g., temporal order vs. physical properties vs. geographic facts).

## Recommendations for Our Experiment

### Recommended datasets
- **Primary**: Our curated ordering_tasks.json with 22 tasks across 5 categories and 3 difficulty levels
- **Secondary**: NBA heights and World Population datasets from OrderByLLM for comparison with prior work
- **Validation**: BIG-Bench word_sorting task for syntactic baseline

### Recommended baselines
- Random ordering (expected Kendall's tau ≈ 0)
- Alphabetical ordering (tests if LLM defaults to alphabetical)
- Pointwise scoring approach (from Access Paths)

### Recommended metrics
- Kendall's tau (primary - measures rank correlation)
- Exact match rate (strictest - binary correct/incorrect)
- Spearman's rho (alternative rank correlation)

### Methodological considerations
- **Positional bias**: Shuffle item order in prompts across multiple runs
- **Temperature**: Use temperature=0 for reproducibility
- **Prompt design**: Simple, direct prompts ("Sort the following items by X, from lowest to highest")
- **Models to test**: At least 3 frontier models (GPT-4o, Claude, Gemini) + 1 open model
- **List length**: Start with 10 items per task (manageable for one-pass ordering)
