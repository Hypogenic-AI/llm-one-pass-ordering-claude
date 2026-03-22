# Cloned Repositories

## 1. OrderByLLM
- **URL**: https://github.com/ZhaoFuheng/OrderByLLM
- **Paper**: "Access Paths for Efficient Ordering with Large Language Models" (arXiv:2509.00303)
- **Purpose**: Provides implementations of various LLM-based sorting algorithms (pointwise, pairwise, listwise, merge sort) and evaluation on factual ordering tasks (NBA heights, world population)
- **Location**: code/OrderByLLM/
- **Key files**:
  - `data/` - NBA heights, population, DL19/DL20 passage ranking data
  - `order_by/` - Implementation of sorting algorithms
  - `prompts/` - Prompt templates for different sorting approaches
- **Relevance**: Contains evaluation code for factual ordering and datasets we can reuse. The pointwise vs comparison-based distinction is central to understanding which properties LLMs can sort by.

## 2. LOTUS
- **URL**: https://github.com/lotus-data/lotus
- **Paper**: "Semantic Operators: A Declarative Model for Rich, AI-based Data Processing" (arXiv:2407.11418)
- **Purpose**: System for semantic data processing with LLMs, including semantic ordering (sem_topk operator)
- **Location**: code/lotus/
- **Key files**:
  - `lotus/` - Core library implementing semantic operators
- **Relevance**: Provides the sem_topk operator which implements LLM-based ordering. Can potentially be used as infrastructure for running ordering experiments.
