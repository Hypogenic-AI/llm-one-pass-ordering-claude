# Datasets for LLM One-pass Ordering Experiments

## Overview

This directory contains datasets for testing LLMs' ability to order items by various factual properties.

## Dataset 1: Ordering Tasks (ordering_tasks.json)

### Overview
- **Source**: Curated from well-known factual knowledge
- **Size**: 22 ordering tasks, each with 6-10 items
- **Format**: JSON array of ordering task objects
- **Task**: Given a list of items and a property, produce the correct ordering

### Structure
Each task contains:
- `id`: Unique task identifier
- `property`: The property to sort by (e.g., "population", "height", "date of birth")
- `category`: Task category (syntactic, factual_wellknown, factual_knowledge, factual_specific, temporal)
- `difficulty`: Expected difficulty (easy, medium, hard)
- `description`: Human-readable description
- `items`: Shuffled list of items to sort
- `correct_order`: Ground truth correct ordering

### Categories
- **syntactic** (2 tasks): Alphabetical ordering - tests basic string comparison
- **factual_wellknown** (3 tasks): Well-known facts like planet order, continent size
- **factual_knowledge** (6 tasks): World knowledge like country population, animal weight
- **temporal** (4 tasks): Chronological ordering of events, presidents, inventions
- **factual_specific** (7 tasks): Specific numeric facts like atomic numbers, city latitudes

### Loading
```python
import json
with open("datasets/ordering_tasks.json") as f:
    tasks = json.load(f)
```

## Dataset 2: NBA Player Heights (nba_heights_200.csv)

### Overview
- **Source**: OpenIntro / OrderByLLM repository
- **Size**: 200 NBA players
- **Format**: CSV with columns: full_name, h_meters
- **Task**: Order players by height
- **License**: Public domain

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/nba_heights_200.csv")
```

## Dataset 3: World Population by Country (population_by_country_2020.csv)

### Overview
- **Source**: OrderByLLM repository (Worldometer data)
- **Size**: 199 countries/regions
- **Format**: CSV with columns: Country, Population (2020), and more
- **Task**: Order countries by population
- **License**: Public domain

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/population_by_country_2020.csv")
```

## Experiment Design

For each ordering task:
1. Present the LLM with a shuffled list of items
2. Ask it to order them by the specified property
3. Compare the output ordering to the ground truth
4. Measure accuracy using Kendall's tau correlation

Key metrics:
- **Kendall's tau**: Measures rank correlation (-1 to 1, higher is better)
- **Exact match rate**: Whether the ordering is perfectly correct
- **Spearman's rho**: Alternative rank correlation metric
