"""
LLM One-pass Ordering Experiment
Tests how well LLMs can order lists of items by various properties in a single pass.
"""

import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI
from scipy import stats

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Config
NUM_SHUFFLES = 3  # shuffles per task
MODELS = ["gpt-4.1", "gpt-4o-mini"]
TEMPERATURE = 0.0
RESULTS_DIR = Path("/workspaces/llm-one-pass-ordering-claude/results")
RESULTS_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def load_tasks():
    with open("/workspaces/llm-one-pass-ordering-claude/datasets/ordering_tasks.json") as f:
        return json.load(f)


def create_prompt(items, property_desc):
    """Create a standardized ordering prompt."""
    items_str = "\n".join(f"- {item}" for item in items)
    return (
        f"Sort the following items by {property_desc}. "
        f"Return ONLY the sorted list, one item per line, with no numbering, bullets, or explanations.\n\n"
        f"{items_str}"
    )


def parse_response(response_text, original_items):
    """Parse LLM response into an ordered list, matching against original items."""
    lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
    # Remove common prefixes like "1.", "- ", "* "
    cleaned = []
    for line in lines:
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-\*]\s*", "", line)
        line = line.strip()
        if line:
            cleaned.append(line)

    # Match to original items (fuzzy: case-insensitive)
    orig_lower = {item.lower(): item for item in original_items}
    matched = []
    for item in cleaned:
        key = item.lower()
        if key in orig_lower:
            matched.append(orig_lower[key])
        else:
            # Try partial match
            for orig_key, orig_val in orig_lower.items():
                if orig_key in key or key in orig_key:
                    matched.append(orig_val)
                    break

    return matched


def kendall_tau(predicted, ground_truth):
    """Compute Kendall's tau between predicted and ground truth orderings."""
    if len(predicted) != len(ground_truth):
        return None
    # Convert to rank arrays
    gt_rank = {item: i for i, item in enumerate(ground_truth)}
    try:
        pred_ranks = [gt_rank[item] for item in predicted]
    except KeyError:
        return None
    gt_ranks = list(range(len(ground_truth)))
    tau, p_value = stats.kendalltau(pred_ranks, gt_ranks)
    return tau


def spearman_rho(predicted, ground_truth):
    """Compute Spearman's rho between predicted and ground truth orderings."""
    if len(predicted) != len(ground_truth):
        return None
    gt_rank = {item: i for i, item in enumerate(ground_truth)}
    try:
        pred_ranks = [gt_rank[item] for item in predicted]
    except KeyError:
        return None
    gt_ranks = list(range(len(ground_truth)))
    rho, p_value = stats.spearmanr(pred_ranks, gt_ranks)
    return rho


def exact_match(predicted, ground_truth):
    return predicted == ground_truth


def avg_displacement(predicted, ground_truth):
    """Average absolute positional displacement per item."""
    if len(predicted) != len(ground_truth):
        return None
    gt_rank = {item: i for i, item in enumerate(ground_truth)}
    try:
        displacements = [abs(i - gt_rank[item]) for i, item in enumerate(predicted)]
    except KeyError:
        return None
    return np.mean(displacements)


def run_ordering_task(model, items, property_desc, task_id):
    """Run a single ordering task and return results."""
    prompt = create_prompt(items, property_desc)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that sorts items accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=1000
        )
        response_text = response.choices[0].message.content
        return {
            "response": response_text,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
    except Exception as e:
        print(f"  ERROR on {task_id}: {e}")
        return {"response": "", "error": str(e)}


def main():
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} ordering tasks")
    print(f"Models: {MODELS}")
    print(f"Shuffles per task: {NUM_SHUFFLES}")
    print(f"Total API calls: {len(tasks) * NUM_SHUFFLES * len(MODELS)}")
    print()

    all_results = {}

    for model in MODELS:
        print(f"{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")
        model_results = []

        for task in tasks:
            task_id = task["id"]
            correct_order = task["correct_order"]
            property_desc = task["property"]
            category = task["category"]
            difficulty = task["difficulty"]
            items = task["items"]

            print(f"\n  Task: {task_id} ({category}/{difficulty})")

            task_runs = []
            for shuffle_idx in range(NUM_SHUFFLES):
                shuffled = items.copy()
                random.shuffle(shuffled)

                result = run_ordering_task(model, shuffled, property_desc, task_id)
                response_text = result.get("response", "")
                parsed = parse_response(response_text, items)

                # Compute metrics
                tau = kendall_tau(parsed, correct_order)
                rho = spearman_rho(parsed, correct_order)
                em = exact_match(parsed, correct_order)
                disp = avg_displacement(parsed, correct_order)

                run_result = {
                    "shuffle_idx": shuffle_idx,
                    "input_order": shuffled,
                    "parsed_output": parsed,
                    "correct_order": correct_order,
                    "raw_response": response_text,
                    "kendall_tau": tau,
                    "spearman_rho": rho,
                    "exact_match": em,
                    "avg_displacement": disp,
                    "n_items_matched": len(parsed),
                    "n_items_expected": len(correct_order),
                }
                task_runs.append(run_result)

                tau_str = f"{tau:.3f}" if tau is not None else "N/A"
                print(f"    Shuffle {shuffle_idx}: tau={tau_str}, exact={em}, matched={len(parsed)}/{len(correct_order)}")

                # Small delay to avoid rate limits
                time.sleep(0.3)

            # Aggregate per task
            taus = [r["kendall_tau"] for r in task_runs if r["kendall_tau"] is not None]
            rhos = [r["spearman_rho"] for r in task_runs if r["spearman_rho"] is not None]
            ems = [r["exact_match"] for r in task_runs]

            task_summary = {
                "task_id": task_id,
                "category": category,
                "difficulty": difficulty,
                "property": property_desc,
                "description": task["description"],
                "n_items": len(correct_order),
                "runs": task_runs,
                "mean_kendall_tau": float(np.mean(taus)) if taus else None,
                "std_kendall_tau": float(np.std(taus)) if taus else None,
                "mean_spearman_rho": float(np.mean(rhos)) if rhos else None,
                "exact_match_rate": float(np.mean(ems)),
                "mean_displacement": float(np.mean([r["avg_displacement"] for r in task_runs if r["avg_displacement"] is not None])) if any(r["avg_displacement"] is not None for r in task_runs) else None,
            }
            model_results.append(task_summary)

            mean_tau = f"{task_summary['mean_kendall_tau']:.3f}" if task_summary['mean_kendall_tau'] is not None else "N/A"
            print(f"    => Mean tau={mean_tau}, EM rate={task_summary['exact_match_rate']:.1%}")

        all_results[model] = model_results

    # Save results
    output_file = RESULTS_DIR / "ordering_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    # Save config
    config = {
        "seed": SEED,
        "num_shuffles": NUM_SHUFFLES,
        "models": MODELS,
        "temperature": TEMPERATURE,
        "n_tasks": len(tasks),
        "timestamp": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
