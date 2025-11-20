import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

MCCE_ROOT = "<TO_BE_FILLED>"
if MCCE_ROOT not in sys.path:
    sys.path.append(MCCE_ROOT)


_embedding_cache = {}

try:
    from algorithm.base import Item  # type: ignore
except (ImportError, ModuleNotFoundError):
    class Item:
        def __init__(self, value, property_dict):
            self.value = value
            self.property = property_dict
            self.total = 0.0

try:
    from problem.motsp import evaluator as motsp_evaluator  # type: ignore
except (ImportError, ModuleNotFoundError):
    motsp_evaluator = None

MCCE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROBLEM_DATA_PATH = os.path.join(MCCE_ROOT, "data", "problems", "MOTSP", "problem.txt")


def _load_problem_data_fallback():
    if not os.path.exists(DEFAULT_PROBLEM_DATA_PATH):
        return None

    cities = []
    with open(DEFAULT_PROBLEM_DATA_PATH, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                x1, y1, x2, y2 = map(float, line.split(","))
                cities.append([x1, y1, x2, y2])
            except ValueError:
                continue
    if not cities:
        return None
    return np.asarray(cities, dtype=np.float32)


def _load_motsp_cities():
    if motsp_evaluator is not None:
        try:
            return motsp_evaluator.load_problem_data(1)
        except Exception as exc:
            print(f"[WARN] load_problem_data failed, fallback to text parsing: {exc}")
    return _load_problem_data_fallback()


def _prepare_distance_matrices(cities):
    if cities is None or len(cities) == 0:
        return None, None
    coords1 = cities[:, :2]
    coords2 = cities[:, 2:]
    dist1 = np.linalg.norm(coords1[:, None, :] - coords1[None, :, :], axis=2).astype(np.float32)
    dist2 = np.linalg.norm(coords2[:, None, :] - coords2[None, :, :], axis=2).astype(np.float32)
    return dist1, dist2


MOTSP_CITIES = _load_motsp_cities()
DIST1, DIST2 = _prepare_distance_matrices(MOTSP_CITIES)


def _parse_motsp_solution(sol_str):
    if not isinstance(sol_str, str) or "tour" not in sol_str:
        return None
    local_vars = {}
    try:
        exec(sol_str, {"np": np}, local_vars)
    except Exception as exc:
        print(f"[WARN] Failed to parse MOTSP solution: {exc}")
        return None
    tour = local_vars.get("tour")
    if tour is None:
        return None
    tour = np.asarray(tour).astype(int).flatten()
    if tour.size == 0:
        return None
    return tour


def _tour_to_embedding(tour):
    if DIST1 is None or DIST2 is None:
        return None
    n = tour.size
    edges_obj1 = np.zeros(n, dtype=np.float32)
    edges_obj2 = np.zeros(n, dtype=np.float32)
    city_count = DIST1.shape[0]

    for idx in range(n):
        a = int(tour[idx])
        b = int(tour[(idx + 1) % n])
        if a < 0 or a >= city_count or b < 0 or b >= city_count:
            return None
        edges_obj1[idx] = DIST1[a, b]
        edges_obj2[idx] = DIST2[a, b]

    embedding = np.concatenate([edges_obj1, edges_obj2])
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        return None
    return (embedding / norm).astype(np.float32)


def get_embedding(sol_str):
    global _embedding_cache
    if sol_str in _embedding_cache:
        return _embedding_cache[sol_str]
    tour = _parse_motsp_solution(sol_str)
    if tour is None:
        _embedding_cache[sol_str] = None
        return None
    emb = _tour_to_embedding(tour)
    _embedding_cache[sol_str] = emb
    return emb


def calculate_similarity(sol1, sol2):
    emb1 = get_embedding(sol1)
    emb2 = get_embedding(sol2)
    if emb1 is None or emb2 is None:
        return 0.0
    sim = float(np.dot(emb1, emb2))
    if not np.isfinite(sim):
        return 0.0
    sim = max(min(sim, 1.0), -1.0)
    return (sim + 1.0) / 2.0


def process_one_prompt(idx, query, high_score_pool_solutions, high_score_extended_pool_solutions,
                       low_score_pool_solutions, solution_to_total):
    prompt_text = query.get('prompt', '')
    parents = query.get('parents', [])

    chosen_solutions = find_similar_solutions(
        parents,
        high_score_pool_solutions,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.3,
        max_similarity=0.95,
        show_progress=False
    )
    if len(chosen_solutions) < 2:
        chosen_solutions = find_similar_solutions(
            parents,
            high_score_extended_pool_solutions,
            similarity_thresholds=[0.7, 0.6, 0.5],
            min_threshold=0.3,
            max_similarity=0.95,
            show_progress=False
        )

    rejected_solutions = find_similar_solutions(
        parents,
        low_score_pool_solutions,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.0,
        max_similarity=0.95,
        show_progress=False
    )

    while len(chosen_solutions) < 2 and len(high_score_pool_solutions) > 0:
        chosen_solutions.append(high_score_pool_solutions[np.random.randint(len(high_score_pool_solutions))][0])
    while len(rejected_solutions) < 2 and len(low_score_pool_solutions) > 0:
        rejected_solutions.append(low_score_pool_solutions[np.random.randint(len(low_score_pool_solutions))][0])

    chosen_response = f"<candidate>{chosen_solutions[0]}</candidate>\n<candidate>{chosen_solutions[1]}</candidate>"
    rejected_response = f"<candidate>{rejected_solutions[0]}</candidate>\n<candidate>{rejected_solutions[1]}</candidate>"

    dpo_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

    parent_values = [parent.get('value', '') for parent in parents if parent.get('value')]
    similarities = {}
    for p_idx, parent_val in enumerate(parent_values[:2]):
        if parent_val:
            for c_idx, chosen in enumerate(chosen_solutions[:2]):
                try:
                    sim = calculate_similarity(parent_val, chosen)
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = 0.0
            for r_idx, rejected in enumerate(rejected_solutions[:2]):
                try:
                    sim = calculate_similarity(parent_val, rejected)
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = 0.0

    for p in range(1, 3):
        for t in ['c', 'r']:
            for m in range(1, 3):
                key = f"similar_{p}_{t}{m}"
                if key not in similarities:
                    similarities[key] = 0.0

    parent_data = []
    for idx_p in range(min(2, len(parents))):
        parent = parents[idx_p]
        parent_data.append({
            "solution": parent.get('value', ''),
            "total": parent.get('total', 0.0)
        })
    while len(parent_data) < 2:
        parent_data.append({"solution": "", "total": 0.0})

    chosen_totals = [solution_to_total.get(sol, 0.0) for sol in chosen_solutions]
    rejected_totals = [solution_to_total.get(sol, 0.0) for sol in rejected_solutions]

    full_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "parents": parent_data,
        "chosen_solutions": [
            {
                "solution": chosen_solutions[idx_c] if idx_c < len(chosen_solutions) else '',
                "total": chosen_totals[idx_c] if idx_c < len(chosen_totals) else 0.0
            }
            for idx_c in range(2)
        ],
        "rejected_solutions": [
            {
                "solution": rejected_solutions[idx_r] if idx_r < len(rejected_solutions) else '',
                "total": rejected_totals[idx_r] if idx_r < len(rejected_totals) else 0.0
            }
            for idx_r in range(2)
        ]
    }
    full_item.update(similarities)
    return idx, dpo_item, full_item


def load_prompt_history(pkl_path):
    pkl_dir = os.path.dirname(pkl_path)
    pkl_basename = os.path.basename(pkl_path).replace('.pkl', '')

    parent_dir = os.path.dirname(pkl_dir)
    prompt_dir = os.path.join(parent_dir, 'prompt')

    if os.path.exists(prompt_dir):
        for filename in os.listdir(prompt_dir):
            if filename.endswith('_prompt.json') and pkl_basename in filename:
                prompt_file_path = os.path.join(prompt_dir, filename)
                try:
                    with open(prompt_file_path, 'r', encoding='utf-8') as fin:
                        prompt_data = json.load(fin)
                    return prompt_data.get('queries', [])
                except Exception as exc:
                    print(f"Failed to load prompt file {prompt_file_path}: {exc}")
                    break

    print("No prompt history found, using default empty list")
    return []


def find_similar_solutions(target_parents, sol_pool,
                           similarity_thresholds=None,
                           min_threshold=0.3,
                           max_similarity=0.95,
                           show_progress=True):
    if similarity_thresholds is None:
        similarity_thresholds = [0.7, 0.6, 0.5]

    if not target_parents or not sol_pool:
        selected = np.random.choice(len(sol_pool), min(2, len(sol_pool)), replace=False)
        return [sol_pool[i][0] for i in selected]

    parent_values = [parent.get('value', '') for parent in target_parents if parent.get('value')]
    if not parent_values:
        selected = np.random.choice(len(sol_pool), min(2, len(sol_pool)), replace=False)
        return [sol_pool[i][0] for i in selected]

    selected_solutions = []
    used_indices = set()

    for threshold in similarity_thresholds:
        if len(selected_solutions) >= 2:
            break

        iterator = tqdm(sol_pool, desc=f"Similarity filtering (threshold={threshold:.1f})", leave=False) if show_progress else sol_pool
        for sol_str, sol_idx in iterator:
            if sol_idx in used_indices or len(selected_solutions) >= 2:
                continue

            max_similarity_score = 0.0
            for parent_val in parent_values:
                try:
                    similarity = calculate_similarity(parent_val, sol_str)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception:
                    continue

            if threshold <= max_similarity_score <= max_similarity:
                selected_solutions.append(sol_str)
                used_indices.add(sol_idx)

        if len(selected_solutions) >= 2:
            break

    if len(selected_solutions) < 2:
        similarities = []
        iterator = tqdm(sol_pool, desc="Finding highest similarity candidates", leave=False) if show_progress else sol_pool
        for sol_str, sol_idx in iterator:
            if sol_idx in used_indices:
                continue
            max_similarity_score = 0.0
            for parent_val in parent_values:
                try:
                    similarity = calculate_similarity(parent_val, sol_str)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception:
                    continue
            if max_similarity_score <= max_similarity:
                similarities.append((max_similarity_score, sol_str, sol_idx))
        similarities.sort(key=lambda x: x[0], reverse=True)
        needed = 2 - len(selected_solutions)
        for i in range(min(needed, len(similarities))):
            selected_solutions.append(similarities[i][1])

    while len(selected_solutions) < 2 and len(sol_pool) > len(selected_solutions):
        available_candidates = []
        iterator = tqdm(sol_pool, desc="Supplementary candidate filtering", leave=False) if show_progress else enumerate(sol_pool)
        iterable = enumerate(iterator) if show_progress else iterator
        for i, (sol_str, sol_idx) in iterable:
            if sol_idx in used_indices:
                continue
            max_similarity_score = 0.0
            for parent_val in parent_values:
                try:
                    similarity = calculate_similarity(parent_val, sol_str)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception:
                    continue
            if max_similarity_score <= max_similarity:
                available_candidates.append(i)

        if available_candidates:
            selected_idx = np.random.choice(available_candidates)
            sol_str = sol_pool[selected_idx][0]
            selected_solutions.append(sol_str)
            used_indices.add(sol_pool[selected_idx][1])
        else:
            break

    return selected_solutions[:2]


def create_dpo_data_from_pkl_v2(pkl_path, output_json_path, num_pairs=None):
    print(f"Loading optimization data from {pkl_path}")
    with open(pkl_path, "rb") as fin:
        data = pickle.load(fin)

    all_items = data.get("all_mols", [])
    if not all_items:
        print("No candidates found in the pkl file.")
        return 0

    sorted_items = sorted(all_items, key=lambda x: x[0].total, reverse=True)
    print(f"Found {len(sorted_items)} candidates, sorted by total score")

    prompt_history = load_prompt_history(pkl_path)
    print(f"Loaded {len(prompt_history)} historical prompts")

    if len(prompt_history) > 128:
        prompt_history = prompt_history[-128:]
        print(f"Limited to latest {len(prompt_history)} prompts")

    if num_pairs is None:
        num_pairs = min(len(prompt_history), data['evaluation'][-1]['all_unique_moles'] // 2)

    if num_pairs == 0:
        print("No prompts available for generating DPO data.")
        return 0

    total_items = len(sorted_items)
    high_score_pool_size = int(total_items * 0.3)
    low_score_pool_size = int(total_items * 0.3)

    high_score_pool = [(sorted_items[i][0], i) for i in range(high_score_pool_size)]
    low_score_pool = [(sorted_items[i][0], i) for i in range(total_items - low_score_pool_size, total_items)]
    high_score_extended_pool = [(sorted_items[i][0], i) for i in range(int(total_items * 0.5))]

    print(f"High score pool: {len(high_score_pool)} candidates")
    print(f"Low score pool: {len(low_score_pool)} candidates")

    high_score_pool_solutions = [(item.value, idx) for item, idx in high_score_pool]
    low_score_pool_solutions = [(item.value, idx) for item, idx in low_score_pool]
    high_score_extended_pool_solutions = [(item.value, idx) for item, idx in high_score_extended_pool]

    solution_to_total = {}
    for item, _ in sorted_items:
        solution_to_total[item.value] = item.total

    dpo_data = [None] * min(num_pairs, len(prompt_history))
    full_data = [None] * min(num_pairs, len(prompt_history))

    total_pairs = min(num_pairs, len(prompt_history))
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(
                process_one_prompt,
                i,
                prompt_history[i],
                high_score_pool_solutions,
                high_score_extended_pool_solutions,
                low_score_pool_solutions,
                solution_to_total,
            )
            for i in range(total_pairs)
        ]
        for fut in tqdm(as_completed(futures), total=total_pairs, desc="Parallel DPO data generation (embedding similarity)", unit="pair"):
            idx, dpo_item, full_item = fut.result()
            dpo_data[idx] = dpo_item
            full_data[idx] = full_item

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as fout:
        json.dump(dpo_data, fout, indent=2, ensure_ascii=False)

    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fulldata_dir = os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training", "fulldata")
    os.makedirs(fulldata_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(output_json_path))[0]
    fulldata_path = os.path.join(fulldata_dir, f"{base_filename}_full.json")
    with open(fulldata_path, "w", encoding="utf-8") as fout:
        json.dump(full_data, fout, indent=2, ensure_ascii=False)

    print(f"DPO dataset completed, {len(dpo_data)} pairs generated, saved to {output_json_path}")
    print(f"Full data saved to {fulldata_path}")
    print("Chosen solutions from high score range, rejected solutions from low score range, selected based on embedding similarity with parents")
    return len(dpo_data)


def main():
    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(
        description="Build embedding-based DPO data for MOTSP task and trigger training."
    )
    parser.add_argument("--exp", required=True, help="Experiment name for output directory naming")
    parser.add_argument("--pkl_path", required=True, help="Path to pkl file containing optimization trajectory")
    parser.add_argument("--data_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training"), 
                       help="Directory to save DPO JSON data")
    parser.add_argument("--model_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_models"), 
                       help="Directory to save trained models")
    parser.add_argument("--prev_exp", help="Previous experiment name for warm start")
    parser.add_argument("--num_pairs", type=int, help="Optional, number of data pairs to generate")
    parser.add_argument(
        "--ref_model_path",
        default="<TO_BE_FILLED>",
        help="Reference model path for DPO training (fixed base)",
    )

    args = parser.parse_args()

    output_json_path = os.path.join(args.data_dir, f"{args.exp}.json")
    model_output_path = os.path.join(args.model_dir, args.exp)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Creating DPO dataset (embedding similarity) for experiment: {args.exp}")
    num_data_pairs = create_dpo_data_from_pkl_v2(args.pkl_path, output_json_path, args.num_pairs)

    if num_data_pairs == 0:
        print("No data pairs generated. Exiting.")
        return

    training_script = os.path.join(MCCE_PROJECT_ROOT, "training", "train_dpo.py")
    print(f"Starting DPO training for experiment: {args.exp}")

    command_list = [
        "python",
        training_script,
        "--train_data_path",
        output_json_path,
        "--output_dir",
        model_output_path,
        "--exp_name",
        args.exp,
        "--ref_model_path",
        args.ref_model_path,
    ]

    if args.prev_exp:
        prev_model_path = os.path.join(args.model_dir, args.prev_exp)
        if os.path.exists(prev_model_path):
            command_list.extend(["--model_name_or_path", prev_model_path])
            print(f"Using policy model from previous experiment: {prev_model_path}")
            print(f"Using reference model (fixed): {args.ref_model_path}")
        else:
            print(f"Warning: Previous model path {prev_model_path} does not exist. Using default base model.")
            print(f"Using reference model (fixed): {args.ref_model_path}")
    else:
        print("Using default base model for policy model")
        print(f"Using reference model (fixed): {args.ref_model_path}")

    command_str = " ".join(shlex.quote(c) for c in command_list)
    final_command = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate verl && {command_str}"

    print("Running command:")
    print(f"cd {MCCE_PROJECT_ROOT} && {final_command}")

    start_time = time.time()

    try:
        subprocess.run(
            final_command,
            check=True,
            cwd=MCCE_PROJECT_ROOT,
            shell=True,
            executable="/bin/bash",
        )
        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"DPO training script finished successfully in {duration:.2f} minutes.")
        print(f"Trained model saved to: {model_output_path}")
    except subprocess.CalledProcessError as exc:
        print(f"Error running DPO training script: {exc}")
        raise
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error during DPO training: {exc}")
        raise


if __name__ == "__main__":
    main()


