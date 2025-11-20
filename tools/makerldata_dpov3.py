import json
import pandas as pd
import os
import sys
import argparse
import subprocess
import shlex
import pickle
import numpy as np
import torch
import time
from tdc import Oracle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

MCCE_ROOT = "<TO_BE_FILLED>"
if MCCE_ROOT not in sys.path:
    sys.path.append(MCCE_ROOT)

_similarity_cache = {}
_oracle_cache = {}

try:
    from algorithm.base import Item
except (ImportError, ModuleNotFoundError):
    class Item:
        def __init__(self, value, property_dict):
            self.value = value
            self.property = property_dict
            self.total = 0.0

def calculate_similarity(smiles1, smiles2):
    global _similarity_cache, _oracle_cache
    key = (smiles1, smiles2)
    if key in _similarity_cache:
        return _similarity_cache[key]
    oracle = _oracle_cache.get(smiles1)
    if oracle is None:
        oracle = Oracle(name='Similarity_Meta', target_smiles=smiles1)
        _oracle_cache[smiles1] = oracle
    try:
        similarity = oracle([smiles2])[0]
    except Exception:
        similarity = 0.0
    _similarity_cache[key] = similarity
    return similarity

def process_one_prompt(idx, query, high_score_pool_smiles, high_score_extended_pool_smiles, low_score_pool_smiles, smiles_to_total):
    prompt_text = query.get('prompt', '')
    parents = query.get('parents', [])

    chosen_molecules = find_similar_molecules(
        parents, high_score_pool_smiles,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.3,
        max_similarity=0.8,
        show_progress=False
    )
    if len(chosen_molecules) < 2:
        chosen_molecules = find_similar_molecules(
            parents, high_score_extended_pool_smiles,
            similarity_thresholds=[0.7, 0.6, 0.5],
            min_threshold=0.3,
            max_similarity=0.8,
            show_progress=False
        )

    rejected_molecules = find_similar_molecules(
        parents, low_score_pool_smiles,
        similarity_thresholds=[0.7, 0.6, 0.5],
        min_threshold=0.0,
        max_similarity=0.8,
        show_progress=False
    )

    while len(chosen_molecules) < 2 and len(high_score_pool_smiles) > 0:
        chosen_molecules.append(high_score_pool_smiles[np.random.randint(len(high_score_pool_smiles))][0])
    while len(rejected_molecules) < 2 and len(low_score_pool_smiles) > 0:
        rejected_molecules.append(low_score_pool_smiles[np.random.randint(len(low_score_pool_smiles))][0])

    chosen_response = f"<candidate>{chosen_molecules[0]}</candidate>\n<candidate>{chosen_molecules[1]}</candidate>"
    rejected_response = f"<candidate>{rejected_molecules[0]}</candidate>\n<candidate>{rejected_molecules[1]}</candidate>"

    dpo_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

    parent_smiles = [parent.get('value', '') for parent in parents if parent.get('value')]
    similarities = {}
    for p_idx, parent_smiles_str in enumerate(parent_smiles[:2]):
        if parent_smiles_str:
            for c_idx, chosen_mol in enumerate(chosen_molecules[:2]):
                try:
                    sim = calculate_similarity(parent_smiles_str, chosen_mol)
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_c{c_idx+1}"] = 0.0
            for r_idx, rejected_mol in enumerate(rejected_molecules[:2]):
                try:
                    sim = calculate_similarity(parent_smiles_str, rejected_mol)
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = sim
                except Exception:
                    similarities[f"similar_{p_idx+1}_r{r_idx+1}"] = 0.0

    for p in range(1, 3):
        for t in ['c', 'r']:
            for m in range(1, 2 + 1):
                key = f"similar_{p}_{t}{m}"
                if key not in similarities:
                    similarities[key] = 0.0

    parent_data = []
    for idx_p in range(min(2, len(parents))):
        parent = parents[idx_p]
        parent_data.append({
            "smiles": parent.get('value', ''),
            "total": parent.get('total', 0.0)
        })
    while len(parent_data) < 2:
        parent_data.append({"smiles": "", "total": 0.0})

    chosen_totals = [smiles_to_total.get(sm, 0.0) for sm in chosen_molecules]
    rejected_totals = [smiles_to_total.get(sm, 0.0) for sm in rejected_molecules]

    full_item = {
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "parents": parent_data,
        "chosen_molecules": [
            {"smiles": chosen_molecules[idx_c] if idx_c < len(chosen_molecules) else '',
             "total": chosen_totals[idx_c] if idx_c < len(chosen_totals) else 0.0}
            for idx_c in range(2)
        ],
        "rejected_molecules": [
            {"smiles": rejected_molecules[idx_r] if idx_r < len(rejected_molecules) else '',
             "total": rejected_totals[idx_r] if idx_r < len(rejected_totals) else 0.0}
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
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        prompt_data = json.load(f)
                    return prompt_data.get('queries', [])
                except Exception as e:
                    print(f"Failed to load prompt file {prompt_file_path}: {e}")
                    break
    
    print("No prompt history found, using default empty list")
    return []

def find_similar_molecules(target_parents, mol_pool, similarity_thresholds=[0.7, 0.6, 0.5], min_threshold=0.3, max_similarity=0.8, show_progress=True):
    if not target_parents or not mol_pool:
        selected = np.random.choice(len(mol_pool), min(2, len(mol_pool)), replace=False)
        smiles_list = []
        for i in selected:
            mol_item = mol_pool[i][0]
            mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
            smiles_list.append(mol_smiles)
        return smiles_list
    
    parent_smiles = [parent.get('value', '') for parent in target_parents if parent.get('value')]
    if not parent_smiles:
        selected = np.random.choice(len(mol_pool), min(2, len(mol_pool)), replace=False)
        smiles_list = []
        for i in selected:
            mol_item = mol_pool[i][0]
            mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
            smiles_list.append(mol_smiles)
        return smiles_list
    
    selected_molecules = []
    used_indices = set()
    
    for threshold in similarity_thresholds:
        if len(selected_molecules) >= 2:
            break
            
        iterator = tqdm(mol_pool, desc=f"Similarity filtering (threshold={threshold:.1f})", leave=False) if show_progress else mol_pool
        for mol_item, mol_idx in iterator:
            if mol_idx in used_indices or len(selected_molecules) >= 2:
                continue
                
            mol_smiles = mol_item.value if hasattr(mol_item, 'value') else mol_item
            max_similarity_score = 0.0
            for parent_smiles_str in parent_smiles:
                try:
                    similarity = calculate_similarity(parent_smiles_str, mol_smiles)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
            
            if threshold <= max_similarity_score <= max_similarity:
                selected_molecules.append(mol_smiles)
                used_indices.add(mol_idx)
        
        if len(selected_molecules) >= 2:
            break
    
    if len(selected_molecules) < 2:
        similarities = []
        iterator = tqdm(mol_pool, desc="Finding highest similarity candidates", leave=False) if show_progress else mol_pool
        for mol_item, mol_idx in iterator:
            if mol_idx in used_indices:
                continue
                
            mol_smiles = mol_item.value if hasattr(mol_item, 'value') else mol_item
            max_similarity_score = 0.0
            for parent_smiles_str in parent_smiles:
                try:
                    similarity = calculate_similarity(parent_smiles_str, mol_smiles)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception as e:
                    continue
            
            if max_similarity_score <= max_similarity:
                similarities.append((max_similarity_score, mol_smiles, mol_idx))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        needed = 2 - len(selected_molecules)
        for i in range(min(needed, len(similarities))):
            selected_molecules.append(similarities[i][1])
    
    while len(selected_molecules) < 2 and len(mol_pool) > len(selected_molecules):
        available_candidates = []
        iterator = tqdm(mol_pool, desc="Supplementary candidate filtering", leave=False) if show_progress else enumerate(mol_pool)
        if show_progress:
            iterable = enumerate(iterator)
        else:
            iterable = iterator
        for i, (mol_item, mol_idx) in iterable:
            if mol_idx in used_indices:
                continue
            
            mol_smiles = mol_item.value if hasattr(mol_item, 'value') else mol_item
            max_similarity_score = 0.0
            for parent_smiles_str in parent_smiles:
                try:
                    similarity = calculate_similarity(parent_smiles_str, mol_smiles)
                    max_similarity_score = max(max_similarity_score, similarity)
                except Exception as e:
                    continue
            
            if max_similarity_score <= max_similarity:
                available_candidates.append(i)
        
        if available_candidates:
            selected_idx = np.random.choice(available_candidates)
            mol_item = mol_pool[selected_idx][0]
            mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
            selected_molecules.append(mol_smiles)
            used_indices.add(mol_pool[selected_idx][1])
        else:
            available_indices = [i for i, (_, mol_idx) in enumerate(mol_pool) if mol_idx not in used_indices]
            if available_indices:
                selected_idx = np.random.choice(available_indices)
                mol_item = mol_pool[selected_idx][0]
                mol_smiles = mol_item.value if hasattr(mol_item, "value") else mol_item
                selected_molecules.append(mol_smiles)
                used_indices.add(mol_pool[selected_idx][1])
            else:
                break
    
    return selected_molecules[:2]

def find_molecule_total_score(smiles, sorted_mols):
    for mol_item, _ in sorted_mols:
        if mol_item.value == smiles:
            return mol_item.total
    return 0.0

def create_dpo_data_from_pkl_v2(pkl_path, output_json_path, num_pairs=None):
    print(f"Loading molecular data from {pkl_path}")
    with open(pkl_path, "rb") as fin:
        data = pickle.load(fin)
    
    all_mols = data.get("all_mols", [])
    if not all_mols:
        print("No molecules found in the pkl file.")
        return 0
        
    sorted_mols = sorted(all_mols, key=lambda x: x[0].total, reverse=True)
    print(f"Found {len(sorted_mols)} molecules, sorted by total score")
    
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
    
    total_mols = len(sorted_mols)
    high_score_pool_size = int(total_mols * 0.3)
    low_score_pool_size = int(total_mols * 0.3)
    
    high_score_pool = [(sorted_mols[i][0], i) for i in range(high_score_pool_size)]
    low_score_pool = [(sorted_mols[i][0], i) for i in range(total_mols - low_score_pool_size, total_mols)]
    
    high_score_extended_pool = [(sorted_mols[i][0], i) for i in range(int(total_mols * 0.5))]
    
    print(f"High score pool: {len(high_score_pool)} molecules")
    print(f"Low score pool: {len(low_score_pool)} molecules")
    
    high_score_pool_smiles = [(mol_item.value, idx) for mol_item, idx in high_score_pool]
    low_score_pool_smiles = [(mol_item.value, idx) for mol_item, idx in low_score_pool]
    high_score_extended_pool_smiles = [(mol_item.value, idx) for mol_item, idx in high_score_extended_pool]

    smiles_to_total = {}
    for mol_item, _ in sorted_mols:
        smiles_to_total[mol_item.value] = mol_item.total

    dpo_data = [None] * min(num_pairs, len(prompt_history))
    full_data = [None] * min(num_pairs, len(prompt_history))

    total_pairs = min(num_pairs, len(prompt_history))
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(
                process_one_prompt,
                i,
                prompt_history[i],
                high_score_pool_smiles,
                high_score_extended_pool_smiles,
                low_score_pool_smiles,
                smiles_to_total,
            )
            for i in range(total_pairs)
        ]
        for fut in tqdm(as_completed(futures), total=total_pairs, desc="Parallel DPO data generation", unit="pair"):
            idx, dpo_item, full_item = fut.result()
            dpo_data[idx] = dpo_item
            full_data[idx] = full_item
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as fout:
        json.dump(dpo_data, fout, indent=2, ensure_ascii=False)
    
    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fulldata_dir = os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training", "fulldata")
    os.makedirs(fulldata_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(output_json_path))[0]
    fulldata_path = os.path.join(fulldata_dir, f"{base_filename}_full.json")
    
    with open(fulldata_path, "w") as fout:
        json.dump(full_data, fout, indent=2, ensure_ascii=False)
    
    print(f"DPO dataset completed, {len(dpo_data)} pairs generated, saved to {output_json_path}")
    print(f"Full data saved to {fulldata_path}")
    print(f"Chosen molecules from high score range, rejected molecules from low score range, selected based on similarity with parents")
    return len(dpo_data)

def main():
    MCCE_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="Process molecular data for DPO training and run the training script (v2 with similarity-based selection).")
    parser.add_argument("--exp", required=True, help="Experiment name, used for output files and directories.")
    parser.add_argument("--pkl_path", required=True, help="Path to the input pkl file containing molecular data.")
    parser.add_argument("--data_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_training"), 
                       help="Directory to save training data (JSON).")
    parser.add_argument("--model_dir", default=os.path.join(MCCE_PROJECT_ROOT, "data", "dpo_models"), 
                       help="Directory to save trained models.")
    parser.add_argument("--prev_exp", help="Experiment name of the previous run to use as a base model.")
    parser.add_argument("--num_pairs", type=int, help="Number of data pairs to generate (optional).")
    parser.add_argument("--ref_model_path", default="<TO_BE_FILLED>", 
                       help="Reference model path (always the original base model)")
    
    args = parser.parse_args()

    output_json_path = os.path.join(args.data_dir, f"{args.exp}.json")
    model_output_path = os.path.join(args.model_dir, args.exp)
    
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Creating DPO dataset for experiment: {args.exp}")
    num_data_pairs = create_dpo_data_from_pkl_v2(args.pkl_path, output_json_path, args.num_pairs)
    
    if num_data_pairs == 0:
        print("No data pairs generated. Exiting.")
        return

    training_script = os.path.join(MCCE_PROJECT_ROOT, "training", "train_dpo.py")
    print(f"Starting DPO training for experiment: {args.exp}")

    command_list = [
        "python", 
        training_script,
        "--train_data_path", output_json_path,
        "--output_dir", model_output_path,
        "--exp_name", args.exp,
        "--ref_model_path", args.ref_model_path
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
        print(f"Using default base model for policy model")
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
            executable="/bin/bash"
        )
        
        end_time = time.time()
        duration = (end_time - start_time) / 60
        print(f"DPO training script finished successfully in {duration:.2f} minutes.")
        print(f"Trained model saved to: {model_output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running DPO training script: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during DPO training: {e}")
        raise

if __name__ == '__main__':
    main()
