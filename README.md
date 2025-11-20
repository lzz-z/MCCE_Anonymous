# MCCE: A FRAMEWORK FOR MULTI-LLM COLLABORATIVE CO-EVOLUTION

MCCE is a multi-objective optimization framework powered by Large Language Models (LLMs), supporting molecular optimization, Multi-Objective Traveling Salesman Problem (MOTSP), Multi-Objective Capacitated Vehicle Routing Problem (MOCVRP), and circle packing problems.

## Key Features

- **Model Collaboration**: Supports collaboration between API models (GPT, Claude, Gemini) and local Qwen models
- **DPO Training**: Integrated Direct Preference Optimization (DPO) training with automatic data generation and model fine-tuning
- **Multi-Problem Support**: Molecular optimization, MOTSP, MOCVRP, and circle packing
- **Self-Contained**: All dependencies and data files are included within the project, no external path dependencies

## Project Structure

```
MCCE/
├── algorithm/          # Core algorithm implementation
│   ├── MOO.py         # Multi-objective optimization algorithm
│   ├── base.py        # Base class definitions
│   └── PromptTemplate.py  # Prompt templates
├── model/             # Model implementations
│   ├── MOLLM.py       # Main model class
│   ├── LLM.py         # LLM interface
│   └── util.py        # Utility functions
├── problem/           # Problem definitions
│   ├── molecules/     # Molecular optimization
│   ├── motsp/         # Multi-Objective TSP
│   ├── mocvrp/        # Multi-Objective CVRP
│   └── circlepacking/ # Circle packing
├── tools/             # Data generation tools
│   ├── makerldata_dpov3.py           # Molecular DPO data
│   ├── makerldata_motsp_embed.py     # MOTSP DPO data
│   ├── makerldata_mocvrp_embed.py    # MOCVRP DPO data
│   └── makerldata_circle_embed.py    # Circle packing DPO data
├── training/          # Training scripts
│   └── train_dpo.py   # DPO training implementation
├── data/              # Data directory
│   ├── problems/      # Problem data files
│   ├── dpo_training/  # DPO training data (auto-generated)
│   └── dpo_models/    # DPO trained models (auto-generated)
├── oracle/            # Molecular evaluation data
├── eval.py            # Evaluation module
└── main.py            # Main entry point
```

## Environment Setup

This project requires two conda environments:

### 1. moorl Environment (Main Execution)

```bash
conda create -n moorl python=3.10
conda activate moorl
pip install -r requirements_moorl.txt

# For molecular optimization, install rdkit:
conda install -c conda-forge rdkit
```

### 2. verl Environment (DPO Training)

```bash
conda create -n verl python=3.10
conda activate verl
pip install -r requirements_verl.txt


## Quick Start

### Run Molecular Optimization

```bash
conda activate moorl
python main.py problem/molecules/config.yaml
```

### Run MOTSP

```bash
conda activate moorl
python main.py problem/motsp/config.yaml
```

### Run MOCVRP

```bash
conda activate moorl
python main.py problem/mocvrp/config.yaml
```

### Run Circle Packing

```bash
conda activate moorl
python main.py problem/circlepacking/config.yaml
```

## Configuration

Each problem has its own configuration file with key parameters:

- `max_generation`: Maximum number of iterations
- `pop_size`: Population size
- `model_collaboration`: Enable model collaboration
- `use_dpo`: Enable DPO training
- `model_name`: API model name (e.g., `gemini-2.5-flash-nothinking`)
- `local_model_path`: Local Qwen model path

## Custom Optimization Problems

To define a new optimization problem, create the following files:

1. **`config.yaml`** - Algorithm parameter configuration
2. **`{problem}.yaml`** - Problem description and objective definitions
3. **`evaluator.py`** - Evaluation function implementation

Refer to the example files in each problem directory for detailed tutorials.

## DPO Training

MCCE automatically:
1. Collects optimization data (chosen/rejected sample pairs)
2. Generates DPO training datasets
3. Launches DPO training (using verl environment)
4. Updates model weights

Training data and models are saved in `data/dpo_training/` and `data/dpo_models/` directories.


## Validation

### Validate moorl Environment
```bash
conda activate moorl
python -c "from algorithm.MOO import MOO; print('✓ moorl environment OK')"
python -c "from model.MOLLM import MOLLM; print('✓ MOLLM import OK')"
```

### Validate verl Environment
```bash
conda activate verl
python -c "import trl; import swanlab; print('✓ verl environment OK')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## System Requirements

- **Python**: 3.10
- **GPU**: A800 40G *8