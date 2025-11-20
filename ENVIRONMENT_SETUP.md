# MCCE Environment Setup Guide

MCCE requires two separate conda environments to run different functional modules.

## 📦 Two Environments Overview

### 1. moorl Environment - Main Optimization Environment
**Purpose**: Run MCCE multi-objective optimization algorithms

**Main Functions**:
- Execute genetic/evolutionary algorithms
- Call LLMs to generate candidate solutions
- Evaluate and select optimal solutions
- Record optimization history

**Usage**:
```bash
conda activate moorl
python main.py problem/molecules/config.yaml
```

### 2. verl Environment - DPO Training Environment
**Purpose**: Execute Direct Preference Optimization (DPO) training

**Main Functions**:
- Generate training data from optimization history
- Fine-tune local Qwen models
- Update model weights
- Experiment tracking and logging

**Usage**:
- Automatic trigger: Called automatically when moorl environment is running
- Manual training:
```bash
conda activate verl
python tools/makerldata_dpov3.py --exp my_exp --pkl_path results/xxx.pkl
```

## 🔧 Installation

### Method 1: Using Requirements Files (Recommended)

#### 1.1 Create moorl Environment
```bash
# Create environment
conda create -n moorl python=3.10

# Activate environment
conda activate moorl

# Install dependencies
pip install -r requirements_moorl.txt

# For molecular optimization, install rdkit
conda install -c conda-forge rdkit
```

#### 1.2 Create verl Environment
```bash
# Create environment
conda create -n verl python=3.10

# Activate environment
conda activate verl

# Install dependencies
pip install -r requirements_verl.txt

# Install PyTorch (according to your CUDA version)
# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Method 2: Clone from Existing Environments

If you already have configured environments:

```bash
# Export existing environments
conda activate moorl
conda env export > environment_moorl.yml

conda activate verl
conda env export > environment_verl.yml

# Create environments on new machine
conda env create -f environment_moorl.yml
conda env create -f environment_verl.yml
```

## 📋 Dependencies Overview

### moorl Environment Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| transformers | >=4.30.0 | HuggingFace models |
| pymoo | >=0.6.0 | Multi-objective optimization |
| pygmo | >=2.19.0 | Global optimization algorithms |
| openai | >=1.0.0 | OpenAI API |
| google-generativeai | >=0.8.0 | Gemini API |
| pytdc | >=0.3.0 | Molecular databases |

### verl Environment Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| transformers | >=4.30.0 | HuggingFace models |
| trl | >=0.7.0 | DPO training core |
| accelerate | >=1.0.0 | Distributed training |
| swanlab | >=0.3.0 | Experiment tracking |
| flash-attn | >=2.0.0 | Attention acceleration |
| peft | >=0.10.0 | Parameter-efficient fine-tuning |

## 🔍 Installation Verification

### Verify moorl Environment
```bash
conda activate moorl

# Check Python version
python --version  # Should be 3.10.x

# Check core packages
python -c "import torch; print('torch:', torch.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import pymoo; print('pymoo OK')"
python -c "from algorithm.MOO import MOO; print('MCCE import OK')"

# Test MCCE imports
python -c "from model.MOLLM import MOLLM; print('MOLLM import OK')"
```

### Verify verl Environment
```bash
conda activate verl

# Check Python version
python --version  # Should be 3.10.x

# Check core packages
python -c "import torch; print('torch:', torch.__version__)"
python -c "import trl; print('trl:', trl.__version__)"
python -c "import swanlab; print('swanlab:', swanlab.__version__)"

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
```

## 🚀 Quick Start

### 1. Run Basic Optimization (moorl environment)
```bash
conda activate moorl
python main.py problem/circle_packing/config.yaml
```

### 2. Run Optimization with DPO Training (automatic environment switching)
```bash
# Run main program in moorl environment
conda activate moorl
python main.py problem/molecules/config.yaml

# DPO training will automatically execute in verl environment
# No need to manually switch environments
```

### 3. Manual DPO Training (verl environment)
```bash
conda activate verl
python tools/makerldata_dpov3.py \
  --exp my_experiment \
  --pkl_path results/your_result.pkl \
  --ref_model_path /path/to/Qwen2.5-7B-Instruct
```

## ⚠️ Common Issues

### Q1: Why do we need two environments?
A: The two environments may have conflicting package versions. Separating them avoids issues:
- moorl: Requires various optimization algorithm libraries
- verl: Requires latest training libraries (trl, peft, etc.)

### Q2: Can we merge into one environment?
A: Technically possible, but not recommended:
- More dependencies, more complex installation
- Increased risk of version conflicts
- Harder to maintain

### Q3: How to run on machines without GPU?
A: moorl environment can run on CPU (slower), but verl environment strongly recommends GPU for DPO training.

### Q4: What if Flash Attention installation fails?
A: Flash Attention requires specific CUDA version and compilation environment:
```bash
# If installation fails, you can skip
# Program will automatically fall back to standard attention mechanism
pip install flash-attn --no-build-isolation
```

### Q5: How to update environments?
```bash
# Update moorl
conda activate moorl
pip install -r requirements_moorl.txt --upgrade

# Update verl
conda activate verl
pip install -r requirements_verl.txt --upgrade
```

## 📝 Environment Export and Migration

### Export Complete Environment Configuration
```bash
# moorl environment
conda activate moorl
pip freeze > requirements_moorl_full.txt
conda env export > environment_moorl.yml

# verl environment
conda activate verl
pip freeze > requirements_verl_full.txt
conda env export > environment_verl.yml
```

### Rebuild Environment on New Machine
```bash
# Using conda yml files (recommended, includes all dependencies)
conda env create -f environment_moorl.yml
conda env create -f environment_verl.yml

# Or using pip requirements (more universal)
conda create -n moorl python=3.10
conda activate moorl
pip install -r requirements_moorl.txt

conda create -n verl python=3.10
conda activate verl
pip install -r requirements_verl.txt
```

## 💾 Disk Space Requirements

- moorl environment: ~5-8 GB
- verl environment: ~8-12 GB
- Total: ~15-20 GB

Recommend reserving at least 30GB disk space for environments and model storage.

## 📊 Environment Comparison

| Feature | moorl Environment | verl Environment |
|---------|------------------|------------------|
| **Main Purpose** | Optimization algorithm execution | DPO model training |
| **Python Version** | 3.10 | 3.10 |
| **Core Libraries** | pymoo, pygmo | trl, peft |
| **GPU Requirement** | Optional | Strongly Recommended |
| **Disk Space** | ~5-8 GB | ~8-12 GB |
| **Run Frequency** | Continuous | Periodic trigger |
| **Key Packages** | ~100 | ~80 |

## 🔧 Maintenance Recommendations

### Regular Updates
```bash
# Check for updates monthly
conda activate moorl
pip list --outdated

conda activate verl
pip list --outdated
```

### Environment Backup
```bash
# Backup environment configuration (recommended after major updates)
conda activate moorl
conda env export > backup/environment_moorl_$(date +%Y%m%d).yml

conda activate verl
conda env export > backup/environment_verl_$(date +%Y%m%d).yml
```

### Dependency Locking
```bash
# Export exact versions (for production environments)
conda activate moorl
pip freeze > requirements_moorl_locked.txt

conda activate verl
pip freeze > requirements_verl_locked.txt
```

## ⚠️ Important Notes

1. **Two environments are not interchangeable**
   - moorl runs main program
   - verl only for DPO training
   - Do not attempt to merge environments

2. **Version Compatibility**
   - PyTorch version must match CUDA version
   - Flash Attention requires specific CUDA environment
   - Some packages may have platform-specific dependencies

3. **GPU Requirements**
   - moorl: CPU-runnable, GPU recommended
   - verl: Strongly recommend GPU (at least 24GB VRAM)

4. **Disk Space**
   - Reserve at least 30GB space
   - Includes environments, models, data

## 📝 Quick Command Reference

```bash
# Create moorl environment
conda create -n moorl python=3.10 && conda activate moorl && pip install -r requirements_moorl.txt

# Create verl environment
conda create -n verl python=3.10 && conda activate verl && pip install -r requirements_verl.txt

# Verify moorl
conda activate moorl && python -c "from algorithm.MOO import MOO; print('✓')"

# Verify verl
conda activate verl && python -c "import trl; import swanlab; print('✓')"

# Run MCCE
conda activate moorl && python main.py problem/molecules/config.yaml
```

## 🔗 Related Resources

- PyTorch Installation: https://pytorch.org/get-started/locally/
- HuggingFace Documentation: https://huggingface.co/docs
- TRL Documentation: https://huggingface.co/docs/trl
- MCCE Project: https://github.com/your-repo/MCCE

---

**Last Updated**: 2025-11-20  
**MCCE Version**: 1.0.0

