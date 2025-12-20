# PRISM: PRuning-Integrated Sparse Matrix Normalization for Joint Pruning and Quantization

**Status**: Research in progress | **Target**: IEEE TNNLS
**Experiments**: Running on Compute Canada HPC (fir cluster)

---

## Overview

PRISM is a novel joint pruning and quantization technique for Large Language Models that introduces **sparse-aware Sinkhorn normalization**. Unlike standard approaches that compute normalization factors on the full weight matrix (including zeros), PRISM recomputes Sinkhorn factors only on the remaining non-zero weights after pruning.

### Key Innovation

Standard Sinkhorn normalization includes pruned (zero) weights in variance calculations, which distorts the scaling factors:

```
Standard: Compute μ₁, μ₂ on full W (including zeros after pruning)
PRISM:    Compute μ₁, μ₂ only on W[mask == 1] (non-zero weights)
```

This simple insight yields significant improvements at moderate-to-high sparsity levels.

### Techniques Compared

| Technique | Type | Description |
|-----------|------|-------------|
| **FP16** | Baseline | Full precision, no compression |
| **SINQ** | Quantization | Sinkhorn-normalized quantization (no sparsity) |
| **Wanda** | Pruning | Magnitude × activation pruning |
| **SparseGPT** | Pruning + Quant | OBS-based pruning with quantization |
| **PRISM** | Pruning + Quant | Sparse-aware Sinkhorn + OBS compensation |

---

## Quick Start

```bash
# Setup
git clone <repo>
cd PRISM
source env/bin/activate

# Run PRISM benchmark
python benchmarks/benchmark_suite.py \
    --model qwen-0.5b \
    --technique prism \
    --precision 4 \
    --sparsity 0.35 \
    --dataset wikitext2
```

---

## Project Structure

```
PRISM/
├── sinq/                    # Core implementation
│   ├── sparse_quant.py      # PRISM + pruning algorithms
│   ├── sinkhorn.py          # Sinkhorn normalization
│   └── dual_shift.py        # Quantization kernels
├── benchmarks/
│   └── benchmark_suite.py   # Main experiment runner
├── sbatch/                  # SLURM job scripts (HPC)
│   ├── run_all_experiments.sh
│   └── run_sparsity_ablation.sh
├── llmdocs/                 # Documentation & research notes
│   └── llm_context.md       # Full context for developers
├── results/                 # Experiment outputs (git-ignored)
├── logs/                    # Job logs (git-ignored)
└── cache/                   # HuggingFace cache (git-ignored)
```

---

## Running Experiments

### Local Testing
```bash
python benchmarks/benchmark_suite.py \
    --model qwen-0.5b \
    --technique prism \
    --precision 4 \
    --sparsity 0.35
```

### HPC Batch Jobs
```bash
# Submit all experiments
./sbatch/run_all_experiments.sh

# Submit sparsity ablation study
./sbatch/run_sparsity_ablation.sh

# Monitor jobs
squeue -u $USER
```

---

## Documentation

For comprehensive project context, research methodology, and coding guidelines, see:

**[llmdocs/llm_context.md](llmdocs/llm_context.md)** - Full developer context

---

## Current Status

- **Experiments**: Running on Compute Canada fir cluster
- **Models**: qwen-0.5b, opt-1.3b, gemma-2b, llama-7b
- **Ablations**: Sparsity sweep 5-95%, Precision 3-8 bit
- **Lead Researcher**: Managing HPC job submissions

---

## Citation

Paper in preparation for IEEE Transactions on Neural Networks and Learning Systems (TNNLS).
