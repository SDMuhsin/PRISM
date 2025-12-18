#!/bin/bash
# ============================================================================
# SINQ Comprehensive Experiment Suite - SLURM Submission Script
# Cluster: fir
# ============================================================================
#
# This script submits all benchmark configurations:
#   - Techniques: FP16, SINQ, SparseGPT, Wanda, PRISM
#   - Models: qwen-0.5b (small), opt-1.3b (medium), gemma-2b (large), llama-7b (xlarge)
#   - Precisions: 3, 4, 5, 6, 7, 8 bits
#   - Sparsities: 5%, 25%, 50% (for sparse methods)
#   - Datasets: wikitext2, ptb, c4, mmlu
#
# Usage:
#   chmod +x sbatch/run_all_experiments.sh
#   ./sbatch/run_all_experiments.sh
#
# ============================================================================

# Note: Don't use 'set -e' - it breaks ((job_count++)) when count is 0

# ============================================================================
# CONFIGURATION
# ============================================================================

# Models - one representative per size category
all_models=("qwen-0.5b" "opt-1.3b" "gemma-2b" "llama-7b")

# Techniques
sparse_techniques=("sparsegpt" "wanda" "prism")
all_techniques=("fp16" "sinq" "sparsegpt" "wanda" "prism")

# Precisions (3-8 bits)
precisions=("3" "4" "5")

# Sparsity levels (5%, 25%, 50%)
sparsities=("0.05" "0.25" "0.50")

# Datasets
all_datasets=("wikitext2" "ptb" "c4" "mmlu")

# Job counter
job_count=0

# Working directory

# Create directories
mkdir -p ./logs ./results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_config() {
    # Returns: gpu_type mem time_limit
    # Tailored to model size and dataset complexity
    local model=$1
    local dataset=$2

    # Base times per model (for PPL datasets like wikitext2)
    # MMLU takes ~2x longer due to multiple choice evaluation
    local base_time_min=0
    local gpu_type=""
    local mem=""

    case $model in
        "qwen-0.5b")
            # ~0.5B params, very fast
            gpu_type="nvidia_h100_80gb_hbm3_2g.20gb:1"
            mem="16000M"
            base_time_min=20
            ;;
        "opt-1.3b")
            # ~1.3B params
            gpu_type="nvidia_h100_80gb_hbm3_2g.20gb:1"
            mem="20000M"
            base_time_min=35
            ;;
        "gemma-2b")
            # ~2B params
            gpu_type="nvidia_h100_80gb_hbm3_4g.40gb:1"
            mem="32000M"
            base_time_min=50
            ;;
        "llama-7b")
            # ~7B params, needs full GPU
            gpu_type="nvidia_h100_80gb_hbm3:1"
            mem="64000M"
            base_time_min=120
            ;;
        *)
            # Default fallback
            gpu_type="nvidia_h100_80gb_hbm3_4g.40gb:1"
            mem="32000M"
            base_time_min=60
            ;;
    esac

    # Adjust time based on dataset
    local time_min=$base_time_min
    case $dataset in
        "wikitext2"|"ptb")
            # Standard PPL evaluation
            time_min=$base_time_min
            ;;
        "c4")
            # C4 is larger, ~1.5x time
            time_min=$((base_time_min * 3 / 2))
            ;;
        "mmlu")
            # MMLU is slow (many samples), ~2.5x time
            time_min=$((base_time_min * 5 / 2))
            ;;
    esac

    # Add 20% buffer and round up to nearest 5 minutes
    time_min=$((time_min * 12 / 10))
    time_min=$(( ((time_min + 4) / 5) * 5 ))

    # Cap minimum at 15 min, maximum at 6 hours
    if [ $time_min -lt 15 ]; then
        time_min=15
    elif [ $time_min -gt 360 ]; then
        time_min=360
    fi

    # Convert to HH:MM:SS
    local hours=$((time_min / 60))
    local mins=$((time_min % 60))
    local time_limit=$(printf "%02d:%02d:00" $hours $mins)

    echo "$gpu_type $mem $time_limit"
}

submit_job() {
    local model=$1
    local technique=$2
    local precision=$3
    local sparsity=$4
    local dataset=$5

    # Get tailored configuration
    read gpu_type mem time_limit <<< $(get_job_config $model $dataset)

    # Build job name
    if [[ -n "$sparsity" ]]; then
        sparsity_pct=$(echo "$sparsity * 100" | bc | cut -d'.' -f1)
        job_name="${model}_${technique}_${precision}b_sp${sparsity_pct}_${dataset}"
    else
        job_name="${model}_${technique}_${precision}b_${dataset}"
    fi

    log_file="./logs/${job_name}"

    # Build python command
    if [[ -n "$sparsity" ]]; then
        python_cmd="python benchmarks/benchmark_suite.py --model $model --technique $technique --precision $precision --sparsity $sparsity --dataset $dataset --quiet"
    else
        python_cmd="python benchmarks/benchmark_suite.py --model $model --technique $technique --precision $precision --dataset $dataset --quiet"
    fi

    echo "  $job_name [GPU: ${gpu_type%%:*}, Mem: $mem, Time: $time_limit]"

    sbatch \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --gpus=$gpu_type \
        --mem=$mem \
        --time=$time_limit \
        --output=${log_file}-%N-%j.out \
        --error=${log_file}-%N-%j.err \
        --job-name=$job_name \
        --wrap="
            module load scipy-stack cuda cudnn
            module load arrow
            source ./env/bin/activate
            echo '========================================'
            echo 'Job: $job_name'
            echo 'Model: $model'
            echo 'Technique: $technique'
            echo 'Precision: ${precision}-bit'
            echo 'Sparsity: $sparsity'
            echo 'Dataset: $dataset'
            echo 'Started: '\$(date)
            echo '========================================'
            nvidia-smi
            export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
            $python_cmd
            echo '========================================'
            echo 'Finished: '\$(date)
            echo '========================================'
        "

    ((job_count++))
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "============================================================================"
echo "SINQ Comprehensive Experiment Suite"
echo "============================================================================"
echo "Cluster: fir"
echo "Models: ${all_models[*]}"
echo "Techniques: ${all_techniques[*]}"
echo "Precisions: ${precisions[*]}"
echo "Sparsities: ${sparsities[*]} (for sparse methods)"
echo "Datasets: ${all_datasets[*]}"
echo "============================================================================"
echo ""

# ============================================================================
# SECTION 1: FP16 BASELINE JOBS
# ============================================================================
echo "=============================================="
echo "Section 1: FP16 Baseline Jobs"
echo "=============================================="

for model in "${all_models[@]}"; do
    for dataset in "${all_datasets[@]}"; do
        submit_job "$model" "fp16" "16" "" "$dataset"
    done
done

# ============================================================================
# SECTION 2: SINQ JOBS (Quantization Only - No Sparsity)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 2: SINQ Jobs (Quantization Only)"
echo "=============================================="

for model in "${all_models[@]}"; do
    for precision in "${precisions[@]}"; do
        for dataset in "${all_datasets[@]}"; do
            submit_job "$model" "sinq" "$precision" "" "$dataset"
        done
    done
done

# ============================================================================
# SECTION 3: SPARSEGPT JOBS (Sparse + Quantization)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 3: SparseGPT Jobs"
echo "=============================================="

for model in "${all_models[@]}"; do
    for precision in "${precisions[@]}"; do
        for sparsity in "${sparsities[@]}"; do
            for dataset in "${all_datasets[@]}"; do
                submit_job "$model" "sparsegpt" "$precision" "$sparsity" "$dataset"
            done
        done
    done
done

# ============================================================================
# SECTION 4: WANDA JOBS (Sparse + Quantization)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 4: Wanda Jobs"
echo "=============================================="

for model in "${all_models[@]}"; do
    for precision in "${precisions[@]}"; do
        for sparsity in "${sparsities[@]}"; do
            for dataset in "${all_datasets[@]}"; do
                submit_job "$model" "wanda" "$precision" "$sparsity" "$dataset"
            done
        done
    done
done

# ============================================================================
# SECTION 5: PRISM JOBS (Sparse-Aware Pruning + Quantization)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 5: PRISM Jobs (Sparse-Aware)"
echo "=============================================="

for model in "${all_models[@]}"; do
    for precision in "${precisions[@]}"; do
        for sparsity in "${sparsities[@]}"; do
            for dataset in "${all_datasets[@]}"; do
                submit_job "$model" "prism" "$precision" "$sparsity" "$dataset"
            done
        done
    done
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================================"
echo "Total jobs submitted: $job_count"
echo ""
echo "Job breakdown:"
n_models=${#all_models[@]}
n_prec=${#precisions[@]}
n_sparse=${#sparsities[@]}
n_data=${#all_datasets[@]}
echo "  - FP16:      $n_models models x $n_data datasets = $((n_models * n_data)) jobs"
echo "  - SINQ:      $n_models models x $n_prec precisions x $n_data datasets = $((n_models * n_prec * n_data)) jobs"
echo "  - SparseGPT: $n_models models x $n_prec prec x $n_sparse sparsities x $n_data datasets = $((n_models * n_prec * n_sparse * n_data)) jobs"
echo "  - Wanda:     $n_models models x $n_prec prec x $n_sparse sparsities x $n_data datasets = $((n_models * n_prec * n_sparse * n_data)) jobs"
echo "  - PRISM:     $n_models models x $n_prec prec x $n_sparse sparsities x $n_data datasets = $((n_models * n_prec * n_sparse * n_data)) jobs"
echo ""
echo "Resource allocation by model:"
echo "  - qwen-0.5b: 20GB MIG slice, 16GB mem"
echo "  - opt-1.3b:  20GB MIG slice, 20GB mem"
echo "  - gemma-2b:  40GB MIG slice, 32GB mem"
echo "  - llama-7b:  Full 80GB H100, 64GB mem"
echo ""
echo "Logs: ./logs/"
echo "Results: ./results/"
echo "============================================================================"
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ./logs/<job_name>-*.out"
echo "============================================================================"
