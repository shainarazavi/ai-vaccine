#!/bin/bash
#SBATCH --job-name=baseline_exp
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gres=gpu:a40:4
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

mkdir -p logs results
module purge
module load pytorch2.1-cuda11.8-python3.10
cd /fs01/projects/aixpert/users/shaina/immunization

EXPERIMENTS=(
    "--output_dir ./results/qwen25_baseline --model_name Qwen/Qwen2.5-7B-Instruct --eval_only"
    "--output_dir ./results/llama31_baseline --model_name meta-llama/Llama-3.1-8B-Instruct --eval_only"
    "--output_dir ./results/gemma2_baseline --model_name google/gemma-2-9b-it --eval_only"
    "--output_dir ./results/phi35_baseline --model_name microsoft/Phi-3.5-mini-instruct --eval_only"
)

echo "Running experiment $SLURM_ARRAY_TASK_ID: ${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
python scripts/train.py --data_dir ./data ${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}