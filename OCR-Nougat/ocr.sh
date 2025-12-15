export HF_ENDPOINT=https://hf-mirror.com
# Single-GPU, Single Batch
# python nougat.py ./distributions_redacted.pdf > distributions.md
# Multi-GPU, Large Batch
accelerate launch --num_processes 2 nougat_multi_gpu.py "distributions_redacted.pdf" --out distributions.mmd --batch 40 --scale 5.0