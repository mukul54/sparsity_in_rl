PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python check_sparsity.py \
    --sft_model deepseek-ai/deepseek-math-7b-instruct \
    --rl_model deepseek-ai/deepseek-math-7b-rl \
    --torch_dtype bfloat16 \
    --device_map auto \
    --cache_dir ./model_cache \
    --tolerances 1e-5 1e-4 1e-3