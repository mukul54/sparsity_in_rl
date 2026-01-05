import argparse
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

def load_model(model_path, dtype, device_map, cache_dir):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir
    )

def compute_statistics_incremental(sft_model, rl_model, tolerances):
    rl_state_dict = rl_model.state_dict()
    sft_state_dict = sft_model.state_dict()
    
    missing_in_rl = [name for name in sft_state_dict if name not in rl_state_dict]
    missing_in_sft = [name for name in rl_state_dict if name not in sft_state_dict]
    
    if missing_in_rl or missing_in_sft:
        print("Missing in RL:", missing_in_rl)
        print("Missing in SFT:", missing_in_sft)
    
    # Initialize counters
    total_params = 0
    num_zeros = 0
    num_close_to_zero = {tol: 0 for tol in tolerances}
    num_nonzero_dict = {}
    
    with torch.no_grad():
        for name, param_sft in tqdm(sft_model.named_parameters(), desc="Computing statistics"):
            delta = rl_state_dict[name] - sft_state_dict[name]
            
            # Move to CPU to save GPU memory
            delta_cpu = delta.cpu()
            
            # Update statistics
            total_params += delta_cpu.numel()
            num_zeros += (delta_cpu == 0).sum().item()
            
            # Check tolerance levels
            zero_tensor = torch.zeros_like(delta_cpu)
            for tol in tolerances:
                num_close_to_zero[tol] += torch.isclose(
                    delta_cpu, zero_tensor, atol=tol
                ).sum().item()
            
            # Store per-parameter sparsity
            num_nonzero_dict[name] = (delta_cpu != 0).sum().item() / delta_cpu.numel()
            
            # Free memory
            del delta, delta_cpu
    
    return total_params, num_zeros, num_close_to_zero, num_nonzero_dict

def compute_layerwise_sparsity(num_nonzero_dict):
    layerwise_sparsity = {}
    for key in num_nonzero_dict:
        if key.startswith("model.layers"):
            layer = key.split(".")[2]
            layerwise_sparsity.setdefault(layer, []).append(num_nonzero_dict[key])
    
    # Average sparsity per layer
    layerwise_sparsity = {
        layer: sum(vals) / len(vals)
        for layer, vals in layerwise_sparsity.items()
    }
    
    # Print
    print("\nLayer-wise sparsity:")
    for layer, sparsity in sorted(layerwise_sparsity.items(), key=lambda x: int(x[0])):
        print(f"Layer {layer}: {sparsity:.4f}")
    
    return layerwise_sparsity

def main(args):
    # Load models
    dtype = getattr(torch, args.torch_dtype)
    model_sft = load_model(args.sft_model, dtype, args.device_map, args.cache_dir)
    model_rl = load_model(args.rl_model, dtype, args.device_map, args.cache_dir)
    
    # Compute statistics incrementally
    total_params, num_zeros, num_close_to_zero, num_nonzero_dict = compute_statistics_incremental(
        model_sft, model_rl, args.tolerances
    )
    
    print(f"\nTotal parameters: {total_params:,}")
    pct_zeros = num_zeros / total_params
    print(f"Percentage of 0 values in the task vector: {pct_zeros:.4f}")
    
    for tol in args.tolerances:
        fraction_close_to_zero = num_close_to_zero[tol] / total_params
        print(f"Tolerance = {tol:.0e} -> Fraction close to zero: {fraction_close_to_zero:.4f}")
    
    # Compute layer-wise sparsity
    compute_layerwise_sparsity(num_nonzero_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SFT and RL model parameter deltas.")
    parser.add_argument("--sft_model", type=str, required=True,
                        help="Path or HF hub name for the SFT model")
    parser.add_argument("--rl_model", type=str, required=True,
                        help="Path or HF hub name for the RL model")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache downloaded models")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        help="Torch dtype, e.g., bfloat16 or float32")
    parser.add_argument("--device_map", type=str, default="cpu",
                        help="Device map for model loading (default: cpu)")
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-5],
                        help="List of tolerances to use for zero-check")
    
    args = parser.parse_args()
    main(args)