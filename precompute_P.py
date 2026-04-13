"""Precompute the null-space projection matrix P for all 32 layers.

Optimized: batch multiple layers per forward pass over Wikipedia to minimize
redundant dataset iterations. ~8 layers per pass to stay within 40GB VRAM.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/kim/CaKE"))
sys.path.insert(0, os.path.expanduser("~/kim/CaKE/EasyEdit"))

import torch, gc, time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from EasyEdit.easyeditor.models.alphaedit_circuit import AlphaEditCircuitHyperParams
from EasyEdit.easyeditor.models.rome.layer_stats import layer_stats

MODEL = "meta-llama/Meta-Llama-3-8B"
HPARAMS = "EasyEdit/hparams/AlphaEdit_Circuit/llama3-8b.yaml"
LAYERS_PER_PASS = 6  # 6 layers per pass — ~1.7GB/layer, ~10GB total + 16GB model, safe on 40GB

hparams = AlphaEditCircuitHyperParams.from_hparams(HPARAMS)

model_name_short = MODEL.rsplit("/")[-1]
size_suffix = "" if hparams.mom2_n_samples is None else f"_{hparams.mom2_n_samples}"
if hparams.mom2_batch_tokens is not None:
    size_suffix = f"_t{hparams.mom2_batch_tokens}" + size_suffix

stats_subdir = Path(hparams.stats_dir) / model_name_short / f"{hparams.mom2_dataset}_stats"
stats_subdir.mkdir(parents=True, exist_ok=True)

P_filepath = stats_subdir / f"null_space_project_{hparams.mom2_dtype}_{size_suffix}.pt"
print(f"P will be saved to: {P_filepath}")

# ── Find uncached layers ──────────────────────────────────────────────────
all_layer_names = [hparams.rewrite_module_tmp.format(i) for i in range(hparams.num_hidden_layers)]
uncached = []
for i, ln in enumerate(all_layer_names):
    fname = stats_subdir / f"{ln}_{hparams.mom2_dtype}_mom2{size_suffix}.npz"
    if not fname.exists():
        uncached.append(i)

print(f"Total layers: {hparams.num_hidden_layers}, cached: {hparams.num_hidden_layers - len(uncached)}, to compute: {len(uncached)}")

if uncached:
    print(f"\nLoading model {MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # ── Compute stats in batched passes ───────────────────────────────────
    batches = [uncached[i:i+LAYERS_PER_PASS] for i in range(0, len(uncached), LAYERS_PER_PASS)]
    print(f"Computing covariance stats in {len(batches)} passes ({LAYERS_PER_PASS} layers/pass)...")

    for batch_idx, layer_indices in enumerate(batches):
        layer_names_batch = [all_layer_names[i] for i in layer_indices]
        print(f"\n--- Pass {batch_idx+1}/{len(batches)}: layers {layer_indices} ---")
        t0 = time.time()
        layer_stats(
            model, tokenizer,
            layer_names_batch,
            hparams.stats_dir,
            hparams.mom2_dataset,
            to_collect=["mom2"],
            sample_size=hparams.mom2_n_samples,
            precision=hparams.mom2_dtype,
            batch_tokens=hparams.mom2_batch_tokens,
            hparams=hparams,
        )
        elapsed = time.time() - t0
        print(f"Pass {batch_idx+1} done in {elapsed/60:.1f} min")
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll covariance stats computed.")

    # ── Compute projection matrices ───────────────────────────────────────
    print(f"\nComputing projection matrices for {hparams.num_hidden_layers} layers ...")
    from EasyEdit.easyeditor.models.alphaedit_circuit.AlphaEdit_Circuit_main import get_project

    P = [None for _ in range(hparams.num_hidden_layers)]
    for layer in tqdm(range(hparams.num_hidden_layers), desc="Projection matrices"):
        P[layer] = get_project(model, tokenizer, layer, hparams).to("cpu")

    print(f"Saving to {P_filepath} ...")
    torch.save(P, P_filepath)
    print("Done.")

else:
    # All stats cached, just compute P
    print(f"\nAll stats cached. Loading model for eigendecomposition only ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    from EasyEdit.easyeditor.models.alphaedit_circuit.AlphaEdit_Circuit_main import get_project

    P = [None for _ in range(hparams.num_hidden_layers)]
    for layer in tqdm(range(hparams.num_hidden_layers), desc="Projection matrices"):
        P[layer] = get_project(model, tokenizer, layer, hparams).to("cpu")

    print(f"Saving to {P_filepath} ...")
    torch.save(P, P_filepath)
    print("Done.")
