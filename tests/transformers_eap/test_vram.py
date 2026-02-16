import gc
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_eap import EAPGraph, EAPTokenizer


class TestVRAMUsage(unittest.TestCase):
    def _load_tiny_model(self):
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
        return tokenizer, model

    def _build_inputs(self, tokenizer, device):
        eap_tokenizer = EAPTokenizer(tokenizer)
        prompts = [
            "Suppose John lives in a houseboat, Ellie Kemper lives in an apartment, and Jane lives in a cabin. Therefore, the person who lives in an apartment is a citizen of",
            "If Alex resides in a tent, Ellie Kemper resides in a condo, and Chris resides in a mansion, then the individual residing in a condo is a citizen of",
            "Given that Sam occupies a treehouse, Ellie Kemper occupies a duplex, and Pat occupies a farmhouse, the person occupying the duplex is a citizen of",
            "In a list of some countries, Brazil is ranked 5th, Croatia is ranked 3rd, and Japan is ranked 1st. The rank of the country that Ellie Kemper is a citizen of is",
            "If in a global sports competition, France secures the 8th position, Croatia claims the 2nd, and Australia is at the 6th position, the position Ellie Kemper's home country holds is",
            "During an international economic summit, Canada is placed 7th, Croatia is 4th, and South Korea is 10th in terms of economic growth. The placement of the country where Ellie Kemper resides as a citizen is"
        ]
        subjects = ["Ellie Kemper"] * len(prompts)
        ans_ids = torch.tensor([1] * len(prompts), device=device)
        input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
        return input_ids, attention_mask, subject_indices, ans_ids

    def test_run_eap_records_peak_vram(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for VRAM tests")

        try:
            tokenizer, model = self._load_tiny_model()
        except Exception as exc:
            self.skipTest(f"Skipping VRAM test due to model load error: {exc}")

        eap = EAPGraph(model)
        device = torch.device(eap.main_device)

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        input_ids, attention_mask, subject_indices, ans_ids = self._build_inputs(tokenizer, device)
        results = eap.run_eap(input_ids, attention_mask, subject_indices, ans_ids, noise_scale=0.1)

        torch.cuda.synchronize(device)
        peak_alloc = torch.cuda.max_memory_allocated(device)
        self.assertGreater(peak_alloc, 0)
        self.assertIn("blocks.0.hook_q_input", results)

    def test_run_eap_peak_vram_stable_across_repeated_runs(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for VRAM tests")

        try:
            tokenizer, model = self._load_tiny_model()
        except Exception as exc:
            self.skipTest(f"Skipping VRAM test due to model load error: {exc}")

        eap = EAPGraph(model)
        device = torch.device(eap.main_device)
        input_ids, attention_mask, subject_indices, ans_ids = self._build_inputs(tokenizer, device)

        def run_once_and_get_peak_bytes():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

            _ = eap.run_eap(input_ids, attention_mask, subject_indices, ans_ids, noise_scale=0.1)

            torch.cuda.synchronize(device)
            return torch.cuda.max_memory_allocated(device)

        first_peak = run_once_and_get_peak_bytes()
        second_peak = run_once_and_get_peak_bytes()

        # Allow modest allocator/runtime variability while catching major regressions.
        tolerance_factor = 1.35
        tolerance_abs = 64 * 1024 * 1024  # 64 MB
        self.assertLessEqual(second_peak, int(first_peak * tolerance_factor) + tolerance_abs)


if __name__ == "__main__":
    unittest.main()
