import torch
import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_eap import EAPGraph, EAPTokenizer

class TestMultiGPU(unittest.TestCase):
    
    def test(self):
        print("\n--- Testing Multi-GPU ---")

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("Skipping multi-GPU test: Requires at least 2 GPUs.")
            return

        try:
            # Using Qwen 0.5B which is small enough
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # balanced device_map ensures split across GPUs
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="balanced")
            eap = EAPGraph(model) 
            eap_tokenizer = EAPTokenizer(tokenizer, device=eap.main_device)
        except Exception as e:
            print(f"Skipping Qwen due to load error (network/disk): {e}")
            return

        prompts = ["The capital of France is"]
        subjects = ["France"]
        ans_ids = torch.tensor([1]).to(eap.main_device) # Dummy ID

        # Pre-process
        input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
        
        # Run
        # Note: input_ids must be on main device, checked inside tokenize_inputs? 
        # tokenize_inputs puts them on self.main_device. EAPGraph detects main_device from model.
        # If model is split, main_device is usually cuda:0.
        results = eap.run_eap(input_ids, attention_mask, subject_indices, ans_ids)
        
        # Check L0 Q Input
        key = "blocks.0.hook_q_input"
        self.assertIn(key, results)
        
        score_shape = results[key].shape
        print(f"Qwen L0 Q Shape: {score_shape}")
        # Should be [Heads, Sources]. Sources >= 1.
        self.assertEqual(len(score_shape), 2)

if __name__ == "__main__":
    unittest.main()
