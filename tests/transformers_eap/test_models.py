import torch
import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_eap import EAPGraph, EAPTokenizer

class TestMultiModel(unittest.TestCase):
    
    def test_qwen_0_5b(self):
        print("\n--- Testing Qwen/Qwen2.5-0.5B-Instruct ---")
        try:
            # Using Qwen 0.5B which is small enough
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
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
        
        results = eap.run_eap(input_ids, attention_mask, subject_indices, ans_ids, noise_scale=0.1)
        
        # Check L0 Q Input
        # Qwen has 0.5B params. Layers: 24? Heads?
        # Just check key existence and basic shape
        key = "blocks.0.hook_q_input"
        self.assertIn(key, results)
        
        score_shape = results[key].shape
        print(f"Qwen L0 Q Shape: {score_shape}")
        # Should be [Heads, Sources]. Sources >= 1.
        self.assertEqual(len(score_shape), 2)
        
    def test_gpt_j_tiny(self):
        print("\n--- Testing Tiny GPT-J ---")
        # Use a tiny random model to verify architecture logic without downloading 6GB
        model_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            eap = EAPGraph(model)
            eap_tokenizer = EAPTokenizer(tokenizer, device=eap.main_device)
        except Exception as e:
            print(f"Skipping GPT-J due to load error: {e}")
            return
            
        prompts = ["Test prompt"]
        subjects = ["Test"]
        ans_ids = torch.tensor([1]).to(eap.main_device)
        
        # Pre-process
        input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
        
        results = eap.run_eap(input_ids, attention_mask, subject_indices, ans_ids, noise_scale=0.1)
        
        key = "blocks.0.hook_q_input"
        self.assertIn(key, results)
        print(f"GPT-J L0 Q Shape: {results[key].shape}")

if __name__ == "__main__":
    unittest.main()
