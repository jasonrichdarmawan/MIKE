import torch
import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_eap import EAPGraph, EAPTokenizer
from transformers_eap.eap_graph import get_nll_metric

class TestSourceCounts(unittest.TestCase):

    def test_counts(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        eap = EAPGraph(model)
        eap_tokenizer = EAPTokenizer(tokenizer)
        
        # Dummy data
        prompts = ["Test subject"]
        subjects = ["subject"]
        ans_ids = torch.tensor([1]).to(eap.main_device)
        
        # Pre-process
        input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
        
        # Run
        # Define Metric
        nll_metric = get_nll_metric(ans_ids)
             
        results = eap.run_eap(
             input_ids, 
             attention_mask, 
             metric_fn=nll_metric,
             subject_indices=subject_indices
        )
        
        # Checks
        print("\n--- Verifying Logic ---")
        
        # L0 Q Input
        # Expected: 1 Source (ResidPre)
        # Shape: [12, 1]
        res_l0_q = results['blocks.0.hook_q_input']
        print(f"L0 Q: {res_l0_q.shape}. Expected: [12, 1]")
        self.assertEqual(res_l0_q.shape, (12, 1))

        # L0 MLP Input
        # Expected: 13 Sources (ResidPre + 12 Heads)
        res_l0_mlp = results['blocks.0.hook_mlp_in']
        print(f"L0 MLP: {res_l0_mlp.shape}. Expected: [13]")
        self.assertEqual(res_l0_mlp.shape, (13,))
        
        # L1 Q Input
        # Expected: 14 Sources (13 + 1 MLP_0)
        res_l1_q = results['blocks.1.hook_q_input']
        print(f"L1 Q: {res_l1_q.shape}. Expected: [12, 14]")
        self.assertEqual(res_l1_q.shape, (12, 14))

        # L1 MLP Input
        # Expected: 26 Sources (14 + 12 Heads)
        res_l1_mlp = results['blocks.1.hook_mlp_in']
        print(f"L1 MLP: {res_l1_mlp.shape}. Expected: [26]")
        self.assertEqual(res_l1_mlp.shape, (26,))
        
        # L10 MLP In (11th Layer)
        # L0..L9 = 10 layers. Each layer adds 13 sources.
        # Total accumulated entering L10 Q = 1 + 10*13 = 131.
        # MLP adds L10 Heads (12).
        # Total = 131 + 12 = 143.
        res_l10_mlp = results['blocks.10.hook_mlp_in']
        print(f"L10 MLP: {res_l10_mlp.shape}. Expected: [143]")
        self.assertEqual(res_l10_mlp.shape, (143,))

        # L11 MLP In (12th Layer)
        # 1 + 11*13 = 144. + 12 Heads = 156.
        res_l11_mlp = results['blocks.11.hook_mlp_in']
        print(f"L11 MLP: {res_l11_mlp.shape}. Expected: [156]")
        self.assertEqual(res_l11_mlp.shape, (156,))

        # L11 Resid Post
        # Adds L11 MLP (1). 156 + 1 = 157.
        res_l11_resid = results['blocks.11.hook_resid_post']
        print(f"L11 Resid: {res_l11_resid.shape}. Expected: [157]")
        self.assertEqual(res_l11_resid.shape, (157,))

        print("ALL TESTS PASSED")

if __name__ == "__main__":
    unittest.main()
