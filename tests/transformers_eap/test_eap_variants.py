import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_eap.eap_graph import EAPGraph, find_peak_mlp_hubs, find_top_edges, get_nll_metric, get_logit_diff_metric
from transformers_eap.tokenizer import EAPTokenizer

def test_eap_with_noise():
    print("Testing EAP with Gaussian Noise...")
    model_id = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"Skipping test due to model loading error: {e}")
        return

    eap = EAPGraph(model)
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
    ans_tokens = [" Croatia", " Croatia", " Croatia", " 3rd", " 2nd", " 4th"]
    
    input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
    ans_ids = [tokenizer(a)["input_ids"][0] for a in ans_tokens]
    ans_ids = torch.tensor(ans_ids).to(eap.main_device)


    # Run Standard EAP
    nll_metric = get_nll_metric(ans_ids)

    results_std = eap.run_eap(
        input_ids, 
        attention_mask, 
        metric_fn=nll_metric,
        subject_indices=subject_indices,
        noise_scale=0.1, 
        integrated_gradients=None
    )
    assert len(results_std) > 0
    print("Standard EAP run successful.")
    find_peak_mlp_hubs(results_std)
    find_top_edges(results_std, n=5)

    # Run EAP-IG
    results_ig = eap.run_eap(
        input_ids, 
        attention_mask, 
        metric_fn=nll_metric,
        subject_indices=subject_indices,
        noise_scale=0.1, 
        integrated_gradients=5
    )
    assert len(results_ig) > 0
    print("EAP-IG run successful.")
    find_peak_mlp_hubs(results_ig)
    find_top_edges(results_ig, n=5)

def test_eap_with_corrupted_prompt():
    print("\nTesting EAP with Corrupted Prompt...")
    model_id = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as e:
        print(f"Skipping test due to model loading error: {e}")
        return

    eap = EAPGraph(model)
    tokenizer.pad_token = tokenizer.eos_token
    
    prompts_clean = [
        "Suppose John lives in a houseboat, Ellie Kemper lives in an apartment, and Keanu Reeves lives in a cabin. Therefore, the person who lives in an apartment is a citizen of",
    ]
    prompts_corrupt = [
        "Suppose John lives in a houseboat, Keanu Reeves lives in an apartment, and Ellie Kemper lives in a cabin. Therefore, the person who lives in an apartment is a citizen of",
    ]
    
    # Simple tokenization
    clean_encoding = tokenizer(prompts_clean, return_tensors="pt", padding=True, truncation=True)
    corrupt_encoding = tokenizer(prompts_corrupt, return_tensors="pt", padding=True, truncation=True)
    
    input_ids = clean_encoding["input_ids"]
    attention_mask = clean_encoding["attention_mask"]
    corrupted_input_ids = corrupt_encoding["input_ids"]
    
    # Ensure lengths match for element-wise subtraction assumption in current implementation
    if input_ids.shape != corrupted_input_ids.shape:
        print("Adjusting prompts to match length for simple test...")
        # Just truncation or padding logic would go here, for now rely on similar prompt structure
        min_len = min(input_ids.shape[1], corrupted_input_ids.shape[1])
        input_ids = input_ids[:, :min_len]
        attention_mask = attention_mask[:, :min_len]
        corrupted_input_ids = corrupted_input_ids[:, :min_len]

    clean_ans_tokens = [" United States"]
    ans_ids = [tokenizer(a)["input_ids"][0] for a in clean_ans_tokens]
    ans_ids = torch.tensor(ans_ids).to(eap.main_device)

    # Run EAP with explicit corrupted input (IG=None)
    nll_metric = get_nll_metric(ans_ids)

    results_std = eap.run_eap(
        input_ids, 
        attention_mask, 
        metric_fn=nll_metric,
        corrupted_input_ids=corrupted_input_ids,
        integrated_gradients=None
    )
    assert len(results_std) > 0
    print("Standard EAP with Corrupted Prompt run successful.")
    find_peak_mlp_hubs(results_std)
    find_top_edges(results_std, n=5)

    # Run EAP-IG with explicit corrupted input
    results_ig = eap.run_eap(
        input_ids, 
        attention_mask, 
        subject_indices=None, 
        metric_fn=nll_metric,
        corrupted_input_ids=corrupted_input_ids,
        integrated_gradients=5
    )
    assert len(results_ig) > 0
    print("EAP-IG with Corrupted Prompt run successful.")
    find_peak_mlp_hubs(results_ig)
    find_top_edges(results_ig, n=5)

def test_eap_with_logit_diff():
    print("\nTesting EAP with Logit Difference (Target - Foil)...")
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    eap = EAPGraph(model)
    eap_tokenizer = EAPTokenizer(tokenizer)

    # Prompt: "The capital of France is" -> Target " Paris", Foil " London"
    prompts = ["The capital of France is"]
    subjects = ["France"]
    
    target_token = " Paris"
    foil_token = " London"
    
    ans_id = tokenizer(target_token)["input_ids"][0]
    foil_id = tokenizer(foil_token)["input_ids"][0]
    
    ans_ids = torch.tensor([ans_id], device=eap.main_device)
    foil_ids = torch.tensor([foil_id], device=eap.main_device)
    
    input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
    
    # Run EAP with Logit Diff
    logit_diff_metric = get_logit_diff_metric(ans_ids, foil_ids)

    results = eap.run_eap(
        input_ids, 
        attention_mask, 
        subject_indices=subject_indices, 
        metric_fn=logit_diff_metric,
        noise_scale=0.1
    )
    
    assert len(results) > 0
    print("EAP with Logit Difference run successful.")
    find_peak_mlp_hubs(results)
    find_top_edges(results, n=5)

def test_eap_logit_diff_with_corrupted_prompt():
    print("\nTesting EAP Logit Difference (Target - Foil) with Explicit Corrupted Prompt...")
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    eap = EAPGraph(model)
    eap_tokenizer = EAPTokenizer(tokenizer)

    # Clean: "The capital of France is" -> Target " Paris", Foil " London"
    prompts = ["The capital of France is"]
    subjects = ["France"]
    # Corrupt: "The capital of Italy is" -> This should naturally lower prob of Paris and increase Rome (but we foil with London)
    # Actually, a better corrupted prompt for "Paris vs London" might be "The capital of England is"
    corrupted_prompts = ["The capital of England is"]
    
    target_token = " Paris"
    foil_token = " London"
    
    ans_id = tokenizer(target_token)["input_ids"][0]
    foil_id = tokenizer(foil_token)["input_ids"][0]
    
    ans_ids = torch.tensor([ans_id], device=eap.main_device)
    foil_ids = torch.tensor([foil_id], device=eap.main_device)
    
    input_ids, attention_mask, subject_indices = eap_tokenizer.tokenize_inputs(prompts, subjects)
    corrupted_input_ids, _, _ = eap_tokenizer.tokenize_inputs(corrupted_prompts, ["England"])

    # Define metric
    logit_diff_metric = get_logit_diff_metric(ans_ids, foil_ids)

    # Run EAP
    results = eap.run_eap(
        input_ids, 
        attention_mask, 
        subject_indices=None, 
        metric_fn=logit_diff_metric,
        corrupted_input_ids=corrupted_input_ids,
        integrated_gradients=None
    )
    
    assert len(results) > 0
    print("EAP Logit Diff + Corrupted Prompt successful.")
    find_peak_mlp_hubs(results)
    find_top_edges(results, n=5)


if __name__ == "__main__":
    test_eap_with_noise()
    test_eap_with_corrupted_prompt()
    test_eap_with_logit_diff()
    test_eap_logit_diff_with_corrupted_prompt()
