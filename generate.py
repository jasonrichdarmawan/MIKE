# %%

if __name__ == "__main__":
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"

# %%

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch
import time

# %%

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
# %%

if __name__ == "__main__":
    past_key_values = DynamicCache()
    messages = [
        [{"role": "user", "content": "Hello, my name is Jeff. what's your name?"}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]
    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", padding=True, return_dict=True).to(model.device)

    generated_ids = inputs.input_ids
    cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
    max_new_tokens = 10

    for _ in range(max_new_tokens):
        outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True) # shpae [batch_size, seq_len, vocab_size]  
        # Greedily sample one next token
        next_token_ids = outputs.logits[:, -1:].argmax(-1) # shape [batch_size, 1]
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1) # shape [batch_size, seq_len+1]

        # Prepare inputs for the next generation step by leaaving unprocessed tokens, in our case we have only one new token
        # and expanding attn mask for the new token, as explained above
        attention_mask = inputs["attention_mask"]
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
        cache_position = cache_position[-1:] + 1 # add one more position for the next token

    for i, generate_text in enumerate(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)):
        print(f"Generated text {i}:\n{generate_text.strip()}")

# %%

if __name__ == "__main__":
    prompts = [
        "Question: Who is the head of state of the country where Fernando Santos hold a citizenship?",
        "Question: Where was the creator of Aslan born?",
        "Question: On which continent was Hari Kunzru born?",
    ]

    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    ).to(model.device)

    start_time = time.time()
    out1 = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        stop_strings=["Retrieved fact:"],
        tokenizer=tokenizer,
        return_dict_in_generate=True,
        use_cache=True,
    )
    end_time = time.time()
    print(f"\n=== stage1 (prefill+first decode) time: {end_time - start_time:.2f}s ===")

    generated_stage1 = tokenizer.batch_decode(
        out1.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    for i, text in enumerate(generated_stage1):
        print(f"\n=== Stage1 sample {i} ===")
        print(text.strip())
        
# %%

if __name__ == "__main__":

    keep_idx = torch.tensor([0, 2], device=model.device)
    retrieved_facts = [
        "\nRetrieved fact: Fernando Santos is a citizen of United Kingdom.\n",
        "\nRetrieved fact: Hari Kunzru was born in London.\n",
    ]

    def select_past_key_values(past_key_values, batch_indices):
        return tuple(
            tuple(state.index_select(0, batch_indices) for state in layer)
            for layer in past_key_values
        )

    def build_prefix_attention_mask(sequences, pad_token_id, trailing_trim_token_ids):
        prefix_mask = (sequences != pad_token_id).long()
        for row in range(sequences.shape[0]):
            last_idx = sequences.shape[1] - 1
            while last_idx >= 0 and prefix_mask[row, last_idx] == 0:
                last_idx -= 1
            while last_idx >= 0 and int(sequences[row, last_idx].item()) in trailing_trim_token_ids:
                prefix_mask[row, last_idx] = 0
                last_idx -= 1
        return prefix_mask

    # out1.past_key_values has shape [num_layers, 2, batch_size, num_key_value_heads, seq_len, head_dim]
    selected_past = select_past_key_values(out1.past_key_values, keep_idx)
    selected_sequences = out1.sequences.index_select(0, keep_idx)

    trailing_trim_token_ids = set()
    if tokenizer.eos_token_id is not None:
        trailing_trim_token_ids.add(tokenizer.eos_token_id)
    for maybe_end_token in ("<|im_end|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(maybe_end_token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            trailing_trim_token_ids.add(token_id)

    tokenizer.padding_side = "left"
    fact_tokens = tokenizer(
        retrieved_facts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    fact_input_ids = fact_tokens["input_ids"]
    fact_attention_mask = fact_tokens["attention_mask"]
    past_length = selected_past[0][0].shape[2]
    cache_position_stage2 = torch.arange(
        past_length,
        past_length + fact_input_ids.shape[1],
        device=model.device,
    )
    prefix_attention_mask = build_prefix_attention_mask(
        selected_sequences,
        tokenizer.pad_token_id,
        trailing_trim_token_ids,
    )
    attention_mask_stage2 = torch.cat([prefix_attention_mask, fact_attention_mask], dim=1)

    start_time = time.time()
    out2 = model.generate(
        input_ids=fact_input_ids,
        attention_mask=attention_mask_stage2,
        past_key_values=selected_past,
        cache_position=cache_position_stage2,
        max_new_tokens=128,
        do_sample=False,
        return_dict_in_generate=True,
        use_cache=True,
    )
    end_time = time.time()
    print(f"\n=== stage2 (append fact + continue) time: {end_time - start_time:.2f}s ===")

    continuation_only_ids = out2.sequences
    continuation_texts = tokenizer.batch_decode(continuation_only_ids, skip_special_tokens=False)
    for source_idx, text in zip(keep_idx.tolist(), continuation_texts):
        print(f"\n=== Stage2 continuation for original sample {source_idx} ===")
        print(text.strip())
    
# %%
