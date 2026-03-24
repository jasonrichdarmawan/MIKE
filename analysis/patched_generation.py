import functools

import numpy as np
from numpy import ndarray
import torch
import torch.nn.functional as F
from torch import Tensor
from baukit import Trace, TraceDict

from jaxtyping import Float, Int, Shaped

from utils import get_prepend_space, get_layer_names, get_attention_modules, get_attention_layers_names, \
    get_norm_module, get_mlp_layers_names


def find_tokens(tokenizer, string_tokens: Int[Tensor, "prompt_size"], substring: str, prepend_space: bool, last=True):
    """
    Example:
    - string_tokens: tokenized "the head of state of the country of citizenship of Stephen Harper is"
    - substring: "the country of citizenship of Stephen Harper" / "is"
    - returns: end index of substring tokens in string_tokens i.e. "Harper" / "is" token position index
    """
    if prepend_space:
        substring = " " + substring
    substring_tokens = tokenizer(substring, add_special_tokens=False, return_tensors="pt").input_ids[0]
    substring_tokens = substring_tokens.to(string_tokens.device)
    for start in range(len(string_tokens) - len(substring_tokens) + 1):
        end = start + len(substring_tokens)
        if torch.all(string_tokens[start:end] == substring_tokens):
            if last:
                return end - 1
            else:
                return start, end - 1
    return None


def get_hidden_states(model, tokenizer, entities: list[str], prompts: list[str]) -> Float[Tensor, "num_prompts num_layers hidden_size"]:
    """
    The goal is to extract the hidden representation at the entity's last token position in the prompt at all layers.

    Example:
    - prompt: "the head of state of the country of citizenship of Stephen Harper is"
    - entity: "the country of citizenship of Stephen Harper" / "is"
    - returns: hidden states of "Harper" / "is" token at all layers, shape (num_prompts, num_layers, hidden_size)
    """
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prepend_space = get_prepend_space(model)
    entity_idx = [find_tokens(tokenizer, q, e, prepend_space) for q, e in zip(prompt_inputs.input_ids, entities)]
    entity_hidden_states = [None for _ in range(len(entity_idx))]

    layers = get_layer_names(model)
    with torch.no_grad(), TraceDict(model, layers) as trace:
        prompt_inputs = prompt_inputs.to(model.device)
        model(**prompt_inputs)
        for i, idx in enumerate(entity_idx):
            if idx is not None:
                entity_hidden_states[i] = torch.stack([trace[layer].output[0][i][idx].cpu() for layer in layers])

    return torch.stack(entity_hidden_states)


def get_hidden_states_with_patching(model, tokenizer, entity, prompt, hidden_state, target_layer):
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    prepend_space = get_prepend_space(model)
    entity_idx = [find_tokens(tokenizer, q, e, prepend_space) for q, e in zip(prompt_inputs.input_ids, entity)]
    entity_hidden_states = [None for _ in range(len(entity_idx))]

    def replace_hidden_state_hook(output):
        hs = output[0]
        if hs.shape[1] == 1:  # After first replacement the hidden state is cached
            return output
        hs[entity_idx] = hidden_state.to(hs.device)
        return (hs,) + output[1:]

    layers = get_layer_names(model)
    with torch.no_grad(), Trace(model, layer=target_layer, edit_output=replace_hidden_state_hook), TraceDict(model,
                                                                                                             layers) as trace:
        prompt_inputs = prompt_inputs.to(model.device)
        model(**prompt_inputs)
        for i, idx in enumerate(entity_idx):
            if idx is not None:
                entity_hidden_states[i] = torch.stack([trace[layer].output[0][i][idx].cpu() for layer in layers])

    return torch.stack(entity_hidden_states)


def generate_patching_inputs(model, tokenizer, target_prompt: list[str], target_token_str: str):
    """
    The goal is to tokenize the target_prompt and find the position of target_token_str in target_prompt.

    Example:
    - target_prompt: "Syria: Syria is a country in the Middle East, Leonardo DiCaprio: Leonardo DiCaprio is an American actor, Samsung: Samsung is a South Korean multinational corporation, x"
    - target_token_str: "x"

    Caveat: target_token_str can be in any position in target_prompt.

    returns: tokenized target_prompt, position of target_token in target_prompt. shape: tuple[row_indices, col_indices], length: batch_size
    """
    if get_prepend_space(model):
        if type(target_token_str) is str:
            target_token_str = " " + target_token_str
        else:
            target_token_str = [" " + t for t in target_token_str]
    target_token = tokenizer(target_token_str, return_tensors="pt", add_special_tokens=False,
                             padding=True).input_ids[..., -1] # shape: (batch_size,)
    inputs = tokenizer(target_prompt, return_tensors="pt", padding=True)
    target_position = (inputs.input_ids == target_token.unsqueeze(1)).cumsum(dim=1).argmax(dim=1) # shape: (batch_size,)
    return inputs.to(model.device), (torch.LongTensor(range(target_position.shape[0])), target_position)


# @jaxtyped(typechecker=typechecker)
def decode_generated(tokenizer, generated: Int[Tensor, "batch_size sequence_size"], target_prompt: list[str], sampled=False):
    r"""
    Example:
    - target_prompt: "Syria: Syria is a country in the Middle East, Leonardo DiCaprio: Leonardo DiCaprio is an American actor, Samsung: Samsung is a South Korean multinational corporation, x"
    
    Caveat: replace "\n" with " " in generated text.

    returns: list of generated text after target_prompt
    """
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)
    batch_size = len(target_prompt)
    sample_size = len(text) // len(target_prompt)
    if sampled:
        target_prompt = np.repeat(target_prompt, sample_size)
    pred = [t[len(p):].strip().replace("\n", " ") for t, p in zip(text, target_prompt)]
    if sampled:
        pred = np.split(np.array(pred), batch_size)
    return pred


def generate_with_patching_layer(model, tokenizer, hidden_state: Float[Tensor, "batch_size hidden_size"], target_layer: str, target_prompt: list[str], target_token_str: str,
                                 do_sample: bool):
    inputs, target_position = generate_patching_inputs(model, tokenizer, target_prompt, target_token_str)

    def replace_hidden_state_hook(output):
        hs = output[0]
        if hs.shape[1] == 1:  # After first replacement the hidden state is cached
            return output
        hs[target_position] = hidden_state.to(hs.device)
        return (hs,) + output[1:]

    with torch.no_grad(), Trace(model, layer=target_layer, retain_output=False, edit_output=replace_hidden_state_hook):
        if do_sample:
            generated = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10, do_sample=True)
        else:
            generated = model.generate(**inputs, do_sample=False, temperature=1, top_p=1, num_beams=1,
                                       pad_token_id=tokenizer.eos_token_id, max_new_tokens=10)

    return decode_generated(tokenizer, generated, target_prompt)


def generate_with_patching_same_layers(model, tokenizer, hidden_states, target_prompt, target_token_str, do_sample):
    generations_by_layer = []
    layer_names = get_layer_names(model)
    for source_layer_idx, target_layer in enumerate(layer_names):
        hidden_state = hidden_states[:, source_layer_idx, :]
        generations = generate_with_patching_layer(model, tokenizer, hidden_state, target_layer, target_prompt,
                                                   target_token_str, do_sample)
        generations_by_layer.append(generations)
    return generations_by_layer


def _generate_with_patch_and_cache(
        model, tokenizer, hidden_state: Float[Tensor, "batch_size hidden_size"],
        target_layer: str, x_input_ids: torch.Tensor, full_attention_mask: torch.Tensor,
        prefix_kv, prefix_input_ids: torch.Tensor, target_prompt: list[str], do_sample: bool):
    """
    KV-cache variant of generate_with_patching_layer.
    The prefix KV cache is precomputed once and reused across all source×target
    layer calls — only the single "x" token is re-processed per call (~32x speedup).

    Uses a first_call flag instead of shape check because x_input_ids is already
    1 token, so hs.shape[1] == 1 on both the patched prefill step and all
    subsequent generation steps.
    """
    first_call = [True]

    def replace_hidden_state_hook(output):
        hs = output[0]
        if first_call[0]:
            first_call[0] = False
            hs[:, -1, :] = hidden_state.to(hs.device)
        return (hs,) + output[1:]

    with torch.no_grad(), Trace(model, layer=target_layer, retain_output=False,
                                edit_output=replace_hidden_state_hook):
        if do_sample:
            new_tokens = model.generate(
                input_ids=x_input_ids, attention_mask=full_attention_mask,
                past_key_values=prefix_kv, pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=10, do_sample=True)
        else:
            new_tokens = model.generate(
                input_ids=x_input_ids, attention_mask=full_attention_mask,
                past_key_values=prefix_kv, do_sample=False,
                temperature=1, top_p=1, num_beams=1,
                pad_token_id=tokenizer.eos_token_id, max_new_tokens=10)

    full_generated = torch.cat([prefix_input_ids, new_tokens], dim=1)
    return decode_generated(tokenizer, full_generated, target_prompt)


def generate_with_patching_all_layers(model, tokenizer, hidden_states: Float[Tensor, "batch_size num_layers hidden_size"], target_prompt: list[str], target_token_str: str, do_sample: bool,
                                      use_kv_cache: bool = False):
    """
    The goal is to replace the hidden representation of target_token_str in target_prompt with a hidden_state at a specific target layer with the hidden representation taken from a specific source layer.

    Example:
    - target_prompt: "Syria: Syria is a country in the Middle East, Leonardo DiCaprio: Leonardo DiCaprio is an American actor, Samsung: Samsung is a South Korean multinational corporation, x"
    - target_token_str: "x"

    use_kv_cache: if True, precomputes the prefix KV cache once and reuses it
    across all source×target layer calls (~32x faster prefill). Assumes
    target_token_str is the last token in the prompt (left-padded batch).
    """
    layer_names = get_layer_names(model)
    layer_count = len(layer_names)
    generations = np.ndarray((layer_count, layer_count), dtype=object)

    if use_kv_cache:
        inputs, _ = generate_patching_inputs(model, tokenizer, target_prompt, target_token_str)
        prefix_input_ids  = inputs.input_ids[:, :-1]
        x_input_ids       = inputs.input_ids[:, -1:]
        full_attention_mask = inputs.attention_mask
        with torch.no_grad():
            prefix_kv = model(
                input_ids=prefix_input_ids,
                attention_mask=inputs.attention_mask[:, :-1],
                use_cache=True,
            ).past_key_values
        for source_layer in range(layer_count):
            for target_layer in range(layer_count):
                hidden_state = hidden_states[:, source_layer, :]
                generations[source_layer, target_layer] = _generate_with_patch_and_cache(
                    model, tokenizer, hidden_state, layer_names[target_layer],
                    x_input_ids, full_attention_mask, prefix_kv,
                    prefix_input_ids, target_prompt, do_sample)
    else:
        for source_layer in range(layer_count):
            for target_layer in range(layer_count):
                hidden_state = hidden_states[:, source_layer, :]
                generations[source_layer, target_layer] = generate_with_patching_layer(model,
                                                                                       tokenizer,
                                                                                       hidden_state,
                                                                                       layer_names[target_layer],
                                                                                       target_prompt,
                                                                                       target_token_str,
                                                                                       do_sample)
    return generations


def get_top_k_tokens(tokenizer, scores: Float[Tensor, "num_prompts num_layers vocab_size"], k=10) -> tuple[Shaped[ndarray, "num_prompts num_layers k"], Float[Tensor, "num_prompts num_layers k"]]:
    with torch.no_grad():
        probabilities = F.softmax(scores, dim=-1) # shape (num_prompts, num_layers, vocab_size)
        top_k_probabilities, top_k_indices = torch.topk(probabilities, k, dim=-1) # shape (num_prompts, num_layers, k)
        top_k_words = np.empty_like(top_k_indices.cpu(), dtype=object)
        for b, layers in enumerate(top_k_indices):
            for l, layer in enumerate(layers):
                top_k_words[b][l] = tokenizer.batch_decode(layer.unsqueeze(-1)) # shape (k,)
        return top_k_words, top_k_probabilities


def get_token_ranks(projections: Float[Tensor, "num_layers vocab_size"], tokens: list[int]) -> Shaped[ndarray, "num_layers num_tokens"]:
    """
    The goal is for each layer, to count how many vocabulary entries have a higher score than that token.
    """
    projections = projections.float().cpu()
    vals = projections[:, tokens] # shape (num_layers, num_tokens)
    ranks = np.empty_like(vals, dtype=int)
    for layer in range(projections.shape[0]):
        for i, val in enumerate(vals[layer]):
            ranks[layer][i] = (projections[layer] > val).sum()
    return ranks


def generate_with_attention_knockout(model, tokenizer, layer_idx: int, prompt: list[str], source: list[str], target_token_str: list[str], default_decoding: bool,
                                     k=0, return_probabilities=False):
    """
    Example:
    - prompt: "the head of state of the country of citizenship of Stephen Harper is" / "the country of citizenship of Stephen Harper is" / "the head of state of Canada is"
    - source: "Stephen Harper" / "citizenship"
    - target_token_str: "the country of citizenship of Stephen Harper" / "the head of state of Canada" / "citizenship" / "state"
    """
    inputs, target_position = generate_patching_inputs(model, tokenizer, prompt, target_token_str)
    if type(source) is not str:
        prepend_space = get_prepend_space(model)
        source_position = torch.LongTensor([find_tokens(tokenizer, q, e, prepend_space) for q, e in
                                            zip(inputs.input_ids, source)])

    def wrap_attention_forward(original_forward_func):
        @functools.wraps(original_forward_func)
        def knockout_attention_forward(*args, **kwargs):
            """
            The goal is to mask out attention weights from a source to a target position at a specific layer, causing these attention heads to contribute nothing to the model's prediction.
            """
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for key, v in kwargs.items():
                new_kwargs[key] = v

            if "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            else:
                hidden_states = args[0]
            batch_size = hidden_states.shape[0]
            num_tokens = hidden_states.shape[1]
            attention_weight_size = (batch_size, model.config.num_attention_heads, num_tokens, num_tokens) # shape: (batch_size, num_heads, query_len, key_len)
            prev_attention_mask = kwargs["attention_mask"]
            new_attention_mask = torch.zeros(attention_weight_size, dtype=prev_attention_mask.dtype).to(
                prev_attention_mask.device) + prev_attention_mask
            if source == "all":
                new_attention_mask[target_position[0], :, :, target_position[1]] = torch.finfo(model.dtype).min
            elif source == "last":
                if num_tokens != 1:
                    # Index 2 is Query (Receiver), Index 3 is Key (Sender)
                    # -1 refers to the LAST token in the sequence (the Query)
                    # target_position[1] refers to the token index of target_token_str (the Key)
                    new_attention_mask[target_position[0], :, -1, target_position[1]] = torch.finfo(model.dtype).min
            else:
                if num_tokens != 1:
                    new_attention_mask[target_position[0], :, source_position, target_position[1]] = torch.finfo(
                        model.dtype).min

            new_kwargs["attention_mask"] = new_attention_mask
            return original_forward_func(*new_args, **new_kwargs)

        return knockout_attention_forward

    if layer_idx != -1:
        attention_modules = get_attention_modules(model, layer_idx, k)
        original_forward_funcs = [attention_modules.forward for attention_modules in attention_modules]
        for attention_module in attention_modules:
            attention_module.forward = wrap_attention_forward(attention_module.forward)

    with torch.no_grad():
        if default_decoding:
            generated = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=10,
                                       return_dict_in_generate=return_probabilities, output_scores=return_probabilities)
        else:
            generated = model.generate(**inputs, do_sample=False, temperature=1, top_p=1, num_beams=1,
                                       pad_token_id=tokenizer.eos_token_id, max_new_tokens=10,
                                       return_dict_in_generate=return_probabilities, output_scores=return_probabilities)

    if layer_idx != -1:
        for i, attention_module in enumerate(attention_modules):
            attention_module.forward = original_forward_funcs[i]

    if return_probabilities:
        generations = decode_generated(tokenizer, generated["sequences"], prompt)
        top_k_words, top_k_probabilities = get_top_k_tokens(tokenizer, generated["scores"][0])
        return generations, top_k_words, top_k_probabilities
    else:
        return decode_generated(tokenizer, generated, prompt)


def generate_with_attention_knockout_all_layers(model, tokenizer, prompt: list[str], source: list[str], target_token_str: list[str], default_decoding: bool,
                                                k=0, return_probabilities=False):
    """
    Example:
    - prompt: "the head of state of the country of citizenship of Stephen Harper is" / "the country of citizenship of Stephen Harper is" / "the head of state of Canada is"
    - source: "Stephen Harper" / "citizenship"
    - target_token_str: "the country of citizenship of Stephen Harper" / "the head of state of Canada" / "citizenship" / "state"
    """
    generations_by_layer = {}
    for layer in range(-1, model.config.num_hidden_layers):
        generations_by_layer[layer] = generate_with_attention_knockout(model, tokenizer, layer, prompt, source,
                                                                       target_token_str, default_decoding, k,
                                                                       return_probabilities)
    return generations_by_layer


def get_entity_ranks(model, tokenizer, projections: Float[Tensor, "num_prompts num_layers vocab_size"], entities: list[list[str]]) -> list[Shaped[ndarray, "num_layers num_entities"]]:
    """
    The goal is for each prompt, for each layer, to count how many vocabulary entries have a higher score than that token.

    Example:
    - entities: [["Charles III of the United Kingdom", ...], ...]

    Caveat: rank based on the first token of each entity
    """
    entities = [ent + [e.lower() for e in ent] for ent in entities]
    if get_prepend_space(model):
        entities = [ent + [" " + e for e in ent] for ent in entities]
    tokenized_entity: list[list[list[int]]] = [tokenizer(ent, add_special_tokens=False).input_ids for ent in entities]
    ranks = []
    for proj, tokens in zip(projections, tokenized_entity):
        first_tokens = [t[0] for t in tokens]
        ranks.append(get_token_ranks(proj, first_tokens))
    return ranks


def get_sublayer_projection(model, tokenizer, prompt: list[str], bridge_entities: list[list[str]], answers: list[list[str]], layers: list[str]
                            ) -> tuple[Shaped[ndarray, "num_prompts num_layers k"], 
                                       Float[Tensor, "num_prompts num_layers k"], 
                                       list[Shaped[ndarray, "num_layers num_bridge_entities"]], 
                                       list[Shaped[ndarray, "num_layers num_answers"]], 
                                       list[Shaped[ndarray, "num_layers 1"]]]:
    """
    The goals are:
    - to project the attention / mlp sublayer outputs to the vocabulary space and get the top k tokens.
    - to rank the bridge entities and final answers based on the projections scores at each layer.

    Example:
    - prompt: ["the head of state of the country of citizenship of Stephen Harper is", ...]
    - bridge_entities: [["Canada", ...], ...]
    - answers: [["Charles III of the United Kingdom", ...], ...]
    """
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device) # shape (num_prompts, seq_len)

    sublayer_values = [] # shape (num_prompts, num_layers, hidden_size)
    with torch.no_grad(), TraceDict(model, layers) as trace:
        logits = model(**prompt_inputs).logits
        for i in range(len(prompt)):
            values = [] # shape (num_layers, hidden_size)
            for layer in layers:
                if type(trace[layer].output) is tuple:
                    output = trace[layer].output[0]
                else:
                    output = trace[layer].output
                values.append(output[i][-1].cpu())
            sublayer_values.append(torch.stack(values))
    sublayer_values = torch.stack(sublayer_values) # shape (num_prompts, num_layers, hidden_size)

    out_embeddings = model.get_output_embeddings() # shape (hidden_size, vocab_size)
    norm = get_norm_module(model)
    norm_device = next(norm.parameters()).device
    with torch.no_grad():
        projections = out_embeddings(norm(sublayer_values.to(norm_device))) # shape (num_prompts, num_layers, vocab_size)
    top_k_words, top_k_probabilities = get_top_k_tokens(tokenizer, projections) # shape (num_prompts, num_layers, k), (num_prompts, num_layers, k)

    answer_ranks = get_entity_ranks(model, tokenizer, projections, answers) # list of shape (num_layers, num_answers)
    bridge_entity_ranks = get_entity_ranks(model, tokenizer, projections, bridge_entities) # list of shape (num_layers, num_bridge_entities)

    prediction_ranks = [] # list of shape (num_layers, 1)
    predicted_tokens = logits[:, -1, :].max(dim=-1).indices # shape (num_prompts,)
    for proj, predicted_token in zip(projections, predicted_tokens):
        prediction_ranks.append(get_token_ranks(proj, [predicted_token]))
    prediction_ranks = [r.flatten().tolist() for r in prediction_ranks]

    return top_k_words, top_k_probabilities, bridge_entity_ranks, answer_ranks, prediction_ranks


def get_attention_projection(model, tokenizer, prompt: list[str], bridge_entities: list[list[str]], answers: list[list[str]]):
    """
    Example:
    - prompt: ["the head of state of the country of citizenship of Stephen Harper is", ...]
    - bridge_entities: [["Canada", ...], ...]
    - answers: [["Charles III of the United Kingdom", ...], ...]
    """
    return get_sublayer_projection(model, tokenizer, prompt, bridge_entities, answers,
                                   get_attention_layers_names(model))


def get_mlp_projection(model, tokenizer, prompt: list[str], bridge_entities: list[list[str]], answers: list[list[str]]):
    return get_sublayer_projection(model, tokenizer, prompt, bridge_entities, answers, get_mlp_layers_names(model))
