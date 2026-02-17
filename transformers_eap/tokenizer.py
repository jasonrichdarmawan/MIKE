import torch

class EAPTokenizer:
    def __init__(self, tokenizer):
        """
        Wrapper for tokenizer to handle EAP-specific formatting (padding side)
        and subject identification.
        """
        self.tokenizer = tokenizer
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_inputs(self, prompts, subjects=None):
        """
        Tokenizes prompts and identify subject token indices.
        Returns: (input_ids, attention_mask, subject_indices)
        """
        # Enforce left padding for correct next-token prediction
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        
        self.tokenizer.padding_side = old_padding_side
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        if subjects is None:
            return input_ids, attention_mask, None

        subject_indices = []
        for i, (p, s) in enumerate(zip(prompts, subjects)):
            p_ids = input_ids[i]
            # Naive subject search strategy
            s_ids = self.tokenizer(" " + s.strip(), add_special_tokens=False)["input_ids"]
            s_ids_t = torch.tensor(s_ids)
            found = False
            for j in range(len(p_ids) - len(s_ids_t) + 1):
                if torch.equal(p_ids[j:j+len(s_ids_t)], s_ids_t):
                    subject_indices.append(list(range(j, j+len(s_ids_t))))
                    found = True
                    break
            if not found: 
                # This print might annoy users if running large batch, maybe use logging or silent?
                # Kept print for consistency with previous code.
                print(f"Warning: Subject '{s}' not found in prompt. Using empty mask.")
                subject_indices.append([])
                
        return input_ids, attention_mask, subject_indices
