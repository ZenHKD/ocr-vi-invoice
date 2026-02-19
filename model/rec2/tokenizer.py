import torch

class Tokenizer:
    def __init__(self, charset):
        """
        Args:
            charset: List of characters in vocabulary
        """
        # 0: Blank, 1: PAD
        self.blank = '[BLANK]'
        self.pad = '[PAD]'
        self.specials = [self.blank, self.pad]
        
        self.blank_id = 0
        self.pad_id = 1
            
        self.charset = list(sorted(list(set(charset))))
        self.token_to_id = {self.blank: self.blank_id, self.pad: self.pad_id}
        self.token_to_id.update({t: i + len(self.specials) for i, t in enumerate(self.charset)})
        
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.num_classes = len(self.token_to_id)

    def encode(self, texts, device='cpu'):
        """
        Encode text to token IDs
        
        Args:
            texts: List of strings
            device: Target device
        
        Returns:
            Tensor of token IDs
        """
        batch_ids = []
        
        # No BOS/EOS, just characters
        for text in texts:
            ids = [self.token_to_id[c] for c in text if c in self.token_to_id]
            batch_ids.append(ids)
        
        # Pad sequences
        max_len = max([len(ids) for ids in batch_ids]) if batch_ids else 1
        padded_ids = []
        for ids in batch_ids:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.pad_id] * (max_len - len(ids))
            padded_ids.append(ids)
        
        return torch.tensor(padded_ids, dtype=torch.long, device=device)


    def decode(self, token_ids):
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of ints or tensor
        
        Returns:
            List of decoded strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        texts = []
        for ids in token_ids:
            text = []
            for i in ids:
                # Skip special tokens
                if i == self.blank_id or i == self.pad_id:
                    continue
                
                if i in self.id_to_token:
                    text.append(self.id_to_token[i])
            texts.append("".join(text))
        return texts

