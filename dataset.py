import torch


class BilingualDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer_source, tokenizer_target, source_language, target_language, sequence_length):
        """
        Constructor for the BilingualDataset class:
        - dataset: the dataset to use
        - tokenizer_source: the tokenizer for the source language
        - tokenizer_target: the tokenizer for the target language
        - source_language: the source language
        - target_language: the target language
        - sequence_length: the sequence length to use
        """
        super(BilingualDataset, self).__init__()
        self.dataset = dataset
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_language = source_language
        self.target_language = target_language
        self.sequence_length = sequence_length

        self.sos_token = torch.tensor([self.tokenizer_source.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_source.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_source.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Get the length of the dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Get an item from the dataset:
        - Get the source and target texts
        - Encode the source text
        - Decode the target text
        - Pad the encoded and decoded texts
        - Return the encoded and decoded texts
        """
        source_target_pair = self.dataset[index]
        source_text = source_target_pair["translation"][self.source_language]
        target_text = source_target_pair["translation"][self.target_language]

        encoded_inputs_tokens = self.tokenizer_source.encode(source_text).ids
        decoded_inputs_tokens = self.tokenizer_source.encode(target_text).ids

        encoder_num_padding_tokens = self.sequence_length - len(encoded_inputs_tokens) - 2
        decoder_num_padding_tokens = self.sequence_length - len(decoded_inputs_tokens) - 1

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("The sequence length is too short for the given input")
        
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(encoded_inputs_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(decoded_inputs_tokens, dtype=torch.int64), 
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        labels = torch.cat(
            [
                torch.tensor(decoded_inputs_tokens, dtype=torch.int64), 
                self.eos_token, 
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.shape[0] == self.sequence_length
        assert decoder_input.shape[0] == self.sequence_length
        assert labels.shape[0] == self.sequence_length

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_attention_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_attention_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]),
            "labels": labels,
            "source_text": source_text,
            "target_text": target_text,
        }

   
def causal_mask(length):
    """
    Create a causal mask for the decoder attention
    """
    mask = torch.triu(torch.ones(1, length, length), diagonal=1).type(dtype=torch.int)
    return mask == 0