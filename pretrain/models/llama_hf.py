import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


class LlamaCausalWrapper(nn.Module):
    def __init__(self, model: LlamaForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, idx, attn_mask=None):
        outputs = self.model(
            input_ids=idx,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits

    def count_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.model.embed_tokens.weight.numel()
            if self.model.lm_head.weight is not self.model.model.embed_tokens.weight:
                n_params -= self.model.lm_head.weight.numel()
        return n_params


def build_llama_model(
    model_name: str,
    vocab_size: int = 50257,
    max_seq_length: int = 2048,
    dropout: float = 0.0,
    initializer_range: float = 0.02,
    tie_word_embeddings: bool = False,
    attn_implementation: str = "flash_attention_2",
):
    if model_name in {"llama2-20M", "llama-20M"}:
        # Keep the plainLM small-LLaMA style simple: use standard MHA and a 4x
        # FFN, but switch to a wider/shallower 4-layer layout that is closer to
        # common community 20M recipes.
        hidden_size = 256
        num_attention_heads = 8
        num_hidden_layers = 4
        intermediate_size = hidden_size * 4
    elif model_name == "llama-60M":
        hidden_size = 512
        num_attention_heads = 8
        num_hidden_layers = 8
        # Common small-LLaMA research setting seen across multiple papers.
        intermediate_size = 1376
    elif model_name == "llama-kaiyue":
        hidden_size = 512
        num_attention_heads = 8
        num_hidden_layers = 32
        intermediate_size = hidden_size * 4
    elif model_name == "llama-93M":
        hidden_size = 512
        num_attention_heads = 16
        num_hidden_layers = 8
        intermediate_size = hidden_size * 4
    elif model_name == "llama-170M":
        hidden_size = 768
        num_attention_heads = 12
        num_hidden_layers = 8
        intermediate_size = hidden_size * 4
    else:
        raise ValueError(f"Unsupported llama model config: {model_name}")

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_seq_length,
        attention_dropout=dropout,
        initializer_range=initializer_range,
        tie_word_embeddings=tie_word_embeddings,
    )
    config._attn_implementation = attn_implementation

    model = LlamaForCausalLM(config)
    model.set_attn_implementation(attn_implementation)
    return LlamaCausalWrapper(model), config
