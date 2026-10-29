import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM


class QwenCausalWrapper(nn.Module):
    def __init__(self, model: Qwen3ForCausalLM):
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


def build_qwen_model(
    model_name: str,
    vocab_size: int = 50257,
    max_seq_length: int = 2048,
    dropout: float = 0.0,
    initializer_range: float = 0.02,
    tie_word_embeddings: bool = True,
    attn_implementation: str = "flash_attention_2",
):
    model_key = model_name.lower()

    if model_key in {"qwen3-0.6b", "qwen3_0.6b", "qwen3-0.6B".lower()}:
        hidden_size = 1024
        intermediate_size = 3072
        num_hidden_layers = 28
        num_attention_heads = 16
        num_key_value_heads = 8
        rope_theta = 1_000_000.0
        max_position_embeddings = max(40960, max_seq_length)
    else:
        raise ValueError(f"Unsupported qwen model config: {model_name}")

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        attention_dropout=dropout,
        initializer_range=initializer_range,
        tie_word_embeddings=tie_word_embeddings,
        rms_norm_eps=1e-6,
        rope_theta=rope_theta,
        attention_bias=False,
    )
    config._attn_implementation = attn_implementation

    model = Qwen3ForCausalLM(config)
    model.set_attn_implementation(attn_implementation)
    return QwenCausalWrapper(model), config
