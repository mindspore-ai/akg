import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False, head_dim=128, num_tokens={}):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa
        self.head_dim = head_dim
        self.num_tokens = num_tokens

    def forward(self, query, key, value, attn_mask=None):
        start_token = 0
        q_heads = query.shape[-1] // self.head_dim
        kv_heads = key.shape[-1] // self.head_dim

        for num_token in self.num_tokens:
            cur_query = query[start_token:start_token+num_token, :]
            cur_query = cur_query.view(1, num_token, q_heads, self.head_dim)
            cur_query = cur_query.permute(0, 2, 1, 3)  # Transpose to [batch, heads, seq_len, head_dim] for scaled_dot_product_attention
            
            cur_key = key[start_token:start_token+num_token, :]
            cur_key = cur_key.view(1, num_token, kv_heads, self.head_dim)
            cur_key = cur_key.permute(0, 2, 1, 3)  # Transpose to [batch, heads, seq_len, head_dim] for scaled_dot_product_attention

            cur_value = value[start_token:start_token+num_token, :]
            cur_value = cur_value.view(1, num_token, kv_heads, self.head_dim)
            cur_value = cur_value.permute(0, 2, 1, 3)  # Transpose to [batch, heads, seq_len, head_dim] for scaled_dot_product_attention
            
            start_token += num_token
            
            cur_out = torch.nn.functional.scaled_dot_product_attention(
                cur_query, cur_key, cur_value, attn_mask=attn_mask, dropout_p=self.dropout_p,
                is_causal=self.is_causal, enable_gqa=self.enable_gqa
            )

            if start_token == 0:
                out = cur_out
            else:
                out = torch.cat((out, cur_out), dim=1)
        return out


def get_inputs():
    # Using shapes that are representative of large model computations in transformer models
    # Shape (32, 1024) represents:
    # - 32 num_tokens
    # - 512 head dimension * heads
    head_dim = 128
    batch_size = 8
    num_tokens, q_head, kv_head = 1024, 512, 256
    num_tokens = [128, 128, 128, 128, 128, 128, 128, 128]
    
    q_shape = (num_tokens, q_head)
    kv_shape = (num_tokens, kv_head)
    
    query = torch.randn(q_shape, dtype=torch.float16)
    key = torch.randn(kv_shape, dtype=torch.float16)
    value = torch.randn(kv_shape, dtype=torch.float16)
    return [query, key, value]


def get_init_inputs():
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = False  # Not using Grouped-Query Attention
    return [dropout_p, is_causal, enable_gqa]