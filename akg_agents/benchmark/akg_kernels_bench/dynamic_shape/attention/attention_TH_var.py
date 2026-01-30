import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dropout_p=0.0, is_causal=False, enable_gqa=False, head_dim=128):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.enable_gqa = enable_gqa
        self.head_dim = head_dim

    def forward(self, query, key, value, attn_mask=None, num_tokens=None):
        start_token = 0
        q_heads = query.shape[-1] // self.head_dim
        kv_heads = key.shape[-1] // self.head_dim

        for i in range(num_tokens.size(0)):
            num_token = num_tokens[i].item()
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


def get_inputs_dyn_list():
    # Case 1: Small tensor size (num_token, 128) (aligned)
    num_tokens1 = torch.randint(1, 10, (8,))
    all_token = torch.sum(num_tokens1, dim=0)
    query1 = torch.randn(all_token, 128, dtype=torch.float16)
    key1 = torch.randn(all_token, 128, dtype=torch.float16)
    value1 = torch.randn(all_token, 128, dtype=torch.float16)
    
    # Case 2: Small tensor size (num_tokens, 256) (aligned)
    num_tokens2 = torch.randint(10, 20, (8,))
    all_token = torch.sum(num_tokens2, dim=0)
    query2 = torch.randn(all_token, 256, dtype=torch.float16)
    key2 = torch.randn(all_token, 256, dtype=torch.float16)
    value2 = torch.randn(all_token, 256, dtype=torch.float16)
    
    # Case 3: Small tensor size (num_tokens, 512) (aligned)
    num_tokens3 = torch.randint(1, 10, (16,))
    all_token = torch.sum(num_tokens3, dim=0)
    query3 = torch.randn(all_token, 512, dtype=torch.float16)
    key3 = torch.randn(all_token, 512, dtype=torch.float16)
    value3 = torch.randn(all_token, 512, dtype=torch.float16)
    
    # Case 4: Medium tensor size (num_tokens, 512) (diff, aligned)
    num_tokens4 = torch.randint(1, 10, (16,))
    all_token = torch.sum(num_tokens4, dim=0)
    query4 = torch.randn(all_token, 512, dtype=torch.float16)
    key4 = torch.randn(all_token, 256, dtype=torch.float16)
    value4 = torch.randn(all_token, 256, dtype=torch.float16)
    
    # Case 5: Medium tensor size (num_tokens, 512) (diff, aligned)
    num_tokens5 = torch.randint(10, 20, (16,))
    all_token = torch.sum(num_tokens5, dim=0)
    query5 = torch.randn(all_token, 512, dtype=torch.float16)
    key5 = torch.randn(all_token, 256, dtype=torch.float16)
    value5 = torch.randn(all_token, 256, dtype=torch.float16)
    
    # Case 6: Extreme tensor size (num_tokens, 512) (aligned)
    num_tokens6 = torch.randint(64, 128, (8,))
    all_token = torch.sum(num_tokens6, dim=0)
    query6 = torch.randn(all_token, 512, dtype=torch.float16)
    key6 = torch.randn(all_token, 512, dtype=torch.float16)
    value6 = torch.randn(all_token, 512, dtype=torch.float16)
    
    return [
        [query1, key1, value1, num_tokens1],
        [query2, key2, value2, num_tokens2],
        [query3, key3, value3, num_tokens3],
        [query4, key4, value4, num_tokens4],
        [query5, key5, value5, num_tokens5],
        [query6, key6, value6, num_tokens6]
    ]


def get_init_inputs():
    # Parameters for scaled_dot_product_attention
    dropout_p = 0.0  # No dropout
    is_causal = False  # Not causal
    enable_gqa = False  # Not using Grouped-Query Attention
    head_dim = 128
    return [dropout_p, is_causal, enable_gqa, head_dim]