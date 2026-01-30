import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, scale='auto'):
        """
        初始化Scaled Dot Product Attention模块
        
        参数:
        scale: 缩放因子，可以是以下值之一:
            - float: 自定义缩放因子
            - None: 不使用缩放（相当于scale=1.0）
        """
        super(Model, self).__init__()
        self.scale = scale

    def forward(self, query, key, value):
        """
        带有可配置缩放因子的Scaled Dot Product Attention实现
        
        参数:
        query: [batch_size, num_heads, seq_len, head_dim]
        key: [batch_size, num_heads, seq_len, head_dim]
        value: [batch_size, num_heads, seq_len, head_dim]
        
        返回:
        output: [batch_size, num_heads, seq_len, head_dim]
        """
        # 1. 确定缩放因子
        if self.scale is None:
            scale_factor = 1.0 / (query.size(-1) ** 0.5)
        else:
            scale_factor = float(self.scale)
        
        # 2. 对query进行缩放
        LOG2E = 1.44269504
        scaled_query = query * scale_factor * LOG2E
        
        # 3. 计算注意力分数 (缩放后的query和key的点积)
        # [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        
        # 4. 应用softmax获取注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 5. 计算输出（注意力权重与value的点积）
        output = torch.matmul(attn_weights, value)
        
        return output

def get_inputs():
    batch, num_heads, seq_len, head_dim = 32, 8, 1024, 64
    shape = (batch, num_heads, seq_len, head_dim)
    
    query = torch.randn(shape, dtype=torch.float16)
    key = torch.randn(shape, dtype=torch.float16)
    value = torch.randn(shape, dtype=torch.float16)
    return [query, key, value]

def get_init_inputs():
    scale = None
    return [scale]