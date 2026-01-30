import torch
import torch.nn as nn

num_groups = 8
total_units = 128
select_count = 4
sequence_length = 16384


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input_scores, unit_group_mapping, position_indices, group_buffer):
        group_buffer[unit_group_mapping, position_indices] = input_scores.flatten()
        return group_buffer

def get_inputs():
    input_scores = torch.randn([sequence_length, select_count], dtype=torch.float32)
    units_per_group = total_units // num_groups
    flat_indices = torch.randint(total_units, [sequence_length * select_count], dtype=torch.int32)
    unit_group_mapping = (flat_indices // units_per_group).to(torch.int32)
    position_indices = torch.arange(flat_indices.shape[0])
    group_buffer = torch.zeros([num_groups, sequence_length * select_count], dtype=torch.float32)
    return [input_scores, unit_group_mapping, position_indices, group_buffer]

def get_init_inputs():
    return []