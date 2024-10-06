import torch
import torch.nn as nn
import torch.nn.functional as F

class PCL_HPOEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim=1, max_seq_length=128):
        super(PCL_HPOEncoder, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim

    def forward(self, vec, mask):
        batch_size, seq_length, _ = vec.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        vec = torch.cat((cls_tokens, vec), dim=1)
        vec = vec.permute(1, 0, 2)
        vec = self.transformer_encoder(vec, src_key_padding_mask=mask)
        cls_embedding = vec[0]
        return cls_embedding, vec


