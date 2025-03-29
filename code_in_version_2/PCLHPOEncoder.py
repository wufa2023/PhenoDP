import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import numpy as np

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

def info_nce_loss(embeddings1, embeddings2, temperature=0.1):
    N = embeddings1.shape[0]
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)  
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    labels = torch.arange(N).repeat(2).to(embeddings.device)
    mask = torch.eye(2 * N, dtype=torch.bool).to(embeddings.device)
    sim_matrix = sim_matrix[~mask].view(2 * N, 2 * N - 1)
    loss = F.cross_entropy(sim_matrix / temperature, labels)
    
    return loss

def get_hps_vec(hps, node_embedding):
    v_list = []
    for hp in hps:
        v_list.append(node_embedding[hp])
    v_list = np.vstack(v_list)
                 
    return v_list

def get_hpos(d, disease_dict, p=0.5):
    hps = disease_dict[d]
    n = int(len(hps) * p)
    s_hps = random.sample(hps, n)
    
    return s_hps

def pad_or_truncate(vec, max_len=128):
    original_length = vec.size(0)
    if original_length > max_len:
        padded_vec = vec[:max_len]
    else:
        pad_size = max_len - original_length
        padded_vec = F.pad(vec, (0, 0, 0, pad_size))
    attention_mask = torch.zeros(max_len, dtype=torch.long)
    attention_mask[:min(original_length, max_len)] = 1
    
    return padded_vec, attention_mask

def get_training_sample(disease_db, disease_dict, node_embedding, n=2000):
    select_disease = random.sample(disease_db, n)
    v1_list = []
    v2_list = []
    mask1_list = []
    mask2_list = []
    for d in select_disease:
        d_hps = get_hpos(d,disease_dict, p=0.7)
        vec = torch.tensor(get_hps_vec(d_hps, node_embedding))
        vec, mask = pad_or_truncate(vec)
        mask = torch.tensor([1] + list(mask))
        v1_list.append(vec)
        mask1_list.append(mask)
        
        d_hps = get_hpos(d, disease_dict, p=0.7)
        vec = torch.tensor(get_hps_vec(d_hps, node_embedding))
        vec, mask = pad_or_truncate(vec)
        mask = torch.tensor([1] + list(mask))
        v2_list.append(vec)
        mask2_list.append(mask)
        
    v1_tensor = torch.stack(v1_list)
    v2_tensor = torch.stack(v2_list)
    mask1_tensor = torch.stack(mask1_list)
    mask2_tensor = torch.stack(mask2_list)
                       
    return [v1_tensor, v2_tensor], [mask1_tensor, mask2_tensor]
                       
def info_nce_loss(embeddings1, embeddings2, temperature=0.1):
    N = embeddings1.shape[0]
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)  
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    labels = torch.arange(N).repeat(2).to(embeddings.device)
    mask = torch.eye(2 * N, dtype=torch.bool).to(embeddings.device)
    sim_matrix = sim_matrix[~mask].view(2 * N, 2 * N - 1)
    loss = F.cross_entropy(sim_matrix / temperature, labels)
                       
    return loss