"""
HPO Encoders for PhenoDP

This module contains different HPO encoding implementations:
- PCL_HPOEncoder: Contrastive learning-based HPO encoder
- PSD_HPOEncoder: Graph-based HPO encoder with denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import dgl

# Other Libraries
from pyhpo.ontology import Ontology


class PCL_HPOEncoder(nn.Module):
    """Contrastive Learning-based HPO Encoder"""
    
    def __init__(self, input_dim=256, num_heads=8, num_layers=3, hidden_dim=512, dropout=0.1, output_dim=1, max_seq_length=128):
        super(PCL_HPOEncoder, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(p=dropout)
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


class GCN(nn.Module):
    """Graph Convolutional Network for PSD_HPOEncoder"""
    
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(h_feats, out_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        z = F.relu(h)
        h = self.conv2(g, z)
        return h, z


class PSD_HPOEncoder:
    """Pre-trained Semantic Denoising HPO Encoder"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def compute_kernel_bias(vecs, n_components):
        vecs = np.concatenate(vecs, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(s**0.5))
        W = np.linalg.inv(W.T)
        W = W[:, :n_components]
        return W, -mu

    @staticmethod
    def transform_and_normalize(vecs, kernel, bias):
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    @staticmethod
    def normalize(vecs):
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    @staticmethod
    def get_average_embedding(text, tokenizer, T5model):
        input_ids = tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = T5model.encoder(input_ids=input_ids)
            embeddings = outputs.last_hidden_state
        embeddings = embeddings[0].numpy()
        average_embedding = np.mean(embeddings, axis=0)
        return average_embedding
        
    @staticmethod
    def get_vec(hp, tokenizer, T5model):
        obj = Ontology.get_hpo_object(hp)
        return PSD_HPOEncoder.get_average_embedding(obj.name, tokenizer, T5model)
        
    @staticmethod
    def nx_to_dgl(nx_graph):
        dgl_graph = dgl.from_networkx(nx_graph)
        dgl_graph = dgl.add_self_loop(dgl_graph)
        features = [nx_graph.nodes[node]['feature'] for node in nx_graph.nodes()]
        dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)
        return dgl_graph

    @staticmethod
    def mask_graph(dgl_graph, node_mask_percentage=0.2, edge_mask_percentage=0.2):
        num_nodes = dgl_graph.num_nodes()
        num_edges = dgl_graph.num_edges()
        
        mask_num_nodes = int(num_nodes * node_mask_percentage)
        mask_node_indices = random.sample(range(num_nodes), mask_num_nodes)
        original_node_features = dgl_graph.ndata['feat'].clone()
        dgl_graph.ndata['feat'][mask_node_indices] = 0  # 掩码特征为0
        
        mask_num_edges = int(num_edges * edge_mask_percentage)
        mask_edge_indices = random.sample(range(num_edges), mask_num_edges)
        original_edges = dgl_graph.edges()[0].clone(), dgl_graph.edges()[1].clone()
        dgl_graph.remove_edges(mask_edge_indices)
        
        return mask_node_indices, original_node_features, original_edges, mask_edge_indices

    @staticmethod
    def train_model(model, graph, epochs=2000, lr=0.001, node_mask_percentage=0.2, edge_mask_percentage=0.2):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        model.train()
        for epoch in range(epochs):
            mask_node_indices, original_node_features, original_edges, mask_edge_indices = PSD_HPOEncoder.mask_graph(
                graph, node_mask_percentage, edge_mask_percentage
            )
            optimizer.zero_grad()
            features = graph.ndata['feat']
            outputs, latent = model(graph, features)
            loss = loss_fn(outputs[mask_node_indices], original_node_features[mask_node_indices])
            loss.backward()
            optimizer.step()
                
            graph.ndata['feat'] = original_node_features
            graph.add_edges(original_edges[0][mask_edge_indices], original_edges[1][mask_edge_indices])
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')


# Utility functions for PCL_HPOEncoder
def info_nce_loss(embeddings1, embeddings2, temperature=0.1):
    """InfoNCE loss for contrastive learning"""
    N = embeddings1.shape[0]
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)  
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    labels = torch.arange(N).repeat(2).to(embeddings.device)
    mask = torch.eye(2 * N, dtype=torch.bool).to(embeddings.device)
    sim_matrix = sim_matrix[~mask].view(2 * N, 2 * N - 1)
    loss = F.cross_entropy(sim_matrix / temperature, labels)  
    return loss


def get_hps_vec(hps, node_embedding):
    """Get HPO term vectors from embeddings"""
    v_list = []
    for hp in hps:
        v_list.append(node_embedding[hp])
    v_list = np.vstack(v_list)
    return v_list


def get_hpos(d, disease_dict, p=0.5):
    """Get random subset of HPO terms for a disease"""
    hps = disease_dict[d]
    n = int(len(hps) * p)
    s_hps = random.sample(hps, n)
    return s_hps


def pad_or_truncate(vec, max_len=128):
    """Pad or truncate vector to fixed length"""
    original_length = vec.size(0)
    if original_length > max_len:
        padded_vec = vec[:max_len]
    else:
        pad_size = max_len - original_length
        padded_vec = F.pad(vec, (0, 0, 0, pad_size))
    attention_mask = torch.zeros(max_len, dtype=torch.bool)
    attention_mask[:min(original_length, max_len)] = True
    return padded_vec, attention_mask


def get_training_sample(disease_db, disease_dict, node_embedding, n=2000):
    """Generate training samples for contrastive learning"""
    select_disease = random.sample(disease_db, n)
    v1_list = []
    v2_list = []
    mask1_list = []
    mask2_list = []
    for d in select_disease:
        d_hps = get_hpos(d, disease_dict, p=0.7)
        vec = torch.tensor(get_hps_vec(d_hps, node_embedding))
        vec, mask = pad_or_truncate(vec)
        mask = torch.tensor([True] + list(mask))
        v1_list.append(vec)
        mask1_list.append(mask)
        
        d_hps = get_hpos(d, disease_dict, p=0.7)
        vec = torch.tensor(get_hps_vec(d_hps, node_embedding))
        vec, mask = pad_or_truncate(vec)
        mask = torch.tensor([True] + list(mask))
        v2_list.append(vec)
        mask2_list.append(mask)
        
    v1_tensor = torch.stack(v1_list)
    v2_tensor = torch.stack(v2_list)
    mask1_tensor = torch.stack(mask1_list)
    mask2_tensor = torch.stack(mask2_list)
                       
    return [v1_tensor, v2_tensor], [mask1_tensor, mask2_tensor] 