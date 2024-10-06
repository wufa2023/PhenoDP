import torch
import torch.nn as nn
import torch.nn.functional as F

def nx_to_dgl(nx_graph):
    dgl_graph = dgl.from_networkx(nx_graph)
    dgl_graph = dgl.add_self_loop(dgl_graph)
    features = [nx_graph.nodes[node]['feature'] for node in nx_graph.nodes()]
    dgl_graph.ndata['feat'] = torch.tensor(features, dtype=torch.float32)
    return dgl_graph

dgl_graph = nx_to_dgl(graph)
feature_dimension = 768

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

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(h_feats, out_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        z = F.relu(h)
        h = self.conv2(g, z)
        return h, z

def train_model(model, graph, epochs=2000, lr=0.001, node_mask_percentage=0.2, edge_mask_percentage=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        mask_node_indices, original_node_features, original_edges, mask_edge_indices = mask_graph(graph, node_mask_percentage, edge_mask_percentage)
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

train_model(model, dgl_graph)
model.eval()
with torch.no_grad():
    reconstructed_features, reconstructed_latent = model(dgl_graph, dgl_graph.ndata['feat'])
