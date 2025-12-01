import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNLayer(nn.Module):
    """A minimal GCN layer using only PyTorch ops."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        if edge_index.numel() == 0:
            return self.linear(x)

        row, col = edge_index
        self_loop = torch.arange(num_nodes, device=device)
        loop_index = torch.stack([self_loop, self_loop], dim=0)
        edge_index = torch.cat([edge_index, loop_index], dim=1)

        values = torch.ones(edge_index.size(1), device=device, dtype=x.dtype)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

        degree = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        norm_values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm_adj = torch.sparse_coo_tensor(edge_index, norm_values, (num_nodes, num_nodes))

        out = torch.sparse.mm(norm_adj, x)
        out = self.linear(out)
        return out


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        last_dim = in_dim
        for layer_idx in range(num_layers):
            next_dim = out_dim if layer_idx == num_layers - 1 else hidden_dim
            layers.append(SimpleGCNLayer(last_dim, next_dim))
            last_dim = next_dim
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = self.activation(h)
        return h


class VectorHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(node_feat)
