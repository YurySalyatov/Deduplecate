import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_node_types, num_edge_types):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in num_node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata=(num_node_types,
                           num_edge_types), heads=4)
            self.convs.append(conv)

        self.out_lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        # Project node features
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        # HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


class DuplicatePredictor(nn.Module):
    def __init__(self, hidden_channels, num_node_types, num_edge_types, num_layers, dropout):
        super().__init__()
        self.encoder = HeteroGNN(hidden_channels, num_layers,
                                 num_node_types, num_edge_types)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Получение эмбеддингов
        embeddings = self.encoder(x_dict, edge_index_dict)

        # Получение эмбеддингов для пар авторов
        author_embeddings = embeddings['author']
        src = author_embeddings[edge_label_index[0]]
        dst = author_embeddings[edge_label_index[1]]

        # Предсказание вероятности дубликата
        pair_embeddings = torch.cat([src, dst], dim=1)
        return self.predictor(pair_embeddings).squeeze()
