import torch
import numpy as np
from typing import Tuple


def split_edges(edge_index: torch.Tensor, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    """Разделение ребер на train/val/test"""
    num_edges = edge_index.size(1)
    indices = torch.randperm(num_edges)

    train_size = int(ratios[0] * num_edges)
    val_size = int(ratios[1] * num_edges)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return train_idx, val_idx, test_idx


def negative_sampling(edge_index: torch.Tensor, num_nodes: int, num_neg_samples: int):
    """Генерация negative samples"""
    # Упрощенная реализация negative sampling
    pos_edges = set([(src.item(), dst.item()) for src, dst in zip(edge_index[0], edge_index[1])])

    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (1,))
        dst = torch.randint(0, num_nodes, (1,))

        if (src.item(), dst.item()) not in pos_edges and src != dst:
            neg_edges.append([src.item(), dst.item()])

    return torch.tensor(neg_edges, dtype=torch.long).t()


def prepare_edge_labels(data, num_neg_samples=None):
    """Подготовка меток для ребер с negative sampling"""
    if num_neg_samples is None:
        num_neg_samples = data['author', 'duplicate', 'author'].edge_index.size(1)

    pos_edge_index = data['author', 'duplicate', 'author'].edge_index
    neg_edge_index = negative_sampling(pos_edge_index, data['author'].num_nodes, num_neg_samples)

    # Объединяем положительные и отрицательные примеры
    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1))
    ])

    # Перемешиваем
    perm = torch.randperm(edge_index.size(1))
    edge_index = edge_index[:, perm]
    edge_label = edge_label[perm]

    return edge_index, edge_label