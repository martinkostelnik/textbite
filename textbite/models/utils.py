from typing import List, Tuple, Set

import torch


def edges_to_edge_indices(edges: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    from_indices = [from_idx for from_idx, _ in edges]
    to_indices = [to_idx for _, to_idx in edges]

    return from_indices, to_indices


def edge_indices_to_edges(from_indices: List[int], to_indices: List[int]) -> List[Tuple[int, int]]:
    assert len(from_indices) == len(to_indices)
    return [(from_index, to_index) for from_index, to_index in zip(from_indices, to_indices)]


def get_transitive_subsets(positive_edges: List[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    positive_subsets = []
    for from_idx, to_idx in positive_edges:
        subset = [(from_idx, to_idx)]
        while True:
            from_indices = [f for f, _ in subset]
            to_indices = [t for _, t in subset]
            left_positive_edges = [(f, t) for f, t in positive_edges if t in from_indices and (f, t) not in subset]
            right_positive_edges = [(f, t) for f, t in positive_edges if f in to_indices and (f, t) not in subset]

            if not left_positive_edges and not right_positive_edges:
                break

            subset.extend(left_positive_edges)
            subset.extend(right_positive_edges)

        subset = set(subset)
        if subset not in positive_subsets:
            positive_subsets.append(set(subset))

    return positive_subsets


def get_similarities(node_features, edge_indices):
    lhs_nodes = torch.index_select(input=node_features, dim=0, index=edge_indices[0, :])
    rhs_nodes = torch.index_select(input=node_features, dim=0, index=edge_indices[1, :])
    similarities = torch.cosine_similarity(lhs_nodes, rhs_nodes, dim=1)
    similarities = torch.sigmoid(similarities)
    return similarities
