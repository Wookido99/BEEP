import torch

def get_edge_gt_from_node_imp(node_imp, edge_index):
    node_gt = torch.max(node_imp, dim=1, keepdim=True)[0]  # shape: [N, 1]
    node_gt_mask = (node_gt == 1).view(-1)  # shape: [N]

    edge_gt = torch.zeros(edge_index.shape[1], 1)

    source_nodes = edge_index[0, :]
    target_nodes = edge_index[1, :]
    edge_gt_mask = node_gt_mask[source_nodes] & node_gt_mask[target_nodes]

    edge_gt[edge_gt_mask] = 1

    return edge_gt
