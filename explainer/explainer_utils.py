import torch

def create_edge_embeds(edge_index, node_embeds):
    embeds_0dim = node_embeds[edge_index[0]]  # [E, embedding_size]
    embeds_1dim = node_embeds[edge_index[1]]  # [E, embedding_size]
    edge_embeds = torch.cat([embeds_0dim, embeds_1dim], dim=1)  # [E, embedding_size * 2]

    return edge_embeds


def sample_graph(sampling_weights, device, temperature=1.0, bias=0.0, training=True):
    if training:
        bias = bias + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(device)
        sampling_weights = sampling_weights.to(device)
        gate_inputs = (gate_inputs + sampling_weights) / temperature
        graph = torch.sigmoid(gate_inputs)
    else:
        graph = torch.sigmoid(sampling_weights)
    return graph.squeeze()