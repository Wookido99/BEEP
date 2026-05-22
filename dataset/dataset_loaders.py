import os
import pickle
import numpy as np
from numpy.random import RandomState

def adj_to_edge_index(adj):
    """
    Convert an adjacency matrix to an edge index
    :param adj: Original adjacency matrix
    :return: Edge index representation of the graphs
    """
    converted = []
    for d in adj:
        edge_index = np.argwhere(d > 0.).T
        mask = edge_index[0] != edge_index[1]
        converted.append(edge_index[:, mask])

    return converted

def load_graph_dataset(dataset_name, shuffle=False):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    path = parent_dir + '/data/BA2/raw/' + "BA-2Motif" + '.pkl'
    with open(path, 'rb') as fin:
        adjs, features, labels = pickle.load(fin)

    n_graphs = adjs.shape[0]
    indices = np.arange(n_graphs)
    if shuffle:
        prng = RandomState(42) 
        indices = prng.permutation(indices)


    adjs = adjs[indices]
    features = features[indices].astype('float32')
    labels = labels[indices]

    n_train = int(n_graphs * 0.7)
    n_val = int(n_graphs * 0.85)
    train_mask = np.zeros(n_graphs, dtype=bool)
    val_mask = np.zeros(n_graphs, dtype=bool)
    test_mask = np.zeros(n_graphs, dtype=bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_val] = True
    test_mask[n_val:] = True

    edge_index = adj_to_edge_index(adjs)

    return edge_index, features, labels, train_mask, val_mask, test_mask


def load_dataset(dataset, skip_preprocessing=False, shuffle=False):
    print(f"Loading {dataset} dataset")
    data = load_graph_dataset(dataset, shuffle)
    if skip_preprocessing:
        return data[:-1] 
    return data