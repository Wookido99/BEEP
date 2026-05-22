import random
import torch

from dataset.MUTAG_dataset import Mutagenicity
from dataset.BA3_dataset import BA3Motif
from dataset.fluoride_carbonyl_dataset import FluorideCarbonyl
from dataset.mnistsp_dataset import MNIST75sp


def _is_positive_graph(graph):
    y = getattr(graph, "y", None)
    if y is None:
        return False
    if torch.is_tensor(y):
        if y.numel() == 0:
            return False
        value = float(y.view(-1)[0].item())
    else:
        try:
            value = float(y)
        except Exception:
            return False
    return int(value) == 1


def _filter_positive_graphs(graphs):
    return [g for g in graphs if _is_positive_graph(g)]


def get_dataset(data_root_path, dataset, **kwargs):
    dataset_name = dataset
    if dataset_name == 'BA3':
        train_dataset = BA3Motif(data_root_path, mode="training")
        val_dataset = BA3Motif(data_root_path, mode="evaluation")
        test_dataset = BA3Motif(data_root_path, mode="testing")
        num_cls = 3

    elif dataset_name == 'MUTAG':
        train_dataset = Mutagenicity(data_root_path, target="explainer", mode="training")
        val_dataset = Mutagenicity(data_root_path, target="explainer", mode="evaluation")
        test_dataset = Mutagenicity(data_root_path, target="explainer", mode="testing")
        num_cls = 2

    elif dataset_name == 'FC':
        train_dataset = FluorideCarbonyl(data_root_path, mode="training")
        val_dataset = FluorideCarbonyl(data_root_path, mode="evaluation")
        test_dataset = FluorideCarbonyl(data_root_path, mode="testing")
        num_cls = 2

    elif dataset_name == 'MNIST':
        train_dataset = MNIST75sp(data_root_path, mode="training")
        val_dataset = MNIST75sp(data_root_path, mode="evaluation")
        test_dataset = MNIST75sp(data_root_path, mode="testing")
        num_cls = 2

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, val_dataset, test_dataset, num_cls
