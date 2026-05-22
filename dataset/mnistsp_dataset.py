import numpy as np
import os.path as osp
import pickle
import random
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data
from collections import defaultdict

def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(- dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A

def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data

class MNIST75sp(InMemoryDataset):
    splits = ["training", "evaluation", "testing"]
    
    def __init__(self, root, mode='training', use_mean_px=True,
                 use_coord=True, node_gt_att_threshold=0,
                 transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits, f"mode must be one of {self.splits}"
        self.mode = mode
        self.node_gt_att_threshold = node_gt_att_threshold
        self.use_mean_px, self.use_coord = use_mean_px, use_coord
        
        super(MNIST75sp, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index(f'mnist_75sp_{mode}.pt')
        # self.data, self.slices = torch.load(self.processed_paths[idx])
        self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
        
    
    @property
    def raw_file_names(self):
        return ['mnist_75sp_train.pkl', 'mnist_75sp_test.pkl']

    @property
    def processed_file_names(self):
        return [
            "mnist_75sp_training.pt",
            "mnist_75sp_evaluation.pt",
            "mnist_75sp_testing.pt",
        ]

    def download(self):
        for file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, file)):
                print(f"raw data of `{file}` doesn't exist, please download from our github.")
                raise FileNotFoundError

    def process(self):
        with open(osp.join(self.raw_dir, "mnist_75sp_train.pkl"), "rb") as f:
            train_labels, train_sp_data = pickle.load(f)

        train_data_list = self._build_data_list(train_labels, train_sp_data)
        
        train_data_list = self._reduce_data_list(train_data_list, ratio=0.1)
        random.shuffle(train_data_list)

        torch.save(self.collate(train_data_list), self.processed_paths[0])

        with open(osp.join(self.raw_dir, "mnist_75sp_test.pkl"), "rb") as f:
            test_labels, test_sp_data = pickle.load(f)

        test_data_list = self._build_data_list(test_labels, test_sp_data)
        
        test_data_list = self._reduce_data_list(test_data_list, ratio=0.1)
        random.shuffle(test_data_list)

        torch.save(self.collate(test_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])

    def _build_data_list(self, labels, sp_data):
        data_list = []
        img_size = 28

        for index, sample in enumerate(sp_data):
            mean_px, coord = sample[:2]
            coord = coord / img_size
            A = compute_adjacency_matrix_images(coord)
            N_nodes = A.shape[0]

            A = torch.FloatTensor((A > 0.1) * A)
            edge_index, edge_attr = dense_to_sparse(A)

            x = None
            if self.use_mean_px:
                x = mean_px.reshape(N_nodes, -1)
            if self.use_coord:
                coord = coord.reshape(N_nodes, 2)
                if self.use_mean_px:
                    x = np.concatenate((x, coord), axis=1)
                else:
                    x = coord
            if x is None:
                x = np.ones((N_nodes, 1))  # dummy features
            
            x = np.pad(x, ((0, 0), (2, 0)), "edge")
            
            if self.node_gt_att_threshold == 0:
                node_gt_att = (mean_px > 0).astype(np.float32)
            else:
                node_gt_att = mean_px.copy()
                node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0
            node_gt_att = torch.LongTensor(node_gt_att).view(-1)

            row, col = edge_index
            edge_gt_att = torch.LongTensor(node_gt_att[row] * node_gt_att[col]).view(-1)

            data_list.append(
                Data(
                    x=torch.tensor(x, dtype=torch.float),
                    y=torch.LongTensor([labels[index]]),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    node_gt_att=node_gt_att,
                    edge_gt=edge_gt_att,
                    name=f"MNISTSP-{index}",
                    idx=index,
                )
            )
        return data_list

    def _reduce_data_list(self, data_list, ratio=0.1):
        from collections import defaultdict
        import random

        label_dict = defaultdict(list)
        for data in data_list:
            lbl = data.y.item()
            label_dict[lbl].append(data)

        new_data_list = []
        for lbl, items in label_dict.items():
            keep_num = int(len(items) * ratio)
            sampled_items = random.sample(items, keep_num)
            new_data_list.extend(sampled_items)

        return new_data_list