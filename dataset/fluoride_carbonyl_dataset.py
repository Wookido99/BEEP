import os.path as osp
import random
import sys

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from dataset.utils import get_edge_gt_from_node_imp

class FluorideCarbonyl(InMemoryDataset):
    splits = ["training", "evaluation", "testing"]

    def __init__(
        self, root, mode="testing", transform=None, pre_transform=None, pre_filter=None
    ):
        assert mode in self.splits
        self.mode = mode
        super(FluorideCarbonyl, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index("{}.pt".format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
        
    @property
    def raw_file_names(self):
        return ["fluoride_carbonyl.npz"]

    @property
    def processed_file_names(self):
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, "fluoride_carbonyl.npz")):
            print(
                "raw data of `fluoride_carbonyl.npz` doesn't exist, please place it in the raw directory."
            )
            raise FileNotFoundError

    def process(self):
        # Load the fluoride_carbonyl data
        data = np.load(osp.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True)
        att, X, y, df = data['attr'], data['X'], data['y'], data['smiles']
        ylist = [y[i][0] for i in range(y.shape[0])]
        ylist = list(map(int, ylist))

        X = X[0]

        data_list = []

        for idx in range(len(X)):
            x = torch.from_numpy(X[idx]['nodes']).float()
            edge_attr = torch.from_numpy(X[idx]['edges']).float()
            y = torch.tensor([ylist[idx]], dtype=torch.long)

            e1 = torch.from_numpy(X[idx]['receivers']).long()
            e2 = torch.from_numpy(X[idx]['senders']).long()
            edge_index = torch.stack([e1, e2], dim=0)

            node_imp = torch.from_numpy(att[idx][0]['nodes']).float()
            edge_gt = get_edge_gt_from_node_imp(node_imp, edge_index)
            
            data = Data(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_gt=edge_gt,
                idx=idx,
                name=f"fluoride_carbonyl_{idx}",
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
            
        # Group data by labels
        from collections import defaultdict
        label_groups = defaultdict(list)
        for data in data_list:
            label = int(data.y.item())
            label_groups[label].append(data)
        # Shuffle within each label group
        for label in label_groups:
            random.shuffle(label_groups[label])
        
        # Split each label group into train/val/test
        train_list, eval_list, test_list = [], [], []
        for label, group in label_groups.items():
            total_len = len(group)
            train_split = int(0.8 * total_len)
            eval_split = int(0.9 * total_len)
            train_list.extend(group[:train_split])
            eval_list.extend(group[train_split:eval_split])
            test_list.extend(group[eval_split:])
        
        
        # Shuffle the final splits to mix labels
        random.shuffle(train_list)
        random.shuffle(eval_list)
        random.shuffle(test_list)

        # Save the splits
        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(eval_list), self.processed_paths[1])
        torch.save(self.collate(test_list), self.processed_paths[2])
