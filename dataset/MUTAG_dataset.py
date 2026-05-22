import os
import os.path as osp
import random

import numpy as np
import sklearn.preprocessing as preprocessing
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

class Mutagenicity(InMemoryDataset):
    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/Mutagenicity.zip"

    splits = ["training", "evaluation", "testing"]

    def __init__(
        self, root, target="explainer", mode="testing", transform=None, pre_transform=None, pre_filter=None
    ):
        assert mode in self.splits
        self.mode = mode
        self.target = target
        super(Mutagenicity, self).__init__(root, transform, pre_transform, pre_filter)
        idx = self.processed_file_names.index(f"{self.mode}.pt")
        # self.data, self.slices = torch.load(self.processed_paths[idx]) # idx 0, 1, 2 each
        self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
        

    @property
    def raw_file_names(self):
        return [
            "Mutagenicity/" + i
            for i in [
                "Mutagenicity_A.txt",
                "Mutagenicity_edge_labels.txt",
                "Mutagenicity_edge_gt.txt",
                "Mutagenicity_graph_indicator.txt",
                "Mutagenicity_graph_labels.txt",
                "Mutagenicity_node_labels.txt",
            ]
        ]

    @property
    def processed_file_names(self):
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self):
        if os.path.exists(osp.join(self.raw_dir, "MUTAG")):
            print("Using existing data in folder MUTAG")
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        edge_index = np.loadtxt(
            osp.join(self.raw_dir, self.raw_file_names[0]), delimiter=","
        ).T
        edge_index = torch.from_numpy(edge_index - 1.0).to(
            torch.long
        )  # node idx from 0

        edge_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]))
        encoder = preprocessing.OneHotEncoder().fit(
            np.unique(edge_label).reshape(-1, 1)
        )
        
        edge_gt = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]))
        edge_gt = torch.unsqueeze(torch.LongTensor(edge_gt), 1).long()
        
        edge_attr = encoder.transform(edge_label.reshape(-1, 1)).toarray()
        edge_attr = torch.Tensor(edge_attr)

        node_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[-1]))
        encoder = preprocessing.OneHotEncoder().fit(
            np.unique(node_label).reshape(-1, 1)
        )
        x = encoder.transform(node_label.reshape(-1, 1)).toarray()
        x = torch.Tensor(x)

        z = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]), dtype=int)

        y = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[4]))
        y = torch.unsqueeze(torch.LongTensor(y), 1).long()
              
        num_graphs = len(y)
        total_edges = edge_index.size(1)
        begin = 0

        data_list = []
        for i in range(num_graphs):
            perm = np.where(z == i + 1)[0]
            bound = max(perm)
            end = begin
            for end in range(begin, total_edges):
                if int(edge_index[0, end]) > bound:
                    break

            data = Data(
                x=x[perm],
                y=y[i],
                z=node_label[perm],
                edge_index=edge_index[:, begin:end] - int(min(perm)),
                edge_attr=edge_attr[begin:end],
                edge_gt=edge_gt[begin:end],
                name="mutag_%d" % i,
                idx=i,
            )
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            # Set dataset differently according to target is whether gnn or explainer
            if self.target == "gnn":
                begin = end
                data_list.append(data)
            elif self.target == "explainer":
                if torch.sum(data.edge_gt, dim=0) > 0:
                    data_list.append(data)
                begin = end
        
        random.shuffle(data_list)
                    
        if self.target == "gnn":
            assert len(data_list) == 4337
            # Ensure the file paths are correct
            torch.save(self.collate(data_list[1000:]), self.processed_paths[0])
            torch.save(self.collate(data_list[500:1000]), self.processed_paths[1])
            torch.save(self.collate(data_list[:500]), self.processed_paths[2])
        
        elif self.target == "explainer":
            assert len(data_list) == 1356
            # Ensure the file paths are correct # 80/10/10
            torch.save(self.collate(data_list[286:]), self.processed_paths[0])
            torch.save(self.collate(data_list[143:286]), self.processed_paths[1])
            torch.save(self.collate(data_list[:143]), self.processed_paths[2])