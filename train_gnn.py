import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

from sklearn.metrics import accuracy_score
from gnn.GCN import GCN
from gnn.GIN import GIN

from tqdm import tqdm
from dataset.MUTAG_dataset import Mutagenicity
from dataset.BA3_dataset import BA3Motif
from dataset.fluoride_carbonyl_dataset import FluorideCarbonyl
from dataset.mnistsp_dataset import MNIST75sp

EPS = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN Model")
    parser.add_argument("--data_path", default=osp.join(osp.dirname(__file__), "data",), help="Input data path.")
    parser.add_argument('--dataset', type=str, default='MNIST',
                choices=['BA3', 'MUTAG', 'FC', 'MNIST'])
    parser.add_argument("--save_path", default="param", help="Parameter save path")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    
    parser.add_argument("--epochs", type=int, default=500, help="Number of epoch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--verbose", type=int, default=1, help="Interval of evaluation.")
    parser.add_argument('--model', type = str, default='GCN', choices=['GCN', 'GIN']) 
    parser.add_argument('--nlayers', type = int, default = 4, help='Number of GNN layers.')  
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    
    return parser.parse_args()

def get_dataset(data_root_path, dataset):
    
    if dataset == 'BA3':
        train_dataset = BA3Motif(data_root_path, mode="training")
        val_dataset = BA3Motif(data_root_path, mode="evaluation")
        test_dataset = BA3Motif(data_root_path, mode="testing")        
        num_cls = 3
        
    elif dataset == 'MUTAG':
        train_dataset = Mutagenicity(data_root_path, target="explainer", mode="training")
        val_dataset = Mutagenicity(data_root_path, target="explainer", mode="evaluation")
        test_dataset = Mutagenicity(data_root_path, target="explainer", mode="testing")        
        num_cls = 2
    
    elif dataset == 'FC':
        train_dataset = FluorideCarbonyl(data_root_path, mode="training")
        val_dataset = FluorideCarbonyl(data_root_path, mode="evaluation")
        test_dataset = FluorideCarbonyl(data_root_path, mode="testing")        
        num_cls = 2
    
    elif dataset == 'MNIST':
        train_dataset = MNIST75sp(data_root_path, mode="training")
        val_dataset = MNIST75sp(data_root_path, mode="evaluation")
        test_dataset = MNIST75sp(data_root_path, mode="testing")   
        num_cls = 10
        
    return train_dataset, val_dataset, test_dataset, num_cls

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")    
    data_path = osp.join(args.data_path, f"{args.dataset}")
    
    train_dataset, val_dataset, test_dataset, num_cls = get_dataset(data_path, args.dataset)    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    criterion = nn.CrossEntropyLoss()  
    
    # Model selection
    if args.model == 'GCN':
        if args.dataset in ("MUTAG", "FC"):
            model = GCN(train_dataset[0].x.shape[1], num_cls, args.nlayers, args.hidden, 0.5, pool_type='mean', use_jk=False)
        elif args.dataset == "MNIST":
            model = GCN(train_dataset[0].x.shape[1], num_cls, args.nlayers, args.hidden, 0.5, pool_type='sum', use_jk=False)
    else:
        if args.dataset == "BA3": 
            model = GIN(train_dataset[0].x.shape[1], num_cls, args.nlayers, args.hidden, 0.5, pool_type='mean')
    
    model = model.to(device)
    model.reset_parameters()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-4)

    best_acc = 0.0
    save_path = osp.join(args.save_path, f"{args.model}")
    all_labels = []
    # for epoch in tqdm(range(1, args.epochs + 1)):
    #     model.train()
    #     loss_accum = 0.0
    #     for i, data in enumerate(train_loader):
    #         data = data.to(device)
    #         optimizer.zero_grad()
    #         _, output, _ = model(data)
    #         loss = criterion(output, data.y)
    #         loss.backward()
    #         optimizer.step()

    #         loss_accum += loss.item()
    #     train_loss = loss_accum / (i + 1)
    #     print(f"Average train loss of epoch [{epoch}] : {train_loss}")

    #     ###==========Validation==========###
    #     y_label = []
    #     y_pred = []
        
    #     model.eval()
    #     with torch.no_grad():
    #         for i, data in enumerate(val_loader):
    #             data = data.to(device)
    #             _, output, _ = model(data)                   
    #             pred = torch.argmax(output, dim=1).long()
    #             y_pred += pred.cpu().detach().numpy().tolist()
    #             y_label += data.y.cpu().detach().numpy().tolist()
        
    #     acc_val = accuracy_score(y_pred, y_label)
    #     print(f"Epoch [{epoch}] Validation accuracy: {acc_val:.4f}")
        
    #     if acc_val > best_acc:
    #         best_acc = acc_val
    #         print("Best model is saved!")
    #         torch.save(model.state_dict(), osp.join(save_path, f"{args.dataset}_{args.model}_best_val.pth"))
        
    #     # Step the scheduler
    #     scheduler.step(acc_val)
    
    ###==========Testing==========###
    model.load_state_dict(torch.load(osp.join(save_path, f"{args.dataset}_{args.model}_best_val.pth")))
    
    y_label = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print(data.x.shape)
            data = data.to(device)
            _, output, _ = model(data)                   
            pred = torch.argmax(output, dim=1).long()
            y_pred += pred.cpu().detach().numpy().tolist()
            y_label += data.y.cpu().detach().numpy().tolist()

    acc_test = accuracy_score(y_pred, y_label)
    print(f"Test accuracy: {acc_test:.4f}")
