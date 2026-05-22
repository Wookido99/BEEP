import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from explainer.explainer_utils import create_edge_embeds, sample_graph
from explainer.baseline_resolver import score_base_explainer_batch

def evaluate_auc(loader, explainer, model, device, training=False):    
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        for batch in loader:
            explainer.eval()
            model.eval()
            batch = batch.to(device)
            mask = score_base_explainer_batch(batch, model, explainer, device)

            edge_gt = batch.edge_gt.squeeze()
            
            all_gt.extend(edge_gt.detach().cpu().numpy())
            all_pred.extend(mask.detach().cpu().numpy())
    
    roc_auc = roc_auc_score(all_gt, all_pred)
    return roc_auc

def evaluate_auc_class(loader, classifier, device, model=None):
    all_gt = []
    all_pred = []
    
    with torch.no_grad():
        for batch in loader:
            classifier.eval()
            batch = batch.to(device)
                        
            if model is not None:
                prob = classifier(batch, model)
            else:
                prob = classifier(batch)    
            # prob = classifier(batch)    
            edge_gt = batch.edge_gt.squeeze()
            
            all_gt.extend(edge_gt.detach().cpu().numpy())
            all_pred.extend(prob.detach().cpu().numpy())
    
    roc_auc = roc_auc_score(all_gt, all_pred)
    
    return roc_auc
