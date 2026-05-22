import torch
import numpy as np
from sklearn.metrics import f1_score
from explainer.explainer_utils import create_edge_embeds, sample_graph
from explainer.baseline_resolver import score_base_explainer_batch

# New: distribution-only metrics (kept separate from F1)
def compute_polarity(scores):
    """Polarity in [0,1]: high when scores near 0/1, low near 0.5."""
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    p = np.asarray(scores, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((2.0 * p - 1.0) ** 2))

def compute_bimodality_coefficient(scores):
    """
    Normalized bimodality coefficient in [0,1].
    Uses Pearson β2 kurtosis: BC = (skew^2 + 1) / kurt, then map [1/3, 1] → [0,1].
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    x = np.asarray(scores, dtype=float)
    if x.size < 2:
        return 0.0
    x = np.clip(x, 0.0, 1.0)
    m = float(np.mean(x))
    c = x - m
    m2 = float(np.mean(c ** 2))
    if m2 <= 1e-12:
        return 0.0
    m3 = float(np.mean(c ** 3))
    m4 = float(np.mean(c ** 4))
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)
    if kurt <= 1e-12:
        return 0.0
    bc = (skew ** 2 + 1.0) / kurt
    bc_norm = (bc - (1.0 / 3.0)) / (1.0 - (1.0 / 3.0))
    return float(np.clip(bc_norm, 0.0, 1.0))

def calculate_f1_fixed_threshold(predictions, ground_truth, threshold=0.5):
    y_pred_binary = (predictions >= threshold).astype(int)
    return f1_score(ground_truth, y_pred_binary, zero_division=0)


def evaluate_f1_fidelity(test_loader, model, explainer_base, explainer_guided, device, dataset_name, second_label: str = 'Round_1'):
    if second_label != 'Round_1': # Round is more than 1
        guided_scores = []
        gt_labels = []
        
        explainer_guided.eval()
        model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)                
                guided_mask_scores = explainer_guided(batch).squeeze()
                
                if guided_mask_scores.dim() > 1 and guided_mask_scores.size(-1) == 2:
                    guided_mask_scores = torch.softmax(guided_mask_scores, dim=-1)[:, 1]  # Use positive class probability
                
                edge_gt = batch.edge_gt.squeeze()                
                guided_scores.extend(guided_mask_scores.detach().cpu().numpy())
                gt_labels.extend(edge_gt.detach().cpu().numpy())
        
        # To numpy
        guided_scores = np.array(guided_scores)
        gt_labels = np.array(gt_labels)

        thresholds2 = np.median(guided_scores)

        # F1 score
        guided_f1_score = calculate_f1_fixed_threshold(guided_scores, gt_labels, thresholds2)
        guided_fidelity = evaluate_fidelity(test_loader, model, explainer_base, explainer_guided,
                                            device, dataset_name, threshold= thresholds2, explainer_type='guided')
        
        print(f"{second_label}_Guided Explainer F1 Score: {guided_f1_score:.4f}") 
        print(f"{second_label}_Guided Explainer Fidelity: {guided_fidelity:.4f}")

        results = {
            'guided_f1_score': guided_f1_score,
            'guided_fidelity': guided_fidelity
        }
    else: # Round is 1
        base_scores = []
        guided_scores = []
        gt_labels = []
        
        explainer_base.eval()
        explainer_guided.eval()
        model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                _, original_pred, _ = model(batch)
                base_mask_scores = score_base_explainer_batch(batch, model, explainer_base, device)
                
                guided_mask_scores = explainer_guided(batch).squeeze()
                if guided_mask_scores.dim() > 1 and guided_mask_scores.size(-1) == 2:
                    guided_mask_scores = torch.softmax(guided_mask_scores, dim=-1)[:, 1]  # Use positive class probability
                
                edge_gt = batch.edge_gt.squeeze()
                
                base_scores.extend(base_mask_scores.detach().cpu().numpy())
                guided_scores.extend(guided_mask_scores.detach().cpu().numpy())
                gt_labels.extend(edge_gt.detach().cpu().numpy())
        
        # To numpy
        base_scores = np.array(base_scores)
        guided_scores = np.array(guided_scores)
        gt_labels = np.array(gt_labels)

        threshold_from_base = np.mean(base_scores)
        threshold_from_guided = np.mean(guided_scores)

        # F1 score
        base_f1_score = calculate_f1_fixed_threshold(base_scores, gt_labels, threshold_from_base)
        guided_f1_score = calculate_f1_fixed_threshold(guided_scores, gt_labels, threshold_from_guided)

        original_fidelity = evaluate_fidelity(test_loader, model, explainer_base, explainer_guided, 
                                              device, dataset_name, threshold=threshold_from_guided, explainer_type='original')
        base_fidelity = evaluate_fidelity(test_loader, model, explainer_base, explainer_guided, 
                                          device, dataset_name, threshold=threshold_from_base, explainer_type='base')
        guided_fidelity = evaluate_fidelity(test_loader, model, explainer_base, explainer_guided, 
                                          device, dataset_name, threshold=threshold_from_guided, explainer_type='guided')

        print(f"Base Explainer F1 Score: {base_f1_score:.4f}")
        print(f"{second_label}_Guided Explainer F1 Score: {guided_f1_score:.4f}") 
        print(f"{second_label}_Guided Explainer Fidelity: {guided_fidelity:.4f}")
        print(f"Original GNN Fidelity: {original_fidelity:.4f}")
        print(f"Base Explainer Fidelity: {base_fidelity:.4f}")
                
        results = {
            'base_f1_score': base_f1_score,
            'guided_f1_score': guided_f1_score,
            'original_fidelity': original_fidelity,
            'base_fidelity': base_fidelity
        }
        
    return results


# New: report polarity and bimodality only (no change to existing F1 functions)
def evaluate_bimod_binar(test_loader, model, explainer_base, explainer_guided, device, dataset_name, second_label: str = 'Round_2'):
    if second_label != 'Round_1':
        guided_scores = []
        explainer_guided.eval()
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                p2 = explainer_guided(batch).squeeze()
                if p2.dim() > 1 and p2.size(-1) == 2:
                    p2 = torch.softmax(p2, dim=-1)[:, 1]
                    print("Something is wrong here...")
                guided_scores.extend(p2.detach().cpu().numpy())
        guided_scores = np.array(guided_scores)

        res = {
            f'{second_label}_polarity': compute_polarity(guided_scores),
            f'{second_label}_bimodality': compute_bimodality_coefficient(guided_scores),
        }
        print(f"{second_label}_Polarity: {res[f'{second_label}_polarity']:.4f}")
        print(f"{second_label}_Bimodality: {res[f'{second_label}_bimodality']:.4f}")

        return res
    else:    
        base_scores = []
        guided_scores = []

        explainer_base.eval()
        explainer_guided.eval()
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                p1 = score_base_explainer_batch(batch, model, explainer_base, device)

                p2 = explainer_guided(batch).squeeze()
                if p2.dim() > 1 and p2.size(-1) == 2:
                    p2 = torch.softmax(p2, dim=-1)[:, 1]
                    print("Something is wrong here...")

                base_scores.extend(p1.detach().cpu().numpy())
                guided_scores.extend(p2.detach().cpu().numpy())

        base_scores = np.array(base_scores)
        guided_scores = np.array(guided_scores)

        res = {
            'base_polarity': compute_polarity(base_scores),
            'base_bimodality': compute_bimodality_coefficient(base_scores),
            f'{second_label}_polarity': compute_polarity(guided_scores),
            f'{second_label}_bimodality': compute_bimodality_coefficient(guided_scores),
        }
        print(f"Base_Polarity: {res['base_polarity']:.4f}")
        print(f"Base_Bimodality: {res['base_bimodality']:.4f}")
        print(f"{second_label}_Polarity: {res[f'{second_label}_polarity']:.4f}")
        print(f"{second_label}_Bimodality: {res[f'{second_label}_bimodality']:.4f}")

    return res


def evaluate_fidelity(
    test_loader,
    model,
    explainer_base,
    explainer_guided,
    device,
    dataset_name,
    threshold,
    explainer_type: str = 'base',
    base_score_fn=None,
):
    if explainer_type == 'original':
        model.eval()
        
        original_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                _, original_pred, _ = model(batch)
                original_pred_class = torch.argmax(original_pred, dim=1)
                true_labels = batch.y
                
                if str(dataset_name) == 'FC':
                    include_mask = (true_labels == 1)
                else:
                    include_mask = torch.ones_like(true_labels, dtype=torch.bool)

                count = int(include_mask.sum().item())
                total_samples += count

                if count > 0:
                    original_correct += (original_pred_class[include_mask] == true_labels[include_mask]).sum().item()

        original_accuracy = original_correct / total_samples        
        return original_accuracy
    
    if explainer_type == 'base':
        explainer_base.eval()
        model.eval()
        
        base_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                _, original_pred, _ = model(batch)
                true_labels = batch.y
                
                if base_score_fn is None:
                    base_mask_scores = score_base_explainer_batch(batch, model, explainer_base, device)
                else:
                    base_mask_scores = base_score_fn(batch, model)
                base_edge_weights = (base_mask_scores > threshold).float()
                
                _, base_pred, _ = model(batch, edge_weights=base_edge_weights)
                base_pred_class = torch.argmax(base_pred, dim=1)
                
                if str(dataset_name) == 'FC':
                    include_mask = (true_labels == 1)
                else:
                    include_mask = torch.ones_like(true_labels, dtype=torch.bool)

                count = int(include_mask.sum().item())
                total_samples += count

                if count > 0:
                    base_correct += (base_pred_class[include_mask] == true_labels[include_mask]).sum().item()

        base_accuracy = base_correct / total_samples
        return base_accuracy
    
    if explainer_type == 'guided':
        explainer_guided.eval()
        model.eval()
        
        guided_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                true_labels = batch.y
                
                guided_mask_scores = explainer_guided(batch).squeeze()
                if guided_mask_scores.dim() > 1 and guided_mask_scores.size(-1) == 2:
                    guided_mask_scores = torch.softmax(guided_mask_scores, dim=-1)[:, 1]
                guided_edge_weights = (guided_mask_scores > threshold).float()
                
                _, guided_pred, _ = model(batch, edge_weights=guided_edge_weights)
                guided_pred_class = torch.argmax(guided_pred, dim=1)
                
                if str(dataset_name) == 'FC':
                    include_mask = (true_labels == 1)
                else:
                    include_mask = torch.ones_like(true_labels, dtype=torch.bool)

                count = int(include_mask.sum().item())
                total_samples += count

                if count > 0:
                    guided_correct += (guided_pred_class[include_mask] == true_labels[include_mask]).sum().item()

        guided_accuracy = guided_correct / total_samples
        return guided_accuracy
