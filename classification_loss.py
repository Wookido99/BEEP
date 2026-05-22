import torch
import torch.nn.functional as F

def focal_loss(mask, edge_gt, pos_weight = 10.0):
        
    alpha = 0.25
    gamma=2.0
    
    mask = mask.view(-1)
    edge_gt = edge_gt.view(-1)

    bce_loss = F.binary_cross_entropy_with_logits(
        mask,
        edge_gt,
        reduction='none',
        pos_weight=torch.tensor(pos_weight, device=mask.device)
    )

    pt = torch.exp(-bce_loss)
    focal_factor = alpha * (1 - pt) ** gamma
    focal_loss = focal_factor * bce_loss

    return focal_loss.mean()

def confidence_margin_loss(mask: torch.Tensor, edge_gt: torch.Tensor, margin: float = 2.0, pos_weight: float = 10.0) -> torch.Tensor:
    
    logits = mask.view(-1)
    labels = edge_gt.view(-1).float()
    
    y = labels * 2.0 - 1.0

    loss_per_edge = F.relu(margin - y * logits)

    weight = labels * pos_weight + (1.0 - labels)
    weighted = weight * loss_per_edge

    return weighted.mean()

def polar_loss(mask, edge_gt, pos_weight=10.0):

    mask = mask.view(-1)
    edge_gt = edge_gt.view(-1).float()

    # Base binary cross entropy loss with class weighting
    bce_loss = F.binary_cross_entropy_with_logits(
        mask,
        edge_gt,
        reduction='none',
        pos_weight=torch.tensor(pos_weight, device=mask.device)
    )

    # Combine all loss components
    total_loss = bce_loss

    return total_loss.mean()

def nnpu_loss(mask, edge_gt, positive_prior=0.1, beta=0.0):
    """
    Non-Negative PU (nnPU) Loss for edge explanation
    
    Args:
        mask: predicted mask scores (logits) [N_edges]
        edge_gt: ground truth labels [N_edges] 
                 1.0 = positive (labeled), -1.0 = unlabeled, 0.0 = negative (if any)
        positive_prior: prior probability of positive class in unlabeled data
        beta: coefficient for non-negative constraint (0 for standard nnPU)
    
    Returns:
        nnPU loss value
    """
    mask = mask.view(-1)
    edge_gt = edge_gt.view(-1).float()
    
    # Separate positive, negative, and unlabeled samples
    positive_mask = (edge_gt == 1.0)
    negative_mask = (edge_gt == 0.0)
    unlabeled_mask = (edge_gt == -1.0)
    
    # Check if we have positive samples
    if positive_mask.sum() == 0:
        # Only unlabeled data - treat as standard unsupervised learning
        if unlabeled_mask.sum() == 0:
            return torch.tensor(0.0, device=mask.device, requires_grad=True)
        
        unlabeled_scores = mask[unlabeled_mask]
        # Encourage diverse predictions on unlabeled data
        sigmoid_scores = torch.sigmoid(unlabeled_scores)
        entropy_loss = -torch.mean(
            sigmoid_scores * torch.log(sigmoid_scores + 1e-8) + 
            (1 - sigmoid_scores) * torch.log(1 - sigmoid_scores + 1e-8)
        )
        return entropy_loss
    
    # Extract scores for each type
    positive_scores = mask[positive_mask]
    
    # For nnPU learning, we primarily use positive and unlabeled data
    # Negative samples (if any) are treated as unlabeled for nnPU
    all_unlabeled_mask = unlabeled_mask | negative_mask
    unlabeled_scores = mask[all_unlabeled_mask]
    
    # Positive risk: R_P^+ = E[ℓ(f(x), +1) | x ∈ P]
    positive_risk = torch.mean(F.binary_cross_entropy_with_logits(
        positive_scores, 
        torch.ones_like(positive_scores), 
        reduction='none'
    ))
    
    if unlabeled_scores.numel() == 0:
        return positive_risk
    
    # Negative risk from unlabeled: R_U^- = E[ℓ(f(x), -1) | x ∈ U]
    unlabeled_negative_risk = torch.mean(F.binary_cross_entropy_with_logits(
        unlabeled_scores,
        torch.zeros_like(unlabeled_scores),
        reduction='none'
    ))
    
    # nnPU risk estimator: R_nnPU = π_p * R_P^+ + max(0, R_U^- - π_p * R_P^+)
    negative_risk = unlabeled_negative_risk - positive_prior * positive_risk
    negative_risk = torch.max(torch.tensor(0.0, device=mask.device), negative_risk)
    
    # Total nnPU loss
    nnpu_risk = positive_prior * positive_risk + negative_risk
    
    return nnpu_risk