import torch
import warnings

def save_sigmoid(logits_mask, eps=1e-7):
    probs = torch.sigmoid(logits_mask)
    
    # Avoid log(0)=-inf
    probs = torch.where(probs <= 0.0, torch.full_like(probs, eps), probs)
    probs = torch.where(probs >= 1.0, torch.full_like(probs, 1.0 - eps), probs)

    return probs

def bernoulli_entropy(logits_mask):
    if not torch.is_tensor(logits_mask):
        logits_mask = torch.tensor(logits_mask)
    logits = logits_mask.float()
    probs = save_sigmoid(logits)

    entropy = - (probs * torch.log(probs) + (1.0 - probs) * torch.log(1 - probs))
    
    if not torch.isfinite(entropy).all():
        warnings.warn("Some elements from entropy are inf or -inf", RuntimeWarning)
    return entropy