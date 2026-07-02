import torch

def EL2N(logits_mask, gt_mask):
    logits = torch.as_tensor(logits_mask, dtype=torch.float32)
    gt = torch.as_tensor(gt_mask, dtype=torch.float32, device=logits.device)
    probs = torch.sigmoid(logits)
    return (probs - gt).square()

def compute_el2n(logits, gt_mask):
    return EL2N(logits, gt_mask)