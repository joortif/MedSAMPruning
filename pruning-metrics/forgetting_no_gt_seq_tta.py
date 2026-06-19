import torch

def forgetting_no_gt_sequential_TTA(all_preds, image=None):
    
    T = all_preds.shape[0]
    
    diffs = all_preds[1:] != all_preds[:-1]   # (T-1, H, W)
    switch_map = diffs.sum(dim=0).to(torch.int32)
    
    return switch_map