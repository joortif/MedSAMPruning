import numpy as np
import albumentations as A
import torch

seq_test_time_augments = [
    A.GaussNoise(p=1.0),

    A.RandomBrightnessContrast(brightness_limit=[0.2,0.2], contrast_limit=[0.2,0.2], p=1.0),

    A.MotionBlur(blur_limit=[9,9], p=1.0),

    A.MedianBlur(blur_limit=[9,9], p=1.0),

    A.GaussianBlur(blur_limit=(3, 11), p=1.0),

    A.ImageCompression(compression_type="jpeg", quality_range=[40,70], p=1.0),

    A.Downscale(scale_range=[0.25,0.25], p=1.0),

    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
]


def apply_seq_tta(image, tta):
    
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()

        if image.ndim == 4:  
            image = image.squeeze(0)

        if image.ndim == 3:  
            image = image.permute(1, 2, 0)

        image = image.numpy()
    
    if image.dtype != np.uint8 and image.dtype != np.float32:
        image = image.astype(np.float32)
    
    return tta(image=image)    

def forgetting_sequential_TTA(all_preds, gt, image=None):
    
    T, H, W = all_preds.shape

    correct = all_preds == gt.unsqueeze(0)
    
    prev_learned = correct[:-1]            
    now_correct = correct[1:]              
    forgetting_events = prev_learned & (~now_correct)
    forgetting_map = forgetting_events.sum(dim=0).to(torch.int32)  
    
    denom = float(max(1, T-1))
    forgetting_map_norm = forgetting_map.float() / denom
    
    return forgetting_map