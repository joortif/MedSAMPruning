import os
from pathlib import Path
import time

import numpy as np
import albumentations as A
import pandas as pd
import torch
from tqdm import tqdm

from models.medsam.medsam import predict_medsam
from models.medsam.download_medsam import download_medsam, load_medsam
from pruning_metrics.utils import average_class_score, preprocess_gt, preprocess_image_and_mask, preprocess_image_tta, weighted_image_score

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

def compute_forgetting_seq_tta_rankings(image_path, mask_path, csv_path, model_path):

    scores_list = []

    image_files = sorted(
        Path(image_path) / f
        for f in os.listdir(image_path)
    )

    mask_files = {
        Path(f).stem: Path(mask_path) / f
        for f in os.listdir(mask_path)
    }

    start = time.time()
    
    medsam_path = download_medsam(output_dir=model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        raise RuntimeError(
            "CUDA is not available. MedSAM and U-Net require a GPU."
        )
        return
    
    device = torch.device(device)
    
    medsam_model = load_medsam(medsam_path, device=device)
    
    for img_file in tqdm(image_files, desc="Computing forgetting scores using TTA..."):
        
        img = img_file.stem
            
        all_preds = []
        
        gt_mask_orig, gt_mask_resize = preprocess_gt(mask_files[img], resize=True, imgsz=1024)
        image_no_resize, img_resize, H, W = preprocess_image_tta(img_file, imgsz=1024)
        
        current = img_resize.copy()
        
        image, gt_256, gt_mask_resize_np = preprocess_image_and_mask(img_resize, gt_mask_resize)
        
        medsam_seg, _ = predict_medsam(image, gt_256, gt_mask_resize_np, H, W, medsam_model=medsam_model, device=device)
        
        all_preds.append(torch.from_numpy(medsam_seg))
        
        current = img_resize
        
        for i, tta in enumerate(seq_test_time_augments):
        
            out = apply_seq_tta(current, tta)
            
            image_np = out["image"]
            
            image_tta = torch.from_numpy(image_np).float()
            if image_tta.ndim == 3:
                image_tta = image_tta.permute(2, 0, 1)

            image_tta = image_tta.unsqueeze(0)
                    
            medsam_seg, box_np = predict_medsam(image_tta, gt_256, gt_mask_resize_np, H, W, medsam_model=medsam_model, device=device, uniform_pad=True)
            
            all_preds.append(torch.from_numpy(medsam_seg))
            
            current = image_np.copy()
            
        all_preds = torch.stack(all_preds, dim=0)
        all_preds_inversed = torch.flip(all_preds, dims=[0]) 
        
        forgetting_map = forgetting_sequential_TTA(all_preds_inversed, gt_mask_orig, image_no_resize)
        score_fg, score_bg = average_class_score(forgetting_map, gt_mask_orig, threshold=None)
        image_score = weighted_image_score(score_fg, score_bg, alpha=1)
        scores_list.append({"id": img, "forgetting_seq_tta": image_score})

    end = time.time()
    total_time = end - start

    print(f"Total elapsed time: {total_time:.2f}s")
    df_scores = pd.DataFrame(scores_list)
    df_scores = df_scores.sort_values(by="forgetting_seq_tta", ascending=False)
    df_scores.to_csv(csv_path, index=False, sep=";", decimal=",")
    print(f"CSV saved in {csv_path}")

    return dict(zip(df_scores["id"], df_scores["forgetting_seq_tta"]))