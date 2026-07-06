import os
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def _normalize_color_tuple(v):
    v = np.asarray(v).flatten()

    if v.size != 3:
        raise ValueError(f"Color inválido: se esperaban 3 canales, llegó {v}")

    if np.issubdtype(v.dtype, np.floating) and v.max() <= 1.0:
        v = np.round(v * 255)

    return tuple(int(x) for x in v)

def extract_instances_from_semantic_mask(sem_mask):
    instances = []
    classes = np.unique(sem_mask)
    for cls in classes:
        if cls == 0:
            continue
        binmask = (sem_mask == cls).astype(np.uint8)

        num_labels, labels = cv2.connectedComponents(binmask, connectivity=8)
        for lab in range(1, num_labels):
            comp_mask = (labels == lab).astype(np.uint8)
            area = int(comp_mask.sum())
            instances.append((int(cls), comp_mask))
    return instances

def png_to_semantic_mask(path: str, colormap: dict = None, background_id: int = 0) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No puedo abrir {path}")

    if len(img.shape) == 2:
        return img.astype(np.int32)

    if img.shape[2] == 4:
        rgb = img[:, :, :3]

        if np.all(rgb[...,0] == rgb[...,1]) and np.all(rgb[...,1] == rgb[...,2]):
            return rgb[...,0].astype(np.int32)

        img = rgb

    if img.shape[2] == 3:
        h, w, _ = img.shape

        if np.all(img[...,0] == img[...,1]) and np.all(img[...,1] == img[...,2]):
            return img[...,0].astype(np.int32)

        if colormap is None:
            flat = img.reshape(-1, 3)
            colors, counts = np.unique(flat, axis=0, return_counts=True)
            if len(colors) != 2:
                raise ValueError(f"Se esperaban 2 colores, hay {len(colors)}.")
            fg = colors[np.argmin(counts)]
            mask = (img == fg).all(axis=2).astype(np.int32)
            return mask

        cmap = {int(k): _normalize_color_tuple(v) for k, v in colormap.items()}
        color_to_id = {tuple(v): k for k, v in cmap.items()}

        mask = np.full((h, w), background_id, dtype=np.int32)

        for color, cls_id in color_to_id.items():
            match = (img == np.array(color)).all(axis=2)
            mask[match] = cls_id

        return mask

    raise ValueError(f"Invalid PNG format: shape={img.shape}")

# Pruning metrics adaptation to semantic segmentation

def average_class_score(scores_mask, gt_mask, threshold=0.5):
    if not torch.is_tensor(scores_mask):
        scores_mask = torch.tensor(scores_mask)

    scores = scores_mask.float()
    
    if threshold is not None:
        gt = gt_mask.float()  
        mask_fg = (gt > threshold)
    else:
        mask_fg = (gt_mask == 1)
    
    mask_bg = ~mask_fg
        
    N_fg = int(mask_fg.sum().item())
    N_bg = int(mask_bg.sum().item())
    
    score_fg = 0.0
    score_bg = 0.0
    
    if N_fg > 0.0:
        score_fg = float(scores[mask_fg].mean().item())
        
    if N_bg > 0.0:
        score_bg = float(scores[mask_bg].mean().item())
        
    return score_fg, score_bg

def weighted_image_score(fg_score, bg_score, alpha=0.9):
    return alpha * fg_score + (1-alpha) * bg_score 

def preprocess_gt(mask_path, resize=False, imgsz=256):
        
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if m.ndim == 3 and m.shape[2] == 4:
        m = m[:, :, :3]
        
    if m.ndim == 3 and m.shape[2] == 3:
        m_gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        m = m_gray
        
    mask_full = (m > 0).astype(np.float32)
    gt_mask = torch.from_numpy(mask_full)

    gt_mask_resize=None
    if resize:
        mask_resized = cv2.resize(mask_full, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.float32)

        gt_mask_resize = torch.from_numpy(mask_resized)
    return gt_mask, gt_mask_resize

def preprocess_image_tta(image_path, resize=True, imgsz=1024):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img.ndim == 2:
        img_3c = np.stack([img, img, img], axis=-1)
    else:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img_3c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    H, W = img_3c.shape[:2]
    
    img_1024 = None
    
    if resize:
    
        img_resized = cv2.resize(img_3c, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        img_1024 = img_resized.astype(np.uint8)
        
        minv = float(img_1024.min())
        maxv = float(img_1024.max())
        denom = max(maxv - minv, 1e-8)
        denom = maxv - minv if (maxv - minv) >= 1e-8 else 1e-8
        img_1024 = (img_1024 - minv) / denom

    return img_3c, img_1024, H, W

def preprocess_image_and_mask(img, gt):
    gt_numpy = gt
    
    if isinstance(gt_numpy, torch.Tensor):
        gt_numpy = gt.detach().cpu()
        if gt_numpy.ndim == 3:
            gt_numpy = gt_numpy.squeeze(0)
        gt_numpy = gt_numpy.numpy()
    
    gt_mask_256 = cv2.resize(gt_numpy, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    image = torch.from_numpy(img).float()
    gt_mask_256 = torch.from_numpy(gt_mask_256).long()
    
    if image.ndim == 3:
        image = image.permute(2, 0, 1)
    
    mask_tta = gt_mask_256.unsqueeze(0)
    image = image.unsqueeze(0)
    
    return image, gt_mask_256, gt_numpy


def get_data_from_name(h5_file, image_name, exclude=None):
    logits_low = None
    logits_upsampled = None
    pred_mask = None
    
    if exclude and any(ex in image_name for ex in exclude):
        return None, None, None
        
    with h5py.File(h5_file, 'r') as f:
        
        grp_path = f"imgs/{image_name}"  
        grp = f[grp_path]
        
        if 'logits_low' in grp:
            logits_low = grp['logits_low'][()]
        if 'logits_upsampled' in grp:
            logits_upsampled = grp['logits_upsampled'][()]
        if 'pred_mask' in grp:
            pred_mask = grp['pred_mask'][()]
        
    return logits_low, logits_upsampled, pred_mask

def compute_ranking_scores(image_path, mask_path, logits_path, metric_fn, metric_name, csv_path):
    scores_list = []
    metric_times = []

    images = sorted(
        Path(img).stem
        for img in os.listdir(image_path)
    )

    mask_files = {
        Path(f).stem: Path(mask_path) / f
        for f in os.listdir(mask_path)
    }

    start = time.time()
    for img in tqdm(images, desc=f"Computing {metric_name} scores"):
        mask_file = mask_files[img]

        _, gt_mask_resized = preprocess_gt(mask_file, resize=True, imgsz=256)
        logits_low, logits_upsampled, pred_mask = get_data_from_name(logits_path, img) #256x256, HxW, HxW

        logits = logits_low.squeeze(0)
        
        t0 = time.perf_counter()
        
        metric_mask = metric_fn(logits, gt_mask_resized)
        score_fg, score_bg = average_class_score(metric_mask, gt_mask_resized)

        image_score = weighted_image_score(score_fg, score_bg, alpha=1)

        t1 = time.perf_counter()

        metric_times.append(t1 - t0)
        
        scores_list.append({"id": img, metric_name: image_score})

    total_time = time.time() - start

    print(f"Total elapsed time: {total_time:.2f}s")

    df_scores = pd.DataFrame(scores_list)

    df_scores = df_scores.sort_values(
        by=metric_name,
        ascending=False,
    )

    df_scores.to_csv(
        csv_path,
        index=False,
        sep=";",
        decimal=",",
    )

    print(f"CSV saved in {csv_path}")

    return dict(zip(df_scores["id"], df_scores[metric_name]))