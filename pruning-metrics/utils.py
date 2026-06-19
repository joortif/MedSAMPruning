import cv2
import numpy as np
import tqdm as tqdm
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
