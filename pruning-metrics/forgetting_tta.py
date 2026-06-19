import torch
import cv2
import albumentations as A
import numpy as np

test_time_augments = [
    A.ReplayCompose([A.HorizontalFlip(p=1.0)]),
    A.ReplayCompose([A.VerticalFlip(p=1.0)]),
    A.ReplayCompose([A.VerticalFlip(p=1.0), A.HorizontalFlip(p=1.0)]),
    
    A.Compose([A.GaussianBlur(p=1.0, sigma_limit=[2, 4])]),
    A.Compose([A.GaussNoise(p=1.0)]),
    A.Compose([A.RandomBrightnessContrast(p=1.0)]),
    A.Compose([A.CLAHE(p=1.0)])
]

def apply_tta(image, mask, tta):
    
    if image.dtype != np.uint8 and image.dtype != np.float32:
        image = image.astype(np.float32)
    
    transformation = tta.transforms[0].__class__.__name__
    needs_mask = any(
        transformation in ["HorizontalFlip", "VerticalFlip", "SafeRotate"]
        for t in tta.transforms
    )
    
    if needs_mask:       
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu()
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask = mask.numpy()
        
        return tta(image=image, mask=mask)
    return tta(image=image)    

def invert_mask_tta(pred_tta, replay):
    pred = pred_tta
    tgt_shape = replay.get("image_shape", None)
    if tgt_shape is not None:
        H_tta, W_tta = tgt_shape[0], tgt_shape[1]
        if (pred.shape[0], pred.shape[1]) != (H_tta, W_tta):
            pred = cv2.resize(pred, (W_tta, H_tta), interpolation=cv2.INTER_NEAREST)

    transforms = replay.get("transforms", [])
    for t in reversed(transforms):
        if not t.get("applied", False):
            continue

        name = t.get("__class_fullname__") or t.get("__class__", None) or ""
        params = t.get("params", {}) or {}

        if name == "HorizontalFlip":
            inv_aug = A.HorizontalFlip(p=1.0)
            out = inv_aug(image=pred)
            pred = out["image"]
            continue

        if name == "VerticalFlip":
            inv_aug = A.VerticalFlip(p=1.0)
            out = inv_aug(image=pred)
            pred = out["image"]
            continue

        if name == "SafeRotate":
            angle = None
            if "rotate" in params:
                lim = params["rotate"]
                if isinstance(lim, (list, tuple)):
                    angle = lim[0]
                else:
                    angle = lim

            inv_angle = (-float(angle)) % 360.0

            inv_aug = A.Rotate(limit=(inv_angle, inv_angle), p=1.0)
            out = inv_aug(image=pred)
            pred = out["image"]
            continue

    pred = np.asarray(pred)
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred[...,0]
    return pred

def forgetting_TTA(all_preds, gt, image=None):

    fn = (all_preds == 0) & (gt.unsqueeze(0) == 1)
    fn_mask = fn.sum(dim=0).to(torch.int32)
    
    fp = (all_preds == 1) & (gt.unsqueeze(0) == 0)
    fp_mask = fp.sum(dim=0).to(torch.int32)

    forgetting_map = fn_mask + fp_mask
    
    return forgetting_map

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