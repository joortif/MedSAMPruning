import cv2
import numpy as np
import torch

def mask_to_bbox(mask, add_noise=False, noise_factor=0.2, uniform_pad=False):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    if add_noise:
        w = x1 - x0
        h = y1 - y0

        if uniform_pad:
            pad_x = int(np.round(noise_factor * w))
            pad_y = int(np.round(noise_factor * h))
            
            x0 -= pad_x
            x1 += pad_x
            y0 -= pad_y
            y1 += pad_y
        else:
            pad_x0 = int(np.round(np.random.uniform(0, noise_factor) * w))
            pad_x1 = int(np.round(np.random.uniform(0, noise_factor) * w))
            pad_y0 = int(np.round(np.random.uniform(0, noise_factor) * h))
            pad_y1 = int(np.round(np.random.uniform(0, noise_factor) * h))

            x0 -= pad_x0
            y0 -= pad_y0
            x1 += pad_x1
            y1 += pad_y1

    return [x0, y0, x1, y1]

def preprocess_image(image_path, imgsz=1024, normalize_per_image=True, device=None):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img_bgr.ndim == 2:
        img_bgr = np.repeat(img_bgr[:, :, None], 3, axis=-1)
        
    if img_bgr.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr
        
    H, W = img_rgb.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    
    img_f = img_resized.astype(np.float32)
    if normalize_per_image:
        minv = img_f.min()
        maxv = img_f.max()
        denom = max(maxv - minv, 1e-8)
        img_f = (img_f - minv) / denom
    else:
        img_f = img_f / 255.0
        
    img_f = np.ascontiguousarray(img_f.transpose(2, 0, 1)) 
    img_tensor = torch.from_numpy(img_f).unsqueeze(0)  

    if device is not None:
        img_tensor = img_tensor.to(device, non_blocking=True)

    return img_tensor, H, W

def preprocess_mask(mask_path, imgsz=256):
    
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if m.ndim == 3:
        m = m[..., 0]

    mask_resized = cv2.resize(m, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.float32)

    mask_full = (m > 0).astype(np.uint8)

    return mask_full, mask_resized
