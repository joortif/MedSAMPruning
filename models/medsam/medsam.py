from pathlib import Path
import time

import h5py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.medsam.download_medsam import download_medsam
from models.medsam.utils import mask_to_bbox, preprocess_image, preprocess_mask

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box, mask, H, W, threshold=0.5):

    box_torch = None
    mask_torch = None
    if box is not None:
        box_torch = torch.as_tensor(box, dtype=torch.float, device=img_embed.device)
        if box_torch.ndim == 1:
            box_torch = box_torch[None, None, :]
        elif box_torch.ndim == 2:
            box_torch = box_torch[:, None, :]

    if mask is not None:
        mask_torch = torch.as_tensor(mask, dtype=torch.float, device=img_embed.device)

        if mask_torch.ndim == 2:
            mask_torch = mask_torch[None, :, :]    # (1, H, W)
        mask_torch = mask_torch[:, None, :, :]

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=mask_torch,
    )

    low_res_logits, a = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )
    

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > threshold).astype(np.uint8)
    return medsam_seg

def predict_medsam(image, mask, mask_original, H, W, bbox=True, uniform_pad=False, medsam_model=None, device="cpu", model_path=None):
    
    if medsam_model is None:
        medsam_path = download_medsam(output_dir=model_path)
        medsam_model = load_medsam(medsam_path)
        
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()

        if mask.ndim == 4:     
            mask = mask[0, 0]
        elif mask.ndim == 3:    
            mask = mask[0]

        mask = mask.numpy()
    
    if bbox is True:
        box_np = mask_to_bbox(mask_original, add_noise=True, uniform_pad=uniform_pad)
        
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(image.to(device)) 

    medsam_seg = medsam_inference(medsam_model, image_embedding, box=box_np, mask=mask, H=H, W=W)

    return medsam_seg, box_np


def process_medsam_batches(
    image_paths,
    mask_paths,
    medsam_model,
    device,
    save_dir,
    imgsz=1024,
    batch_size=4,
    h5_name="medsam_logits.h5",
):
   
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    h5_path = save_dir / h5_name

    mask_lookup = {}
    for mp in mask_paths:
        stem = Path(mp).stem.lower()
        if stem not in mask_lookup:
            mask_lookup[stem] = mp

    medsam_model = medsam_model.to(device).eval()

    start_time = time.perf_counter()

    with h5py.File(str(h5_path), "a") as h5_f:
        pbar = tqdm(total=len(image_paths), desc="Processing images using MedSAM...", unit="img")

        try:
            for start_idx in range(0, len(image_paths), batch_size):
                end_idx = min(len(image_paths), start_idx + batch_size)

                batch_imgs = []
                batch_sizes = []
                batch_boxes = []
                batch_ids = []
                batch_meta = []

                for i in range(start_idx, end_idx):
                    img_path = str(image_paths[i])
                    img_id = Path(img_path).stem
                    mask_path = mask_lookup.get(img_id.lower())

                    if mask_path is None:
                        pbar.update(1)
                        continue

                    img_tensor, H, W = preprocess_image(img_path, imgsz=imgsz)
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0)

                    original_mask, _ = preprocess_mask(mask_path, imgsz=imgsz)
                    box_xyxy_original = mask_to_bbox(original_mask, add_noise=True)
                    box_normalized_1024 = (
                        box_xyxy_original / np.array([W, H, W, H]) * imgsz
                    ).astype(np.float32)

                    batch_imgs.append(img_tensor)
                    batch_sizes.append((H, W))
                    batch_boxes.append(box_normalized_1024)
                    batch_ids.append(img_id)
                    batch_meta.append((img_path, mask_path, box_xyxy_original, H, W))

                if len(batch_imgs) == 0:
                    continue

                batch_img_tensor = torch.cat(batch_imgs, dim=0).to(device, non_blocking=True)
                boxes_tensor = torch.tensor(np.stack(batch_boxes, axis=0), dtype=torch.float32, device=device)

                with torch.no_grad():
                    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        img_embeddings = medsam_model.image_encoder(batch_img_tensor)

                        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                            points=None,
                            boxes=boxes_tensor,
                            masks=None,
                        )

                        low_res_logits, _ = medsam_model.mask_decoder(
                            image_embeddings=img_embeddings,
                            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )

                        logits_upsampled = F.interpolate(
                            low_res_logits,
                            size=(imgsz, imgsz),
                            mode="bilinear",
                            align_corners=False,
                        )

                low_res_logits_cpu = low_res_logits.detach().cpu().numpy().astype(np.float32)
                logits_upsampled_cpu = logits_upsampled.detach().cpu().numpy().astype(np.float32)

                for idx, img_id in enumerate(batch_ids):
                    img_path, mask_path, box_xyxy_original, H, W = batch_meta[idx]

                    grp_name = f"imgs/{img_id}"
                    if grp_name in h5_f:
                        del h5_f[grp_name]
                    g = h5_f.create_group(grp_name)

                    g.create_dataset(
                        "logits_low",
                        data=low_res_logits_cpu[idx],
                        compression="lzf",
                    )
                    g.create_dataset(
                        "logits_upsampled",
                        data=logits_upsampled_cpu[idx],
                        compression="lzf",
                    )

                    g.attrs["image_path"] = img_path
                    g.attrs["mask_path"] = mask_path
                    g.attrs["orig_H"] = np.int32(H)
                    g.attrs["orig_W"] = np.int32(W)
                    g.attrs["box_xyxy_original"] = np.asarray(box_xyxy_original, dtype=np.float32)
                    g.attrs["box_normalized_1024"] = np.asarray(batch_boxes[idx], dtype=np.float32)

                    pbar.update(1)

                h5_f.flush()

                del batch_img_tensor, img_embeddings, low_res_logits, logits_upsampled
                del low_res_logits_cpu, logits_upsampled_cpu
                torch.cuda.empty_cache() if device.type == "cuda" else None

        finally:
            pbar.close()

    total_time = time.perf_counter() - start_time
    return total_time