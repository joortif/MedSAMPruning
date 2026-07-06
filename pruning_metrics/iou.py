import os
from pathlib import Path
import time
import h5py
import torch
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

from pruning_metrics.utils import get_data_from_name, preprocess_gt

def compute_iou(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union != 0 else 1.0

def compute_iou_ranking(
    image_path,
    mask_path,
    logits_path,
    csv_path,
):
    scores_list = []
    iou_times = []

    images = sorted(Path(img).stem for img in os.listdir(image_path))

    mask_files = {
        Path(f).stem: Path(mask_path) / f
        for f in os.listdir(mask_path)
    }

    for img in tqdm(images, desc="Computing IoU scores"):

        mask_file = mask_files[img]

        t0 = time.perf_counter()

        _, logits_upsampled, _ = get_data_from_name(
            logits_path,
            img
        )
        
        with h5py.File(logits_path, "r") as f:
            grp = f[f"imgs/{img}"]
            H = int(grp.attrs["orig_H"])
            W = int(grp.attrs["orig_W"])

        prob = torch.sigmoid(
            torch.from_numpy(logits_upsampled)
        ).numpy()

        prob = np.squeeze(prob)

        prob_resized = cv2.resize(
            prob,
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )

        pred_mask = (prob_resized >= 0.5).astype(np.uint8)

        gt_mask = cv2.imread(
            str(mask_file),
            cv2.IMREAD_GRAYSCALE
        )

        gt_mask = (gt_mask > 0).astype(np.uint8)

        iou_score = compute_iou(pred_mask, gt_mask)

        t1 = time.perf_counter()
        iou_times.append(t1 - t0)

        scores_list.append({
            "id": img,
            "iou": float(iou_score)
        })

    df = pd.DataFrame(scores_list)

    df = df.sort_values("iou", ascending=False)

    output_dir = Path(csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(
        csv_path,
        index=False,
        sep=";",
        decimal=","
    )

    print(f"CSV saved in {csv_path}")
    print(f"Mean IoU time: {np.mean(iou_times):.4f}s")

    return dict(zip(df["id"], df["iou"]))