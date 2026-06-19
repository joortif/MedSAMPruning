import time
import cv2
import math
from collections import defaultdict
import numpy as np
import tqdm as tqdm
import csv

from .utils import extract_instances_from_semantic_mask

def contour_perimeter_and_area(binary_mask):
    img = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    
    perimeter = float(cv2.arcLength(cnt, True))
    area = int(cv2.contourArea(cnt))  

    return perimeter, area

def scs(perimeter, area):
    return perimeter / area

def si_scs(perimeter, area):
    return perimeter / (2.0 * math.sqrt(math.pi * area))

def cb_scs(masks, out_csv_path=None):

    all_instances = []  
    per_image_instances = {}

    start = time.time()
    for i, mask in tqdm(masks.items(), desc="Processing masks"):
        per_image_instances[i] = []
        instances = extract_instances_from_semantic_mask(mask)
        for cls, inst_mask in instances:
            perim, area = contour_perimeter_and_area(inst_mask)
            if perim == 0 or area == 0:
                #print(f"Perimeter {perim} and area {area} found in mask {i}.png")
                continue
            scs_mask = scs(perim, area)
            si_mask = si_scs(perim, area)
            all_instances.append((i, cls, si_mask))
            per_image_instances[i].append({'class': cls, 'area': area, 'perimeter': perim, 'scs': scs_mask, 'si_scs': si_mask, 'mask': inst_mask})

    # Compute per-class totals of SI-SCS across dataset (for CB normalization)
    class_totals = defaultdict(float)
    for (img_idx, cls, si) in all_instances:
        class_totals[cls] += float(si)

    image_cb_scores = {}
    for mask_name, cls, si_val in all_instances:
        denom = class_totals.get(cls)
        norm_score = si_val / denom
        image_cb_scores[mask_name] = image_cb_scores.get(mask_name, 0.0) + norm_score
        
    end = time.time()
    total_time = end - start
    
    print(f"Total time: {total_time}")

    rows = sorted(image_cb_scores.items(), key=lambda x: x[0])
    with open(out_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(['id', 'cb-scs'])
        for img_id, score in image_cb_scores.items():
            writer.writerow([img_id, score])

    return image_cb_scores, per_image_instances, class_totals