import os
import csv
import shutil
from random import sample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def pruning(images_dir, pruning_rate, imgs_sublist=None, csv_file=None, desc=True, img_ext=".png", mixed=False, mixing_percentage=0.5):
    all_files = os.listdir(images_dir)
        
    images = [img for img in all_files if img.endswith(img_ext) or img.endswith(".jpg")]

    if imgs_sublist is not None:
        imgs_sublist = set(imgs_sublist)  
        images = [img for img in images if img in imgs_sublist]
        
    N = len(images)
    n_images = int(N * (1 - pruning_rate))
    
    if csv_file is None:
        random_images = sample(images, n_images)
        print(f"{len(random_images)} randomly selected from {images_dir}")
        return random_images
    
    print(f"Selecting {n_images}...")
    data = {}
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for key, value in reader:
            filename = f"{key}{img_ext}"
            if filename in images:  
                data[filename] = float(value.replace(",","."))            
    
    ordered_dict = dict(sorted(data.items(), key= lambda x: x[1], reverse=desc))
    ordered_images = list(ordered_dict.keys())
    
    if not mixed:
        reduced_subset = ordered_images[:n_images]
        return reduced_subset
        
    n_top = int(round(n_images * mixing_percentage))
    n_bottom = n_images - n_top

    print(f"Picking {n_top} images from top and {n_bottom} from bottom.")
    top_images = ordered_images[:n_top]                
    bottom_images = ordered_images[-n_bottom:]
    
    merged = top_images + bottom_images
    merged = list(dict.fromkeys(merged))

    print(f"Merged top and bottom images: {len(merged)} images obtained.")
    
    return merged

def split_with_pruning(images_dir, labels_dir, output_dir, pruning_rate, csv_file=None, desc=True,
                       image_ext=".png", mask_ext=".png", train=0.7, val=0.2, test=0.1, random_state=123, verbose=False, mixed=False, mixed_percentage=0.5):
    
    for part in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, part, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, part, "labels"), exist_ok=True)
    
    images = [img for img in os.listdir(images_dir) if img.endswith(".png") or img.endswith(".jpg")]
            
    if test > 0:
        train_val_imgs, test_imgs = train_test_split(
            images,
            test_size=test,
            shuffle=True,
            random_state=random_state
        )

        test_images_dir = os.path.join(output_dir, "test", "images")
        test_labels_dir = os.path.join(output_dir, "test", "labels")

        for img_name in test_imgs:
            base = os.path.splitext(img_name)[0]
            mask_name = base + mask_ext

            shutil.copy(os.path.join(images_dir, img_name),
                        os.path.join(test_images_dir, img_name))
            shutil.copy(os.path.join(labels_dir, mask_name),
                        os.path.join(test_labels_dir, mask_name))

        if verbose:
            print(f"{len(test_imgs)} images saved in {os.path.join(output_dir, 'test')}")
    else:
        train_val_imgs = images
        test_imgs = []
        if verbose:
            print("No test split requested (test=0). All images used for train/val.")
        
    reduced_subset = train_val_imgs
    if pruning_rate > 0:
        reduced_subset = pruning(images_dir, pruning_rate, train_val_imgs, csv_file, desc, img_ext=image_ext, mixed=mixed, mixing_percentage=mixed_percentage)
    
    if verbose:
        print(f"{len(reduced_subset)} images saved for training and validation split after pruning.")
    
    rel_val = val / (train + val)
    train_subset, val_subset = train_test_split(reduced_subset, test_size=rel_val, shuffle=True, random_state=random_state)

    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    for img_name in train_subset:
        base = os.path.splitext(img_name)[0]
        mask_name = base + mask_ext

        shutil.copy(os.path.join(images_dir, img_name),
                    os.path.join(train_images_dir, img_name))
        shutil.copy(os.path.join(labels_dir, mask_name),
                    os.path.join(train_labels_dir, mask_name))

    if verbose:
        print(f"{len(train_subset)} images saved for training.")

    val_images_dir = os.path.join(output_dir, "val", "images")
    val_labels_dir = os.path.join(output_dir, "val", "labels")
    for img_name in val_subset:
        base = os.path.splitext(img_name)[0]
        mask_name = base + mask_ext

        shutil.copy(os.path.join(images_dir, img_name),
                    os.path.join(val_images_dir, img_name))
        shutil.copy(os.path.join(labels_dir, mask_name),
                    os.path.join(val_labels_dir, mask_name))

        
    if verbose:
        print(f"{len(val_subset)} images saved for validation.")