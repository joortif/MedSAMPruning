import shutil

from anyio import Path
import cv2
import numpy as np
import os
import albumentations as A

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as BaseDataset

VALID_EXTENSIONS: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

class Dataset(BaseDataset):
    
    def __init__(self, images_dir, masks_dir, classes, augmentation=None, background=None, binary=True):
        self.images_fps = []
        self.ids = []
        for fname in os.listdir(images_dir):
            if fname.lower().endswith(tuple(VALID_EXTENSIONS)):
                fpath = os.path.join(images_dir, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    self.images_fps.append(fpath)
                    self.ids.append(fname)
        
        masks_ids = [os.path.splitext(image_id)[0]+'.png' for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in masks_ids]
        self.background_class = background
        
        self.classes = classes
        self.binary = binary
        self.class_values = [self.classes.index(cls.lower()) for cls in classes]
        
        if self.binary:
            self.class_map = {v: 0 if v == self.background_class else 1 for v in self.class_values}
        else:
            self.class_map = {self.background_class: 255} if self.background_class is not None else {}
            self.class_map.update({v: i for i, v in enumerate(self.class_values) if v != self.background_class})
            
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

        mask = cv2.imread(self.masks_fps[i], 0)
        
        if self.binary:
            mask_remap = np.where(mask == 0, 0, 1).astype(np.uint8)
        else:
            mask_remap = np.full_like(mask, 255 if self.background_class is not None else 0, dtype=np.uint8)
            for class_value, new_value in self.class_map.items():
                mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]
         
        
        image = image.transpose(2, 0, 1)
        
        return image, mask_remap, self.ids[i]

    def __len__(self):
        return len(self.ids)
    
    def get_class_map(self):
        return self.class_map
    
def get_training_augmentation(resize_height = 512, resize_width=512, background=None):
    if background is None:
        background = 0
        
    train_transform = [
        A.Resize(height=resize_height, width=resize_width),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),  
            translate_percent={"x": 0.1, "y": 0.1}, 
            rotate=10, 
            fill=background, 
            fill_mask=background,  
            border_mode=cv2.BORDER_CONSTANT, 
            p=1
        ),
        A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, fill=background, fill_mask=background),
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.1,0.2), p=1),
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(8,32), hole_width_range=(8,32), fill=background, fill_mask=background, p=0.3),
        A.GridDistortion(p=0.2),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(resize_height = 512, resize_width=512, background=None):
    if background is None:
        background = 0
        
    test_transform = [
       A.Resize(height=resize_height, width=resize_width),
       A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, fill=background, fill_mask=background)
    ]
    return A.Compose(test_transform)

def split_dataset(images_dir, labels_dir, output_base, val_size, seed=42):

    if val_size is None:
        print("Validation size not specified. Using default value of 0.2.")
        val_size = 0.2

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    train_img_dir = Path(output_base) / "train" / "images"
    train_lbl_dir = Path(output_base) / "train" / "labels"
    val_img_dir = Path(output_base) / "val" / "images"
    val_lbl_dir = Path(output_base) / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in images_dir.iterdir() if f.is_file()])

    train_files, val_files = train_test_split(
        image_files,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )

    def copy_samples(files, img_dst, lbl_dst):
        for img_file in files:
            label_file = labels_dir / img_file.name

            if not label_file.exists():
                raise FileNotFoundError(
                    f"Label not found for {img_file.name}"
                )

            shutil.copy2(img_file, img_dst / img_file.name)
            shutil.copy2(label_file, lbl_dst / label_file.name)

    copy_samples(train_files, train_img_dir, train_lbl_dir)
    copy_samples(val_files, val_img_dir, val_lbl_dir)

    print(f"Dataset split completed: {int((1 - val_size) * 100)}% training, {int(val_size * 100)}% validation.")