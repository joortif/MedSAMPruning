import cv2
import numpy as np
import os

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