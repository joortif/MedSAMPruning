from collections import defaultdict
import os
from time import time
from typing import Optional
import torch
import numpy as np
import re
import torch
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import pandas as pd

from models.smp.exceptions import NoValidAutobatchConfigException
from models.smp.metrics import compute_metrics
from models.smp.utils import autobatch

class SemanticSegmentationModel:

    def __init__(self, classes: np.ndarray, epochs:int, imgsz:int, metrics: np.ndarray, selection_metric: str, model_name:str, model_size:str, output_path:str, 
                 val_fold: Optional[int], fraction:Optional[int]=0.6):
        
        self.classes = classes
        self.n_output_classes = len([cls for cls in self.classes if cls.lower() !="background"])
        self.epochs = epochs
        self.imgsz = imgsz
        self.metrics = metrics
        self.selection_metric = selection_metric
        self.model_name = model_name
        self.model_size = model_size
        self.output_path = output_path
        self.fraction = fraction
        self.val_fold = str(val_fold)

        self.lr = 2e-4

    def save_model(self, output_dir, weights_only=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        model_save_name = f"{self.model_name}-{self.model_size}-ep{self.epochs}-fold{self.val_fold}.pt"
        output_path = os.path.join(output_dir,model_save_name)

        if weights_only:
            torch.save(self.model.state_dict(), output_path)
            # print(f"Model weights saved in {output_path}")
        else:
            torch.save(self.model, output_path)
            # print(f"Complete model saved in {output_dir}")
        
        return output_path
    
    def show_metrics(self, metrics, stage):
    
        print(f"{stage} metrics:\n")

        general_metrics = {k: v for k, v in metrics.items() if '_class_' not in k}

        print(f"{'Metric':<20} {'Value':>8}")
        print("-" * 30)
        for metric, value in general_metrics.items():
            print(f"{metric:<20} {value:>8.4f}")

        print("\nMetrics by Class:\n")

        class_pattern = re.compile(r'(.+)_class_(.+)')

        class_metrics = {}

        for key, value in metrics.items():
            match = class_pattern.match(key)
            if match:
                metric_name = match.group(1)
                class_idx = match.group(2)
                class_metrics.setdefault(metric_name, {})[class_idx] = value

        all_classes = sorted(set(idx for metric_dict in class_metrics.values() for idx in metric_dict))

        for metric_name, class_dict in class_metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}:")
            print("-" * 30)
            for c in all_classes:
                val = class_dict.get(c, None)
                if val is not None:
                    print(f"Class {c:<2} : {val:>8.4f}")
                else:
                    print(f"Class {c:<2} : {'N/A':>8}")
            print()


    def autobatch_imgsz(self):
        device = next(self.model.parameters()).device

        if device.type == "cpu" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            
        try:
            self.batch = autobatch(model=self.model, imgsz=self.imgsz, fraction=self.fraction)
        except NoValidAutobatchConfigException as e:
            print(f"Autobatch failed: {e}")
        
        if self.batch < 16:
            self.lr = 2e-5
            print(f"Reducing learning rate to {self.lr}")
        
    def run_training():
        raise NotImplementedError("Subclasses must implement this method") 

    def save_metrics():
        raise NotImplementedError("Subclasses must implement this method")
    


class Model(pl.LightningModule, SemanticSegmentationModel):
    def __init__(self, in_channels: int , classes: int, metrics: np.ndarray, imgsz:int, selection_metric: str, epochs:int, t_max: Optional[int], output_path:str, 
                 fraction:Optional[int], val_fold: Optional[int]=0, model_name: str="unet", encoder_name: str="resnet34", batch_size: Optional[int]=32, **kwargs):
        
        super().__init__()

        SemanticSegmentationModel.__init__(self, classes=classes, epochs=epochs, imgsz=imgsz, metrics=metrics, 
                                           selection_metric=selection_metric, 
                                           model_name=model_name, model_size=encoder_name, output_path=output_path, fraction=fraction, val_fold=val_fold)
        self.model_name = model_name.replace("-", "")
        self.in_channels = in_channels

        self.model = smp.create_model(
            arch=self.model_name,
            encoder_name=self.model_size,
            in_channels=self.in_channels,
            classes=self.n_output_classes,
            **kwargs,
        )

        self.t_max = t_max
        
        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(self.model_size)
        self.number_of_classes = self.n_output_classes
        self.binary = self.n_output_classes == 1
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.ignore_index = None
        self.val_fold = val_fold
        self.batch = batch_size
        
        self.prev_correct_maps_train = {}
        self.pixel_forgetting_train = {}
        self.image_forgetting_train = defaultdict(int)

        self.prev_correct_maps_val = {}
        self.pixel_forgetting_val = {}
        self.image_forgetting_val = defaultdict(int)
        
        self.forgetting_update_time = 0.0
        self.forgetting_update_calls = 0
        self.fit_total_time = 0.0
        self.forgetting_post_time = 0.0

        if self.binary:
            self.loss_mode = smp.losses.BINARY_MODE
        else:
            self.loss_mode = smp.losses.MULTICLASS_MODE
            self.ignore_index = 255
            
        self.loss_fn = smp.losses.DiceLoss(self.loss_mode, from_logits=True, ignore_index=self.ignore_index)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        if self.batch is None:
            self.autobatch_imgsz()
        else:
            print(f"Using defined batch size of {self.batch}")

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
    
    def validate_segmentation_batch(self, image, mask):

        assert image.ndim == 4, f"Expected image ndim=4, got {image.ndim}" # [batch_size, channels, H, W]
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f"Image dimensions must be divisible by 32, got {h}x{w}"

        if self.binary: 
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            assert mask.ndim == 4, f"Expected binary mask ndim=4, got {mask.ndim}"
            assert mask.max() <= 1.0 and mask.min() >= 0.0, "Binary mask values must be in range [0, 1]"
        else:
            assert mask.ndim == 3, f"Expected multiclass mask ndim=3, got {mask.ndim}"
            mask = mask.long()
        
        return image, mask

    def shared_step(self, batch, stage):
        image, mask, sample_ids = batch

        image, mask = self.validate_segmentation_batch(image, mask)
        
        logits_mask = self.forward(image)

        logits_mask = logits_mask.contiguous()

        loss = self.loss_fn(logits_mask, mask)
        
        self.log(f"loss_{stage}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.size(0))

        if self.binary:
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            valid = torch.ones_like(pred_mask, dtype=torch.bool)
        
        else:
            prob_mask = logits_mask.softmax(dim=1)
            pred_mask = prob_mask.argmax(dim=1)
            valid = mask != self.ignore_index
            
        correct_map = (pred_mask == mask) & valid
        
        if self.binary:
            metric_args = {"mode": "binary"}
        else:
            metric_args = {"mode": "multiclass", "num_classes": self.number_of_classes, "ignore_index": self.ignore_index}

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), **metric_args)
                
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "sample_ids": sample_ids,
            "correct_map": correct_map.detach(),
        }

    def shared_epoch_end(self, outputs, stage):
        results = compute_metrics(outputs, self.metrics, self.classes, stage)
        results = {f"{k}_{stage}": v for k, v in results.items()}
        self.log_dict(results, prog_bar=True)

    def training_step(self, batch):
        train_loss_info = self.shared_step(batch, "train")
        self._update_forgetting(train_loss_info["sample_ids"], train_loss_info["correct_map"], split="train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()
        super().on_train_epoch_end()

    def validation_step(self, batch):
        valid_loss_info = self.shared_step(batch, "valid")
        self._update_forgetting(valid_loss_info["sample_ids"], valid_loss_info["correct_map"], split="valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def save_metrics(self, metrics, experiment_name, filename, training_time=None):
        if not metrics:
            print("No metrics to save.")
            return metrics

        metrics_dict = metrics[0]  
        
        keys_to_remove = [k for k in metrics_dict.keys() if "loss" in k.lower()]
        for k in keys_to_remove:
            metrics_dict.pop(k, None)

        df = pd.DataFrame([metrics_dict])  
        df.insert(0, "Experiment", experiment_name)  
        
        if training_time is not None:
            df["Training Time (min)"] = round(training_time / 60.0, 2)

        if os.path.exists(filename):
            df_existing = pd.read_csv(filename, sep=';')
            df_combined = pd.concat([df_existing, df], ignore_index=True)
        else:
            df_combined = df

        file_output_path = os.path.join(os.path.dirname(self.output_path), filename)
        if os.path.exists(file_output_path):
            df.to_csv(file_output_path, sep=';', mode='a', header=False, index=False)
        else:
            df.to_csv(file_output_path, sep=';', index=False)

        self.show_metrics(metrics_dict, "Test")  

        print(f"Metrics saved in file {file_output_path}")

        evaluation_metric = metrics_dict.get(f"{self.selection_metric}_test")
        return evaluation_metric
    
    def fit_model(self, train_loader, val_loader=None):
        self.val_loader_for_viz = val_loader

        self.trainer = pl.Trainer(
            max_epochs=self.epochs,
            log_every_n_steps=1
        )
        
        start_time = time.time()
        self.trainer.fit(self, train_dataloaders=train_loader, val_dataloaders = val_loader)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time / 60:.2f} minutes")        
        
        return total_time
    
    def validate(self, valid_loader):
        trainer = pl.Trainer(logger=False, enable_checkpointing=False)  
        valid_metrics = trainer.validate(self, dataloaders=valid_loader, verbose=False)
        return valid_metrics
        
    def test(self, test_loader):
        trainer = pl.Trainer(logger=False, enable_checkpointing=False)  
        test_metrics = trainer.test(self, dataloaders=test_loader, verbose=True)
        return test_metrics
    
    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)
    
    def save_pretrained(self, output_path):
        output_dir = os.path.join(output_path, 'pretrained_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        self.model.save_pretrained(output_dir)