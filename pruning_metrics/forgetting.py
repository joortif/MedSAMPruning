import os
from pathlib import Path

from datasets.dataset import split_dataset
from models.smp.train_model import train_pruning_model
from models.smp.unet import process_unet_batches
from pruning_metrics.forgetting_seq_tta import compute_forgetting_seq_tta_rankings

def compute_forgetting_scores(images_dir: str, labels_dir: str, model: str, csv_path: str, output_path: str, batch_size: int, val_size: float, epochs: int, seed: int):

    model = model.lower()

    if model.lower() == "medsam":
        
        image_path = Path(images_dir)
        mask_path = Path(labels_dir)

        forgetting_results = compute_forgetting_seq_tta_rankings(image_path=image_path, mask_path=mask_path, csv_path=csv_path, model_path=output_path)

    elif model.lower() == "unet":

        output_path_split = Path(output_path) / "split"

        split_dataset(images_dir, labels_dir, output_base=output_path_split, val_size=val_size, seed=seed)

        _, forgetting_results = train_pruning_model(
            full_img_dir=images_dir,
            data_img_dir=output_path_split,
            model_name="unet",
            encoder_name="resnet34",
            n_input_channels=3,
            n_output_classes=1,
            classes=["background", "foreground"],
            epochs=epochs,
            num_workers=4,
            background=None,
            fraction=0.7,
            output_path=os.path.join(output_path, "logits", model),
            batch_size=batch_size,
            forgetting=True
        )

    else:
        raise ValueError(f"Unsupported model: {model}. Supported models are 'medsam' and 'unet'.")
    
    return forgetting_results 