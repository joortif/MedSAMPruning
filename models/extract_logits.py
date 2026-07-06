import os
import torch
from pathlib import Path

from datasets.dataset import split_dataset
from models.medsam.download_medsam import download_medsam, load_medsam
from models.medsam.medsam import process_medsam_batches
from models.smp.train_model import train_pruning_model
from models.smp.unet import process_unet_batches


def extract_logits(images_dir: str, labels_dir: str, model: str, batch_size: int, output_path: str, val_size: float, epochs: int, forgetting: bool = False):

    model = model.lower()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        raise RuntimeError(
            "CUDA is not available. MedSAM and U-Net require a GPU."
        )
        return
    
    device = torch.device(device)

    if model== "medsam":
        medsam_path = download_medsam(output_dir = Path(output_path) / "medsam" )
        
        medsam_model = load_medsam(medsam_path, device=device)

        image_path = Path(images_dir)
        mask_path = Path(labels_dir)

        output_dir = Path(output_path) / "logits"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [p for p in image_path.iterdir() if p.is_file()]

        mask_paths = sorted([str(mask_path / f) for f in os.listdir(mask_path)])

        total_time = process_medsam_batches(
            image_paths=image_paths,
            mask_paths=mask_paths,
            medsam_model=medsam_model,
            batch_size=batch_size,
            save_dir=str(output_dir),
            device = device,
            h5_name=f"logits_{model.lower()}.h5",
        )

        print(f"MedSAM logits saved in {output_dir}")
        print(f"Total processing time: {total_time:.2f} s")
    
    elif model == "unet":

        output_path_split = Path(output_path) / "split"

        split_dataset(images_dir, labels_dir, output_base=output_path_split, val_size=val_size, seed=42)

        unet_model, forgetting_results = train_pruning_model(
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
            forgetting=forgetting
        )

        if not forgetting:
            process_unet_batches(
                model=unet_model.model,
                images_dir=images_dir,
                labels_dir=labels_dir,
                batch_size=batch_size,
                save_dir=os.path.join(output_path, "logits"),
                h5_name=f"logits_{model.lower()}.h5",
                device=device
            )

    else:
        raise ValueError(f"Unsupported model: {model}. Supported models are 'medsam' and 'unet'.")
    
    return forgetting_results if forgetting else None