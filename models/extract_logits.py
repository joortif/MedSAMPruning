import os
from pathlib import Path

from models.medsam.download_medsam import load_medsam
from models.medsam.medsam import process_medsam_batches

def extract_logits(dataset: str, model: str, model_path: str, batch_size: str):

    repo_root = Path(__file__).resolve().parents[1]  
    dataset = dataset.upper()
    model = model.lower()

    if model.lower() == "medsam":
        medsam_model = load_medsam(model_path)

        dataset_path = repo_root / "data" / dataset
        image_path = dataset_path / "images"
        mask_path = dataset_path / "labels"

        output_dir = repo_root / "logits" / dataset / model
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [p for p in image_path.iterdir() if p.is_file()]

        mask_paths = sorted([str(mask_path / f) for f in os.listdir(mask_path)])

        total_time = process_medsam_batches(
            image_paths=image_paths,
            mask_paths=mask_paths,
            medsam_model=medsam_model,
            batch_size=batch_size,
            save_dir=str(output_dir),
            h5_name=f"logits_{dataset.lower()}.h5",
        )

        print(f"MedSAM logits saved in {output_dir}")
        print(f"Total processing time: {total_time:.2f} s")
    
    elif model.lower() == "unet":
        
        
    else:
        raise ValueError(f"Model not supported. Try using MedSAM or UNet.")