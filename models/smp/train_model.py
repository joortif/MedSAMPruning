import os

from torch.utils.data import DataLoader


from datasets.dataset import Dataset, get_training_augmentation, get_validation_augmentation
from models.smp.unet import Model
from models.smp.utils import calculate_closest_resize, image_sizes


def train_pruning_model(
    full_img_dir,
    data_img_dir,
    metrics=[],
    model_name="unet",
    encoder_name="resnet34",
    n_input_channels=3,
    n_output_classes=1,
    classes=["background", "foreground"],
    epochs=None,
    num_workers=4,
    background=None,
    fraction=0.7,
    output_path=None,
    batch_size=None,
    forgetting=False
):
    if epochs is None:
        print("No epochs specified. Using default value of 10.")
        epochs = 10

    mode_h, mode_w = image_sizes(full_img_dir)

    imgsz = calculate_closest_resize(mode_h, mode_w)
    print(f"Using imgsz={imgsz}")

    x_train_dir = os.path.join(data_img_dir, "train", "images")
    y_train_dir = os.path.join(data_img_dir, "train", "labels")

    x_val_dir = os.path.join(data_img_dir, "val", "images")
    y_val_dir = os.path.join(data_img_dir, "val", "labels")
        
    valid_dataset = Dataset(
        x_val_dir,
        y_val_dir,
        classes=classes,
        augmentation=get_validation_augmentation(imgsz, imgsz, background),
        background=background
    )

    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=classes,
        augmentation=get_validation_augmentation(imgsz, imgsz, background) if forgetting else get_training_augmentation(imgsz, imgsz, background),
        background=background
    )

    model = Model(
        model_name=model_name,
        encoder_name=encoder_name,
        in_channels=n_input_channels,
        metrics=metrics,
        selection_metric="dice",
        classes=classes,
        output_path=output_path,
        t_max=None,
        epochs=epochs,
        imgsz=imgsz,
        fraction=fraction,
        batch_size=batch_size,
        forgetting = forgetting
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=model.batch,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=model.batch,
        shuffle=False,
        num_workers=num_workers
    )

    model.t_max = epochs * len(train_loader)

    fit_total_time, forg_results = model.fit_model(
        train_loader=train_loader,
        val_loader=valid_loader
    )

    """forget_total = model.forgetting_update_time
    forget_mean = forget_total / max(model.forgetting_update_calls, 1)

    rest_time = fit_total_time - forget_total

    print("\n========== TIME STATS ==========")
    print(f"FIT TOTAL TIME (train + val + forgetting): {fit_total_time:.4f} s")
    print(f"FIT TOTAL TIME: {fit_total_time / 60:.4f} min")

    print("\n--- FORGETTING ---")
    print(f"Forgetting total time: {forget_total:.4f} s")
    print(f"Forgetting mean time per call: {forget_mean:.6f} s")
    print(f"Forgetting calls: {model.forgetting_update_calls}")

    print("\n--- TRAIN + VALIDATION ---")
    print(f"Training + validation time: {rest_time:.4f} s")
    print(f"Training + validation time: {rest_time / 60:.4f} min")

    if fit_total_time > 0:
        forget_pct = (forget_total / fit_total_time) * 100
        print(f"Forgetting percentage over total: {forget_pct:.2f}%")

    print("================================\n")"""

    return model, forg_results if forg_results is not None else None