from pathlib import Path
import random
import numpy as np
import os

from models.extract_logits import extract_logits
from pruning.selection import save_pruned_samples, select_samples
from pruning_metrics.binary_entropy import compute_entropy
from pruning_metrics.cb_scs import cb_scs
from pruning_metrics.el2n import compute_el2n
from pruning_metrics.forgetting import compute_forgetting_scores
from pruning_metrics.fusion import fuse_cb_el2n
from pruning_metrics.iou import compute_iou_ranking
from pruning_metrics.utils import compute_ranking_scores

METRICS = {
    "el2n": compute_el2n,
    "entropy": compute_entropy,
}


def prune_dataset(
    dataset_path: str,
    pruning_model: str,
    pruning_rate: float,
    strategy: str,
    selection: str,
    output_path: str,
    seed: int,
    logits_path: str = None,
    batch_size: int = None,
    epochs: int = None,
    val_size: float = None,
    alpha: float = None
):
    """
    Prune a dataset based on the specified parameters.

    Args:
        dataset_path (str): Path to the dataset.
        pruning_model (str): Model to use for pruning ('medsam' or 'unet').
        save_logits (bool): Whether to save logits during pruning.
        pruning_rate (float): Rate at which to prune the dataset.
        strategy (str): Strategy to use for pruning.
        selection (str): Selection method ('easy', 'hard', or 'mixed').
        seed (int): Random seed for reproducibility.
        logits_path (str): Path to the logits file (if applicable).
        batch_size (int): Batch size for processing.
        epochs (int): Number of epochs for training the pruning model.
        val_size (float): Proportion of the dataset to use for validation.
        alpha (float): Weighting factor for the CB-SCS+EL2N fusion pruning strategy.

    Returns:
        None
    """
    # Set random seed for reproducibility

    random.seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    samples_ranking = None

    csv_strategy_path = Path(output_path) / "ranking" / f"{strategy}.csv"

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Images or labels directory not found in {dataset_path}. Subdirectories 'images' and 'labels' are required.")
    
    # First step: Extract logits and save them using MedSAM or UNet model. 
    # If the strategy is "forgetting" or "cb-scs", we don't need to save logits in either case. 
    # With fusion, we calculate the cb-scs scores and then the el2n scores using the logits extracted with the specified model.

    if strategy not in ["forgetting", "cb-scs", "fusion"]:
        if logits_path is None:
            print(f"Extracting logits using {pruning_model} model...")
            extract_logits(images_dir=images_dir, labels_dir=labels_dir, model=pruning_model, batch_size=batch_size, output_path = output_path, 
                        val_size=val_size, epochs=epochs, forgetting=False)
    
    elif strategy == "cb-scs":
        print(f"Calculating cb-scs scores using ground truth labels...")
        samples_ranking = cb_scs(mask_path=labels_dir, out_csv_path=csv_strategy_path)

    elif strategy == "forgetting":
        print(f"Extracting logits and calculating scores using {pruning_model} model for forgetting strategy...")
        samples_ranking = compute_forgetting_scores(images_dir=images_dir, labels_dir=labels_dir, model=pruning_model, csv_path=csv_strategy_path, 
                                                    output_path=output_path, batch_size=batch_size, val_size=val_size, epochs=epochs, seed=seed)
    elif strategy == "fusion":
        print(f"Calculating cb-scs and el2n scores for fusion strategy...")
        samples_ranking_cb_scs = cb_scs(mask_path=labels_dir, output_path=csv_strategy_path)
        
        if logits_path is None:
            print(f"Extracting logits using {pruning_model} model...")
            extract_logits(images_dir=images_dir, labels_dir=labels_dir, model=pruning_model, batch_size=batch_size, output_path = output_path, 
                        val_size=val_size, epochs=epochs, forgetting=False)
            
            logits_path = Path(output_path) / "logits" / f"logits_{pruning_model.lower()}.h5"
            
        samples_ranking_el2n = compute_ranking_scores(image_path = images_dir, mask_path=labels_dir, logits_path=logits_path,
                                                     metric_fn=METRICS["el2n"], metric_name="el2n", csv_path=csv_strategy_path)
        
        samples_ranking = fuse_cb_el2n(samples_ranking_cb_scs, samples_ranking_el2n, alpha=alpha, output_path=csv_strategy_path)

    # Second step: Calculate the ranking scores based on the specified strategy.
    # If the strategy is "forgetting", "cb-scs" or "fusion", we already have the scores calculated in the previous step.

    if samples_ranking is None:
        if strategy in ["el2n", "entropy"]:
            print(f"Calculating {strategy} scores using logits...")
            samples_ranking = compute_ranking_scores(image_path = images_dir, mask_path=labels_dir, logits_path=logits_path,
                                                     metric_fn=METRICS[strategy], metric_name=strategy, csv_path=csv_strategy_path)

        if strategy == "iou":
            print(f"Calculating IoU scores using logits and ground truth labels...")
            samples_ranking = compute_iou_ranking(image_path=images_dir, mask_path=labels_dir, logits_path=logits_path, 
                                                  csv_path=csv_strategy_path)

    # Third step: Prune the dataset based on the calculated score, the specified pruning rate and the selection method.
    print(f"Pruning dataset using {strategy} strategy with the '{selection}' selection method and pruning rate of {pruning_rate}...")
    reduced_subset = select_samples(scores_dict=samples_ranking, pruning_rate=pruning_rate, strategy=selection, difficulty_high_score=False if strategy == "iou" else True)

    # Save the pruned dataset
    print(f"Saving pruned dataset to {output_path}...")
    save_pruned_samples(selected_ids=reduced_subset, images_dir=images_dir, labels_dir=labels_dir, output_dir=Path(output_path) / "pruned_dataset")