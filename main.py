import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import yaml

from datasets.download_datasets import download_dataset
from pruning.pruning import prune_dataset

pruning_strategies = ["iou","cb-scs","el2n","entropy","forgetting","fusion"]

def float_0_1(value):
    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not in the range [0, 1]")
    
    return value

def pruning_rate_type(value):
    value = float(value)

    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not in the range (0, 1)")
    
    return value

def build_parser():

    parser = argparse.ArgumentParser(
        description="Semantic segmentation dataset pruning"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download MMOTU or BUSI datasets."
    )

    download_parser.add_argument(
        "--dataset",
        choices=["BUSI", "MMOTU"],
        required=True
    )

    download_parser.add_argument(
        "--data-dir",
        required=True
    )

    pruning_parser = subparsers.add_parser(
        "prune",
        help="Generate pruned dataset split."
    )

    pruning_parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )

    pruning_parser.add_argument(
        "--dataset-path",
    )

    pruning_parser.add_argument(
        "--pruning-model",
        choices=["medsam", "unet"]
    )

    pruning_parser.add_argument(
        "--pruning-rate",
        type=pruning_rate_type,
    )

    pruning_parser.add_argument(
        "--strategy",
        choices=pruning_strategies
    )

    pruning_parser.add_argument(
        "--selection",
        choices=[
            "easy",
            "hard",
            "mixed",
        ],
    )

    pruning_parser.add_argument(
        "--logits-path",
        type=str,
    )

    pruning_parser.add_argument(
        "--batch-size",
        type=int
    )

    pruning_parser.add_argument(
        "--output-path",
        type=str
    )

    pruning_parser.add_argument(
        "--seed",
        default=123,
        type=int,
    )

    pruning_parser.add_argument(
        "--epochs",
        default=None,
        type=int,
    )

    pruning_parser.add_argument(
        "--val-size",
        default=None,
        type=float_0_1
    )

    pruning_parser.add_argument(
        "--alpha",
        default=None,
        type=float_0_1
    )

    return parser


def main():

    config = {}

    parser = build_parser()

    args_config, _ = parser.parse_known_args()

    if getattr(args_config, "config", None):
        with open(args_config.config, "r") as f:
            config = yaml.safe_load(f) or {}

        config = {
            k.replace("-", "_"): v
            for k, v in config.items()
        }

    args = parser.parse_args()

    for key, value in config.items():

        if hasattr(args, key):

            current_value = getattr(args, key)

            if current_value is None:
                setattr(args, key, value)

    if args.command == "download":

        download_dataset(
            dataset_name=args.dataset,
            outdir=args.data_dir,
        )

        return

    if args.command == "prune":

        required_args = [
            "dataset_path",
            "pruning_rate",
            "strategy",
            "selection",
            "output_path",
        ]

        for arg in required_args:
            if getattr(args, arg) is None:
                parser.error(
                    f"--{arg.replace('_', '-')} is required"
                )

        if args.strategy != "fusion" and args.alpha is not None:
            parser.error(
                "--alpha can only be used with strategy=fusion"
            )

        prune_kwargs = dict(
            dataset_path=args.dataset_path,
            pruning_model=args.pruning_model,
            pruning_rate=args.pruning_rate,
            strategy=args.strategy,
            selection=args.selection,
            seed=args.seed,
            batch_size=args.batch_size,
            output_path=args.output_path,
            alpha=args.alpha,
            logits_path=args.logits_path,
        )

        if args.pruning_model == "unet":

            prune_kwargs.update(
                epochs=args.epochs,
                val_size=args.val_size,
            )

        prune_dataset(**prune_kwargs)

        return

    raise ValueError(f"Unknown command {args.command}")

if __name__ == "__main__": 
    main()