import os
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.dataset import CSAMDataset
from training.train import train_pipeline
from training.validate import evaluate_model
from training.loss import CSAMLoss
from training.checkpoint import load_checkpoint


# ============================================================
# BUILD DATASETS AND DATALOADERS
# ============================================================

def build_datasets_and_loaders(
    train_csv,
    val_csv,
    test_csv,
    image_dir,
    batch_size=4,
    precompute_logic=True
):

    print("\n" + "=" * 70)
    print("BUILDING DATASETS")
    print("=" * 70)

    print(f"[INFO] Train CSV : {train_csv}")
    print(f"[INFO] Val CSV   : {val_csv}")
    print(f"[INFO] Test CSV  : {test_csv}")
    print(f"[INFO] Image dir : {image_dir}")

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "-" * 70)
    print("LOADING TRAIN DATASET")
    print("-" * 70)

    train_dataset = CSAMDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        precompute_logic=precompute_logic
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "-" * 70)
    print("LOADING VALIDATION DATASET")
    print("-" * 70)

    val_dataset = CSAMDataset(
        csv_file=val_csv,
        image_dir=image_dir,
        precompute_logic=precompute_logic
    )

    # ========================================================
    # TEST
    # ========================================================

    print("\n" + "-" * 70)
    print("LOADING TEST DATASET")
    print("-" * 70)

    test_dataset = CSAMDataset(
        csv_file=test_csv,
        image_dir=image_dir,
        precompute_logic=precompute_logic
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\n[OK] DataLoaders created")

    print(
        f"[OK] Train batches : {len(train_loader)}"
    )

    print(
        f"[OK] Val batches   : {len(val_loader)}"
    )

    print(
        f"[OK] Test batches  : {len(test_loader)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train CSAM Detection Model"
    )

    # ========================================================
    # DATASET PATHS
    # ========================================================

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset_updated_organized",
        help="Root directory containing train/val/test"
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default=(
            "dataset_updated_organized/"
            "train/metadata.csv"
        ),
        help="Path to training metadata CSV"
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        default=(
            "dataset_updated_organized/"
            "val/metadata.csv"
        ),
        help="Path to validation metadata CSV"
    )

    parser.add_argument(
        "--test_csv",
        type=str,
        default=(
            "dataset_updated_organized/"
            "test/metadata.csv"
        ),
        help="Path to test metadata CSV"
    )

    # ========================================================
    # TRAINING PARAMETERS
    # ========================================================

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--csam_weight",
        type=float,
        default=2.0,
        help="CSAM class loss weight"
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )

    # ========================================================
    # ABLATION MODE
    # ========================================================

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "full",
            "image_only",
            "text_only",
            "no_logic",
            "no_cross_attention",
            "no_fusion"
        ],
        help="Ablation mode"
    )

    # ========================================================
    # LOGIC PRECOMPUTATION
    # ========================================================

    parser.add_argument(
        "--no_precompute_logic",
        action="store_true",
        help="Disable logic vector precomputation"
    )

    # ========================================================
    # PARSE ARGUMENTS
    # ========================================================

    args = parser.parse_args()

    if not os.path.exists(args.train_csv):
        args.train_csv = "5_fold_splits/fold_1/train.csv" if os.path.exists("5_fold_splits/fold_1/train.csv") else os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(args.val_csv):
        args.val_csv = "5_fold_splits/fold_1/test.csv" if os.path.exists("5_fold_splits/fold_1/test.csv") else os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(args.test_csv):
        args.test_csv = "5_fold_splits/fold_1/test.csv" if os.path.exists("5_fold_splits/fold_1/test.csv") else os.path.join(args.data_dir, "metadata.csv")

    # ========================================================
    # CREATE CHECKPOINT DIRECTORY
    # ========================================================

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("\n" + "=" * 70)
    print("CSAM TRAINING PIPELINE")
    print("=" * 70)

    print(
        f"Image Directory : {args.data_dir}"
    )

    print(
        f"Train CSV       : {args.train_csv}"
    )

    print(
        f"Val CSV         : {args.val_csv}"
    )

    print(
        f"Test CSV        : {args.test_csv}"
    )

    print(
        f"Batch Size      : {args.batch_size}"
    )

    print(
        f"Epochs          : {args.epochs}"
    )

    print(
        f"Learning Rate   : {args.lr}"
    )

    print(
        f"CSAM Weight     : {args.csam_weight}"
    )

    print(
        f"Mode            : {args.mode}"
    )

    print(
        f"Save Directory  : {args.save_dir}"
    )

    print("=" * 70)

    # ========================================================
    # VERIFY PATHS BEFORE LOADING
    # ========================================================

    print("\n" + "=" * 70)
    print("VERIFYING DATASET PATHS")
    print("=" * 70)

    paths_to_check = {
        "Dataset root": args.data_dir,
        "Train CSV": args.train_csv,
        "Val CSV": args.val_csv,
        "Test CSV": args.test_csv
    }

    all_paths_valid = True

    for name, path in paths_to_check.items():

        if os.path.exists(path):

            print(
                f"[OK] {name}: {path}"
            )

        else:

            print(
                f"[ERROR] {name} NOT FOUND: {path}"
            )

            all_paths_valid = False

    if not all_paths_valid:

        raise FileNotFoundError(
            "\nOne or more dataset paths are invalid. "
            "Please check the paths above."
        )

    # ========================================================
    # BUILD DATASETS
    # ========================================================

    precompute_logic = not args.no_precompute_logic

    (
        train_loader,
        val_loader,
        test_loader
    ) = build_datasets_and_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        image_dir=args.data_dir,
        batch_size=args.batch_size,
        precompute_logic=precompute_logic
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n" + "=" * 70)
    print("DEVICE")
    print("=" * 70)

    print(
        f"[INFO] Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"[INFO] CUDA version: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "[WARNING] CUDA is not available. "
            "Training will run on CPU."
        )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    model = train_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        csam_weight=args.csam_weight,
        save_dir=args.save_dir,
        device=device,
        mode=args.mode
    )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("=" * 70)

    best_path = os.path.join(
        args.save_dir,
        "best_model.pt"
    )

    if os.path.exists(best_path):

        print(
            f"[OK] Best checkpoint found: "
            f"{best_path}"
        )

        load_checkpoint(
            model,
            checkpoint_path=best_path,
            device=device
        )

    else:

        print(
            "[WARNING] best_model.pt not found."
        )

        print(
            "[WARNING] Evaluating current model."
        )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = CSAMLoss(
        csam_weight=args.csam_weight
    )

    # ========================================================
    # TEST EVALUATION
    # ========================================================

    test_metrics, _, _ = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )

    # ========================================================
    # TEST RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST SET METRICS REPORT")
    print("=" * 70)

    print(
        f"Test Loss   : "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Accuracy    : "
        f"{test_metrics['accuracy']:.4f} "
        f"({test_metrics['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Precision   : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1-Score    : "
        f"{test_metrics['f1_score']:.4f}"
    )

    print(
        f"Confusion   : "
        f"TP={test_metrics['tp']}, "
        f"FP={test_metrics['fp']}, "
        f"TN={test_metrics['tn']}, "
        f"FN={test_metrics['fn']}"
    )

    print("=" * 70)

    print("\n[OK] Training and evaluation completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()