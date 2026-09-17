import os
import gc
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from dataset.dataset import CSAMDataset
from training.train import train_pipeline
from training.validate import evaluate_model
from training.loss import CSAMLoss
from training.checkpoint import load_checkpoint
from metrics_utils import plot_and_save_confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

SPLIT_DIR = "5_fold_splits"

IMAGE_DIR = "dataset_updated_organized"

BATCH_SIZE = 4

EPOCHS = 3

LR = 1e-4

CSAM_WEIGHT = 2.0

N_FOLDS = 5

# Internal validation taken only from the 4 training folds
INTERNAL_VAL_SIZE = 0.10

SAVE_DIR = "cv_checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD ALL CAPTIONS
# ============================================================

def load_all_captions():

    caption_map = {}

    print("\n" + "=" * 70)
    print("LOADING ALL CAPTIONS")
    print("=" * 70)

    caption_files = [

        os.path.join(
            IMAGE_DIR,
            "csam_non_sens.csv"
        ),

        os.path.join(
            IMAGE_DIR,
            "non_csam.csv"
        ),

        os.path.join(
            IMAGE_DIR,
            "train",
            "train_csam_non_sens.csv"
        ),

        os.path.join(
            IMAGE_DIR,
            "train",
            "train_non_csam.csv"
        ),

        os.path.join(
            IMAGE_DIR,
            "val",
            "val_csam_non_sens.csv"
        ),

        os.path.join(
            IMAGE_DIR,
            "val",
            "val_non_csam.csv"
        ),

        os.path.join(
            IMAGE_DIR,
            "test",
            "test_csam_non_sens.csv"
        ),

        os.path.join(
            IMAGE_DIR,
            "test",
            "test_non_csam.csv"
        )
    ]

    total_rows = 0

    for caption_file in caption_files:

        if not os.path.exists(caption_file):

            print(
                f"[WARNING] Missing caption file:"
            )

            print(
                f"          {caption_file}"
            )

            continue

        df = pd.read_csv(
            caption_file
        )

        if (
            "filename" not in df.columns
            or "caption" not in df.columns
        ):

            print(
                f"[WARNING] Invalid caption file:"
            )

            print(
                f"          {caption_file}"
            )

            continue

        print(
            f"[OK] {caption_file} "
            f"-> {len(df)} rows"
        )

        total_rows += len(df)

        for _, row in df.iterrows():

            filename = (
                str(row["filename"])
                .strip()
                .lower()
            )

            caption = (
                str(row["caption"])
                .strip()
            )

            if filename not in caption_map:

                caption_map[filename] = caption

    print()

    print(
        f"[OK] Total caption rows read: "
        f"{total_rows}"
    )

    print(
        f"[OK] Unique captions loaded: "
        f"{len(caption_map)}"
    )

    return caption_map


# ============================================================
# PREPARE FOLD CSV
# ============================================================

def prepare_csv(
    source_csv,
    output_csv,
    caption_map
):

    df = pd.read_csv(
        source_csv
    )

    # ========================================================
    # IMAGE COLUMN
    # ========================================================

    if "image name" in df.columns:

        df["image"] = (
            df["image name"]
            .astype(str)
            .str.strip()
        )

    elif "image" in df.columns:

        df["image"] = (
            df["image"]
            .astype(str)
            .str.strip()
        )

    else:

        raise ValueError(
            f"No image column found in:\n"
            f"{source_csv}"
        )


    # ========================================================
    # LABEL
    # ========================================================

    if "label" not in df.columns:

        raise ValueError(
            f"No label column found in:\n"
            f"{source_csv}"
        )


    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # ========================================================
    # CAPTIONS
    # ========================================================

    df["text"] = (

        df["image"]

        .astype(str)

        .str.strip()

        .str.lower()

        .map(
            caption_map
        )

        .fillna("")

        .astype(str)

    )


    # ========================================================
    # FILEPATH
    # ========================================================

    if "filepath" in df.columns:

        df["filepath"] = (

            df["filepath"]

            .astype(str)

            .str.strip()

        )


    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        os.path.dirname(
            output_csv
        ),
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False
    )

    return df


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(
    train_df,
    test_df,
    fold
):

    # ========================================================
    # SAMPLE COUNT
    # ========================================================

    expected_total = (
        len(train_df)
        +
        len(test_df)
    )

    print(
        f"Total samples in fold {fold}: "
        f"{expected_total}"
    )


    # ========================================================
    # GROUP LEAKAGE
    # ========================================================

    if (
        "group_id" in train_df.columns
        and
        "group_id" in test_df.columns
    ):

        train_groups = set(
            train_df[
                "group_id"
            ]
        )

        test_groups = set(
            test_df[
                "group_id"
            ]
        )

        overlap = (
            train_groups &
            test_groups
        )

        print(
            f"Group overlap : "
            f"{len(overlap)}"
        )

        if overlap:

            raise RuntimeError(

                f"GROUP LEAKAGE FOUND "
                f"IN FOLD {fold}: "
                f"{len(overlap)} overlapping groups"

            )

    else:

        print(
            "[WARNING] group_id not available."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "5-FOLD CROSS VALIDATION - FIXED SHARED SPLITS"
    )
    print("=" * 70)

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Split directory     : {SPLIT_DIR}"
    )

    print(
        f"Folds               : {N_FOLDS}"
    )

    print(
        f"Epochs              : {EPOCHS}"
    )

    print(
        f"Batch Size          : {BATCH_SIZE}"
    )

    print(
        f"Internal Val Size   : "
        f"{INTERNAL_VAL_SIZE * 100:.0f}%"
    )

    print("=" * 70)


    # ========================================================
    # VERIFY DIRECTORIES
    # ========================================================

    if not os.path.exists(
        SPLIT_DIR
    ):

        raise FileNotFoundError(
            f"Split directory not found:\n"
            f"{SPLIT_DIR}"
        )


    if not os.path.exists(
        IMAGE_DIR
    ):

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{IMAGE_DIR}"
        )


    # ========================================================
    # CREATE SAVE DIRECTORY
    # ========================================================

    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )


    # ========================================================
    # LOAD CAPTIONS
    # ========================================================

    caption_map = load_all_captions()


    # ========================================================
    # FOLD RESULTS
    # ========================================================

    fold_results = []
    all_cv_preds = []
    all_cv_targets = []


    # ========================================================
    # 5 FOLDS
    # ========================================================

    for fold in range(
        1,
        N_FOLDS + 1
    ):

        print("\n\n")

        print("=" * 70)
        print(
            f"FOLD {fold}/{N_FOLDS}"
        )
        print("=" * 70)


        # ====================================================
        # SHARED SPLIT PATHS
        # ====================================================

        fold_input_dir = os.path.join(
            SPLIT_DIR,
            f"fold_{fold}"
        )

        source_train_csv = os.path.join(
            fold_input_dir,
            "train.csv"
        )

        source_test_csv = os.path.join(
            fold_input_dir,
            "test.csv"
        )


        if not os.path.exists(
            source_train_csv
        ):

            raise FileNotFoundError(
                f"Missing:\n"
                f"{source_train_csv}"
            )


        if not os.path.exists(
            source_test_csv
        ):

            raise FileNotFoundError(
                f"Missing:\n"
                f"{source_test_csv}"
            )


        # ====================================================
        # READ SHARED SPLITS
        # ====================================================

        source_train_df = pd.read_csv(
            source_train_csv
        )

        source_test_df = pd.read_csv(
            source_test_csv
        )


        print(
            f"Original train samples : "
            f"{len(source_train_df)}"
        )

        print(
            f"Original test samples  : "
            f"{len(source_test_df)}"
        )


        # ====================================================
        # VERIFY SHARED SPLIT
        # ====================================================

        verify_split(
            source_train_df,
            source_test_df,
            fold
        )


        # ====================================================
        # FOLD SAVE DIRECTORY
        # ====================================================

        fold_dir = os.path.join(
            SAVE_DIR,
            f"fold_{fold}"
        )

        os.makedirs(
            fold_dir,
            exist_ok=True
        )


        prepared_dir = os.path.join(
            fold_dir,
            "prepared_data"
        )

        os.makedirs(
            prepared_dir,
            exist_ok=True
        )


        prepared_train_csv = os.path.join(
            prepared_dir,
            "train.csv"
        )

        prepared_test_csv = os.path.join(
            prepared_dir,
            "test.csv"
        )


        # ====================================================
        # PREPARE TRAIN CSV
        # ====================================================

        print(
            "\nPreparing training CSV..."
        )

        train_df = prepare_csv(
            source_train_csv,
            prepared_train_csv,
            caption_map
        )


        # ====================================================
        # PREPARE TEST CSV
        # ====================================================

        print(
            "Preparing test CSV..."
        )

        test_df = prepare_csv(
            source_test_csv,
            prepared_test_csv,
            caption_map
        )


        # ====================================================
        # CAPTION STATISTICS
        # ====================================================

        train_caption_count = (

            train_df["text"]

            .str.strip()

            .ne("")

            .sum()

        )


        test_caption_count = (

            test_df["text"]

            .str.strip()

            .ne("")

            .sum()

        )


        print()

        print(
            f"[OK] Train captions: "
            f"{train_caption_count}/"
            f"{len(train_df)}"
        )

        print(
            f"[OK] Test captions: "
            f"{test_caption_count}/"
            f"{len(test_df)}"
        )


        # ====================================================
        # LOAD TRAIN DATASET
        # ====================================================

        print(
            "\nLoading training dataset..."
        )


        train_dataset = CSAMDataset(

            csv_file=prepared_train_csv,

            image_dir=IMAGE_DIR,

            precompute_logic=True,

            skip_caption_loading=True

        )


        # ====================================================
        # LOAD OUTER TEST DATASET
        # ====================================================

        print(
            "\nLoading outer test dataset..."
        )


        test_dataset = CSAMDataset(

            csv_file=prepared_test_csv,

            image_dir=IMAGE_DIR,

            precompute_logic=True,

            skip_caption_loading=True

        )


        print()

        print(
            f"[OK] Train dataset : "
            f"{len(train_dataset)}"
        )

        print(
            f"[OK] Outer test dataset : "
            f"{len(test_dataset)}"
        )


        # ====================================================
        # INTERNAL VALIDATION SPLIT
        # ====================================================

        print(
            "\nCreating internal validation split..."
        )


        labels = np.array(
            train_dataset.data[
                "label"
            ]
        )


        indices = np.arange(
            len(train_dataset)
        )


        internal_train_idx, internal_val_idx = (

            train_test_split(

                indices,

                test_size=INTERNAL_VAL_SIZE,

                random_state=42 + fold,

                stratify=labels

            )

        )


        print(
            f"Internal train samples : "
            f"{len(internal_train_idx)}"
        )

        print(
            f"Internal val samples   : "
            f"{len(internal_val_idx)}"
        )

        print(
            f"Outer test samples     : "
            f"{len(test_dataset)}"
        )


        # ====================================================
        # CREATE SUBSETS
        # ====================================================

        internal_train_dataset = Subset(

            train_dataset,

            internal_train_idx

        )


        internal_val_dataset = Subset(

            train_dataset,

            internal_val_idx

        )


        # ====================================================
        # DATALOADERS
        # ====================================================

        train_loader = DataLoader(

            internal_train_dataset,

            batch_size=BATCH_SIZE,

            shuffle=True,

            num_workers=0,

            pin_memory=torch.cuda.is_available()

        )


        val_loader = DataLoader(

            internal_val_dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=0,

            pin_memory=torch.cuda.is_available()

        )


        test_loader = DataLoader(

            test_dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=0,

            pin_memory=torch.cuda.is_available()

        )


        print()

        print(
            f"[OK] Training batches : "
            f"{len(train_loader)}"
        )

        print(
            f"[OK] Validation batches : "
            f"{len(val_loader)}"
        )

        print(
            f"[OK] Outer test batches : "
            f"{len(test_loader)}"
        )


        # ====================================================
        # TRAIN MODEL
        # ====================================================

        print("\n")

        print("=" * 70)

        print(
            f"TRAINING FOLD {fold}"
        )

        print("=" * 70)


        model = train_pipeline(

            train_loader=train_loader,

            val_loader=val_loader,

            epochs=EPOCHS,

            lr=LR,

            csam_weight=CSAM_WEIGHT,

            save_dir=fold_dir,

            device=DEVICE,

            mode="full"

        )


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        best_path = os.path.join(

            fold_dir,

            "best_model.pt"

        )


        if not os.path.exists(
            best_path
        ):

            raise FileNotFoundError(

                f"Best model checkpoint not found:\n"
                f"{best_path}"

            )


        print()

        print(
            "[OK] Loading best model:"
        )

        print(
            f"     {best_path}"
        )


        load_checkpoint(

            model,

            checkpoint_path=best_path,

            device=DEVICE

        )


        model.eval()


        # ====================================================
        # OUTER TEST EVALUATION
        # ====================================================

        print("\n")

        print("=" * 70)

        print(
            f"OUTER TEST EVALUATION - FOLD {fold}"
        )

        print("=" * 70)


        criterion = CSAMLoss(

            csam_weight=CSAM_WEIGHT

        )


        with torch.no_grad():

            test_metrics, predictions, targets = (

                evaluate_model(

                    model,

                    test_loader,

                    criterion,

                    DEVICE

                )

            )


        # ====================================================
        # FOLD RESULT
        # ====================================================

        result = {

            "fold": fold,

            "accuracy":
                test_metrics["accuracy"],

            "precision":
                test_metrics["precision"],

            "recall":
                test_metrics["recall"],

            "f1_score":
                test_metrics["f1_score"]

        }


        fold_results.append(
            result
        )


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()

        print("-" * 70)

        print(
            f"FOLD {fold} OUTER TEST RESULTS"
        )

        print("-" * 70)


        print(

            f"Accuracy  : "
            f"{result['accuracy']:.4f}"

        )


        print(

            f"Precision : "
            f"{result['precision']:.4f}"

        )


        print(

            f"Recall    : "
            f"{result['recall']:.4f}"

        )


        print(

            f"F1-Score  : "
            f"{result['f1_score']:.4f}"

        )


        # ====================================================
        # OPTIONAL CONFUSION VALUES
        # ====================================================

        if "tp" in test_metrics:

            print()

            print(
                "Confusion Matrix:"
            )

            print(
                f"TP = {test_metrics['tp']}"
            )

            print(
                f"FP = {test_metrics['fp']}"
            )

            print(
                f"TN = {test_metrics['tn']}"
            )

            print(
                f"FN = {test_metrics['fn']}"
            )


        all_cv_preds.extend(predictions)
        all_cv_targets.extend(targets)

        # ====================================================
        # SAVE FOLD TEST RESULT
        # ====================================================

        fold_result_path = os.path.join(

            fold_dir,

            "outer_test_results.txt"

        )


        with open(
            fold_result_path,
            "w"
        ) as f:

            f.write(
                f"FOLD {fold} OUTER TEST RESULTS\n"
            )

            f.write(
                "=" * 60 + "\n\n"
            )

            f.write(
                f"Accuracy  : "
                f"{result['accuracy']:.6f}\n"
            )

            f.write(
                f"Precision : "
                f"{result['precision']:.6f}\n"
            )

            f.write(
                f"Recall    : "
                f"{result['recall']:.6f}\n"
            )

            f.write(
                f"F1-Score  : "
                f"{result['f1_score']:.6f}\n"
            )

        # Save Per-Fold Detailed Metrics JSON
        fold_json_path = os.path.join(fold_dir, "outer_test_results.json")
        with open(fold_json_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"[OK] Saved fold {fold} metrics JSON        -> {fold_json_path}")

        # Plot & Save Per-Fold Confusion Matrix PNG
        fold_cm_path = os.path.join(fold_dir, "confusion_matrix.png")
        plot_and_save_confusion_matrix(
            y_true=targets,
            y_pred=predictions,
            class_names=["Non_CSAM", "CSAM"],
            save_path=fold_cm_path,
            title=f"Complete CSAM Model - Fold {fold} Outer Test Confusion Matrix"
        )
        print(f"[OK] Saved fold {fold} confusion matrix PNG -> {fold_cm_path}")


        # ====================================================
        # CLEAN MEMORY
        # ====================================================

        del model

        del train_dataset

        del test_dataset

        del train_loader

        del val_loader

        del test_loader

        del internal_train_dataset

        del internal_val_dataset

        gc.collect()


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # FINAL RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        fold_results
    )


    print("\n\n")

    print("=" * 70)

    print(
        "FINAL 5-FOLD CROSS VALIDATION RESULTS"
    )

    print("=" * 70)


    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # MEAN STD
    # ========================================================

    metrics = [

        "accuracy",

        "precision",

        "recall",

        "f1_score"

    ]


    print()

    print("-" * 70)

    print(
        "MEAN ± STD"
    )

    print("-" * 70)


    summary = {}


    for metric in metrics:

        mean = (

            results_df[
                metric
            ]

            .mean()

        )


        std = (

            results_df[
                metric
            ]

            .std()

        )


        summary[metric] = {

            "mean": mean,

            "std": std

        }


        print(

            f"{metric.capitalize():10s}: "
            f"{mean:.4f} ± {std:.4f}"

        )


    # ========================================================
    # SAVE RESULTS CSV
    # ========================================================

    results_path = os.path.join(

        SAVE_DIR,

        "5_fold_results.csv"

    )


    results_df.to_csv(

        results_path,

        index=False

    )


    print()

    print(
        f"[OK] Results saved:"
    )

    print(
        f"     {results_path}"
    )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = os.path.join(

        SAVE_DIR,

        "5_fold_summary.txt"

    )


    with open(

        summary_path,

        "w"

    ) as f:

        f.write(
            "5-FOLD CROSS VALIDATION RESULTS\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )


        for metric in metrics:

            f.write(

                f"{metric.capitalize():10s}: "
                f"{summary[metric]['mean']:.6f} "
                f"+/- "
                f"{summary[metric]['std']:.6f}\n"

            )


    print(

        f"[OK] Summary saved:"
    )

    print(
        f"     {summary_path}"
    )

    # ========================================================
    # SAVE SUMMARY JSON & OVERALL CONFUSION MATRIX PNG
    # ========================================================

    summary_json_path = os.path.join(
        SAVE_DIR,
        "5_fold_summary.json"
    )
    cv_summary_data = {
        "fold_results": fold_results,
        "summary_metrics": summary,
        "total_samples": len(all_cv_targets)
    }
    with open(summary_json_path, "w") as f:
        json.dump(cv_summary_data, f, indent=4)
    print(f"[OK] Summary JSON saved        -> {summary_json_path}")

    results_json_path = os.path.join(
        SAVE_DIR,
        "5_fold_results.json"
    )
    with open(results_json_path, "w") as f:
        json.dump(fold_results, f, indent=4)
    print(f"[OK] Results JSON saved        -> {results_json_path}")

    if len(all_cv_targets) > 0:
        overall_cm_path = os.path.join(
            SAVE_DIR,
            "5_fold_confusion_matrix.png"
        )
        plot_and_save_confusion_matrix(
            y_true=all_cv_targets,
            y_pred=all_cv_preds,
            class_names=["Non_CSAM", "CSAM"],
            save_path=overall_cm_path,
            title="Complete CSAM Model - 5-Fold Cross Validation Confusion Matrix"
        )
        print(f"[OK] Overall Confusion Matrix -> {overall_cm_path}")


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "[OK] 5-FOLD CROSS VALIDATION COMPLETE"
    )

    print(
        "[OK] Fixed shared splits used."
    )

    print(
        "[OK] 4 folds used for training."
    )

    print(
        "[OK] 1 fold used as outer test."
    )

    print(
        "[OK] No group leakage detected."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()