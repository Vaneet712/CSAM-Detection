import os

import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.full_pipeline import FullCSAMPipeline
from training.loss import CSAMLoss
from training.validate import evaluate_model
from training.checkpoint import save_checkpoint

import matplotlib.pyplot as plt


def train_pipeline(
    train_loader,
    mode="full",
    val_loader=None,
    epochs=5,
    lr=1e-4,
    csam_weight=2.0,
    save_dir="checkpoints",
    device=None,
    use_logic=True
):

    """
    Complete training loop for the CSAM Detection Pipeline.
    Supports full model and ablation experiments.
    """

    # ============================================================
    # DEVICE
    # ============================================================

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("=" * 70)
    print(
        f"Starting CSAM Training Pipeline on Device: {device}"
    )
    print(f"ABLATION MODE: {mode}")
    print("=" * 70)

    # ============================================================
    # MODEL
    # ============================================================

    model = FullCSAMPipeline(
        mode=mode
    ).to(device)

    # ============================================================
    # LOSS
    # ============================================================

    criterion = CSAMLoss(
        csam_weight=csam_weight
    )

    # ============================================================
    # OPTIMIZER
    # ============================================================

    trainable_params = filter(
        lambda p: p.requires_grad,
        model.parameters()
    )

    optimizer = AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-2
    )

    # ============================================================
    # LR SCHEDULER
    # ============================================================

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    best_f1 = -1.0

    patience = 5
    patience_counter = 0

    # ============================================================
    # HISTORY
    # ============================================================

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    # ============================================================
    # CHECKPOINT DIRECTORY
    # ============================================================

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        # ========================================================
        # TRAINING BATCHES
        # ========================================================

        for step, batch in enumerate(
            train_loader,
            1
        ):

            images = batch["image"].to(device)

            texts = batch["text"]

            logic_vector = batch[
                "logic_vector"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            # ----------------------------------------------------
            # ZERO GRADIENT
            # ----------------------------------------------------

            optimizer.zero_grad()

            # ----------------------------------------------------
            # FORWARD
            # ----------------------------------------------------

            logits, attention = model(
                images,
                texts,
                logic_vector
            )

            # ----------------------------------------------------
            # LOSS
            # ----------------------------------------------------

            loss = criterion(
                logits,
                labels
            )

            # ----------------------------------------------------
            # BACKPROPAGATION
            # ----------------------------------------------------

            loss.backward()

            optimizer.step()

            # ----------------------------------------------------
            # TRAINING STATISTICS
            # ----------------------------------------------------

            train_loss += (
                loss.item() * len(labels)
            )

            preds = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += len(labels)

        # ========================================================
        # LR SCHEDULER
        # ========================================================

        scheduler.step()

        # ========================================================
        # TRAIN METRICS
        # ========================================================

        epoch_train_loss = (
            train_loss / total
            if total > 0
            else 0.0
        )

        epoch_train_acc = (
            correct / total
            if total > 0
            else 0.0
        )

        train_losses.append(
            epoch_train_loss
        )

        train_accuracies.append(
            epoch_train_acc
        )

        # ========================================================
        # VALIDATION
        # ========================================================

        val_metrics = {}

        if (
            val_loader is not None
            and len(val_loader.dataset) > 0
        ):

            model.eval()

            val_metrics, _, _ = evaluate_model(
                model,
                val_loader,
                criterion,
                device
            )

            val_loss = val_metrics[
                "loss"
            ]

            val_acc = val_metrics[
                "accuracy"
            ]

            val_prec = val_metrics[
                "precision"
            ]

            val_rec = val_metrics[
                "recall"
            ]

            val_f1 = val_metrics[
                "f1_score"
            ]

            val_losses.append(
                val_loss
            )

            val_accuracies.append(
                val_acc
            )

            # ----------------------------------------------------
            # PRINT EPOCH RESULTS
            # ----------------------------------------------------

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Train Acc: {epoch_train_acc:.4f} || "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val Prec: {val_prec:.4f} | "
                f"Val Rec: {val_rec:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

            # ----------------------------------------------------
            # BEST MODEL
            # ----------------------------------------------------

            if val_f1 > best_f1:

                best_f1 = val_f1

                patience_counter = 0

                best_path = os.path.join(
                    save_dir,
                    "best_model.pt"
                )

                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    best_path
                )

                print(
                    f"[OK] Best model saved: "
                    f"{best_path}"
                )

            else:

                patience_counter += 1

                print(
                    f"Early stopping patience: "
                    f"{patience_counter}/{patience}"
                )

            # ----------------------------------------------------
            # EARLY STOPPING
            # ----------------------------------------------------

            if patience_counter >= patience:

                print(
                    "[OK] Early stopping triggered."
                )

                break

        # ========================================================
        # LATEST CHECKPOINT
        # ========================================================

        latest_path = os.path.join(
            save_dir,
            "latest_checkpoint.pt"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics,
            latest_path
        )

    # ============================================================
    # TRAINING COMPLETE
    # ============================================================

    print("=" * 70)

    print(
        "[OK] Training Pipeline Execution Complete"
    )

    print("=" * 70)

    # ============================================================
    # TRAINING PLOTS
    # ============================================================

    epochs_completed = range(
        1,
        len(train_losses) + 1
    )

    # ============================================================
    # LOSS CURVE
    # ============================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs_completed,
        train_losses,
        label="Training Loss"
    )

    if len(val_losses) > 0:

        plt.plot(
            epochs_completed,
            val_losses,
            label="Validation Loss"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"Training and Validation Loss - {mode}"
    )

    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(
        save_dir,
        "loss_curve.png"
    )

    plt.savefig(
        loss_plot_path
    )

    plt.close()

    print(
        f"[OK] Loss curve saved at: "
        f"{loss_plot_path}"
    )

    # ============================================================
    # ACCURACY CURVE
    # ============================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs_completed,
        train_accuracies,
        label="Training Accuracy"
    )

    if len(val_accuracies) > 0:

        plt.plot(
            epochs_completed,
            val_accuracies,
            label="Validation Accuracy"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"Training and Validation Accuracy - {mode}"
    )

    plt.legend()
    plt.grid(True)

    accuracy_plot_path = os.path.join(
        save_dir,
        "accuracy_curve.png"
    )

    plt.savefig(
        accuracy_plot_path
    )

    plt.close()

    print(
        f"[OK] Accuracy curve saved at: "
        f"{accuracy_plot_path}"
    )

    return model