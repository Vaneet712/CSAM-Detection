import torch


class FLConfig:
    """Federated Learning configuration."""

    NUM_CLIENTS = 5
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    CSAM_WEIGHT = 2.0
    DATA_SPLITS_DIR = "federated_splits"
    IMAGE_DIR = "dataset_updated_organized"
    RESULTS_DIR = "federated/results"
    CHECKPOINT_DIR = "federated/checkpoints"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __repr__(self):
        return (
            f"FLConfig(clients={self.NUM_CLIENTS}, rounds={self.NUM_ROUNDS}, "
            f"local_epochs={self.LOCAL_EPOCHS}, batch_size={self.BATCH_SIZE}, "
            f"lr={self.LEARNING_RATE})"
        )

