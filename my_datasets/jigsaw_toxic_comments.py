import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data.sampler import WeightedRandomSampler
from my_datasets.utils import get_sampling_weights


class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, biases=None, embeddings=None):
        self.biases = biases
        self.texts = texts
        self.targets = labels
        self.embeddings = embeddings

        if self.embeddings is not None:
            # Ensure embeddings are a PyTorch tensor
            self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)

        # Ensure labels are a PyTorch tensor and unsqueezed for BCEWithLogitsLoss
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        if self.biases is not None:
            # Ensure embeddings are a PyTorch tensor
            self.biases = torch.tensor(self.biases, dtype=torch.long)
        else:
            self.biases = self.targets  # workaround in order not to have nans in biases

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Return embeddings if available, otherwise return text for on-the-fly encoding
        input_data = (
            self.embeddings[idx] if self.embeddings is not None else self.texts[idx]
        )

        return {
            "index": idx,
            "inputs": input_data,
            "targets": self.targets[idx],
            "bias": self.biases[idx],
        }


def load_and_prepare_data(csv_path, target_label, bias_label=None):
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded '{csv_path}' with {len(df)} rows.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{csv_path}' not found.")
        return None, None, None
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from '{csv_path}': {e}")

    if (
        "comment_text" not in df.columns
        or target_label not in df.columns
        or (bias_label not in df.columns and bias_label != None)
    ):
        raise ValueError(
            f"Error: CSV must contain 'comment_text', '{target_label}', and '{bias_label}' columns."
        )

    if bias_label == None:
        bias_label = target_label
    df["comment_text"] = df["comment_text"].fillna("")
    df[target_label] = (
        pd.to_numeric(df[target_label], errors="coerce").fillna(0).astype(int)
    )
    df[bias_label] = (
        pd.to_numeric(df[bias_label], errors="coerce").fillna(0).astype(int)
    )

    return (
        df["comment_text"].tolist(),
        df[target_label].tolist(),
        df[bias_label].tolist(),
    )


def get_jigsaw_toxic_comments_loaders(
    root,
    train="train_identity_hate_biases.csv",
    val="test_identity_hate.csv",
    test="test_chatgpt.csv",
    bias="bias",
    target="identity_hate",
    batch_size=128,
    num_workers=4,
    encoder_name="all-MiniLM-L6-v2",
    sampler=None,
):

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Data Loading ---

    X_train_text, y_train, b_train = load_and_prepare_data(
        os.path.join(root, train), target, bias
    )
    X_val_text, y_val, b_val = load_and_prepare_data(
        os.path.join(root, val), target, bias
    )
    X_test_text, y_test, b_test = load_and_prepare_data(
        os.path.join(root, test), target, bias
    )

    print("\n--- Target Class Distribution (Identity_hate) ---")
    print("Training Set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("Validation Set:")
    print(pd.Series(y_val).value_counts(normalize=True))
    print("\nTest Set:")
    print(pd.Series(y_test).value_counts(normalize=True))

    # --- 2. Generate Embeddings using Sentence-BERT ---
    print(f"\n--- Loading Sentence-BERT model: '{encoder_name}' for embeddings ---")
    embedding_model = SentenceTransformer(encoder_name)
    embedding_model.to(device)  # Move embedding model to GPU if available
    print("Embedding model loaded successfully.")

    print("\n--- Generating Embeddings (This may take a while) ---")
    # Use tqdm for progress bar
    X_train_embeddings = embedding_model.encode(
        X_train_text,
        show_progress_bar=True,  # tqdm handles it
        convert_to_numpy=True,
        device=str(device),  # Ensure encoding happens on selected device
    )
    print(f"Training/Validation embeddings shape: {X_train_embeddings.shape}")

    X_val_embeddings = embedding_model.encode(
        X_val_text,
        show_progress_bar=True,  # tqdm handles it
        convert_to_numpy=True,
        device=str(device),  # Ensure encoding happens on selected device
    )
    print(f"Training/Validation embeddings shape: {X_val_embeddings.shape}")

    X_test_embeddings = embedding_model.encode(
        X_test_text,
        show_progress_bar=True,  # tqdm handles it
        convert_to_numpy=True,
        device=str(device),  # Ensure encoding happens on selected device
    )
    print(f"Test embeddings shape: {X_test_embeddings.shape}")

    # --- 4. PyTorch DataLoaders ---
    print("\n--- Creating PyTorch DataLoaders ---")

    # Create TensorDatasets
    train_dataset = CustomTextDataset(
        X_train_text, y_train, b_train, embeddings=X_train_embeddings
    )
    val_dataset = CustomTextDataset(
        X_val_text, y_val, b_val, embeddings=X_val_embeddings
    )
    test_dataset = CustomTextDataset(
        X_test_text, y_test, b_test, embeddings=X_test_embeddings
    )

    if sampler == "weighted":
        weights = get_sampling_weights(
            train_dataset.targets,
            train_dataset.biases,
        )
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
        )
    else:
        sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("DataLoaders created.")
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
