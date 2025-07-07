import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import (
    train_test_split,
)  # Still imported but not used for the main split

# import kagglehub # Removed as automatic download is no longer used
import torchaudio  # Needed for loading audio files
from tqdm.auto import tqdm  # For progress bars
import torch.nn.functional as F
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wav):
        for t in self.transforms:
            wav = t(wav)
        return wav

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "     {0}".format(t)
        format_string += "\n)"
        return format_string


class Pad:
    def __init__(
        self, pad_length=16000, mode="constant", value=0, pad_position="center"
    ):
        self.pad_length = pad_length
        self.mode = mode
        self.value = value
        self.pad_position = pad_position

    def __call__(self, wav):
        wav_length = wav.shape[1]
        delta = self.pad_length - wav_length
        if self.pad_position == "center":
            left_crop_len = int(delta / 2)
            right_crop_len = delta - int(delta / 2)
        elif self.pad_position == "right":
            left_crop_len = 0
            right_crop_len = delta
        elif self.pad_position == "left":
            left_crop_len = delta
            right_crop_len = 0
        elif self.pad_position == "random":
            left_crop_len = int(random.random() * delta)
            right_crop_len = delta - left_crop_len

        wav = F.pad(wav, (left_crop_len, right_crop_len), self.mode, self.value)
        return wav


class Crop:
    def __init__(self, crop_length=16000, crop_position="center"):
        self.crop_length = crop_length
        self.crop_position = crop_position

    def __call__(self, wav):
        wav_length = wav.shape[1]
        delta = wav_length - self.crop_length

        if self.crop_position == "left":
            i = 0

        elif (
            self.crop_position == "left"
        ):  # This 'elif' should likely be 'elif self.crop_position == "right"'
            i = delta

        elif self.crop_position == "center":
            i = int(delta / 2)

        elif self.crop_position == "random":
            i = random.randint(0, delta)

        wav = wav[:, i : i + self.crop_length]
        return wav


class PadCrop:
    def __init__(
        self,
        pad_crop_length=16000,
        crop_position="center",
        pad_mode="constant",
        pad_value=0,
        pad_position="center",
    ):

        self.pad_crop_length = pad_crop_length
        self.crop_position = crop_position
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.pad_position = pad_position
        self.crop_transform = Crop(
            crop_length=self.pad_crop_length, crop_position=self.crop_position
        )
        self.pad_transform = Pad(
            pad_length=self.pad_crop_length,
            mode=self.pad_mode,
            value=self.pad_value,
            pad_position=self.pad_position,
        )

    def __call__(self, wav):
        wav_length = wav.shape[1]
        delta = wav_length - self.pad_crop_length

        if delta > 0:
            wav = self.crop_transform(wav)
        elif delta < 0:
            wav = self.pad_transform(wav)
        return wav


# --- Real Audio Embedding Logic using Wav2Vec2 ---
# --- MFCC Feature Extractor ---
class MFCCEncoder:
    """
    Extracts MFCC features from audio files using torchaudio.
    """

    def __init__(
        self, sample_rate=16000, n_mfcc=40, n_fft=400, hop_length=160, device="cpu"
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.device = device

        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": n_fft, "hop_length": hop_length},
        ).to(device)

        # Amplitude to DB for better representation
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

        # The output dimension will be n_mfcc (number of coefficients)
        # The time dimension depends on the audio length after PadCrop
        # For a fixed length of 128000 samples (8 seconds at 16kHz),
        # with hop_length=160, the number of frames would be (128000 // 160) + 1 = 801
        # So, the embedding dimension is n_mfcc * num_frames_after_pooling
        # If we average across time, it's just n_mfcc. Let's assume we mean pool across time.
        self.embedding_dim = n_mfcc

    def encode(
        self,
        audio_file_paths,
        sample_rate=16000,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu",
        transforms=None,
    ):
        """
        Processes a list of audio file paths and returns their MFCC embeddings.
        Args:
            audio_file_paths (list): List of paths to audio files.
            sample_rate (int): Target sample rate for audio processing.
            show_progress_bar (bool): Whether to display a progress bar.
            convert_to_numpy (bool): Whether to convert embeddings to NumPy array.
            device (str): Device to run the model on ('cpu' or 'cuda').
            transforms (Compose, optional): Audio transformations to apply (e.g., PadCrop). Defaults to None.
        Returns:
            np.ndarray or list: MFCC embeddings.
        """
        embeddings = []
        iterator = (
            tqdm(audio_file_paths, desc="Extracting MFCCs")
            if show_progress_bar
            else audio_file_paths
        )

        for audio_path in iterator:
            try:
                speech, sr = torchaudio.load(audio_path)

                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=sample_rate
                    )
                    speech = resampler(speech)

                if speech.shape[0] > 1:
                    speech = torch.mean(speech, dim=0, keepdim=True)

                if transforms is not None:
                    speech = transforms(speech)

                # Move speech to device before MFCC transformation
                speech = speech.to(self.device)

                # Compute MFCCs
                mfccs = self.mfcc_transform(speech)
                # Convert to dB scale
                mfccs_db = self.amplitude_to_db(mfccs)

                # Mean pool across the time dimension to get a fixed-size embedding per audio clip
                features = mfccs_db  # .mean(
                #     dim=-1
                # )  # Assuming shape is [batch_size, n_mfcc, time_frames]

                embeddings.append(features.cpu().numpy())
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Append a placeholder (e.g., NaNs) for failed files, ensuring correct dimension
                embeddings.append(np.full((1, self.embedding_dim), np.nan))

        if convert_to_numpy:
            final_embeddings = np.vstack(embeddings)
            final_embeddings = np.nan_to_num(final_embeddings)
            return final_embeddings
        return embeddings


# --- Custom Audio Dataset ---
class CustomAudioDataset(Dataset):
    def __init__(self, audio_embeddings, targets, biases=None):
        self.audio_embeddings = audio_embeddings  # This now directly holds the pre-computed audio embeddings
        self.targets = targets
        self.biases = biases

        # Ensure embeddings are a PyTorch tensor
        self.audio_embeddings = torch.tensor(self.audio_embeddings, dtype=torch.float32)

        # Ensure targets are a PyTorch tensor
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        if self.biases is not None:
            # Ensure biases are a PyTorch tensor
            # The salience can be float, so let's keep it float for now.
            # If it's used as a categorical bias, it might need to be cast to long or one-hot encoded later.
            self.biases = torch.tensor(self.biases, dtype=torch.long)
        else:
            self.biases = torch.zeros_like(
                self.targets, dtype=torch.long
            )  # Default to zeros if no bias, avoid NaN

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "index": idx,
            "inputs": self.audio_embeddings[idx],  # Input is now audio embedding
            "targets": self.targets[idx],
            "bias": self.biases[idx],
        }


def compute_or_load_embeddings_audio(
    audio_file_paths, split_name, encoder_instance, cache_dir, device
):
    """
    Generates or loads audio embeddings.
    `encoder_instance` should be an instance of a class with an `encode` method
    that takes a list of audio file paths.
    """
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, f"{split_name}_audio_embeddings.npy")
    file_paths_cache_path = os.path.join(
        cache_dir, f"{split_name}_audio_file_paths.txt"
    )

    # To ensure consistency, store the list of file paths used to generate embeddings
    if os.path.exists(emb_path) and os.path.exists(file_paths_cache_path):
        with open(file_paths_cache_path, "r") as f:
            cached_paths = [line.strip() for line in f]

        # Check if the list of audio paths matches the cached ones
        if cached_paths == audio_file_paths:
            print(
                f"Loading cached audio embeddings for '{split_name}' from: {emb_path}"
            )
            embeddings = np.load(emb_path)
            return embeddings
        else:
            print(
                f"Cached audio file paths for '{split_name}' do not match current paths. Re-generating embeddings."
            )

    # Define transforms for audio before encoding (e.g., Pad/Crop to a fixed length)
    transforms = Compose(
        [PadCrop(pad_crop_length=64000)]
    )  # Keep 8 seconds as in previous code

    print(f"Generating audio embeddings for '{split_name}'...")
    embeddings = encoder_instance.encode(  # Renamed audio_encoder to encoder_instance for clarity
        audio_file_paths,
        show_progress_bar=True,
        device=str(device),
        transforms=transforms,
    )
    np.save(emb_path, embeddings)
    with open(file_paths_cache_path, "w") as f:
        for path in audio_file_paths:
            f.write(f"{path}\n")
    print(f"Saved audio embeddings to: {emb_path}")
    return embeddings


def get_sampling_weights(targets):
    """
    Computes sampling weights to balance the target classes (is_english_speaker).
    Weights are inversely proportional to the class frequency.
    """
    print("Computing sampling weights to balance target classes.")
    class_counts = torch.bincount(targets)
    # Handle cases where a class might have 0 samples to avoid division by zero
    class_weights = 1.0 / (
        class_counts.float() + 1e-6
    )  # Add small epsilon to avoid division by zero
    # Create a weight for each sample based on its target class
    sample_weights = class_weights[targets]
    return sample_weights


# --- Main DataLoader Generation Function ---
def get_urbansounds_dataloaders(
    root,  # This is where the dataset will be downloaded and cache will be stored
    batch_size=128,
    num_workers=4,
    encoder_name="MFCCEncoder",  # Specifies which encoder to use
    random_seed=42,  # For reproducible splits
    sampler=None,
):
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if encoder_name == "MFCCEncoder":
        # Instantiate the MFCCEncoder
        audio_encoder = MFCCEncoder(device=str(device))
    else:
        raise NotImplementedError(
            f"Encoder '{encoder_name}' is not supported for audio encoding."
        )

    # --- 1. Data Loading and Preprocessing ---
    print("\n--- Loading UrbanSound8K Data ---")

    # Define the expected path to the UrbanSound8K dataset
    dataset_path = root

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}. "
            "Please ensure the UrbanSound8K dataset is manually downloaded "
            "and extracted to this location."
        )
    print(f"Dataset found at: {dataset_path}")

    # Correct CSV file path for UrbanSound8K
    csv_file_path = os.path.join(
        dataset_path, "metadata", "UrbanSound8K.csv"
    )  # Corrected metadata path

    try:
        df = pd.read_csv(csv_file_path)
        print("CSV loaded successfully.")
    except FileNotFoundError:
        print(f"Error: 'UrbanSound8K.csv' not found at {csv_file_path}.")
        raise  # Re-raise to stop if file not found

    # Construct full audio file paths
    # Audio files are structured as 'audio/fold{fold}/{slice_file_name}'
    audio_base_dir = os.path.join(dataset_path, "audio")
    df["audio_filepath"] = df.apply(
        lambda row: os.path.join(
            audio_base_dir, f"fold{row['fold']}", row["slice_file_name"]
        ),
        axis=1,
    )

    # Filter out rows where audio file doesn't exist
    df["audio_exists"] = df["audio_filepath"].apply(os.path.exists)
    df_filtered = df[
        df["audio_exists"]
    ].copy()  # Create a copy to avoid SettingWithCopyWarning

    # Ensure 'classID' and 'salience' are available and cast appropriately
    if "classID" not in df_filtered.columns or "salience" not in df_filtered.columns:
        raise ValueError("Missing 'classID' or 'salience' columns in the metadata.")

    # Target: 'classID' (0-9)
    df_filtered["target"] = df_filtered["classID"].astype(int)
    # Bias: 'salience' (1 or 2)
    df_filtered["bias"] = df_filtered["salience"].map({1: 0, 2: 1}).astype(float)
    print(f"\nOriginal DataFrame size: {len(df)} rows")
    print(f"Filtered DataFrame size (audio exists): {len(df_filtered)} rows")
    print("\nClass Distribution (Target - classID) in Filtered Data:")
    print(df_filtered["target"].value_counts().sort_index())
    print("\nSalience Distribution (Bias) in Filtered Data:")
    print(df_filtered["bias"].value_counts().sort_index())

    # --- Train/Test Split (Fold-based) ---
    print("\n--- Performing Train/Test Split based on Folds ---")

    # Test set: fold 10
    test_df = df_filtered[df_filtered["fold"] == 10].copy()
    # Training set: rest of the folds (1-9)
    train_df = df_filtered[df_filtered["fold"] != 10].copy()

    print(f"Train set size (Folds 1-9): {len(train_df)} rows")
    print(f"Test set size (Fold 10): {len(test_df)} rows")

    # Cache split dataframes (no longer using test_split_ratio or bias_ratio in filename)
    train_df_cache_path = os.path.join(
        root, f"train_df_folds_1_9_seed_{random_seed}.pkl"
    )
    test_df_cache_path = os.path.join(root, f"test_df_fold_10_seed_{random_seed}.pkl")

    # Check if cached dataframes exist
    if os.path.exists(train_df_cache_path) and os.path.exists(test_df_cache_path):
        print(
            f"Loading cached train/test split from: {train_df_cache_path} and {test_df_cache_path}"
        )
        train_df = pd.read_pickle(train_df_cache_path)
        test_df = pd.read_pickle(test_df_cache_path)
    else:
        print(
            f"Cached train/test split not found or outdated. Saving new split to cache."
        )
        train_df.to_pickle(train_df_cache_path)
        test_df.to_pickle(test_df_cache_path)
        print(
            f"Saved train/test split to: {train_df_cache_path} and {test_df_cache_path}"
        )

    print("\n--- Class and Bias Distribution in Train Set ---")
    print("Class (Target - classID):")
    print(train_df["target"].value_counts().sort_index())
    print("\nSalience (Bias):")
    print(train_df["bias"].value_counts().sort_index())

    print("\n--- Class and Bias Distribution in Test Set ---")
    print("Class (Target - classID):")
    print(test_df["target"].value_counts().sort_index())
    print("\nSalience (Bias):")
    print(test_df["bias"].value_counts().sort_index())

    # --- 3. Generate Audio Embeddings ---
    print(
        f"\n--- Loading/Generating Audio Embeddings with {type(audio_encoder).__name__} ---"
    )

    X_train_audio_embeddings = compute_or_load_embeddings_audio(
        train_df["audio_filepath"].tolist(),
        "train",
        audio_encoder,
        root,  # Use root for caching embeddings
        device,
    )
    X_test_audio_embeddings = compute_or_load_embeddings_audio(
        test_df["audio_filepath"].tolist(),
        "test",
        audio_encoder,
        root,  # Use root for caching embeddings
        device,
    )

    # --- 4. PyTorch DataLoaders ---
    print("\n--- Creating PyTorch DataLoaders ---")

    train_dataset = CustomAudioDataset(
        audio_embeddings=X_train_audio_embeddings,
        targets=train_df["target"].values,  # Use 'target' column for targets
        biases=train_df["bias"].values,  # Use 'bias' column for biases
    )
    test_dataset = CustomAudioDataset(
        audio_embeddings=X_test_audio_embeddings,
        targets=test_df["target"].values,  # Use 'target' column for targets
        biases=test_df["bias"].values,  # Use 'bias' column for biases
    )

    # No weighted sampling for target balancing as per request
    if sampler == "weighted":
        weights = get_sampling_weights(
            train_dataset.targets,
        )
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(train_dataset), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=train_sampler,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("DataLoaders created successfully.")

    return (
        train_loader,
        test_loader,
        test_loader,
        train_dataset,
        test_dataset,
        test_dataset,
    )


# --- Example Usage ---
if __name__ == "__main__":
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main execution using device: {device}")

    # Example usage for UrbanSound8K
    train_loader, test_loader, train_dataset, test_dataset = (
        get_urbansounds_dataloaders(
            root="./data/urbansounds/UrbanSound8K",  # Specify a directory for data download and cache
            batch_size=32,
            num_workers=2,
            encoder_name="MFCCEncoder",  # Use the MFCCEncoder
            random_seed=42,
        )
    )

    print("\n--- Example Batch from Train DataLoader ---")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Inputs (Audio Embeddings) shape: {batch['inputs'].shape}")
        print(
            f"  Targets (Class ID) shape: {batch['targets'].shape}, Values: {batch['targets']}"
        )
        print(
            f"  Bias (Salience) shape: {batch['bias'].shape}, Values: {batch['bias']}"
        )
        if i == 0:  # Print only the first batch for brevity
            break

    print("\n--- Example Batch from Test DataLoader ---")
    for i, batch in enumerate(test_loader):
        print(f"Batch {i+1}:")
        print(f"  Inputs (Audio Embeddings) shape: {batch['inputs'].shape}")
        print(
            f"  Targets (Class ID) shape: {batch['targets'].shape}, Values: {batch['targets']}"
        )
        print(
            f"  Bias (Salience) shape: {batch['bias'].shape}, Values: {batch['bias']}"
        )
        if i == 0:  # Print only the first batch for brevity
            break
