import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import kagglehub
import torchaudio  # Needed for loading audio files
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
)  # For Wav2Vec2 audio embeddings
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
            format_string += "    {0}".format(t)
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

        elif self.crop_position == "left":
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
class RealAudioEncoder:
    """
    Encodes audio files into embeddings using a pre-trained Wav2Vec2 model.
    """

    def __init__(self, model_name="facebook/wav2vec2-base-960h", device="cpu"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.embedding_dim = (
            self.model.config.hidden_size
        )  # Get embedding dimension from model config

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
        Processes a list of audio file paths and returns their embeddings.
        Args:
            audio_file_paths (list): List of paths to audio files.
            sample_rate (int): Target sample rate for audio processing.
            show_progress_bar (bool): Whether to display a progress bar.
            convert_to_numpy (bool): Whether to convert embeddings to NumPy array.
            device (str): Device to run the model on ('cpu' or 'cuda').
        Returns:
            np.ndarray or list: Audio embeddings.
        """
        embeddings = []
        # Use tqdm for a progress bar if requested
        iterator = (
            tqdm(audio_file_paths, desc="Encoding Audio")
            if show_progress_bar
            else audio_file_paths
        )

        for audio_path in iterator:
            try:
                # Load audio
                speech, sr = torchaudio.load(audio_path)

                # Resample if needed to match model's expected sample rate (16kHz for Wav2Vec2)
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=sample_rate
                    )
                    speech = resampler(speech)

                # Ensure audio is mono (Wav2Vec2 expects mono input)
                if speech.shape[0] > 1:
                    speech = torch.mean(speech, dim=0, keepdim=True)

                if transforms is not None:
                    # print(speech.shape)
                    speech = transforms(speech)
                    # print(speech.shape)
                    # print(50 * "-")
                # Process and encode
                # Squeeze to remove channel dimension if it's 1 (e.g., [1, samples] -> [samples])
                inputs = self.processor(
                    speech.squeeze().numpy(),
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                )

                with torch.no_grad():
                    input_values = inputs.input_values.to(self.device)
                    # The last_hidden_state are the embeddings. Mean pool them for utterance-level embedding.
                    features = self.model(
                        input_values
                    ).last_hidden_state  # .mean(dim=1)
                    embeddings.append(features.cpu().numpy())
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                # Append a placeholder (e.g., NaNs) for failed files
                embeddings.append(np.full((1, self.embedding_dim), np.nan))

        if convert_to_numpy:
            # Stack all embeddings. Handle potential NaN rows if some audio files failed.
            final_embeddings = np.vstack(embeddings)
            final_embeddings = np.nan_to_num(final_embeddings)  # Replace NaNs with 0
            return final_embeddings
        return embeddings


# --- Your provided CustomTextDataset, adapted for Audio ---
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
            self.biases = torch.tensor(self.biases, dtype=torch.long)
        else:
            self.biases = torch.zeros_like(
                self.targets
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
    audio_file_paths, split_name, encoder_name, cache_dir, device
):
    """
    Generates or loads audio embeddings.
    `encoder_name` should be an instance of a class with an `encode` method
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

    transforms = Compose([PadCrop(pad_crop_length=128000)])
    print(f"Generating audio embeddings for '{split_name}'...")
    embeddings = encoder_name.encode(
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
def get_speech_accent_dataloaders(
    root,  # This is where the dataset will be downloaded and cache will be stored
    batch_size=128,
    num_workers=4,
    encoder_name=None,  # Pass an instance of your audio encoder
    sampler=None,  # Not using sampler in this example's current setup as it's not needed for the bias/target selection
    random_seed=42,  # For reproducible splits and sampling
    test_split_ratio=0.5,  # Hyperparameter for train/test split ratio
    desired_bias_ratio=3,  # Hyperparameter for desired correlation (e.g., 9 for 90% female/male)
):
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if encoder_name == "RealAudioEncoder":
        # Instantiate the RealAudioEncoder
        audio_encoder = RealAudioEncoder(device=str(device))
    else:
        raise NotImplementedError(
            "only RealAudioEncoder is supported for encoding audio"
        )

    # --- 1. Data Loading and Preprocessing ---
    print("\n--- Loading Speech Accent Archive Data ---")

    # Download the dataset to get the path
    try:
        dataset_path = kagglehub.dataset_download("rtatman/speech-accent-archive")
        print(f"Dataset downloaded to: {dataset_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure 'kagglehub' is installed (`pip install kagglehub`) ")
        print("and you have configured your Kaggle API credentials if necessary.")
        raise  # Re-raise to stop execution if dataset isn't available

    csv_file_path = os.path.join(dataset_path, "speakers_all.csv")

    try:
        df = pd.read_csv(csv_file_path)
        print("CSV loaded successfully.")
    except FileNotFoundError:
        print(f"Error: 'speakers_all.csv' not found at {csv_file_path}.")
        raise  # Re-raise to stop if file not found

    # Clean up column names and unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()

    # Correct 'famale' to 'female' in the 'sex' column
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.lower().str.strip()
        df["sex"].replace("famale", "female", inplace=True)

    # Create binary target 'is_english_speaker' (1 for English, 0 for Non-English)
    english_language = [
        "english"
    ]  # Make sure this matches the case in your data if not normalized
    df["is_english_speaker"] = (
        df["native_language"].astype(str).str.lower().isin(english_language).astype(int)
    )

    # Encode 'gender': 0 for female, 1 for male
    df["gender_encoded"] = df["sex"].map({"female": 0, "male": 1})

    # Construct full audio file paths
    audio_dir = os.path.join(dataset_path, "recordings", "recordings")
    df["audio_filepath"] = df["filename"].astype(str) + ".mp3"
    df["audio_filepath"] = df["audio_filepath"].apply(
        lambda x: os.path.join(audio_dir, x)
    )

    # Filter out rows where audio file doesn't exist or critical data is missing
    df["audio_exists"] = df["audio_filepath"].apply(os.path.exists)
    df_filtered = df[
        df["audio_exists"]
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    df_filtered.dropna(
        subset=["age", "age_onset", "gender_encoded", "is_english_speaker"],
        inplace=True,
    )

    print(f"\nOriginal DataFrame size: {len(df)} rows")
    print(f"Filtered DataFrame size (audio exists & no NaNs): {len(df_filtered)} rows")
    print("\nInitial Speaker & Gender Distribution (Filtered Data):")
    print(pd.crosstab(df_filtered["is_english_speaker"], df_filtered["gender_encoded"]))

    # --- Balance by Target (is_english_speaker) first ---
    print("\n" + "=" * 70)
    print("--- Balancing Data by Target (is_english_speaker) ---")
    print("=" * 70)

    english_speakers_all = df_filtered[df_filtered["is_english_speaker"] == 1]
    non_english_speakers_all = df_filtered[df_filtered["is_english_speaker"] == 0]

    min_class_count = min(len(english_speakers_all), len(non_english_speakers_all))

    if min_class_count == 0:
        print(
            "\nWarning: One of the target classes (English/Non-English) has no samples. Cannot balance by target."
        )
        df_balanced_target = (
            df_filtered.copy()
        )  # Proceed with original filtered data if balancing not possible
    else:
        # Undersample the majority class to match the minority class
        if len(english_speakers_all) > min_class_count:
            english_speakers_balanced = english_speakers_all.sample(
                n=min_class_count, random_state=random_seed
            )
            non_english_speakers_balanced = non_english_speakers_all
        else:  # len(non_english_speakers_all) > min_class_count
            non_english_speakers_balanced = non_english_speakers_all.sample(
                n=min_class_count, random_state=random_seed
            )
            english_speakers_balanced = english_speakers_all

        df_balanced_target = pd.concat(
            [english_speakers_balanced, non_english_speakers_balanced]
        ).reset_index(drop=True)
        print(f"Balanced target classes. Each class now has {min_class_count} samples.")

    print(f"\nDataFrame size after target balancing: {len(df_balanced_target)} rows")
    print("\nSpeaker Distribution after Target Balancing:")
    print(df_balanced_target["is_english_speaker"].value_counts())

    # --- Apply Gender Bias Subset Selection on the TARGET-BALANCED DataFrame ---
    print("\n" + "=" * 70)
    print(
        "--- Creating Subset with Specific Gender Proportions (on balanced target data) ---"
    )
    print("=" * 70)

    selected_df_parts = []

    # --- English Speakers (is_english_speaker == 1): Aim for `desired_bias_ratio` Female ---
    english_speakers_df = df_balanced_target[
        df_balanced_target["is_english_speaker"] == 1
    ].copy()
    english_females = english_speakers_df[english_speakers_df["gender_encoded"] == 0]
    english_males = english_speakers_df[english_speakers_df["gender_encoded"] == 1]

    num_english_females = len(english_females)
    num_english_males = len(english_males)

    if num_english_females == 0:
        print(
            "\nWarning: No English female speakers found in target-balanced data. Cannot achieve desired female ratio for English speakers."
        )
    else:
        desired_english_males = round(num_english_females / desired_bias_ratio)
        if num_english_males >= desired_english_males:
            sampled_english_males = english_males.sample(
                n=int(desired_english_males), random_state=random_seed
            )
            print(
                f"Selected {len(sampled_english_males)} out of {num_english_males} English male speakers to meet {100 * desired_bias_ratio / (desired_bias_ratio + 1):.0f}% female target."
            )
            selected_df_parts.append(english_females)
            selected_df_parts.append(sampled_english_males)
        else:
            print(
                f"\nWarning: Not enough English male speakers ({num_english_males}) in target-balanced data to achieve exactly {100 * desired_bias_ratio / (desired_bias_ratio + 1):.0f}% female for English speakers while keeping all available females."
            )
            if num_english_males > 0:
                actual_desired_english_females = round(
                    num_english_males * desired_bias_ratio
                )
                if num_english_females >= actual_desired_english_females:
                    sampled_english_females = english_females.sample(
                        n=int(actual_desired_english_females), random_state=random_seed
                    )
                    print(
                        f"Adjusting: Keeping all {num_english_males} English males and sampling {len(sampled_english_females)} English females to get closest to {100 * desired_bias_ratio / (desired_bias_ratio + 1):.0f}% female."
                    )
                    selected_df_parts.append(sampled_english_females)
                    selected_df_parts.append(english_males)
                else:
                    print(
                        f"Cannot achieve desired female ratio for English speakers even by taking all available males and females. Using all available English speakers for this group."
                    )
                    selected_df_parts.append(english_females)
                    selected_df_parts.append(english_males)
            else:
                print(
                    "No English males available in target-balanced data. Keeping all English females for this group."
                )
                selected_df_parts.append(english_females)

    # --- Non-English Speakers (is_english_speaker == 0): Aim for `desired_bias_ratio` Male ---
    non_english_speakers_df = df_balanced_target[
        df_balanced_target["is_english_speaker"] == 0
    ].copy()
    non_english_females = non_english_speakers_df[
        non_english_speakers_df["gender_encoded"] == 0
    ]
    non_english_males = non_english_speakers_df[
        non_english_speakers_df["gender_encoded"] == 1
    ]

    num_non_english_females = len(non_english_females)
    num_non_english_males = len(non_english_males)

    if num_non_english_males == 0:
        print(
            "\nWarning: No Non-English male speakers found in target-balanced data. Cannot achieve desired male ratio for Non-English speakers."
        )
    else:
        desired_non_english_females = round(num_non_english_males / desired_bias_ratio)
        if num_non_english_females >= desired_non_english_females:
            sampled_non_english_females = non_english_females.sample(
                n=int(desired_non_english_females), random_state=random_seed
            )
            print(
                f"Selected {len(sampled_non_english_females)} out of {num_non_english_females} Non-English female speakers to meet {100 * desired_bias_ratio / (desired_bias_ratio + 1):.0f}% male target."
            )
            selected_df_parts.append(non_english_males)
            selected_df_parts.append(sampled_non_english_females)
        else:
            print(
                f"\nWarning: Not enough Non-English female speakers ({num_non_english_females}) in target-balanced data to achieve exactly {100 * desired_bias_ratio / (desired_bias_ratio + 1):.0f}% male for Non-English speakers while keeping all available males."
            )
            if num_non_english_females > 0:
                actual_desired_non_english_males = round(
                    num_non_english_females * desired_bias_ratio
                )
                if num_non_english_males >= actual_desired_non_english_males:
                    sampled_non_english_males = non_english_males.sample(
                        n=int(actual_desired_non_english_males),
                        random_state=random_seed,
                    )
                    print(
                        f"Adjusting: Keeping all {num_non_english_females} Non-English females and sampling {len(sampled_non_english_males)} Non-English males to get closest to {100 * desired_bias_ratio / (desired_bias_ratio + 1):.0f}% male."
                    )
                    selected_df_parts.append(sampled_non_english_males)
                    selected_df_parts.append(non_english_females)
                else:
                    print(
                        f"Cannot achieve desired male ratio for Non-English speakers even by taking all available females and males. Using all available Non-English speakers for this group."
                    )
                    selected_df_parts.append(non_english_females)
                    selected_df_parts.append(non_english_males)
            else:
                print(
                    "No Non-English females available in target-balanced data. Keeping all Non-English males for this group."
                )
                selected_df_parts.append(non_english_males)

    # Concatenate the selected parts into the new DataFrame
    df_subset = pd.concat(selected_df_parts).reset_index(drop=True)

    print(
        f"\n--- New Subset DataFrame size after proportion adjustment: {len(df_subset)} rows ---"
    )

    print("\n--- Verifying Gender Proportions in the Subset ---")
    speaker_types = {1: "English Speaker", 0: "Non-English Speaker"}
    for lang_type_val in sorted(df_subset["is_english_speaker"].unique()):
        label = speaker_types.get(lang_type_val, f"Unknown ({lang_type_val})")
        subset_lang = df_subset[df_subset["is_english_speaker"] == lang_type_val]
        gender_counts = subset_lang["gender_encoded"].value_counts()

        total_in_group = gender_counts.sum()
        if total_in_group > 0:
            female_prop = gender_counts.get(0, 0) / total_in_group  # 0 for female
            male_prop = gender_counts.get(1, 0) / total_in_group  # 1 for male
            print(f"\n{label} (Total: {total_in_group}):")
            print(f"  Female (0): {int(gender_counts.get(0, 0))} ({female_prop:.2%})")
            print(f"  Male (1): {int(gender_counts.get(1, 0))} ({male_prop:.2%})")
        else:
            print(f"\n{label} has no data in the subset.")

    # --- 2. Train/Test Split ---
    print(
        f"\n--- Performing {100 * (1 - test_split_ratio):.0f}/{100 * test_split_ratio:.0f} Train/Test Split ---"
    )

    # Define cache paths for train/test dataframes, including bias ratio and target balancing in the name
    train_df_cache_path = os.path.join(
        root,
        f"train_df_split_target_balanced_test_ratio_{test_split_ratio}_bias_ratio_{desired_bias_ratio}_seed_{random_seed}.pkl",
    )
    test_df_cache_path = os.path.join(
        root,
        f"test_df_split_target_balanced_test_ratio_{test_split_ratio}_bias_ratio_{desired_bias_ratio}_seed_{random_seed}.pkl",
    )

    # Check if cached dataframes exist
    if os.path.exists(train_df_cache_path) and os.path.exists(test_df_cache_path):
        print(
            f"Loading cached train/test split from: {train_df_cache_path} and {test_df_cache_path}"
        )
        train_df = pd.read_pickle(train_df_cache_path)
        test_df = pd.read_pickle(test_df_cache_path)
    else:
        print(
            f"Cached train/test split not found or outdated. Performing new split and saving to cache."
        )
        train_df, test_df = train_test_split(
            df_subset,
            test_size=test_split_ratio,  # Use the hyperparameter here
            random_state=random_seed,
            stratify=df_subset[
                ["is_english_speaker", "gender_encoded"]
            ],  # Stratify by both target and bias for balanced splits
        )
        # Save the split dataframes to cache
        train_df.to_pickle(train_df_cache_path)
        test_df.to_pickle(test_df_cache_path)
        print(
            f"Saved train/test split to: {train_df_cache_path} and {test_df_cache_path}"
        )

    print(f"Train set size: {len(train_df)} rows")
    print(f"Test set size: {len(test_df)} rows")

    # --- 3. Generate Audio Embeddings ---
    print(
        f"\n--- Loading/Generating Audio Embeddings with {type(audio_encoder).__name__} ---"
    )

    X_train_audio_embeddings = compute_or_load_embeddings_audio(
        train_df["audio_filepath"].tolist(),
        "train",
        audio_encoder,
        root,
        device,
    )
    X_test_audio_embeddings = compute_or_load_embeddings_audio(
        test_df["audio_filepath"].tolist(),
        "test",
        audio_encoder,
        root,
        device,
    )

    # --- 4. PyTorch DataLoaders ---
    print("\n--- Creating PyTorch DataLoaders ---")

    train_dataset = CustomAudioDataset(
        audio_embeddings=X_train_audio_embeddings,
        targets=train_df["is_english_speaker"].values,
        biases=train_df["gender_encoded"].values,
    )
    test_dataset = CustomAudioDataset(
        audio_embeddings=X_test_audio_embeddings,
        targets=test_df["is_english_speaker"].values,
        biases=test_df["gender_encoded"].values,
    )

    # Note: Sampler logic is typically applied to train_loader.
    # If you use a weighted sampler, `shuffle=False` should be set for DataLoader.
    # The dummy `get_sampling_weights` is provided.
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
            shuffle=False,  # Must be False when using a sampler
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

    # Example usage with new hyperparameters
    train_loader, test_loader, train_dataset, test_dataset = (
        get_speech_accent_dataloaders(
            root="./speech_accent_data_cache",  # Specify a directory for data download and cache
            batch_size=32,
            num_workers=2,
            encoder_name="RealAudioEncoder",  # Use the RealAudioEncoder
            random_seed=42,
            test_split_ratio=0.3,  # Example: 30% for test set
            desired_bias_ratio=7,  # Example: Aim for 7:1 ratio (e.g., 87.5% female for English, 87.5% male for Non-English)
            sampler="weighted",  # Enable weighted sampling to balance target classes
        )
    )

    print("\n--- Example Batch from Train DataLoader ---")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Inputs (Audio Embeddings) shape: {batch['inputs'].shape}")
        print(
            f"  Targets (English Speaker) shape: {batch['targets'].shape}, Values: {batch['targets']}"
        )
        print(f"  Bias (Gender) shape: {batch['bias'].shape}, Values: {batch['bias']}")
        if i == 0:  # Print only the first batch for brevity
            break

    print("\n--- Example Batch from Test DataLoader ---")
    for i, batch in enumerate(test_loader):
        print(f"Batch {i+1}:")
        print(f"  Inputs (Audio Embeddings) shape: {batch['inputs'].shape}")
        print(
            f"  Targets (English Speaker) shape: {batch['targets'].shape}, Values: {batch['targets']}"
        )
        print(f"  Bias (Gender) shape: {batch['bias'].shape}, Values: {batch['bias']}")
        if i == 0:  # Print only the first batch for brevity
            break
