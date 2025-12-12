"""
Unified data loading utilities for training and validation.
"""
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
from typing import List, Tuple, Any, Optional
from domain_adaptation.data.sensor_dataset import SensorDataset
import os
import json
import hashlib
from datetime import datetime
from tqdm import tqdm


class DataManager:
    """Handle dataset loading and preprocessing."""

    @staticmethod
    def load_datasets(
            config: Any,
            device: torch.device,
            data_dirs: Optional[List[str]] = None,
    ) -> ConcatDataset:
        """Load all datasets from configured paths.

        Args:
            config: Configuration object
            device: Device to load data to
            data_dirs: Optional override for data directories. If None, uses config.data_dirs
        """
        print("Loading dataset...")

        # Use provided data_dirs or fall back to config.data_dirs
        if data_dirs is None:
            data_dirs = config.data_dirs

        action_patterns = getattr(config, 'action_patterns', None)
        if action_patterns:
            print(f"Filtering actions with patterns: {action_patterns}")

        datasets = []
        sides = config.side if isinstance(config.side, list) else [config.side]

        for data_dir in data_dirs:
            for side in sides:
                dataset = SensorDataset(
                    data_dir=data_dir,
                    input_names=[name.replace("*", side) for name in config.input_names],
                    label_names=[name.replace("*", side) for name in config.label_names],
                    side=side,
                    participant_masses=config.participant_masses,
                    action_patterns=action_patterns,
                    device=device
                )
                datasets.append(dataset)
                print(f"  - Loaded {len(dataset)} trials from {data_dir} (side: {side})")

        full_dataset = ConcatDataset(datasets)
        print(f"Total dataset size: {len(full_dataset)} trials")

        return full_dataset

    @staticmethod
    def load_sim_datasets(
            config: Any,
            device: torch.device,
            data_dirs: Optional[List[str]] = None,
    ) -> ConcatDataset:
        """Load simulated sensor datasets from *_imu_sim.csv files."""
        print("Loading simulated sensor dataset...")

        if data_dirs is None:
            data_dirs = config.data_dirs

        action_patterns = getattr(config, 'action_patterns', None)
        if action_patterns:
            print(f"Filtering actions with patterns: {action_patterns}")

        datasets = []
        sides = config.side if isinstance(config.side, list) else [config.side]

        for data_dir in data_dirs:
            for side in sides:
                dataset = SensorDataset(
                    data_dir=data_dir,
                    input_names=[name.replace("*", side) for name in config.input_names],
                    label_names=[name.replace("*", side) for name in config.label_names],
                    side=side,
                    participant_masses=config.participant_masses,
                    action_patterns=action_patterns,
                    device=device,
                    input_file_suffix="_exo_sim.csv",
                    label_file_suffix="_moment_filt_bio.csv",
                )
                datasets.append(dataset)
                print(f"  - Loaded {len(dataset)} sim trials from {data_dir} (side: {side})")

        full_dataset = ConcatDataset(datasets)
        print(f"Total simulated dataset size: {len(full_dataset)} trials")
        return full_dataset

    @staticmethod
    def get_or_compute_valid_indices(
            full_dataset: ConcatDataset,
            config: Any,
            cache_dir: str = 'cache',
            cache_suffix: str = ''
    ) -> List[int]:
        """Get or compute valid indices (non-NaN samples) with caching.

        Args:
            full_dataset: The full dataset to validate
            config: Configuration object
            cache_dir: Directory for cache files
            cache_suffix: Additional suffix for cache file naming
        """
        if not getattr(config, 'filter_nan_trials', True):
            return list(range(len(full_dataset)))

        os.makedirs(cache_dir, exist_ok=True)

        if isinstance(config.side, list):
            side_str = '_'.join(sorted(config.side))
        else:
            side_str = config.side

        action_str = ""
        if hasattr(config, 'action_patterns') and config.action_patterns:
            action_str = hashlib.md5(str(config.action_patterns).encode()).hexdigest()[:8]

        # Add cache suffix to differentiate train/test caches
        cache_key = str(getattr(config, 'data_dirs', '')) + str(config.input_names) + side_str + action_str + cache_suffix
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f'valid_indices_{cache_hash}.json')

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                if (cache_data['total_trials'] == len(full_dataset) and
                        cache_data.get('side') == (side_str if isinstance(config.side, list) else config.side) and
                        cache_data.get('action_patterns_hash', '') == action_str):
                    print(f"Loaded cached valid indices: {len(cache_data['valid_indices'])} valid trials")
                    return cache_data['valid_indices']
            except:
                pass

        print("Filtering trials with NaN...")
        valid_indices = []

        if isinstance(config.side, list):
            num_sides = len(config.side)
            print(f"Checking trials for {num_sides} sides...")

        for i in tqdm(range(len(full_dataset)), desc="Checking trials"):
            # Note: This now returns 4 items
            inputs, labels, seq_lengths, trial_names = full_dataset[i]
            if not torch.isnan(inputs).any() and not torch.isnan(labels).any():
                valid_indices.append(i)

        print(f"Valid trials: {len(valid_indices)}/{len(full_dataset)} "
              f"({100 * len(valid_indices) / len(full_dataset):.1f}%)")

        if isinstance(config.side, list):
            trials_per_side = len(valid_indices) // len(config.side)
            print(f"  Per side: approximately {trials_per_side} trials")

        cache_data = {
            'valid_indices': valid_indices,
            'total_trials': len(full_dataset),
            'side': side_str if isinstance(config.side, list) else config.side,
            'action_patterns_hash': action_str,
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f"Saved cache to: {cache_path}")

        return valid_indices

    @staticmethod
    def create_random_train_val_split(
        config: Any,
        val_split: float = 0.1,
        device: Optional[torch.device] = torch.device("cpu"),
        max_samples: Optional[int] = None,
    ) -> Tuple[Subset, Subset]:
        """Create train/validation split from dataset using random split."""

        val_split = getattr(config, 'val_split', 0.1)

        full_dataset = DataManager.load_datasets(
            config=config,
            device=device,
        )

        valid_indices = DataManager.get_or_compute_valid_indices(full_dataset, config)

        if max_samples and len(valid_indices) > max_samples:
            valid_indices = valid_indices[:max_samples]
            print(f"Limited to {max_samples} samples")

        filtered_dataset = Subset(full_dataset, valid_indices)

        val_size = int(len(filtered_dataset) * val_split)
        train_size = len(filtered_dataset) - val_size
        train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

        print(f"Dataset split (random): {train_size} train trials, {val_size} validation trials")

        return train_dataset, val_dataset

    @staticmethod
    def create_manual_split(
            config: Any,
            device: torch.device,
            max_samples: Optional[int] = None
    ) -> Tuple[Subset, Subset]:
        """Create train/test split from dataset using manual directory-based splitting.

        Args:
            config: Configuration object containing train_data_dirs and val_dataset
            device: Device to load data to
            max_samples: Optional maximum number of samples per dataset

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        print("=" * 50)
        print("Creating manual train/test split by directories")
        print("=" * 50)

        # Load training dataset
        print("\n[Training Dataset]")

        train_full_dataset = DataManager.load_datasets(
            config, device, config.train_data_dirs,
        )

        train_valid_indices = DataManager.get_or_compute_valid_indices(
            train_full_dataset, config, cache_suffix='_train'
        )

        if max_samples and len(train_valid_indices) > max_samples:
            train_valid_indices = train_valid_indices[:max_samples]
            print(f"Limited training set to {max_samples} samples")

        train_dataset = Subset(train_full_dataset, train_valid_indices)

        # Load test dataset (always use original mode)
        print("\n[Test Dataset]")
        test_full_dataset = DataManager.load_datasets(
            config, device, config.val_dataset,
        )
        test_valid_indices = DataManager.get_or_compute_valid_indices(
            test_full_dataset, config, cache_suffix='_test'
        )
        if max_samples and len(test_valid_indices) > max_samples:
            test_valid_indices = test_valid_indices[:max_samples]
            print(f"Limited test set to {max_samples} samples")

        test_dataset = Subset(test_full_dataset, test_valid_indices)

        print("\n" + "=" * 50)
        print(f"Final dataset split (manual):")
        print(f"  - Training: {len(train_dataset)} trials from {len(config.train_data_dirs)} directories")
        print(f"  - Testing:  {len(test_dataset)} trials from {len(config.val_dataset)} directories")
        print("=" * 50 + "\n")

        return train_dataset, test_dataset

    @staticmethod
    def create_dataloaders_trail(
        train_dataset: Subset,
        val_dataset: Subset,
        batch_size: int,
        device: torch.device
    ) -> Tuple[DataLoader, DataLoader]:
        """Create DataLoaders for training and validation/test."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: DataManager.collate_function(x, device)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: DataManager.collate_function(x, device)
        )

        return train_loader, val_loader


    @staticmethod
    def collate_function(batch: List, device: torch.device) -> Tuple:
        """Custom collate function for batching sequences with trial names or window info."""
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        seq_lengths = [item[2][0] for item in batch]

        # Handle trial names or window metadata
        metadata = []
        for item in batch:
            # item[3] could be trial names (list of strings) or window metadata (list of dicts)
            if isinstance(item[3][0], str):
                # Original trial names
                if len(item[3]) == 1:  # Single trial
                    metadata.append(item[3][0])
                else:  # Multiple trials (shouldn't happen in typical usage)
                    metadata.extend(item[3])
            else:
                # Window metadata (dict)
                metadata.extend(item[3])

        # Remove extra dimensions
        inputs = [x.squeeze(0) for x in inputs]
        labels = [x.squeeze(0) for x in labels]

        # Calculate max sequence length in batch
        max_seq_len = max([x.shape[-1] for x in inputs])

        # Pad to same length
        padded_inputs = []
        padded_labels = []
        for inp, lab in zip(inputs, labels):
            pad_length = max_seq_len - inp.shape[-1]
            padded_inp = torch.nn.functional.pad(inp, (0, pad_length), mode='constant', value=0)
            padded_lab = torch.nn.functional.pad(lab, (0, pad_length), mode='constant', value=0)
            padded_inputs.append(padded_inp)
            padded_labels.append(padded_lab)

        return torch.stack(padded_inputs), torch.stack(padded_labels), seq_lengths, metadata
