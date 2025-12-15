import os
import re
import json
import hashlib
from typing import List, Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


class SensorDataset(Dataset):
    '''Dataset for dynamically loading exoskeleton input and label data (real/sim).'''

    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: str,
                 participant_masses: Dict[str, float] = {},
                 action_patterns: Optional[List[str]] = None,
                 device: torch.device = torch.device("cpu"),
                 input_file_suffix: str = "_exo.csv",
                 label_file_suffix: str = "_moment_filt.csv",
                 cache_dir: str = "cache",
                 normalize_inputs: bool = True):
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.participant_masses = participant_masses
        self.action_patterns = action_patterns
        self.device = device
        self.input_file_suffix = input_file_suffix.lower()
        self.label_file_suffix = label_file_suffix.lower() if label_file_suffix else None
        self.cache_dir = cache_dir
        self.normalize_inputs = normalize_inputs
        self.trial_names = self._get_trial_names()
        self.channel_stats = self._get_or_compute_channel_stats()
        self._mean_tensor = torch.tensor(self.channel_stats["mean"], dtype=torch.float32, device=self.device).view(1, -1, 1)
        std = torch.tensor(self.channel_stats["std"], dtype=torch.float32, device=self.device)
        std = torch.clamp(std, min=1e-6)
        self._std_tensor = std.view(1, -1, 1)

        if self.action_patterns:
            print(f"  - Action patterns: {self.action_patterns}")
            print(f"  - Matched trials: {len(self.trial_names)}")

    def __len__(self):
        '''Returns number of files found.'''
        return len(self.trial_names)

    def __getitem__(self, idx: int or List[int] or slice):
        '''Loads data based on provided indices. Returns trial names along with data.'''
        # Get list of desired file names based on idx
        if isinstance(idx, list):
            trial_names = [self.trial_names[i] for i in idx]
        else:
            trial_names = self.trial_names[idx]
            trial_names = [trial_names] if not isinstance(trial_names, list) else trial_names

        # Load data
        data = [list(self._load_trial_data_train(trial_name)) for trial_name in trial_names]

        # add zero padding to allow for concatenation
        data, trial_sequence_lengths = self._add_zero_padding(data)

        # concatenate tensors
        input_data, label_data = zip(*data)
        input_data = torch.cat(input_data, dim=0)
        if self.normalize_inputs:
            input_data = (input_data - self._mean_tensor) / self._std_tensor
        label_data = torch.cat(label_data, dim=0)

        # Return trial names along with data
        return input_data, label_data, trial_sequence_lengths, trial_names

    def get_trial_names(self):
        return self.trial_names

    def extract_action_type(self, trial_name: str) -> str:
        """
        Extract action type from trial name.
        Examples:
            'BT01/normal_walk_1_0-6_on' -> 'normal_walk_0-6'
            'BT01/normal_walk_1_1_0-6_on' -> 'normal_walk_0-6'
            'BT01/normal_walk_1_shuffle_on' -> 'normal_walk_shuffle'
            'BT02/jump_1_fb_on' -> 'jump'
            'BT03/dynamic_walk_1_high-knees_on' -> 'dynamic_walk'
        """
        # Get folder name (remove participant prefix)
        if '/' in trial_name:
            folder_name = trial_name.split('/')[-1]
        elif '\\' in trial_name:
            folder_name = trial_name.split('\\')[-1]
        else:
            folder_name = trial_name

        # Split by underscore
        parts = folder_name.split('_')

        # Handle compound action names
        compound_actions = ['normal_walk', 'dynamic_walk', 'incline_walk', 'walk_backward',
                            'weighted_walk', 'obstacle_walk', 'sit_to_stand', 'curb_down',
                            'curb_up', 'lift_weight', 'side_shuffle', 'tug_of_war',
                            'turn_and_step', 'tire_run', 'start_stop', 'step_ups']

        # Check for compound actions
        if len(parts) >= 2:
            potential_compound = f"{parts[0]}_{parts[1]}"

            # Special handling for normal_walk - extract speed/type information
            if potential_compound == 'normal_walk':
                # Define normal_walk speed/type identifiers
                normal_walk_types = ['0-6', '1-2', '1-8', '2-0', '2-5', 'shuffle', 'skip']

                # Search for speed/type identifier in the remaining parts
                for part in parts[2:]:  # Start from index 2 (after 'normal_walk')
                    if part in normal_walk_types:
                        return f"normal_walk_{part}"

                # If no specific type found, return just 'normal_walk'
                return 'normal_walk'

            # For other compound actions, return as before
            elif potential_compound in compound_actions:
                return potential_compound

        # Return first part as action type
        return parts[0]

    def _match_action_patterns(self, trial_folder_name: str) -> bool:
        """Check if trial folder name matches any specified regex patterns."""
        if not self.action_patterns:
            return True

        for pattern in self.action_patterns:
            if re.match(pattern, trial_folder_name):
                return True
        return False

    def _cache_key(self) -> str:
        key = json.dumps({
            "data_dir": self.data_dir,
            "side": self.side,
            "action_patterns": self.action_patterns,
            "input_suffix": self.input_file_suffix,
        }, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()

    def _load_cached_trials(self) -> Optional[List[str]]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"trials_{self._cache_key()}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("data_dir") == self.data_dir:
                    return data.get("trials", [])
            except Exception:
                return None
        return None

    def _save_cached_trials(self, trials: List[str]) -> None:
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"trials_{self._cache_key()}.json")
        payload = {
            "data_dir": self.data_dir,
            "trials": trials,
            "side": self.side,
            "action_patterns": self.action_patterns,
        }
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass

    def _get_trial_names(self):
        '''Get all trial names in data_dir, filtered by action patterns if specified.'''
        cached = self._load_cached_trials()
        if cached is not None:
            print(f"Loaded cached trial list ({len(cached)} entries) from {self.cache_dir}")
            return cached

        participants = [participant for participant in os.listdir(self.data_dir)
                        if "." not in participant and participant != "LICENSE"]

        trial_names = []
        action_stats = {}

        for participant in participants:
            participant_dir = os.path.join(self.data_dir, participant)
            for trial_name in os.listdir(participant_dir):
                if self._match_action_patterns(trial_name):
                    full_trial_name = os.path.join(participant, trial_name)
                    trial_names.append(full_trial_name)

                    # Statistics
                    action_type = self.extract_action_type(full_trial_name)
                    action_stats[action_type] = action_stats.get(action_type, 0) + 1

        if self.action_patterns and action_stats:
            print(f"  - Action distribution: {action_stats}")

        self._save_cached_trials(trial_names)
        return trial_names

    def _channel_stats_cache_path(self) -> Optional[str]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        key = json.dumps({
            "data_dir": self.data_dir,
            "side": self.side,
            "inputs": self.input_names,
            "input_suffix": self.input_file_suffix,
        }, sort_keys=True)
        cache_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"channel_stats_{cache_hash}.json")

    def _get_or_compute_channel_stats(self) -> Dict[str, List[float]]:
        cache_path = self._channel_stats_cache_path()
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("input_names") == self.input_names:
                    return data
            except Exception:
                pass

        channel_count = len(self.input_names)
        sum_vec = torch.zeros(channel_count)
        sum_sq = torch.zeros(channel_count)
        count = torch.zeros(channel_count)

        for trial in self.trial_names:
            inputs, _ = self._load_trial_data_train(trial)
            data = inputs.squeeze(0)  # (C, T)
            mask = torch.isfinite(data)
            safe_data = torch.where(mask, data, torch.zeros_like(data))
            sum_vec += safe_data.sum(dim=1)
            sum_sq += (safe_data ** 2).sum(dim=1)
            count += mask.sum(dim=1)

        count = torch.clamp(count, min=1.0)
        mean = sum_vec / count
        var = torch.clamp(sum_sq / count - mean ** 2, min=1e-6)
        std = torch.sqrt(var)

        stats = {
            "input_names": self.input_names,
            "mean": mean.tolist(),
            "variance": var.tolist(),
            "std": std.tolist(),
            "norm_variance": (var / torch.clamp(std ** 2, min=1e-6)).tolist(),
        }
        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, ensure_ascii=False)
            except Exception:
                pass
        return stats

    @property
    def channel_std(self) -> List[float]:
        return self.channel_stats["std"]
    @property
    def channel_variance(self) -> List[float]:
        return self.channel_stats.get("norm_variance", self.channel_stats["variance"])

    def get_channel_std_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        tensor = torch.tensor(self.channel_std, dtype=torch.float32, device=device)
        return tensor.view(1, -1, 1)

    def _load_trial_data_train(self, trial_name: str):
        '''Loads data from a single trial.'''
        trial_dir = os.path.join(self.data_dir, trial_name)
        input_file_path = self._find_input_file(trial_dir)

        participant = trial_name.split("/")[0].split("\\")[0]
        if participant not in self.participant_masses:
            print(f"Warning - {participant} mass was not provided.")
        input_data = self._load_input_data(input_file_path, body_mass=self.participant_masses.get(participant, 1.))

        label_data = self._load_label_data(trial_dir)

        return input_data, label_data

    def _find_input_file(self, trial_dir: str) -> str:
        for file in os.listdir(trial_dir):
            file_lower = file.lower()
            if not file_lower.endswith(self.input_file_suffix):
                continue
            if self.input_file_suffix.endswith("_exo.csv") and file_lower.endswith("power_exo.csv"):
                continue
            return os.path.join(trial_dir, file)
        raise FileNotFoundError(
            f"No file ending with '{self.input_file_suffix}' found in {trial_dir}"
        )

    def _load_input_data(self, file_path: str, body_mass: float):
        '''Loads input data from a single file and returns as a 3D torch.FloatTensor.'''
        df = pd.read_csv(file_path)

        df.loc[:, "insole_l_force_y"] /= body_mass
        df.loc[:, "insole_r_force_y"] /= body_mass

        if self.side == "l":
            df.loc[:, "foot_imu_l_gyro_x"] *= -1.
            df.loc[:, "foot_imu_l_gyro_y"] *= -1.
            df.loc[:, "foot_imu_l_accel_z"] *= -1.
            df.loc[:, "shank_imu_l_gyro_x"] *= -1.
            df.loc[:, "shank_imu_l_gyro_y"] *= -1.
            df.loc[:, "shank_imu_l_accel_z"] *= -1.
            df.loc[:, "thigh_imu_l_gyro_x"] *= -1.
            df.loc[:, "thigh_imu_l_gyro_y"] *= -1.
            df.loc[:, "thigh_imu_l_accel_z"] *= -1.
            df.loc[:, "insole_l_cop_z"] *= -1.

        input_data = torch.tensor(df[self.input_names].values, device=self.device).transpose(0, 1).unsqueeze(0).float()
        return input_data

    def _load_label_data(self, trial_dir: str):
        '''Loads label data from a single file and returns as a 3D torch.FloatTensor.'''
        if not self.label_file_suffix:
            raise RuntimeError("label_file_suffix 未设置，无法加载标签数据")
        label_path = None
        for file in os.listdir(trial_dir):
            file_lower = file.lower()
            if file_lower.endswith(self.label_file_suffix):
                label_path = os.path.join(trial_dir, file)
                break
        if label_path is None:
            raise FileNotFoundError(
                f"No file ending with '{self.label_file_suffix}' found in {trial_dir}"
            )
        df = pd.read_csv(label_path)
        label_data = torch.tensor(df[self.label_names].values, device=self.device).transpose(0, 1).unsqueeze(0).float()
        return label_data

    def _add_zero_padding(self, data: List[List[torch.FloatTensor]]):
        '''Adds zero padding to the end of each trial to match sequence lengths.'''
        trial_sequence_lengths = [trial_data[0].shape[-1] for trial_data in data]
        max_sequence_length = max(trial_sequence_lengths)

        for i in range(len(data)):
            trial_sequence_length = trial_sequence_lengths[i]
            if trial_sequence_length < max_sequence_length:
                padding_length = max_sequence_length - trial_sequence_length
                for j in range(len(data[i])):
                    data[i][j] = torch.cat(
                        (data[i][j], torch.zeros((1, data[i][j].shape[1], padding_length), device=self.device)), dim=2)

        return data, trial_sequence_lengths
