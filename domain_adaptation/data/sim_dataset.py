"""Dataset for simulated sensor data (*_imu_sim.csv)."""
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


_LEFT_SIGN_FLIP_COLS = [
    "foot_imu_l_gyro_x",
    "foot_imu_l_gyro_y",
    "foot_imu_l_accel_z",
    "shank_imu_l_gyro_x",
    "shank_imu_l_gyro_y",
    "shank_imu_l_accel_z",
    "thigh_imu_l_gyro_x",
    "thigh_imu_l_gyro_y",
    "thigh_imu_l_accel_z",
    "insole_l_cop_z",
]


class SimDataset(Dataset):
    """Dataset that mirrors TcnDataset but loads *_imu_sim.csv. """

    def __init__(
        self,
        data_dir: str,
        sim_input_names: List[str],
        label_names: List[str],
        side: str,
        participant_masses: Dict[str, float],
        action_patterns: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.data_dir = data_dir
        self.sim_input_names = sim_input_names
        self.label_names = label_names
        self.side = side
        self.participant_masses = participant_masses
        self.action_patterns = action_patterns
        self.device = device
        self.trial_names = self._get_trial_names()

        if self.action_patterns:
            print(f"  - Action patterns: {self.action_patterns}")
            print(f"  - Matched trials: {len(self.trial_names)}")

    def __len__(self) -> int:
        return len(self.trial_names)

    def __getitem__(self, idx: Union[int, List[int], slice]):
        if isinstance(idx, list):
            trial_names = [self.trial_names[i] for i in idx]
        else:
            trial_names = self.trial_names[idx]
            trial_names = [trial_names] if not isinstance(trial_names, list) else trial_names

        data = [list(self._load_trial_data(trial_name)) for trial_name in trial_names]
        data, trial_sequence_lengths = self._add_zero_padding(data)

        input_data, label_data = zip(*data)
        input_data = torch.cat(input_data, dim=0)
        label_data = torch.cat(label_data, dim=0)
        return input_data, label_data, trial_sequence_lengths, trial_names

    def _get_trial_names(self) -> List[str]:
        participants = [p for p in os.listdir(self.data_dir) if "." not in p and p != "LICENSE"]
        trial_names: List[str] = []
        for participant in participants:
            participant_dir = os.path.join(self.data_dir, participant)
            for trial in os.listdir(participant_dir):
                if self._match_action_patterns(trial):
                    trial_names.append(os.path.join(participant, trial))
        return trial_names

    def _match_action_patterns(self, trial_folder_name: str) -> bool:
        if not self.action_patterns:
            return True
        for pattern in self.action_patterns:
            if re.match(pattern, trial_folder_name):
                return True
        return False

    def _load_trial_data(self, trial_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        trial_dir = os.path.join(self.data_dir, trial_name)
        sim_path = None
        for file in os.listdir(trial_dir):
            file_lower = file.lower()
            if file_lower.endswith("_imu_sim.csv"):
                sim_path = os.path.join(trial_dir, file)
                break
        if sim_path is None:
            raise FileNotFoundError(f"No *_imu_sim.csv found under {trial_dir}")

        participant = trial_name.split("/")[0].split("\\")[0]
        body_mass = self.participant_masses.get(participant, 1.0)
        input_data = self._load_input_data(sim_path, body_mass=body_mass)

        label_path = None
        for file in os.listdir(trial_dir):
            file_lower = file.lower()
            if file_lower.endswith("_moment_filt_bio.csv"):
                label_path = os.path.join(trial_dir, file)
                break
        if label_path is None:
            raise FileNotFoundError(f"No *_moment_filt_bio.csv found under {trial_dir}")
        label_data = self._load_label_data(label_path)

        return input_data, label_data

    def _load_input_data(self, file_path: str, body_mass: float) -> torch.Tensor:
        df = pd.read_csv(file_path)
        for col in ("insole_l_force_y", "insole_r_force_y"):
            if col in df.columns:
                df.loc[:, col] /= body_mass

        if self.side == "l":
            for col in _LEFT_SIGN_FLIP_COLS:
                if col in df.columns:
                    df.loc[:, col] *= -1.0

        missing_cols = [col for col in self.sim_input_names if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns missing in {file_path}: {missing_cols}")

        tensor = (
            torch.tensor(df[self.sim_input_names].values, device=self.device)
            .transpose(0, 1)
            .unsqueeze(0)
            .float()
        )
        return tensor

    def _load_label_data(self, file_path: str) -> torch.Tensor:
        df = pd.read_csv(file_path)
        tensor = (
            torch.tensor(df[self.label_names].values, device=self.device)
            .transpose(0, 1)
            .unsqueeze(0)
            .float()
        )
        return tensor

    def _add_zero_padding(self, data: List[List[torch.Tensor]]):
        sequence_lengths = [trial[0].shape[-1] for trial in data]
        max_length = max(sequence_lengths)
        for idx, trial in enumerate(data):
            length = sequence_lengths[idx]
            if length < max_length:
                pad_len = max_length - length
                zero_pad = torch.zeros((1, trial[0].shape[1], pad_len), device=self.device)
                data[idx][0] = torch.cat((trial[0], zero_pad), dim=2)
                zero_pad_label = torch.zeros((1, trial[1].shape[1], pad_len), device=self.device)
                data[idx][1] = torch.cat((trial[1], zero_pad_label), dim=2)
        return data, sequence_lengths
