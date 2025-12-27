# python standard library
import os
from typing import Dict, List, Tuple, Optional
import pickle

# third-party library
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import hydra

# custom library


class HDF5Reader:
    @staticmethod
    def read(
        file_path: str,
        sdo_wavelengths: List[str],
        omni_variables: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sdo_data = {}
        omni_data = {}
        
        with h5py.File(file_path, 'r') as f:
            for wavelength in sdo_wavelengths:
                dataset_name = f"sdo/{wavelength}"
                if dataset_name not in f:
                    raise KeyError(f"SDO wavelength {wavelength} not found")
                sdo_data[wavelength] = f[dataset_name][:]

            for variable in omni_variables:
                dataset_name = f"omni/{variable}"
                if dataset_name not in f:
                    raise KeyError(f"OMNI variable {variable} not found")
                omni_data[variable] = f[dataset_name][:]
        
        return sdo_data, omni_data
    

class Normalizer:
    def __init__(self, stat_dict: Optional[Dict[str, Dict[str, float]]] = None):
        self.stat_dict = stat_dict

    def normalize_sdo(self, data: np.ndarray) -> np.ndarray:
        # 현재 8-bit 영상 사용 중.
        # 향후 raw 데이터 사용 시 수정 해야 함.
        return (data * (2.0 / 255.0)) - 1.0
    
    def normalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        if variable not in self.stat_dict:
            raise KeyError(f"Statistics not found for variable: {variable}")

        mean = self.stat_dict[variable]['mean']
        std = self.stat_dict[variable]['std']
        return (data - mean) / std
    
    def denormalize_omni(self, data: np.ndarray, variable: str) -> np.ndarray:
        if variable not in self.stat_dict:
            raise KeyError(f"Statistics not found for variable: {variable}")

        mean = self.stat_dict[variable]['mean']
        std = self.stat_dict[variable]['std']
        return data * std + mean


class OnlineStatistics:
    """
    Compute mean and std using Welford's online algorithm.
    Memory efficient - O(1) space complexity.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
    
    def update(self, batch: np.ndarray):
        """
        Update statistics with a new batch of data.
        
        Args:
            batch: Data array of any shape
        """
        # Flatten and filter finite values
        values = batch.flatten()
        valid_values = values[np.isfinite(values)]
        
        for x in valid_values:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
    
    @property
    def std(self) -> float:
        """Return standard deviation."""
        if self.n < 2:
            return 1.0  # Fallback for insufficient data
        return float(np.sqrt(self.M2 / self.n))
    
    def get_stats(self) -> Dict[str, float]:
        """Return statistics as dictionary."""
        return {
            'mean': float(self.mean),
            'std': self.std
        }


def compute_statistics(
    stat_file_path: str,
    data_root: str,
    data_dir_name: str,
    data_file_list: List[str],
    variables: List[str],    
    overwrite: bool = False
) -> Dict[str, Dict[str, float]]:
    if os.path.exists(stat_file_path) and not overwrite:
        try:
            with open(stat_file_path, 'rb') as f:
                loaded_stats = pickle.load(f)
            
            # Verify all variables are present
            if all(var in loaded_stats for var in variables):
                print(f"Loaded statistics from {stat_file_path}")
                return {var: loaded_stats[var] for var in variables}
            else:
                print("Incomplete statistics, recomputing...")
        except (pickle.PickleError, KeyError) as e:
            print(f"Failed to load statistics: {e}, recomputing...")
    
    # Filter for h5 files only
    h5_files = [f"{data_root}/{data_dir_name}/{f}" for f in data_file_list if f.endswith('.h5')]
    
    if not h5_files:
        raise ValueError("No valid .h5 files found in data file list")
    
    # Initialize online statistics for each variable
    stats_computers = {var: OnlineStatistics() for var in variables}
    
    # Process files
    valid_files = 0
    for file_path in h5_files:
        if not os.path.exists(file_path):
            continue
        
        try:
            with h5py.File(file_path, 'r') as f:
                for variable in variables:
                    dataset_name = f"omni/{variable}"
                    if dataset_name in f:
                        data = f[dataset_name][:]
                        stats_computers[variable].update(data)
            
            valid_files += 1
            
            if valid_files % 100 == 0:
                print(f"Processed {valid_files}/{len(h5_files)} files...")
                
        except (OSError, KeyError) as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            continue
    
    if valid_files == 0:
        raise ValueError("No valid data files found for statistics computation")
    
    # Compile final statistics
    stat_dict = {}
    for variable in variables:
        stat_dict[variable] = stats_computers[variable].get_stats()
        print(f"{variable}: mean={stat_dict[variable]['mean']:.3f}, "
              f"std={stat_dict[variable]['std']:.3f}")
    
    # Save statistics
    try:
        os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
        with open(stat_file_path, 'wb') as f:
            pickle.dump(stat_dict, f)
        print(f"Statistics saved to {stat_file_path}")
    except (OSError, pickle.PickleError) as e:
        print(f"Warning: Failed to save statistics: {e}")
    
    return stat_dict


def split_by_class(file_list: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    positive = []
    negative = []
    for file_name, label in file_list:
        if label == 0:
            negative.append((file_name, label))
        else:
            positive.append((file_name, label))
    positive.sort(key=lambda x: x[0])
    negative.sort(key=lambda x: x[0])
    return positive, negative


def undersample(
    file_list: List[Tuple[str, int]],
    num_subsample: int,
    subsample_index: int
) -> Tuple[List[Tuple[str, int]], int, int]:
    positive, negative = split_by_class(file_list)
    
    # Shuffle negative samples using global random state
    import random
    random.shuffle(negative)
    
    # Split into folds
    n = len(negative)
    base_size = n // num_subsample
    remainder = n % num_subsample
    
    start = 0
    for i in range(num_subsample):
        size = base_size + (1 if i < remainder else 0)
        if i == subsample_index:
            selected_negative = negative[start:start + size]
            break
        start += size
    
    # Combine positive and selected negative
    sampled_list = positive + selected_negative
    
    return sampled_list, len(positive), len(negative)


class BaseDataset(Dataset):
    def __init__(self, config):
        self.data_root = config.environment.data_root
        self.data_dir_name = config.data.data_dir_name
        self.dataset_name = config.data.dataset_name

        self.sdo_wavelengths = sorted(config.data.sdo_wavelengths)
        self.sdo_start_index = config.data.sdo_start_index
        self.sdo_end_index = config.data.sdo_end_index

        self.input_variables = sorted(config.data.input_variables)
        self.input_start_index = config.data.input_start_index
        self.input_end_index = config.data.input_end_index

        self.target_variables = sorted(config.data.target_variables)
        self.target_start_index = config.data.target_start_index
        self.target_end_index = config.data.target_end_index
        self.omni_variables = list(set(
            self.input_variables + self.target_variables
        ))

        self.target_days = config.experiment.target_days

        train_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_name}_train.csv"
        )
        self.train_list = self._load_file_list(train_list_path)

        validation_list_path = os.path.join(
            self.data_root,
            f"{self.dataset_name}_validation.csv"
        )
        self.validation_list = self._load_file_list(validation_list_path)

        statistics_file_path = os.path.join(
            self.data_root,
            f"{self.dataset_name}_validation.pkl"
        )
        self.stat_dict = compute_statistics(
            stat_file_path = statistics_file_path,
            data_root = self.data_root,
            data_dir_name = self.data_dir_name,
            data_file_list = self.train_list,
            variables = self.omni_variables,
            overwrite=False
        )
        self.normalizer = Normalizer(stat_dict=self.stat_dict)

    def _load_file_list(self, csv_path: str) -> List[str]:
        df = pd.read_csv(csv_path)
        file_names = df['file_name'].tolist()
        list_labels = {}
        labels = []
        for day in self.target_days :
            key = f'class_day{day}'
            list_labels[key] = df[key].tolist()
        for idx in range(len(file_names)):
            tmp = []
            for day in self.target_days :
                key = f'class_day{day}'
                tmp.append(list_labels[key][idx])
            labels.append(max(tmp))
        return list(zip(file_names, labels))
    
    def process_sdo(self, sdo_data: Dict[str, np.ndarray]) -> np.ndarray:
        sdo_arrays = []
        for wavelength in self.sdo_wavelengths:
            data = sdo_data[wavelength]
            # Normalize [0, 255] -> [-1, 1]
            data = self.normalizer.normalize_sdo(data)
            sdo_arrays.append(data)
        # Stack: (T, C, H, W)
        sdo_array = np.concatenate(sdo_arrays, axis=1)
        # Select timesteps
        sdo_array = sdo_array[self.sdo_start_index:self.sdo_end_index]
        # Transpose to (C, T, H, W)
        sdo_array = np.transpose(sdo_array, (1, 0, 2, 3))
        return sdo_array
    
    def process_omni_input(self, omni_data: Dict[str, np.ndarray]) -> np.ndarray:
        omni_arrays = []
        for variable in self.input_variables:
            data = omni_data[variable]
            # Z-score normalization
            data = self.normalizer.normalize_omni(data, variable)
            omni_arrays.append(data)
        # Stack: (T, C)
        omni_array = np.stack(omni_arrays, axis=-1)
        # Select timesteps
        omni_array = omni_array[self.input_start_index:self.input_end_index]
        return omni_array
    
    def process_omni_target(self, omni_data: Dict[str, np.ndarray]) -> np.ndarray:
        omni_arrays = []
        for variable in self.target_variables:
            data = omni_data[variable]
            # Z-score normalization
            data = self.normalizer.normalize_omni(data, variable)
            omni_arrays.append(data)
        # Stack: (T, C)
        omni_array = np.stack(omni_arrays, axis=-1)
        # Select timesteps
        omni_array = omni_array[self.target_start_index:self.target_end_index]
        return omni_array


class TrainDataset(BaseDataset):
    def __init__(self, config):
        super(TrainDataset, self).__init__(config)

        self.enable_undersampling = config.experiment.enable_undersampling
        self.num_subsample = config.experiment.num_subsample
        self.subsample_index = config.experiment.subsample_index

        if self.enable_undersampling:
            subsample, _, _ = undersample(
                self.train_list,
                num_subsample=self.num_subsample,
                subsample_index=self.subsample_index
            )
            self.file_list = subsample
        else :
            self.file_list = self.train_list
        self.num = len(self.file_list)

    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        file_path = os.path.join(
            self.data_root,
            self.data_dir_name,
            file_name
        )

        sdo_data, omni_data = HDF5Reader.read(
            file_path,
            self.sdo_wavelengths,
            self.omni_variables
        )

        sdo_array = self.process_sdo(sdo_data)
        input_array = self.process_omni_input(omni_data)
        target_array = self.process_omni_target(omni_data)
        label_array = np.array([[label]], dtype=np.float32)

        data_dict = {
            'sdo': torch.tensor(sdo_array, dtype=torch.float32),
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            "file_names" : file_name
        }
        return data_dict


class ValidationDataset(BaseDataset):
    def __init__(self, config):
        super(ValidationDataset, self).__init__(config)
        self.file_list = self.validation_list
        self.num = len(self.file_list)

    def __len__(self):
        return self.num
    
    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        file_path = os.path.join(
            self.data_root,
            self.data_dir_name,
            file_name
        )

        sdo_data, omni_data = HDF5Reader.read(
            file_path,
            self.sdo_wavelengths,
            self.omni_variables
        )

        sdo_array = self.process_sdo(sdo_data)
        input_array = self.process_omni_input(omni_data)
        target_array = self.process_omni_target(omni_data)
        label_array = np.array([[label]], dtype=np.float32)

        data_dict = {
            'sdo': torch.tensor(sdo_array, dtype=torch.float32),
            'inputs': torch.tensor(input_array, dtype=torch.float32),
            'targets': torch.tensor(target_array, dtype=torch.float32),
            'labels': torch.tensor(label_array, dtype=torch.float32),
            "file_names" : file_name
        }
        return data_dict


class TestDataset(BaseDataset):
    def __init__(self, config):
        super(TestDataset, self).__init__(config)


def create_dataloader(config, phase="train") -> DataLoader:
    phase = phase.upper()
    if phase == "TRAIN":
        dataset = TrainDataset(config)
    elif phase == "VALIDATION":
        dataset = ValidationDataset(config)
    elif phase == "TEST" :
        dataset = TestDataset(config)
    else :
        raise ValueError(f"Not valid phase: {phase}")
        
    dataloader = DataLoader(
        dataset,
        batch_size = config.experiment.batch_size,
        shuffle = phase == "TRAIN",
        num_workers = config.environment.num_workers,
        pin_memory = config.environment.device == 'cuda',
        drop_last = False,
        persistent_workers = (config.environment.num_workers > 0),
    )

    print(f"{phase} dataloader created")
    print(f"# of data: {len(dataset)} # of batch: {len(dataloader)}")    

    return dataloader


@hydra.main(config_path="./configs", version_base=None)
def main(config):
    train_dataloader = create_dataloader(config, "train")
    validation_dataloader = create_dataloader(config, "validation")

    print("train dataloader")
    for i, data_dict in enumerate(train_dataloader):
        for k, v in data_dict.items():
            try :
                v = v.numpy()
                print(k, v.shape, v.min(), v.max(), v.mean(), v.std())
            except :
                print(k, len(v))
        break

    print("")
    print("validation dataloader")
    for i, data_dict in enumerate(validation_dataloader):
        for k, v in data_dict.items():
            try :
                v = v.numpy()
                print(k, v.shape, v.min(), v.max(), v.mean(), v.std())
            except :
                print(k, len(v))
        break


if __name__ == "__main__" :
    main()
