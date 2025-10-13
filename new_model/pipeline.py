import os
import sys
import time
import pickle
import random
from glob import glob
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
import hydra


def read_h5(file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Read data from HDF5 file.
    
    Args:
        file_path: Path to HDF5 file.
        input_variables: List of input variable names.
        target_variables: List of target variable names.
        
    Returns:
        Tuple of (sdo_193, sdo_211, omni_inputs, omni_targets).
        
    Raises:
        FileNotFoundError: If file doesn't exist.
        KeyError: If required datasets are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        with h5py.File(file_path, 'r') as f:
            sdo = f['sdo'][:]  
            inputs = f['inputs'][:]
            targets = f['targets'][:]
            labels = f['labels'][()]
    except (OSError, h5py.error.HDF5Error) as e:
        raise OSError(f"Failed to read HDF5 file {file_path}: {e}")

    return sdo, inputs, targets, labels


def get_statistics(stat_file_path: str, dataset_path:str,
                   file_list: List[str], overwrite: bool = False) -> Dict[str, Dict[str, float]]:
    """Compute and cache statistics for data normalization.
    
    Args:
        stat_file_path: Path to save/load statistics pickle file.
        data_root: Root directory containing data files.
        data_file_list: List of data file names.
        variables: List of variable names to compute statistics for.
        overwrite: Whether to recompute statistics even if cache exists.
        
    Returns:
        Dictionary containing mean and std for each variable.
        
    Raises:
        FileNotFoundError: If data files cannot be found.
        ValueError: If no valid data is found for statistics computation.
    """
    # Filter for h5 files only

    data_file_list = []
    for (file_name, file_class) in file_list :
        data_file_list.append(f"{dataset_path}/{file_class}/{file_name}")

    if not data_file_list:
        raise ValueError("No valid .h5 files found in data file list")
    
    stat_dict = {}
    
    if os.path.exists(stat_file_path) and not overwrite:
        # Load existing statistics
        try:
            stat_dict = pickle.load(open(stat_file_path, 'rb'))
        except (pickle.PickleError, KeyError) as e:
            print(f"Warning: Failed to load statistics from {stat_file_path}: {e}")
            print("Recomputing statistics...")
            overwrite = True
    
    if not os.path.exists(stat_file_path) or overwrite:
        # Compute statistics from scratch
        all_data = []
        for data_file_path in data_file_list:
            with h5py.File(data_file_path, 'r') as f:
                data = f['inputs'][:]
                all_data.append(data)
        
        # Compute final statistics
        all_data = np.concatenate(all_data, axis=1)  # Shape: (num_samples, num_variables)
        
        mean = np.mean(all_data, axis=1)
        std = np.std(all_data, axis=1)
        mean = np.expand_dims(mean, axis=0)
        std = np.expand_dims(std, axis=0)
        stat_dict = {
            'mean': mean,
            'std': std
        }
        
        # Save statistics
        try:
            os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
            pickle.dump(stat_dict, open(stat_file_path, 'wb'))
            print(f"Statistics saved to {stat_file_path}")
        except (OSError, pickle.PickleError) as e:
            print(f"Warning: Failed to save statistics to {stat_file_path}: {e}")
    
    return stat_dict


class CustomDataset(Dataset):
    def __init__(self, options, logger=None):
        self.data_root = options.environment.data_root
        self.dataset_name = options.data.dataset_name
        self.dataset_path = f"{self.data_root}/{self.dataset_name}"
        self.variables = self.input_variables = options.data.input_variables

        self.train_list_path = f"{self.dataset_path}/train_list.csv"
        self.validation_list_path = f"{self.dataset_path}/validation_list.csv"
        self.stat_file_path = f"{self.dataset_path}/statistics.pkl"

        train_df = pd.read_csv(self.train_list_path)
        validation_df = pd.read_csv(self.validation_list_path)

        train_file_names = train_df['filename'].tolist()
        train_file_class = train_df['class'].tolist()
        self.train_file_list = list(zip(train_file_names, train_file_class))

        self.validation_file_names = validation_df['filename'].tolist()
        self.validation_file_class = validation_df['class'].tolist()
        self.validation_file_list = list(zip(self.validation_file_names, self.validation_file_class))

        print(f"Training samples: {len(self.train_file_list)}, Validation samples: {len(self.validation_file_list)}")

        if options.experiment.phase == 'train':
            self.list_data = self.train_file_list
        elif options.experiment.phase == 'validation':
            self.list_data = self.validation_file_list
        else:
            raise ValueError(f"Unknown phase: {options.experiment.phase}. Must be 'train' or 'validation'.")
        
        self.nb_data = len(self.list_data)

        try:
            self.stat_dict = get_statistics(
                self.stat_file_path, self.dataset_path, 
                self.train_file_list
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load/compute statistics: {e}")
        print(f"Loaded statistics for {len(self.stat_dict)} variables.")

        self.memory_cache = {}
        self.cache_enabled = True  # Can be disabled for low-memory scenarios

    def __len__(self):
        return self.nb_data
    
    def __getitem__(self, idx):
        file_name, file_class = self.list_data[idx]
        if self.cache_enabled and file_name in self.memory_cache:
            return self.memory_cache[file_name]
        
        file_path = f"{self.dataset_path}/{file_class}/{file_name}"

        sdo, inputs, targets, labels = read_h5(file_path)
        inputs = np.transpose(inputs, (1, 0))  # Shape: (sequence_length, num_variables)
        targets = np.transpose(targets, (1, 0))  # Shape: (num_groups, num_vectors)
        labels = np.expand_dims(labels, axis=-1)  # Shape: (num_groups, 1)

        inputs = (inputs - self.stat_dict['mean']) / self.stat_dict['std']
        
        sdo_tensor = torch.tensor(sdo, dtype=torch.float32)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        if self.cache_enabled:
            self.memory_cache[file_name] = {
                "sdo": sdo_tensor,
                "inputs": inputs_tensor,
                "targets": targets_tensor,
                "labels": labels_tensor,
                "file_names": os.path.basename(file_path)
            }
        
        return {
            "sdo": sdo_tensor,
            "inputs": inputs_tensor,
            "targets": targets_tensor,
            "labels": labels_tensor,
            "file_names": os.path.basename(file_path)
        }


def create_dataloader(config, logger=None):
    dataset = CustomDataset(config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.experiment.batch_size,
        shuffle=(config.experiment.phase == 'train'),
        num_workers=config.experiment.num_workers,
        pin_memory=(config.environment.device == 'cuda'),
        drop_last=False  # Keep all samples
    )
    return dataloader


@hydra.main(config_path="./configs", version_base=None)
def main(config):

    dataloader = create_dataloader(config)

    for batch in dataloader:
        print(batch['sdo'].shape, batch['inputs'].shape, batch['targets'].shape, batch['labels'].shape, batch['file_names'])
        tmp = batch['inputs'].numpy()
        print(tmp.mean(), tmp.std())
        break


if __name__ == "__main__" :
    main()



# def create_dataloader(options, logger=None):
#     """Create data loader for training or validation.
    
#     Args:
#         options: Configuration object containing dataloader parameters.
#         logger: Optional logger for output.
        
#     Returns:
#         DataLoader instance.
        
#     Raises:
#         RuntimeError: If dataset creation fails.
#     """
#     try:
#         dataset = CustomDataset(options, logger=logger)
#         dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=options.batch_size,
#             shuffle=(options.phase == 'train'),
#             num_workers=options.num_workers,
#             pin_memory=(options.device == 'cuda'),
#             drop_last=False  # Keep all samples
#         )
        
#         message = f"Dataloader created with {len(dataloader)} batches."
#         if logger:
#             logger.info(message)
#         else:
#             print(message)
            
#         return dataloader
        
#     except Exception as e:
#         error_msg = f"Failed to create dataloader: {e}"
#         if logger:
#             logger.error(error_msg)
#         else:
#             print(f"Error: {error_msg}")
#         raise RuntimeError(error_msg)

# class CustomDataset(Dataset):
#     """Custom dataset for multi-modal solar wind and image data.
    
#     Loads and preprocesses solar wind time series data and SDO image sequences
#     with strict NaN exclusion and efficient memory management.
    
#     Args:
#         options: Configuration object containing dataset parameters.
#         logger: Optional logger for output.
#     """
    
#     def __init__(self, options, logger=None):
#         self.logger = logger
#         self.data_root = options.data_root
#         self.train_list_path = options.train_list_path
#         self.validation_list_path = options.validation_list_path
        
#         # Load data lists
#         try:
#             self.train_df = pd.read_csv(self.train_list_path)
#             self.validation_df = pd.read_csv(self.validation_list_path)
#         except (FileNotFoundError, pd.errors.EmptyDataError) as e:
#             raise FileNotFoundError(f"Failed to load data lists: {e}")

#         self.train_file_list = self.train_df['file_name'].tolist()
#         self.validation_file_list = self.validation_df['file_name'].tolist()

#         # Select appropriate file list based on phase
#         if options.phase == 'train':
#             self.list_data = self.train_file_list
#         elif options.phase == 'validation':
#             self.list_data = self.validation_file_list
#         else:
#             raise ValueError(f"Unknown phase: {options.phase}. Must be 'train' or 'validation'.")
            
#         self.nb_data = len(self.list_data)
#         self._log_info(f"Using {self.nb_data} samples for {options.phase} phase.")

#         # Store configuration parameters
#         self.input_variables = options.input_variables
#         self.input_sequence_length = options.input_sequence_length
#         self.num_input_variables = options.num_input_variables

#         self.target_variables = options.target_variables
#         self.target_sequence_length = options.target_sequence_length
#         self.num_target_variables = options.num_target_variables

#         # Target transformation parameters
#         self.group_size = getattr(options, 'group_size', 24)
#         self.threshold = getattr(options, 'threshold', 48)

#         # ìˆ˜ì •ëœ ë¶€ë¶„: ì •ë¦¬ëœ config ì‚¬ìš©
#         self.convlstm_input_channels = options.convlstm_input_channels
#         self.convlstm_input_image_frames = options.convlstm_input_image_frames
#         self.image_size = options.image_size

#         # Define expected shapes for validation
#         # Only input variables need statistics (targets are not normalized)
#         self.variables = self.input_variables  # Changed from list(set(...))
#         self.sdo_shape = (
#             self.convlstm_input_channels,
#             self.convlstm_input_image_frames,
#             self.image_size,
#             self.image_size
#         )
#         self.inputs_shape = (self.input_sequence_length, self.num_input_variables)
        
#         # Update target shape based on transformation
#         num_groups = self.target_sequence_length // self.group_size
#         self.targets_shape = (num_groups, self.num_target_variables)

#         # Load/compute statistics (only for input variables)
#         self.stat_file_path = options.stat_file_path
#         try:
#             self.stat_dict = get_statistics(
#                 self.stat_file_path, self.data_root, 
#                 self.train_file_list, self.variables
#             )
#         except Exception as e:
#             raise RuntimeError(f"Failed to load/compute statistics: {e}")

#         # Initialize memory management
#         self.memory_cache = {}
#         self.cache_enabled = True  # Can be disabled for low-memory scenarios
        
#         # Validate and cache data
#         self.validate_data()

#     def _log_info(self, message: str):
#         """Log information message."""
#         if self.logger:
#             self.logger.info(message)
#         else:
#             print(message)

#     def _log_warning(self, message: str):
#         """Log warning message."""
#         pass
#         # if self.logger:
#         #     self.logger.warning(message)
#         # else:
#         #     print(f"Warning: {message}")

#     def validate_data(self):
#         """Validate and cache data items with strict NaN exclusion.
        
#         Checks each data file for validity and caches successfully processed items.
#         Any files containing NaN values are logged and excluded from the dataset.
#         """
#         valid_files = []
#         invalid_files = []
#         processing_errors = []
#         nan_excluded_files = []
        
#         for file_name in self.list_data:
#             try:
#                 file_path = f"{self.data_root}/{file_name}"
                
#                 # Read data
#                 sdo_193, sdo_211, omni_inputs, omni_targets = read_h5(
#                     file_path, self.input_variables, self.target_variables
#                 )

#                 # Process and validate data
#                 processed_data = self._process_data_item(
#                     sdo_193, sdo_211, omni_inputs, omni_targets, file_name
#                 )
                
#                 if processed_data is not None:
#                     valid_files.append(file_name)
#                     if self.cache_enabled:
#                         self.cache_item(file_name, processed_data)
#                 else:
#                     nan_excluded_files.append(file_name)
#                     invalid_files.append(file_name)
                    
#             except Exception as e:
#                 invalid_files.append(file_name)
#                 processing_errors.append(f"{file_name}: {str(e)}")

#         # Update file list to only include valid files
#         self.list_data = valid_files
#         self.nb_data = len(self.list_data)
#         self.valid_files = valid_files
#         self.invalid_files = invalid_files

#         # Log validation results
#         total_files = len(valid_files) + len(invalid_files)
#         self._log_info(
#             f"Validation complete: {len(valid_files)}/{total_files} files valid, "
#             f"{len(invalid_files)} files invalid."
#         )
#         self._log_info(f"NaN exclusions: {len(nan_excluded_files)} files contained NaN values")
        
#         if processing_errors and self.logger:
#             self.logger.debug("Processing errors:")
#             for error in processing_errors[:10]:  # Log first 10 errors
#                 self.logger.debug(f"  {error}")
        
#         if len(valid_files) == 0:
#             raise RuntimeError("No valid data files found after validation")

#     def _process_data_item(self, sdo_193: np.ndarray, sdo_211: np.ndarray, 
#                           omni_inputs: Dict[str, np.ndarray], 
#                           omni_targets: Dict[str, np.ndarray], 
#                           file_name: str) -> Optional[Dict[str, torch.Tensor]]:
#         """Process a single data item with strict NaN exclusion.
        
#         Args:
#             sdo_193: SDO 193 channel data.
#             sdo_211: SDO 211 channel data.
#             omni_inputs: Dictionary of input time series data.
#             omni_targets: Dictionary of target time series data.
#             file_name: Name of the source file.
            
#         Returns:
#             Processed data dictionary or None if any NaN values are found.
#         """
#         try:
#             # Process SDO data
#             sdo = np.concatenate([sdo_193, sdo_211], axis=1)  # Shape: (frames, 2, H, W)
#             sdo = np.transpose(sdo, (1, 0, 2, 3))  # Shape: (2, frames, H, W)

#             # Check for NaN in image data first
#             if np.isnan(sdo).any() or not np.isfinite(sdo).all():
#                 self._log_warning(f"NaN or non-finite values found in image data for {file_name}")
#                 return None
            
#             # Process time series data with strict NaN checking
#             inputs = self._process_time_series(
#                 omni_inputs, self.input_variables, 
#                 self.input_sequence_length, is_input=True
#             )
            
#             if inputs is None:  # NaN found in inputs
#                 return None

#             # Process targets WITHOUT normalization
#             targets = self._process_time_series(
#                 omni_targets, self.target_variables,
#                 self.target_sequence_length, is_input=False,
#                 offset=self.input_sequence_length,
#                 normalize=False  # Do not normalize targets
#             )
            
#             if targets is None:  # NaN found in targets
#                 return None

#             # Normalize image data from [0, 255] to [-1, 1]
#             sdo = self._normalize_image_data(sdo)

#             # Convert to tensors
#             sdo_tensor = torch.tensor(sdo, dtype=torch.float32)
#             inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
#             targets_tensor = torch.tensor(targets, dtype=torch.float32)
            
#             # Apply target transformation
#             targets_transformed = make_target(targets_tensor, self.group_size, self.threshold)

#             # Validate shapes and content
#             if not self._validate_processed_data(sdo, inputs, targets_transformed.numpy()):
#                 return None

#             return {
#                 "sdo": sdo_tensor,
#                 "inputs": inputs_tensor,
#                 "targets": targets_transformed,
#                 "file_names": file_name
#             }
            
#         except Exception as e:
#             self._log_warning(f"Failed to process {file_name}: {e}")
#             return None

#     def _process_time_series(self, data_dict: Dict[str, np.ndarray], 
#                            variables: List[str], sequence_length: int, 
#                            is_input: bool, offset: int = 0, normalize: bool = True) -> Optional[np.ndarray]:
#         """Process time series data with strict NaN exclusion.
        
#         Args:
#             data_dict: Dictionary containing time series data.
#             variables: List of variable names to process.
#             sequence_length: Length of sequences to extract.
#             is_input: Whether this is input data (affects indexing).
#             offset: Offset for data extraction.
#             normalize: Whether to normalize the data (default: True).
            
#         Returns:
#             Processed time series array or None if NaN values are found.
            
#         Raises:
#             ValueError: If data contains NaN values or invalid shapes.
#         """
#         processed_vars = []
        
#         for variable in variables:
#             if variable not in data_dict:
#                 raise KeyError(f"Variable {variable} not found in data")
            
#             raw_data = data_dict[variable]
            
#             # Extract relevant sequence
#             if is_input:
#                 var_data = raw_data[:sequence_length]
#             else:
#                 end_idx = offset + sequence_length
#                 var_data = raw_data[offset:end_idx]
            
#             # Check for sufficient data length
#             if len(var_data) < sequence_length:
#                 raise ValueError(
#                     f"Insufficient data for {variable}: got {len(var_data)}, "
#                     f"need {sequence_length}"
#                 )
            
#             # Strict NaN check - exclude any data with NaN values
#             if np.isnan(var_data).any():
#                 self._log_warning(f"NaN values found in {variable} - excluding this sample")
#                 return None
            
#             # Check for infinite values
#             if not np.isfinite(var_data).all():
#                 self._log_warning(f"Non-finite values found in {variable} - excluding this sample")
#                 return None
            
#             # Normalize using statistics (only if normalize=True)
#             if normalize:
#                 if variable in self.stat_dict:
#                     mean = self.stat_dict[variable]['mean']
#                     std = self.stat_dict[variable]['std']
#                     if std > 0:
#                         var_data = (var_data - mean) / std
#                     else:
#                         self._log_warning(f"Zero std for {variable}, using raw values")
#                 else:
#                     self._log_warning(f"No statistics found for {variable}")
            
#             processed_vars.append(var_data)
        
#         return np.stack(processed_vars, axis=-1)

#     def _normalize_image_data(self, sdo: np.ndarray) -> np.ndarray:
#         """Normalize image data from [0, 255] to [-1, 1].
        
#         Args:
#             sdo: Image data array.
            
#         Returns:
#             Normalized image data.
#         """
#         # Clip extreme values to prevent normalization issues
#         sdo = np.clip(sdo, 0, 255)
#         return (sdo / 255.0) * 2.0 - 1.0

#     def _validate_processed_data(self, sdo: np.ndarray, inputs: np.ndarray, 
#                                targets: np.ndarray) -> bool:
#         """Validate processed data shapes and content with strict NaN checking.
        
#         Args:
#             sdo: Processed SDO image data.
#             inputs: Processed input time series data.
#             targets: Processed target time series data.
            
#         Returns:
#             True if data is valid, False otherwise.
#         """
#         # Check shapes
#         if sdo.shape != self.sdo_shape:
#             self._log_warning(f"Invalid SDO shape: {sdo.shape}, expected {self.sdo_shape}")
#             return False
        
#         if inputs.shape != self.inputs_shape:
#             self._log_warning(f"Invalid inputs shape: {inputs.shape}, expected {self.inputs_shape}")
#             return False
        
#         if targets.shape != self.targets_shape:
#             self._log_warning(f"Invalid targets shape: {targets.shape}, expected {self.targets_shape}")
#             return False
        
#         # Strict check for any NaN or infinite values - reject immediately
#         arrays_to_check = [("SDO", sdo), ("inputs", inputs), ("targets", targets)]
#         for name, array in arrays_to_check:
#             if np.isnan(array).any():
#                 self._log_warning(f"NaN values found in {name} - sample rejected")
#                 return False
#             if not np.isfinite(array).all():
#                 self._log_warning(f"Non-finite values found in {name} - sample rejected")
#                 return False
        
#         return True

#     def cache_item(self, file_name: str, data_dict: Dict[str, torch.Tensor]):
#         """Cache processed data item.
        
#         Args:
#             file_name: Name of the file.
#             data_dict: Processed data dictionary to cache.
#         """
#         if self.cache_enabled:
#             self.memory_cache[file_name] = data_dict

#     def disable_cache(self):
#         """Disable memory caching to reduce memory usage."""
#         self.cache_enabled = False
#         self.memory_cache.clear()

#     def __len__(self):
#         return self.nb_data

#     def __getitem__(self, idx):
#         """Get a data item by index.
        
#         Args:
#             idx: Index of the item to retrieve.
            
#         Returns:
#             Dictionary containing the data item.
            
#         Raises:
#             RuntimeError: If data loading fails and fallback is not possible.
#         """
#         file_name = self.list_data[idx]
        
#         # Try to get from cache first
#         if self.cache_enabled and file_name in self.memory_cache:
#             return self.memory_cache[file_name]
        
#         # Load and process data on-demand
#         try:
#             file_path = f"{self.data_root}/{file_name}"
#             sdo_193, sdo_211, omni_inputs, omni_targets = read_h5(
#                 file_path, self.input_variables, self.target_variables
#             )
            
#             processed_data = self._process_data_item(
#                 sdo_193, sdo_211, omni_inputs, omni_targets, file_name
#             )
            
#             if processed_data is not None:
#                 if self.cache_enabled:
#                     self.cache_item(file_name, processed_data)
#                 return processed_data
#             else:
#                 raise RuntimeError(f"Failed to process data for {file_name}")
                
#         except Exception as e:
#             self._log_warning(f"Failed to load data for {file_name}: {e}")
#             # Return dummy data as fallback
#             return self._create_dummy_data()

#     def _create_dummy_data(self) -> Dict[str, torch.Tensor]:
#         """Create dummy data as fallback when loading fails.
        
#         Returns:
#             Dictionary containing dummy tensors with appropriate shapes.
#         """
#         dummy_sdo = torch.zeros(self.sdo_shape, dtype=torch.float32)
#         dummy_inputs = torch.zeros(self.inputs_shape, dtype=torch.float32)
#         dummy_targets = torch.zeros(self.targets_shape, dtype=torch.float32)
        
#         return {
#             "sdo": dummy_sdo,
#             "inputs": dummy_inputs,
#             "targets": dummy_targets,
#             "file_names": "dummy_data"
#         }




# def get_statistics(stat_file_path: str, data_root: str, data_file_list: List[str], 
#                   variables: List[str], overwrite: bool = False) -> Dict[str, Dict[str, float]]:
#     """Compute and cache statistics for data normalization.
    
#     Args:
#         stat_file_path: Path to save/load statistics pickle file.
#         data_root: Root directory containing data files.
#         data_file_list: List of data file names.
#         variables: List of variable names to compute statistics for.
#         overwrite: Whether to recompute statistics even if cache exists.
        
#     Returns:
#         Dictionary containing mean and std for each variable.
        
#     Raises:
#         FileNotFoundError: If data files cannot be found.
#         ValueError: If no valid data is found for statistics computation.
#     """
#     # Filter for h5 files only
#     data_file_list = [f"{data_root}/{f}" for f in data_file_list if f.endswith('.h5')]
    
#     if not data_file_list:
#         raise ValueError("No valid .h5 files found in data file list")
    
#     stat_dict = {}
    
#     if os.path.exists(stat_file_path) and not overwrite:
#         # Load existing statistics
#         try:
#             loaded = pickle.load(open(stat_file_path, 'rb'))
#             for variable in variables:
#                 stat_dict[variable] = loaded.get(variable, {})
#         except (pickle.PickleError, KeyError) as e:
#             print(f"Warning: Failed to load statistics from {stat_file_path}: {e}")
#             print("Recomputing statistics...")
#             overwrite = True
    
#     if not os.path.exists(stat_file_path) or overwrite:
#         # Compute statistics from scratch
#         total_dict = {variable: [] for variable in variables}
        
#         valid_files = 0
#         for data_file_path in data_file_list:
#             if not os.path.exists(data_file_path):
#                 print(f"Warning: File not found: {data_file_path}")
#                 continue
                
#             try:
#                 with h5py.File(data_file_path, 'r') as f:
#                     for variable in variables:
#                         dataset_name = f"omni_{variable}"
#                         if dataset_name in f:
#                             data = f[dataset_name][:]
#                             # Filter out NaN and infinite values
#                             valid_data = data[np.isfinite(data)]
#                             if len(valid_data) > 0:
#                                 total_dict[variable].append(valid_data)
#                         else:
#                             print(f"Warning: Variable {variable} not found in {data_file_path}")
#                 valid_files += 1
#             except (OSError, KeyError) as e:
#                 print(f"Warning: Failed to read {data_file_path}: {e}")
#                 continue
        
#         if valid_files == 0:
#             raise ValueError("No valid data files found for statistics computation")
        
#         # Compute final statistics
#         for variable in variables:
#             if total_dict[variable]:
#                 concatenated_data = np.concatenate(total_dict[variable], axis=0)
#                 if len(concatenated_data) > 0:
#                     stat_dict[variable] = {
#                         'mean': float(np.mean(concatenated_data)),
#                         'std': float(np.std(concatenated_data))
#                     }
#                 else:
#                     print(f"Warning: No valid data found for variable {variable}")
#                     stat_dict[variable] = {'mean': 0.0, 'std': 1.0}
#             else:
#                 print(f"Warning: No data found for variable {variable}")
#                 stat_dict[variable] = {'mean': 0.0, 'std': 1.0}
        
#         # Save statistics
#         try:
#             os.makedirs(os.path.dirname(stat_file_path), exist_ok=True)
#             pickle.dump(stat_dict, open(stat_file_path, 'wb'))
#             print(f"Statistics saved to {stat_file_path}")
#         except (OSError, pickle.PickleError) as e:
#             print(f"Warning: Failed to save statistics to {stat_file_path}: {e}")
    
#     return stat_dict



# class CustomDataset(Dataset):
#     """Custom dataset for multi-modal solar wind and image data.
    
#     Loads and preprocesses solar wind time series data and SDO image sequences
#     with strict NaN exclusion and efficient memory management.
    
#     Args:
#         options: Configuration object containing dataset parameters.
#         logger: Optional logger for output.
#     """
    
#     def __init__(self, options, logger=None):
#         self.logger = logger
#         self.data_root = options.data_root
#         self.train_list_path = options.train_list_path
#         self.validation_list_path = options.validation_list_path
        
#         # Load data lists
#         try:
#             self.train_df = pd.read_csv(self.train_list_path)
#             self.validation_df = pd.read_csv(self.validation_list_path)
#         except (FileNotFoundError, pd.errors.EmptyDataError) as e:
#             raise FileNotFoundError(f"Failed to load data lists: {e}")

#         self.train_file_list = self.train_df['file_name'].tolist()
#         self.validation_file_list = self.validation_df['file_name'].tolist()

#         # Select appropriate file list based on phase
#         if options.phase == 'train':
#             self.list_data = self.train_file_list
#         elif options.phase == 'validation':
#             self.list_data = self.validation_file_list
#         else:
#             raise ValueError(f"Unknown phase: {options.phase}. Must be 'train' or 'validation'.")
            
#         self.nb_data = len(self.list_data)
#         self._log_info(f"Using {self.nb_data} samples for {options.phase} phase.")

#         # Store configuration parameters
#         self.input_variables = options.input_variables
#         self.input_sequence_length = options.input_sequence_length
#         self.num_input_variables = options.num_input_variables

#         self.target_variables = options.target_variables
#         self.target_sequence_length = options.target_sequence_length
#         self.num_target_variables = options.num_target_variables

#         # Target transformation parameters
#         self.group_size = getattr(options, 'group_size', 24)
#         self.threshold = getattr(options, 'threshold', 48)

#         # ìˆ˜ì •ëœ ë¶€ë¶„: ì •ë¦¬ëœ config ì‚¬ìš©
#         self.convlstm_input_channels = options.convlstm_input_channels
#         self.convlstm_input_image_frames = options.convlstm_input_image_frames
#         self.image_size = options.image_size

#         # Define expected shapes for validation
#         # Only input variables need statistics (targets are not normalized)
#         self.variables = self.input_variables  # Changed from list(set(...))
#         self.sdo_shape = (
#             self.convlstm_input_channels,
#             self.convlstm_input_image_frames,
#             self.image_size,
#             self.image_size
#         )
#         self.inputs_shape = (self.input_sequence_length, self.num_input_variables)
        
#         # Update target shape based on transformation
#         num_groups = self.target_sequence_length // self.group_size
#         self.targets_shape = (num_groups, self.num_target_variables)

#         # Load/compute statistics (only for input variables)
#         self.stat_file_path = options.stat_file_path
#         try:
#             self.stat_dict = get_statistics(
#                 self.stat_file_path, self.data_root, 
#                 self.train_file_list, self.variables
#             )
#         except Exception as e:
#             raise RuntimeError(f"Failed to load/compute statistics: {e}")

#         # Initialize memory management
#         self.memory_cache = {}
#         self.cache_enabled = True  # Can be disabled for low-memory scenarios
        
#         # Validate and cache data
#         self.validate_data()

#     def _log_info(self, message: str):
#         """Log information message."""
#         if self.logger:
#             self.logger.info(message)
#         else:
#             print(message)

#     def _log_warning(self, message: str):
#         """Log warning message."""
#         pass
#         # if self.logger:
#         #     self.logger.warning(message)
#         # else:
#         #     print(f"Warning: {message}")

#     def validate_data(self):
#         """Validate and cache data items with strict NaN exclusion.
        
#         Checks each data file for validity and caches successfully processed items.
#         Any files containing NaN values are logged and excluded from the dataset.
#         """
#         valid_files = []
#         invalid_files = []
#         processing_errors = []
#         nan_excluded_files = []
        
#         for file_name in self.list_data:
#             try:
#                 file_path = f"{self.data_root}/{file_name}"
                
#                 # Read data
#                 sdo_193, sdo_211, omni_inputs, omni_targets = read_h5(
#                     file_path, self.input_variables, self.target_variables
#                 )

#                 # Process and validate data
#                 processed_data = self._process_data_item(
#                     sdo_193, sdo_211, omni_inputs, omni_targets, file_name
#                 )
                
#                 if processed_data is not None:
#                     valid_files.append(file_name)
#                     if self.cache_enabled:
#                         self.cache_item(file_name, processed_data)
#                 else:
#                     nan_excluded_files.append(file_name)
#                     invalid_files.append(file_name)
                    
#             except Exception as e:
#                 invalid_files.append(file_name)
#                 processing_errors.append(f"{file_name}: {str(e)}")

#         # Update file list to only include valid files
#         self.list_data = valid_files
#         self.nb_data = len(self.list_data)
#         self.valid_files = valid_files
#         self.invalid_files = invalid_files

#         # Log validation results
#         total_files = len(valid_files) + len(invalid_files)
#         self._log_info(
#             f"Validation complete: {len(valid_files)}/{total_files} files valid, "
#             f"{len(invalid_files)} files invalid."
#         )
#         self._log_info(f"NaN exclusions: {len(nan_excluded_files)} files contained NaN values")
        
#         if processing_errors and self.logger:
#             self.logger.debug("Processing errors:")
#             for error in processing_errors[:10]:  # Log first 10 errors
#                 self.logger.debug(f"  {error}")
        
#         if len(valid_files) == 0:
#             raise RuntimeError("No valid data files found after validation")

#     def _process_data_item(self, sdo_193: np.ndarray, sdo_211: np.ndarray, 
#                           omni_inputs: Dict[str, np.ndarray], 
#                           omni_targets: Dict[str, np.ndarray], 
#                           file_name: str) -> Optional[Dict[str, torch.Tensor]]:
#         """Process a single data item with strict NaN exclusion.
        
#         Args:
#             sdo_193: SDO 193 channel data.
#             sdo_211: SDO 211 channel data.
#             omni_inputs: Dictionary of input time series data.
#             omni_targets: Dictionary of target time series data.
#             file_name: Name of the source file.
            
#         Returns:
#             Processed data dictionary or None if any NaN values are found.
#         """
#         try:
#             # Process SDO data
#             sdo = np.concatenate([sdo_193, sdo_211], axis=1)  # Shape: (frames, 2, H, W)
#             sdo = np.transpose(sdo, (1, 0, 2, 3))  # Shape: (2, frames, H, W)

#             # Check for NaN in image data first
#             if np.isnan(sdo).any() or not np.isfinite(sdo).all():
#                 self._log_warning(f"NaN or non-finite values found in image data for {file_name}")
#                 return None
            
#             # Process time series data with strict NaN checking
#             inputs = self._process_time_series(
#                 omni_inputs, self.input_variables, 
#                 self.input_sequence_length, is_input=True
#             )
            
#             if inputs is None:  # NaN found in inputs
#                 return None

#             # Process targets WITHOUT normalization
#             targets = self._process_time_series(
#                 omni_targets, self.target_variables,
#                 self.target_sequence_length, is_input=False,
#                 offset=self.input_sequence_length,
#                 normalize=False  # Do not normalize targets
#             )
            
#             if targets is None:  # NaN found in targets
#                 return None

#             # Normalize image data from [0, 255] to [-1, 1]
#             sdo = self._normalize_image_data(sdo)

#             # Convert to tensors
#             sdo_tensor = torch.tensor(sdo, dtype=torch.float32)
#             inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
#             targets_tensor = torch.tensor(targets, dtype=torch.float32)
            
#             # Apply target transformation
#             targets_transformed = make_target(targets_tensor, self.group_size, self.threshold)

#             # Validate shapes and content
#             if not self._validate_processed_data(sdo, inputs, targets_transformed.numpy()):
#                 return None

#             return {
#                 "sdo": sdo_tensor,
#                 "inputs": inputs_tensor,
#                 "targets": targets_transformed,
#                 "file_names": file_name
#             }
            
#         except Exception as e:
#             self._log_warning(f"Failed to process {file_name}: {e}")
#             return None

#     def _process_time_series(self, data_dict: Dict[str, np.ndarray], 
#                            variables: List[str], sequence_length: int, 
#                            is_input: bool, offset: int = 0, normalize: bool = True) -> Optional[np.ndarray]:
#         """Process time series data with strict NaN exclusion.
        
#         Args:
#             data_dict: Dictionary containing time series data.
#             variables: List of variable names to process.
#             sequence_length: Length of sequences to extract.
#             is_input: Whether this is input data (affects indexing).
#             offset: Offset for data extraction.
#             normalize: Whether to normalize the data (default: True).
            
#         Returns:
#             Processed time series array or None if NaN values are found.
            
#         Raises:
#             ValueError: If data contains NaN values or invalid shapes.
#         """
#         processed_vars = []
        
#         for variable in variables:
#             if variable not in data_dict:
#                 raise KeyError(f"Variable {variable} not found in data")
            
#             raw_data = data_dict[variable]
            
#             # Extract relevant sequence
#             if is_input:
#                 var_data = raw_data[:sequence_length]
#             else:
#                 end_idx = offset + sequence_length
#                 var_data = raw_data[offset:end_idx]
            
#             # Check for sufficient data length
#             if len(var_data) < sequence_length:
#                 raise ValueError(
#                     f"Insufficient data for {variable}: got {len(var_data)}, "
#                     f"need {sequence_length}"
#                 )
            
#             # Strict NaN check - exclude any data with NaN values
#             if np.isnan(var_data).any():
#                 self._log_warning(f"NaN values found in {variable} - excluding this sample")
#                 return None
            
#             # Check for infinite values
#             if not np.isfinite(var_data).all():
#                 self._log_warning(f"Non-finite values found in {variable} - excluding this sample")
#                 return None
            
#             # Normalize using statistics (only if normalize=True)
#             if normalize:
#                 if variable in self.stat_dict:
#                     mean = self.stat_dict[variable]['mean']
#                     std = self.stat_dict[variable]['std']
#                     if std > 0:
#                         var_data = (var_data - mean) / std
#                     else:
#                         self._log_warning(f"Zero std for {variable}, using raw values")
#                 else:
#                     self._log_warning(f"No statistics found for {variable}")
            
#             processed_vars.append(var_data)
        
#         return np.stack(processed_vars, axis=-1)

#     def _normalize_image_data(self, sdo: np.ndarray) -> np.ndarray:
#         """Normalize image data from [0, 255] to [-1, 1].
        
#         Args:
#             sdo: Image data array.
            
#         Returns:
#             Normalized image data.
#         """
#         # Clip extreme values to prevent normalization issues
#         sdo = np.clip(sdo, 0, 255)
#         return (sdo / 255.0) * 2.0 - 1.0

#     def _validate_processed_data(self, sdo: np.ndarray, inputs: np.ndarray, 
#                                targets: np.ndarray) -> bool:
#         """Validate processed data shapes and content with strict NaN checking.
        
#         Args:
#             sdo: Processed SDO image data.
#             inputs: Processed input time series data.
#             targets: Processed target time series data.
            
#         Returns:
#             True if data is valid, False otherwise.
#         """
#         # Check shapes
#         if sdo.shape != self.sdo_shape:
#             self._log_warning(f"Invalid SDO shape: {sdo.shape}, expected {self.sdo_shape}")
#             return False
        
#         if inputs.shape != self.inputs_shape:
#             self._log_warning(f"Invalid inputs shape: {inputs.shape}, expected {self.inputs_shape}")
#             return False
        
#         if targets.shape != self.targets_shape:
#             self._log_warning(f"Invalid targets shape: {targets.shape}, expected {self.targets_shape}")
#             return False
        
#         # Strict check for any NaN or infinite values - reject immediately
#         arrays_to_check = [("SDO", sdo), ("inputs", inputs), ("targets", targets)]
#         for name, array in arrays_to_check:
#             if np.isnan(array).any():
#                 self._log_warning(f"NaN values found in {name} - sample rejected")
#                 return False
#             if not np.isfinite(array).all():
#                 self._log_warning(f"Non-finite values found in {name} - sample rejected")
#                 return False
        
#         return True

#     def cache_item(self, file_name: str, data_dict: Dict[str, torch.Tensor]):
#         """Cache processed data item.
        
#         Args:
#             file_name: Name of the file.
#             data_dict: Processed data dictionary to cache.
#         """
#         if self.cache_enabled:
#             self.memory_cache[file_name] = data_dict

#     def disable_cache(self):
#         """Disable memory caching to reduce memory usage."""
#         self.cache_enabled = False
#         self.memory_cache.clear()

#     def __len__(self):
#         return self.nb_data

#     def __getitem__(self, idx):
#         """Get a data item by index.
        
#         Args:
#             idx: Index of the item to retrieve.
            
#         Returns:
#             Dictionary containing the data item.
            
#         Raises:
#             RuntimeError: If data loading fails and fallback is not possible.
#         """
#         file_name = self.list_data[idx]
        
#         # Try to get from cache first
#         if self.cache_enabled and file_name in self.memory_cache:
#             return self.memory_cache[file_name]
        
#         # Load and process data on-demand
#         try:
#             file_path = f"{self.data_root}/{file_name}"
#             sdo_193, sdo_211, omni_inputs, omni_targets = read_h5(
#                 file_path, self.input_variables, self.target_variables
#             )
            
#             processed_data = self._process_data_item(
#                 sdo_193, sdo_211, omni_inputs, omni_targets, file_name
#             )
            
#             if processed_data is not None:
#                 if self.cache_enabled:
#                     self.cache_item(file_name, processed_data)
#                 return processed_data
#             else:
#                 raise RuntimeError(f"Failed to process data for {file_name}")
                
#         except Exception as e:
#             self._log_warning(f"Failed to load data for {file_name}: {e}")
#             # Return dummy data as fallback
#             return self._create_dummy_data()

#     def _create_dummy_data(self) -> Dict[str, torch.Tensor]:
#         """Create dummy data as fallback when loading fails.
        
#         Returns:
#             Dictionary containing dummy tensors with appropriate shapes.
#         """
#         dummy_sdo = torch.zeros(self.sdo_shape, dtype=torch.float32)
#         dummy_inputs = torch.zeros(self.inputs_shape, dtype=torch.float32)
#         dummy_targets = torch.zeros(self.targets_shape, dtype=torch.float32)
        
#         return {
#             "sdo": dummy_sdo,
#             "inputs": dummy_inputs,
#             "targets": dummy_targets,
#             "file_names": "dummy_data"
#         }


# def create_dataloader(options, logger=None):
#     """Create data loader for training or validation.
    
#     Args:
#         options: Configuration object containing dataloader parameters.
#         logger: Optional logger for output.
        
#     Returns:
#         DataLoader instance.
        
#     Raises:
#         RuntimeError: If dataset creation fails.
#     """
#     try:
#         dataset = CustomDataset(options, logger=logger)
#         dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=options.batch_size,
#             shuffle=(options.phase == 'train'),
#             num_workers=options.num_workers,
#             pin_memory=(options.device == 'cuda'),
#             drop_last=False  # Keep all samples
#         )
        
#         message = f"Dataloader created with {len(dataloader)} batches."
#         if logger:
#             logger.info(message)
#         else:
#             print(message)
            
#         return dataloader
        
#     except Exception as e:
#         error_msg = f"Failed to create dataloader: {e}"
#         if logger:
#             logger.error(error_msg)
#         else:
#             print(f"Error: {error_msg}")
#         raise RuntimeError(error_msg)

# if __name__ == "__main__":
#     # Example usage

#     from config import Config

#     options = Config()
#     options = Config().from_args_and_yaml(yaml_path="configs/config_dev.yaml")

#     dataloader = create_dataloader(options, logger=None)

#     print(len(dataloader))

#     for i, data_dict in enumerate(dataloader):
#         sdo = data_dict['sdo']
#         inputs = data_dict['inputs']
#         targets = data_dict['targets']
#         file_names = data_dict['file_names']
#         print(sdo.shape, inputs.shape, targets.shape, file_names)
#         print(targets)
#         if i >= 2:
#             break