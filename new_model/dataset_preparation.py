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


@hydra.main(version_base=None, config_path="./configs")
def main(config):

    data_path = f"{config.data.data_root}/original"

    files = glob(f"{data_path}/*.h5")
    num_files = len(files)
    print(f"Number of .h5 files in {data_path}: {num_files}")

if __name__ == "__main__" :
    main()


# def _validate_data(self):
#     """Validate and cache data items with strict NaN exclusion.
    
#     Checks each data file for validity and caches successfully processed items.
#     Any files containing NaN values are logged and excluded from the dataset.
#     """
#     valid_files = []
#     invalid_files = []
#     processing_errors = []
#     nan_excluded_files = []
    
#     for file_name in self.list_data:
#         try:
#             file_path = f"{self.data_root}/{file_name}"
            
#             # Read data
#             sdo_193, sdo_211, omni_inputs, omni_targets = read_h5(
#                 file_path, self.input_variables, self.target_variables
#             )

#             # Process and validate data
#             processed_data = self._process_data_item(
#                 sdo_193, sdo_211, omni_inputs, omni_targets, file_name
#             )
            
#             if processed_data is not None:
#                 valid_files.append(file_name)
#                 if self.cache_enabled:
#                     self.cache_item(file_name, processed_data)
#             else:
#                 nan_excluded_files.append(file_name)
#                 invalid_files.append(file_name)
                
#         except Exception as e:
#             invalid_files.append(file_name)
#             processing_errors.append(f"{file_name}: {str(e)}")

#     # Update file list to only include valid files
#     self.list_data = valid_files
#     self.nb_data = len(self.list_data)
#     self.valid_files = valid_files
#     self.invalid_files = invalid_files

#     # Log validation results
#     total_files = len(valid_files) + len(invalid_files)
#     self._log_info(
#         f"Validation complete: {len(valid_files)}/{total_files} files valid, "
#         f"{len(invalid_files)} files invalid."
#     )
#     self._log_info(f"NaN exclusions: {len(nan_excluded_files)} files contained NaN values")
    
#     if processing_errors and self.logger:
#         self.logger.debug("Processing errors:")
#         for error in processing_errors[:10]:  # Log first 10 errors
#             self.logger.debug(f"  {error}")
    
#     if len(valid_files) == 0:
#         raise RuntimeError("No valid data files found after validation")



# def read_h5(file_path: str, input_variables: List[str], target_variables: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Data file not found: {file_path}")
    
#     try:
#         with h5py.File(file_path, 'r') as f:
#             # Read image data
#             sdo_193 = f['sdo_193'][:]
#             sdo_211 = f['sdo_211'][:]
            
#             # Read input variables
#             omni_inputs = {}
#             for variable in input_variables:
#                 dataset_name = f"omni_{variable}"
#                 if dataset_name in f:
#                     omni_inputs[variable] = f[dataset_name][:]
#                 else:
#                     raise KeyError(f"Input variable {variable} not found in {file_path}")
            
#             # Read target variables
#             omni_targets = {}
#             for variable in target_variables:
#                 dataset_name = f"omni_{variable}"
#                 if dataset_name in f:
#                     omni_targets[variable] = f[dataset_name][:]
#                 else:
#                     raise KeyError(f"Target variable {variable} not found in {file_path}")
                    
#     except (OSError, h5py.error.HDF5Error) as e:
#         raise OSError(f"Failed to read HDF5 file {file_path}: {e}")
    
#     return sdo_193, sdo_211, omni_inputs, omni_targets