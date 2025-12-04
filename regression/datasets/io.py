"""
File I/O operations for multimodal dataset.

This module handles reading data from HDF5 files.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import h5py


class HDF5Reader:
    """Handle HDF5 file reading with proper error handling."""
    
    @staticmethod
    def read(
        file_path: str,
        sdo_wavelengths: List[str],
        omni_variables: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Read SDO and OMNI data from HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            sdo_wavelengths: List of SDO wavelengths to read
            omni_variables: List of OMNI variables to read
            
        Returns:
            (sdo_data, omni_data) dictionaries
            
        Raises:
            FileNotFoundError: If file doesn't exist
            KeyError: If required dataset not found
            OSError: If HDF5 file is corrupted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        sdo_data = {}
        omni_data = {}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Read SDO data
                for wavelength in sdo_wavelengths:
                    dataset_name = f"sdo_{wavelength}"
                    if dataset_name not in f:
                        raise KeyError(f"SDO wavelength {wavelength} not found in {file_path}")
                    sdo_data[wavelength] = f[dataset_name][:]
                
                # Read OMNI data
                for variable in omni_variables:
                    dataset_name = f"omni_{variable}"
                    if dataset_name not in f:
                        raise KeyError(f"OMNI variable {variable} not found in {file_path}")
                    omni_data[variable] = f[dataset_name][:]
        
        except KeyError:
            raise
        except Exception as e:
            raise OSError(f"Failed to read HDF5 file {file_path}: {e}")
        
        return sdo_data, omni_data
