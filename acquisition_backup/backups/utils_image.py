from typing import Tuple, Union, Optional, Literal

import numpy as np
import pandas as pd
from scipy import ndimage


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    order: int = 1,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Resize an image to the specified size.
    
    Parameters
    ----------
    image : np.ndarray
        Input image. Can be 2D (grayscale) or 3D (color, channel last).
        Supports any numpy dtype (uint8, uint16, float32, float64, etc.)
    size : Tuple[int, int]
        Target size as (height, width).
    order : int, optional
        Interpolation order (default=1):
        - 0: nearest-neighbor
        - 1: bilinear
        - 2: bi-quadratic
        - 3: bi-cubic
    preserve_range : bool, optional
        If True, preserve the original data range and dtype (default=True).
        
    Returns
    -------
    np.ndarray
        Resized image with the same dtype as input.
        
    Examples
    --------
    >>> img = np.random.rand(100, 100).astype(np.float32)
    >>> resized = resize_image(img, (50, 50))
    >>> resized.shape
    (50, 50)
    >>> resized.dtype
    dtype('float32')
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")
    
    original_dtype = image.dtype
    target_height, target_width = size
    
    # Calculate zoom factors
    if image.ndim == 2:
        zoom_factors = (target_height / image.shape[0], target_width / image.shape[1])
    else:
        # For 3D images (H, W, C), don't zoom the channel dimension
        zoom_factors = (
            target_height / image.shape[0],
            target_width / image.shape[1],
            1.0
        )
    
    # Perform resize using scipy.ndimage.zoom
    resized = ndimage.zoom(image.astype(np.float64), zoom_factors, order=order)
    
    if preserve_range:
        # Clip to original dtype range if integer type
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            resized = np.clip(resized, info.min, info.max)
        resized = resized.astype(original_dtype)
    
    return resized


def rotate_image(
    image: np.ndarray,
    angle: float,
    order: int = 1,
    reshape: bool = False,
    cval: Union[float, int] = 0,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Rotate an image by the specified angle (in degrees).
    
    Parameters
    ----------
    image : np.ndarray
        Input image. Can be 2D (grayscale) or 3D (color, channel last).
        Supports any numpy dtype (uint8, uint16, float32, float64, etc.)
    angle : float
        Rotation angle in degrees. Positive values rotate counter-clockwise.
    order : int, optional
        Interpolation order (default=1):
        - 0: nearest-neighbor
        - 1: bilinear
        - 2: bi-quadratic
        - 3: bi-cubic
    reshape : bool, optional
        If True, output shape is adapted to contain the entire rotated image.
        If False, output shape is the same as input (default=False).
    cval : float or int, optional
        Value used for points outside the boundaries (default=0).
    preserve_range : bool, optional
        If True, preserve the original data range and dtype (default=True).
        
    Returns
    -------
    np.ndarray
        Rotated image with the same dtype as input.
        
    Examples
    --------
    >>> img = np.random.rand(100, 100).astype(np.float32)
    >>> rotated = rotate_image(img, 45)
    >>> rotated.shape
    (100, 100)
    >>> rotated.dtype
    dtype('float32')
    
    >>> # Rotate with expanded canvas
    >>> rotated_full = rotate_image(img, 45, reshape=True)
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")
    
    original_dtype = image.dtype
    
    if image.ndim == 2:
        # 2D grayscale image
        rotated = ndimage.rotate(
            image.astype(np.float64),
            angle,
            order=order,
            reshape=reshape,
            cval=float(cval)
        )
    else:
        # 3D image (H, W, C) - rotate each channel separately
        # or use axes parameter to rotate only spatial dimensions
        rotated = ndimage.rotate(
            image.astype(np.float64),
            angle,
            axes=(0, 1),  # Rotate in the (H, W) plane
            order=order,
            reshape=reshape,
            cval=float(cval)
        )
    
    if preserve_range:
        # Clip to original dtype range if integer type
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            rotated = np.clip(rotated, info.min, info.max)
        rotated = rotated.astype(original_dtype)
    
    return rotated


def bytescale_image(
    data: np.ndarray,
    imin: Optional[Union[float, int]] = None,
    imax: Optional[Union[float, int]] = None,
    omin: int = 0,
    omax: int = 255
) -> np.ndarray:
    """
    Scale input data to byte range [omin, omax].
    
    Linearly maps input values from [imin, imax] to output range [omin, omax],
    then converts to uint8. Useful for visualizing scientific data (e.g., FITS images)
    or preparing data for display/saving.
    
    Parameters
    ----------
    data : np.ndarray
        Input array of any numeric dtype.
    imin : float or int, optional
        Input minimum value for scaling. If None, uses data.min().
    imax : float or int, optional
        Input maximum value for scaling. If None, uses data.max().
    omin : int, optional
        Output minimum value (default=0).
    omax : int, optional
        Output maximum value (default=255).
        
    Returns
    -------
    np.ndarray
        Scaled array with dtype=uint8.
        
    Raises
    ------
    ValueError
        If imin >= imax (invalid input range).
        
    Examples
    --------
    >>> # Scale 16-bit solar image to 8-bit for display
    >>> img_16bit = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
    >>> img_8bit = bytescale_image(img_16bit, imin=0, imax=65535)
    
    >>> # Auto-scale float data with contrast stretch
    >>> data = np.random.rand(100, 100) * 1000 - 500  # range: [-500, 500]
    >>> scaled = bytescale_image(data)  # auto imin/imax from data
    
    >>> # Apply percentile-based scaling for better contrast
    >>> imin, imax = np.percentile(data, [1, 99])
    >>> scaled = bytescale_image(data, imin=imin, imax=imax)
    """
    # Convert to float64 for precision during calculations
    data = np.asarray(data, dtype=np.float64)
    
    # Auto-detect input range if not specified
    if imin is None:
        imin = np.nanmin(data)
    if imax is None:
        imax = np.nanmax(data)
    
    # Validate input range
    if imin >= imax:
        raise ValueError(f"imin ({imin}) must be less than imax ({imax})")
    
    # Linear scaling: [imin, imax] -> [0, 1] -> [omin, omax]
    scaled = (data - imin) / (imax - imin)
    scaled = scaled * (omax - omin) + omin
    
    # Clip to output range and convert to uint8
    scaled = np.clip(scaled, omin, omax).astype(np.uint8)
    
    return scaled


def circle_mask(
    image_size: Union[int, Tuple[int, int]],
    radius: float,
    center: Optional[Tuple[float, float]] = None,
    mask_type: Literal['inner', 'outer'] = 'inner'
) -> np.ndarray:
    """
    Generate a circular boolean mask.
    
    Creates a 2D boolean array with True values inside (or outside) a circle.
    Commonly used for solar disk masking in SDO/AIA or H-alpha imagery.
    
    Parameters
    ----------
    image_size : int or tuple of int
        Size of the output mask. If int, creates a square mask (size, size).
        If tuple, creates a mask with shape (height, width).
    radius : float
        Radius of the circle in pixels.
    center : tuple of float, optional
        Center coordinates as (cy, cx). If None, defaults to image center.
    mask_type : {'inner', 'outer'}, optional
        - 'inner': True inside the circle (default)
        - 'outer': True outside the circle
        
    Returns
    -------
    np.ndarray
        Boolean mask with shape (height, width).
        
    Raises
    ------
    ValueError
        If mask_type is not 'inner' or 'outer'.
        
    Examples
    --------
    >>> # Mask for 4096x4096 SDO image with solar disk radius ~1600 pixels
    >>> disk_mask = circle_mask(4096, radius=1600, mask_type='inner')
    >>> masked_image = np.where(disk_mask, image, 0)
    
    >>> # Rectangular image with off-center circle
    >>> mask = circle_mask((512, 1024), radius=200, center=(256, 600))
    
    >>> # Mask out the solar disk to analyze corona
    >>> corona_mask = circle_mask(4096, radius=1600, mask_type='outer')
    """
    # Handle image_size as int or tuple
    if isinstance(image_size, (int, np.integer)):
        height, width = image_size, image_size
    else:
        height, width = image_size
    
    # Default center: image center
    if center is None:
        cy, cx = height / 2.0, width / 2.0
    else:
        cy, cx = center
    
    # Create coordinate grids (memory-efficient with ogrid)
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Generate mask based on type
    if mask_type == 'inner':
        mask = distance_from_center < radius
    elif mask_type == 'outer':
        mask = distance_from_center >= radius
    else:
        raise ValueError(f"mask_type must be 'inner' or 'outer', got '{mask_type}'")
    
    return mask


def annulus_mask(
    image_size: Union[int, Tuple[int, int]],
    inner_radius: float,
    outer_radius: float,
    center: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Generate an annular (ring-shaped) boolean mask.
    
    Useful for analyzing solar features at specific radial distances,
    such as the chromosphere or inner corona.
    
    Parameters
    ----------
    image_size : int or tuple of int
        Size of the output mask.
    inner_radius : float
        Inner radius of the annulus in pixels.
    outer_radius : float
        Outer radius of the annulus in pixels.
    center : tuple of float, optional
        Center coordinates as (cy, cx). If None, defaults to image center.
        
    Returns
    -------
    np.ndarray
        Boolean mask with True values in the annular region.
        
    Examples
    --------
    >>> # Analyze region between 1.0 and 1.3 solar radii
    >>> solar_radius = 1600  # pixels
    >>> corona_ring = annulus_mask(4096, inner_radius=solar_radius, 
    ...                            outer_radius=solar_radius * 1.3)
    """
    inner = circle_mask(image_size, inner_radius, center=center, mask_type='inner')
    outer = circle_mask(image_size, outer_radius, center=center, mask_type='inner')
    return outer & ~inner


# Convenience aliases
resize = resize_image
rotate = rotate_image
bytescale = bytescale_image







# def get_median_value(smap):
#     data = smap.data
#     meta = smap.meta
#     radius = meta["R_SUN"]
#     mask = circle_mask(
#         image_size=data.shape[0],
#         radius = radius,# * (np.sqrt(3.0) / 2.),
#         mask_type = 'inner'
#     )
#     masked = data[mask==1]
#     median_value = np.median(masked)
#     return median_value


# def get_reference_median(reference_median_path):
#     if os.path.exists(reference_median_path) is False :
#         df = pd.read_csv(CSV_PATH, parse_dates=['datetime'])
#         reference_date = datetime.datetime(
#             year = 2011,
#             month = 1,
#             day = 1,
#             hour = 0,
#         )
#         mask = df['datetime'] == reference_date
#         row = df.loc[mask].iloc[0]
#         aia_193 = row['aia_193']
#         aia_211 = row['aia_211']
#         map_aia_193 = Map(f"{DATA_ROOT}/aia/{aia_193}")
#         map_aia_193 = register_map(map_aia_193)
#         median_val_aia_193 = get_median_value(map_aia_193)
#         map_aia_211 = Map(f"{DATA_ROOT}/aia/{aia_211}")
#         map_aia_211 = register_map(map_aia_211)
#         median_val_aia_211 = get_median_value(map_aia_211)
#         median_dict = {"aia_193" : median_val_aia_193, "aia_211" : median_val_aia_211}
#         pickle.dump(median_dict, open(reference_median_path, 'wb'))
#         print(f"Reference median values saved: {reference_median_path}")
#         del median_dict

#     median_dict = pickle.load(open(reference_median_path, 'rb'))
#     print(f"Reference median values loaded: {reference_median_path}")
#     return median_dict


# def normalize_map(smap, median_dict):
#     median_value = get_median_value(smap)
#     data = smap.data
#     meta = smap.meta
#     wavelength = int(meta["WAVELNTH"])
#     reference_median_value = median_dict[f"aia_{wavelength}"]
#     data = data * (reference_median_value/median_value)
#     smap_new = Map(data, meta)
#     return smap_new
