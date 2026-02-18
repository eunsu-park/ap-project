"""Data acquisition module for solar physics research.

This module provides functions to fetch solar and solar wind observation data
from PostgreSQL databases using the egghouse.database library.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from egghouse.database import PostgresManager, to_dataframe


# SDO Channel mapping: legacy name -> (telescope, channel)
SDO_CHANNEL_MAP = {
    'aia_193': ('aia', '193'),
    'aia_211': ('aia', '211'),
    'hmi_magnetogram': ('hmi', 'm_45s'),
}

# Default time range for SDO closest match (±minutes)
DEFAULT_TIME_RANGE_MINUTES = 12

# Default maximum quality value (0-9 allowed)
DEFAULT_MAX_QUALITY = 9


# Database configurations (new schema)
import os

DB_CONFIG_SPACE_WEATHER = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'database': 'space_weather',
    'user': os.environ.get('DB_USER', 'eunsupark'),
    'password': os.environ.get('DB_PASSWORD', 'eunsupark'),
    'log_queries': False
}

DB_CONFIG_SOLAR_IMAGES = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'database': 'solar_images',
    'user': os.environ.get('DB_USER', 'eunsupark'),
    'password': os.environ.get('DB_PASSWORD', 'eunsupark'),
    'log_queries': False
}

# Backward compatibility aliases
DB_CONFIG_OMNI = DB_CONFIG_SPACE_WEATHER
DB_CONFIG_SDO = DB_CONFIG_SOLAR_IMAGES


def get_omni_data(
    start_date: datetime,
    end_date: datetime,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """Fetches OMNI solar wind data from the omni_low_resolution table.

    Args:
        start_date: Start date for the query range (inclusive).
        end_date: End date for the query range (exclusive).
        columns: List of column names to retrieve. If None, retrieves all columns.

    Returns:
        DataFrame containing the OMNI solar wind data within the specified
        date range.

    Raises:
        Exception: If database connection or query fails.
    """
    with PostgresManager(**DB_CONFIG_SPACE_WEATHER) as db:
        results = db.select_date_range(
            table_name='omni_low_resolution',
            date_column='datetime',
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            order_by='datetime'
        )
        return to_dataframe(results, parse_dates=['datetime'])


def get_sdo_closest_to_time(
    target_time: datetime,
    telescope: str,
    channel: str,
    time_range_minutes: int = DEFAULT_TIME_RANGE_MINUTES,
    max_quality: int = DEFAULT_MAX_QUALITY,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """Fetches the SDO image closest to target time within the specified range.

    Finds data where:
    - telescope and channel match
    - quality is between 0 and max_quality (inclusive)
    - datetime is within ±time_range_minutes of target_time
    - Returns the single closest record to target_time

    Args:
        target_time: Target datetime to find the closest image.
        telescope: Telescope name ('aia' or 'hmi').
        channel: Channel name ('193', '211', 'm_45s', etc.).
        time_range_minutes: Time range in minutes (±). Default is 12.
        max_quality: Maximum allowed quality value (inclusive). Default is 9.
        columns: List of column names to retrieve. If None, retrieves all.

    Returns:
        DataFrame with at most one row containing the closest matching image.
        Empty DataFrame if no matching data found.

    Raises:
        Exception: If database connection or query fails.
    """
    start_time = target_time - timedelta(minutes=time_range_minutes)
    end_time = target_time + timedelta(minutes=time_range_minutes)

    col_str = ', '.join(columns) if columns else '*'

    query = f"""
        SELECT {col_str} FROM sdo
        WHERE telescope = %s
          AND channel = %s
          AND quality >= 0 AND quality <= %s
          AND datetime >= %s
          AND datetime <= %s
        ORDER BY ABS(EXTRACT(EPOCH FROM (datetime - %s::timestamp)))
        LIMIT 1
    """

    with PostgresManager(**DB_CONFIG_SOLAR_IMAGES) as db:
        results = db.execute(
            query,
            params=(telescope, channel, max_quality, start_time, end_time, target_time),
            fetch=True
        )
        return to_dataframe(results, parse_dates=['datetime'])


def get_sdo_data(
    start_date: datetime,
    end_date: datetime,
    telescope: str,
    channel: str,
    columns: Optional[list] = None,
    quality: Optional[int] = 0
) -> pd.DataFrame:
    """Fetches SDO observation data from the unified sdo table.

    Args:
        start_date: Start date for the query range (inclusive).
        end_date: End date for the query range (exclusive).
        telescope: Telescope name ('aia' or 'hmi').
        channel: Channel name ('193', '211', 'm_45s', etc.).
        columns: List of column names to retrieve. If None, retrieves all.
        quality: Quality flag filter. Default is 0 (good quality only).
                 Set to None to include all quality values.

    Returns:
        DataFrame containing the SDO observation data.

    Raises:
        Exception: If database connection or query fails.
    """
    where_conditions = {'telescope': telescope, 'channel': channel}
    if quality is not None:
        where_conditions['quality'] = quality

    with PostgresManager(**DB_CONFIG_SOLAR_IMAGES) as db:
        results = db.select_date_range(
            table_name='sdo',
            date_column='datetime',
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            where=where_conditions,
            order_by='datetime'
        )
        return to_dataframe(results, parse_dates=['datetime'])


def get_sdo_aia_193(
    start_date: datetime,
    end_date: datetime,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """Fetches SDO AIA 193 wavelength observation data.

    Args:
        start_date: Start date for the query range (inclusive).
        end_date: End date for the query range (exclusive).
        columns: List of column names to retrieve. If None, retrieves all columns.

    Returns:
        DataFrame containing the AIA 193 observation data within the specified
        date range.

    Raises:
        Exception: If database connection or query fails.
    """
    return get_sdo_data(start_date, end_date, 'aia', '193', columns)


def get_sdo_aia_211(
    start_date: datetime,
    end_date: datetime,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """Fetches SDO AIA 211 wavelength observation data.

    Args:
        start_date: Start date for the query range (inclusive).
        end_date: End date for the query range (exclusive).
        columns: List of column names to retrieve. If None, retrieves all columns.

    Returns:
        DataFrame containing the AIA 211 observation data within the specified
        date range.

    Raises:
        Exception: If database connection or query fails.
    """
    return get_sdo_data(start_date, end_date, 'aia', '211', columns)


def get_sdo_hmi_magnetogram(
    start_date: datetime,
    end_date: datetime,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """Fetches SDO HMI magnetogram observation data.

    Args:
        start_date: Start date for the query range (inclusive).
        end_date: End date for the query range (exclusive).
        columns: List of column names to retrieve. If None, retrieves all columns.

    Returns:
        DataFrame containing the HMI magnetogram data within the specified
        date range.

    Raises:
        Exception: If database connection or query fails.
    """
    return get_sdo_data(start_date, end_date, 'hmi', 'm_45s', columns)


def get_all_data_at_time(
    t: datetime,
    time_range_minutes: int = DEFAULT_TIME_RANGE_MINUTES,
    max_quality: int = DEFAULT_MAX_QUALITY
) -> dict:
    """Fetches all observation data at a specific time.

    For OMNI data, retrieves exact datetime match.
    For SDO data, finds the closest image within ±time_range_minutes
    where quality is between 0 and max_quality.

    Args:
        t: Target datetime to query.
        time_range_minutes: Time range in minutes for SDO matching.
                            Default is 12 (±12 minutes).
        max_quality: Maximum allowed quality value (inclusive). Default is 9.

    Returns:
        Dictionary containing DataFrames for each data source:
        - 'omni': OMNI solar wind data (datetime == t)
        - 'aia_193': SDO AIA 193 data (closest within range)
        - 'aia_211': SDO AIA 211 data (closest within range)
        - 'hmi_magnetogram': SDO HMI magnetogram data (closest within range)

    Raises:
        Exception: If database connection or query fails.
    """
    result = {}

    # Fetch OMNI data (exact datetime match)
    with PostgresManager(**DB_CONFIG_SPACE_WEATHER) as db:
        omni_results = db.select(
            table_name='omni_low_resolution',
            where={'datetime': t}
        )
        result['omni'] = to_dataframe(omni_results, parse_dates=['datetime'])

    # Fetch SDO data (closest match within time range)
    for legacy_name, (telescope, channel) in SDO_CHANNEL_MAP.items():
        result[legacy_name] = get_sdo_closest_to_time(
            target_time=t,
            telescope=telescope,
            channel=channel,
            time_range_minutes=time_range_minutes,
            max_quality=max_quality
        )

    return result
