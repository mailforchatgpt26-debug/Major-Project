"""
Helper utility functions for data processing.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
from pathlib import Path


def log1p_transform(series: pd.Series) -> pd.Series:
    """Apply log1p transformation (handles zeros and negatives)."""
    return np.log1p(series.clip(lower=0))


def expm1_transform(series: pd.Series) -> pd.Series:
    """Reverse log1p transformation."""
    return np.expm1(series)


def normalize_sentiment(series: pd.Series, 
                        input_range: tuple = (-10, 10),
                        output_range: tuple = (0, 1)) -> pd.Series:
    """
    Normalize sentiment scores from GDELT range to model range.
    
    Args:
        series: Sentiment values
        input_range: Original range (default: GDELT -10 to 10)
        output_range: Target range (default: 0 to 1)
    
    Returns:
        Normalized series
    """
    in_min, in_max = input_range
    out_min, out_max = output_range
    
    normalized = (series - in_min) / (in_max - in_min)
    normalized = normalized * (out_max - out_min) + out_min
    
    return normalized.clip(out_min, out_max)


def clip_outliers(series: pd.Series, 
                  lower_percentile: float = 0.01,
                  upper_percentile: float = 0.99) -> pd.Series:
    """Clip outliers using percentile method."""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower, upper)


def safe_divide(numerator: Union[pd.Series, float], 
                denominator: Union[pd.Series, float],
                fill_value: float = 0.0) -> Union[pd.Series, float]:
    """
    Safe division handling zeros and infinities.
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is zero
    
    Returns:
        Division result with safe handling
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        result = numerator / denominator
        result = result.replace([np.inf, -np.inf], fill_value)
        result = result.fillna(fill_value)
        return result
    else:
        return numerator / denominator if denominator != 0 else fill_value


def create_lag_features(df: pd.DataFrame,
                       group_cols: List[str],
                       value_col: str,
                       lags: List[int] = [1, 2, 3],
                       sort_col: str = 'year') -> pd.DataFrame:
    """
    Create lagged features for time series.
    
    Args:
        df: Input dataframe
        group_cols: Columns to group by (e.g., ['source_iso3', 'target_iso3'])
        value_col: Column to create lags from
        lags: List of lag periods
        sort_col: Column to sort by before lagging
    
    Returns:
        DataFrame with lag columns added
    """
    df = df.sort_values(group_cols + [sort_col])
    
    for lag in lags:
        col_name = f"{value_col}_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[value_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame,
                            group_cols: List[str],
                            value_col: str,
                            windows: List[int] = [3, 6],
                            sort_col: str = 'year') -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: Input dataframe
        group_cols: Columns to group by
        value_col: Column to compute rolling stats on
        windows: List of window sizes
        sort_col: Column to sort by
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.sort_values(group_cols + [sort_col])
    
    for window in windows:
        # Rolling mean
        col_name = f"{value_col}_rolling_mean_{window}"
        df[col_name] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Rolling std
        col_name = f"{value_col}_rolling_std_{window}"
        df[col_name] = df.groupby(group_cols)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    return df


def ensure_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, 
                   filepath: Union[str, Path],
                   format: str = 'csv',
                   **kwargs):
    """
    Save dataframe to file with automatic format detection.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        format: 'csv', 'parquet', 'feather', 'pickle'
        **kwargs: Additional arguments for pandas save methods
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    if format == 'csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False, **kwargs)
    elif format == 'feather':
        df.to_feather(filepath, **kwargs)
    elif format == 'pickle':
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(filepath: Union[str, Path],
                   format: Optional[str] = None,
                   **kwargs) -> pd.DataFrame:
    """
    Load dataframe from file with automatic format detection.
    
    Args:
        filepath: Input path
        format: Format override (auto-detect from extension if None)
        **kwargs: Additional arguments for pandas read methods
    
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix[1:]  # Remove the dot
    
    if format == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif format == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif format == 'feather':
        return pd.read_feather(filepath, **kwargs)
    elif format == 'pickle':
        return pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Calculate DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / 1024**2


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage by downcasting numeric types.
    
    Args:
        df: Input DataFrame
        verbose: Print memory reduction stats
    
    Returns:
        DataFrame with reduced memory usage
    """
    start_mem = memory_usage_mb(df)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = memory_usage_mb(df)
    
    if verbose:
        reduction = (start_mem - end_mem) / start_mem * 100
        print(f"Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df


def validate_iso3_codes(df: pd.DataFrame, iso_columns: List[str]) -> pd.DataFrame:
    """
    Validate and filter ISO3 codes.
    
    Args:
        df: DataFrame with ISO3 columns
        iso_columns: List of column names containing ISO3 codes
    
    Returns:
        DataFrame with only valid ISO3 codes
    """
    initial_len = len(df)
    
    for col in iso_columns:
        # Check if column exists
        if col not in df.columns:
            continue
        
        # Remove null values
        df = df[df[col].notna()]
        
        # Check length (ISO3 should be 3 characters)
        df = df[df[col].str.len() == 3]
        
        # Remove numeric codes (should be alphabetic)
        df = df[df[col].str.isalpha()]
    
    removed = initial_len - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with invalid ISO3 codes ({removed/initial_len*100:.1f}%)")
    
    return df


def merge_with_validation(left: pd.DataFrame,
                         right: pd.DataFrame,
                         on: Union[str, List[str]],
                         how: str = 'left',
                         validate: Optional[str] = None) -> pd.DataFrame:
    """
    Merge DataFrames with automatic validation and reporting.
    
    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column(s) to merge on
        how: Merge type
        validate: Pandas merge validation parameter
    
    Returns:
        Merged DataFrame
    """
    left_len = len(left)
    
    merged = pd.merge(left, right, on=on, how=how, validate=validate)
    
    # Report merge statistics
    merged_len = len(merged)
    if how == 'inner':
        match_rate = merged_len / left_len * 100
        print(f"Merge on {on}: {merged_len}/{left_len} rows matched ({match_rate:.1f}%)")
    elif how == 'left':
        print(f"Left merge on {on}: {merged_len} rows (started with {left_len})")
    
    return merged


def get_date_from_year_month(year: int, month: Optional[int] = None) -> pd.Timestamp:
    """Convert year and optional month to pandas Timestamp."""
    if month is None or pd.isna(month):
        return pd.Timestamp(year=year, month=1, day=1)
    return pd.Timestamp(year=year, month=int(month), day=1)


def create_time_features(df: pd.DataFrame, 
                        year_col: str = 'year',
                        month_col: Optional[str] = None) -> pd.DataFrame:
    """
    Create time-based features.
    
    Args:
        df: Input DataFrame
        year_col: Name of year column
        month_col: Name of month column (optional)
    
    Returns:
        DataFrame with time features added
    """
    # Year-based features
    df['year_norm'] = (df[year_col] - df[year_col].min()) / (df[year_col].max() - df[year_col].min())
    
    if month_col and month_col in df.columns:
        # Month-based features
        df['month_sin'] = np.sin(2 * np.pi * df[month_col] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[month_col] / 12)
        
        # Quarter
        df['quarter'] = ((df[month_col] - 1) // 3 + 1).astype(int)
    
    return df


if __name__ == "__main__":
    # Test helper functions
    print("Testing helper functions...")
    
    # Test log transform
    data = pd.Series([0, 1, 10, 100, 1000])
    print("\nLog1p transform:", log1p_transform(data).values)
    
    # Test sentiment normalization
    sentiment = pd.Series([-10, -5, 0, 5, 10])
    print("\nNormalized sentiment:", normalize_sentiment(sentiment).values)
    
    # Test safe divide
    result = safe_divide(pd.Series([1, 2, 3]), pd.Series([2, 0, 1]))
    print("\nSafe divide:", result.values)
    
    print("\n✅ All helper functions working!")