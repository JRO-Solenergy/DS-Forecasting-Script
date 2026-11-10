import pandas as pd
from datetime import datetime

def  set_datetime_col(df: pd.DataFrame, datetime_col: str = "Datetime") -> pd.DataFrame:
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    return df

def floor_datetime_col(df: pd.DataFrame, datetime_col: str = "Datetime", resolution: str = "5min") -> pd.DataFrame:
    df[datetime_col] = df[datetime_col].dt.floor(resolution)
    return df

def reindex_datetime_col(df: pd.DataFrame, start_datetime:str = None, end_datetime:str = None, resolution:str="5min") -> pd.DataFrame:
    actual_start = pd.Timestamp(start_datetime) if start_datetime is not None else df.index.min()
    actual_end = pd.Timestamp(end_datetime) if end_datetime is not None else df.index.max()
    full_index = pd.date_range(start=actual_start.floor(resolution), end=actual_end.floor(resolution), freq=resolution)
    return df.reindex(full_index)

def fill_missing_time_intervals(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    return df.asfreq(freq)

def extend_future_rows(df: pd.DataFrame, periods: int = 288, freq: str = "5min", start_datetime: datetime = None) -> pd.DataFrame:
    if start_datetime is None:
        start_datetime = df.index[-1]
    future_index = pd.date_range(start_datetime, periods=periods + 1, freq=freq)[1:]
    return df.reindex(df.index.append(future_index))