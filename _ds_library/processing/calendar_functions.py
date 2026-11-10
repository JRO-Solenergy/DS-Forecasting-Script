import numpy as np
import pandas as pd

TAU = 2 * np.pi

def add_cyclical_min_of_day(df: pd.DataFrame) -> pd.DataFrame:
    minute_of_day = df.index.hour * 60 + df.index.minute
    total_minutes_per_day = 24 * 60
    phase = TAU * minute_of_day / total_minutes_per_day
    df["min_of_day_cos"] = np.cos(phase)
    df["min_of_day_sin"] = np.sin(phase)
    return df

def add_day_of_week(df: pd.DataFrame, col="day_of_week") -> pd.DataFrame:
    df[col] = df.index.dayofweek
    return df