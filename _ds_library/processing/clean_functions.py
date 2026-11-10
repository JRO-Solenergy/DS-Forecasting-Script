import pandas as pd

def clip_cols(df: pd.DataFrame, cols: list[str] = None, upper: float = None, lower: float = None) -> pd.DataFrame:
    if cols is None:
        df = df.clip(lower=lower, upper=upper)
    else:
        df[cols] = df[cols].clip(lower=lower, upper=upper)
    return df

def interpolate_then_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.interpolate().bfill().ffill()

def zero_on_dayofweek(df: pd.DataFrame, col: str, day: int = 6) -> pd.DataFrame:
    df.loc[df.index.dayofweek == day, col] = 0
    return df