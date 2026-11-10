import pandas as pd

def set_index_col(df: pd.DataFrame, col: str = "Datetime") -> pd.DataFrame:
    return df.set_index(col).sort_index().rename_axis(None)

