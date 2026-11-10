import inspect
import numpy as np
import onnxruntime as ort
import pandas as pd

def run_onnx(onnx_path, inputs) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path))
    expected_inputs = {i.name for i in session.get_inputs()}
    if set(inputs) != expected_inputs:
        raise ValueError(f"ONNX expects inputs {sorted(expected_inputs)}, got {sorted(inputs)}")
    return session.run(None, inputs)[0][0]

def build_onnx_inputs(window_df: pd.DataFrame, window_size: int, target_col: str,
                      time_cols: list[str], weekday_col: str) -> dict[str, np.ndarray]:
    past = window_df.iloc[:window_size]
    future = window_df.iloc[window_size:]

    return {
        "x_past":         past[time_cols].to_numpy(np.float32)[None],      # [1, 144, n_features]
        "x_future":       future[time_cols].to_numpy(np.float32)[None],    # [1, 288, n_features]
        "y_past":         past[target_col].to_numpy(np.float32)[None],     # [1, 144]
        "weekday_past":   past[weekday_col].to_numpy(np.int64)[None],      # [1, 144]
        "weekday_future": future[weekday_col].to_numpy(np.int64)[None],    # [1, 288]
    }