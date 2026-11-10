import pandas as pd
import json
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def affine_transformer(col: pd.Series, scale: float, offset: float):
    return (col * scale) + offset


def inverse_affine_transformer(col: np.ndarray, scale: float, offset: float):
    return (col - offset) / scale


def _extract_affine_params(scaler):
    scale_to_dict = lambda scale, offset: {"scale": scale[0], "offset": offset[0]}
    match scaler:
        case StandardScaler():
            return scale_to_dict(1 / scaler.scale_, -scaler.mean_ / scaler.scale_)
        case MinMaxScaler():
            return scale_to_dict(scaler.scale_, scaler.min_)
        case RobustScaler():
            return scale_to_dict(1 / scaler.scale_, -scaler.center_ / scaler.scale_)
        case _:
            raise TypeError(f"no affine extraction for {type(scaler).__name__}")


class FeatureScaler:
    def __init__(
        self,
        site_artifact_directory: Path,
        standard_scaled_features: list[str] = None,
        minmax_scaled_features: list[str] = None,
        robust_scaled_features: list[str] = None,
    ):
        self.json_directory = site_artifact_directory / "scale_parameters.json"

        self.standard_scaled_features = standard_scaled_features or []
        self.minmax_scaled_features = minmax_scaled_features or []
        self.robust_scaled_features = robust_scaled_features or []

        self._scalers = (
            (self.standard_scaled_features, StandardScaler(), "standard"),
            (self.minmax_scaled_features, MinMaxScaler(), "minmax"),
            (self.robust_scaled_features, RobustScaler(), "robust"),
        )
        self.affine_params = {}

    def save_affine_params(self):
        self.json_directory.parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_directory, "w") as file:
            json.dump(self.affine_params, file, indent=2)

    def load_affine_params(self):
        with open(self.json_directory, "r") as file:
            self.affine_params = json.load(file)

    def fit(self, df):
        df = df.copy()
        for features, scaler, name in self._scalers:
            if features:
                for feature in features:
                    df[[feature]] = scaler.fit_transform(df[[feature]])
                    self.affine_params[feature] = _extract_affine_params(scaler)
        self.save_affine_params()
        return self

    def transform(self, df):
        df = df.copy()
        for features, _, name in self._scalers:
            if features:
                for feature in features:
                    affine_params = self.affine_params[feature]
                    df[[feature]] = affine_transformer(df[[feature]], **affine_params)
        return df

    def inverse_transform_col(self, target: np.ndarray, target_col_name: str):
        affine_params = self.affine_params[target_col_name]
        return inverse_affine_transformer(target, **affine_params)


def scale_df(df: pd.DataFrame, site_artifact_directory: Path, scaler_features: dict):
    scaler = FeatureScaler(site_artifact_directory=site_artifact_directory, **scaler_features)
    scaler.load_affine_params()
    scaled_df = scaler.transform(df)
    return scaled_df, scaler