from _ds_library.staging import stage
from _ds_library.processing import (
    dataframe_functions as frame,
    datetime_functions as dates,
    clean_functions as clean,
    scaler_functions as scale,
    calendar_functions as cal,
    onnx_functions as onnx_fcn,
)

from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta


def forecast_demand(demand_df: pd.DataFrame, current_datetime: datetime):
    artifact_directory = Path(__file__).parent / "artifacts"
    current_datetime = pd.Timestamp(current_datetime).floor("5min")
    start_datetime = current_datetime - timedelta(hours=12)
    end_datetime = current_datetime - timedelta(minutes=5)
    with stage("cleaning"):
        demand_df = (
            demand_df.pipe(dates.set_datetime_col, datetime_col="timestamp")
            .pipe(dates.floor_datetime_col, datetime_col="timestamp")
            .pipe(frame.set_index_col, col="timestamp")
            .pipe(dates.reindex_datetime_col, start_datetime=start_datetime, end_datetime=end_datetime)
            .pipe(dates.fill_missing_time_intervals)
            .pipe(clean.interpolate_then_fill)
            .pipe(clean.clip_cols, lower=0)
        )
    with stage("feature"):
        demand_df = (
            demand_df
            .pipe(dates.extend_future_rows)
            .pipe(cal.add_cyclical_min_of_day)
            .pipe(cal.add_day_of_week)
        )
    with stage("scale"):
        scaled_demand_df, scaler = scale.scale_df(
            demand_df, artifact_directory,
            scaler_features={"minmax_scaled_features": ["value"]}
        )
    with stage("predict"):
        onnx_directory = artifact_directory / "test.onnx"
        inputs = onnx_fcn.build_onnx_inputs(scaled_demand_df, 144, "value", ["min_of_day_cos", "min_of_day_sin"], "day_of_week")
        prediction = onnx_fcn.run_onnx(onnx_directory, inputs)
    with stage("postprocess"):
        forecast = pd.DataFrame({
            "forecast": scaler.inverse_transform_col(prediction, "value"),
        }, index = demand_df.index[-288:]).clip(lower=0).pipe(clean.zero_on_dayofweek, col="forecast")

    return forecast["forecast"].tolist()

def main():
    # Use datetime.now() if used in the actual process, might have to check if this gives UTC time or UTC8 depending on the machine
    current_datetime = datetime(2026, 10, 24, 13)
    demand_df = pd.read_csv("DemandDataOctoberTestingHub.csv")
    forecast = forecast_demand(demand_df, current_datetime)
    print(forecast)

if __name__ == '__main__':
    main()
