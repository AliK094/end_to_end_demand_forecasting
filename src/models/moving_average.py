import pandas as pd

class MovingAverageForecaster:
    def __init__(self, forecast_horizon=28, window=28):
        self.forecast_horizon = forecast_horizon
        self.window = window

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if "id" not in df.columns or "sales" not in df.columns or "date" not in df.columns:
            raise ValueError("Input DataFrame must contain 'id', 'sales', and 'date' columns.")

        df = df.sort_values("date")

        recent = (
            df.groupby("id", group_keys=False)[["id", "sales", "date"]]
              .apply(lambda x: x.tail(self.window))
              .reset_index(drop=True)
        )

        result = recent.groupby("id")["sales"].mean().reset_index()

        forecast_values = result["sales"].values.reshape(-1, 1)
        forecast_df = pd.DataFrame(
            forecast_values.repeat(self.forecast_horizon, axis=1),
            columns=[f"F{i}" for i in range(1, self.forecast_horizon + 1)]
        )
        forecast_df.insert(0, "id", result["id"].values)

        return forecast_df
