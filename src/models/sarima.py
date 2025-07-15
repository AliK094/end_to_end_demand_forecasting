import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

class SARIMAForecaster:
    def __init__(self, forecast_horizon=28, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), seasonal=True):
        self.forecast_horizon = forecast_horizon
        self.order = order
        self.seasonal_order = seasonal_order if seasonal else (0, 0, 0, 0)

    def _aggregate_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly_df = (
            df.groupby(["id", "week"], observed=True)["sales"]
              .sum()
              .reset_index()
              .sort_values(["id", "week"])
        )
        return weekly_df
    
    def plot_sales_by_id(self, id_val: str, series: pd.Series, save=True):
        if not isinstance(series, pd.Series):
            raise ValueError("Input 'series' must be a pandas Series.")

        if series.index.inferred_type not in ["datetime64", "period"]:
            raise ValueError("The series index must be datetime-like for plotting time series.")

        plt.figure(figsize=(12, 4))
        plt.plot(series.index, series.values, label=f"{id_val} weekly sales")
        plt.title(f"Weekly Sales - ID: {id_val}")
        plt.xlabel("Week")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.tight_layout()

        if save:
            os.makedirs("results/plots", exist_ok=True)
            plt.savefig(f"results/plots/weekly_sales_id_{id_val}.png")
            plt.close()
        else:
            plt.show()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Starting SARIMA forecasting...")

        if "id" not in df.columns or "sales" not in df.columns or "date" not in df.columns:
            raise ValueError("Input DataFrame must contain 'id', 'sales', and 'date' columns.")

        # Aggregate daily sales to weekly sales
        weekly_df = self._aggregate_to_weekly(df)
        print("Weekly data shape:", weekly_df.shape)

        forecasts = []
        ids = weekly_df["id"].unique()
        weekly_steps = -(-self.forecast_horizon // 7)  # ceiling division

        for id_val in tqdm(ids, desc="Fitting SARIMA models on weekly data"):
            subset = weekly_df[weekly_df["id"] == id_val].sort_values("week")
            series = pd.Series(subset["sales"].values, index=pd.DatetimeIndex(subset["week"]))
            series.index.freq = series.index.inferred_freq or "W-MON"


            # self.plot_sales_by_id(id_val, series, save=True)

            # Clean series before modeling
            if series.isnull().any() or len(series) < 10 or series.std() == 0 or not np.isfinite(series).all():
                mean_val = series.mean() if not series.empty else 0.0
                fallback = [mean_val] * self.forecast_horizon
                forecasts.append([id_val] + fallback)
                continue

            # Clip extreme values
            q_low, q_high = series.quantile([0.01, 0.99])
            series = series.clip(lower=q_low, upper=q_high)

            # self.plot_sales_by_id(id_val, series, save=False)

            # Fit SARIMA model
            try:
                model = SARIMAX(series, 
                                order=self.order, 
                                seasonal_order=self.seasonal_order,
                                enforce_stationarity=False, 
                                enforce_invertibility=False, 
                                simple_differencing=True)
                results = model.fit(disp=False)
                weekly_forecast = results.forecast(steps=weekly_steps)
            except Exception as e:
                print(f"Warning: SARIMA failed for {id_val} with error: {e}")
                mean_val = series.mean() if not series.empty else 0.0
                weekly_forecast = [mean_val] * weekly_steps

            # Distribute each weekly value over 7 days
            daily_forecast = [v / 7 for v in weekly_forecast for _ in range(7)]
            daily_forecast = daily_forecast[:self.forecast_horizon]  # trim to exact horizon

            forecasts.append([id_val] + daily_forecast)

            # exit(0)

        # Create forecast DataFrame
        columns = ["id"] + [f"F{i+1}" for i in range(self.forecast_horizon)]
        forecast_df = pd.DataFrame(forecasts, columns=columns)

        return forecast_df
