import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ForecastEvaluator:
    def __init__(self, forecast_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "id"):
        self.forecast_df = forecast_df.copy()
        self.test_df = test_df.copy()
        self.id_col = id_col
        self.merged_df = None

    def prepare_data(self):
        # Pivot test_df to match forecast format (wide with F1...F28)
        grouped = (
            self.test_df.sort_values(["id", "date"])
                        .groupby("id", group_keys=False, observed=True)
        )
        actual_df = grouped["sales"].apply(list).apply(pd.Series).reset_index()
        actual_df.columns = [self.id_col] + [f"F{i+1}" for i in range(actual_df.shape[1] - 1)]

        # Save forecast
        # actual_df.to_csv(f"results/lgbm_actual.csv", index=False)
        # print(f"Actual data saved to results/lgbm_actual.csv")

        # Melt both forecast and actual into long format
        forecast_long = self.forecast_df.melt(id_vars=self.id_col, var_name="F", value_name="forecast")
        actual_long = actual_df.melt(id_vars=self.id_col, var_name="F", value_name="actual")

        # Merge on id and F
        self.merged_df = pd.merge(forecast_long, actual_long, on=[self.id_col, "F"])


    def evaluate(self):
        if self.merged_df is None:
            self.prepare_data()

        y_true = self.merged_df["actual"].values
        y_pred = self.merged_df["forecast"].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "sMAPE": round(smape, 4)
        }
