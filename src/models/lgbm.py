import pandas as pd
import numpy as np
import lightgbm as lgb

class LightGBMForecaster:
    def __init__(self, forecast_horizon=28):
        self.forecast_horizon = forecast_horizon
        self.model = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["dayofweek"] = df["date"].dt.dayofweek
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["day"] = df["date"].dt.day

        for lag in [1, 7, 14, 28]:
            df[f"lag_{lag}"] = df.groupby("id", observed=True)["sales"].shift(lag)

        for window in [7, 14, 28]:
            df[f"rolling_mean_{window}"] = (
                df.groupby("id", observed=True)["sales"].shift(1).rolling(window).mean().reset_index(level=0, drop=True)
            )

        # df["sales"] = np.log1p(df["sales"])
        df["sales"] = df["sales"].astype(float)

        return df

    def train(self, train_df: pd.DataFrame):
        df = self.create_features(train_df)
        required_columns = [col for col in df.columns if col.startswith("lag_") or col.startswith("rolling_")]
        required_columns += ["sales"]  # target column
        df = df.dropna(subset=required_columns)

        features = [col for col in df.columns if col not in ["id", "sales", "date"]]
        print(f'Features: {features}')  # print features
        X_train = df[features]
        y_train = df["sales"]

        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1
        }
        self.model = lgb.train(params, train_data, num_boost_round=100)

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        df = self.create_features(test_df)
        predictions = []

        for i in range(self.forecast_horizon):
            current_date = df["date"].min() + pd.Timedelta(days=i)
            X = df[df["date"] == current_date]
            X_features = X.drop(columns=["id", "sales", "date"], errors='ignore')
            preds = self.model.predict(X_features)
            df.loc[X.index, "sales"] = preds
            # linear_preds = np.expm1(preds)
            predictions.append(preds)

        forecast_df = df[df["date"] >= df["date"].min()][["id", "date"]].copy()
        forecast_df["prediction"] = np.concatenate(predictions)

        forecast_df = forecast_df.groupby("id", observed=True)["prediction"].apply(list).reset_index()
        forecast_df[[f"F{i+1}" for i in range(self.forecast_horizon)]] = pd.DataFrame(
            forecast_df["prediction"].to_list(), index=forecast_df.index
        )
        forecast_df.drop(columns="prediction", inplace=True)

        return forecast_df
