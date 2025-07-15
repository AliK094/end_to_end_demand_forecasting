import os
import pandas as pd
from src.data.data_loader import DataLoader
from src.models.moving_average import MovingAverageForecaster
from src.models.sarima import SARIMAForecaster
from src.models.lgbm import LightGBMForecaster

from src.evaluation.evaluator import ForecastEvaluator
from src.data.visualization import DataVisualizer 

RESULTS_DIR = "results"
METRICS_FILE = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")

def save_forecast(model_name: str, forecast_df):
    # Save forecast
    os.makedirs(RESULTS_DIR, exist_ok=True)
    forecast_path = os.path.join(RESULTS_DIR, f"{model_name}_forecast.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecast saved to {forecast_path}")

def evaluate_and_save(model_name: str, forecast_df, test_df):
    # Evaluate
    evaluator = ForecastEvaluator(forecast_df=forecast_df, test_df=test_df)
    metrics = evaluator.evaluate()
    metrics["model"] = model_name
    print(f"{model_name} Evaluation Metrics:\n{metrics}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    if os.path.exists(METRICS_FILE):
        existing = pd.read_csv(METRICS_FILE)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"Metrics saved to {METRICS_FILE}")

def read_and_visualize_data():
    # Load data
    data_loader = DataLoader(cache_filename="cleaned_data_test_HOUSEHOLD_CA_1.parquet")
    train_df, test_df = data_loader.load_data(force_clean=False)

    # Visualize data
    vis = DataVisualizer()
    # vis.plot_total_sales_by_category(train_df)
    # vis.plot_weekly_sales_by_category(train_df)
    vis.plot_distribution(train_df, column="sales")
    # Uncomment to visualize sample series
    # vis.plot_sample_series(train_df, n=3)

    # print(train_df.head())

    print("Data types:", train_df.dtypes)
    print("Memory usage - train (MB):", train_df.memory_usage(deep=True).sum() / 1e6)
    print("Memory usage - test (MB):", test_df.memory_usage(deep=True).sum() / 1e6)

    return train_df, test_df

def run_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame):

    # Select and run model
    if model_name == "moving_average":
        forecaster = MovingAverageForecaster(forecast_horizon=28, window=28)
        forecast_df = forecaster.predict(train_df)
        save_forecast(model_name, forecast_df)
        evaluate_and_save(model_name, forecast_df, test_df)

    elif model_name == "sarima":
        forecaster = SARIMAForecaster(forecast_horizon=28, seasonal=True, order=(1, 0, 0), seasonal_order=(1, 0, 0, 26))
        forecast_df = forecaster.predict(train_df)
        save_forecast(model_name, forecast_df)
        evaluate_and_save(model_name, forecast_df, test_df)

    elif model_name == "lgbm":
        forecaster = LightGBMForecaster(forecast_horizon=28)
        forecaster.train(train_df)
        forecast_df = forecaster.predict(test_df)
        save_forecast(model_name, forecast_df)
        evaluate_and_save(model_name, forecast_df, test_df)

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def main():
    # Read and visualize data
    train_df, test_df = read_and_visualize_data()

    print("Train Data Shape:", train_df.shape)
    print("Test Data Shape:", test_df.shape)

    # Change to test different models
    # run_model("moving_average", train_df, test_df)
    # run_model("sarima", train_df, test_df)
    run_model("lgbm", train_df, test_df)

if __name__ == "__main__":
    main()
