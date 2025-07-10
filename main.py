import os
import pandas as pd
from src.data.data_loader import DataLoader
from src.models.moving_average import MovingAverageForecaster
from src.evaluation.evaluator import ForecastEvaluator

def run_model(model_name: str):
    # Load data
    data_loader = DataLoader()
    train_df, test_df = data_loader.load_data()

    # Select model
    if model_name == "moving_average":
        forecaster = MovingAverageForecaster(forecast_horizon=28, window=28)
        forecast_df = forecaster.predict(train_df)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Save forecast
    os.makedirs("results", exist_ok=True)
    forecast_path = f"results/{model_name}_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"Forecast saved to {forecast_path}")

    # Evaluate
    evaluator = ForecastEvaluator(forecast_df=forecast_df, test_df=test_df)
    metrics = evaluator.evaluate()
    print(f"{model_name} Evaluation Metrics:")
    print(metrics)

    # Save metrics
    metrics["model"] = model_name
    metrics_df = pd.DataFrame([metrics])
    metrics_path = "results/evaluation_metrics.csv"
    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

def main():
    # Change this to test other models later
    run_model("moving_average")

if __name__ == "__main__":
    main()
