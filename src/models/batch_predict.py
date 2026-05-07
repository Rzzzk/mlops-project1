import os
import prefect
from prefect import task, flow
import duckdb
import pandas as pd
import wandb
import joblib
import hydra
from omegaconf import DictConfig

@task(name="1. Extract Data from MotherDuck", retries=2)
def extract_data(query: str) -> pd.DataFrame:
    print("Connecting to MotherDuck to extract data...")
    con = duckdb.connect("md:titanic_db")
    df = con.execute(query).df()
    con.close()
    return df

@task(name="2. Fetch Production Model from W&B", retries=2)
def fetch_model(model_registry_path: str) -> str:
    print(f"Downloading model from {model_registry_path}...")
    api = wandb.Api()
    artifact = api.artifact(model_registry_path)
    model_dir = artifact.download()
    return os.path.join(model_dir, "best_model.pkl")

@task(name="3. Run Batch Predictions")
def predict(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    print("Loading model into memory and running predictions...")
    model = joblib.load(model_path)
    
    # Run the batch prediction
    predictions = model.predict(df)
    
    # Attach predictions back to the original dataframe
    results_df = df.copy()
    
    # Map the integers back to readable text (optional, but good for BI dashboards)
    results_df["predicted_outcome"] = ["Survived" if p == 1 else "Did Not Survive" for p in predictions]
    return results_df

@task(name="4. Load Predictions to MotherDuck", retries=2)
def load_predictions(results_df: pd.DataFrame, table_name: str):
    print(f"Writing {len(results_df)} predictions back to MotherDuck table: {table_name}...")
    con = duckdb.connect("md:titanic_db")
    # Overwrite if exists so we can run this batch job repeatedly without duplicating data
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM results_df")
    con.close()
    print("Batch load complete!")

@flow(name="Titanic Batch Forecasting Pipeline", log_prints=True)
def batch_forecasting_flow(cfg: DictConfig):
    # Step 1: Extract
    # We query the table we just created in the last step
    raw_data = extract_data("SELECT * FROM titanic_test_data")
    
    # Drop the target column if it exists in the test set so we don't cheat/crash the model
    features_df = raw_data.drop(columns=[cfg.dataset.target], errors='ignore')

    # Step 2: Fetch Model
    # Notice we ask specifically for the 'production' alias!
    wb_model_path = "rezk-ahmed-rezk1-iti/titanic-pipeline/winning-classifier:latest"
    model_file_path = fetch_model(wb_model_path)
    
    # Step 3: Predict
    results_df = predict(features_df, model_file_path)
    
    # Step 4: Load
    load_predictions(results_df, "titanic_forecast_results")

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_batch_job(cfg: DictConfig):
    # Start the Prefect orchestration
    batch_forecasting_flow(cfg)

if __name__ == "__main__":
    run_batch_job()