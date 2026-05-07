import duckdb
import hydra
import pandas as pd
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def upload_test_data(cfg: DictConfig):
    print("Connecting to MotherDuck...")
    # The 'md:' prefix tells duckdb to connect to the MotherDuck cloud
    # It automatically looks for the motherduck_token environment variable
    # con = duckdb.connect("md:titanic_db")
    con = duckdb.connect("md:") # Connect to the default account space
    con.execute("CREATE DATABASE IF NOT EXISTS titanic_db") # Create it
    con.execute("USE titanic_db") # Switch to it

    print("Loading test data using Hydra config...")
    # Assuming your Hydra config has a path for the test data. 
    # If it's named differently, just update this path to match your yaml!
    test_data_path = cfg.dataset.processed_val_path # Or processed_test_path if you have one
    test_df = pd.read_csv(test_data_path)

    print(f"Uploading {len(test_df)} rows to MotherDuck...")
    # DuckDB can read the 'test_df' pandas dataframe directly from the local python memory
    con.execute("CREATE TABLE IF NOT EXISTS titanic_test_data AS SELECT * FROM test_df")
    
    print("Upload complete! Verify it in the MotherDuck UI.")
    con.close()

if __name__ == "__main__":
    upload_test_data()