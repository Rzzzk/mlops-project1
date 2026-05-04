import os
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def prepare_data(cfg: DictConfig) -> None:
    print(f"Loading raw data from {cfg.dataset.train_path}...")
    df = pd.read_csv(cfg.dataset.train_path)

    print("Splitting data into train and validation sets...")
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.dataset.test_size,
        random_state=cfg.dataset.random_state,
    )

    # Ensure the processed directory exists
    os.makedirs(os.path.dirname(cfg.dataset.processed_train_path), exist_ok=True)
    
    # Save the split datasets
    train_df.to_csv(cfg.dataset.processed_train_path, index=False)
    val_df.to_csv(cfg.dataset.processed_val_path, index=False)
    
    print(f"Saved train split to {cfg.dataset.processed_train_path}")
    print(f"Saved val split to {cfg.dataset.processed_val_path}")

if __name__ == "__main__":
    prepare_data()