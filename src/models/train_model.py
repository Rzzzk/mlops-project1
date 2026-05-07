import os
import hydra
import joblib
import pandas as pd
import wandb  ### W&B: Import the library
from omegaconf import DictConfig, OmegaConf  ### W&B: OmegaConf is needed to parse Hydra configs
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def create_pipeline(classifier, cfg: DictConfig):
    numeric_features = list(cfg.dataset.numeric_features)
    categorical_features = list(cfg.dataset.categorical_features)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_training(cfg: DictConfig) -> None:
    ### W&B: 1. Convert Hydra config to a standard Python dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    ### W&B: 2. Initialize the run. 
    # W&B will automatically grab your system metrics (CPU/Memory usage) in the background.
    run = wandb.init(
        project="titanic-pipeline", # Group all runs under this project
        name="multi-model-evaluation", 
        config=config_dict,         # Log all Hydra parameters
        job_type="train"
    )

    print("Loading processed datasets...")
    train_df = pd.read_csv(cfg.dataset.processed_train_path)
    val_df = pd.read_csv(cfg.dataset.processed_val_path)

    ### W&B: 3. Log the processed datasets as a lineage artifact
    # This proves exactly which data files resulted in the final model.
    dataset_artifact = wandb.Artifact(
        name="processed-dataset", 
        type="dataset",
        description="Preprocessed train and validation sets"
    )
    dataset_artifact.add_file(cfg.dataset.processed_train_path)
    dataset_artifact.add_file(cfg.dataset.processed_val_path)
    run.log_artifact(dataset_artifact)

    # Feature selection
    X_train = train_df.drop(cfg.dataset.target, axis=1)
    y_train = train_df[cfg.dataset.target]
    
    X_val = val_df.drop(cfg.dataset.target, axis=1)
    y_val = val_df[cfg.dataset.target]

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=cfg.model.logistic_regression.max_iter
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=cfg.model.random_forest.n_estimators,
            random_state=cfg.model.random_forest.random_state,
        ),
    }

    best_acc = 0
    best_model_pipeline = None
    best_model_name = ""

    print("Training models...")
    for name, model in models.items():
        pipeline = create_pipeline(model, cfg)
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"Model: {name} | Validation Accuracy: {acc:.4f}")

        ### W&B: 4. Log the accuracy for each model type dynamically
        wandb.log({f"{name}_accuracy": acc})

        if acc > best_acc:
            best_acc = acc
            best_model_pipeline = pipeline
            best_model_name = name

    ### W&B: 5. Log a final summary metric for the overall run
    wandb.run.summary["best_model_name"] = best_model_name
    wandb.run.summary["best_val_accuracy"] = best_acc

    if best_model_pipeline:
        model_path = cfg.model.output_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model_pipeline, model_path)
        print(f"\nSaved Best Model: {best_model_name} to {model_path}")

        ### W&B: 6. Save the winning model to the Model Registry
        # We include metadata so you can see the winning accuracy without downloading the file.
        model_artifact = wandb.Artifact(
            name="winning-classifier", 
            type="model",
            metadata={
                "model_type": best_model_name,
                "validation_accuracy": best_acc
            }
        )
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

    ### W&B: 7. Safely close the run to sync all final files to the cloud
    wandb.finish()

if __name__ == "__main__":
    run_training()