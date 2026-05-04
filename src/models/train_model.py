import os
import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
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
    print("Loading processed datasets...")
    train_df = pd.read_csv(cfg.dataset.processed_train_path)
    val_df = pd.read_csv(cfg.dataset.processed_val_path)

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

        if acc > best_acc:
            best_acc = acc
            best_model_pipeline = pipeline
            best_model_name = name

    if best_model_pipeline:
        model_path = cfg.model.output_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model_pipeline, model_path)
        print(f"\nSaved Best Model: {best_model_name} to {model_path}")

if __name__ == "__main__":
    run_training()