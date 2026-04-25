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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_pipeline(classifier, cfg: DictConfig):
    """
    Creates a full Scikit-Learn pipeline including preprocessing.
    Feature column names are driven by the Hydra dataset config.
    """
    numeric_features = list(cfg.dataset.numeric_features)
    categorical_features = list(cfg.dataset.categorical_features)

    # Preprocessing for numeric data: Impute missing + Scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical data: Impute missing + One-Hot Encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run_training(cfg: DictConfig) -> None:
    # 1. Data Loading
    train_df = pd.read_csv(cfg.dataset.train_path)

    # Feature selection
    X = train_df.drop(cfg.dataset.target, axis=1)
    y = train_df[cfg.dataset.target]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.dataset.test_size,
        random_state=cfg.dataset.random_state,
    )

    # 2. Define models to compare
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

    # 3. Automated Training & Evaluation
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

    # 4. Model Saving
    if best_model_pipeline:
        model_path = cfg.model.output_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model_pipeline, model_path)
        print(f"\nSaved Best Model: {best_model_name} to {model_path}")


if __name__ == "__main__":
    run_training()
