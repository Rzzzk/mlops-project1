import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_pipeline(classifier):
    """
    Creates a full Scikit-Learn pipeline including preprocessing.
    This ensures that preprocessing steps are saved with the model.
    """
    # Define features
    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = ["Embarked", "Sex", "Pclass"]

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

    # Create the full pipeline
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def run_training():
    # 1. Data Loading
    train_df = pd.read_csv("data/raw/train.csv")

    # Feature selection
    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Define models to compare
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_acc = 0
    best_model_pipeline = None
    best_model_name = ""

    # 3. Automated Training & Evaluation
    for name, model in models.items():
        pipeline = create_pipeline(model)
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
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        joblib.dump(best_model_pipeline, model_path)
        print(f"\nSaved Best Model: {best_model_name} to {model_path}")


if __name__ == "__main__":
    run_training()
