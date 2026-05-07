import wandb
import joblib
import pandas as pd
import os

def load_production_model_and_predict():
    print("Fetching production model from W&B...")
    # 1. Initialize the W&B API
    api = wandb.Api()

    # 2. Point to the Model Registry using the 'latest' alias
    # Replace 'Titanic-Classifier' if you named it something else in the UI
    artifact_path = "rezk-ahmed-rezk1-iti/titanic-pipeline/winning-classifier:latest"
    
    # 3. Download the artifact
    # W&B is smart: if you already downloaded this version locally, 
    # it uses the cached version instead of re-downloading.
    artifact = api.artifact(artifact_path)
    model_dir = artifact.download()
    
    print(f"Model downloaded to: {model_dir}")

    # 4. Load the scikit-learn pipeline using joblib
    # The filename must match whatever you called it in your training script
    model_path = os.path.join(model_dir, "best_model.pkl") 
    model = joblib.load(model_path)
    
    print("Model loaded successfully. Preparing prediction...")

    # 5. Create some dummy data for a new passenger
    # Your model expects a Pandas DataFrame because of the ColumnTransformer
    new_passenger = pd.DataFrame([{
        "Pclass": 3,
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }])

    # 6. Make the prediction
    # Because you saved the entire Pipeline (preprocessor + classifier), 
    # you don't need to manually scale or encode the data here!
    prediction = model.predict(new_passenger)
    
    # Map the binary output to a human-readable result
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    
    print(f"\n--- Prediction Result ---")
    print(f"Passenger Details:\n{new_passenger.iloc[0].to_string()}")
    print(f"Outcome: {result}")

if __name__ == "__main__":
    load_production_model_and_predict()