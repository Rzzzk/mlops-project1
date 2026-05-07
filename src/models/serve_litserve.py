import litserve as ls
import wandb
import joblib
import pandas as pd
import os

class TitanicLitAPI(ls.LitAPI):
    def setup(self, device):
        print("Starting LitServe Setup...")
        print("Downloading production model from W&B Registry...")
        
        # 1. Pull the model from W&B exactly like your test script did
        api = wandb.Api()
        artifact_path = "rezk-ahmed-rezk1-iti/titanic-pipeline/winning-classifier:latest"
        artifact = api.artifact(artifact_path)
        model_dir = artifact.download()
        
        # 2. Load the pipeline into memory (happens ONLY ONCE on startup)
        model_path = os.path.join(model_dir, "best_model.pkl") 
        self.model = joblib.load(model_path)
        print("Model loaded successfully. Server is ready!")

    def decode_request(self, request):
        # The request payload will look like: {"passengers": [{...}, {...}]}
        # We extract the list of passengers.
        return request["passengers"]

    def batch(self, inputs):
        # MAGIC HAPPENS HERE:
        # LitServe groups multiple separate HTTP requests together. 
        # So 'inputs' is a list of lists of passengers: [[req1_p1, req1_p2], [req2_p1]]
        
        # We need to remember the size of each request so we can unbatch later
        self.request_sizes = [len(req) for req in inputs]
        
        # Flatten the list of lists into one giant list for pandas
        flat_list = [passenger for req in inputs for passenger in req]
        
        # Convert the flattened list into a single Pandas DataFrame
        return pd.DataFrame(flat_list)

    def predict(self, df):
        # The model processes the massive batched DataFrame all at once
        return self.model.predict(df)

    def unbatch(self, predictions):
        # 'predictions' is a flat numpy array (e.g., [1, 0, 1])
        # We split it back into chunks corresponding to the original HTTP requests
        results = []
        start = 0
        for size in self.request_sizes:
            chunk = predictions[start : start + size]
            # Convert 0/1 to human-readable format
            readable_chunk = ["Survived" if p == 1 else "Did Not Survive" for p in chunk]
            results.append(readable_chunk)
            start += size
        return results

    def encode_response(self, output):
        # Wrap the final specific chunk back into JSON for the user
        return {"predictions": output}

if __name__ == "__main__":
    # FIX 1: Pass batching parameters to the API class, not the LitServer
    api = TitanicLitAPI(max_batch_size=8, batch_timeout=1)
    
    server = ls.LitServer(api)
    
    # FIX 2: Let's use port 8080 to cleanly bypass the stuck process on 8000
    print("Starting server on http://127.0.0.1:8080")
    server.run(port=8080)