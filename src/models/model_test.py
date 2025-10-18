import dagshub
import mlflow
import unittest
import os

from mlflow.tracking import MlflowClient

dags_hub_token = os.getenv('DAGS_HUB_SECRET')

if not dags_hub_token:
    raise EnvironmentError("Dagshub token is not available")

# Authenticate DagsHub non-interactively
dagshub.auth.add_app_token(dags_hub_token)

os.environ['MLFLOW_TRACKING_USERNAME'] = dags_hub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dags_hub_token

model_alias = "challenger"
model_name = "Tuned_RandomForest"

mlflow.set_tracking_uri('https://dagshub.com/Ashbipbinu/Water_Potability_MLFLOW.mlflow')

class TestModelLoading(unittest.TestCase):

    def test_model_challenger(self):
        
        client = MlflowClient()
        version = client.get_model_version_by_alias(model_name, model_alias)

        self.assertGreater(len(version), 0, "No challenger models found")
    
    def test_model_loading(self):
        client = MlflowClient()
        version = client.get_model_version_by_alias(model_name, model_alias)

        if not version:
            self.fail("No model found with the alias challenger")
        
        latest_version = version[0].version
        run_id = version[0].run_id

        logged_model = f"runs:/{run_id}/{model_name}"

        try:

            loaded_model = mlflow.pyfunc.load_model(logged_model)
        except Exception as e:
            self.fail(f"Failed to load model: {e}")
        
        self.assertIsNotNone(loaded_model, "The loaded model is none")
        print(f"Model succesfully loaded from {logged_model}")

if __name__ == "__main__":
    unittest.main()