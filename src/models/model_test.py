import dagshub
import mlflow
import os

from mlflow.tracking import MlflowClient

dags_hub_token = os.getenv('DAGS_HUB_SECRET')

if not dags_hub_token:
    raise EnvironmentError("Dagshub token is not available")

# Authenticate DagsHub non-interactively
dagshub.auth.add_app_token(dags_hub_token)

os.environ['MLFLOW_TRACKING_USERNAME'] = dags_hub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dags_hub_token

model_name = "challenger"

mlflow.set_tracking_uri('https://dagshub.com/Ashbipbinu/Water_Potability_MLFLOW.mlflow')