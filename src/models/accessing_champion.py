import dagshub
import mlflow
import pandas as pd
import os

from mlflow.tracking import MlflowClient

dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability_MLFLOW', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Ashbipbinu/Water_Potability_MLFLOW.mlflow')
model_name = "Tuned_RandomForest"

client = MlflowClient()

champion_model_uri = f"models:/{model_name}@champion"
loaded_model = mlflow.pyfunc.load_model(champion_model_uri)

data = pd.read_csv(os.path.join(os.getcwd(), "data", "processed", "train_processed_mean.csv"))

cols = data.columns

input_data = {}
for col in cols:
    value = int(input(f"Enter value for {col}: "))
    input_data[col] = value

loaded_model_pred = loaded_model.predict(pd.DataFrame(input_data, index=0))

print(loaded_model_pred)