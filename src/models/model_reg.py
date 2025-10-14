import mlflow
import os
import json
import dagshub


from mlflow.tracking import MlflowClient

dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability_MLFLOW', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Ashbipbinu/Water_Potability_MLFLOW.mlflow')

new_Experiment = "Finally Model Registry"

if mlflow.get_experiment_by_name(new_Experiment) == None:
    experiment_id = mlflow.create_experiment(name = new_Experiment)

mlflow.set_experiment(new_Experiment)

client = MlflowClient()


reports_path = os.path.join(os.getcwd(), "reports")
file_path = os.path.join(reports_path, "run_details.json")
with open(file_path, 'r') as file:
    run_details = json.load(file)

run_id = run_details["run_id"]
model_path = os.path.join(os.getcwd(), "models", "Tuned_RandomForest.pkl")
model_name =  run_details["model_name"]


model_uri = f'run://{run_id}/{model_path}'

reg = mlflow.register_model(model_uri, model_name)

new_stage = "Production"

alias_name = "champion"
client.set_registered_model_alias(
    name=model_name,
    alias=alias_name,
    version=reg.version
)

print(f"Model '{model_name}' version {reg.version} has been championed to the 'Production' stage.")

print("Model Registration Completed Succesfully")