import dagshub
import mlflow

dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability_MLFLOW', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Ashbipbinu/Water_Potability_MLFLOW.mlflow')


import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)