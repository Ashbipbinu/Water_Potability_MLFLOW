import json
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import dagshub
import mlflow

dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability_MLFLOW', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/Ashbipbinu/Water_Potability_MLFLOW.mlflow')

new_Experiment = "Final Model"

if mlflow.get_experiment_by_name(new_Experiment) == None:
    experiment_id = mlflow.create_experiment(name = new_Experiment)

mlflow.set_experiment(new_Experiment)

def load_data(file_path):
    return pd.read_csv(file_path)

def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    return model

def splitting_Data_XY(data):
    X_test = data.drop(columns=['Potability'])
    y_test = data['Potability']

    return (X_test, y_test)

def make_prdiction(model: RandomForestClassifier, X_data):
    y_pred = model.predict(X_data)

    return y_pred

def evaluation(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision_score": precision,
        "recall_score": recall
    }

def save_metrics(metrics):
    with open('metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)
    

def main():
    with mlflow.start_run(run_name="DVC-MLFLOW") as run:
        print("Loading the test data")
        file_path = os.path.join(os.getcwd(), "data", "processed", "test_processed_mean.csv")
        test_data = load_data(file_path)

        print("Splitting the data")
        X_test, y_test = splitting_Data_XY(test_data)

        print("Loading the model")
        model = load_model()

        print("Making predictions from the model")
        y_pred = make_prdiction(model, X_test)

        print("Evaluating the model")
        metrics = evaluation(y_test, y_pred)

        accuracy = metrics['accuracy']
        f1_score_ = metrics['f1_score']
        precision_score_ = metrics['precision_score']
        recall_score_ = metrics['recall_score']

        mlflow.log_metric("acc", accuracy)
        mlflow.log_metric("f1_score", f1_score_)
        mlflow.log_metric("precision_score", precision_score_)
        mlflow.log_metric("recall_score", recall_score_)

        print("Save the metrics")
        save_metrics(metrics)

        plt.figure(figsize=(5,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion matrix for tuned Random Forest")
        
        path_reports = os.path.join(os.getcwd(), "reports")
        os.makedirs(path_reports, exist_ok=True)
        filename = f"confusion_metrix_Tuned_RandomForest.png"

        # 3. Create the full, complete file path
        full_path = os.path.join(path_reports, filename)

        # 4. Save the figure using the full path
        plt.savefig(full_path)

        mlflow.log_artifact(full_path)
        mlflow.log_artifact(__file__)

        with open("params.yaml", 'r') as file:
            params = yaml.safe_load(file)
            mlflow.log_params(params)

        model_path = os.path.join(os.getcwd(), "models")
        os.makedirs(model_path, exist_ok=True)

        model_file_path = os.path.join(model_path, "Tuned_RandomForest.pkl")
        with open(f"{model_file_path}", 'wb') as file:
            pickle.dump(model, file)

        run_info = {"run_id": run.info.run_id, "model_name": model_file_path}
        
        # 1. Define the filename for the run info
        run_info_filename = "run_details.json"
        
        # 2. Construct the full file path (Directory + Filename)
        run_info_full_path = os.path.join(path_reports, run_info_filename)

        # 3. Open the file path for writing
        with open(run_info_full_path, 'w') as file:
            json.dump(run_info, file, indent=4)
        
        mlflow.log

if __name__ == '__main__':
    main()
    print("Finished the model evaluation")

