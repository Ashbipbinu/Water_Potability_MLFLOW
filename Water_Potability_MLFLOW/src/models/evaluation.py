import json
import pickle
import pandas as pd
import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

    print("Save the metrics")
    save_metrics(metrics)

if __name__ == '__main__':
    main()

    print("Finished the model evaluation")

