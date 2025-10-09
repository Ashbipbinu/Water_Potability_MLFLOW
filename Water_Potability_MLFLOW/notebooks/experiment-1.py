import mlflow
import mlflow.sklearn

import yaml
import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub

dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability', mlflow=True)
new_Experiment = "Water_Potability_Classification-1"

if mlflow.get_experiment_by_name(new_Experiment) == None:
    experiment_id = mlflow.create_experiment(name = new_Experiment)

mlflow.set_experiment(new_Experiment)

mlflow.set_tracking_uri("https://dagshub.com/Ashbipbinu/Water_Potability.mlflow") 
#mlflow.set_tracking_uri("http://127.0.0.1:5000") 



with mlflow.start_run():


    def load_data(file_path):
        return pd.read_csv(file_path)

    def splitting_data_to_XY(data: pd.DataFrame):
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']

        return (X, y)

    def model_training(model, X_data, y_data):
        model.fit(X_data, y_data)
        return model

    

    file_path = os.path.join(os.getcwd(), "data", "processed", "train_processed_mean.csv")
    load_train_data = load_data(file_path)

    split_data = splitting_data_to_XY(load_train_data)
    X_train, y_train = split_data
    n_estimators = 100
    max_depth = 3

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model = model_training(clf, X_train, y_train)


    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    def load_data(file_path):
        return pd.read_csv(file_path)


    def make_prdiction(model, X_data):
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

    file_path = os.path.join(os.getcwd(), "data", "processed", "test_processed_mean.csv")
    load_test_data = load_data(file_path)
    X_test, y_test = splitting_data_to_XY(load_test_data)
    y_pred = make_prdiction(clf, X_test)

    metrics = evaluation(y_test, y_pred)

    train_df = mlflow.data.from_pandas(load_train_data)
    test_df = mlflow.data.from_pandas(load_test_data)

    accuracy = metrics['accuracy']
    f1_score = metrics['f1_score']
    precision_score = metrics['precision_score']
    recall_score = metrics['recall_score']

    mlflow.log_metric("acc", accuracy)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("precision_score", precision_score)
    mlflow.log_metric("recall_score", recall_score)

    plt.figure(figsize=(5,5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")

    plt.savefig("confusion_metrix.png")

    mlflow.log_artifact("confusion_metrix.png")

    mlflow.log_artifact(__file__)

    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test") 

    mlflow.set_tag("author", "Ashbi")
    mlflow.set_tags({"model" : "GradientBoostingClassifier", "Experiment-1" : "Water_Potability_Classification"})

    mlflow.sklearn.log_model(sk_model=clf, artifact_path="RandomForestClassifier")
