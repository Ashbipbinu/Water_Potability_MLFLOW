import mlflow
import mlflow.sklearn

import yaml
import os
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub


dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability', mlflow=True)
new_Experiment = "Water_Potability_Classification-3"

if mlflow.get_experiment_by_name(new_Experiment) == None:
    experiment_id = mlflow.create_experiment(name = new_Experiment)

mlflow.set_experiment(new_Experiment)

mlflow.set_tracking_uri("https://dagshub.com/Ashbipbinu/Water_Potability.mlflow") 
#mlflow.set_tracking_uri("http://127.0.0.1:5000") 

def load_data(file_path):
        return pd.read_csv(file_path)

def splitting_data_to_XY(data: pd.DataFrame):
    X = data.drop(columns=['Potability'], axis=1)
    y = data['Potability']

    return (X, y)

def model_training(model, X_data, y_data):
    model.fit(X_data, y_data)
    return model


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

models = {
    "Logistic Regression" : LogisticRegression(),
    "Random Forest" : RandomForestClassifier(),
    "Support Vector Machine" : SVC(),
    "Decision Tree" :  DecisionTreeClassifier(),
    "K Nearest Neighbour": KNeighborsClassifier(),
    "XG Boost" :  XGBClassifier()
}

file_path_train = os.path.join(os.getcwd(), "data", "processed", "train_processed_mean.csv")
file_path_test = os.path.join(os.getcwd(), "data", "processed", "test_processed_mean.csv")


train_data = load_data(file_path_train)
test_data = load_data(file_path_test)

X_train, y_train = splitting_data_to_XY(train_data)
X_test, y_test = splitting_data_to_XY(test_data)



with mlflow.start_run(run_name="Water Potability Models Experiments") as parent:

    for model_name, model in models.items():
         
         model_name = f'{model_name.replace(" ", "_")}.pkl'
         with mlflow.start_run(run_name=model_name, nested=True) as child:

               
            model.fit(X_train, y_train)

            model_path = os.path.join(os.getcwd(), "Water_Potability_MLFLOW", "models")
            os.makedirs(model_path, exist_ok=True)

            model_file_path = os.path.join(model_path, f"{model_name}")
            with open(f"{model_file_path}", 'wb') as file:
                pickle.dump(model, file)
            
            y_pred = model.predict(X_test)
            
            metrics = evaluation(y_test, y_pred)



            accuracy = metrics['accuracy']
            f1_score_ = metrics['f1_score']
            precision_score_ = metrics['precision_score']
            recall_score_ = metrics['recall_score']

            mlflow.log_metric("acc", accuracy)
            mlflow.log_metric("f1_score", f1_score_)
            mlflow.log_metric("precision_score", precision_score_)
            mlflow.log_metric("recall_score", recall_score_)

            plt.figure(figsize=(5,5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion matrix for {model_name}")

            path_figure = os.path.join(os.getcwd(), "Water_Potability_MLFLOW", "reports")
            os.makedirs(path_figure, exist_ok=True)
            filename = f"confusion_metrix_{model_name.replace(" ", "_")}.png"

            # 3. Create the full, complete file path
            full_path = os.path.join(path_figure, filename)

            # 4. Save the figure using the full path
            plt.savefig(full_path)

            mlflow.log_artifact(f"confusion_metrix_{model_name.replace(" ", "_")}.png")
            mlflow.log_artifact(__file__)
            mlflow.log_artifact(f"{model_name.replace(" ", "_")}")
            

            mlflow.set_tag("author", "Ashbi")
            mlflow.set_tags({"model" : f"{model_name.replace(" ", "_")}", f"{new_Experiment}" : "Water_Potability_Classification"})
