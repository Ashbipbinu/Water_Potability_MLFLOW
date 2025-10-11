import mlflow
import os
import pandas as pd
import pickle
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub


dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability', mlflow=True)
new_Experiment = "Hyperparameter_Tuning_Water_Potability_Classification"

if mlflow.get_experiment_by_name(new_Experiment) == None:
    experiment_id = mlflow.create_experiment(name = new_Experiment)

mlflow.set_experiment(new_Experiment)

mlflow.set_tracking_uri("https://dagshub.com/Ashbipbinu/Water_Potability.mlflow") 

def load_data(file_path):
        return pd.read_csv(file_path)

def splitting_data_to_XY(data: pd.DataFrame):
    X = data.drop(columns=['Potability'], axis=1)
    y = data['Potability']

    return (X, y)

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

def make_prdiction(model, X_data):
    y_pred = model.predict(X_data)
    return y_pred


with mlflow.start_run(run_name="Hyperparameter_Tuning_Water_Potability_Classification") as parent:

    rf = RandomForestClassifier(random_state=42)
    
    file_path = os.path.join(os.getcwd(), "data", "processed")
    
    test_data = load_data(os.path.join(file_path, "test_processed_mean.csv"))
    train_data = load_data(os.path.join(file_path, "train_processed_mean.csv"))

    X_train, y_train = splitting_data_to_XY(train_data)
    X_test, y_test = splitting_data_to_XY(test_data)

    rf.fit(X_train, y_train)

    parameter_dist = {
    'n_estimators': [100, 110, 200, 500, 1000],

    # Maximum depth of the tree (integer or None for full depth)
    'max_depth': [10,20,50, 100],
    # You can also include fixed values like None: [10, 20, 50, 100, None]

    # Minimum number of samples required to split an internal node (integer)
    'min_samples_split': [2,4,6,8,10,20],

    # Minimum number of samples required to be at a leaf node (integer)
    'min_samples_leaf': [1,5,10],

    # Number of features to consider for the best split (categorical or fraction/integer)
    'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9, 1.0], 
    # For classification, 'sqrt' (or $\sqrt{p}$) and for regression, 'log2' (or $p/3$) are common defaults.

    # Function to measure the quality of a split (categorical)
    'criterion': ['gini', 'entropy'], # For classification; use ['squared_error', 'absolute_error'] for regression

    # Whether bootstrap samples are used (categorical/boolean)
    'bootstrap': [True, False]

    }

    rs = RandomizedSearchCV(estimator=rf, random_state=42, param_distributions=parameter_dist, cv=5, n_jobs=-1, verbose=2)
    rs.fit(X_train, y_train)
    
    params = rs.best_params_
    estimator = rs.best_estimator_

    y_pred = estimator.predict(X_test)

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
    plt.title("Confusion matrix for tuned Random Forest")
    
    path_figure = os.path.join(os.getcwd(), "reports")
    os.makedirs(path_figure, exist_ok=True)
    filename = f"confusion_metrix_Tuned_RandomForest.png"

    # 3. Create the full, complete file path
    full_path = os.path.join(path_figure, filename)

    # 4. Save the figure using the full path
    plt.savefig(full_path)

    mlflow.log_artifact(full_path)
    mlflow.log_artifact(__file__)

    config = {
    "model_params": rs.best_params_
    }

    with open("params.yaml", 'w') as file:
        yaml.safe_dump(config, file)

    model_path = os.path.join(os.getcwd(), "models")
    os.makedirs(model_path, exist_ok=True)

    model_file_path = os.path.join(model_path, "Tuned_RandomForest.pkl")
    with open(f"{model_file_path}", 'wb') as file:
        pickle.dump(estimator, file)

