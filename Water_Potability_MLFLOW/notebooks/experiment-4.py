# Importing necessary libraries
import mlflow
import pandas as pd
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Ashbipbinu', repo_name='Water_Potability', mlflow=True)


# Instantiating each models as a dictionary
models = {
    "Logistic Regression" : LogisticRegression(),
    "Random Forest" : RandomForestClassifier(),
    "Support Vector Machine" : SVC(),
    "Decision Tree" :  DecisionTreeClassifier(),
    "K Nearest Neighbour": KNeighborsClassifier(),
    "XG Boost" :  XGBClassifier()
}

def load_data(file_path):
        return pd.read_csv(file_path)

def splitting_data_to_XY(data: pd.DataFrame):
    X = data.drop(columns=['Potability'], axis=1)
    y = data['Potability']

    return (X, y)


def make_prdiction(model, X_data):
    y_pred = model.predict(X_data)
    return y_pred

def KnImpute(data):
     kn = KNNImputer(n_neighbors=5)

     if data.isna().any().any():
          imputed_arr = kn.fit_transform(data)
          imputed_Data = pd.DataFrame(imputed_arr, columns=data.columns, index=data.index)
          return imputed_Data
     else:
          return data

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


new_Experiment = 'Data with KNN imputer'
is_Experiment_already_exist = mlflow.get_experiment_by_name(new_Experiment)

if is_Experiment_already_exist == None:
    mlflow.set_experiment(new_Experiment)

mlflow.set_tracking_uri("https://dagshub.com/Ashbipbinu/Water_Potability.mlflow") 


# Parent Run
with mlflow.start_run(run_name="Imputer - Water Potability Models Experiments") as parent :

    file_path_train = os.path.join(os.getcwd(), "data", "raw", "train.csv")
    file_path_test = os.path.join(os.getcwd(), "data", "raw", "test.csv")

    data_train = load_data(file_path_train)
    data_test = load_data(file_path_test)

    imputed_train_data = KnImpute(data_train)
    imputed_test_data = KnImpute(data_test)

    # Saving the processed data
    path = os.path.join(os.getcwd(), "data", "raw")

    os.makedirs(path, exist_ok=True)
    train_data_path = os.path.join(path, "train_processed.csv")
    test_data_path = os.path.join(path, "test_processed.csv")


    imputed_test_data.to_csv(train_data_path, index=False)
    imputed_train_data.to_csv(test_data_path, index=False) 
    
    X_train, y_train = splitting_data_to_XY(imputed_train_data)
    X_test, y_test = splitting_data_to_XY(imputed_test_data)

    for model_name, model in models.items():
        model_name = model_name.replace(" ", "_")
        # Child run
        with mlflow.start_run(run_name=f"Model: {model_name}", nested=True) as child:
            model.fit(X_train, y_train)
            
            # Making predictions and evaluating the model
            y_pred = make_prdiction(model, X_test)
            metrics = evaluation(y_test, y_pred)

            # Evaluation results
            accuracy = metrics['accuracy']
            f1_score_ = metrics['f1_score']
            precision_score_ = metrics['precision_score']
            recall_score_ = metrics['recall_score']

            # Logging the metrics
            mlflow.log_metric("acc", accuracy)
            mlflow.log_metric("f1_score", f1_score_)
            mlflow.log_metric("precision_score", precision_score_)
            mlflow.log_metric("recall_score", recall_score_)

            # SAving the plots as artifacts
            plt.figure(figsize=(5,5))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion matrix for {model_name}")

            path_figure = os.path.join(os.getcwd(), "Water_Potability_MLFLOW", "reports")
            os.makedirs(path_figure, exist_ok=True)
            filename = f"Imputer_confusion_metrix_{model_name.replace(" ", "_")}.png"

            # 3. Create the full, complete file path
            full_path = os.path.join(path_figure, filename)

            # 4. Save the figure using the full path
            plt.savefig(full_path)

            model_path = os.path.join(os.getcwd(), "Water_Potability_MLFLOW", "models")
            os.makedirs(model_path, exist_ok=True)

            model_file_path = os.path.join(model_path, f"{model_name}")
            with open(f"{model_file_path}", 'wb') as file:
                pickle.dump(model, file)