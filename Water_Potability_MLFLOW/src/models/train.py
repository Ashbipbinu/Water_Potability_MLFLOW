import os
import pandas  as pd
import pickle
import yaml

from sklearn.ensemble import RandomForestClassifier


training_params = {
    'model_training': {
        'n_estimators': 130,  # Your desired value
        'max_depth': 15       # Your desired value
    }
}

def save_params(params_dict, file_path='params.yaml'):

    with open(file_path, 'w') as file:
        yaml.dump(params_dict, file, default_flow_style=False)


with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

n_estimator = params['model_training']['n_estimators']
max_depth = params['model_training']['max_depth']


def load_data(file_path):
    return pd.read_csv(file_path)

def splitting_data_to_XY(data: pd.DataFrame):
    X = data.drop(columns=['Potability'], axis=1)
    y = data['Potability']

    return (X, y)

def model_training(model, X_data, y_data):
    model.fit(X_data, y_data)

    return model

def saving_model(model):

    directory = os.path.join(os.getcwd(), 'models')
    os.makedirs(directory, exist_ok=True)

    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)


def main():

    save_params(training_params)

    print("Loading the data")
    file_path = os.path.join(os.getcwd(), 'data', 'processed', 'train_processed_mean.csv')
    train_data = load_data(file_path)

    print("Splitting the data into features and target")
    X_train, y_train = splitting_data_to_XY(train_data)

    print(f"Training the model with parameters n_estimator: {n_estimator} and max_depth: {max_depth}")
    rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)

    print("Model training started")   
    model = model_training(rf, X_train, y_train)   

    print("Saving the model") 
    saving_model(model)


if __name__ == "__main__":
    main()

    print("Finished training")