import pandas as pd
import os
import yaml

from sklearn.model_selection import train_test_split

with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

test_size = params['data_collection']['test_size']

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def splitting_raw_data(df):
    train, test = train_test_split(df, test_size=test_size, random_state=42)

    return (train, test)

def saving_file(data, file_name):
    directory = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, file_name)
    data.to_csv(path, index=False)

def main():
    data = 'water_potability.csv'
    
    print("Loading data")
    df = load_data(data)

    print("Splitting train and test data")
    train, test = splitting_raw_data(df)

    print("Saving the train data as train.csv")
    saving_file(train, 'train.csv')

    print("Saving the test data as test.csv")
    saving_file(test, 'test.csv')


if __name__ == '__main__':
    main()
    print("Finished data collection successfully")