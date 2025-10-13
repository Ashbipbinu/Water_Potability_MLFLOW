import pandas as pd
import os
import yaml

from sklearn.impute import KNNImputer


def load_data(filepath):
    return pd.read_csv(filepath)

def check_for_missing_values(data: pd.DataFrame):
    
    data_copy = data.copy()

    for col in data_copy.columns:
        if data_copy[col].isna().any():
            print("Found missing values")
            mean_value = data_copy[col].mean()
            data_copy[col] = data_copy[col].fillna(mean_value)
    
    return data_copy
    
def save_data(data, file_name):
    directory = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, file_name)
    data.to_csv(path, index=False)



def main():
    
    print("Loading training data")
    train_data = load_data(r'data/raw/train.csv')
    
    print("Loading testing data")
    test_data = load_data(r'data/raw/test.csv')

    print("Cheking missing values in the training data")
    train_processed = check_for_missing_values(train_data)

    print("Cheking missing values in the testing data")
    test_processed = check_for_missing_values(test_data)

    print("Saving train data")
    save_data(train_processed, 'train_processed_mean.csv')
    
    print("Saving test data")
    save_data(test_processed, 'test_processed_mean.csv')


if __name__ == '__main__':
    main()
    print("Finished data pre_processing successfully")
