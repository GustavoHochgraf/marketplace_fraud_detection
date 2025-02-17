import pandas as pd

def split_data_by_time(df: pd.DataFrame, train_weeks: list = [10,11,12,13,14], test_weeks: list = [15,16,17]):
    """
    Splits the dataset into training and testing sets based on time.
    It assumes that the DataFrame contains a 'week' column and a target column 'fraude'.

    Parameters:
        df (pd.DataFrame): The full dataset.
        train_weeks (list): Weeks to use for training.
        test_weeks (list): Weeks to use for testing.

    Returns:
        X_train, X_test, y_train, y_test: The features and target splits.
    """
    train_df = df[df['week'].isin(train_weeks)]
    test_df = df[df['week'].isin(test_weeks)]
    
    # Drop the target column from features
    X_train = train_df.drop(columns=['fraude'])
    y_train = train_df['fraude']
    X_test = test_df.drop(columns=['fraude'])
    y_test = test_df['fraude']
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    from data.load_data import load_data
    df = load_data("../../data/raw/dados.csv")
    X_train, X_test, y_train, y_test = split_data_by_time(df)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
