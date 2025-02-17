import pandas as pd

def load_data(filepath: str = 'data/raw/dados.csv') -> pd.DataFrame:
    """
    Loads the raw CSV file into a DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

if __name__ == '__main__':
    df = load_data()
    print(df.head())
