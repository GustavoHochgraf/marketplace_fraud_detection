import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame):
    """
    Preprocesses the dataset:
      - Fills missing numeric values with the median.
      - Scales features (excluding the target 'fraude').
    Returns the scaled features and the target series.
    """
    df = df.copy()
    # Fill missing numeric values.
    df.fillna(df.median(numeric_only=True), inplace=True)
    # Separate features and target.
    target = df['fraude']
    features = df.drop(columns=['fraude'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=df.index)
    return features_scaled_df, target

if __name__ == '__main__':
    from load_data import load_data
    df = load_data()
    df_scaled, target = preprocess_data(df)
    print(df_scaled.head())
