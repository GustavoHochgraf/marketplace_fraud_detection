import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.constants import PREDEFINED_J_MAPPING

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms the raw fraud dataset with cleaning and feature engineering.
    """
    def __init__(self, j_mapping: dict = None):
        # Use provided mappings; if not provided, use defaults.
        self.j_mapping_ = j_mapping if j_mapping is not None else PREDEFINED_J_MAPPING

    @staticmethod
    def _remove_accents(text: str) -> str:
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    @staticmethod
    def _count_special_chars(text: str) -> int:
        return len(re.findall(r'[^A-Za-z0-9\s]', text))

    @staticmethod
    def _count_numbers(text: str) -> int:
        return len(re.findall(r'\d+', text))

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.lower()
        text = FraudFeatureEngineer._remove_accents(text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    @staticmethod
    def _trata_o(x):
        if isinstance(x, str):
            if x.lower() == 'n':
                return 0
            elif x.lower() == 'y':
                return 1
        return 2

    @staticmethod
    def gera_k(x):
        if x <= 0.247:
            return 0
        elif x <= 0.496:
            return 1
        elif x <= 0.746:
            return 2
        else:
            return 3

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # DATE FEATURES
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['ymd'] = df['fecha'].dt.strftime('%Y%m%d').astype(int)
        df['week'] = df['fecha'].dt.isocalendar().week.astype(int)
        df['day_of_week'] = df['fecha'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['hour'] = df['fecha'].dt.hour
        bins = [0, 5, 12, 18, 24]
        labels = ['madrugada', 'manha', 'tarde', 'noite']
        df['periodo'] = pd.cut(df['hour'], bins=bins, right=False, labels=labels)
        df['periodo_num'] = df['periodo'].map({'madrugada': 3, 'manha': 0, 'tarde': 1, 'noite': 2}).astype(int)
        df['OOT'] = 'train'
        df.loc[df['week'].isin([15, 16, 17]), 'OOT'] = 'test'

        # VARIABLE A: binarize (example: 1-3 high fraud, 4 low)
        df['a_bin'] = (df['a'] <= 3).astype(int)

        # VARIABLES B, C, D, E, F: fill missings/outliers
        df['b'] = df['b'].fillna(1.1)
        df['c'] = df['c'].fillna(1.1)
        df['d'] = df['d'].fillna(-1)
        df['e'] = df['e'].replace(0, np.nan)
        e_outlier = df['e'].quantile(0.99)
        df['e'] = df['e'].fillna(e_outlier)
        df['f'] = df['f'].replace(0, np.nan)
        f_outlier = df['f'].quantile(0.99)
        df['f'] = df['f'].fillna(f_outlier)

        # VARIABLE G: country grouping
        df['g_agrup_simples'] = df['g'].apply(lambda x: x if x in ['BR', 'AR'] else 'others')

        # VARIABLE I: text processing
        df['i'] = df['i'].astype(str)
        df['i_len'] = df['i'].str.len()
        df['i_special_chars'] = df['i'].apply(self._count_special_chars)
        df['i_special_chars_agrup'] = df['i_special_chars'].apply(lambda x: str(x) if x <= 2 else ">2")
        df['i_num_count'] = df['i'].apply(self._count_numbers)
        df['i_num_count_agrup'] = df['i_num_count'].apply(lambda x: '<=2' if x <= 2 else '>2')
        df['i_cleaned'] = df['i'].apply(self._clean_text)
        df['i_word_original'] = df['i'].str.lower().str.contains('original').astype(int)
        df['i_word_kit'] = df['i'].str.lower().str.contains('kit').astype(int)
        df['i_word_gb'] = df['i'].str.lower().str.contains('gb').astype(int)
        df['i_word_ram'] = df['i'].str.lower().str.contains('ram').astype(int)

        # VARIABLE J: cluster product categories
        df['j_cluster'] = df['j'].map(self.j_mapping_).fillna(-1).astype(int)
        df['j_cluster_agrup'] = df['j_cluster'].apply(lambda x: 1 if x in [3,4,5] else 0)

        # VARIABLE K: discretize using hard-coded function
        df['k_bin'] = df['k'].apply(self.gera_k)

        # VARIABLES L, M, N, P, MONTO: keep as is (n and p are binary)
        # VARIABLE O: custom transformation (N/n -> 0, Y/y -> 1, else 2)
        df['o_transformed'] = df['o'].apply(self._trata_o)

        return df

if __name__ == '__main__':
    df_raw = pd.read_csv('../data/raw/dados.csv')
    engineer = FraudFeatureEngineer()
    df_engineered = engineer.fit_transform(df_raw)
    df_engineered.to_csv('../data/processed/processed_data.csv', index=False)
