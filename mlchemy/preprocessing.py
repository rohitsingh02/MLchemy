import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self):
        self.encoders = {}

    def fit_transform(self, df):
        df = df.copy()
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        df.fillna(df.median(), inplace=True)
        return df

    def transform(self, df):
        df = df.copy()
        for col, le in self.encoders.items():
            df[col] = le.transform(df[col])
        df.fillna(df.median(), inplace=True)
        return df
