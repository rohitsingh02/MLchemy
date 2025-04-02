import pandas as pd
from mlchemy.model import LightGBMPredictor
from mlchemy.preprocessing import Preprocessor
from mlchemy.validation import get_kfold

# Sample dataset
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": ["A", "B", "A", "B", "A"],
    "target": [0, 1, 0, 1, 0]
})

# Preprocessing
prep = Preprocessor()
df = prep.fit_transform(df)
X, y = df.drop(columns=["target"]), df["target"]

# Training
model = LightGBMPredictor(task="classification")
model.fit(X, y)

# Prediction
preds = model.predict(X)
print("Predictions:", preds)
