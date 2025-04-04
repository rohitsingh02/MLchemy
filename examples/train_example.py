import pandas as pd
import numpy as np
from mlchemy.model import LightGBMPredictor
from mlchemy.preprocessing import Preprocessor
from mlchemy.validation import get_kfold
from mlchemy.config import TrainerConfig
from mlchemy.trainer import Trainer

# Sample dataset
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    # "feature3": [1, 2, 4, 5, np.NAN],
    "feature3": [1, 2, 4, 5, np.NAN],
    "feature2": ["A", "B", "A", "B", "A"],
    "feature4": ["A", "B", "A", "B", np.NaN],

    "target": [0, 1, 0, 1, 0]
})


cat_cols = df.select_dtypes(include=['object', 'category']).columns


config = TrainerConfig(
    num_folds=5,
    num_repeats=1,
    target_col="target",
    fillna_num=None,
    fillna_cat="NA",
    cat_features=cat_cols,
    cat_encoding='label' # Literal['label', 'onehot', 'frequency'] = 'label'


)
trainer = Trainer(config=config, data=df)
# trainer.fit()
trainer.fit(train_data=df)






# # Preprocessing
# prep = Preprocessor()
# df = prep.fit_transform(df)
# X, y = df.drop(columns=["target"]), df["target"]

# # Training
# model = LightGBMPredictor(task="classification")
# model.fit(X, y)

# # Prediction
# preds = model.predict(X)
# print("Predictions:", preds)
