import numpy as np
import pandas as pd
from mlchemy.model import LightGBMPredictor

print(LightGBMPredictor)

def test_lightgbm():
    X = pd.DataFrame(np.random.rand(100, 5))
    y = np.random.randint(0, 2, 100)

    model = LightGBMPredictor(task="classification")
    model.fit(X, y)
    preds = model.predict(X)
    
    assert len(preds) == len(y), "Prediction length mismatch!"
    print("✅ LightGBMPredictor test passed!")

test_lightgbm()