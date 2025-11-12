# lgb_infer.py
import lightgbm as lgb
import joblib
import numpy as np

bst = lgb.Booster(model_file="lgb_model.txt")
scaler = joblib.load("lgb_scaler.pkl")

def predict_single(sample_dict, feature_order):
    # sample_dict: mapping feature->value (or provide a numpy array in feature order)
    x = np.array([sample_dict.get(f, 0.0) for f in feature_order], dtype=float).reshape(1, -1)
    x = scaler.transform(x)
    p = bst.predict(x)[0]
    return p, int(p>=0.5)
