# rf_infer.py
import joblib
import numpy as np
rf = joblib.load("rf_model.joblib")
scaler = joblib.load("rf_scaler.pkl")

def rf_predict_vector(vec):
    x = scaler.transform(vec.reshape(1, -1))
    p = rf.predict_proba(x)[0,1]
    return p, int(p>=0.5)
