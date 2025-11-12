# evaluate_models.py
import joblib
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from cnn_train import Simple1DCNN

print("\n=== Intrusion Detection - Model Evaluation ===\n")

# ==========================================================
# Load Test Data
# ==========================================================
print("[1] Loading test data...")
data = joblib.load("test_data.pkl")   # Must contain {'X_test', 'y_test'}
X_test = data['X_test']
y_test = data['y_test']
print(f"✅ Loaded test set: {X_test.shape[0]} samples, {X_test.shape[1]} features\n")

# ==========================================================
# Evaluate LightGBM
# ==========================================================
print("[2] Evaluating LightGBM...")
lgb_model = lgb.Booster(model_file="lgb_model.txt")
lgb_scaler = joblib.load("lgb_scaler.pkl")
X_lgb = lgb_scaler.transform(X_test)
y_pred_lgb_prob = lgb_model.predict(X_lgb)
y_pred_lgb = (y_pred_lgb_prob >= 0.5).astype(int)

# Metrics
acc_lgb = accuracy_score(y_test, y_pred_lgb)
prec_lgb = precision_score(y_test, y_pred_lgb)
rec_lgb = recall_score(y_test, y_pred_lgb)
f1_lgb = f1_score(y_test, y_pred_lgb)
auc_lgb = roc_auc_score(y_test, y_pred_lgb_prob)

print(f"✅ LightGBM -> Acc: {acc_lgb:.4f}, Prec: {prec_lgb:.4f}, Rec: {rec_lgb:.4f}, F1: {f1_lgb:.4f}, AUC: {auc_lgb:.4f}\n")

# ==========================================================
# Evaluate Random Forest
# ==========================================================
print("[3] Evaluating Random Forest...")
rf_model = joblib.load("rf_model.joblib")
rf_scaler = joblib.load("rf_scaler.pkl")
X_rf = rf_scaler.transform(X_test)
y_pred_rf_prob = rf_model.predict_proba(X_rf)[:, 1]
y_pred_rf = (y_pred_rf_prob >= 0.5).astype(int)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_rf_prob)

print(f"✅ RandomForest -> Acc: {acc_rf:.4f}, Prec: {prec_rf:.4f}, Rec: {rec_rf:.4f}, F1: {f1_rf:.4f}, AUC: {auc_rf:.4f}\n")

# ==========================================================
# Evaluate CNN
# ==========================================================
print("[4] Evaluating CNN...")
NUM_FEATURES = X_test.shape[1]
cnn_model = Simple1DCNN(seq_len=NUM_FEATURES)
cnn_model.load_state_dict(torch.load("cnn_model.pt", map_location='cpu'))
cnn_model.eval()
cnn_scaler = joblib.load("cnn_scaler.pkl")

X_cnn = cnn_scaler.transform(X_test).astype(np.float32)
X_cnn_t = torch.tensor(X_cnn).unsqueeze(1)  # (N,1,L)
with torch.no_grad():
    probs = torch.softmax(cnn_model(X_cnn_t), dim=1)[:, 1].numpy()
y_pred_cnn = (probs >= 0.5).astype(int)

acc_cnn = accuracy_score(y_test, y_pred_cnn)
prec_cnn = precision_score(y_test, y_pred_cnn)
rec_cnn = recall_score(y_test, y_pred_cnn)
f1_cnn = f1_score(y_test, y_pred_cnn)
auc_cnn = roc_auc_score(y_test, probs)

print(f"✅ CNN -> Acc: {acc_cnn:.4f}, Prec: {prec_cnn:.4f}, Rec: {rec_cnn:.4f}, F1: {f1_cnn:.4f}, AUC: {auc_cnn:.4f}\n")

# ==========================================================
# Comparison Summary
# ==========================================================
print("📊 === Model Performance Comparison ===")
print(f"{'Model':<15}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'F1':<10}{'AUC':<10}")
print(f"{'-'*60}")
print(f"{'LightGBM':<15}{acc_lgb:<10.4f}{prec_lgb:<10.4f}{rec_lgb:<10.4f}{f1_lgb:<10.4f}{auc_lgb:<10.4f}")
print(f"{'RandomForest':<15}{acc_rf:<10.4f}{prec_rf:<10.4f}{rec_rf:<10.4f}{f1_rf:<10.4f}{auc_rf:<10.4f}")
print(f"{'CNN':<15}{acc_cnn:<10.4f}{prec_cnn:<10.4f}{rec_cnn:<10.4f}{f1_cnn:<10.4f}{auc_cnn:<10.4f}")

print("\n✅ Evaluation complete!")
