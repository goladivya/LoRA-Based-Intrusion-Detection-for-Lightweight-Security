# lightgbm_train.py
import lightgbm as lgb
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from preprocess import load_and_preprocess

# ---- config ----

CSV = "c:/Users/HP/Desktop/project_ids_/data/combined_data.csv"
LABEL_COL = "attack"  # Change to match your actual label column name (e.g., "attack", "intrusion", "class", etc.)
MODEL_OUT = "lgb_model.txt"
# ----------------


X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(CSV, label_col=LABEL_COL)

train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_names)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbosity': -1,
    'n_jobs': -1
}
bst = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

bst.save_model(MODEL_OUT)   # save LightGBM native model
joblib.dump(scaler, "lgb_scaler.pkl")

joblib.dump({'X_test': X_test, 'y_test': y_test}, "test_data.pkl")
print("✅ Test data saved as test_data.pkl")

# evaluate
y_pred_prob = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred = (y_pred_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_prob))
print("Confusion:", confusion_matrix(y_test, y_pred))
