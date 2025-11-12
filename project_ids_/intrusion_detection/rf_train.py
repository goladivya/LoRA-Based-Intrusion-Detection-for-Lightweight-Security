# rf_train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from preprocess import load_and_preprocess

CSV = "c:/Users/HP/Desktop/project_ids_/data/combined_data.csv"
LABEL_COL = "attack"
MODEL_OUT = "rf_model.joblib"

X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(CSV, label_col=LABEL_COL)

# RandomForest works better with original scale sometimes, but scaled is OK.
rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
joblib.dump(rf, MODEL_OUT)
joblib.dump(scaler, "rf_scaler.pkl")

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("Confusion:", confusion_matrix(y_test, y_pred))
