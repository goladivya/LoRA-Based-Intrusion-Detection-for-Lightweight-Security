import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import load_and_preprocess
import joblib
import numpy as np

# ==========================================
# Configuration
# ==========================================
CSV = r"C:\Users\HP\Desktop\project_ids_\data\combined_data.csv"
LABEL_COL = "attack"
MODEL_OUT = "cnn_model.pt"
SCALER_OUT = "cnn_scaler.pkl"
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3

# ==========================================
# Model Definition
# ==========================================
class Simple1DCNN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ==========================================
# Training Code
# ==========================================
def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(CSV, label_col=LABEL_COL)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    # DataLoaders (⚠️ num_workers=0 for Windows)
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple1DCNN(seq_len=X_train.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on device: {device}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)

        # Evaluation
        model.eval()
        preds, probs = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = model(xb)
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds.extend((p >= 0.5).astype(int).tolist())
                probs.extend(p.tolist())

        auc = roc_auc_score(y_test, probs)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")

    # Save model + scaler
    torch.save(model.state_dict(), MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"\n✅ Model saved to {MODEL_OUT}")
    print(f"✅ Scaler saved to {SCALER_OUT}")

# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    main()
