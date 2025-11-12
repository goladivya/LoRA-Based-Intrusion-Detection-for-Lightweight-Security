import torch
import joblib
import numpy as np
from cnn_train import Simple1DCNN

# ==========================================
# Configuration
# ==========================================
NUM_FEATURES = 45   # <-- Set this to X_train.shape[1] printed in cnn_train
MODEL_FILE = "cnn_model.pt"
SCALER_FILE = "cnn_scaler.pkl"

# ==========================================
# Load Model and Scaler
# ==========================================
scaler = joblib.load(SCALER_FILE)
model = Simple1DCNN(seq_len=NUM_FEATURES)
model.load_state_dict(torch.load(MODEL_FILE, map_location='cpu'))
model.eval()

# ==========================================
# Single Sample Prediction
# ==========================================
def predict_vector(vec):
    """
    vec: 1D numpy array of raw feature values
    returns: (probability, predicted_class)
    """
    x = scaler.transform(vec.reshape(1, -1)).astype(np.float32)
    x_t = torch.tensor(x).unsqueeze(1)  # (1,1,L)
    with torch.no_grad():
        out = model(x_t)
        p = torch.softmax(out, dim=1)[0, 1].item()
    return p, int(p >= 0.5)

# Example usage
if __name__ == "__main__":
    sample = np.random.rand(NUM_FEATURES)  # dummy sample, replace with real feature vector
    prob, pred = predict_vector(sample)
    print(f"Predicted probability: {prob:.4f} | Predicted class: {pred}")

    # Export to TorchScript for deployment
    example_input = torch.randn(1, 1, NUM_FEATURES)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("cnn_model_traced.pt")
    print("✅ TorchScript model saved as cnn_model_traced.pt")
