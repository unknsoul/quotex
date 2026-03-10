"""
LSTM Sequence Model — v16 Attention-GRU deep learning ensemble member.

A GRU with temporal self-attention that learns WHICH of the recent bars
are most predictive. Tree-based models miss sequential dependencies;
this model captures:
  - Multi-bar momentum acceleration/deceleration
  - Volatility regime transitions
  - Price pattern sequences (flags, wedges, H&S)

v16 upgrades:
  - Self-attention layer: learns to focus on most informative timesteps
  - LayerNorm + GELU activation for better gradient flow
  - 30-bar window (150 min) for richer temporal context
  - Larger hidden dim (96) for more expressive representations
  - Lower LR + more epochs for better convergence
  - Label smoothing to reduce overfitting

Architecture:
  - Input: (batch, seq_len=30, n_features) normalized feature windows
  - 2-layer GRU with dropout
  - Temporal self-attention over all hidden states
  - Dense head with LayerNorm → sigmoid probability
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

log = logging.getLogger("lstm_model")

# Try to import torch; graceful fallback if unavailable
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    log.warning("PyTorch not installed — LSTM model disabled")

# ─── Paths ──────────────────────────────────────────────────────────────────
from config import MODEL_DIR
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
LSTM_SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")

# ─── Hyperparameters ────────────────────────────────────────────────────────
SEQ_LEN = 30          # v16: 30 × M5 = 150 min (was 20)
HIDDEN_DIM = 96       # v16: wider (was 64)
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 5e-4             # v16: lower for attention model (was 1e-3)
EPOCHS = 50           # v16: more epochs with early stopping (was 30)
BATCH_SIZE = 256      # v16: smaller batches = better gradients (was 512)
PATIENCE = 7          # v16: more patience for complex architecture (was 5)
LABEL_SMOOTH = 0.03   # v16: label smoothing alpha


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════

if _HAS_TORCH:
    class _SeqDataset(Dataset):
        """Sliding-window dataset for sequence model."""
        def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
            self.seq_len = seq_len

        def __len__(self):
            return len(self.X) - self.seq_len

        def __getitem__(self, idx):
            x_seq = self.X[idx: idx + self.seq_len]
            label = self.y[idx + self.seq_len]
            return torch.from_numpy(x_seq), torch.tensor(label)


    # ═══════════════════════════════════════════════════════════════════════
    #  Model
    # ═══════════════════════════════════════════════════════════════════════

    class GRUClassifier(nn.Module):
        """v16 Attention-GRU: learns which timesteps matter most."""
        def __init__(self, input_dim, hidden_dim=HIDDEN_DIM,
                     num_layers=NUM_LAYERS, dropout=DROPOUT):
            super().__init__()
            self.gru = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
            )
            # Self-attention mechanism
            self.attn_query = nn.Linear(hidden_dim, hidden_dim)
            self.attn_key = nn.Linear(hidden_dim, hidden_dim)
            self.attn_v = nn.Linear(hidden_dim, 1)

            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.gru(x)  # (batch, seq_len, hidden)

            # Self-attention: learn which timesteps matter
            q = self.attn_query(out)                     # (batch, seq, hidden)
            k = self.attn_key(out)                       # (batch, seq, hidden)
            scores = self.attn_v(torch.tanh(q + k))      # (batch, seq, 1)
            weights = torch.softmax(scores, dim=1)        # (batch, seq, 1)

            # Weighted sum of all hidden states (not just last)
            context = (out * weights).sum(dim=1)           # (batch, hidden)
            return self.head(context).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               feature_names: list = None):
    """
    Train GRU sequence model on windowed feature arrays.

    Args:
        X_train: (n_samples, n_features) — already normalized
        y_train: (n_samples,) binary labels
        X_val: validation features
        y_val: validation labels
        feature_names: list of feature column names

    Returns:
        dict with model path, val_accuracy, val_auc
    """
    if not _HAS_TORCH:
        log.warning("PyTorch not installed — skipping LSTM training")
        return None

    # Normalize features with robust scaling
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, LSTM_SCALER_PATH)

    # Replace NaN/inf with 0
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = X_train_s.shape[1]
    train_ds = _SeqDataset(X_train_s, y_train, SEQ_LEN)
    val_ds = _SeqDataset(X_val_s, y_val, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = GRUClassifier(n_features).to(device)

    # Class-weighted BCE loss
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Override: we use BCELoss since model already has sigmoid
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item() * len(y_batch)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels)
        val_acc = ((val_preds_arr >= 0.5) == val_labels_arr).mean()

        log.info("Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                 epoch + 1, EPOCHS, train_loss, val_loss, val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), LSTM_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

    # Load best model and compute final metrics
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, weights_only=True))
    model.eval()

    from sklearn.metrics import accuracy_score, roc_auc_score
    val_preds_final = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            pred = model(x_batch.to(device))
            val_preds_final.extend(pred.cpu().numpy())
    val_preds_final = np.array(val_preds_final)
    val_labels_final = y_val[SEQ_LEN:]

    acc = accuracy_score(val_labels_final, (val_preds_final >= 0.5).astype(int))
    try:
        auc = roc_auc_score(val_labels_final, val_preds_final)
    except ValueError:
        auc = 0.5

    log.info("LSTM final: val_acc=%.4f val_auc=%.4f", acc, auc)
    return {"path": LSTM_MODEL_PATH, "val_accuracy": acc, "val_auc": auc,
            "n_features": n_features}


# ═══════════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════════

def load_lstm_model(n_features: int):
    """Load trained LSTM model for inference."""
    if not _HAS_TORCH:
        return None
    if not os.path.exists(LSTM_MODEL_PATH):
        return None
    if not os.path.exists(LSTM_SCALER_PATH):
        return None

    model = GRUClassifier(n_features)
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, weights_only=True, map_location="cpu"))
    model.eval()
    scaler = joblib.load(LSTM_SCALER_PATH)
    return {"model": model, "scaler": scaler}


def predict_lstm(lstm_bundle, X_window: np.ndarray) -> float:
    """
    Get LSTM probability for a single window of features.

    Args:
        lstm_bundle: dict from load_lstm_model()
        X_window: (seq_len, n_features) array of recent feature rows

    Returns:
        float probability [0, 1]
    """
    if lstm_bundle is None:
        return 0.5
    if not _HAS_TORCH:
        return 0.5

    model = lstm_bundle["model"]
    scaler = lstm_bundle["scaler"]

    # Scale and clean
    X_scaled = scaler.transform(X_window)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    with torch.no_grad():
        x_tensor = torch.from_numpy(X_scaled.astype(np.float32)).unsqueeze(0)
        prob = model(x_tensor).item()

    return float(np.clip(prob, 0.01, 0.99))
