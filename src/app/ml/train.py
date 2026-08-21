import logging
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

# --- EIGENE IMPORTE ---
from app.ml.dataset import MarketDataset, build_tensor_dataset
from app.ml.model import MarketLSTM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- HYPERPARAMETER & KONFIGURATION ---
BATCH_SIZE = 64  # Reduziert für feingranulareres Lernen
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_lstm_market.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")  # Dateipfad für den Scaler


def get_device() -> torch.device:
    """Findet die beste verfügbare Hardware zur Beschleunigung."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model():
    """Trainiert das PyTorch LSTM Multi-Asset-Modell."""
    device = get_device()
    logger.info(f"Starte PyTorch Training. Genutzte Hardware: {device}")

    # ==========================
    # 1. DATEN LADEN
    # ==========================
    logger.info("Lade 3D-Tensoren aus der InfluxDB (dies kann kurz dauern)...")
    X_train_t, y_train_t, X_test_t, y_test_t, scaler = build_tensor_dataset()

    logger.info(f"Trainings-Batches: ~{len(X_train_t) // BATCH_SIZE}")

    # 2. PyTorch Datasets initialisieren
    train_dataset = MarketDataset(X_train_t, y_train_t)
    test_dataset = MarketDataset(X_test_t, y_test_t)

    # 3. DataLoaders konfigurieren
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================
    # 4. MODELL & LOSS SETUP
    # ==========================
    logger.info("Berechne Klassen-Gewichte (Class Weights)...")
    pos_weight = MarketDataset.compute_pos_weight(y_train_t).to(device)

    logger.info("Initialisiere MarketLSTM Modell...")
    model = MarketLSTM(
        input_size=X_train_t.shape[-1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================
    # 5. DER TRAININGSLOOP
    # ==========================
    logger.info("Beginne Epochen-Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        # Tracking für die Trainings-Accuracy
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward Pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            # Backward Pass
            loss.backward()

            # --- WICHTIGER FIX: GRADIENT CLIPPING ---
            # Verhindert "explodierende Gradienten" und stabilisiert das LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy-Berechnung für den aktuellen Batch
            with torch.no_grad():
                probs = torch.sigmoid(predictions)
                preds = (probs >= 0.5).float()
                correct_train += (preds == batch_y).sum().item()
                total_train += batch_y.size(0)

        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Logging von Loss UND Trainings-Accuracy
        logger.info(
            f"Epoche [{epoch:02d}/{EPOCHS}] - Loss: {avg_loss:.4f} "
            f"| Train Acc: {train_acc:.2%}"
        )

    # ==========================
    # 6. EVALUIERUNG
    # ==========================
    logger.info("Starte Evaluierung auf unbekannten Testdaten...")
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    acc = accuracy_score(all_targets, all_preds)

    logger.info("===============================================")
    logger.info("=== EVALUIERUNG (PYTORCH LSTM MODELL) ===")
    logger.info("===============================================")
    logger.info(f"Klassifikations-Genauigkeit (Accuracy): {acc:.2%}")
    print(classification_report(all_targets, all_preds, zero_division=0))

    # ==========================
    # 7. MODELLE SPEICHERN
    # ==========================
    os.makedirs(MODEL_DIR, exist_ok=True)

    # LSTM-Gewichte sichern
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Erfolg! Das LSTM-Modell wurde unter '{MODEL_PATH}' gesichert.")

    # Scaler via Pickle sichern
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler erfolgreich unter '{SCALER_PATH}' gesichert.")


if __name__ == "__main__":
    train_model()
