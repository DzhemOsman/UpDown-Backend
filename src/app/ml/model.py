import torch
import torch.nn as nn


class MarketLSTM(nn.Module):
    """
    Ein Deep Learning Modell für multivariate Finanzzeitreihen.
    Nimmt 3D-Tensoren (Batch, Sequence, Features) und gibt einen Logit aus.
    """

    def __init__(
        self,
        input_size: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_rate: int | float = 0.2,
    ):
        super(MarketLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Das Herzstück: Das LSTM-Netzwerk
        # batch_first=True ist wichtig, da unsere Daten (Batch, Seq, Feature) sind
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Regularisierungsschicht gegen Overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # Letzte Schicht: Mappt den Hidden State auf eine einzige Zahl (Logit)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch_Size, Sequence_Length, Features)

        # Initialisiere Hidden State und Cell State implizit mit Nullen
        # (PyTorch macht das automatisch, wenn (h0, c0) nicht übergeben wird)
        lstm_out, _ = self.lstm(x)

        # lstm_out shape: (Batch_Size, Sequence_Length, Hidden_Size)
        # Wir wollen nur die Vorhersage nach dem ALLERLETZTEN Tag der Sequenz:
        last_time_step_out = lstm_out[:, -1, :]

        # last_time_step_out shape: (Batch_Size, Hidden_Size)
        out = self.dropout(last_time_step_out)

        # Logit berechnen (Kein Sigmoid hier, das macht BCEWithLogitsLoss!)
        out = self.fc(out)

        return out
