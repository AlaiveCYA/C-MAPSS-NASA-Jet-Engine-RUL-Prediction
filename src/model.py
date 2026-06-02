"""LSTM for RUL regression."""

import torch
import torch.nn as nn


class LSTMRUL(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layers(x)["wyjscie"]

    def forward_layers(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with activations from each LSTM layer and the head."""
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden) — last timestep per layer
        layers: dict[str, torch.Tensor] = {}
        for i in range(self.num_layers):
            layers[f"LSTM_{i + 1}"] = h_n[i]
        h_last = h_n[-1]
        z = self.head[0](h_last)
        layers["FC_1"] = z
        layers["wyjscie"] = self.head[2](z).squeeze(-1)
        return layers
