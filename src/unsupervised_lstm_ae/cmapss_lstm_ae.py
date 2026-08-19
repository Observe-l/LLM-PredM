"""Conditional LSTM autoencoder used by the C-MAPSS HI experiment.

The implementation follows the useful part of de Pater & Mitici (2023):
operating conditions are supplied to both recurrent parts, while only sensor
measurements are reconstructed.  A Luong-style attention layer combines the
decoder state with all encoder states before reconstructing each sensor.
"""

from __future__ import annotations

import torch
from torch import nn


class ConditionalLSTMAutoencoder(nn.Module):
    """LSTM sequence autoencoder conditioned on operating conditions."""

    def __init__(
        self,
        sensor_dim: int,
        operating_condition_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if sensor_dim < 1 or operating_condition_dim < 1:
            raise ValueError("sensor_dim and operating_condition_dim must be positive")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        input_dim = sensor_dim + operating_condition_dim
        self.sensor_dim = sensor_dim
        self.operating_condition_dim = operating_condition_dim
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.attention_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, sensor_dim)

    def forward(
        self,
        sensors: torch.Tensor,
        operating_conditions: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a batch of sensor sequences.

        Teacher forcing is used for the decoder input: at time t the decoder
        receives the sensor observation from t-1 and the operating condition
        at t.  This makes the reconstruction error a useful cycle-local
        anomaly signal at inference time while keeping training stable.
        """

        if sensors.ndim != 3 or operating_conditions.ndim != 3:
            raise ValueError("sensors and operating_conditions must be [batch, time, feature]")
        if sensors.shape[:2] != operating_conditions.shape[:2]:
            raise ValueError("sensors and operating_conditions must share batch/time dimensions")

        encoder_input = torch.cat([sensors, operating_conditions], dim=-1)
        encoder_states, (hidden, cell) = self.encoder(encoder_input)

        previous_sensors = torch.zeros_like(sensors)
        previous_sensors[:, 1:] = sensors[:, :-1]
        decoder_input = torch.cat([previous_sensors, operating_conditions], dim=-1)
        decoder_states, _ = self.decoder(decoder_input, (hidden, cell))

        # Luong-style dot-product attention over the encoder time dimension.
        query = self.attention_query(decoder_states)
        value = self.attention_value(encoder_states)
        attention_scores = torch.bmm(query, value.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_weights, encoder_states)

        fused = torch.tanh(self.attention_fusion(torch.cat([decoder_states, context], dim=-1)))
        return self.output(fused)
