"""The conditional LSTM-AE architecture used for the N-CMAPSS paper replay."""

from __future__ import annotations

import torch
from torch import nn


class PaperConditionalLSTMAE(nn.Module):
    """LSTM-AE with informative operating conditions and local Luong attention.

    This follows the paper's sequence convention: the encoder sees all
    measurements, while the decoder reconstructs time steps 2..n. During
    training the previous true sensor vector is used (teacher forcing); during
    evaluation the previous reconstructed vector is fed back.
    """

    def __init__(
        self,
        sensor_dim: int = 13,
        operating_condition_dim: int = 4,
        hidden_size: int = 4,
        attention_window: int = 5,
        fully_connected_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.sensor_dim = sensor_dim
        self.operating_condition_dim = operating_condition_dim
        self.hidden_size = hidden_size
        self.attention_window = attention_window

        recurrent_input_dim = sensor_dim + operating_condition_dim
        self.encoder = nn.LSTM(recurrent_input_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(recurrent_input_dim, hidden_size, batch_first=True)

        # Eq. (8): bilinear Luong alignment matrix W_a.
        self.attention_matrix = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.attention_matrix)
        # Eq. (11): h-tilde = tanh(W_h [context, decoder_state]).
        self.attention_fusion = nn.Linear(hidden_size * 2, hidden_size)

        # The paper uses l=3 fully connected layers, with 128 nodes in the
        # first l-1 layers and tanh on those hidden layers.
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size + operating_condition_dim, fully_connected_hidden),
            nn.Tanh(),
            nn.Linear(fully_connected_hidden, fully_connected_hidden),
            nn.Tanh(),
            nn.Linear(fully_connected_hidden, sensor_dim),
        )

    def _local_attention(
        self,
        decoder_states: torch.Tensor,
        encoder_states: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply local Luong attention for a batch of equal-length sequences."""

        # scores[b,t,j] = decoder[b,t]^T W_a encoder[b,j]
        query = torch.matmul(decoder_states, self.attention_matrix)
        scores = torch.bmm(query, encoder_states.transpose(1, 2))
        distance = torch.abs(target_indices[:, None] - torch.arange(
            encoder_states.shape[1], device=encoder_states.device
        )[None, :])
        mask = distance <= self.attention_window
        scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_states)
        return torch.tanh(
            self.attention_fusion(torch.cat([context, decoder_states], dim=-1))
        )

    def _reconstruct_teacher_forced(
        self,
        sensors: torch.Tensor,
        operating_conditions: torch.Tensor,
    ) -> torch.Tensor:
        encoder_input = torch.cat([sensors, operating_conditions], dim=-1)
        encoder_states, (hidden, cell) = self.encoder(encoder_input)

        decoder_input = torch.cat(
            [sensors[:, :-1], operating_conditions[:, 1:]], dim=-1
        )
        decoder_states, _ = self.decoder(decoder_input, (hidden, cell))
        target_indices = torch.arange(1, sensors.shape[1], device=sensors.device)
        augmented = self._local_attention(decoder_states, encoder_states, target_indices)
        head_input = torch.cat([augmented, operating_conditions[:, 1:]], dim=-1)
        return self.reconstruction_head(head_input)

    def _reconstruct_autoregressive(
        self,
        sensors: torch.Tensor,
        operating_conditions: torch.Tensor,
    ) -> torch.Tensor:
        encoder_input = torch.cat([sensors, operating_conditions], dim=-1)
        encoder_states, (hidden, cell) = self.encoder(encoder_input)
        reconstructed: list[torch.Tensor] = []
        previous = sensors[:, 0]
        for target_index in range(1, sensors.shape[1]):
            decoder_input = torch.cat(
                [previous, operating_conditions[:, target_index]], dim=-1
            ).unsqueeze(1)
            decoder_state, (hidden, cell) = self.decoder(
                decoder_input, (hidden, cell)
            )
            index = torch.tensor([target_index], device=sensors.device)
            augmented = self._local_attention(decoder_state, encoder_states, index)
            head_input = torch.cat(
                [augmented, operating_conditions[:, target_index].unsqueeze(1)], dim=-1
            )
            previous = self.reconstruction_head(head_input).squeeze(1)
            reconstructed.append(previous)
        return torch.stack(reconstructed, dim=1)

    def forward(
        self,
        sensors: torch.Tensor,
        operating_conditions: torch.Tensor,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        if sensors.ndim != 3 or operating_conditions.ndim != 3:
            raise ValueError("inputs must have shape [batch, time, feature]")
        if sensors.shape[:2] != operating_conditions.shape[:2]:
            raise ValueError("sensor and operating-condition sequences must align")
        if sensors.shape[1] < 2:
            raise ValueError("a flight must contain at least two time steps")
        if teacher_forcing:
            return self._reconstruct_teacher_forced(sensors, operating_conditions)
        return self._reconstruct_autoregressive(sensors, operating_conditions)
