from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


@dataclass
class MaintenanceSFTOutput:
    loss: Optional[torch.Tensor]
    action_loss: Optional[torch.Tensor]
    priority_loss: Optional[torch.Tensor]
    action_logits: torch.Tensor
    priority_logits: torch.Tensor


class NumericFeatureProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_numeric_tokens: int, dropout: float = 0.1):
        super().__init__()
        self.num_numeric_tokens = num_numeric_tokens
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size * num_numeric_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, numeric_values: torch.Tensor) -> torch.Tensor:
        x = self.norm(numeric_values)
        x = self.proj(x)
        x = x.view(numeric_values.size(0), self.num_numeric_tokens, -1)
        return self.out_norm(x)


class MaintenanceSFTModel(nn.Module):
    def __init__(
        self,
        base_model_name_or_path: str,
        numeric_feature_dim: int,
        num_actions: int,
        num_priorities: int,
        num_numeric_tokens: int = 2,
        dropout: float = 0.1,
        load_in_4bit: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        alpha: float = 0.5,
        device_map=None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        self.llm = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
            dtype=torch_dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        hidden_size = int(self.llm.config.hidden_size)
        self.numeric_projector = NumericFeatureProjector(
            input_dim=numeric_feature_dim,
            hidden_size=hidden_size,
            num_numeric_tokens=num_numeric_tokens,
            dropout=dropout,
        )
        self.task_tokens = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_actions),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_priorities),
        )
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.priority_loss_fn = nn.CrossEntropyLoss()

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        for param in module.parameters():
            return param.dtype
        return torch.float32

    def move_task_modules(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> None:
        modules = [
            self.numeric_projector,
            self.action_head,
            self.priority_head,
            self.action_loss_fn,
            self.priority_loss_fn,
        ]
        for module in modules:
            if dtype is None:
                module.to(device=device)
            else:
                module.to(device=device, dtype=dtype)
        if dtype is None:
            self.task_tokens.data = self.task_tokens.data.to(device=device)
        else:
            self.task_tokens.data = self.task_tokens.data.to(device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_values: torch.Tensor,
        action_labels: Optional[torch.Tensor] = None,
        priority_labels: Optional[torch.Tensor] = None,
    ) -> MaintenanceSFTOutput:
        text_embeds = self.get_input_embeddings()(input_ids)
        numeric_dtype = self._module_dtype(self.numeric_projector)
        numeric_tokens = self.numeric_projector(numeric_values.to(device=text_embeds.device, dtype=numeric_dtype))
        numeric_tokens = numeric_tokens.to(dtype=text_embeds.dtype)

        batch_size = input_ids.size(0)
        task_tokens = self.task_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
        task_tokens = task_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        full_embeds = torch.cat([text_embeds, numeric_tokens, task_tokens], dim=1)
        extra_mask = torch.ones(
            batch_size,
            numeric_tokens.size(1) + task_tokens.size(1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat([attention_mask, extra_mask], dim=1)

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        action_state = last_hidden[:, -2, :].to(dtype=self._module_dtype(self.action_head))
        priority_state = last_hidden[:, -1, :].to(dtype=self._module_dtype(self.priority_head))
        action_logits = self.action_head(action_state)
        priority_logits = self.priority_head(priority_state)

        action_loss = None
        priority_loss = None
        loss = None
        if action_labels is not None and priority_labels is not None:
            action_loss = self.action_loss_fn(action_logits, action_labels.long())
            priority_loss = self.priority_loss_fn(priority_logits, priority_labels.long())
            loss = self.alpha * action_loss + (1.0 - self.alpha) * priority_loss

        return MaintenanceSFTOutput(
            loss=loss,
            action_loss=action_loss,
            priority_loss=priority_loss,
            action_logits=action_logits,
            priority_logits=priority_logits,
        )
