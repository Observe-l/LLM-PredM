from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


@dataclass
class PredMOutput:
    loss: Optional[torch.Tensor]
    rul_loss: Optional[torch.Tensor]
    action_loss: Optional[torch.Tensor]
    rul_preds: torch.Tensor
    action_logits: torch.Tensor


class InstancePatchEmbedder(nn.Module):
    def __init__(self, input_dim: int, patch_len: int, patch_stride: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim * patch_len, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, series: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = series.shape
        x = self.norm(series)
        if seq_len < self.patch_len:
            pad_len = self.patch_len - seq_len
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.size(1)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.contiguous().view(batch_size, -1, self.patch_len * feat_dim)
        return self.proj(patches)


class PatchReprogrammer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_prototypes: int, dropout: float = 0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, patch_tokens: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        prototypes = F.adaptive_avg_pool1d(text_embeds.transpose(1, 2), self.num_prototypes).transpose(1, 2)
        prototypes = self.prototype_proj(prototypes)
        reprogrammed, _ = self.attn(query=patch_tokens, key=prototypes, value=prototypes, need_weights=False)
        return self.output_proj(reprogrammed + patch_tokens)


class MultimodalQwenForPredM(nn.Module):
    def __init__(
        self,
        base_model_name_or_path: str,
        ts_feature_dim: int,
        num_actions: int,
        patch_len: int = 4,
        patch_stride: int = 4,
        num_heads: int = 4,
        num_prototypes: int = 8,
        dropout: float = 0.1,
        load_in_4bit: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        alpha: float = 0.5,
        rul_loss_scale: float = 1.0,
        device_map=None,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.rul_loss_scale = max(float(rul_loss_scale), 1e-6)
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
        self.hidden_size = hidden_size
        self.patch_embedder = InstancePatchEmbedder(
            input_dim=ts_feature_dim,
            patch_len=patch_len,
            patch_stride=patch_stride,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.patch_reprogrammer = PatchReprogrammer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_prototypes=num_prototypes,
            dropout=dropout,
        )
        self.task_tokens = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.rul_loss_fn = nn.MSELoss()
        self.action_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        for param in module.parameters():
            return param.dtype
        for buffer in module.buffers():
            return buffer.dtype
        return torch.float32

    def move_task_modules(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> None:
        modules = [
            self.patch_embedder,
            self.patch_reprogrammer,
            self.rul_head,
            self.action_head,
            self.rul_loss_fn,
            self.action_loss_fn,
        ]
        for module in modules:
            if dtype is None:
                module.to(device=device)
            else:
                module.to(device=device, dtype=dtype)

        task_tokens = self.task_tokens.data
        if dtype is None:
            self.task_tokens.data = task_tokens.to(device=device)
        else:
            self.task_tokens.data = task_tokens.to(device=device, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ts_values: torch.Tensor,
        rul_labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
    ) -> PredMOutput:
        text_embeds = self.get_input_embeddings()(input_ids)
        patch_dtype = self._module_dtype(self.patch_embedder)
        ts_values = ts_values.to(device=text_embeds.device, dtype=patch_dtype)
        patch_tokens = self.patch_embedder(ts_values)
        reprog_dtype = self._module_dtype(self.patch_reprogrammer)
        reprog_text = text_embeds.to(dtype=reprog_dtype)
        patch_tokens = patch_tokens.to(dtype=reprog_dtype)
        ts_embeds = self.patch_reprogrammer(patch_tokens, reprog_text)
        ts_embeds = ts_embeds.to(dtype=text_embeds.dtype)

        batch_size = input_ids.size(0)
        task_tokens = self.task_tokens.to(device=text_embeds.device, dtype=text_embeds.dtype)
        task_tokens = task_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        full_embeds = torch.cat([text_embeds, ts_embeds, task_tokens], dim=1)
        ts_mask = torch.ones(batch_size, ts_embeds.size(1) + task_tokens.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([attention_mask, ts_mask], dim=1)

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        head_dtype = self._module_dtype(self.rul_head)
        rul_state = last_hidden[:, -2, :].to(dtype=head_dtype)
        action_state = last_hidden[:, -1, :].to(dtype=self._module_dtype(self.action_head))
        rul_preds = self.rul_head(rul_state).squeeze(-1)
        action_logits = self.action_head(action_state)

        loss = None
        rul_loss = None
        action_loss = None
        if rul_labels is not None:
            rul_preds_scaled = rul_preds.float() / self.rul_loss_scale
            rul_labels_scaled = rul_labels.float() / self.rul_loss_scale
            rul_loss = self.rul_loss_fn(rul_preds_scaled, rul_labels_scaled)
            if action_labels is not None and torch.any(action_labels >= 0):
                action_loss = self.action_loss_fn(action_logits, action_labels.long())
                loss = self.alpha * rul_loss + (1.0 - self.alpha) * action_loss
            else:
                loss = rul_loss

        return PredMOutput(
            loss=loss,
            rul_loss=rul_loss,
            action_loss=action_loss,
            rul_preds=rul_preds,
            action_logits=action_logits,
        )
