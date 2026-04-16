from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .model import MultimodalQwenForPredM


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


class WindowedPredMDataset(Dataset):
    def __init__(self, artifacts_dir: str, split: str, spec_feature_names: List[str]) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        with open(self.artifacts_dir / "vehicle_store.pkl", "rb") as f:
            self.vehicle_store: Dict[str, Dict] = pickle.load(f)
        with open(self.artifacts_dir / "feature_schema.json", "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.index = pd.read_csv(self.artifacts_dir / "sample_index.csv")
        self.index = self.index[self.index["split"] == split].reset_index(drop=True)
        self.spec_feature_names = spec_feature_names

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.index.iloc[idx]
        vehicle_id = str(row["vehicle_id"])
        item = self.vehicle_store[vehicle_id]
        endpoint_pos = int(row["endpoint_pos"])
        window_len = int(self.schema["window_len"])
        start = max(0, endpoint_pos - window_len + 1)
        sensor = item["sensor_values"][start : endpoint_pos + 1]
        time_values = item["time_values"][start : endpoint_pos + 1]
        current_len = sensor.shape[0]
        if current_len < window_len:
            pad_len = window_len - current_len
            sensor = np.pad(sensor, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0.0)
            time_values = np.pad(time_values, ((pad_len, 0), (0, 0)), mode="constant", constant_values=0.0)
        ts_values = np.concatenate([sensor, time_values], axis=1).astype(np.float32)

        spec_parts = [
            f"{name}=Cat{int(value)}"
            for name, value in zip(self.spec_feature_names, np.asarray(item["spec_ids"]).reshape(-1))
        ]
        prompt = (
            self.schema["domain_prompt"]
            + " Vehicle profile: "
            + ", ".join(spec_parts)
            + f". Current observation time: {float(row['endpoint_time']):.2f}."
            + " Predict the remaining useful life and the best maintenance recommendation for this observation."
        )
        return {
            "prompt": prompt,
            "ts_values": torch.tensor(ts_values, dtype=torch.float32),
            "rul_labels": torch.tensor(float(row["rul"]), dtype=torch.float32),
            "action_labels": torch.tensor(int(row["action_id"]), dtype=torch.long),
        }


class PredMCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in examples]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "ts_values": torch.stack([x["ts_values"] for x in examples]),
            "rul_labels": torch.stack([x["rul_labels"] for x in examples]),
            "action_labels": torch.stack([x["action_labels"] for x in examples]),
        }


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    output: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        output[key] = value.to(device)
    return output


def ensure_finite_batch(batch: Dict[str, torch.Tensor]) -> None:
    for key in ("ts_values", "rul_labels"):
        value = batch[key]
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite values found in batch tensor: {key}")


@torch.no_grad()
def evaluate(model: MultimodalQwenForPredM, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    rul_losses: List[float] = []
    action_losses: List[float] = []
    rul_preds_all: List[np.ndarray] = []
    rul_labels_all: List[np.ndarray] = []
    action_preds_all: List[np.ndarray] = []
    action_labels_all: List[np.ndarray] = []

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(**batch)
        if outputs.loss is not None:
            losses.append(float(outputs.loss.detach().cpu()))
        if outputs.rul_loss is not None:
            rul_losses.append(float(outputs.rul_loss.detach().cpu()))
        if outputs.action_loss is not None:
            action_losses.append(float(outputs.action_loss.detach().cpu()))
        rul_preds_all.append(outputs.rul_preds.detach().cpu().numpy())
        rul_labels_all.append(batch["rul_labels"].detach().cpu().numpy())
        action_preds_all.append(outputs.action_logits.argmax(dim=-1).detach().cpu().numpy())
        action_labels_all.append(batch["action_labels"].detach().cpu().numpy())

    rul_preds = np.concatenate(rul_preds_all)
    rul_labels = np.concatenate(rul_labels_all)
    action_preds = np.concatenate(action_preds_all)
    action_labels = np.concatenate(action_labels_all)
    labeled_mask = action_labels >= 0
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "rul_loss": float(np.mean(rul_losses)) if rul_losses else float("nan"),
        "action_loss": float(np.mean(action_losses)) if action_losses else float("nan"),
        "rul_mae": float(np.mean(np.abs(rul_preds - rul_labels))),
        "rul_rmse": float(np.sqrt(np.mean((rul_preds - rul_labels) ** 2))),
        "action_acc": float(np.mean(action_preds[labeled_mask] == action_labels[labeled_mask])) if labeled_mask.any() else float("nan"),
        "action_support": int(labeled_mask.sum()),
    }


def ensure_trainable_base(model_path: str) -> None:
    placeholder_markers = [
        "/path/to/",
        "hf-compatible-qwen-checkpoint",
        "your-model-path",
    ]
    if any(marker in model_path for marker in placeholder_markers):
        raise ValueError(
            "You are still using the placeholder `--base_model_path` from the README. "
            "Please replace it with a real Hugging Face model ID like `Qwen/Qwen2.5-3B-Instruct` "
            "or an actual local checkpoint directory path."
        )
    if model_path.endswith(".gguf") or ":" in model_path:
        raise ValueError(
            "LoRA fine-tuning here expects a Hugging Face compatible Qwen checkpoint path, not an Ollama model name or GGUF file. "
            "Ollama qwen3.5:4b can be used for deployment, but training needs the corresponding Transformers weights."
        )
    looks_like_local_path = model_path.startswith("/") or model_path.startswith("./") or model_path.startswith("../")
    if looks_like_local_path:
        local_path = Path(model_path)
        if not local_path.exists():
            raise FileNotFoundError(
                f"`--base_model_path` points to a local path that does not exist: {model_path}"
            )
        if local_path.is_dir() and not (local_path / "config.json").exists():
            raise FileNotFoundError(
                f"Local model directory is missing `config.json`: {model_path}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multimodal Qwen model for predictive maintenance.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts/qwen_predm")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_predm_lora")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_text_length", type=int, default=256)
    parser.add_argument("--patch_len", type=int, default=1)
    parser.add_argument("--patch_stride", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_prototypes", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_trainable_base(args.base_model_path)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    with open(Path(args.artifacts_dir) / "feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    spec_feature_names = list(schema["spec_feature_names"])
    num_actions = len(schema["action_to_id"])
    ts_feature_dim = len(schema["sensor_feature_names"]) + len(schema["time_feature_names"])
    sample_index = pd.read_csv(Path(args.artifacts_dir) / "sample_index.csv")
    train_index = sample_index[sample_index["split"] == "train"].reset_index(drop=True)
    rul_loss_scale = float(train_index["rul"].std())
    if not np.isfinite(rul_loss_scale) or rul_loss_scale <= 1e-6:
        rul_loss_scale = float(train_index["rul"].abs().mean())
    if not np.isfinite(rul_loss_scale) or rul_loss_scale <= 1e-6:
        rul_loss_scale = 100.0

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = WindowedPredMDataset(args.artifacts_dir, "train", spec_feature_names)
    collator = PredMCollator(tokenizer, args.max_text_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    test_index = sample_index[sample_index["split"] == "test"].reset_index(drop=True)
    test_loader = None
    if len(test_index) > 0:
        test_dataset = WindowedPredMDataset(args.artifacts_dir, "test", spec_feature_names)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = MultimodalQwenForPredM(
        base_model_name_or_path=args.base_model_path,
        ts_feature_dim=ts_feature_dim,
        num_actions=num_actions,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        num_heads=args.num_heads,
        num_prototypes=args.num_prototypes,
        dropout=args.dropout,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=dtype,
        alpha=args.alpha,
        rul_loss_scale=rul_loss_scale,
        device_map={"": device.index or 0} if args.load_in_4bit and device.type == "cuda" else None,
    )
    if args.load_in_4bit:
        model.llm = prepare_model_for_kbit_training(model.llm)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.move_task_modules(device=device, dtype=torch.float32)
    if not args.load_in_4bit:
        model.llm.to(device)
    model.train()

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_rul_loss = 0.0
        running_action_loss = 0.0
        running_action_batches = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            ensure_finite_batch(batch)
            outputs = model(**batch)
            if outputs.loss is None or not torch.isfinite(outputs.loss):
                raise FloatingPointError("Non-finite loss detected during training.")
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += float(outputs.loss.detach().cpu())
            if outputs.rul_loss is not None:
                running_rul_loss += float(outputs.rul_loss.detach().cpu())
            if outputs.action_loss is not None:
                running_action_loss += float(outputs.action_loss.detach().cpu())
                running_action_batches += 1

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                    max_norm=args.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "total_loss": running_loss / step,
                            "rul_loss": running_rul_loss / step,
                            "action_loss": running_action_loss / running_action_batches if running_action_batches > 0 else None,
                            "rul_loss_scale": rul_loss_scale,
                        }
                    ),
                    flush=True,
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        if test_loader is not None:
            metrics = evaluate(model, test_loader, device)
            metrics["epoch"] = epoch + 1
            history.append(metrics)
            print(json.dumps(metrics, indent=2), flush=True)
        else:
            history.append({"epoch": epoch + 1})

        epoch_dir = Path(args.output_dir) / f"epoch_{epoch + 1}"
        adapter_dir = epoch_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.llm.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        torch.save(
            {
                "patch_embedder": model.patch_embedder.state_dict(),
                "patch_reprogrammer": model.patch_reprogrammer.state_dict(),
                "task_tokens": model.task_tokens.detach().cpu(),
                "rul_head": model.rul_head.state_dict(),
                "action_head": model.action_head.state_dict(),
                "schema": schema,
            },
            epoch_dir / "multimodal_heads.pt",
        )

    with open(Path(args.output_dir) / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
