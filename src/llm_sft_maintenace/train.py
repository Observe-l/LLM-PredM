from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .model import MaintenanceSFTModel


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


class MaintenanceSFTDataset(Dataset):
    def __init__(self, artifacts_dir: str, split: str) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        with open(self.artifacts_dir / "records.pkl", "rb") as f:
            all_records: List[Dict] = pickle.load(f)
        with open(self.artifacts_dir / "feature_schema.json", "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.records = [record for record in all_records if record["split"] == split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[idx]
        prompt = (
            self.schema["domain_prompt"]
            + " Event metadata: "
            + f"event_id={record['event_id']}, case_id={record['case_id']}, work_order_id={record['work_order_id']}, "
            + f"vehicle_id={record['vehicle_id']}, system={record['system']}, subsystem={record['subsystem']}, component={record['component']}. "
            + f"Maintenance note: {record['maintenance_note']}"
        )
        return {
            "prompt": prompt,
            "numeric_values": torch.tensor(record["numeric_values"], dtype=torch.float32),
            "action_labels": torch.tensor(int(record["action_label"]), dtype=torch.long),
            "priority_labels": torch.tensor(int(record["priority_label"]), dtype=torch.long),
        }


class MaintenanceSFTCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_text_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in examples]
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
            "numeric_values": torch.stack([item["numeric_values"] for item in examples]),
            "action_labels": torch.stack([item["action_labels"] for item in examples]),
            "priority_labels": torch.stack([item["priority_labels"] for item in examples]),
        }


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: MaintenanceSFTModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    action_losses: List[float] = []
    priority_losses: List[float] = []
    action_preds_all: List[np.ndarray] = []
    action_labels_all: List[np.ndarray] = []
    priority_preds_all: List[np.ndarray] = []
    priority_labels_all: List[np.ndarray] = []

    for batch in loader:
        outputs = model(**move_batch(batch, device))
        if outputs.loss is not None:
            losses.append(float(outputs.loss.detach().cpu()))
        if outputs.action_loss is not None:
            action_losses.append(float(outputs.action_loss.detach().cpu()))
        if outputs.priority_loss is not None:
            priority_losses.append(float(outputs.priority_loss.detach().cpu()))
        action_preds_all.append(outputs.action_logits.argmax(dim=-1).detach().cpu().numpy())
        priority_preds_all.append(outputs.priority_logits.argmax(dim=-1).detach().cpu().numpy())
        action_labels_all.append(batch["action_labels"].numpy())
        priority_labels_all.append(batch["priority_labels"].numpy())

    action_preds = np.concatenate(action_preds_all)
    action_labels = np.concatenate(action_labels_all)
    priority_preds = np.concatenate(priority_preds_all)
    priority_labels = np.concatenate(priority_labels_all)
    joint_acc = np.mean((action_preds == action_labels) & (priority_preds == priority_labels))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "action_loss": float(np.mean(action_losses)) if action_losses else float("nan"),
        "priority_loss": float(np.mean(priority_losses)) if priority_losses else float("nan"),
        "action_acc": float(np.mean(action_preds == action_labels)),
        "priority_acc": float(np.mean(priority_preds == priority_labels)),
        "joint_acc": float(joint_acc),
    }


def ensure_trainable_base(model_path: str) -> None:
    placeholder_markers = ["/path/to/", "hf-compatible-qwen-checkpoint", "your-model-path"]
    if any(marker in model_path for marker in placeholder_markers):
        raise ValueError("Replace --base_model_path with a real Hugging Face model ID or local checkpoint path.")
    if model_path.endswith(".gguf") or ":" in model_path:
        raise ValueError(
            "This training pipeline requires a Hugging Face-compatible checkpoint path, not an Ollama model name or GGUF file."
        )
    if model_path.startswith("/") or model_path.startswith("./") or model_path.startswith("../"):
        local_path = Path(model_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_path}")
        if local_path.is_dir() and not (local_path / "config.json").exists():
            raise FileNotFoundError(f"Local model directory is missing config.json: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a maintenance action SFT model with text and numeric inputs.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts/maintenance_sft")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/maintenance_sft_lora")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--num_numeric_tokens", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    model: MaintenanceSFTModel,
    tokenizer: AutoTokenizer,
    schema: Dict,
    output_dir: Path,
) -> None:
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(
        {
            "numeric_projector": model.numeric_projector.state_dict(),
            "task_tokens": model.task_tokens.detach().cpu(),
            "action_head": model.action_head.state_dict(),
            "priority_head": model.priority_head.state_dict(),
            "schema": schema,
        },
        output_dir / "multimodal_heads.pt",
    )


def main() -> None:
    args = parse_args()
    ensure_trainable_base(args.base_model_path)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    with open(Path(args.artifacts_dir) / "feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    numeric_feature_dim = len(schema["numeric_feature_names"])
    num_actions = len(schema["action_to_id"])
    num_priorities = len(schema["priority_to_id"])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = MaintenanceSFTDataset(args.artifacts_dir, "train")
    val_dataset = MaintenanceSFTDataset(args.artifacts_dir, "validation")
    test_dataset = MaintenanceSFTDataset(args.artifacts_dir, "test")
    collator = MaintenanceSFTCollator(tokenizer, args.max_text_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    model = MaintenanceSFTModel(
        base_model_name_or_path=args.base_model_path,
        numeric_feature_dim=numeric_feature_dim,
        num_actions=num_actions,
        num_priorities=num_priorities,
        num_numeric_tokens=args.num_numeric_tokens,
        dropout=args.dropout,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=dtype,
        alpha=args.alpha,
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
    model.llm.config.use_cache = False
    if not args.disable_gradient_checkpointing:
        model.llm.gradient_checkpointing_enable()
    model.move_task_modules(device=device, dtype=torch.float32)
    if not args.load_in_4bit:
        model.llm.to(device)
    model.train()

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(args.output_dir, exist_ok=True)
    best_joint_acc = -1.0
    global_step = 0
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_action_loss = 0.0
        running_priority_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            outputs = model(**move_batch(batch, device))
            if outputs.loss is None or not torch.isfinite(outputs.loss):
                raise FloatingPointError("Non-finite loss detected during training.")
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += float(outputs.loss.detach().cpu())
            running_action_loss += float(outputs.action_loss.detach().cpu()) if outputs.action_loss is not None else 0.0
            running_priority_loss += float(outputs.priority_loss.detach().cpu()) if outputs.priority_loss is not None else 0.0

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
                            "action_loss": running_action_loss / step,
                            "priority_loss": running_priority_loss / step,
                        }
                    ),
                    flush=True,
                )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        val_metrics = evaluate(model, val_loader, device)
        val_metrics["epoch"] = epoch + 1
        history.append({"split": "validation", **val_metrics})
        print(json.dumps({"split": "validation", **val_metrics}, indent=2), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        epoch_dir = Path(args.output_dir) / f"epoch_{epoch + 1}"
        save_checkpoint(model, tokenizer, schema, epoch_dir)

        if val_metrics["joint_acc"] > best_joint_acc:
            best_joint_acc = val_metrics["joint_acc"]
            save_checkpoint(model, tokenizer, schema, Path(args.output_dir) / "best")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    test_metrics = evaluate(model, test_loader, device)
    history.append({"split": "test", **test_metrics})
    print(json.dumps({"split": "test", **test_metrics}, indent=2), flush=True)
    with open(Path(args.output_dir) / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
