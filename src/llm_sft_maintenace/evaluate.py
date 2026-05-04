from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .model import MaintenanceSFTModel
from .train import evaluate as run_evaluate
from .train import resolve_dtype
from .train import move_batch  # noqa: F401  # kept for parity/import stability
from .train import ensure_trainable_base
from .train import MaintenanceSFTCollator
from .train import MaintenanceSFTDataset


class AblationMaintenanceSFTDataset(MaintenanceSFTDataset):
    def __init__(self, artifacts_dir: str, split: str, prompt_mode: str) -> None:
        super().__init__(artifacts_dir=artifacts_dir, split=split)
        self.prompt_mode = prompt_mode

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = super().__getitem__(idx)
        if self.prompt_mode == "full":
            return item
        if self.prompt_mode == "no_prompt":
            item["prompt"] = " "
            return item
        raise ValueError(f"Unsupported prompt mode: {self.prompt_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate maintenance SFT checkpoints and run prompt ablations.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts/maintenance_sft")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def load_base_model_path(checkpoint_dir: Path) -> str:
    adapter_config_path = checkpoint_dir / "lora_adapter" / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {adapter_config_path}")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    base_model_path = str(adapter_config["base_model_name_or_path"])
    ensure_trainable_base(base_model_path)
    return base_model_path


def load_model_and_tokenizer(
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    load_in_4bit: bool,
    num_numeric_tokens: int,
) -> tuple[MaintenanceSFTModel, AutoTokenizer, Dict]:
    heads_path = checkpoint_dir / "multimodal_heads.pt"
    if not heads_path.exists():
        raise FileNotFoundError(f"Missing multimodal heads checkpoint: {heads_path}")
    checkpoint = torch.load(heads_path, map_location="cpu")
    schema = checkpoint["schema"]
    base_model_path = load_base_model_path(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "lora_adapter", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = MaintenanceSFTModel(
        base_model_name_or_path=base_model_path,
        numeric_feature_dim=len(schema["numeric_feature_names"]),
        num_actions=len(schema["action_to_id"]),
        num_priorities=len(schema["priority_to_id"]),
        num_numeric_tokens=num_numeric_tokens,
        load_in_4bit=load_in_4bit,
        torch_dtype=dtype,
        alpha=0.5,
        device_map={"": device.index or 0} if load_in_4bit and device.type == "cuda" else None,
    )
    model.llm = PeftModel.from_pretrained(model.llm, checkpoint_dir / "lora_adapter", is_trainable=False)
    model.numeric_projector.load_state_dict(checkpoint["numeric_projector"])
    model.action_head.load_state_dict(checkpoint["action_head"])
    model.priority_head.load_state_dict(checkpoint["priority_head"])
    model.task_tokens.data = checkpoint["task_tokens"].to(dtype=torch.float32)
    model.llm.config.use_cache = False
    model.move_task_modules(device=device, dtype=torch.float32)
    if not load_in_4bit:
        model.llm.to(device)
    model.eval()
    return model, tokenizer, schema


def build_loader(
    artifacts_dir: str,
    split: str,
    prompt_mode: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_text_length: int,
) -> DataLoader:
    dataset = AblationMaintenanceSFTDataset(
        artifacts_dir=artifacts_dir,
        split=split,
        prompt_mode=prompt_mode,
    )
    collator = MaintenanceSFTCollator(tokenizer, max_text_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def infer_num_numeric_tokens(checkpoint: Dict) -> int:
    proj_weight = checkpoint["numeric_projector"]["proj.0.weight"]
    out_norm_weight = checkpoint["numeric_projector"]["out_norm.weight"]
    hidden_size = int(out_norm_weight.shape[0])
    return int(proj_weight.shape[0] // hidden_size)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    heads_path = checkpoint_dir / "multimodal_heads.pt"
    checkpoint = torch.load(heads_path, map_location="cpu")
    num_numeric_tokens = infer_num_numeric_tokens(checkpoint)

    model, tokenizer, _ = load_model_and_tokenizer(
        checkpoint_dir=checkpoint_dir,
        device=device,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        num_numeric_tokens=num_numeric_tokens,
    )
    # Rebuild with the actual inferred token count if needed.
    if model.numeric_projector.num_numeric_tokens != num_numeric_tokens:
        raise RuntimeError("Loaded numeric projector token count does not match checkpoint.")

    full_loader = build_loader(
        artifacts_dir=args.artifacts_dir,
        split=args.split,
        prompt_mode="full",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_text_length=args.max_text_length,
    )
    no_prompt_loader = build_loader(
        artifacts_dir=args.artifacts_dir,
        split=args.split,
        prompt_mode="no_prompt",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_text_length=args.max_text_length,
    )

    results = {
        "split": args.split,
        "checkpoint_dir": str(checkpoint_dir),
        "full_prompt": run_evaluate(model, full_loader, device),
        "no_prompt": run_evaluate(model, no_prompt_loader, device),
    }
    print(json.dumps(results, indent=2), flush=True)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
