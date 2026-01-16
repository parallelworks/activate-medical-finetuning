#!/usr/bin/env python3
"""
Utility script that performs PEFT (QLoRA) fine-tuning over medical instruction
data. Designed for non-interactive execution inside Parallel Works ACTIVATE
workflows using Singularity containers.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on medical instruction data with QLoRA."
    )
    parser.add_argument(
        "--base-model-id",
        required=True,
        help="Base Hugging Face model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["huggingface", "local", "bucket"],
        default="huggingface",
        help="Dataset source type.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset repository or local path that contains the prompt column.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset configuration name when loading from the Hub.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use for supervised fine-tuning.",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Column in the dataset that stores the formatted prompt.",
    )
    parser.add_argument(
        "--local-dataset-path",
        default=None,
        help="Path to local dataset file or directory.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["json", "csv", "parquet", "arrow", "dataset"],
        default="json",
        help="Format of local dataset file.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory containing dataset downloaded from bucket.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for dataset shuffling and trainer.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where adapters (and merged weights if requested) are stored.",
    )
    parser.add_argument(
        "--num-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Per-device micro batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum packed sequence length.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Frequency (in steps) for logging metrics.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="How often checkpoints are saved.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoint folders to retain.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="Rank used for the LoRA adapters.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="Alpha scaling factor used for LoRA adapters.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout probability applied inside the LoRA adapters.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma separated list of modules to wrap with LoRA.",
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization scheme to use for the base model.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token used to access gated models or datasets.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="If set, push the adapter checkpoint to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-model-id",
        default=None,
        help="Optional Hub repo name used when pushing results.",
    )
    parser.add_argument(
        "--merge-full-weights",
        action="store_true",
        help="When enabled, merge adapters back into the base model for vLLM.",
    )
    parser.add_argument(
        "--merged-save-format",
        choices=["safetensors", "bin"],
        default="safetensors",
        help="Serialization format for merged checkpoints.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code to transformers loader.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the base model.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Train using bfloat16 where possible.",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable dataset packing to reduce padding for short samples.",
    )
    return parser.parse_args()


def maybe_login(token: Optional[str]) -> None:
    if token:
        login(token=token, add_to_git_credential=True)
        LOG.info("Logged in to Hugging Face Hub.")


def load_training_dataset(args: argparse.Namespace) -> Dataset:
    dataset_source = getattr(args, "dataset_source", "huggingface")
    LOG.info("Loading dataset from source: %s", dataset_source)

    if dataset_source == "huggingface":
        # Load from HuggingFace Hub
        dataset_kwargs = {
            "path": args.dataset_name,
            "split": args.dataset_split,
        }
        if args.dataset_config:
            dataset_kwargs["name"] = args.dataset_config
        if args.hf_token:
            dataset_kwargs["token"] = args.hf_token
        dataset = load_dataset(**dataset_kwargs)
        LOG.info("Loaded dataset %s (split=%s) from HuggingFace Hub", args.dataset_name, args.dataset_split)

    elif dataset_source == "local":
        # Load from local file or directory
        local_path = getattr(args, "local_dataset_path", None)
        if not local_path:
            raise ValueError("local_dataset_path is required for local dataset source")

        dataset_format = getattr(args, "dataset_format", "json")

        if dataset_format == "dataset":
            # Load as HuggingFace dataset directory
            dataset = load_dataset(local_path, split=args.dataset_split or "train")
        else:
            # Load from file (json, csv, parquet, arrow)
            dataset = load_dataset(
                dataset_format,
                data_files=local_path,
                split="train",
            )
        LOG.info("Loaded dataset from local path: %s (format: %s)", local_path, dataset_format)

    elif dataset_source == "bucket":
        # Load from downloaded bucket data (already prepared in DATASET_DIR)
        dataset_dir = getattr(args, "dataset_dir", None)
        if not dataset_dir:
            raise ValueError("dataset_dir is required for bucket dataset source")

        # Try to load as HuggingFace dataset directory first
        try:
            dataset = load_dataset(dataset_dir, split=args.dataset_split or "train")
            LOG.info("Loaded dataset from bucket directory: %s", dataset_dir)
        except Exception as e:
            # If that fails, try to detect and load from files
            LOG.warning("Could not load as dataset directory: %s. Trying file formats...", e)

            # Look for common dataset files
            import os
            files = []
            for ext in ["*.jsonl", "*.json", "*.parquet", "*.csv"]:
                import glob
                found = glob.glob(os.path.join(dataset_dir, ext))
                files.extend(found)

            if not files:
                raise ValueError(f"No supported dataset files found in {dataset_dir}")

            # Use the first found file
            data_file = files[0]
            if data_file.endswith((".json", ".jsonl")):
                dataset = load_dataset("json", data_files=data_file, split="train")
            elif data_file.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=data_file, split="train")
            elif data_file.endswith(".csv"):
                dataset = load_dataset("csv", data_files=data_file, split="train")
            else:
                raise ValueError(f"Unsupported file format: {data_file}")

            LOG.info("Loaded dataset from bucket file: %s", data_file)

    else:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")

    # Apply sample limit
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Validate prompt field
    if args.prompt_field not in dataset.column_names:
        raise ValueError(
            f"Dataset does not contain '{args.prompt_field}' column. "
            f"Available columns: {dataset.column_names}"
        )

    dataset = dataset.shuffle(seed=args.seed)
    LOG.info("Final dataset size: %d samples", len(dataset))
    return dataset


def build_base_model(
    args: argparse.Namespace,
) -> tuple[AutoModelForCausalLM | PeftModel, AutoTokenizer]:
    quant_config = None
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_config,
        use_auth_token=args.hf_token,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    target_modules: List[str] = [
        module.strip()
        for module in args.lora_target_modules.split(",")
        if module.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules or None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
        use_auth_token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.config.use_cache = False
    return model, tokenizer


def train(args: argparse.Namespace) -> Path:
    maybe_login(args.hf_token)
    dataset = load_training_dataset(args)

    # Pre-format dataset: rename 'prompt' field to 'text' for TRL
    # Newer TRL versions expect a 'text' field or use formatting_func
    LOG.info("Pre-formatting dataset for SFTTrainer...")
    if args.prompt_field != "text":
        dataset = dataset.map(lambda x: {"text": x[args.prompt_field]}, remove_columns=[args.prompt_field])
        LOG.info("Renamed '%s' field to 'text'", args.prompt_field)

    model, tokenizer = build_base_model(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    adapter_dir = output_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=args.packing,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    LOG.info("Saved LoRA adapters to %s", adapter_dir)

    if args.push_to_hub:
        hub_target = args.hub_model_id or f"{Path(args.base_model_id).name}-medical"
        trainer.model.push_to_hub(hub_target)
        tokenizer.push_to_hub(hub_target)

    if args.merge_full_weights:
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=args.merged_save_format == "safetensors",
        )
        tokenizer.save_pretrained(merged_dir)
        LOG.info("Saved merged weights for vLLM to %s", merged_dir)

    return adapter_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    args = parse_args()
    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN")
    train(args)


if __name__ == "__main__":
    main()
