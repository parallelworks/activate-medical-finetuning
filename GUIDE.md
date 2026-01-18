# Medical LLM Fine-tuning - Detailed Guide

Complete reference for the Parallel Works ACTIVATE Medical LLM Fine-tuning Workflow.

## Table of Contents

- [Workflow Architecture](#workflow-architecture)
- [Complete Parameter Reference](#complete-parameter-reference)
- [Dataset Sources Guide](#dataset-sources-guide)
- [Output Artifacts](#output-artifacts)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Resources](#resources)

## Workflow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Setup Job     │───▶│  Build Container │───▶│   Fine-tune     │
│  - GPU Check    │    │  - Singularity   │    │  - QLoRA Train  │
│  - Clone Repo   │    │    Build         │    │  - Merge Weights│
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Upload Outputs  │◀─────┐
                                                │  - Adapters     │      │
                                                │  - Merged Model │      │
                                                └────────┬────────┘      │
                                                         │               │
                                                         ▼               │
                                                ┌─────────────────┐      │
                                                │     Cleanup     │──────┘
                                                │  - Summary      │
                                                │  - Validation   │
                                                └─────────────────┘
```

## Complete Parameter Reference

The workflow is organized into 7 logical sections for easy configuration:

### Section 1: Infrastructure & Compute

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cluster` | compute-clusters | *required* | GPU cluster for training |
| `container.source` | dropdown | `lfs` | Container source (lfs/path/pull/build) |
| `container.finetune_path` | string | `~/pw/singularity/finetune.sif` | Path for existing/build container (path/build/lfs) |
| `container.lfs_repo` | string | `~/singularity-containers` | LFS repo containing finetune SIF parts |
| `container.bucket` | storage | `""` | Bucket containing finetune.sif (pull only) |
| `output_bucket` | storage | `""` | Bucket to store trained weights (empty = skip upload, outputs stay on cluster) |

**Container Behavior:**
- `lfs` (default): pulls finetune parts from `container.lfs_repo`, assembles to `container.finetune_path`, and installs git-lfs if needed
- `path`: uses an existing SIF at `container.finetune_path`
- `pull`: downloads `finetune.sif` from `container.bucket` to the run directory
- `build`: builds from `singularity/finetune.def` into `container.finetune_path` (requires sudo/fakeroot)

**Output Bucket Behavior:**
- Leave empty to skip bucket upload - outputs remain on the cluster at `${HOME}/pw/outputs/medical-finetune-{RUN_ID}/`
- The `upload-outputs` job only runs if a bucket is specified
- Useful for quick tests or when you don't need persistent storage

### Section 2: Model Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_source` | dropdown | `huggingface` | Use local path or clone from HuggingFace (git-lfs) |
| `model_config` | string | `llama3-8b-medical` | Predefined configuration preset |
| `base_model_id` | string | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID (for git-lfs clone) |
| `local_model_path` | string | `/models/Llama-3.1-8B-Instruct` | Local model directory (local source) |
| `model_cache_dir` | string | `~/pw/models` | Cache directory for cloned models |
| `hf_token` | password | `${{ org.HF_TOKEN }}` | HuggingFace token for gated models |

**Supported Base Models:**
- `meta-llama/Llama-3.1-8B-Instruct` (Llama 3)
- `meta-llama/Llama-2-7b-chat-hf` (Llama 2)
- `mistralai/Mistral-7B-Instruct-v0.2` (Mistral)
- `google/gemma-1.1-7b-it` (Gemma)

**Model Cache Behavior:**
- HuggingFace models are cloned via git-lfs into `model_cache_dir` (default `~/pw/models`)
- If the model already exists with `config.json`, the clone step is skipped

### Section 3: Dataset Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_source` | dropdown | `huggingface` | Dataset source type |
| `dataset_name` | string | `Shekswess/medical_llama3_instruct_dataset` | HuggingFace dataset (HF only) |
| `dataset_bucket` | storage | `""` | PW storage bucket (bucket only) |
| `dataset_bucket_path` | string | `""` | Path within bucket (bucket only) |
| `local_dataset_path` | string | `""` | Local file path (local only) |
| `dataset_format` | dropdown | `json` | File format (local only) |
| `dataset_split` | string | `train` | Dataset split to use |
| `dataset_config_name` | string | `""` | Optional dataset config (HF only) |
| `prompt_field` | string | `prompt` | Column with formatted prompts |
| `max_samples` | number | `0` | Sample limit (0 = all) |

**Supported Local File Formats:**
- `json` / `jsonl` - JSON or JSON Lines
- `csv` - Comma-separated values
- `parquet` - Apache Parquet
- `arrow` - Apache Arrow
- `dataset` - HuggingFace dataset directory

**Pre-configured Medical Datasets (HuggingFace):**
- `Shekswess/medical_llama3_instruct_dataset` - Full medical instructions for Llama 3
- `Shekswess/medical_llama3_instruct_dataset_short` - Quick test dataset (500 samples)
- `Shekswess/medical_mistral_instruct_dataset` - Mistral-formatted medical data
- `Shekswess/medical_gemma_instruct_dataset` - Gemma-formatted medical data
- `Shekswess/llama3_medical_meadow_wikidoc_instruct_dataset` - WikiDoc medical knowledge
- `Shekswess/llama3_medquad_instruct_dataset` - MedQuAD QA pairs

### Section 4: Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | number | `3.0` | Training epochs |
| `learning_rate` | number | `0.0002` | Learning rate (2e-4) |
| `micro_batch_size` | number | `1` | Per-GPU batch size |
| `gradient_accumulation` | number | `16` | Gradient accumulation steps |
| `max_seq_length` | number | `2048` | Max sequence length (tokens) |

**Effective Batch Size Calculation:**
```
effective_batch_size = micro_batch_size × gradient_accumulation × num_gpus
```

### Section 5: LoRA / PEFT Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_r` | number | `64` | LoRA rank (higher = more params) |
| `lora_alpha` | number | `16` | LoRA alpha scaling factor |
| `lora_dropout` | number | `0.05` | LoRA dropout rate |
| `quantization` | dropdown | `4bit` | Quantization mode (4bit/8bit/none) |
| `gradient_checkpointing` | boolean | `true` | Enable gradient checkpointing |
| `bf16` | boolean | `false` | Use bfloat16 (requires GPU support) |
| `packing` | boolean | `false` | Pack multiple samples per sequence |

**Memory Optimization Tips:**
- Use `4bit` quantization for models >7B parameters
- Enable `gradient_checkpointing` to reduce memory by ~30%
- Reduce `micro_batch_size` if encountering OOM errors
- Increase `gradient_accumulation` to maintain effective batch size

### Section 6: Output & Publishing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merge_full_weights` | boolean | `true` | Merge LoRA into full weights |
| `merged_save_format` | dropdown | `safetensors` | Format (safetensors/bin) |
| `push_to_hub` | boolean | `false` | Push to HuggingFace Hub |
| `hub_model_id` | string | `""` | Target Hub repo (username/model) |
| `hf_token` | password | `""` | HuggingFace access token |

## Dataset Sources Guide

### Option 1: HuggingFace Hub (Default)

Direct download from HuggingFace - best for public datasets and quick testing.

**Workflow Configuration:**
- `dataset_source`: `huggingface`
- `dataset_name`: `Shekswess/medical_llama3_instruct_dataset`
- `dataset_split`: `train`
- `dataset_config_name`: (optional) for datasets with multiple configurations

**Pros:** No preprocessing, automatic caching, wide selection
**Cons:** Requires internet, depends on HF availability

### Option 2: PW Storage Bucket

Store datasets in Parallel Works storage for repeated use and sharing across teams.

**Upload Dataset to Bucket:**
```bash
# Upload your prepared dataset to a PW bucket
pw storage cp my-medical-dataset.jsonl my-bucket/datasets/
```

**Workflow Configuration:**
- `dataset_source`: `bucket`
- `dataset_bucket`: Select your storage bucket
- `dataset_bucket_path`: `datasets/` (path within bucket)

**Pros:** No re-downloads, team sharing, works offline after first upload
**Cons:** Requires initial upload step

### Option 3: Local File Path

Load datasets directly from the cluster filesystem.

**Workflow Configuration:**
- `dataset_source`: `local`
- `local_dataset_path`: `/path/to/cluster/data/my-medical-dataset.jsonl`
- `dataset_format`: `json` (or `csv`, `parquet`, `arrow`, `dataset`)

**Supported Formats:**
| Format | Extension | Example |
|--------|-----------|---------|
| JSON/JSONL | `.json`, `.jsonl` | `/data/medical.jsonl` |
| CSV | `.csv` | `/data/medical.csv` |
| Parquet | `.parquet` | `/data/medical.parquet` |
| Arrow | `.arrow` | `/data/medical.arrow` |
| HF Dataset | Directory | `/data/medical_dataset/` |

**Pros:** No network dependency, full control, fastest loading
**Cons:** Dataset must exist on cluster, manual management

## Dataset Format Requirements

All dataset sources require a `prompt` column with formatted instruction data. Example JSONL format:

```json
{"prompt": "<|start_header_id|>system<|end_header_id|>You are a medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>What is diabetes?<|eot_id|><|start_header_id|>assistant<|end_header_id|>Diabetes is a chronic disease...</|eot_id|>"}
{"prompt": "<|start_header_id|>system<|end_header_id|>You are a medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Explain hypertension.<|eot_id|><|start_header_id|>assistant<|end_header_id|>Hypertension is high blood pressure...</|eot_id|>"}
```

**Creating a Custom Dataset:**

```python
import json
from datasets import Dataset

# Your data in Q&A format
data = [
    {
        "question": "What is diabetes?",
        "answer": "Diabetes is a chronic disease that affects how your body turns food into energy."
    },
    # ... more samples
]

# Format with prompts
formatted = []
for item in data:
    prompt = f"""<|start_header_id|>system<|end_header_id|>You are a medical assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{item['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{item['answer']}<|eot_id|>"""
    formatted.append({"prompt": prompt})

# Save as JSONL
with open("medical-dataset.jsonl", "w") as f:
    for item in formatted:
        f.write(json.dumps(item) + "\n")
```

## Output Artifacts

### Directory Structure

```
medical-finetune-{RUN_ID}/
├── adapters/                    # LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── merged/                      # Merged full weights (vLLM ready)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── logs/
│   └── finetune.log             # Training logs
└── training.env                 # Training configuration
```

### Using Fine-tuned Weights with vLLM

After training, use the merged weights with vLLM:

```bash
# Pull merged weights from storage or copy from output directory
vllm serve {OUTPUT_DIR}/merged \
  --model {OUTPUT_DIR}/merged \
  --tensor-parallel-size {NUM_GPUS} \
  --max-model-len 2048
```

### Using LoRA Adapters

Load adapters directly without merging:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)
model = PeftModel.from_pretrained(
    base_model,
    "{OUTPUT_DIR}/adapters"
)
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:** CUDA out of memory during training

**Solutions:**
1. Reduce `micro_batch_size` to 1 or lower
2. Increase `gradient_accumulation` to maintain effective batch size
3. Ensure `quantization` is set to `4bit`
4. Enable `gradient_checkpointing`
5. Reduce `max_seq_length` from 2048 to 1024

### Slow Training

**Symptoms:** Training takes longer than expected

**Solutions:**
1. Increase `micro_batch_size` if memory allows
2. Reduce `gradient_accumulation` accordingly
3. Use `bf16: true` if GPU supports it (A100, RTX 30xx+)
4. Check GPU utilization with `nvidia-smi`

### HuggingFace Authentication Errors

**Symptoms:** Cannot access gated model/dataset

**Solutions:**
1. Set `hf_token` with valid HuggingFace access token
2. Accept model license agreement on HuggingFace website
3. Ensure token has `read` permissions for gated models

### Singularity Build Failures

**Symptoms:** Container build fails

**Solutions:**
1. Set `container.source` to `lfs` (default) or `build` for cluster builds
2. Ensure `apptainer` or `singularity` is installed on cluster
3. Check network connectivity for pulling base images (nvcr.io)
4. If building manually on the cluster:
   ```bash
   cd ~/pw/activate-medical-finetune
   singularity build ~/pw/singularity/finetune.sif singularity/finetune.def
   ```
5. For pre-built containers, set `container.source` to `path` and provide `container.finetune_path`

## Advanced Usage

### Hyperparameter Tuning

Key parameters to tune for better results:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `learning_rate` | 1e-5 to 5e-4 | Lower = more stable, slower convergence |
| `lora_r` | 8 to 128 | Higher = more capacity, more memory |
| `num_epochs` | 1 to 10 | More epochs = better fit, risk of overfitting |
| `micro_batch_size` | 1 to 8 | Larger = faster training, more memory |

### Multi-GPU Training

For multi-GPU clusters, the workflow automatically distributes training. The effective batch size scales with GPU count:

```
# Single GPU: micro_batch_size=1, gradient_accumulation=16
# Effective batch: 1 × 16 × 1 = 16

# 4 GPUs: micro_batch_size=1, gradient_accumulation=16
# Effective batch: 1 × 16 × 4 = 64
```

## Performance Benchmarks

Training time on single GPU (A100 40GB):

| Model | Dataset | Samples | Epochs | Quantization | Time |
|-------|---------|---------|--------|--------------|------|
| Llama 3 8B | medical-llama3-short | 500 | 1 | 4bit | ~25 min |
| Llama 3 8B | medical-llama3-full | 15,000 | 3 | 4bit | ~4.5 hours |
| Mistral 7B | medical-mistral-full | 12,000 | 3 | 4bit | ~3.5 hours |
| Gemma 7B | medical-gemma-full | 10,000 | 3 | 4bit | ~3 hours |

## Resources

- [LLM-Medical-Finetuning GitHub](https://github.com/Shekswess/LLM-Medical-Finetuning)
- [Parallel Works Documentation](https://parallelworks.com/docs)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PEFT Library](https://github.com/huggingface/peft)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in the output directory
3. Consult the [Parallel Works documentation](https://parallelworks.com/docs)
4. Open an issue in the LLM-Medical-Finetuning GitHub repository
