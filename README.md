# Medical LLM Fine-tuning Workflow

Parallel Works ACTIVATE workflow for fine-tuning LLMs (Llama2/3, Mistral, Gemma) on medical datasets using QLoRA/LoRA via Singularity containers. Generates vLLM-compatible merged weights.

## Quick Start

### 1. Build Singularity Container (Optional)

The workflow can pull a prebuilt container (default: Git LFS) or build from source. You can also pre-build locally:

```bash
./scripts/build_container.sh          # Build to default location
./scripts/build_container.sh --sudo   # Use sudo if needed
```

Output: `~/pw/singularity/finetune.sif`

### 2. Run in Parallel Works

1. Select a GPU cluster with 16GB+ VRAM
2. Choose a configuration preset:
   - `llama3-8b-medical` - Production training (15K samples, ~4-6 hours)
   - `llama3-8b-medical-quick` - Quick test (500 samples, ~30 min)
   - `mistral-7b-medical` - Mistral 7B training
   - `gemma-7b-medical` - Gemma 7B training
3. Leave **Singularity Image Path** as default (auto-builds on cluster if needed)
4. Leave **Output Storage Bucket** empty to skip upload (outputs stay on cluster)
5. Click **Execute** and monitor in the **Runs** tab

### 3. Use Fine-tuned Model

```bash
# Serve with vLLM
vllm serve {OUTPUT_DIR}/merged \
  --tensor-parallel-size {NUM_GPUS} \
  --max-model-len 2048
```

## Turnkey Examples

### Example 1: Quick Test Run
```yaml
# Uses default preset: llama3-8b-medical-quick
# Model: meta-llama/Llama-3.1-8B-Instruct
# Dataset: Shekswess/medical_llama3_instruct_dataset_short (500 samples)
# Time: ~30 minutes on single GPU
```
Just select `llama3-8b-medical-quick` preset and execute.

### Example 2: Production Training
```yaml
# Preset: llama3-8b-medical
# Model: meta-llama/Llama-3.1-8B-Instruct
# Dataset: Shekswess/medical_llama3_instruct_dataset (15K samples)
# Training: 3 epochs, 4-bit quantization
# Time: ~4-6 hours on single GPU
```
Select `llama3-8b-medical` preset, configure output bucket, execute.

### Example 3: Custom Dataset from PW Bucket
```yaml
# Upload dataset first:
pw storage cp my-medical-dataset.jsonl my-bucket/datasets/

# Workflow inputs:
dataset_source: bucket
dataset_bucket: my-bucket
dataset_bucket_path: datasets/
base_model_id: mistralai/Mistral-7B-Instruct-v0.2
```

### Example 4: Local Dataset File
```yaml
# Workflow inputs:
dataset_source: local
local_dataset_path: /path/to/cluster/data/medical.jsonl
dataset_format: json
base_model_id: google/gemma-1.1-7b-it
```

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU Memory | 16 GB | 24 GB+ |
| GPU Count | 1x NVIDIA GPU | 2-4x NVIDIA GPUs |
| System RAM | 32 GB | 64 GB+ |
| Storage | 50 GB | 100 GB+ |

**Supported GPUs:** A100, V100, RTX 3090/4090, RTX A5000/A6000, L40

## Output Structure

```
medical-finetune-{RUN_ID}/
├── adapters/           # LoRA adapter weights
├── merged/             # Merged full weights (vLLM ready)
├── logs/finetune.log   # Training logs
└── training.env        # Configuration snapshot
```

## Workflow Parameters (Quick Reference)

| Section | Key Parameters |
|---------|----------------|
| **Infrastructure** | `container.source`, `container.finetune_path`, `container.lfs_repo`/`container.bucket`, `output_bucket` (optional) |
| **Model** | `model_source`, `base_model_id` or `local_model_path`, `model_cache_dir` |
| **Dataset** | `dataset_source` (huggingface/local/bucket), `dataset_name` |
| **Training** | `num_epochs`, `learning_rate`, `micro_batch_size` |
| **LoRA** | `lora_r`, `lora_alpha`, `quantization` (4bit/8bit/none) |
| **Output** | `merge_full_weights`, `push_to_hub` |

## Dataset Sources

| Source | When to Use |
|--------|-------------|
| **HuggingFace Hub** | Public datasets, quick testing |
| **Local File Path** | Custom datasets, offline training |
| **PW Storage Bucket** | Large datasets, team sharing |

Supported formats: `json`, `jsonl`, `csv`, `parquet`, `arrow`, HF dataset directory

## Documentation

- **[Detailed Guide](GUIDE.md)** - Complete parameter reference, troubleshooting, advanced usage
- **[Workflow Builder's Guide](WORKFLOW_BUILDER_GUIDE.md)** - Guide for building PW ACTIVATE workflows
- [LLM-Medical-Finetuning GitHub](https://github.com/Shekswess/LLM-Medical-Finetuning)
- [Parallel Works Documentation](https://parallelworks.com/docs)

## Local Testing

Test workflow components locally without the PW platform:

```bash
# Run all tests
./scripts/test_local.sh

# Test specific component
./scripts/test_local.sh yaml      # Validate YAML syntax
./scripts/test_local.sh dryrun    # Dry run with test environment
```

## Local Execution (For Debugging)

Run the complete fine-tuning workflow locally without PW for development and debugging:

```bash
# Quick test with preset (100 samples, ~10 min)
./scripts/run_local.sh --config llama3-8b-medical-quick --max-samples 100

# Full training with preset
./scripts/run_local.sh --config llama3-8b-medical

# Use local dataset
./scripts/run_local.sh --local-dataset ./data.jsonl

# Custom model and settings
./scripts/run_local.sh --model-id "mistralai/Mistral-7B-v0.1" --epochs 1
```

The local runner handles:
- Container retrieval/build (LFS, bucket, path, or build)
- Dataset downloading/preparation
- Full training execution
- Output organization and logging

Output: `./output/` directory with adapters, merged weights, and logs.

## License

This workflow uses components from the LLM-Medical-Finetuning project. Please refer to the original repository for licensing information.
