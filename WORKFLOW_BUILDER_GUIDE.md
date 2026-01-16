# Parallel Works ACTIVATE Workflow Builder's Guide

This document serves as a post-mortem and guide for building Parallel Works ACTIVATE workflows. It covers the requirements, best practices, and lessons learned from creating the Medical LLM Fine-tuning workflow.

## Table of Contents

- [Workflow Structure](#workflow-structure)
- [YAML Requirements](#yaml-requirements)
- [Input Parameters](#input-parameters)
- [Jobs and Steps](#jobs-and-steps)
- [Scripts and Containerization](#scripts-and-containerization)
- [README Documentation](#readme-documentation)
- [Thumbnail Requirements](#thumbnail-requirements)
- [Input to Execution Translation](#input-to-execution-translation)
- [Rules of Thumb](#rules-of-thumb)
- [Common Patterns](#common-patterns)
- [Testing and Validation](#testing-and-validation)

---

## Workflow Structure

A complete Parallel Works ACTIVATE workflow consists of:

```
activate-workflow-name/
├── workflow.yaml              # Main workflow definition (required)
├── thumbnail.png              # Workflow thumbnail (400x250px recommended)
├── README.md                  # User-facing documentation (required)
├── GUIDE.md                   # Detailed reference (optional but recommended)
├── scripts/                   # Execution scripts
│   ├── run_script.sh          # Main entry point
│   └── helper.py              # Python helper scripts
├── singularity/               # Container definitions
│   └── container.def          # Singularity definition file
└── requirements.txt           # Python dependencies
```

---

## YAML Requirements

### Basic Structure

```yaml
on:
  execute:
    inputs:
      # Input parameters go here

permissions: ["*"]                     # Required for cluster/storage access

jobs:
  job-name:
    steps:
      - name: Step name
        run: |
          echo "Commands here"
        ssh:
          remoteHost: ${{ inputs.cluster.ip }}

configurations:
  preset-name:
    variables:
      param1: value1
```

**Note:** The workflow name is derived from the repository/directory name, not from a `name:` field in the YAML.

### Key YAML Sections

1. **`on.execute.inputs`**: Workflow input parameters (required)
2. **`permissions`**: Always include for cluster/storage access
3. **`jobs`**: Execution steps with SSH to cluster nodes
4. **`configurations`**: Presets that override default inputs (optional)

---

## Input Parameters

### Group Organization

Organize inputs into logical sections using `type: group`:

```yaml
infrastructure:                         # Section name
  type: group
  label: Infrastructure & Compute       # User-facing label
  collapsed: false                       # true = collapsed by default
  items:
    cluster:
      label: GPU Cluster
      type: compute-clusters
      optional: false

    singularity_image:
      label: Singularity Image Path
      type: string
      default: "/home/$USER/pw/singularity/container.sif"
      optional: true
```

### Input Types

| Type | Usage | Example |
|------|-------|---------|
| `string` | Text input | `/path/to/file` |
| `number` | Numeric values | `3.0`, `0.0002` |
| `boolean` | True/False toggle | `true`, `false` |
| `password` | Sensitive data (tokens) | `hf_token` |
| `dropdown` | Fixed options | See below |
| `compute-clusters` | GPU cluster selection | *required* |
| `storage` | Bucket selection | `output_bucket` |

### Dropdown Pattern

```yaml
model_config:
  label: Configuration Preset
  type: dropdown
  default: preset-1                      # Default selected value
  optional: true
  options:
    - value: preset-1                    # Internal value
      label: Preset 1 Display Name       # User-facing label
    - value: preset-2
      label: Preset 2 Display Name
```

**Best Practice**: Use descriptive labels with internal values that are simple identifiers.

### Optional vs Required

```yaml
required_param:
  type: string
  # optional: false                      # Default, can omit

optional_param:
  type: string
  optional: true
  default: ""                            # Always provide default for optional
```

**Rule**: Make parameters optional when reasonable defaults exist. Required fields block workflow execution.

---

## Jobs and Steps

### Job Dependencies

Use `needs` to define execution order:

```yaml
jobs:
  setup:
    steps:
      - name: Initialize
        run: echo "Setup"

  prepare-data:
    needs: [setup]                       # Runs after setup completes
    steps:
      - name: Get data
        run: echo "Preparing"

  build-container:
    needs: [setup]                       # Can run in parallel with prepare-data
    steps:
      - name: Build
        run: echo "Building"

  train:
    needs: [prepare-data, build-container]  # Runs after both complete
    steps:
      - name: Train
        run: echo "Training"
```

### Conditional Execution

```yaml
upload-outputs:
  needs: [train]
  if: ${{ inputs.infrastructure.output_bucket }}  # Only runs if bucket specified
  steps:
    - name: Upload
      run: pw storage cp outputs/* bucket/

cleanup:
  needs: [upload-outputs]
  if: ${{ always }}                     # Always runs, even if previous fails
  steps:
    - name: Summary
      run: echo "Done"
```

### SSH Execution

All steps that run on cluster nodes need `ssh` block:

```yaml
steps:
  - name: Run on cluster
    run: |
      cd $HOME/workspace
      python script.py
    ssh:
      remoteHost: ${{ inputs.cluster.ip }}
    timeout: 24h                        # Optional: max runtime
```

**Important**: Use absolute paths or `$HOME` in remote commands. Relative paths may fail.

---

## Scripts and Containerization

### Singularity Container Definition

```def
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:24.03-py3   # Base image

%labels
    Author Your.Name
    Maintainer YourOrganization
    Description "Container description"

%environment
    export HF_HOME=/workspace/.cache/huggingface
    export PYTHONUNBUFFERED=1

%post
    set -euo pipefail
    apt-get update && apt-get install -y --no-install-recommends git
    pip install -r /opt/workdir/requirements.txt

%runscript
    exec "$@"

%files
    requirements.txt /opt/workdir/requirements.txt
```

### Script Organization

**Main wrapper script** (`run_script.sh`):
```bash
#!/bin/bash
set -euo pipefail

# Load environment variables from workflow
: "${BASE_MODEL_ID:?Required}"
: "${OUTPUT_DIR:?Required}"
: "${NUM_EPOCHS:=3.0}"                  # Default value

# Run Python script
python /workspace/scripts/helper.py \
  --model-id "$BASE_MODEL_ID" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$NUM_EPOCHS"
```

**Python helper script** (`helper.py`):
```python
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    args = parser.parse_args()

    # Training logic here
    print(f"Training {args.model_id} for {args.epochs} epochs")

if __name__ == "__main__":
    main()
```

---

## Input to Execution Translation

### How Inputs Flow to Execution

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Workflow Inputs │────▶│ Training env    │────▶│ Container exec  │
│ (workflow.yaml) │     │ training.env    │     │ run_script.sh   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Pattern: Environment Variable File

**Step 1: Create environment file from inputs**
```yaml
- name: Set training parameters
  run: |
    cat << 'EOF' > "${WORKDIR}/training.env"
    BASE_MODEL_ID=${{ inputs.model_selection.base_model_id }}
    NUM_EPOCHS=${{ inputs.training_hyperparameters.num_epochs }}
    LEARNING_RATE=${{ inputs.training_hyperparameters.learning_rate }}
    EOF
  ssh:
    remoteHost: ${{ inputs.cluster.ip }}
```

**Step 2: Source environment file in execution**
```yaml
- name: Execute training
  run: |
    source "${WORKDIR}/training.env"

    singularity exec \
      --nv \
      --bind "${OUTPUT_DIR}:/workspace" \
      "${SIF_PATH}" \
      bash /workspace/scripts/run_script.sh
  ssh:
    remoteHost: ${{ inputs.cluster.ip }}
```

**Step 3: Script uses environment variables**
```bash
# In run_script.sh
: "${BASE_MODEL_ID:?Required}"
: "${NUM_EPOCHS:=3.0}"
```

### Conditional Hidden Fields Pattern

Show/hide fields based on previous selections:

```yaml
dataset_source:
  label: Dataset Source
  type: dropdown
  options:
    - value: huggingface
      label: HuggingFace Hub
    - value: local
      label: Local File Path

# Only show for HuggingFace source
dataset_name:
  label: Dataset Name
  type: string
  hidden: ${{ inputs.dataset_config.dataset_source != 'huggingface' }}

# Only show for local source
local_path:
  label: Local Path
  type: string
  hidden: ${{ inputs.dataset_config.dataset_source != 'local' }}
```

---

## Rules of Thumb

### Input Design

1. **Provide sensible defaults** - Users should only need to change what's necessary
2. **Use grouped sections** - Keep related parameters together (max 6-8 items per group)
3. **Make optional truly optional** - Workflow should work without optional fields
4. **Use dropdowns for fixed options** - Better UX than text input with validation
5. **Label with "Optional" in name** - Clear when a field is optional

### Reference Syntax

```yaml
# Top-level input (not in group)
${{ inputs.cluster.ip }}

# Grouped input
${{ inputs.infrastructure.singularity_image }}
${{ inputs.dataset_config.dataset_name }}
${{ inputs.model_selection.base_model_id }}
```

**Rule**: Reference pattern is always `inputs.<group_name>.<item_name>` for grouped inputs.

### Environment Variables

```bash
# Always use quotes for expansion
"${VARIABLE:?Required}"                 # Fail if not set
"${VARIABLE:=default}"                   # Use default if not set

# Never use unquoted variables
$VARIABLE                                # BAD: May expand to nothing
```

### File Paths

```yaml
# Use $HOME instead of ~ in workflow
default: "/home/$USER/pw/singularity/container.sif"  # GOOD
default: "~/pw/singularity/container.sif"            # BAD

# In scripts, both work
cd ~/pw/workspace                        # OK
cd "${HOME}/pw/workspace"                # Better (explicit)
```

### Conditional Job Execution

```yaml
# Only run if condition is truthy
if: ${{ inputs.field_name }}             # Runs if value exists and is not ""

# Always run (even if previous job fails)
if: ${{ always }}
```

---

## Common Patterns

### Pattern 1: Auto-Build Container

**In workflow.yaml:**
```yaml
build-container:
  steps:
    - name: Build Singularity container
      run: |
        CONTAINER_DIR="${HOME}/pw/singularity"
        mkdir -p "${CONTAINER_DIR}"

        if [ -n "${{ inputs.infrastructure.singularity_image }}" ]; then
          # Use provided SIF path
          export SIF_PATH="${{ inputs.infrastructure.singularity_image }}"
          if [ ! -f "${SIF_PATH}" ]; then
            echo "Error: SIF not found at ${SIF_PATH}"
            exit 1
          fi
        else
          # Build from definition
          DEF_FILE="${HOME}/pw/workflow/singularity/container.def"
          SIF_PATH="${CONTAINER_DIR}/container.sif"

          if [ ! -f "${SIF_PATH}" ]; then
            singularity build "${SIF_PATH}" "${DEF_FILE}"
          fi
        fi

        echo "SIF_PATH=${SIF_PATH}" >> $GITHUB_ENV
      ssh:
        remoteHost: ${{ inputs.cluster.ip }}
```

**Separate build script (optional):**

Provide a standalone build script for users to pre-build containers locally:

```bash
#!/bin/bash
# scripts/build_container.sh
set -euo pipefail

DEF_FILE="$(dirname "$0")/../singularity/container.def"
OUTPUT_DIR="${HOME}/pw/singularity"
OUTPUT_FILE="${OUTPUT_DIR}/container.sif"

mkdir -p "${OUTPUT_DIR}"

if command -v singularity &> /dev/null; then
    singularity build "${OUTPUT_FILE}" "${DEF_FILE}"
elif command -v apptainer &> /dev/null; then
    apptainer build "${OUTPUT_FILE}" "${DEF_FILE}"
else
    echo "Error: Neither singularity nor apptainer found"
    exit 1
fi

echo "Built: ${OUTPUT_FILE}"
```

**Benefits:**
- Users can pre-build and test containers locally
- Faster workflow execution (no build time during run)
- Enables local development and testing

### Pattern 2: Dataset Preparation

```yaml
prepare-dataset:
  needs: [setup]
  steps:
    - name: Prepare dataset based on source
      run: |
        OUTPUT_DIR="${HOME}/pw/outputs/workflow-${PW_RUN_ID}"
        DATASET_DIR="${OUTPUT_DIR}/dataset"
        mkdir -p "${DATASET_DIR}"

        SOURCE="${{ inputs.dataset_config.dataset_source }}"

        if [ "${SOURCE}" == "bucket" ]; then
          # Download from PW storage
          pw storage cp -r "${BUCKET}/${PATH}" "${DATASET_DIR}/"
        elif [ "${SOURCE}" == "local" ]; then
          # Copy from local filesystem
          cp "${LOCAL_PATH}" "${DATASET_DIR}/"
        fi
        # HuggingFace: downloaded during training
      ssh:
        remoteHost: ${{ inputs.cluster.ip }}
```

### Pattern 3: Output Upload (Optional)

```yaml
upload-outputs:
  needs: [train]
  if: ${{ inputs.infrastructure.output_bucket }}  # Only if bucket specified
  steps:
    - name: Upload to bucket
      run: |
        OUTPUT_DIR="${HOME}/pw/outputs/workflow-${PW_RUN_ID}"
        BUCKET="${{ inputs.infrastructure.output_bucket }}"

        pw storage cp -r "${OUTPUT_DIR}/results" "${BUCKET}/workflow-${PW_RUN_ID}/"
      ssh:
        remoteHost: ${{ inputs.cluster.ip }}
```

---

## README Documentation

### Essential README Sections

```markdown
# Workflow Name

Brief one-line description of what the workflow does.

## Quick Start

### 1. Run in Parallel Works

1. Step one
2. Step two
3. Click **Execute**

### 2. Use Output

```bash
# Example command
```

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU Memory | 16 GB | 24 GB+ |

## Output Structure

```
output-directory/
├── results/
└── logs/
```

## Workflow Parameters (Quick Reference)

| Section | Key Parameters |
|---------|----------------|

## Documentation

- **[Detailed Guide](GUIDE.md)** - Complete reference
```

### README Best Practices

1. **Keep it short** - Under 150 lines for main README
2. **Quick Start first** - Users want to run immediately
3. **Turnkey examples** - Show common use cases
4. **Link to detailed guide** - Put extensive docs in separate file
5. **Requirements table** - Clear resource needs

---

## Thumbnail Requirements

### Specifications

- **Format**: PNG
- **Recommended size**: 400x250 pixels
- **File size**: Under 100KB preferred
- **Naming**: `thumbnail.png` in workflow root

### Design Guidelines

1. **Show workflow purpose** - Visual representation of what workflow does
2. **Include branding** - Organization logo if applicable
3. **Simple text** - Workflow name, short subtitle
4. **High contrast** - Works on light/dark backgrounds

### Creating Thumbnails

```bash
# Using ImageMagick
convert -size 400x250 xc:#1a1a2e \
  -font Helvetica-Bold -pointsize 32 -fill white \
  -gravity center -annotate +0-30 "Workflow Name" \
  -pointsize 16 -annotate +0+20 "Short description" \
  thumbnail.png

# Or use online tools like Canva, Figma
```

---

## Testing and Validation

### Local Testing Script

A `test_local.sh` script is provided for testing workflow components locally without the PW platform:

```bash
./scripts/test_local.sh [test_type]

# Test types:
#   all       - Run all tests (default)
#   yaml      - Validate YAML syntax
#   config    - Test configuration presets
#   scripts   - Test shell scripts
#   container - Test Singularity container
#   build     - Test build script
#   dryrun    - Dry run with fake training
#   readme    - Test README documentation

# Examples:
./scripts/test_local.sh              # Run all tests
./scripts/test_local.sh yaml         # Test YAML only
./scripts/test_local.sh dryrun       # Run dry run test
```

The test script validates:
- YAML syntax (using Python or basic checks)
- Configuration presets and required variables
- Shell script syntax and executability
- Python script syntax
- Container definition structure
- Requirements.txt for essential packages
- Environment variable handling (dry run)

### Local Execution Script

A `run_local.sh` script is provided for running the complete workflow locally for debugging:

```bash
./scripts/run_local.sh [--config PRESET] [OPTIONS]

# Examples:
./scripts/run_local.sh --config llama3-8b-medical-quick --max-samples 100
./scripts/run_local.sh --local-dataset ./data.jsonl
./scripts/run_local.sh --model-id "mistralai/Mistral-7B-v0.1" --epochs 1
```

The local execution script:
- Validates Singularity/Apptainer installation
- Checks GPU availability
- Builds or uses existing Singularity container
- Downloads/prepare datasets based on source
- Executes full training with all workflow parameters
- Outputs results to local directory with logs

**Key features:**
- Full workflow execution without PW platform
- Same configuration presets as PW workflow
- Supports all dataset sources (HuggingFace, local)
- Generates same outputs as PW workflow
- Real-time logging for debugging

### Manual Local Testing

For manual testing of individual components:

```bash
# 1. Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('workflow.yaml'))"

# 2. Test bash scripts syntax
bash -n scripts/run_finetune.sh

# 3. Test Python script syntax
python3 -m py_compile scripts/pw_finetune.py

# 4. Check container definition
singularity singularity/finetune.def --dry-run

# 5. Test environment file sourcing
cat > test.env << 'EOF'
VAR1=value1
VAR2=value2
EOF
source test.env
echo "${VAR1}"
```

### Pre-Flight Checklist

- [ ] All required fields marked correctly
- [ ] Optional fields have defaults
- [ ] Group references use correct syntax
- [ ] Dropdown options have values and labels
- [ ] Conditional hidden fields reference correct inputs
- [ ] SSH blocks include `remoteHost`
- [ ] File paths use absolute paths or `$HOME`
- [ ] Environment variables use quotes: `"${VAR}"`
- [ ] Jobs have proper `needs` dependencies
- [ ] Optional uploads use `if` condition

### Validation Command

```bash
# If PW provides a YAML validator
pw workflow validate workflow.yaml

# Or check syntax
python -c "import yaml; yaml.safe_load(open('workflow.yaml'))"
```

---

## Lessons Learned

### What Worked Well

1. **Grouped inputs** - Much cleaner UI with 6 logical sections
2. **Conditional hidden fields** - Reduces confusion, only shows relevant options
3. **Auto-build containers** - Users don't need to manually build SIF files
4. **Optional bucket upload** - Quick tests don't require storage setup
5. **Configuration presets** - Dropdown with descriptive labels better than text input

### Common Pitfalls

1. **Wrong group references** - `inputs.singularity_image` vs `inputs.infrastructure.singularity_image`
2. **Missing SSH block** - Steps run on local machine, not cluster
3. **Unquoted variables** - Bash expansion issues
4. **Required optional fields** - Workflow fails when user doesn't provide value
5. **File path issues** - `~` doesn't expand in workflow, use `/home/$USER`
6. **Conditional logic errors** - Hidden fields still referenced in jobs

### Debugging Tips

```yaml
# Add debug step to check environment
- name: Debug environment
  run: |
    echo "Cluster: ${{ inputs.cluster.ip }}"
    echo "Model: ${{ inputs.model_selection.base_model_id }}"
    env | sort
  ssh:
    remoteHost: ${{ inputs.cluster.ip }}
```

---

## Quick Reference Card

### Input Reference Pattern

```
Top-level:       inputs.field_name
Grouped:         inputs.group_name.field_name
```

### Common Types

```
String:          type: string
Number:          type: number
Boolean:         type: boolean
Dropdown:        type: dropdown
Password:        type: password
Cluster:         type: compute-clusters
Storage:         type: storage
```

### Job Patterns

```
Sequential:      needs: [previous-job]
Parallel:        needs: [setup]  (multiple jobs)
Conditional:     if: ${{ inputs.field }}
Always run:      if: ${{ always }}
```

### Environment Variables

```
Required:        "${VAR:?Required}"
Default:         "${VAR:=default}"
Optional:        "${VAR:-}"
```

---

## Additional Resources

- [Parallel Works Documentation](https://parallelworks.com/docs)
- [Singularity User Guide](https://sylabs.io/guides/3.7/user-guide/)
- [YAML Syntax Reference](https://yaml.org/spec/1.2/spec.html)
