# System Architecture — Medical Term Simplifier

> Fine-tuning Mistral-7B with LoRA to translate clinical jargon into patient-friendly plain English.

---

## Table of Contents

- [1. High-Level Overview](#1-high-level-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Configuration Layer](#3-configuration-layer)
- [4. Training Pipeline — `training_pipeline.ipynb`](#4-training-pipeline--training_pipelineipynb)
  - [4.1 Cell 1 — Environment Setup](#41-cell-1--environment-setup)
  - [4.2 Cell 2 — File Upload Check](#42-cell-2--file-upload-check)
  - [4.3 Cell 3 — `main()`](#43-cell-3--main)
    - [load_config()](#load_config)
    - [load_local_dataset()](#load_local_dataset)
    - [build_dataset()](#build_dataset)
    - [_extract_row()](#_extract_row)
    - [load_source()](#load_source)
    - [load_model()](#load_model)
    - [prepare_splits()](#prepare_splits)
    - [build_trainer() + run_training()](#build_trainer--run_training)
    - [export_model()](#export_model)
- [5. Demo Pipeline — `demo.ipynb`](#5-demo-pipeline--demoipynb)
  - [5.1 Cell 1 — Install](#51-cell-1--install)
  - [5.2 Cell 2 — `load_demo()`](#52-cell-2--load_demo)
  - [5.3 Cell 3 — `main()` + `demo()`](#53-cell-3--main--demo)
- [6. Data Sources](#6-data-sources)
- [7. Data Flow](#7-data-flow)
- [8. Model Architecture](#8-model-architecture)
- [9. YAML → API Mapping](#9-yaml--api-mapping)
- [10. Function Dependency Map](#10-function-dependency-map)
- [11. Error Handling Strategy](#11-error-handling-strategy)
- [12. Key Design Decisions](#12-key-design-decisions)

---

## 1. High-Level Overview

The system is split into two fully independent notebooks sharing one config file and one trained model artefact on HuggingFace Hub.

```
┌─────────────────────────────────────────────────────────────────┐
│                         config.yaml                             │
│   model · lora · dataset · training · inference · export        │
│         Single source of truth — both notebooks read from here  │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
┌──────────────────┐    ┌──────────────────────────────────────┐
│ training_        │    │ demo.ipynb                           │
│ pipeline.ipynb   │    │                                      │
│                  │    │  Cell 1: install                     │
│ Cell 1: install  │    │  Cell 2: load_demo()                 │
│ Cell 2: check    │───▶│           reads config               │
│         files    │    │           loads model FROM HUB       │
│ Cell 3: main()   │    │  Cell 3: main()                      │
│   full pipeline  │    │           demo() × 5 showcase terms  │
│   push to Hub    │    │                                      │
└──────────────────┘    └──────────────────────────────────────┘
          │                              ▲
          │   pushes adapter             │ pulls adapter
          ▼                              │
┌─────────────────────────────────────────────────────────────────┐
│                    HuggingFace Hub                              │
│           Akhilesh-2308/medical-term-simplifier                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key principle:** Training and inference are completely decoupled. You train once, push to Hub, and all future runs pull the trained adapter from Hub without re-training.

---

## 2. Repository Structure

```
medical-term-simplifier/
│
├── training_pipeline.ipynb   # Run once to train. Cell 3 = main()
├── demo.ipynb                # Run during presentation. Cell 3 = main()
├── config.yaml               # All hyperparameters — YAML keys match API params exactly
├── med_dataset.json          # Local curated dataset (upload to Colab, not in repo)
└── README.md                 # GitHub documentation
```

---

## 3. Configuration Layer

`config.yaml` is the single source of truth for the entire system. Every key is named to **exactly match** the Python API parameter it maps to, enabling direct `**` unpacking with zero remapping.

```
config.yaml
│
├── model          → FastLanguageModel.from_pretrained(**model_cfg)
├── lora           → FastLanguageModel.get_peft_model(model, **lora_cfg)
├── dataset        → build_dataset(dataset_cfg, ...)
├── training       → TrainingArguments(**training_cfg)
├── inference      → model.generate(**inference_cfg, ...)
└── export         → push_to_hub(hub_model_id)
```

### Config sections and their values

| Section | Key parameters | Purpose |
|---------|---------------|---------|
| `model` | `model_name`, `max_seq_length`, `load_in_4bit`, `dtype` | Base model loading |
| `lora` | `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `use_gradient_checkpointing` | LoRA adapter setup |
| `dataset` | `local_path`, `sources[]`, `total_cap`, `val_split`, `seed`, `instruction_variants` | Data pipeline config |
| `training` | `max_steps`, `learning_rate`, `lr_scheduler_type`, `per_device_train_batch_size`, `optim` | Training loop |
| `inference` | `max_new_tokens`, `do_sample`, `temperature`, `repetition_penalty` | Generation parameters |
| `export` | `hub_model_id`, `hub_token_env`, `push_to_hub`, `lora_dir` | Model export |

---

## 4. Training Pipeline — `training_pipeline.ipynb`

### 4.1 Cell 1 — Environment Setup

**Responsibility:** Silence warnings, set memory flags, install all ML dependencies.

**Environment variables set:**

| Variable | Value | Reason |
|----------|-------|--------|
| `TOKENIZERS_PARALLELISM` | `false` | Prevents deadlock in Colab's tokeniser workers |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Prevents GPU OOM by allowing dynamic memory segments |

**Dependencies installed:** `unsloth`, `pyyaml`, `datasets`, `evaluate`, `rouge_score`, `huggingface_hub`

---

### 4.2 Cell 2 — File Upload Check

**Responsibility:** Verify that `config.yaml` and `med_dataset.json` are present before running the expensive Cell 3.

**Logic:** Checks `pathlib.Path(fname).exists()` for each required file and prints a status message. Fails gracefully — does not crash, just warns.

---

### 4.3 Cell 3 — `main()`

The entire training pipeline runs inside a single `main()` call. All functions are defined above it in the same cell and called sequentially.

```
main()
  │
  ├─ 1/6  load_config()
  ├─ 2/6  load_local_dataset()
  ├─ 3/6  build_dataset()
  │         ├─ load_source()  [called per enabled source]
  │         │    └─ _extract_row()  [called per row]
  │         └─ returns DatasetDict
  ├─ 4/6  load_model()
  ├─ 5/6  prepare_splits() → build_trainer() → run_training()
  └─ 6/6  export_model()
```

---

#### `load_config()`

```
Inputs  : config_path (str, default "config.yaml")
Outputs : cfg (dict) — full parsed YAML
Raises  : FileNotFoundError — if file missing
          KeyError          — if required section absent
```

**What it does:**

1. Wraps the path in `pathlib.Path` and calls `.exists()`
2. Opens the file and calls `yaml.safe_load()` — parses into nested Python dict
3. Validates all six required sections are present: `model`, `lora`, `dataset`, `training`, `inference`, `export`
4. Returns `cfg`

**Called from:** `main()` — once, at the top. All other functions receive their section of `cfg` as an argument, they never call `load_config` themselves.

---

#### `load_local_dataset()`

```
Inputs  : dataset_cfg (dict) — cfg["dataset"]
Outputs : list of dicts (one dict per sample)
Raises  : FileNotFoundError — if med_dataset.json missing
```

**What it does:**

1. Reads `local_path` from `dataset_cfg` — never hard-coded
2. Checks file existence with `pathlib.Path(local_path).exists()`
3. Opens and parses with `json.load()`
4. Returns raw list

**Called from:** `main()` — once, result passed into `build_dataset()`

---

#### `build_dataset()`

```
Inputs  : dataset_cfg (dict), local_rows (list)
Outputs : DatasetDict with keys "train" and "validation"
```

**What it does:**

1. Seeds `random` with `dataset_cfg["seed"]` (42) for reproducibility
2. Iterates all `dataset_cfg["sources"]` — skips any with `enabled: false`
3. Calls `load_source()` for each enabled source inside a `try/except` — one failing source does not abort the pipeline
4. `random.shuffle(all_rows)` — mixes all sources together
5. Slices to `total_cap` (5,000)
6. Calculates split index: `cut = int(len(all_rows) * (1 - val_split))` → 95% train / 5% val
7. Returns `DatasetDict({"train": ..., "validation": ...})`

**Called from:** `main()` — once. Result passed into `prepare_splits()`

---

#### `_extract_row()`

```
Inputs  : row (dict), src_cfg (dict) — one source entry from YAML
Outputs : tuple(question: str, answer: str)
          Returns ("", "") when data is missing — caller skips the row
```

**What it does:** Handles three different answer storage layouts, all driven by YAML config keys.

| Layout | Used by | YAML keys read |
|--------|---------|----------------|
| Direct column | ChatDoctor, Flashcards, WikiDoc, MedQuad, local | `input_col`, `output_col` |
| Nested dict/list | PubMedQA | `fallback_context_col`, `fallback_context_key` |
| MCQ index → option | MedMCQA | `correct_idx_col`, `option_cols` |

**PubMedQA fallback logic:** `long_answer` is often empty. The function falls back to joining passages from `row["context"]["contexts"]`.

**MedMCQA MCQ logic:** `cop` column holds an integer (0–3). `option_cols[int(cop)]` resolves to the column name (e.g. `"opc"`). That column's value becomes the answer.

**Called from:** `load_source()` — once per row

---

#### `load_source()`

```
Inputs  : src (dict), variants (list of str), local_rows (list)
Outputs : list of normalised dicts [{instruction, input, output}, ...]
```

**What it does:**

1. Reads `max_samples` limit from `src`
2. Branches on `src["type"]`:
   - `"local"` → uses the already-loaded `local_rows` directly
   - `"huggingface"` → calls `load_dataset(src["path"], **kwargs)`, selects up to `limit` rows, converts each `Example` to `dict`
3. For each row calls `_extract_row(row, src)` — skips rows where q or a is empty
4. Assigns a random instruction variant: `random.choice(variants)`
5. Returns list of `{"instruction": ..., "input": q, "output": a}`

**Called from:** `build_dataset()` — once per enabled source

---

#### `load_model()`

```
Inputs  : model_cfg (dict), lora_cfg (dict)
Outputs : (model, tokenizer)
```

**What it does:**

1. Calls `FastLanguageModel.from_pretrained(**model_cfg)` — downloads Mistral-7B at 4-bit quantisation (~6 GB GPU RAM)
2. Calls `FastLanguageModel.get_peft_model(model, **lora_cfg)` — attaches LoRA adapters to `q_proj`, `k_proj`, `v_proj`, `o_proj` in each attention layer
3. Freezes all base model weights — only adapter matrices require gradients
4. Prints trainable parameter count and percentage

**LoRA target layers:**

```
Mistral-7B Attention Block
├── q_proj  ← LoRA adapter attached
├── k_proj  ← LoRA adapter attached
├── v_proj  ← LoRA adapter attached
└── o_proj  ← LoRA adapter attached
```

**Called from:** `main()` — once. Returns `(model, tokenizer)` used in all subsequent steps.

---

#### `prepare_splits()`

```
Inputs  : splits (DatasetDict), tokenizer, model_cfg (dict)
Outputs : (train_ds, val_ds) — formatted and filtered HuggingFace Datasets
```

**What it does:**

1. Calls `build_prompt_formatters()` to create two closures: `format_row` and `filter_long`
2. `format_row` converts each `{instruction, input, output}` dict into a single `text` string using the Alpaca template:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}{eos_token}
```

3. `filter_long` tokenises the text and returns `True` only if `len(token_ids) <= max_seq_length` (512)
4. Applies `.map(format_row, remove_columns=[...])` to drop the original three columns
5. Applies `.filter(filter_long)` to remove sequences that would be silently truncated

**Called from:** `main()` — once. Returns `(train_ds, val_ds)` passed into `build_trainer()`

---

#### `build_trainer()` + `run_training()`

```
build_trainer()
  Inputs  : model, tokenizer, train_ds, val_ds, model_cfg, training_cfg
  Outputs : SFTTrainer instance

run_training()
  Inputs  : trainer (SFTTrainer)
  Outputs : metrics dict
```

**`build_trainer()` key behaviour:** `TrainingArguments(**training_cfg)` — every training parameter comes from YAML with zero manual key listing. Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps` = `2 × 4 = 8`.

**Training configuration:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_steps` | 500 | ~20–40 min on T4 |
| `learning_rate` | 2e-4 | Standard for LoRA |
| `lr_scheduler_type` | cosine | Gradual LR decay |
| `per_device_train_batch_size` | 2 | |
| `gradient_accumulation_steps` | 4 | Effective batch = 8 |
| `optim` | adamw_8bit | Memory-efficient Adam |
| `fp16` | true | Half-precision for speed |
| `warmup_steps` | 20 | Prevent early instability |

**`run_training()` calls `trainer.train()`** — handles: batch feed → cross-entropy loss → backprop → LoRA weight update. Prints runtime, samples/sec, and final loss.

**Called from:** `main()` — sequentially, `build_trainer()` then `run_training()`

---

#### `export_model()`

```
Inputs  : model, tokenizer, export_cfg (dict)
Outputs : None (side effects: files on disk + Hub upload)
```

**What it does:**

1. Reads `hub_token_env` from `export_cfg` (default `"HF_TOKEN"`)
2. Fetches the token from Colab secrets: `userdata.get(token_env)` — never hard-coded
3. Sets `os.environ[token_env]` and calls `login(token=hf_token)`
4. Calls `model.push_to_hub(hub_id)` and `tokenizer.push_to_hub(hub_id)`
5. Only the LoRA adapter weights are uploaded — not the full 7B base model

**Called from:** `main()` — once, at the end

---

## 5. Demo Pipeline — `demo.ipynb`

### 5.1 Cell 1 — Install

Same environment setup as training Cell 1, but only installs `unsloth` and `pyyaml`. No training libraries needed.

---

### 5.2 Cell 2 — `load_demo()`

```
Inputs  : config_path (str, default "config.yaml")
Outputs : (model, tokenizer, cfg)
```

**What it does:**

1. Reads and parses `config.yaml`
2. Fetches `HF_TOKEN` from Colab secrets and calls `login()`
3. Reads `hub_model_id` from `export_cfg` — loads the **already-trained** adapter from Hub:

```python
FastLanguageModel.from_pretrained(
    model_name     = hub_id,        # ← Akhilesh-2308/medical-term-simplifier
    max_seq_length = model_cfg["max_seq_length"],
    dtype          = model_cfg["dtype"],
    load_in_4bit   = model_cfg["load_in_4bit"],
)
```

4. Calls `FastLanguageModel.for_inference(model)` — enables Unsloth's optimised inference mode (2× faster token generation)

**No training happens in this notebook.**

---

### 5.3 Cell 3 — `main()` + `demo()`

#### `demo()`

```
Inputs  : term (str), instruction (str, optional)
Outputs : response (str) — also prints formatted box
```

**What it does:**

1. Uses `instruction` from `dataset_cfg["instruction_variants"][0]` if none provided
2. Builds Alpaca prompt — stops at `### Response:\n` (no answer included)
3. Tokenises with `return_tensors="pt"` and moves to CUDA
4. Records `input_len = inputs["input_ids"].shape[1]` — used to slice out only the generated tokens
5. Calls `model.generate(**inference_cfg, ...)` — all generation params from YAML
6. Decodes `outputs[0][input_len:]` — only the new tokens, not the prompt
7. Formats and prints the result in a bordered box

#### `main()`

Iterates `SHOWCASE_TERMS` list (5 terms) and calls `demo()` on each. Prints a counter `[1/5]`, `[2/5]` etc. so the audience can follow along.

**Showcase terms:**

```python
SHOWCASE_TERMS = [
    "Myocardial infarction",
    "Pulmonary embolism",
    "Sepsis",
    "Hyperglycemia",
    "Atrial fibrillation",
]
```

---

## 6. Data Sources

| Name | Type | HuggingFace path | Samples | Answer layout |
|------|------|------------------|---------|---------------|
| chatdoctor | huggingface | `lavita/ChatDoctor-HealthCareMagic-100k` | 1,500 | Direct column (`output`) |
| pubmedqa | huggingface | `pubmed_qa` (config: `pqa_labeled`) | 700 | Nested — `context.contexts[]` |
| medmcqa | huggingface | `medmcqa` | 700 | MCQ index — `cop` → `opa/opb/opc/opd` |
| flashcards | huggingface | `medalpaca/medical_meadow_medical_flashcards` | 500 | Direct column (`output`) |
| wikidoc | huggingface | `medalpaca/medical_meadow_wikidoc_patient_information` | 800 | Direct column (`output`) |
| medquad | huggingface | `keivalya/MedQuad-MedicalQnADataset` | 800 | Direct column (`Answer`) |
| local_curated | local | `med_dataset.json` | all | Direct column (`output`) |

**Total cap:** 5,000 samples after shuffle
**Split:** 95% train / 5% validation
**Seed:** 42 (reproducible)

---

## 7. Data Flow

```
med_dataset.json          HuggingFace datasets
       │                        │
       ▼                        ▼
load_local_dataset()      load_dataset(path, **kwargs)
       │                        │
       └────────┬───────────────┘
                ▼
         load_source()
         [per source]
                │
                ▼
         _extract_row()
         [per row — handles direct / nested / MCQ]
                │
                ▼
         normalised row:
         { instruction, input, output }
                │
                ▼
         build_dataset()
         shuffle → cap 5000 → 95/5 split
                │
                ▼
         DatasetDict
         { "train": ..., "validation": ... }
                │
                ▼
         prepare_splits()
         Alpaca format → filter > 512 tokens
                │
                ▼
         train_ds   val_ds
                │
                ▼
         SFTTrainer.train()
```

---

## 8. Model Architecture

### Base model

```
Mistral-7B (unsloth/mistral-7b-bnb-4bit)
├── 32 transformer layers
├── 4-bit quantised weights (~6 GB GPU RAM vs ~28 GB at full precision)
├── max_seq_length: 512 tokens
└── ALL base weights FROZEN during fine-tuning
```

### LoRA adapters (trained layers only)

```
Each of 32 attention layers:
├── q_proj  → LoRA(r=16, alpha=16)  ← TRAINABLE
├── k_proj  → LoRA(r=16, alpha=16)  ← TRAINABLE
├── v_proj  → LoRA(r=16, alpha=16)  ← TRAINABLE
└── o_proj  → LoRA(r=16, alpha=16)  ← TRAINABLE

Total trainable: ~0.5-1% of 7B parameters
Adapter size on disk: ~300–500 MB
```

### Inference path (demo.ipynb)

```
User input: "Bradycardia"
      │
      ▼
Alpaca prompt builder
      │
      ▼
tokenizer(prompt, return_tensors="pt").to("cuda")
      │
      ▼
model.generate(
    input_ids, attention_mask,
    **inference_cfg,           ← max_new_tokens=200, temperature=1.0
    eos_token_id, use_cache=True
)
      │
      ▼
tokenizer.decode(outputs[0][input_len:])
      │
      ▼
format_output() → print plain-English explanation
```

---

## 9. YAML → API Mapping

The `**` unpacking pattern works because every YAML key matches its target API parameter name exactly.

```
config.yaml                      Python call
─────────────────────────────────────────────────────────────────────
model:
  model_name: "..."         ─┐
  max_seq_length: 512        │  FastLanguageModel.from_pretrained(
  load_in_4bit: true         ├─    model_name     = "...",
  dtype: null               ─┘    max_seq_length = 512,
                                   load_in_4bit   = True,
                                   dtype          = None
                                )

lora:
  r: 16                     ─┐
  lora_alpha: 16             │  FastLanguageModel.get_peft_model(
  lora_dropout: 0.05         │      model,
  target_modules: [...]      ├─    r              = 16,
  bias: "none"               │    lora_alpha      = 16,
  use_gradient_              │    lora_dropout    = 0.05,
    checkpointing: "unsloth"─┘    ...
                                )

training:
  output_dir: "..."         ─┐
  num_train_epochs: 1        │
  max_steps: 500             │  TrainingArguments(
  learning_rate: 2e-4        ├─    output_dir        = "...",
  lr_scheduler_type: cosine  │    num_train_epochs  = 1,
  ...                       ─┘    ...
                                )

inference:
  max_new_tokens: 200        ─┐
  do_sample: false            │  model.generate(
  temperature: 1.0            ├─    input_ids = ...,
  repetition_penalty: 1.1    ─┘    **inference_cfg
                                )
```

---

## 10. Function Dependency Map

```
main()  [training_pipeline.ipynb Cell 3]
├── load_config()
│     └── yaml.safe_load()
│
├── load_local_dataset(dataset_cfg)
│     └── json.load()
│
├── build_dataset(dataset_cfg, local_rows)
│     ├── load_source(src, variants, local_rows)  [× N sources]
│     │     ├── load_dataset()          [if huggingface type]
│     │     └── _extract_row(row, src)  [× M rows]
│     └── DatasetDict()
│
├── load_model(model_cfg, lora_cfg)
│     ├── FastLanguageModel.from_pretrained(**model_cfg)
│     └── FastLanguageModel.get_peft_model(model, **lora_cfg)
│
├── prepare_splits(splits, tokenizer, model_cfg)
│     ├── build_prompt_formatters()
│     │     ├── format_row()   [closure]
│     │     └── filter_long()  [closure]
│     ├── Dataset.map(format_row)
│     └── Dataset.filter(filter_long)
│
├── build_trainer(model, tokenizer, train_ds, val_ds, model_cfg, training_cfg)
│     └── SFTTrainer(args=TrainingArguments(**training_cfg))
│
├── run_training(trainer)
│     └── trainer.train()
│
└── export_model(model, tokenizer, export_cfg)
      ├── userdata.get(token_env)
      ├── login(token)
      └── model.push_to_hub(hub_id)


main()  [demo.ipynb Cell 3]
├── demo(term, instruction)         [× 5 showcase terms]
│     ├── tokenizer(prompt)
│     ├── model.generate(**inference_cfg)
│     └── tokenizer.decode(outputs[0][input_len:])
└── print formatted results


load_demo()  [demo.ipynb Cell 2]
├── yaml.safe_load()
├── login(token)
└── FastLanguageModel.from_pretrained(model_name=hub_id)
```

---

## 11. Error Handling Strategy

| Location | What can fail | How it is handled |
|----------|---------------|-------------------|
| `load_config()` | File not found | `raise FileNotFoundError` with upload instructions |
| `load_config()` | Missing YAML section | `raise KeyError` listing missing sections |
| `load_local_dataset()` | `med_dataset.json` missing | `raise FileNotFoundError` with upload instructions |
| `build_dataset()` | One HF source fails to download | `try/except Exception` — prints warning, continues to next source |
| `load_source()` | Empty dataset returned | Returns `[]` with warning print — does not crash |
| `_extract_row()` | Bad MCQ index | `try/except (IndexError, ValueError, TypeError)` — skips row |
| `_extract_row()` | `nan` / `None` string values | Explicit `not in ("nan", "none", "")` check before using answer |
| `prepare_splits()` | Sequences too long | `.filter(filter_long)` removes them rather than silently truncating |
| `export_model()` | Hub push fails | Propagates exception — failure is visible, not silent |

---

## 12. Key Design Decisions

### 1 — YAML keys match API parameter names exactly
Eliminates all manual key remapping. Adding a new hyperparameter to YAML makes it available immediately without Python changes. Changing a value in YAML is the only thing needed to change behaviour.

### 2 — Two-notebook separation
Training and inference are completely decoupled. The demo notebook has no knowledge of how the model was trained — it only knows the Hub ID. This means the presentation can run even if the training notebook is unavailable.

### 3 — Single `main()` call per notebook
A viewer of either notebook sees one entry point. All internal complexity is hidden inside functions defined above `main()`. The call chain is explicit in the progress printout.

### 4 — `_extract_row()` is config-driven with zero hard-coding
All column names, fallback keys, and MCQ option lists come from YAML. Adding a new dataset only requires a new YAML entry — no Python function changes.

### 5 — Fail-fast on config, fail-safe on data
Config errors (missing file, missing section) raise immediately so you know before any compute runs. Data source failures are caught and warned so one broken dataset does not abort the entire pipeline.

### 6 — `try/except` per source, not around the whole loop
Each source is wrapped independently. If PubMedQA is temporarily unavailable, the other five sources still load and training proceeds with reduced data rather than crashing entirely.
