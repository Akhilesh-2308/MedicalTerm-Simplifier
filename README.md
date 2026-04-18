# Medical Term Simplifier

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-Mistral--7B-blueviolet)
![Method](https://img.shields.io/badge/Fine--tuning-LoRA-orange)
![Framework](https://img.shields.io/badge/Framework-Unsloth-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model%20Hub-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Use Case](#3-use-case)
- [4. How We Got the Idea](#4-how-we-got-the-idea)
- [5. Architecture](#5-architecture)
- [6. What Have We Done](#6-what-have-we-done)
  - [6.1 Data Collection](#61-data-collection)
  - [6.2 Dataset Pipeline](#62-dataset-pipeline)
  - [6.3 Model Fine-Tuning](#63-model-fine-tuning)
- [7. Results](#7-results)
- [8. How to Run](#8-how-to-run)
- [9. Repository Structure](#9-repository-structure)
- [10. Future Work](#10-future-work)

---

## 1. Overview

**Have you ever been told you have a medical condition and had no idea what it meant?**

Most patients walk out of a clinic with a diagnosis they cannot explain to their own family. The words are precise — *bradycardia*, *pulmonary embolism*, *myocardial infarction* — but they mean nothing to someone without a medical background.

**Welcome to the Medical Term Simplifier.**

This project fine-tunes **Mistral-7B** using **LoRA** on a curated multi-source medical dataset to translate clinical language into plain, patient-friendly English. You input a medical term or question — the model explains it the way a doctor would explain it to a patient.

| Input (Medical Jargon) | Output (Plain English) |
|---|---|
| *Bradycardia* | Your heart is beating slower than normal — fewer than 60 beats per minute. Most people feel dizzy or fatigued when this happens. |
| *Myocardial infarction* | A heart attack. A blockage cuts off blood supply to part of the heart, causing that part of the muscle to die. |
| *Pulmonary embolism* | A blood clot that has travelled to your lungs. It blocks blood flow and can be life-threatening if not treated quickly. |

---

## 2. Problem Statement

Medical jargon is a barrier between doctors and patients. This barrier is not just an inconvenience — it has real consequences:

- Patients misunderstand their own diagnosis and treatment plan
- Medication instructions are followed incorrectly because the condition was not understood
- Patients delay seeking care because they did not recognise symptoms as serious
- People turn to unverified internet searches to decode what their doctor told them

General-purpose language models like base GPT or Llama can attempt to explain medical terms, but they were not specifically trained for this task. They produce inconsistent depth, sometimes use technical language in the explanation itself, and were not trained with patient-education framing in mind.

**There is no widely available, fine-tuned model specifically trained to simplify medical terminology into patient-friendly language across a diverse range of medical domains.**

This project solves that.

---

## 3. Use Case

**For patients**
Ask about any medical term, condition, or diagnosis in plain language and get an explanation that is actually understandable.

**For healthcare workers**
Use as a drafting tool for patient communication — quickly generate plain-English explanations of terms before a consultation.

**For medical students**
Use as a study aid — get concept-level plain-language summaries to complement technical study material.

**For developers**
The pipeline is fully config-driven via `config.yaml`. Every hyperparameter, dataset source, and instruction variant is defined in YAML — swap datasets, change the base model, or add new instruction styles without touching any Python code.

---

## 4. How We Got the Idea

When someone receives a medical diagnosis, the first thing they do is search the internet. The results they get are either too technical (Wikipedia, medical journals) or too unreliable (forums, unverified blogs).

There is no tool that does one thing well: take a medical term and explain it clearly, accurately, and in plain language.

Looking at existing large language models, they can explain medical terms — but inconsistently. The quality varies by term, the framing is not always patient-oriented, and there is no guarantee of accuracy since they were not specifically trained for this task.

The idea was simple: take a strong open-source model, gather high-quality medical Q&A data from multiple verified sources, and fine-tune it specifically for the task of patient-friendly medical explanation.

---

## 5. Architecture

### Pipeline Overview

The full training pipeline runs end to end from one `main()` call in `training_pipeline.ipynb`.

```
config.yaml
    │
    ▼
load_config()
    │
    ▼
load_local_dataset()   ←── med_dataset.json (your curated data)
    │
    ▼
build_dataset()        ←── 6 HuggingFace sources + local
    │  _extract_row()      handles different column layouts per source
    │  load_source()       normalises each source independently
    │
    ▼
load_model()           ←── Mistral-7B 4-bit + LoRA adapters
    │  from_pretrained(**model_cfg)
    │  get_peft_model(**lora_cfg)
    │
    ▼
prepare_splits()       ←── Alpaca prompt format + length filter
    │
    ▼
SFTTrainer(TrainingArguments(**training_cfg))
    │
    ▼
push_to_hub()          ←── LoRA adapter → HuggingFace Hub
```

### Two-Notebook Design

| Notebook | Purpose | Cells |
|---|---|---|
| `training_pipeline.ipynb` | Full training — run once | 3 cells → `main()` |
| `demo.ipynb` | Presentation — no training | 3 cells → `main()` |

The demo notebook loads the already-trained model from HuggingFace Hub and runs `main()` which demonstrates 5 terms live.

### Config-Driven Design

All hyperparameters live in `config.yaml`. Keys are named to match the exact API parameter names — so every section unpacks directly into its function call with `**`:

```python
FastLanguageModel.from_pretrained(**model_cfg)
FastLanguageModel.get_peft_model(model, **lora_cfg)
TrainingArguments(**training_cfg)
model.generate(**inference_cfg)
```

No manual key listing. Change a parameter in YAML — it flows through automatically.

---

## 6. What Have We Done

### 6.1 Data Collection

We assembled training data from **5 tiers** of medical sources, each contributing a different type of medical knowledge:

| Tier | Dataset | Samples | What it teaches |
|---|---|---|---|
| 1 | ChatDoctor (HealthCareMagic-100k) | 1,500 | Conversational doctor–patient tone |
| 2 | PubMedQA (pqa_labeled) | 700 | Scientific medical reasoning |
| 3 | MedMCQA | 700 | Medical concept understanding |
| 4 | Medical Flashcards + WikiDoc + MedQuad | 500 + 800 + 800 | Terminology + patient education |
| 5 | Local curated dataset | All (no cap) | Custom hand-curated high-quality pairs |

**Total cap: 5,000 samples. Train/val split: 95% / 5%.**

Each source has a different internal structure — different column names, nested answer fields, MCQ indices. The `_extract_row()` function handles all layouts using only the YAML config, with zero source-specific code.

### 6.2 Dataset Pipeline

Every sample is normalised into a standard format before training:

```
{
  "instruction": "Explain this medical term in simple plain English",
  "input":       "Bradycardia",
  "output":      "Bradycardia means your heart is beating slower than normal..."
}
```

6 instruction variants are defined in `config.yaml` and randomly assigned to each sample during normalisation. This teaches the model to respond to varied patient question styles rather than one rigid format.

The formatted training prompt follows the Alpaca template:

```
### Instruction:
Explain this medical term in simple plain English

### Input:
Bradycardia

### Response:
Bradycardia means your heart is beating slower than normal...
```

### 6.3 Model Fine-Tuning

**Base model:** `unsloth/mistral-7b-bnb-4bit`
**Method:** LoRA (Low-Rank Adaptation)
**Framework:** Unsloth (2× faster training, 50% less VRAM)
**Quantisation:** 4-bit (reduces GPU requirement from ~28GB to ~6GB)

LoRA adds small trainable matrices to the attention layers only. The base model is frozen — only the adapter weights are updated during training. This makes fine-tuning feasible on a single consumer GPU in under an hour.

| LoRA Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Gradient checkpointing | unsloth |

| Training Parameter | Value |
|---|---|
| Max steps | 500 |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Batch size (effective) | 8 (2 × 4 accumulation) |
| Optimiser | adamw_8bit |

---

## 7. Results

The trained model is available on HuggingFace Hub:

**[Akhilesh-2308/medical-term-simplifier](https://huggingface.co/Akhilesh-2308/medical-term-simplifier)**

### Sample outputs

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Term   : Myocardial infarction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A myocardial infarction is the medical name for a heart attack.
  It happens when one of the arteries supplying blood to your heart
  gets blocked. The part of the heart muscle that loses its blood
  supply begins to die. Symptoms include chest pain, shortness of
  breath, and pain spreading to your left arm.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Term   : Pulmonary embolism
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A pulmonary embolism is a blood clot that has travelled to your
  lungs. The clot blocks blood flow through the lung, making it
  hard to breathe and reducing oxygen in your blood. It can be
  life-threatening and needs immediate medical attention.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 8. How to Run

### Training (once)

1. Open `training_pipeline.ipynb` in Google Colab (GPU runtime required — T4 minimum)
2. Upload `config.yaml` and `med_dataset.json` via the file panel
3. Add your HuggingFace token to Colab Secrets as `HF_TOKEN`
4. Run Cell 1 (install), Cell 2 (check files), Cell 3 (`main()`)

Training runs for 500 steps (~20–40 min on T4) and pushes the adapter to your HuggingFace Hub automatically.

### Demo / Presentation

1. Open `demo.ipynb` in Google Colab
2. Upload `config.yaml` and add `HF_TOKEN` to Colab Secrets
3. Run Cell 1 (install), Cell 2 (load model from Hub), Cell 3 (`main()`)

`main()` automatically runs 5 showcase terms. You can also call `demo("any term")` live during the presentation.

```python
# After main(), call any term on the fly
demo("Rheumatoid arthritis")
demo("What is a stroke?")
demo("Cirrhosis", instruction="Explain for a 10-year-old")
```

### Loading the model independently

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "Akhilesh-2308/medical-term-simplifier",
    max_seq_length = 512,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)
```

---

## 9. Repository Structure

```
medical-term-simplifier/
│
├── training_pipeline.ipynb   # Full training pipeline — run once
│                             # Cell 3: main() trains + pushes to Hub
│
├── demo.ipynb                # Presentation notebook — no training
│                             # Cell 3: main() runs 5 showcase terms live
│
├── config.yaml               # All hyperparameters — single source of truth
│                             # Keys match API param names exactly (**unpack)
│
└── med_dataset.json          # Your local curated dataset
                              # Upload to Colab before running training
```

---

## 10. Future Work

- **ROUGE / BERTScore evaluation** — quantify how well the model simplifies compared to reference explanations
- **More instruction styles** — add domain-specific variants (paediatric, elderly, low-literacy)
- **Larger base model** — test with Mistral-7B-Instruct or Llama-3-8B for improved baseline quality
- **Interactive web demo** — deploy via Gradio or Streamlit on HuggingFace Spaces
- **Multilingual support** — extend to Hindi and other Indian languages for broader accessibility
- **Symptom-to-condition mapping** — expand beyond term simplification to symptom explanation

---

## About

Built with [Unsloth](https://github.com/unslothai/unsloth) · [HuggingFace](https://huggingface.co) · [Google Colab](https://colab.research.google.com)

**Model on Hub:** [Akhilesh-2308/medical-term-simplifier](https://huggingface.co/Akhilesh-2308/medical-term-simplifier)
