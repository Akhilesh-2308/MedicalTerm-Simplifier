# ╔══════════════════════════════════════════════════════════════════╗
# ║  Medical LLM Fine-Tuner  —  Production Notebook                 ║
# ║  Google Colab T4 GPU                                            ║
# ║  All settings driven by CONFIG_YAML. Edit that block only.      ║
# ╚══════════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────────────────────────────
# CELL 1 ── Suppress noisy warnings + install dependencies
# ─────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"]     = "expandable_segments:True"

!pip uninstall -y unsloth transformers peft trl accelerate xformers bitsandbytes 2>/dev/null
!pip install --no-cache-dir \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    pyyaml datasets evaluate rouge_score huggingface_hub -q


# ─────────────────────────────────────────────────────────────────────
# CELL 2 ── Config  (edit ONLY this block to change any behaviour)
# ─────────────────────────────────────────────────────────────────────

import yaml, pathlib, json

CONFIG_YAML = """
model:
  name: "unsloth/mistral-7b-bnb-4bit"
  max_seq_length: 512       # hard ceiling — sequences longer than this are filtered out
  load_in_4bit: true
  dtype: null               # null = auto (fp16 on T4, bf16 on A100)

lora:
  r: 16
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  bias: "none"
  use_gradient_checkpointing: true

dataset:
  local_path: "medical_dataset.json"   # path to the JSON file uploaded to Colab
  sources:
    - name: "medalpaca_flashcards"
      type: "huggingface"
      path: "medalpaca/medical_meadow_medical_flashcards"
      split: "train"
      input_col: "input"
      output_col: "output"
      max_samples: 500
      enabled: true

    - name: "medalpaca_wikidoc_patient"
      type: "huggingface"
      path: "medalpaca/medical_meadow_wikidoc_patient_information"
      split: "train"
      input_col: "input"
      output_col: "output"
      max_samples: 500
      enabled: true

    - name: "medquad"
      type: "huggingface"
      path: "keivalya/MedQuad-MedicalQnADataset"
      split: "train"
      input_col: "Question"
      output_col: "Answer"
      max_samples: 500
      enabled: true

    - name: "local_curated"
      type: "local"
      input_col: "input"
      output_col: "output"
      instruction_col: "instruction"
      enabled: true

  total_cap: 1500
  val_split: 0.05
  seed: 42
  instruction_variants:
    - "Explain this medical term in simple plain English"
    - "What does this medical term mean in everyday language?"
    - "Describe this medical condition as if explaining to a patient with no medical background"
    - "Simplify this medical term for someone with no medical knowledge"

training:
  output_dir: "outputs/checkpoints"
  epochs: 1
  max_steps: 200
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  lr_scheduler: "cosine"
  warmup_steps: 20
  logging_steps: 25
  optim: "adamw_8bit"
  fp16: true                 # T4 uses fp16; set false + bf16: true on A100
  eval_strategy: "epoch"     # evaluate once per epoch
  save_strategy: "no"        # no checkpoint saving needed
  report_to: "none"          # set to "wandb" if you have an account
  seed: 42

inference:
  max_new_tokens: 200
  do_sample: false
  temperature: 1.0           # only used when do_sample is true
  repetition_penalty: 1.1

export:
  lora_dir: "outputs/lora_adapter"
  push_to_hub: true
  hub_model_id: "your-username/medical-mistral-7b-lora"
  hub_token_env: "HF_TOKEN"              # set this secret in Colab: Runtime > Secrets
"""

cfg = yaml.safe_load(CONFIG_YAML)
pathlib.Path("config.yaml").write_text(CONFIG_YAML)

mc = cfg["model"]
lc = cfg["lora"]
dc = cfg["dataset"]
tc = cfg["training"]
ic = cfg["inference"]
ec = cfg["export"]

print("✅ Config loaded")
print(f"   Model         : {mc['name']}")
print(f"   Max seq len   : {mc['max_seq_length']}")
print(f"   HF sources    : {[s['name'] for s in dc['sources'] if s['enabled'] and s['type']=='huggingface']}")
print(f"   Local dataset : {dc['local_path']}")
print(f"   Total cap     : {dc['total_cap']} samples")
print(f"   Max steps     : {tc['max_steps']}")
print(f"   Push to Hub   : {ec['push_to_hub']}  →  {ec['hub_model_id']}")


# ─────────────────────────────────────────────────────────────────────
# CELL 3 ── Upload medical_dataset.json
# ─────────────────────────────────────────────────────────────────────
# Upload the medical_dataset.json file via the Colab file panel (left sidebar)
# before running this cell. It must be at the path set in dataset.local_path above.

local_path = dc["local_path"]

if not pathlib.Path(local_path).exists():
    raise FileNotFoundError(
        f"'{local_path}' not found. "
        "Please upload medical_dataset.json via the Colab file panel first."
    )

with open(local_path) as f:
    local_data = json.load(f)

print(f"✅ Local dataset loaded: {len(local_data)} samples from '{local_path}'")
print(f"   Sample output preview: {local_data[0]['output'][:120]} …")


# ─────────────────────────────────────────────────────────────────────
# CELL 4 ── Build merged dataset from all sources
# ─────────────────────────────────────────────────────────────────────

import random
from datasets import load_dataset, Dataset, DatasetDict

def load_source(src: dict, variants: list, local_rows: list) -> list:
    name    = src["name"]
    limit   = src.get("max_samples", 9999)
    in_col  = src["input_col"]
    ou_col  = src["output_col"]
    in_instr = src.get("instruction_col")

    print(f"  Loading '{name}' …", end=" ", flush=True)

    if src["type"] == "local":
        rows = local_rows

    elif src["type"] == "huggingface":
        ds   = load_dataset(src["path"], split=src.get("split", "train"))
        rows = [dict(r) for r in ds.select(range(min(limit, len(ds))))]

    else:
        raise ValueError(f"Unknown source type: {src['type']}")

    normalised = []
    for row in rows:
        q = str(row.get(in_col, "") or "").strip()
        a = str(row.get(ou_col, "") or "").strip()
        if not q or not a:
            continue
        if in_instr and row.get(in_instr):
            instr = str(row[in_instr]).strip()
        else:
            instr = random.choice(variants)
        normalised.append({"instruction": instr, "input": q, "output": a})

    print(f"{len(normalised)} samples")
    return normalised


def build_dataset(cfg: dict, local_rows: list) -> DatasetDict:
    seed     = dc.get("seed", 42)
    cap      = dc.get("total_cap", 1500)
    val_frac = dc.get("val_split", 0.05)
    variants = dc["instruction_variants"]

    random.seed(seed)
    all_rows = []

    for src in dc["sources"]:
        if not src.get("enabled", True):
            print(f"  Skipping '{src['name']}'")
            continue
        try:
            all_rows.extend(load_source(src, variants, local_rows))
        except Exception as e:
            print(f"  ⚠️  Failed to load '{src['name']}': {e}")

    random.shuffle(all_rows)
    all_rows = all_rows[:cap]
    print(f"\n✅ Merged dataset: {len(all_rows)} samples (cap={cap})")

    cut = int(len(all_rows) * (1 - val_frac))
    return DatasetDict({
        "train":      Dataset.from_list(all_rows[:cut]),
        "validation": Dataset.from_list(all_rows[cut:]),
    })


print("Building dataset …")
splits = build_dataset(cfg, local_data)
print(f"   Train: {len(splits['train'])}  |  Val: {len(splits['validation'])}")


# ─────────────────────────────────────────────────────────────────────
# CELL 5 ── Load model + LoRA  (all params from cfg)
# ─────────────────────────────────────────────────────────────────────

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = mc["name"],
    max_seq_length = mc["max_seq_length"],
    dtype          = mc["dtype"],
    load_in_4bit   = mc["load_in_4bit"],
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = lc["r"],
    target_modules             = lc["target_modules"],
    lora_alpha                 = lc["alpha"],
    lora_dropout               = lc["dropout"],
    bias                       = lc["bias"],
    use_gradient_checkpointing = lc["use_gradient_checkpointing"],
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ Model + LoRA ready — {trainable:,} trainable params ({100*trainable/total:.2f}%)")


# ─────────────────────────────────────────────────────────────────────
# CELL 6 ── Format + filter dataset
# ─────────────────────────────────────────────────────────────────────

EOS     = tokenizer.eos_token
MAX_LEN = mc["max_seq_length"]   # pulled from config

def format_row(example):
    # Richer prompt — adds "Explanation:" label to encourage longer, structured answers
    return {
        "text": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}{EOS}"
        )
    }

def filter_long(example):
    # Remove samples that exceed MAX_LEN when tokenized — prevents shape mismatch errors
    ids = tokenizer(example["text"], truncation=False, add_special_tokens=True)["input_ids"]
    return len(ids) <= MAX_LEN

train_ds = splits["train"].map(format_row, remove_columns=["instruction", "input", "output"])
val_ds   = splits["validation"].map(format_row, remove_columns=["instruction", "input", "output"])

print("Filtering long sequences …")
train_ds = train_ds.filter(filter_long, num_proc=1)
val_ds   = val_ds.filter(filter_long, num_proc=1)
print(f"After filter — Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"\nSample formatted text:\n{train_ds[0]['text'][:300]} …")


# ─────────────────────────────────────────────────────────────────────
# CELL 7 ── Train  (all params from cfg)
# ─────────────────────────────────────────────────────────────────────

from trl import SFTTrainer
from transformers import TrainingArguments
import torch

torch._dynamo.config.disable = True   # prevents Dynamo/cross_entropy shape mismatch with Unsloth

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_LEN,      # from config
    packing            = False,
    args               = TrainingArguments(
        output_dir                  = tc["output_dir"],
        num_train_epochs            = tc["epochs"],
        max_steps                   = tc["max_steps"],
        per_device_train_batch_size = tc["per_device_train_batch_size"],
        gradient_accumulation_steps = tc["gradient_accumulation_steps"],
        learning_rate               = tc["learning_rate"],
        lr_scheduler_type           = tc["lr_scheduler"],
        warmup_steps                = tc["warmup_steps"],
        fp16                        = tc["fp16"],            # training.fp16
        logging_steps               = tc["logging_steps"],
        eval_strategy               = tc["eval_strategy"],  # training.eval_strategy
        save_strategy               = tc["save_strategy"],  # training.save_strategy
        optim                       = tc["optim"],
        seed                        = tc["seed"],
        report_to                   = tc["report_to"],      # training.report_to
    ),
)

print("🚀 Starting training …")
train_result = trainer.train()
print("\n✅ Training complete")
print(f"   Runtime     : {train_result.metrics['train_runtime']:.0f}s")
print(f"   Samples/sec : {train_result.metrics['train_samples_per_second']:.1f}")
print(f"   Final loss  : {train_result.metrics['train_loss']:.4f}")

# Evaluate immediately — must be in same cell as train() to keep trainer state
eval_metrics = trainer.evaluate()
print("\n📊 Validation metrics:")
for k, v in eval_metrics.items():
    print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")


# ─────────────────────────────────────────────────────────────────────
# CELL 8 ── Inference  (all params from cfg)
# ─────────────────────────────────────────────────────────────────────

import torch
from typing import Optional

# Use model.eval() only — do NOT call FastLanguageModel.for_inference()
# after training in the same session (causes rotary embedding shape errors)
model.eval()

def ask(term: str, instruction: Optional[str] = None) -> str:
    if instruction is None:
        instruction = "Explain this medical term in simple plain English"

    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{term}\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors      = "pt",
        truncation          = True,
        max_length          = MAX_LEN,   # from config via Cell 6
    ).to("cuda")

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids          = inputs["input_ids"],
            attention_mask     = inputs["attention_mask"],
            max_new_tokens     = ic["max_new_tokens"],        # inference.max_new_tokens
            do_sample          = ic["do_sample"],             # inference.do_sample
            temperature        = ic["temperature"],           # inference.temperature
            repetition_penalty = ic["repetition_penalty"],   # inference.repetition_penalty
            eos_token_id       = tokenizer.eos_token_id,
            pad_token_id       = tokenizer.eos_token_id,
            use_cache          = True,
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


test_terms = [
    "Bradycardia",
    "Hyperglycemia",
    "Sepsis",
    "Cirrhosis",
    "Pulmonary embolism",
    "Atrial fibrillation",
    "Hypothyroidism",
    "Rheumatoid arthritis",
]

print("\n─── Inference Results ───────────────────────────────────────────")
for term in test_terms:
    print(f"\n🔹 {term}")
    print(f"   {ask(term)}")
print("─────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────
# CELL 9 ── Save LoRA adapter + push to Hugging Face Hub
# ─────────────────────────────────────────────────────────────────────

import os
from huggingface_hub import login

LORA_DIR = ec["lora_dir"]    # export.lora_dir

# Always save the adapter locally first
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"✅ LoRA adapter saved locally → {LORA_DIR}/")

# Push to Hugging Face Hub if enabled in config
if ec["push_to_hub"]:
    hf_token = os.environ.get(ec["hub_token_env"])

    if not hf_token:
        # Try Colab secrets (userdata API available in Colab)
        try:
            from google.colab import userdata
            hf_token = userdata.get(ec["hub_token_env"])
        except Exception:
            hf_token = None

    if not hf_token:
        print(
            f"\n⚠️  push_to_hub is true but '{ec['hub_token_env']}' secret is not set.\n"
            "   To push to the Hub:\n"
            "   1. Go to Runtime > Secrets in Colab\n"
            f"   2. Add a secret named '{ec['hub_token_env']}' with your HF write token\n"
            "   3. Re-run this cell"
        )
    else:
        login(token=hf_token)
        model.push_to_hub(ec["hub_model_id"], token=hf_token)
        tokenizer.push_to_hub(ec["hub_model_id"], token=hf_token)
        print(f"\n✅ Model pushed to: https://huggingface.co/{ec['hub_model_id']}")
