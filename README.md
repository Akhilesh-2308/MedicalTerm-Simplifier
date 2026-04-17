# 🧠 Medical Term Simplifier (LLM Fine-Tuning Project)

## 📌 Overview

This project fine-tunes a Large Language Model (LLM) to convert complex medical terms into simple, easy-to-understand explanations.

---

## ❗ Problem Statement

Medical terminology is difficult for non-medical users to understand. Existing explanations are often too technical and inconsistent.

---

## 💡 Solution

We build a fine-tuned LLM that:

* Simplifies medical terms into layman language
* Learns from multiple datasets
* Uses efficient training techniques (LoRA + quantization)

---

## ⚙️ Key Features

* Config-driven pipeline (YAML)
* Multi-source dataset support
* LoRA-based efficient training
* 4-bit quantization for low memory usage
* Modular and reusable code

---

## 🏗️ Project Structure

```
.
├── notebook.ipynb
├── config.yaml
├── dataset.json
├── README.md
├── requirements.txt
```

---

## 🔄 Pipeline Overview

1. Load Config (YAML)
2. Load & Merge Datasets
3. Format Prompts
4. Load Model + Apply LoRA
5. Train Model
6. Run Inference

---

## 🧪 Example Output

**Input:**

```
Myocardial infarction
```

**Output:**

```
A myocardial infarction, commonly known as a heart attack,
occurs when blood flow to the heart is blocked...
```

---

## ⚙️ Configuration

All parameters are controlled via `config.yaml`:

* Model settings
* Dataset sources
* Training hyperparameters
* Inference settings

---

## 🚀 How to Run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Upload dataset + config

3. Run notebook cells sequentially

---

## 🧠 Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* TRL (SFTTrainer)
* Unsloth

---

## 📌 Future Improvements

* Add UI (Streamlit / Web App)
* Improve dataset diversity
* Add evaluation metrics
* Deploy model as API

---

## 📜 License

MIT License
