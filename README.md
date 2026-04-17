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

## 📚 Dataset Sources

This project uses a combination of publicly available medical datasets, each contributing different types of knowledge:

---

### 🩺 ChatDoctor Dataset

https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k

A large dataset of doctor-patient conversations collected from medical forums.
It contains real-world health queries and responses, making it useful for training conversational medical explanations.

---

### 🔬 PubMedQA

https://huggingface.co/datasets/pubmed_qa

A biomedical question-answering dataset based on scientific research papers.
It focuses on evidence-based answers derived from PubMed articles, improving factual accuracy.

---

### 🧠 MedMCQA

https://huggingface.co/datasets/medmcqa

A multiple-choice question dataset from medical entrance exams.
It helps the model learn precise medical concepts and correct answers through structured options.

---

### 📘 Medical Meadow (Flashcards + WikiDoc)

https://huggingface.co/datasets/medalpaca/medical_meadow

A collection of medical flashcards and patient-friendly explanations.
It is useful for simplifying complex medical concepts into easy-to-understand language.

---

### 🏥 MedQuAD

https://huggingface.co/datasets/medquad

A curated dataset of medical question-answer pairs from trusted health websites.
It provides high-quality, structured medical information for common health conditions.

---

### 🧾 Local Curated Dataset

(Custom JSON dataset)

A manually created dataset containing simplified medical explanations.
It is tailored specifically for this project to improve clarity and readability of outputs.

---

## 🤖 Trained Model

The fine-tuned model is available on Hugging Face:

👉 **Model Link:**
https://huggingface.co/Akhilesh-2308/Medical-Term_simplifier

> Replace the above link with your actual Hugging Face model URL

---

## 🏗️ Project Structure

```
.
├── notebook.ipynb
├── config.yaml
├── dataset.json
├── README.md
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


2. Upload dataset + config

3. Run notebook cells sequentially

