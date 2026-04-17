Medical Term Simplifier (LLM Fine-Tuning Project)

**📌 Overview**

This project fine-tunes a Large Language Model (LLM) to convert complex medical terms into simple, easy-to-understand explanations.

It uses:

LoRA (Low-Rank Adaptation)
4-bit quantization
Multi-source datasets
YAML-based configuration





**❗ Problem Statement**

Medical terminology is difficult for non-medical users to understand. Existing explanations are often too technical and inconsistent across sources.

**💡 Solution**

We build a fine-tuned LLM that:
Simplifies medical terms into layman language
Learns from multiple datasets
Uses efficient training techniques to run on limited hardware


**⚙️ Key Features**

✅ Config-driven pipeline (YAML)
✅ Multi-source dataset support
✅ LoRA-based efficient training
✅ 4-bit quantization for low memory usage
✅ Modular and reusable code structure


**🏗️ Project Structure**

.
├── config.yaml          # All configurations

├── notebook.ipynb       # Main training pipeline

├── dataset.json         # Local dataset

├── README.md            # Project documentation


**🔄 Pipeline Overview**

Load Config (YAML)
Load & Merge Datasets
Format Prompts
Load Model + Apply LoRA
Train using SFTTrainer
Run Inference


**🧪 Example Output**

Input:

Myocardial infarction

Output:

A myocardial infarction, commonly known as a heart attack,
occurs when blood flow to the heart is blocked...


**⚙️ Configuration (config.yaml)**

All parameters are controlled via YAML:

Model settings

Dataset sources

Training hyperparameters

Inference settings



**🚀 How to Run**

Install dependencies

pip install -r requirements.txt

Upload dataset + config

Run notebook cells sequentially


**📊 Training Techniques Used**

LoRA (Low-Rank Adaptation)

Quantization (4-bit)

Supervised Fine-Tuning (SFT)
