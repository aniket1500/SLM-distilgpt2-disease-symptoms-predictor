# 🧠 SLM Disease Symptoms Predictor using DistilGPT2

This project fine-tunes a lightweight GPT-2 model (`distilgpt2`) to generate probable symptoms based on input disease names. It demonstrates how to build a sequence-to-sequence language model tailored for medical datasets using the Hugging Face `transformers` library.

---

## 📊 Dataset

- **Source**: [QuyenAnhDE/Diseases_Symptoms](https://huggingface.co/datasets/QuyenAnhDE/Diseases_Symptoms)
- The dataset contains diseases along with their associated symptoms.
- A `pandas` DataFrame is created and preprocessed to standardize symptom formatting.

---

## 🧪 Model Training

### 🔧 Architecture

- **Model Base**: `distilgpt2`
- **Type**: Causal Language Model
- **Framework**: PyTorch

### 🛠 Training Steps

1. Dataset loaded and cleaned.
2. Custom PyTorch `Dataset` class defined for dynamic tokenization.
3. Used a GPT-style training loop where:
   - Input: `"Disease Name"`
   - Target: `"Disease Name | Symptom1, Symptom2, ..."`
4. Optimizer: `AdamW`
5. Loss: `CrossEntropyLoss` with `ignore_index` for padding token

---

## 📤 Model Upload

The fine-tuned model and tokenizer are uploaded to the [Hugging Face Hub](https://huggingface.co/aniketsalunkhe15/SLM-distilgpt2-disease-symptoms-predictor):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("aniketsalunkhe15/SLM-distilgpt2-disease-symptoms-predictor")
tokenizer = AutoTokenizer.from_pretrained("aniketsalunkhe15/SLM-distilgpt2-disease-symptoms-predictor")
