# BioMistral Downstream Clinical Classifier

This repository provides a pipeline to extract clinical embeddings from patient symptom notes using **BioMistral-7B** (quantized via 4-bit `bitsandbytes`) and train a downstream **Scikit-Learn Logistic Regression** model for medical risk classification.

## 🚀 Features
* **BioMistral Embeddings:** Leverages a domain-specific 7B parameter medical LLM.
* **4-Bit Quantization:** Uses NF4 quantization via `bitsandbytes` to run efficiently on consumer GPUs with lower VRAM footprints.
* **Feature Extraction:** Pools the final hidden states layer to generate static vectors.
* **Downstream Classification:** High-speed classification using Scikit-Learn.

## 📦 Prerequisites

Ensure you have a GPU environment setup (CUDA compatible) and install the following dependencies:

*See requirements.txt

## 🛠️ Usage

*The pipeline automatically tokenizes raw medical notes, extracts 4096-dimensional embeddings, splits data stratifically, and evaluates the final downstream metric performance.



## ⚠️ Notes on Limitations
* **Mean Pooling Artifacts:** The basic text embedding function averages across all padded elements. For real applications, implement an attention-mask-aware mean pooling scheme to avoid padding noises distorting clinical text representation vectors.
* **Dataset Scale:** This architecture fits optimally when scaling up to thousands of unstructured electronic health records (EHR notes).
