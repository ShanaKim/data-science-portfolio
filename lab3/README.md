# Language-to-Brain Modeling with fMRI

This project investigates how linguistic representations map to neural activity in the human brain. Using fMRI data, we build machine learning models that predict voxel-level brain responses from language stimuli and analyze which semantic features drive neural activation.

The work progresses from classical language representations to modern transformer-based embeddings, combining predictive modeling with interpretability methods.

---

## Project Overview

- Modeled fMRI BOLD responses across **100,000+ voxels per subject**
- Built modular ML pipelines using **ridge regression** with large-scale cross-validation
- Compared linguistic representations including **Bag-of-Words, Word2Vec, GloVe, and BERT**
- Applied **SHAP and LIME** to interpret word-level contributions to neural activity
- Fine-tuned **BERT with LoRA** for parameter-efficient adaptation

---

## Repository Structure

```text
lab3/
├── lab3.1/   # Baseline encoding models (BoW, Word2Vec, GloVe)
├── lab3.2/   # Pre-trained BERT embeddings for voxel prediction
├── lab3.3/   # LoRA fine-tuning + SHAP/LIME interpretability
```

Each subfolder follows a consistent internal structure:
```
code/         # Model training and evaluation scripts
figs/         # Visualizations and plots
report/       # PDF report with results and interpretation
ridge_utils/  # Shared utilities for ridge regression and evaluation
```

## Lab 3.1 — Baseline Language Encoding Models

- Implemented ridge regression models linking linguistic features to voxel responses  
- Evaluated Bag-of-Words, Word2Vec, and GloVe embeddings  
- Performed voxel-wise cross-validation and correlation-based performance analysis  
- Established baseline encoding performance for later comparison  

---

## Lab 3.2 — Pre-trained BERT Representations

- Extracted contextualized word embeddings using **BERT-base-uncased**  
- Aligned embeddings temporally with fMRI responses  
- Achieved significant performance gains over classical embeddings  
- Analyzed voxel-level correlation improvements across subjects  

---

## Lab 3.3 — Fine-tuning & Interpretability

- Fine-tuned BERT using **LoRA (Low-Rank Adaptation)** for parameter-efficient learning  
- Evaluated whether fine-tuning improves voxel prediction performance  
- Applied **SHAP** and **LIME** to identify influential words driving neural responses  
- Mapped semantic features back to brain regions and assessed interpretability stability  

---

## Key Takeaways

- Transformer-based embeddings substantially outperform traditional language models  
- Fine-tuning yields limited gains with small datasets, highlighting data constraints  
- SHAP and LIME consistently identify meaningful linguistic drivers of neural activity  
- Interpretability reveals sparse, voxel-specific semantic sensitivity patterns  

---

## Tools & Methods

- Python, NumPy, SciPy  
- scikit-learn (ridge regression, cross-validation)  
- PyTorch, HuggingFace Transformers  
- SHAP, LIME  
- fMRI voxel-wise modeling  

