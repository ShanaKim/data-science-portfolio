# Cloud Detection in Polar Regions  
**Python · PyTorch · Machine Learning · Remote Sensing**

## Overview
This project develops a machine learning pipeline to classify **cloud vs. non-cloud pixels in polar satellite imagery**, a challenging task due to the visual similarity between clouds and ice-covered surfaces. Using **satellite data from MISR**, we combine domain-informed features with **deep learning–based representation learning** to improve classification performance.

The final model achieves **92%+ test accuracy and ROC-AUC above 0.98**, demonstrating strong generalization and robustness.

---

## Data
- **200+ satellite images** from the Multi-angle Imaging SpectroRadiometer (MISR)
- Expert-labeled cloud/non-cloud pixels for supervised learning
- Additional unlabeled images used for unsupervised feature learning

---

## Methods
### Exploratory Data Analysis (EDA)
- Visualized spatial patterns, feature distributions, and class imbalance across labeled images

### Feature Engineering
- Original domain features (e.g., NDAI, SD, CORR, radiance angles)
- Spatial context features (local mean / variance)
- **Unsupervised feature extraction using pretrained deep autoencoders (PyTorch)**

### Modeling
- LightGBM, XGBoost, KNN classifiers
- Cross-validation with image-based splits (to prevent data leakage)
- Hyperparameter tuning and architecture comparisons

### Evaluation & Stability Checks
- Accuracy, Precision, Recall, F1, ROC-AUC
- Noise injection (Gaussian perturbations)
- Bootstrap resampling for robustness

---

## Results
- **Best model:** LightGBM with autoencoder-generated embeddings  
- **Performance:**
  - Test Accuracy: **~92.4%**
  - ROC-AUC: **0.98–0.99**
- Learned embeddings significantly outperformed raw features, confirming the value of transfer learning for satellite imagery.

---

## Repository Structure
```
lab2/
├── code/ # Model training, feature engineering, autoencoders
├── figs/ # Plots and visualizations
├── report/ # Final PDF report
├── results/ # Model outputs and evaluation results
```

---

## Key Takeaways
- Demonstrates a **full ML pipeline** from raw satellite data to production-ready models  
- Combines **statistical learning + deep learning** in a principled way  
- Emphasizes **robust evaluation and reproducibility**, not just accuracy

---

📄 **Full report:** See `report/` for methodology, figures, and detailed analysis.


