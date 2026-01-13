# Pediatric Traumatic Brain Injury (TBI) Prediction

**Course:** STAT 214 – Statistical Modeling  
**Author:** Soohyun Kim  
**Date:** Feb–Mar 2025  

## Overview
This project analyzes the **PECARN Pediatric Traumatic Brain Injury (TBI) dataset**, a large multi-center clinical dataset used to develop decision rules for identifying children at risk of **clinically important TBI (ciTBI)**.  

The goal is to:
- Explore data quality and clinical patterns
- Identify key predictors of ciTBI
- Build and interpret predictive models that balance sensitivity and specificity in a highly imbalanced medical setting

The dataset contains **43,000+ pediatric patient records** with **125 original variables**, which were carefully cleaned and reduced for modeling.

---

## Data
- **Source:** PECARN TBI Public Use Dataset  
- **Population:** Children under 18 presenting with minor head trauma  
- **Outcome:** Clinically important TBI (ciTBI)  
- **Final cleaned dataset:** 43,379 observations × 55 features  

Key preprocessing steps:
- Removed irrelevant identifiers and redundant clinical fields  
- Handled missing and inapplicable values using structured, domain-aware rules  
- Compared imputation vs. row-removal strategies for stability analysis  

---

## Exploratory Analysis
Key findings from EDA include:
- ciTBI occurs in **<2%** of cases → severe class imbalance
- Many children receive CT scans despite not having ciTBI
- Strong clinical indicators include:
  - Altered Mental Status (AMS)
  - Loss of Consciousness (LOC)
  - Acting Normally (protective)
  - Scalp and skull fractures
- Age and gender patterns show higher ciTBI risk among **older male adolescents**

Visualizations include:
- Missingness diagnostics
- CT scan vs. outcome comparisons
- Risk factor distributions
- Age–gender histograms

---

## Modeling
Multiple classification models were developed and evaluated:
- Logistic Regression (primary focus for interpretability)
- Random Forest
- LDA / QDA
- Support Vector Machine (SVM)

Modeling highlights:
- Used **cross-validation and class-weighting** to address imbalance
- Performed **backward feature elimination** to optimize sensitivity
- Best logistic regression model achieved:
  - **Sensitivity:** ~86%
  - **Specificity:** ~86%
  - **Accuracy:** >85%

Interpretability was emphasized using:
- Odds ratios
- Clinical risk interpretation
- Stability checks under different data-cleaning assumptions

---

## Key Takeaways
- Core PECARN predictors were strongly validated by the data
- Some commonly assumed symptoms (e.g., headache alone) showed weak predictive power
- Handling missing clinical data materially affects model behavior
- A small, interpretable feature set can achieve strong predictive performance

---

## Repository Structure
```
lab1/
├── code/        # Data cleaning, modeling, visualization scripts
├── data/        # Raw and cleaned datasets (where permitted)
├── documents/   # Assignment instructions and references
├── figs/        # Generated figures and plots
├── report/      # Final PDF report
└── README.md
```
