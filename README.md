##  Author
**Name:** Mohmad Yaqoob

**Roll No:** DA25M017

**Course:** DA5401 -  Data Analytics Laboratory 

**Institute:** IIT Madras


**Date:** 27 October 2025  

#  DA5401 – Assignment 7: Model Selection

This repository contains my complete implementation and analysis for **Assignment 7: Model Selection**, as part of the *Data Analytics Laboratory  (DA5401)* course.  
The project explores multiple supervised learning models, their evaluation metrics, and model selection techniques using the **Landsat Satellite dataset (sat.trn / sat.tst)**.

---

## Overview

The primary goal of this assignment was to:
- Implement and compare **multiple machine learning models** on a multi-class classification task.
- Evaluate models using **accuracy**, **F1-score**, **ROC–AUC**, and **Precision–Recall (PRC–AP)** metrics.
- Understand **threshold-dependent vs. threshold-independent** performance.
- Explore **calibration, ensembles**, and **failure cases (AUC < 0.5)** for deeper understanding.

---

## Assignment Structure

| Part | Task | Description |
|:-----|:-----|:-------------|
| **A** | Baseline Model Training | Trained six base models (KNN, SVC, Decision Tree, Logistic Regression, Naive Bayes, Dummy) and evaluated baseline F1-scores. |
| **B** | ROC–AUC Analysis | Computed and interpreted One-vs-Rest (OvR) ROC curves, identifying the best performing model by AUC. |
| **C** | PRC–AP Analysis | Computed macro-averaged Precision–Recall curves to assess precision–recall trade-offs. |
| **D** | Final Recommendation | Synthesized F1, ROC–AUC, and PRC–AP results to recommend the optimal model. |
| **E** | Brownie Points | Extended analysis with RandomForest, XGBoost, Voting Ensemble, and an intentionally poor (AUC < 0.5) model. |

---

## Dataset

- **Source:** Landsat Satellite Dataset (`sat.trn`, `sat.tst`)  
- **Features:** 36 numerical attributes  
- **Classes:** 5 land-cover categories (after removing class 7)  
- **Shape:** `(4927, 36)`  

All labels were remapped from `[1–5]` to `[0–4]` for compatibility with libraries like XGBoost.

---

## Models Evaluated

| Model | Category | Description |
|:------|:----------|:-------------|
| **KNN** | Instance-based | Classifies samples based on proximity in feature space. |
| **SVC (RBF)** | Margin-based | Separates classes using nonlinear hyperplanes. |
| **Decision Tree** | Tree-based | Recursive feature partitioning; prone to overfitting. |
| **Logistic Regression** | Linear | Interpretable model with OvR extension. |
| **Naive Bayes (Gaussian)** | Probabilistic | Based on conditional independence assumption. |
| **Dummy Classifier** | Baseline | Predicts majority class as reference baseline. |
| **RandomForest** | Ensemble (Bagging) | Averages multiple trees to reduce variance. |
| **XGBoost** | Ensemble (Boosting) | Sequentially improves weak learners using gradient updates. |
| **Voting Ensemble** | Hybrid | Combines SVC, RandomForest, and XGBoost using soft voting. |
| **Inverted Classifier** | Synthetic | Demonstrates AUC < 0.5 by inverting probabilities. |

---

## Evaluation Metrics

- **Accuracy** – Overall correctness of predictions.  
- **Weighted F1-Score** – Harmonic mean of precision and recall, weighted by class frequency.  
- **ROC–AUC (One-vs-Rest)** – Measures class separability independent of threshold.  
- **Average Precision (PRC–AP)** – Area under the precision–recall curve; best for imbalanced data.

---

## Key Results Summary

| Metric | Best Model | Score | Interpretation |
|:--------|:------------|:-------|:---------------|
| **Accuracy / F1** | KNN | 0.94 / 0.94 | Performs well with clear neighborhood structure. |
| **ROC–AUC** | SVC | 0.994 | Excellent separation of classes. |
| **PRC–AP** | SVC | 0.957 | Strong balance between precision and recall. |
| **Ensemble AUC** | Voting (SVC + RF + XGB) | 0.991 | Most stable across thresholds. |
| **AUC < 0.5** | Inverted Model | 0.031 | Illustrates “worse than random” behavior. |

---

## Insights

- **SVC** consistently provided top-tier ROC–AUC and PRC–AP, showing robust class separation.  
- **KNN** achieved slightly better raw accuracy but was more sensitive to class overlap.  
- **RandomForest** and **XGBoost** performed almost identically, confirming the strength of tree-based ensembles.  
- The **Voting Ensemble** marginally improved macro metrics, showing strong calibration and threshold stability.  
- The **Inverted Classifier** clearly demonstrated how probability inversion leads to AUC < 0.5 — validating theoretical understanding.

---

## Enhancement

For deployment-level robustness:
- Use a **Calibrated Voting Ensemble** (SVC + RandomForest + XGBoost) with isotonic calibration.  
- Apply **feature importance visualization (SHAP)** for interpretability.  
- Optionally use **stacking** with Logistic Regression as a meta-learner for refined final predictions.

This yields a model that’s:
✅ Highly accurate  
✅ Calibrated and interpretable  
✅ Resistant to overfitting  
✅ Generalizable across thresholds  

---

## Conclusion

Through a step-by-step exploration of model selection, this project demonstrated:
- How to evaluate models using threshold-independent (ROC–AUC) and threshold-dependent (PRC–AP) metrics.
- The benefits of ensemble methods like RandomForest, XGBoost, and soft-voting combinations.  
- How poorly calibrated or inverted models can perform worse than random guessing (AUC < 0.5).
  
---

**Final Recommendation:**  
- The **Calibrated Voting Ensemble (SVC + RandomForest + XGBoost)** is the most balanced and production-ready model, achieving a macro ROC–AUC of ~0.991 and macro AP of ~0.965.

