## Author
**Name:** Mohmad Yaqoob  
**Roll No:** DA25M017  
**Course:** DA5401 - Data Analytics Laboratory  
**Institute:** IIT Madras  
**Date:** 27 October 2025  

# DA5401 – Assignment 7: Model Selection

This report presents the complete implementation and analysis for **Assignment 7: Model Selection**, part of the *Data Analytics Laboratory (DA5401)* course.  
The project evaluates multiple supervised learning models using the **Landsat Satellite dataset (`sat.trn` / `sat.tst`)**.

---

## Overview

The goals of this assignment were to:  
- Implement and compare **multiple machine learning models** on a multi-class classification task.  
- Evaluate models using **Accuracy, Weighted F1, ROC–AUC, and Average Precision (PRC–AP)**.  
- Understand **threshold-dependent vs. threshold-independent performance**.  
- Explore **probability calibration, ensemble learning**, and **failure cases (AUC < 0.5)**.

---

## Assignment Structure

| Part | Task | Description |
|:-----|:-----|:-------------|
| **A** | Baseline Model Training | Trained six base models (KNN, SVC, Decision Tree, Logistic Regression, Naive Bayes, Dummy) and evaluated F1-scores. |
| **B** | ROC–AUC Analysis | Computed One-vs-Rest ROC curves and identified the best performing model by AUC. |
| **C** | PRC–AP Analysis | Computed macro-averaged Precision–Recall curves to assess precision–recall trade-offs. |
| **D** | Final Recommendation | Synthesized F1, ROC–AUC, and PRC–AP results to recommend the optimal model. |
| **E** | Brownie Points | Extended analysis with RandomForest, XGBoost, Voting Ensemble, and an intentionally poor (AUC < 0.5) model. |

---

## Dataset

- **Source:** Landsat Satellite Dataset (`sat.trn`, `sat.tst`)  
- **Features:** 36 numerical attributes  
- **Classes:** 5 land-cover categories (after removing class 7)  
- **Shape:** `(4927, 36)`  

Labels were remapped from `[1–5]` to `[0–4]` for compatibility with scikit-learn and XGBoost.

---

## Models Evaluated

| Model | Category | Description |
|:------|:--------|:------------|
| **KNN** | Instance-based | Classifies samples based on nearest neighbors in feature space. |
| **SVC (RBF)** | Margin-based | Separates classes using nonlinear hyperplanes. |
| **Decision Tree** | Tree-based | Recursive partitioning; can overfit small data subsets. |
| **Logistic Regression** | Linear | OvR classifier with interpretable coefficients. |
| **Naive Bayes (Gaussian)** | Probabilistic | Assumes conditional independence of features. |
| **Dummy Classifier** | Baseline | Predicts the majority class as reference. |
| **RandomForest** | Ensemble (Bagging) | Averages multiple decision trees to reduce variance. |
| **XGBoost** | Ensemble (Boosting) | Sequentially improves weak learners using gradient updates. |
| **Voting Ensemble** | Hybrid | Combines SVC, RandomForest, and XGBoost using soft voting. |
| **Inverted Classifier** | Synthetic | Demonstrates AUC < 0.5 by inverting predicted probabilities. |

---

## Evaluation Metrics

- **Accuracy** – Fraction of correct predictions.  
- **Weighted F1-Score** – Harmonic mean of precision and recall, weighted by class frequency.  
- **Macro ROC–AUC (OvR)** – Measures class separability, independent of threshold.  
- **Macro Average Precision (PRC–AP)** – Area under precision-recall curve; robust for imbalanced classes.

---

## Baseline Evaluation Results

| Model            | Macro ROC–AUC | Avg Precision (AP) | Accuracy  | Weighted F1 |
|:-----------------|:-------------:|:-----------------:|:---------:|:-----------:|
| VotingEnsemble    | 0.991080      | 0.965060          | 0.933063  | 0.931648    |
| RandomForest      | 0.991007      | 0.967117          | 0.932049  | 0.930617    |
| XGBoost           | 0.989647      | 0.966189          | 0.929006  | 0.927847    |
| SVC               | 0.988418      | 0.957941          | 0.923935  | 0.922663    |
| KNN               | 0.987383      | 0.953977          | 0.933063  | 0.932003    |
| LogisticRegression| 0.968913      | 0.894157          | 0.883367  | 0.877395    |
| NaiveBayes        | 0.959927      | 0.863124          | 0.831643  | 0.834948    |
| DecisionTree      | 0.926282      | 0.806677          | 0.896552  | 0.895857    |
| Dummy             | 0.500000      | 0.200000          | 0.311359  | 0.147853    |
| InvertedModel     | 0.031087      | 0.110304          | 0.001014  | 0.001563    |

---

## Key Insights

- **Voting Ensemble** achieved the **highest ROC–AUC (0.9911)** and strong **Average Precision (0.9651)**, slightly outperforming individual models.  
- **RandomForest** and **XGBoost** performed almost equally well, confirming the effectiveness of tree-based ensembles.  
- **SVC** and **KNN** performed strongly but slightly lower than ensemble models.  
- **Logistic Regression** showed moderate performance, limited by linear boundaries.  
- **Naive Bayes** and **Decision Tree** were weaker, while **Dummy Classifier** performed at random.  
- **Inverted Classifier** illustrates AUC < 0.5: flipping probabilities reverses correct vs. incorrect ranking.

---

## Conceptual Takeaways

- **AUC < 0.5** indicates an inverted decision boundary; flipping predictions would immediately improve performance.  
- Ensemble methods like **VotingClassifier, RandomForest, and XGBoost** balance bias and variance, yielding stable probability estimates across thresholds.  
- Calibrated ensembles produce the most reliable performance in high-dimensional and overlapping datasets.

---

## Final Recommendation

- **Best Individual Model:** `XGBoost` – High precision, stable recall, excellent ROC separation.  
- **Best Overall Model:** `Voting Ensemble (SVC + RF + XGB)` – Highest overall metrics and smooth performance across thresholds.  
- **Educational Highlight:** `Inverted Model` demonstrates the concept of *AUC < 0.5*, emphasizing the importance of probability calibration.

---
