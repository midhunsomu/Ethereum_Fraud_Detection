# Ethereum Fraud Detection System

A machine learning pipeline that classifies Ethereum wallet addresses as fraudulent or legitimate based on on-chain transaction behavior, applying data science to a real-world cybersecurity problem.

## Problem Statement

Fraudulent activity on the Ethereum blockchain (scams, phishing wallets, exit scams) is difficult to detect manually due to transaction volume and the irreversible nature of blockchain transfers. This project builds a binary classifier to flag suspicious addresses using historical transaction patterns.

## Dataset

- **Source:** [Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) (Kaggle)
- **Size:** 9,841 wallet addresses, 51 raw features
- **Target:** `FLAG` (1 = fraudulent, 0 = legitimate)
- **Class distribution:** ~78% legitimate / ~22% fraudulent (imbalanced)
- **Features:** transaction timing/frequency stats, sent/received Ether volumes, ERC20 token activity, contract interactions

## Pipeline

| Stage | Approach |
|---|---|
| **Data cleaning** | Dropped identifier columns (`Address`, `Index`); imputed missing ERC20 numeric fields with 0 and categorical fields with `'Unknown'` |
| **Feature encoding** | One-hot encoded categorical features (`pd.get_dummies`), expanding to 815 features |
| **Train/test split** | 70/30 stratified split to preserve class ratio |
| **Modeling** | Logistic Regression with `class_weight='balanced'` to counter class imbalance |
| **Evaluation** | Precision, recall, F1-score, confusion matrix |

## Results

| Metric | Class 0 (Legit) | Class 1 (Fraud) |
|---|---|---|
| Precision | 0.83 | **0.84** |
| Recall | 0.98 | **0.30** |
| F1-score | 0.90 | 0.45 |

**Overall accuracy: 83.3%**

- **High precision (0.84)** on fraud predictions means flagged addresses are very likely to actually be fraudulent — low false-positive risk.
- **Low recall (0.30)** means the model only catches 30% of actual fraud cases — most fraudulent wallets currently go undetected.

## Key Insight & Next Steps

Precision-recall tradeoff is the central finding: the model is conservative, prioritizing confidence over coverage. For a fraud-detection system, **missed fraud (false negatives) is typically costlier than false alarms**, so improving recall is the priority.

**Planned improvements:**
- Resampling techniques (SMOTE, undersampling) instead of/alongside class weighting
- Tree-based models (Random Forest, XGBoost) to capture nonlinear patterns
- Threshold tuning on predicted probabilities to shift the precision/recall balance
- Feature selection/dimensionality reduction (815 features is high relative to 9,841 samples)

## Tech Stack
Python · pandas · scikit-learn · seaborn/matplotlib

## How to Run
1. Open `ethereum_fraud_detection.ipynb` in Jupyter/Colab
2. Run all cells sequentially (dataset loads automatically via `kagglehub`)
