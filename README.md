
# E‑commerce Purchasing Intention – ML Classification (6 Models)

This repo implements six classification models on the Online Shoppers Purchasing Intention** dataset (UCI) to predict whether an e‑commerce session ends in a purchase (Revenue ). It computes the required metrics: **Accuracy, AUC, Precision, Recall, F1, and MCC**.

> Dataset: UCI Machine Learning Repository — *Online Shoppers Purchasing Intention*
Number of Instances: 12,330 sessions
Number of Features: 18
Data Types: Numerical and categorical
Class Balance:
  ~84.5% sessions did not result in a purchase (negative class)
  ~15.5% sessions resulted in a purchase (positive class)
Tasks: Classification and clustering
Missing Values: None  

> Source: https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset

##  Project Structure
```
ML_ASSIGNMENT/
│
├── .venv/                      # Virtual environment (optional)
├── .vs/                        # Visual Studio files
├── .vscode/                    # VS Code settings
│   └── launch.json
│
├── reports/
│   ├── metrics_summary.csv     # Model evaluation metrics
│   └── roc_curves.png          # ROC curve visualization
│__runtime.txt
├── src/
│   ├── __init__.py
│   ├── train_ecommerce.py      # Model training pipeline
│   └── __pycache__/
│
├── app.py                      # Application entry point
├── online_shoppers_intention.csv  # Dataset
├── requirements.txt            # Dependencies
└── README.md

                    

```

## Outputs:
After execution, the following files will be created inside the reports/ directory:
  metrics_summary.csv → Model evaluation metrics comparison
  roc_curves.png → ROC curves for all implemented models

##  Models Implemented (all on the SAME dataset)
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

##  Metrics Reported
- Accuracy
- ROC AUC 
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)


## Model Comparison Table

| ML Model Name        | Accuracy     | AUC          | Precision     | Recall        | F1            | MCC           |
|----------------------|--------------|--------------|--------------|--------------|--------------|--------------|
| RandomForest         | 0.896999189  | 0.919791531  | 0.771186441  | 0.476439791  | 0.588996764  | 0.554107428  |
| XGBoost              | 0.880373074  | 0.916103496  | 0.602836879  | 0.667539267  | 0.633540373  | 0.563324743  |
| LogisticRegression   | 0.841038118  | 0.893244214  | 0.491349481  | 0.743455497  | 0.591666667  | 0.514501483  |
| KNN                  | 0.873073804  | 0.826662379  | 0.714285714  | 0.30104712   | 0.423572744  | 0.40859792   |
| GaussianNB           | 0.272911598  | 0.733018837  | 0.17262181   | 0.97382199   | 0.293259756  | 0.128884013  |
| DecisionTree         | 0.851987024  | 0.717874155  | 0.522193211  | 0.523560209  | 0.522875817  | 0.435283523  |


## Observations on Model Performance

| ML Model Name        | Observation about Model Performance |
|----------------------|------------------------------------|
| Logistic Regression  | Good at identifying purchasing sessions (high recall 0.74), but predicts many false positives (precision 0.49). Linear model works reasonably well on this dataset. |
| Decision Tree        | Balanced precision and recall (~0.52), but lower AUC (0.72) indicates weak ranking ability and possible overfitting. |
| kNN                  | High precision (0.71) but very low recall (0.30), meaning it predicts purchase correctly when it does, but misses many actual buyers. |
| Naive Bayes          | Extremely high recall (0.97) but very low precision (0.17) and poor accuracy (0.27); model predicts almost all sessions as purchase due to strong independence assumption. |
| Random Forest        | Highest AUC (0.92) with good precision (0.77) and MCC (0.55); strong generalization and reliable ranking of sessions. |
| XGBoost              | Balanced precision (0.60) and recall (0.67) with highest MCC (0.56) and high AUC (0.92); best overall balanced performance considering class imbalance. |



