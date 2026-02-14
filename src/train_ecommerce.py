
import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Please download the dataset and place it there.")
    df = pd.read_csv(path)
    # Some mirrors use lowercase/underscore names; normalize column names minimally
    df.columns = [c.strip().replace(' ', '_') for c in df.columns]
    return df


def build_preprocessor(df: pd.DataFrame):
    # Define target and features based on UCI schema
    target = 'Revenue'
    if target not in df.columns:
        # try lowercase
        if 'Revenue'.lower() in [c.lower() for c in df.columns]:
            # map original to exact name
            for c in df.columns:
                if c.lower() == 'revenue':
                    target = c
                    break
        else:
            raise KeyError("Target column 'Revenue' not found in CSV.")

    # Numeric features per UCI description
    numeric_features = [
        'Administrative','Administrative_Duration',
        'Informational','Informational_Duration',
        'ProductRelated','ProductRelated_Duration',
        'BounceRates','ExitRates','PageValues','SpecialDay'
    ]
    # Handle case-insensitive / variant column names
    colmap = {c.lower(): c for c in df.columns}
    resolved_numeric = [colmap[n.lower()] for n in numeric_features if n.lower() in colmap]

    # Categorical features (treat integer coded as categorical)
    categorical_features = ['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
    resolved_categorical = [colmap[n.lower()] for n in categorical_features if n.lower() in colmap]

    # Safety: coerce Weekend/Revenue to int/bool
    if 'Weekend' in resolved_categorical:
        df[colmap['weekend']] = df[colmap['weekend']].astype(str)
    # Ensure target is binary 0/1
    y_series = df[target].astype(int) if df[target].dtype != bool else df[target].astype(int)

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, resolved_numeric),
            ('cat', categorical_transformer, resolved_categorical)
        ]
    )

    X = df.drop(columns=[target])
    y = y_series.values
    return X, y, preprocessor


def build_models(pos_weight: float) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        'LogisticRegression': LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=None, solver='lbfgs'),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=15),
        'GaussianNB': GaussianNB(),
        'RandomForest': RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight='balanced'),
        'XGBoost': XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=pos_weight
        )
    }
    return models


def evaluate_models(models: Dict[str, Any], preprocessor, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(10, 8))
    for name, clf in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Probabilities for AUC (class 1)
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            # Plot ROC curve
            RocCurveDisplay.from_predictions(y_test, y_proba, name=name, alpha=0.8)
        else:
            # Fallback: use decision function if available, else set NaN
            if hasattr(pipe.named_steps['clf'], 'decision_function'):
                scores = pipe.decision_function(X_test)
                # Convert to [0,1] range via rank-based min-max as approximation
                s_min, s_max = scores.min(), scores.max()
                y_proba = (scores - s_min) / (s_max - s_min + 1e-9)
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        rows.append({
            'model': name,
            'accuracy': acc,
            'auc': auc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'mcc': mcc
        })

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.title('ROC Curves – Online Shoppers Purchasing Intention')
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/roc_curves.png', dpi=180)

    df_metrics = pd.DataFrame(rows).sort_values(by='auc', ascending=False)
    df_metrics.to_csv('reports/metrics_summary.csv', index=False)
    return df_metrics


def main():
    parser = argparse.ArgumentParser(description='Train 6 ML models on the Online Shoppers dataset and compute metrics.')
    parser.add_argument('--data_csv', type=str, required=True, help='Path to online_shoppers_intention.csv')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data_csv)
    X, y, preprocessor = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # Compute positive class weight for XGBoost
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = (neg / max(pos, 1))

    models = build_models(pos_weight)
    metrics_df = evaluate_models(models, preprocessor, X_train, X_test, y_train, y_test)

    print('\n=== Metrics Summary ===')
    print(metrics_df.to_string(index=False))
    print("\nSaved: reports/metrics_summary.csv and reports/roc_curves.png")


if __name__ == '__main__':
    main()
