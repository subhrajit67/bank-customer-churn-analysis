#!/usr/bin/env python3
# =============================================================================
# Bank Customer Churn Prediction — Full Pipeline
# =============================================================================
# Author  : Subhrajit Majumder
# GitHub  : https://github.com/subhrajit67
# LinkedIn: https://www.linkedin.com/in/subhrajit-majumder-b45421252/
# Dataset : Kaggle — Credit Card Customer Churn (10,000 records)
# Tech    : Python · Pandas · Scikit-Learn · XGBoost · SMOTE · Matplotlib
# =============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipeline
import joblib

# ── Output Directory ──────────────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)

# ── Plot Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
PALETTE  = {'Retained': '#4CAF50', 'Churned': '#E53935'}
DIVIDER  = '=' * 70


def save(name: str) -> None:
    """Save current figure to outputs/ and close."""
    path = f'outputs/{name}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved  outputs/{name}")


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print(DIVIDER)
print("1. LOADING DATASET")
print(DIVIDER)

df = pd.read_csv('Churn_Modelling.csv')

print(f"Shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"Missing values : {df.isna().sum().sum()}")
print(f"\nPreview:\n{df.head(3).to_string()}")
print(f"\nStatistical Summary:\n{df.describe().round(2).to_string()}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
print(f"\n{DIVIDER}")
print("2. EXPLORATORY DATA ANALYSIS")
print(DIVIDER)

# 2.1 Target Distribution
churn_rate = df['Exited'].mean() * 100
print(f"\nOverall churn rate : {churn_rate:.1f}%")
print(f"Retained           : {(df['Exited']==0).sum():,}")
print(f"Churned            : {(df['Exited']==1).sum():,}")

fig, ax = plt.subplots(figsize=(5, 5))
counts = df['Exited'].value_counts()
ax.pie(counts, labels=['Retained', 'Churned'],
       autopct='%1.1f%%', startangle=140,
       colors=['#4CAF50', '#E53935'],
       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Customer Churn Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
save('churn_distribution.png')

# 2.2 Numerical Outliers
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df[col], ax=ax, color='#64B5F6',
                flierprops=dict(marker='o', markerfacecolor='#E53935', markersize=4))
    ax.set_title(col, fontsize=9)
plt.suptitle('Outlier Check - Numerical Features', fontsize=12, fontweight='bold')
plt.tight_layout()
save('boxplots_numerical.png')

# 2.3 Zero-Balance Insight
zero_bal  = df['Balance'] == 0
print(f"\nZero-balance customers   : {zero_bal.sum():,} ({zero_bal.mean()*100:.1f}%)")
print(f"Churn - zero balance     : {df.loc[zero_bal,'Exited'].mean()*100:.1f}%")
print(f"Churn - non-zero balance : {df.loc[~zero_bal,'Exited'].mean()*100:.1f}%  (higher)")

# 2.4 Categorical Distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cat_plots = [
    ('Geography',      df['Geography'].value_counts(),                            '#42A5F5'),
    ('Gender',         df['Gender'].value_counts(),                               '#EF5350'),
    ('Has Credit Card',df['HasCrCard'].map({1:'Yes',0:'No'}).value_counts(),     '#66BB6A'),
    ('Is Active Member',df['IsActiveMember'].map({1:'Active',0:'Inactive'}).value_counts(),'#FFA726'),
]
for ax, (title, counts, color) in zip(axes.flat, cat_plots):
    ax.bar(counts.index, counts.values, color=color, edgecolor='white')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Count')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.suptitle('Categorical Feature Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
save('categorical_distributions.png')

# 2.5 Feature vs Churn
df['Churn Status'] = df['Exited'].map({0: 'Retained', 1: 'Churned'})
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature Distributions by Churn Status', fontsize=14, fontweight='bold')
for (feat, title, ax) in [
    ('CreditScore',     'Credit Score vs Churn',   axes[0,0]),
    ('Age',             'Age vs Churn',             axes[0,1]),
    ('Tenure',          'Tenure vs Churn',          axes[1,0]),
    ('Balance',         'Balance vs Churn',         axes[1,1]),
    ('EstimatedSalary', 'Est. Salary vs Churn',     axes[1,2]),
]:
    sns.boxplot(data=df, y=feat, x='Churn Status',
                order=['Retained', 'Churned'], palette=PALETTE, ax=ax)
    ax.set_title(title); ax.set_xlabel('')
axes[0,2].axis('off')
plt.tight_layout()
save('feature_vs_churn.png')

# 2.6 Geography & Gender Churn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
geo_churn = (df.groupby('Geography')['Exited'].mean() * 100).sort_values(ascending=False)
axes[0].bar(geo_churn.index, geo_churn.values, color=['#E53935','#FB8C00','#43A047'])
for i, v in enumerate(geo_churn.values):
    axes[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Churn Rate by Geography (%)', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)'); axes[0].set_ylim(0, 40)

gender_churn = (df.groupby('Gender')['Exited'].mean() * 100).sort_values(ascending=False)
axes[1].bar(gender_churn.index, gender_churn.values, color=['#42A5F5','#EC407A'])
for i, v in enumerate(gender_churn.values):
    axes[1].text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
axes[1].set_title('Churn Rate by Gender (%)', fontweight='bold')
axes[1].set_ylabel('Churn Rate (%)'); axes[1].set_ylim(0, 32)
plt.tight_layout()
save('geo_gender_churn.png')

# 2.7 Activity & Products vs Churn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
activity = (df.groupby(df['IsActiveMember'].map({1:'Active',0:'Inactive'}))['Exited']
             .mean() * 100).reindex(['Inactive','Active'])
axes[0].bar(activity.index, activity.values, color=['#E53935','#4CAF50'])
for i, v in enumerate(activity.values):
    axes[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('Churn Rate: Active vs Inactive', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)'); axes[0].set_ylim(0, 35)

prod_churn = (df.groupby('NumOfProducts')['Exited'].mean() * 100)
axes[1].bar(prod_churn.index.astype(str), prod_churn.values, color='#7E57C2')
for i, v in enumerate(prod_churn.values):
    axes[1].text(i, v+1, f'{v:.0f}%', ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('Churn Rate by Number of Products', fontweight='bold')
axes[1].set_xlabel('Number of Products')
axes[1].set_ylabel('Churn Rate (%)'); axes[1].set_ylim(0, 120)
plt.tight_layout()
save('active_products_churn.png')


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print(f"\n{DIVIDER}")
print("3. FEATURE ENGINEERING")
print(DIVIDER)

df['CreditUtilization']    = df['Balance']  / (df['CreditScore'] + 1)
df['EngagementScore']      = df['NumOfProducts'] + df['HasCrCard'] + df['IsActiveMember']
df['BalanceToSalaryRatio'] = df['Balance']  / (df['EstimatedSalary'] + 1)
df['AgeTenureRatio']       = df['Age']      / (df['Tenure'] + 1)
df['CreditScoreGroup']     = pd.cut(df['CreditScore'],
                                    bins=[0,579,669,739,799,850],
                                    labels=['Very Poor','Fair','Good','Very Good','Exceptional'],
                                    include_lowest=True)

new_feats = ['CreditUtilization','EngagementScore','BalanceToSalaryRatio',
             'AgeTenureRatio','CreditScoreGroup']
print(f"New features: {new_feats}")

# 3.1 Correlation Heatmap
drop_for_corr = ['RowNumber','CustomerId','Surname','CreditScoreGroup','Churn Status']
num_df = df.drop([c for c in drop_for_corr if c in df.columns], axis=1
                 ).select_dtypes(include='number')
corr   = num_df.corr()
mask   = np.triu(np.ones_like(corr, dtype=bool), k=1)

plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
save('correlation_heatmap.png')

print(f"\nTop correlations with Exited:")
print(corr['Exited'].drop('Exited').abs().sort_values(ascending=False).head(8).round(3).to_string())


# =============================================================================
# 4. PREPROCESSING
# =============================================================================
print(f"\n{DIVIDER}")
print("4. PREPROCESSING")
print(DIVIDER)

le = LabelEncoder()
for col in ['Geography', 'Gender', 'CreditScoreGroup']:
    df[col] = le.fit_transform(df[col])

drop_cols = ['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Churn Status']
X = df.drop([c for c in drop_cols if c in df.columns], axis=1).select_dtypes(include='number')
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scale_cols = ['Age','CreditScore','Balance','EstimatedSalary',
              'CreditUtilization','BalanceToSalaryRatio','AgeTenureRatio']
scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

print(f"Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,} | Features : {X_train.shape[1]}")
print(f"Train churn rate : {y_train.mean()*100:.1f}%")


# =============================================================================
# 5. MODEL TRAINING
# =============================================================================
print(f"\n{DIVIDER}")
print("5. MODEL TRAINING & EVALUATION")
print(DIVIDER)

models = {
    'Logistic Regression'   : LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'K-Nearest Neighbors'   : imb_pipeline(SMOTE(random_state=42), KNeighborsClassifier(n_neighbors=7)),
    'Support Vector Machine': imb_pipeline(SMOTE(random_state=42), SVC(probability=True, random_state=42)),
    'Random Forest'         : RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    'XGBoost'               : XGBClassifier(
                                  eval_metric='logloss', verbosity=0, random_state=42,
                                  scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                  n_estimators=200),
    'Gradient Boosting'     : imb_pipeline(SMOTE(random_state=42),
                                  GradientBoostingClassifier(n_estimators=200, random_state=42)),
}

results = []
for name, model in models.items():
    print(f"\n  Training : {name}")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"  Accuracy={acc:.4f}  Recall={rec:.4f}  F1={f1:.4f}  ROC-AUC={roc:.4f}")
    results.append({'Model': name, 'Accuracy': acc, 'Recall': rec, 'F1 Score': f1, 'ROC AUC': roc})

results_df = (pd.DataFrame(results)
              .sort_values('ROC AUC', ascending=False)
              .reset_index(drop=True))

print(f"\n{DIVIDER}")
print("MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(results_df)); w = 0.20
for i, (m, c) in enumerate(zip(['Accuracy','Recall','F1 Score','ROC AUC'],
                                ['#42A5F5','#66BB6A','#FFA726','#EF5350'])):
    ax.bar(x + i*w, results_df[m], w, label=m, color=c, alpha=0.9)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(results_df['Model'], rotation=18, ha='right', fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.8)
plt.tight_layout()
save('model_comparison.png')


# =============================================================================
# 6. HYPERPARAMETER TUNING
# =============================================================================
print(f"\n{DIVIDER}")
print("6. HYPERPARAMETER TUNING — GridSearchCV (Gradient Boosting)")
print(DIVIDER)

param_grid = {
    'gradientboostingclassifier__n_estimators' : [200, 300],
    'gradientboostingclassifier__max_depth'    : [3, 4, 5],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1],
    'gradientboostingclassifier__subsample'    : [0.8, 1.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(
    imb_pipeline(SMOTE(random_state=42), GradientBoostingClassifier(random_state=42)),
    param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

tuned = gs.best_estimator_
y_pred_t = tuned.predict(X_test)
y_prob_t = tuned.predict_proba(X_test)[:, 1]

print(f"\nBest params : {gs.best_params_}")
print(f"Best CV AUC : {gs.best_score_:.4f}")
print(f"\nTuned model — test set:")
print(classification_report(y_test, y_pred_t, target_names=['Retained','Churned']))
print(f"ROC AUC  : {roc_auc_score(y_test, y_prob_t):.4f}")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_t),
                       display_labels=['Retained','Churned']).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix - Tuned Gradient Boosting', fontweight='bold')
plt.tight_layout()
save('confusion_matrix.png')


# =============================================================================
# 7. SAVE ARTEFACTS
# =============================================================================
print(f"\n{DIVIDER}")
print("7. SAVING ARTEFACTS")
print(DIVIDER)

joblib.dump(tuned,  'best_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("  -> best_churn_model.pkl  (tuned GradientBoosting + SMOTE pipeline)")
print("  -> scaler.pkl            (StandardScaler fitted on training data)")

# =============================================================================
# 8. INFERENCE DEMO
# =============================================================================
print(f"\n{DIVIDER}")
print("8. INFERENCE DEMO")
print(DIVIDER)

sample  = X_test.iloc[:5].copy()
preds   = tuned.predict(sample)
probas  = tuned.predict_proba(sample)[:, 1]
for i, (p, pr) in enumerate(zip(preds, probas)):
    label = 'Churned  [!]' if p == 1 else 'Retained [OK]'
    print(f"  Customer {i+1}: {label}  (churn probability: {pr:.1%})")

print(f"\n{DIVIDER}")
print("PIPELINE COMPLETE — all plots saved to outputs/")
print(DIVIDER)
