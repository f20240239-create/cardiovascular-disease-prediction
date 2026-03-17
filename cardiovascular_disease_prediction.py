# =============================================================================
# PROJECT 1: CARDIOVASCULAR DISEASE PREDICTION
# =============================================================================
# Requirements: pip install pandas numpy matplotlib seaborn scikit-learn
# Dataset: cardio_train.csv (semicolon-separated)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             ConfusionMatrixDisplay)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set global style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'

print("=" * 65)
print("     CARDIOVASCULAR DISEASE PREDICTION — MINOR PROJECT")
print("=" * 65)

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("\n[1/6] Loading dataset...")

try:
    df = pd.read_csv("cardio_train.csv", sep=";")
    print(f"      Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print("      'cardio_train.csv' not found. Generating synthetic demo data...")
    np.random.seed(42)
    n = 5000
    age_days = np.random.randint(10000, 25000, n)
    gender = np.random.randint(1, 3, n)
    height = np.random.randint(150, 200, n)
    weight = np.random.uniform(50, 130, n)
    ap_hi = np.random.randint(90, 200, n)
    ap_lo = np.random.randint(60, 130, n)
    cholesterol = np.random.randint(1, 4, n)
    gluc = np.random.randint(1, 4, n)
    smoke = np.random.randint(0, 2, n)
    alco = np.random.randint(0, 2, n)
    active = np.random.randint(0, 2, n)
    cardio = ((age_days > 18000).astype(int) +
              (ap_hi > 140).astype(int) +
              (cholesterol > 1).astype(int) +
              (weight > 90).astype(int) > 1).astype(int)
    df = pd.DataFrame({
        'id': range(n), 'age': age_days, 'gender': gender,
        'height': height, 'weight': weight, 'ap_hi': ap_hi,
        'ap_lo': ap_lo, 'cholesterol': cholesterol, 'gluc': gluc,
        'smoke': smoke, 'alco': alco, 'active': active, 'cardio': cardio
    })
    print(f"      Synthetic dataset created: {df.shape[0]} rows × {df.shape[1]} columns")

# =============================================================================
# SECTION 2: DATA PRE-PROCESSING
# =============================================================================
print("\n[2/6] Data pre-processing...")

# Drop ID column if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

print(f"\n--- Basic Info ---")
print(df.dtypes)
print(f"\n--- Missing Values ---")
print(df.isnull().sum())
print(f"\n--- Statistical Summary ---")
print(df.describe().round(2))

# Convert age from days to years
if df['age'].max() > 1000:
    df['age'] = (df['age'] / 365.25).round(1)
    print("\n      Age converted from days → years.")

# Feature engineering
df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

# Remove physiologically impossible blood pressure values
before = len(df)
df = df[(df['ap_hi'] >= 60)  & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 40)  & (df['ap_lo'] <= 200)]
df = df[df['ap_hi'] > df['ap_lo']]
df = df[(df['height'] >= 100) & (df['height'] <= 220)]
df = df[(df['weight'] >= 30)  & (df['weight'] <= 200)]
after = len(df)
print(f"\n      Outliers removed: {before - after} rows | Remaining: {after}")

print("\n      Pre-processing complete ✔")

# =============================================================================
# SECTION 3: DATA VISUALISATIONS
# =============================================================================
print("\n[3/6] Generating visualisations...")

TARGET_COL = 'cardio'
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

# ── PLOT 1: Target Distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("1. Target Variable Distribution", fontsize=14, fontweight='bold')

counts = df[TARGET_COL].value_counts()
labels = ['No Disease', 'Disease']
axes[0].pie(counts, labels=labels, autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[0].set_title("Pie Chart")

axes[1].bar(labels, counts.values, color=['#4CAF50', '#F44336'],
            edgecolor='white', linewidth=1.5)
for i, v in enumerate(counts.values):
    axes[1].text(i, v + 100, str(v), ha='center', fontweight='bold')
axes[1].set_title("Count Plot")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("plot1_target_distribution.png", bbox_inches='tight')
plt.show()
print("      plot1_target_distribution.png saved")

# ── PLOT 2: Age Distribution by Cardio ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("2. Age Distribution", fontsize=14, fontweight='bold')

for label, grp in df.groupby(TARGET_COL):
    axes[0].hist(grp['age'], bins=30, alpha=0.6,
                 label=labels[label], color=['#4CAF50', '#F44336'][label])
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Age Histogram by Target")
axes[0].legend()

sns.boxplot(data=df, x=TARGET_COL, y='age', ax=axes[1],
            palette=['#4CAF50', '#F44336'])
axes[1].set_xticklabels(labels)
axes[1].set_title("Age Boxplot by Target")
axes[1].set_xlabel("")
axes[1].set_ylabel("Age (years)")

plt.tight_layout()
plt.savefig("plot2_age_distribution.png", bbox_inches='tight')
plt.show()
print("      plot2_age_distribution.png saved")

# ── PLOT 3: Blood Pressure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("3. Blood Pressure Analysis", fontsize=14, fontweight='bold')

sns.boxplot(data=df, x=TARGET_COL, y='ap_hi', ax=axes[0],
            palette=['#4CAF50', '#F44336'])
axes[0].set_xticklabels(labels)
axes[0].set_title("Systolic BP (ap_hi) by Target")
axes[0].set_xlabel("")

sns.boxplot(data=df, x=TARGET_COL, y='ap_lo', ax=axes[1],
            palette=['#4CAF50', '#F44336'])
axes[1].set_xticklabels(labels)
axes[1].set_title("Diastolic BP (ap_lo) by Target")
axes[1].set_xlabel("")

plt.tight_layout()
plt.savefig("plot3_blood_pressure.png", bbox_inches='tight')
plt.show()
print("      plot3_blood_pressure.png saved")

# ── PLOT 4: BMI Distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("4. BMI & Weight Distribution", fontsize=14, fontweight='bold')

for label, grp in df.groupby(TARGET_COL):
    axes[0].hist(grp['bmi'], bins=40, alpha=0.6,
                 label=labels[label], color=['#4CAF50', '#F44336'][label])
axes[0].set_xlim(10, 60)
axes[0].set_xlabel("BMI")
axes[0].set_ylabel("Frequency")
axes[0].set_title("BMI Histogram")
axes[0].legend()

sns.violinplot(data=df, x=TARGET_COL, y='weight', ax=axes[1],
               palette=['#4CAF50', '#F44336'])
axes[1].set_xticklabels(labels)
axes[1].set_title("Weight Violin Plot by Target")
axes[1].set_xlabel("")

plt.tight_layout()
plt.savefig("plot4_bmi_weight.png", bbox_inches='tight')
plt.show()
print("      plot4_bmi_weight.png saved")

# ── PLOT 5: Categorical Features ──────────────────────────────────────────────
cat_feats = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender']
cat_labels = {
    'cholesterol': ['Normal', 'Above Normal', 'Well Above'],
    'gluc':        ['Normal', 'Above Normal', 'Well Above'],
    'smoke':       ['No', 'Yes'],
    'alco':        ['No', 'Yes'],
    'active':      ['No', 'Yes'],
    'gender':      ['Female', 'Male']
}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("5. Categorical Features vs Target", fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, feat in enumerate(cat_feats):
    ct = pd.crosstab(df[feat], df[TARGET_COL], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[i], color=['#4CAF50', '#F44336'],
            edgecolor='white', linewidth=0.8)
    axes[i].set_title(feat.capitalize())
    axes[i].set_ylabel("Percentage (%)")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].legend(labels, title='Cardio', fontsize=8)

plt.tight_layout()
plt.savefig("plot5_categorical_features.png", bbox_inches='tight')
plt.show()
print("      plot5_categorical_features.png saved")

# ── PLOT 6: Pairplot (sample) ─────────────────────────────────────────────────
sample = df[['age', 'bmi', 'ap_hi', 'ap_lo', TARGET_COL]].sample(
    min(1500, len(df)), random_state=42)
pair_fig = sns.pairplot(sample, hue=TARGET_COL, plot_kws={'alpha': 0.4},
                        palette={0: '#4CAF50', 1: '#F44336'}, corner=True)
pair_fig.figure.suptitle("6. Pairplot: Key Features", y=1.01, fontsize=13,
                          fontweight='bold')
pair_fig.savefig("plot6_pairplot.png", bbox_inches='tight')
plt.show()
print("      plot6_pairplot.png saved")

# =============================================================================
# SECTION 4: CORRELATION MATRIX
# =============================================================================
print("\n[4/6] Correlation matrix...")

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap=cmap, center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.7}, ax=ax)
ax.set_title("Correlation Matrix of All Features", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("plot7_correlation_matrix.png", bbox_inches='tight')
plt.show()
print("      plot7_correlation_matrix.png saved")

# Top correlations with target
print("\n      Top correlations with 'cardio':")
corr_target = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
print(corr_target.to_string())

# =============================================================================
# SECTION 5: MODEL TRAINING & ACCURACY
# =============================================================================
print("\n[5/6] Training ML models...")

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

MODELS = {
    "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True,
                                         random_state=42, C=1.0),
    "K-Nearest Neighbor (KNN)":     KNeighborsClassifier(n_neighbors=7),
    "Decision Tree (DT)":           DecisionTreeClassifier(max_depth=8,
                                                            random_state=42),
    "Logistic Regression (LR)":     LogisticRegression(max_iter=1000,
                                                        random_state=42),
    "Random Forest (RF)":           RandomForestClassifier(n_estimators=100,
                                                            random_state=42,
                                                            n_jobs=-1),
}

results = {}
print(f"\n{'Model':<35} {'Train Acc':>10} {'Test Acc':>10} {'CV Mean':>10}")
print("-" * 68)

for name, model in MODELS.items():
    model.fit(X_train_sc, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_sc))
    test_acc  = accuracy_score(y_test,  model.predict(X_test_sc))
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5,
                                scoring='accuracy', n_jobs=-1)
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
    }
    print(f"{name:<35} {train_acc*100:>9.2f}% {test_acc*100:>9.2f}% "
          f"{cv_scores.mean()*100:>9.2f}%")

# ── PLOT 8: Accuracy Comparison ───────────────────────────────────────────────
model_names  = [n.split('(')[-1].rstrip(')') for n in results.keys()]
test_accs    = [v['test_acc'] * 100 for v in results.values()]
cv_means     = [v['cv_mean']  * 100 for v in results.values()]
cv_stds      = [v['cv_std']   * 100 for v in results.values()]

x = np.arange(len(model_names))
width = 0.35
colors = ['#2196F3', '#9C27B0', '#FF9800', '#4CAF50', '#F44336']

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy',
               color=colors, alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, cv_means,  width, label='CV Accuracy (5-fold)',
               color=colors, alpha=0.45, edgecolor='white',
               yerr=cv_stds, capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Accuracy (%)")
ax.set_title("8. Model Accuracy Comparison", fontsize=14, fontweight='bold')
ax.set_ylim(50, 100)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.4)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("plot8_accuracy_comparison.png", bbox_inches='tight')
plt.show()
print("\n      plot8_accuracy_comparison.png saved")

# =============================================================================
# SECTION 6: BEST MODEL — DETAILED EVALUATION
# =============================================================================
print("\n[6/6] Detailed evaluation of best model...")

best_name = max(results, key=lambda k: results[k]['test_acc'])
best      = results[best_name]
best_model = best['model']

print(f"\n      Best Model: {best_name}")
print(f"      Test Accuracy: {best['test_acc']*100:.2f}%")

y_pred  = best_model.predict(X_test_sc)
y_prob  = best_model.predict_proba(X_test_sc)[:, 1]

print(f"\n--- Classification Report: {best_name} ---")
print(classification_report(y_test, y_pred,
                             target_names=['No Disease', 'Disease']))

# ── PLOT 9: Confusion Matrix + ROC Curve ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"9. Best Model Evaluation — {best_name}",
             fontsize=13, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No Disease', 'Disease'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='#2196F3', lw=2,
             label=f'ROC AUC = {roc_auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend(loc='lower right')
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig("plot9_confusion_roc.png", bbox_inches='tight')
plt.show()
print("      plot9_confusion_roc.png saved")

# ── PLOT 10: Feature Importance (RF) ─────────────────────────────────────────
rf_model = results["Random Forest (RF)"]['model']
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = ['#F44336' if importances.index[i] == importances.idxmax()
              else '#2196F3' for i in range(len(importances))]
importances.plot(kind='barh', ax=ax, color=colors_bar[::-1], edgecolor='white')
ax.set_title("10. Feature Importance (Random Forest)", fontsize=13,
             fontweight='bold')
ax.set_xlabel("Importance Score")
ax.axvline(importances.mean(), color='orange', linestyle='--',
           linewidth=1.5, label=f'Mean = {importances.mean():.3f}')
ax.legend()
plt.tight_layout()
plt.savefig("plot10_feature_importance.png", bbox_inches='tight')
plt.show()
print("      plot10_feature_importance.png saved")

# ── PLOT 11: All Confusion Matrices ──────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("11. Confusion Matrices — All Models", fontsize=13,
             fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['model'].predict(X_test_sc))
    short = name.split('(')[-1].rstrip(')')
    ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{short}\n{res['test_acc']*100:.1f}%", fontsize=10)

plt.tight_layout()
plt.savefig("plot11_all_confusion_matrices.png", bbox_inches='tight')
plt.show()
print("      plot11_all_confusion_matrices.png saved")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("                  PROJECT SUMMARY")
print("=" * 65)
print(f"  Dataset Shape    : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Features Used    : {X.shape[1]}")
print(f"  Train / Test     : {len(X_train)} / {len(X_test)}")
print()
print(f"  {'Model':<35} {'Test Acc':>10} {'CV Acc':>10}")
print("  " + "-" * 57)
for name, res in sorted(results.items(),
                         key=lambda x: x[1]['test_acc'], reverse=True):
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<35} {res['test_acc']*100:>9.2f}% "
          f"{res['cv_mean']*100:>9.2f}%{marker}")
print()
print(f"  Best Model       : {best_name}")
print(f"  Best Test Acc    : {best['test_acc']*100:.2f}%")
print(f"  ROC-AUC Score    : {roc_auc:.4f}")
print()
print("  Plots saved:")
plots = [
    "plot1_target_distribution.png",
    "plot2_age_distribution.png",
    "plot3_blood_pressure.png",
    "plot4_bmi_weight.png",
    "plot5_categorical_features.png",
    "plot6_pairplot.png",
    "plot7_correlation_matrix.png",
    "plot8_accuracy_comparison.png",
    "plot9_confusion_roc.png",
    "plot10_feature_importance.png",
    "plot11_all_confusion_matrices.png",
]
for p in plots:
    print(f"    ✔  {p}")
print("=" * 65)
print("  Minor Project Complete!")
print("=" * 65)
