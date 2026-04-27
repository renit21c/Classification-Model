import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# STEP 1: Load Dataset

print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv('citrus.csv')
print(f"Dataset Shape  : {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn Names   : {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")

# STEP 2: Exploratory Data Analysis
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print(f"\nClass Distribution:")
print(df['name'].value_counts())
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nStatistical Summary:\n{df.describe()}")

# Plot EDA
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis - Orange vs Grapefruit', fontsize=14, fontweight='bold')

features = ['diameter', 'weight', 'red', 'green', 'blue']
colors = ['#FF8C00', "#DA3939"]

for i, feature in enumerate(features):
    ax = axes[i // 3][i % 3]
    for j, label in enumerate(df['name'].unique()):
        subset = df[df['name'] == label][feature]
        ax.hist(subset, bins=30, alpha=0.6, color=colors[j], label=label)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()

# Class distribution pie chart
ax = axes[1][2]
counts = df['name'].value_counts()
ax.pie(counts, labels=counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90)
ax.set_title('Class Distribution')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] eda_plots.png")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] correlation_heatmap.png")


# STEP 3: Data Preprocessing
print("\n" + "=" * 60)
print("STEP 3: DATA PREPROCESSING")
print("=" * 60)

# Encode target label
le = LabelEncoder()
df['label'] = le.fit_transform(df['name'])
print(f"Label Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Define features (X) and target (y)
X = df[['diameter', 'weight', 'red', 'green', 'blue']]
y = df['label']

print(f"\nFeatures shape : {X.shape}")
print(f"Target shape   : {y.shape}")

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set size : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test set size  : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")

# Feature Scaling (needed for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n[Done] Feature scaling applied (StandardScaler)")

# STEP 4: Model Training
print("\n" + "=" * 60)
print("STEP 4: MODEL TRAINING")
print("=" * 60)

# Model 1: Decision Tree
print("\n[1] Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
dt_model.fit(X_train, y_train)
print("    Decision Tree Berhasil Ditrain!")

# Model 2: Naive Bayes
print("\n[2] Training Naive Bayes (Gaussian) Classifier...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
print("    Naive Bayes Berhasil Ditrain!")

# Model 3: Support Vector Machine
print("\n[3] Training Support Vector Machine (SVM) Classifier...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)
svm_model.fit(X_train_scaled, y_train)
print("    SVM Berhasil Ditrain!")

# STEP 5: Prediction
print("\n" + "=" * 60)
print("STEP 5: PREDICTION")
print("=" * 60)

y_pred_dt  = dt_model.predict(X_test)
y_pred_nb  = nb_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test_scaled)

print("Predictions completed for all 3 models.")

# STEP 6: Evaluation
print("\n" + "=" * 60)
print("STEP 6: MODEL EVALUATION")
print("=" * 60)

models = {
    'Decision Tree': y_pred_dt,
    'Naive Bayes':   y_pred_nb,
    'SVM':           y_pred_svm
}

results = []
for model_name, y_pred in models.items():
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')
    results.append({
        'Model': model_name,
        'Accuracy':  round(acc, 4),
        'Precision': round(prec, 4),
        'Recall':    round(rec, 4),
        'F1-Score':  round(f1, 4)
    })
    print(f"\n{'─'*40}")
    print(f"  {model_name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(results_df.to_string(index=False))

# STEP 7: Cross Validation
print("\n" + "=" * 60)
print("STEP 7: CROSS VALIDATION (5-Fold)")
print("=" * 60)

cv_models = {
    'Decision Tree': (dt_model,  X, y),
    'Naive Bayes':   (nb_model,  X, y),
    'SVM':           (svm_model, pd.DataFrame(scaler.fit_transform(X), columns=X.columns), y)
}

cv_results = {}
for name, (model, Xd, yd) in cv_models.items():
    scores = cross_val_score(model, Xd, yd, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"\n  {name}:")
    print(f"    CV Scores : {scores.round(4)}")
    print(f"    Mean      : {scores.mean():.4f}")
    print(f"    Std Dev   : {scores.std():.4f}")


# STEP 8: Visualization of Results
print("\n" + "=" * 60)
print("STEP 8: VISUALIZING RESULTS")
print("=" * 60)

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

model_preds = [
    ('Decision Tree', y_pred_dt),
    ('Naive Bayes',   y_pred_nb),
    ('SVM',           y_pred_svm)
]

for ax, (name, y_pred) in zip(axes, model_preds):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_, ax=ax)
    acc = accuracy_score(y_test, y_pred)
    ax.set_title(f'{name}\nAccuracy: {acc:.4f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] confusion_matrices.png")

# Model Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25
bar_colors = ['#2196F3', '#4CAF50', '#FF5722']

ax = axes[0]
for i, row in results_df.iterrows():
    vals = [row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']]
    bars = ax.bar(x + i * width, vals, width, label=row['Model'], color=bar_colors[i], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

ax.set_ylim(0.8, 1.02)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_title('Metrics Comparison')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Cross-Validation Box Plot
ax2 = axes[1]
cv_data = [cv_results['Decision Tree'], cv_results['Naive Bayes'], cv_results['SVM']]
bp = ax2.boxplot(cv_data, labels=['Decision\nTree', 'Naive\nBayes', 'SVM'],
                 patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Cross-Validation Accuracy')
ax2.set_title('5-Fold CV Score Distribution')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] model_comparison.png")

# Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns.tolist(),
          class_names=le.classes_, filled=True, fontsize=8,
          max_depth=3, rounded=True)
plt.title('Decision Tree (max_depth=3 displayed)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print("[Saved] decision_tree_plot.png")


# STEP 9: Conclusion
print("\n" + "=" * 60)
print("STEP 9: CONCLUSION")
print("=" * 60)

best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_acc   = results_df['Accuracy'].max()

print(f"\n  Best Model : {best_model}")
print(f"  Accuracy   : {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"\n  Full Comparison:")
print(results_df.to_string(index=False))
print("\n  Output Gambar Berhasil Disimpan. Selesai!")
print("=" * 60)
