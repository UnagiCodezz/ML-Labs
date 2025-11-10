from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'annual_income':[45000, 85000, 32000, 95000, 52000, 78000, 28000, 110000, 63000, 38000, 88000, 72000, 41000, 92000, 58000],
    'credit_score':[620, 750, 580, 780, 680, 720, 560, 800, 710, 590, 760, 730, 610, 790, 690],
    'loan_ammount':[15000, 35000, 12000, 45000, 22000, 30000, 10000, 50000, 28000, 14000, 38000, 32000, 18000, 42000, 25000],
    'employment_years':[3, 8, 1, 12, 5, 7, 2, 15, 6, 2, 10, 8, 4, 11, 6],
    'debt_to_income':[35, 22, 45, 18, 30, 25, 48, 15, 28, 42, 20, 24, 38, 17, 32]
}

loan_approved = [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

df = pd.DataFrame(data)
X = df.values
y = np.array(loan_approved)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.4f}%')

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(cm)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC: {roc_auc:.2f})')
plt.plot([0,1],[0,1], 'r--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.tight_layout()
plt.savefig('dt_roc.png')

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for idx, feature in enumerate(df.columns):
    axes[idx].hist(df[feature], bins=6, edgecolor='black')
    axes[idx].set_title(f'{feature.replace('_', ' ').title()}')
    axes[idx].set_xlabel(feature.replace('_', ' ').title())
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True)

axes[5].axis('off')
plt.suptitle('Distribution of Loan Application Features')
plt.tight_layout()
plt.savefig("hist-dt.png")