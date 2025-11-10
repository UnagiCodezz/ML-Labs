from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'account_age_months': [24, 6, 36, 12, 48, 18, 60, 3, 30, 9, 42, 15, 54, 21, 27],
    'monthly_charges': [85, 120, 65, 110, 70, 105, 60, 130, 75, 115, 68, 108, 62, 112, 80],
    'support_tickets': [5, 12, 2, 8, 1, 7, 0, 15, 3, 10, 2, 9, 1, 11, 4],
    'usage_frequency': [45, 20, 60, 30, 65, 35, 70, 15, 55, 25, 62, 32, 68, 28, 50],
    'satisfaction_score': [6, 3, 8, 5, 9, 5, 9, 2, 7, 4, 8, 5, 9, 4, 7]
}

churn = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

df = pd.DataFrame(data)
X = df.values
y = np.array(churn)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
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
plt.savefig('logistic_roc.png')


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for idx, feature in enumerate(df.columns):
    axes[idx].hist(df[feature], bins=6, edgecolor="black")
    axes[idx].set_title(f'{feature.replace('_', " ").title()}')
    axes[idx].set_xlabel(feature.replace('_', ' ').title())
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True)

axes[5].axis('off')
plt.suptitle('Distribution of Customer Features')
plt.tight_layout()
plt.savefig('hist-log.png')