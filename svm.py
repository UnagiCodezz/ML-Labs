from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'word_frequency': [12, 45, 8, 52, 15, 48, 5, 60, 20, 10, 50, 42, 18, 55, 25],
    'special_chars': [2, 15, 1, 20, 3, 18, 0, 25, 5, 2, 19, 16, 4, 22, 8],
    'capital_letters': [8, 35, 5, 42, 10, 38, 3, 48, 12, 7, 40, 36, 11, 45, 15],
    'link_count': [1, 8, 0, 10, 2, 9, 0, 12, 3, 1, 9, 8, 2, 11, 4],
    'email_length': [150, 420, 120, 500, 180, 450, 100, 550, 200, 140, 480, 440, 190, 520, 250]
}

spam = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

df = pd.DataFrame(data)
X = df.values
y = np.array(spam)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

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
plt.savefig('svm_roc.png')

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for idx, feature in enumerate(df.columns):
    axes[idx].hist(df[feature], bins=6, edgecolor='black')
    axes[idx].set_title(f'{feature.replace('_', ' ').title()}')
    axes[idx].set_xlabel(feature.replace('_', ' ').title())
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True)

axes[5].axis('off')
plt.suptitle('Distribution of Email Features')
plt.tight_layout()
plt.savefig('hist-svm.png')