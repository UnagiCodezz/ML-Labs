from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'amount': [200, 300, 150, 400, 250, 330, 180, 410, 270, 360, 230, 380, 210, 390, 340],
    'previous_purchases': [2, 5, 1, 7, 3, 6, 2, 8, 3, 7, 2, 6, 2, 8, 5]
}

target = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]

df = pd.DataFrame(data)
X = df.values
y = np.array(target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.4f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC: {auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.title('ROC Curve')
plt.tight_layout()
plt.savefig('bagging_roc.png')


plt.figure()
plt.hist(df['amount'], bins=6, edgecolor='black')
plt.title('Purchase Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hist-bagging.png')