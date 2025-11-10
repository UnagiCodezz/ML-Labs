from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    "amount":[220, 330, 180, 410, 260, 320, 170, 420, 270, 370, 240, 390, 230, 430, 380],
    "previous_visits":[1, 7, 2, 8, 3, 6, 1, 9, 3, 8, 2, 7, 2, 9, 5]
}

target = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]

df = pd.DataFrame(data)
X = df.values
y = np.array(target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GradientBoostingClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(F"\nConfusion Matrix:")
print(cm)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC: {auc:.2f})')
plt.plot([0, 1],[0, 1], 'r--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('boosting_roc.png')

plt.figure()
plt.hist(df['amount'], bins=6, edgecolor='black')
plt.title('Purchase Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hist-boosting.png')