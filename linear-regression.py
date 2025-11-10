from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'opening_price': [150, 162, 145, 178, 155, 170, 185, 195, 148, 160, 180, 172, 158, 183, 192]
}
closing_prices = [152, 164, 148, 179, 157, 172, 187, 197, 150, 162, 181, 173, 159, 184, 193]

df = pd.DataFrame(data)
X = df.values
y = np.array(closing_prices)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R^2 score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

# Scatter plot: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Closing Price')
plt.ylabel('Predicted Closing Price')
plt.title('Predicted vs Actual')
plt.tight_layout()
plt.savefig('regplot.png')

# Histogram: Opening Price
plt.figure()
plt.hist(df['opening_price'], bins=6, edgecolor='black')
plt.title('Opening Price Distribution')
plt.xlabel('Opening Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hist-linear.png')