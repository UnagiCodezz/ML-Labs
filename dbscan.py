from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'purchase_frequency': [2, 15, 3, 18, 5, 16, 1, 20, 4, 2, 17, 14, 6, 19, 7],
    'avg_transaction': [50, 280, 65, 310, 85, 295, 45, 340, 95, 55, 305, 275, 105, 330, 115]
}

df = pd.DataFrame(data)
X = df.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DBSCAN(eps=0.8, min_samples=2)
clusters = model.fit_predict(X_scaled)

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f'Number of clusters: {n_clusters}')
print(f'Number of Noise Points: {n_noise}')
print(f'Cluster labels: {clusters}')

if n_clusters>1 and n_noise<len(clusters):
    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)
    print(f'\nSilhouette Score: {silhouette:.4f}')
    print(f'\nDavies Bouldin Score: {davies_bouldin:.4f}')

plt.scatter(df['purchase_frequency'], df['avg_transaction'], c=clusters)
plt.xlabel('Purchase Frequency')
plt.ylabel('Average Transaction')
plt.title('DBSCAN Clusters')
plt.savefig('dbscan-scatter.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['purchase_frequency'], bins=6, edgecolor='black')
axes[0].set_title('Purchase Frequency')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['avg_transaction'], bins=6, edgecolor='black')
axes[1].set_title('Average Transaction')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('hist-dbscan.png')