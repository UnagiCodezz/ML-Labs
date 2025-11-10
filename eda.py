import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    'age': [25, 30, 22, np.nan, 28, 35, np.nan, 40, 23, 31],
    'education_level': ['Bachelor', 'Master', 'PhD', 'Master', np.nan, 'Bachelor', 'PhD', 'Bachelor', 'Master', np.nan]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

df['age'].fillna(df['age'].mean(), inplace=True)

df['education_level'].fillna(df['education_level'].mode()[0], inplace=True)

df['education_encoded'] = df['education_level'].astype('category').cat.codes

print("\nAfter Preprocessing:")
print(df)

plt.hist(df['age'], bins=6, edgecolor='black', color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hist-eda.png')

edu_counts = df['education_level'].value_counts()
plt.figure()
plt.bar(edu_counts.index, edu_counts.values, color='lightgreen', edgecolor='black')
plt.title('Education Level Counts')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda-bar.png')