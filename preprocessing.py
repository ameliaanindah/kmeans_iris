import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('iris.csv')  # Ganti dengan nama file yang sesuai

print("\nData yang telah dibaca:")
print(df.head())  # Hanya menampilkan 5 data pertama
print("============================================================")

# Detect missing values
print("\nDeteksi data yang memiliki missing value:")
missvalue = df.isna().sum()
print(missvalue)
print("============================================================")

print("\nBaris dengan missing values:")
print(df[df.isna().any(axis=1)])
print("============================================================")

# Function to detect outliers using z-score
def detect_outlier(column_data, threshold=3):
    mean = np.mean(column_data)
    std = np.std(column_data)
    z_scores = np.abs((column_data - mean) / std)
    return z_scores > threshold

# Columns to check for outliers
columns_to_check = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
outlier_indices = []

# Identify and collect outliers
for column in columns_to_check:
    outliers = detect_outlier(df[column])
    outlier_indices.extend(df[outliers].index)

# Drop duplicate indices (if any) and remove outliers
outlier_indices = list(set(outlier_indices))
df_cleaned = df.drop(outlier_indices)

print("\nData setelah menghapus outliers:")
print(df_cleaned.head())
print("Jumlah data setelah penghapusan outliers:", df_cleaned.shape[0])
print("============================================================")

# Handle missing values by dropping rows with any missing values
df_cleaned = df_cleaned.dropna()

print("\nMissing values setelah semua replace, drop, dan outlier removal:")
print(df_cleaned.isna().sum())
print("============================================================")

# Save cleaned data to a new CSV file
output_file_cleaned = 'iris_cleaned.csv'
df_cleaned.to_csv(output_file_cleaned, index=False)
print(f"\nData yang telah dipre-processing disimpan ke {output_file_cleaned}")
