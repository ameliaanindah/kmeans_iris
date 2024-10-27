# Import tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import data
df = pd.read_csv('iris_cleaned.csv')  # Pastikan file ini ada di direktori yang sama
print(df.head())  # Melihat 5 data pertama

# Amati bentuk data
print("\nShape of the dataset:", df.shape)

# Melihat ringkasan statistik deskriptif dari DataFrame 
print("\nRingkasan statistik deskriptif:")
print(df.describe())

# Cek null data
print("\nCek data yang missing value:")
print(df.isnull().sum())

# Cek outlier dengan boxplot
plt.figure(figsize=(12, 6))
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].boxplot()
plt.title('Outlier Check Using Boxplot')
plt.show()

# Amati bentuk visual masing-masing fitur
plt.style.use('fivethirtyeight')
plt.figure(1, figsize=(15, 6))
n = 0
for x in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    n += 1
    plt.subplot(2, 2, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], kde=True, stat="density", kde_kws=dict(cut=3), bins=20)
    plt.title('Distplot of {}'.format(x))
plt.show()

# Merancang K-Means untuk variabel yang relevan
# Menggunakan kolom yang akan digunakan dalam clustering
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Menentukan nilai k yang sesuai dengan Elbow-Method
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=111)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot bentuk visual elbow
plt.figure(1, figsize=(15, 6))
plt.plot(range(1, 11), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Membangun K-Means
optimal_k = 3  # Tentukan nilai optimal k berdasarkan elbow plot
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, max_iter=300, random_state=111)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Melihat bentuk visual cluster
plt.figure(1, figsize=(15, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=200, alpha=0.5, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', alpha=0.5, label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Customer Segmentation Based on K-Means Clustering')
plt.legend()
plt.show()

# Melihat nilai Silhouette Score
score = silhouette_score(X, labels)
print("Silhouette Score: ", score)
