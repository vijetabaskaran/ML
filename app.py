import warnings
warnings.filterwarnings("ignore")
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Dataset
data = {
    "Age":[19,21,23,35,40,45,52,23,25,48],
    "Income":[15,15,16,30,50,60,70,18,20,65],
    "Spending":[39,81,6,77,60,40,20,90,55,35]
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)
# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
print("\nK-Means Clusters:")
print(kmeans_labels)
# DBSCAN Clustering
dbscan = DBSCAN(eps=1.2, min_samples=2)
dbscan_labels = dbscan.fit_predict(X_scaled)
print("\nDBSCAN Clusters:")
print(dbscan_labels)
# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X_scaled)
print("\nHierarchical Clusters:")
print(hc_labels)
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nPCA Reduced Data:")
print(X_pca)
# LDA
y = [0,1,0,1,1,0,0,1,1,0]
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled,y)
print("\nLDA Reduced Data:")
print(X_lda)
# Visualization
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels)
plt.title("Customer Clusters using K-Means")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
