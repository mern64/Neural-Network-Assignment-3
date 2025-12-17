import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
FILE_NAME = 'STINK3014_Assignment03_Customer_Lifestyle.csv'
RANDOM_STATE = 42

# --- 1. DATA LOADING & PREPROCESSING ---
try:
    data = pd.read_csv(FILE_NAME)
    # Clean Column Names (removing special characters for compatibility)
    data.columns = data.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
except FileNotFoundError:
    print(f"ERROR: {FILE_NAME} not found.")
    exit()

# Scale features (K-Means is distance-based and requires scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# --- 2. TASK 2a: ELBOW METHOD ---
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Task 2a: Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('k_means_elbow_plot.png')
print("Step 1: Elbow plot saved as 'k_means_elbow_plot.png'.")

# --- 3. TASK 2c & 2d: FINAL MODEL & CENTROIDS ---
# Based on the assignment requirements, we use K=4
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
kmeans_final.fit(X_scaled)

# Get centroids and inverse transform back to original scale for interpretation
cluster_centroids_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
centroids_df = pd.DataFrame(cluster_centroids_original, columns=data.columns)
centroids_df.index = [f'Cluster {i+1}' for i in range(optimal_k)]

print("\n--- Task 2d: K-Means Cluster Centroids ---")
print(centroids_df.round(2))
centroids_df.to_csv('kmeans_centroids_results.csv')