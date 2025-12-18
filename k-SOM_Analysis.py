import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from math import sqrt, ceil

# --- CONFIGURATION ---
FILE_NAME = 'STINK3014_Assignment03_Customer_Lifestyle.csv'
RANDOM_STATE = 42
LEARNING_RATE = 0.5
SIGMA = 1.0
EPOCHS = 500

# --- 1. DATA LOADING & PREPROCESSING ---
data = pd.read_csv(FILE_NAME)
# Remove non-numeric columns if they exist (like 'ID' or 'Cluster_Label')
data_numeric = data.select_dtypes(include=[np.number]).drop(columns=['ID'], errors='ignore')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_numeric)

# --- 2. KSOM EXPERIMENTS ---
K_values = [4]
som_results = []


def get_grid_shape(N):
    r = int(ceil(sqrt(N)))
    c = int(ceil(N / r))
    return (r, c)


print(f"Starting KSOM Training (LR={LEARNING_RATE}, Iterations={EPOCHS})...")

for N in K_values:
    M_rows, N_cols = get_grid_shape(N)
    som = MiniSom(M_rows, N_cols, X_scaled.shape[1],
                  sigma=SIGMA, learning_rate=LEARNING_RATE, random_seed=RANDOM_STATE)

    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, num_iteration=EPOCHS)

    # Task 3c(i): Quantisation Error (QE)
    qe = som.quantization_error(X_scaled)

    # Task 3c(ii): Topographic Error (TE)
    # TE is the proportion of all data vectors for which first and second BMUs are not adjacent.
    te = som.topographic_error(X_scaled)

    som_results.append({
        'K': N,
        'Grid': f'{M_rows}x{N_cols}',
        'QE': round(qe, 4),
        'TE': round(te, 4)
    })

results_df = pd.DataFrame(som_results)
print("\n--- Task 3c: KSOM Performance Metrics ---")
print(results_df.set_index('K'))
results_df.to_csv('ksom_performance_metrics.csv')

# --- 3. TASK 3d: EXTRACT CLUSTER CHARACTERISTICS FOR LLM INTERPRETATION ---
print("\n--- Task 3d: Extracting Cluster Profiles ---")

# 1. Get the weight vectors (scaled centroids) from the trained SOM
# Note: Since your loop only ran for K=4, the 'som' variable still holds that model.
weights = som.get_weights()

# 2. Reshape the weights from a 2D grid (2x2) into a flat list of 4 clusters
centroids_scaled = weights.reshape(-1, X_scaled.shape[1])

# 3. Inverse transform the weights back to the original scale (Age, BMI, etc.)
centroids_original = scaler.inverse_transform(centroids_scaled)

# 4. Create a clean DataFrame for the results
ksom_profiles = pd.DataFrame(centroids_original, columns=data_numeric.columns)
ksom_profiles.index = [f'KSOM_Cluster_{i+1}' for i in range(len(centroids_original))]

# 5. Display and save the profiles
print("\nKSOM Cluster Centroids (Original Scale):")
print(ksom_profiles.round(2))
ksom_profiles.to_csv('ksom_cluster_profiles.csv')

print("\nSuccess: Use the table above or 'ksom_cluster_profiles.csv' for your LLM interpretation task.")