from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv('Mall_Customers.csv')
print(data.shape)

# Encode categorical data
encoder = LabelEncoder()
data.iloc[:, 1] = encoder.fit_transform(data.iloc[:, 1])

# Standardize the data
scalar = StandardScaler()
scaled_data = scalar.fit_transform(data)  # Keep the scaled data separate

# Calculate WCSS for the elbow method
wscc = []
for k in range(1, 20):
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    model.fit(scaled_data)
    wscc.append(model.inertia_)

# Plot the elbow curve
plt.plot(range(1, 20), wscc)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Fit the model with optimal number of clusters
optimal_k = 18
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(scaled_data)

# Create a DataFrame with clusters
data_with_clusters = pd.DataFrame(scaled_data, columns=data.columns)  # Use original column names
data_with_clusters['Cluster'] = kmeans.labels_

# Plot the clusters
plt.scatter(data_with_clusters['Annual Income (k$)'], data_with_clusters['Spending Score (1-100)'],
            c=data_with_clusters['Cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()
