# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the CSV file
# df = pd.read_csv('Mall_Customers.csv')

# # Display the first few rows of the dataframe
# print(df.head())

# # Encode the 'Gender' column to numerical values
# label_encoder = LabelEncoder()
# df['Gender'] = label_encoder.fit_transform(df['Gender'])

# # Select the relevant features for clustering
# X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']]

# # Normalize the features
# scaler = StandardScaler()
# X_normalized = scaler.fit_transform(X)

# # Calculate WCSS for different number of clusters
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X_normalized)
#     wcss.append(kmeans.inertia_)

# # Plot the Elbow Method graph
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# # From the Elbow Method, let's assume the optimal number of clusters is 4
# optimal_clusters = 4

# # Apply K-means with the optimal number of clusters
# kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
# cluster_labels = kmeans.fit_predict(X_normalized)

# # Add cluster labels to the dataframe
# df['Cluster'] = cluster_labels

# # Display the dataframe with cluster labels
# print(df.head())

# # Plot the clusters for the first two features
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
# plt.title('Customer Segments')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# # Load the data from CSV file
# file_path = 'Mall_Customers.csv'  # Replace with your actual file path
# data = pd.read_csv(file_path)

# # Selecting features for clustering (Annual Income and Spending Score)
# X = data.iloc[:, [3, 4]].values  # Assuming Annual Income and Spending Score are columns 3 and 4 (0-indexed)

# # Elbow method to find optimal number of clusters (K)
# wcss = []  # Within-cluster sum of squares

# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)

# # Plotting the elbow method graph
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
# plt.title('Elbow Method')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('WCSS')
# plt.show()

# # Based on the Elbow graph, choose the optimal number of clusters (K)
# # Let's assume from the graph K=5 seems to be a good choice

# # Applying K-means clustering with K=5
# kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
# y_kmeans = kmeans.fit_predict(X)

# # Visualizing the clusters
# plt.figure(figsize=(10, 8))
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', marker='*', label='Centroids')
# plt.title('Clusters of customers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()

# # Adding cluster labels to the original dataframe
# data['Cluster'] = y_kmeans

# # Displaying the first few rows of the dataframe with cluster labels
# print(data.head())


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data
data = pd.read_csv("Mall_Customers.csv")

# Create DataFrame
df = pd.DataFrame(data)

# Selecting the features for clustering
X = df[['Age','Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Add the cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Display the dataframe with cluster labels
print(df)

# Plot the clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.show()
