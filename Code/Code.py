import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
import requests
from sklearn.decomposition import  PCA
from sklearn.cluster import  KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import cross_val_score
import warnings
%matplotlib inline

# Importing data
df = pd.read_csv("https://raw.githubusercontent.com/AVINASH-ANGILIKAM/TV-Show-Analytics-Unveiling-Popularity-Patterns/main/Data%20Set/movies.csv")

# Check the first few rows of the dataframe
df.head()


# Check for duplicate rows
duplicate_rows = df.duplicated()

# Remove duplicate rows
df.drop_duplicates(inplace=True)

df.info()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Calculate mean, median, standard deviation, skewness, and kurtosis
statistics = numeric_columns.agg(['mean', 'median', 'std', 'skew', 'kurt'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Get basic statistics using describe()
basic_statistics = numeric_columns.describe()

# Print the results
print("Mean, Median, Standard Deviation, Skewness, Kurtosis:")
print(statistics)

print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\nBasic Statistics:")
print(basic_statistics)

def plot_top_N_bar_chart(df, sort_column, n, bar_color='skyblue'):
    # Select top N entries based on sort_column
    top_N = df.nlargest(n, sort_column)
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_N['Name'], top_N[sort_column], color=bar_color)
    plt.title(f'Top {n} Entries based on {sort_column}')
    plt.xlabel('Name')
    plt.ylabel(sort_column)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_with_custom_font():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams['font.family'] = 'Arial'
        plot_top_N_bar_chart(df, 'Vote_average', 10)

plot_with_custom_font()

def plot_top_50_comparison(df, sort_column, x_column, y_column, marker='o', color='orange'):
    # Select top 50 entries based on sort_column
    top_50 = df.nlargest(50, sort_column)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(top_50[x_column], top_50[y_column], marker=marker, color=color)
    plt.title(f'Top 50 {sort_column} - {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
plot_top_50_comparison(df, 'Popularity', 'Vote_average', 'Vote_count')

def plot_correlation_heatmap(df):
    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print(numeric_df.dtypes)  # Print data types of remaining columns
    corr = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
plot_correlation_heatmap(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Select appropriate numerical features for clustering
features = ['Popularity', 'Vote_average', 'Vote_count']
X = df[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate silhouette score to find optimal number of clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

# Select the optimal number of clusters using the elbow method
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 because range starts from 2

# Perform K-means clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original DataFrame
df['Cluster'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(10, 7))

# Define colors for each cluster
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink', 'yellow', 'brown', 'gray']

for i in range(optimal_clusters):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Popularity'], cluster_data['Vote_count'], color=colors[i], label=f'Cluster {i}')

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Back scale cluster centers
for i, center in enumerate(centers):
    plt.scatter(center[0], center[2], color='black', marker='x', label=f'Cluster {i} Center')

plt.xlabel('Popularity')
plt.ylabel('Vote Count')
plt.title(f'K-Means Clustering with {optimal_clusters} Clusters')
plt.legend()
plt.grid(True)
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression


# Select appropriate numerical features for line fitting
x = df['Popularity']
y = df['Vote_count']

# Number of bootstrap samples
num_bootstraps = 1000

# Create arrays to store predictions
predictions = np.zeros((num_bootstraps, len(x)))

# Perform bootstrapping
for i in range(num_bootstraps):
    # Resample the dataset with replacement
    x_boot, y_boot = resample(x, y, replace=True, random_state=i)
    
    # Fit a linear regression model to the resampled data
    model = LinearRegression()
    model.fit(x_boot.values.reshape(-1, 1), y_boot)
    
    # Make predictions on the original data
    predictions[i] = model.predict(x.values.reshape(-1, 1))

# Calculate mean and standard deviation of predictions
mean_predictions = np.mean(predictions, axis=0)
std_predictions = np.std(predictions, axis=0)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data')

# Plot several predictions with uncertainties
for i in range(num_bootstraps):
    plt.plot(x, predictions[i], color='gray', alpha=0.1)

# Plot the mean prediction
plt.plot(x, mean_predictions, color='red', label='Mean Prediction')

# Plot confidence intervals
plt.fill_between(x, mean_predictions - 1.96 * std_predictions, mean_predictions + 1.96 * std_predictions, color='gray', alpha=0.3, label='95% Confidence Interval')

plt.xlabel('Popularity')
plt.ylabel('Vote Count')
plt.title('Linear Regression with 95% Confidence Interval')
plt.legend()
plt.grid(True)
plt.show()


