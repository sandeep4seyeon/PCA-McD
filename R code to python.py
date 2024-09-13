# Used Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import sklearn.utils
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Load the data
mcdonalds = pd.read_csv('mcdonalds.csv')
# Display all the columns
print(mcdonalds.columns.tolist())
# Display the dimensions
print(mcdonalds.shape)
# Display the first 3 rows
print(mcdonalds.head(3))
# Extract Data
MD_x = mcdonalds.iloc[:, 0:11]
# Extract from the First 11 columns & Convert "Yes" to 1 and others to 0
MD_x = (MD_x == "Yes").astype(int)
col_means = MD_x.mean().round(2)
# Display the column means
print(col_means)
# Perform PCA
pca = PCA()
pca.fit(MD_x)
std_dev = np.sqrt(pca.explained_variance_)
proportion_variance = pca.explained_variance_
cumulative_variance = np.cumsum(proportion_variance)
# Display
print("\nImportance of components:")

results = pd.DataFrame({
    'Standard deviation': np.round(std_dev, 5),
    'Proportion of Variance': np.round(proportion_variance, 5),
    'Cumulative Proportion': np.round(cumulative_variance, 5)
})

# Rename the index to match PC1, PC2, etc.
results.index = [f'PC{i+1}' for i in range(len(std_dev))]
results = results.T
# Display the results
print(results)
loadings = pca.components_.T

# Create a DataFrame for loadings with appropriate row and column names
loadings_df = pd.DataFrame(np.round(loadings, 3),
                           index=MD_x.columns,
                           columns=[f'PC{i+1}' for i in range(loadings.shape[1])])

# Print the standard deviations
print("\nStandard deviations (1, .., p=11):")
print(np.round(std_dev, 1))

# Print the rotation matrix (loadings)
print("\nRotation (n x k) = (11 x 11):")
print(loadings_df)

MD_x_pca = pca.fit_transform(MD_x)
plt.figure(figsize=(10, 7))
plt.scatter(MD_x_pca[:, 0], MD_x_pca[:, 1], color='grey', alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')

# Plotting projection axes similar to projAxes in R
# Scale factor for visual clarity
scale_factor = 2

# Loadings for PC1 and PC2
loadings = pca.components_.T[:, :2]

# Plot projection lines for each variable
for i, (x, y) in enumerate(loadings):
    plt.arrow(0, 0, x * scale_factor, y * scale_factor, color='red', alpha=0.7,
              head_width=0.05, head_length=0.1)
    plt.text(x * scale_factor * 1.1, y * scale_factor * 1.1, MD_x.columns[i],
             color='blue', ha='center', va='center')

plt.grid(True)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.show()
# Load the data (assuming it's already loaded)
# mcdonalds = pd.read_csv('mcdonalds.csv')  # Uncomment if loading for the first time

# Extract the first 11 columns and convert "Yes" to 1 and "No" to 0
MD_x = mcdonalds.iloc[:, 0:11]
MD_x = (MD_x == "Yes").astype(int)

# Optional: Scaling the data (not strictly necessary for binary data but shown for completeness)
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x)

# Set random seed for reproducibility
np.random.seed(1234)

# Range of cluster numbers to test
cluster_range = range(2, 9)
inertia = []  # For within-cluster sum of squares

# Perform k-means clustering for each number of clusters
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x_scaled)

    # Append the inertia (within-cluster sum of squares) to the list
    inertia.append(kmeans.inertia_)

# Plotting the inertia values as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(cluster_range, inertia, color='skyblue')
plt.xlabel('number of segments')
plt.ylabel('sum of within cluster distance')
plt.title('Bar Graph of Inertia for Different Cluster Counts')
plt.xticks(cluster_range)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Define parameters for bootstrapping and clustering
cluster_range = range(2, 9)
n_boot = 100  # Number of bootstrap samples
n_rep = 10    # Number of repetitions per clustering
ari_scores = {k: [] for k in cluster_range}  # Dictionary to store ARI scores for each cluster number

# Perform bootstrapping and clustering
for k in cluster_range:
    # Perform clustering on the original dataset
    kmeans_original = KMeans(n_clusters=k, n_init=n_rep, random_state=1234)
    kmeans_original.fit(MD_x_scaled)
    original_labels = kmeans_original.labels_

    # Perform bootstrapping and compute ARI scores
    for _ in range(n_boot):
        # Create a bootstrap sample
        MD_x_bootstrap, _ = sklearn.utils.resample(MD_x_scaled, original_labels, random_state=None)

        # Fit KMeans to the bootstrap sample
        kmeans_bootstrap = KMeans(n_clusters=k, n_init=n_rep, random_state=None)
        kmeans_bootstrap.fit(MD_x_bootstrap)

        # Compute ARI between original clustering and bootstrapped clustering
        bootstrap_labels = kmeans_bootstrap.labels_
        # Ensure the labels length matches for ARI computation
        if len(original_labels) == len(bootstrap_labels):
            ari = adjusted_rand_score(original_labels, bootstrap_labels)
            ari_scores[k].append(ari)

# Convert the ARI scores into a DataFrame for easy plotting
ari_df = pd.DataFrame.from_dict(ari_scores, orient='index').transpose()

# Plotting the boxplot of ARI scores
plt.figure(figsize=(10, 6))
ari_df.boxplot(grid=False)
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Stability of Clusters: Adjusted Rand Index (ARI)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Fit K-Means with 4 clusters
kmeans_4 = KMeans(n_clusters=4, n_init=10, random_state=1234)
cluster_labels = kmeans_4.fit_predict(MD_x_scaled)

# Create a DataFrame that includes cluster labels
clustered_data = pd.DataFrame(MD_x_scaled, columns=MD_x.columns)
clustered_data['Cluster'] = cluster_labels

# Plotting the histograms (Gorge Plot Equivalent)
plt.figure(figsize=(12, 8))

# Loop through each cluster to plot histograms
for cluster_num in range(4):
    cluster_data = clustered_data[clustered_data['Cluster'] == cluster_num]
    plt.hist(cluster_data.values.flatten(), bins=20, range=(0, 1), alpha=0.6, label=f'Cluster {cluster_num + 1}')

plt.xlabel('Value Range (Scaled)')
plt.ylabel('Frequency')
plt.title('Gorge Plot for 4-Cluster Solution')
plt.xlim(0, 1)  # Match the xlim as requested
plt.legend(title='Clusters')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Fit KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
kmeans.fit(MD_x_scaled)
original_labels = kmeans.labels_

# Number of bootstrap samples
n_bootstrap = 100
n_clusters = 4
segment_stabilities = np.zeros(n_clusters)

# Iterate over each cluster number (0 through 3 for 4 clusters)
for cluster_num in range(n_clusters):
    stability_scores = []

    # Bootstrap sampling
    for _ in range(n_bootstrap):
        # Resample the data with replacement
        X_resampled, y_resampled = sklearn.utils.resample(MD_x_scaled, original_labels, random_state=None)

        # Fit KMeans on resampled data
        kmeans_resampled = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
        kmeans_resampled.fit(X_resampled)
        resampled_labels = kmeans_resampled.labels_

        # Calculate the stability score using Adjusted Rand Index (ARI) for each cluster
        ari_score = adjusted_rand_score(y_resampled, resampled_labels)
        stability_scores.append(ari_score)

    # Append average stability score for each cluster
    segment_stabilities[cluster_num] = np.mean(stability_scores)

# Plotting the segment stability
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_clusters + 1), segment_stabilities, color='skyblue')
plt.ylim(0, 1)  # Set the y-axis limit to match the R plot
plt.xlabel("Segment Number")
plt.ylabel("Segment Stability")
plt.title("Segment Level Stability Within Solutions (SLSW) Plot")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Define the range of clusters to test
k_range = range(2, 9)
bic_scores = []
aic_scores = []

# Fit Gaussian Mixture Models for different numbers of clusters
for k in k_range:
    gmm = GaussianMixture(n_components=k, n_init=10, random_state=1234)
    gmm.fit(MD_x_scaled)
    bic_scores.append(gmm.bic(MD_x_scaled))
    aic_scores.append(gmm.aic(MD_x_scaled))

# Plot BIC and AIC scores to determine the optimal number of clusters
plt.figure(figsize=(12, 6))
plt.plot(k_range, bic_scores, marker='o', label='BIC')
plt.plot(k_range, aic_scores, marker='o', label='AIC')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Model Selection: BIC and AIC for Different Numbers of Clusters')
plt.legend()
plt.grid(True)
plt.show()

# Display results
best_k_bic = k_range[np.argmin(bic_scores)]
best_k_aic = k_range[np.argmin(aic_scores)]
print(f"Optimal number of clusters based on BIC: {best_k_bic}")
print(f"Optimal number of clusters based on AIC: {best_k_aic}")
# Define the range of clusters to test
k_range = range(2, 9)
bic_scores = []
aic_scores = []

# Fit Gaussian Mixture Models for different numbers of clusters
for k in k_range:
    gmm = GaussianMixture(n_components=k, n_init=10, random_state=1234)
    gmm.fit(MD_x_scaled)
    bic_scores.append(gmm.bic(MD_x_scaled))
    aic_scores.append(gmm.aic(MD_x_scaled))

# Plot BIC and AIC scores
plt.figure(figsize=(12, 6))
plt.plot(k_range, bic_scores, marker='o', label='BIC')
plt.plot(k_range, aic_scores, marker='o', label='AIC')
plt.xlabel('Number of Clusters')
plt.ylabel('Value of Information Criteria')
plt.title('Model Selection: BIC and AIC for Different Numbers of Clusters')
plt.legend()
plt.grid(True)
plt.show()

# Fit KMeans and GMM with 4 clusters
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
kmeans_labels = kmeans.fit_predict(MD_x_scaled)

gmm_4 = GaussianMixture(n_components=4, n_init=10, random_state=1234)
gmm_labels = gmm_4.fit_predict(MD_x_scaled)

# Compare clustering solutions
kmeans_cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
gmm_cluster_counts = pd.Series(gmm_labels).value_counts().sort_index()

print("KMeans Clustering Counts:")
print(kmeans_cluster_counts)
print("GMM Clustering Counts:")
print(gmm_cluster_counts)

# Compute log-likelihood for GMM
log_likelihood_gmm = gmm_4.score_samples(MD_x_scaled).sum()
print(f'Log-Likelihood for GMM: {log_likelihood_gmm}')

# Compare cluster assignments between KMeans and GMM
ari_score = adjusted_rand_score(kmeans_labels, gmm_labels)
print(f'Adjusted Rand Index (ARI) between KMeans and GMM: {ari_score:.3f}')

mcdonalds['Like'] = pd.Categorical(mcdonalds['Like']).codes
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Print the frequency of the transformed 'Like.n'
like_n_counts = mcdonalds['Like.n'].value_counts().sort_index()
print("Frequency of 'Like.n':")
print(like_n_counts)

# Create a formula string for the regression model
columns = mcdonalds.columns[0:11]  # Assuming the first 11 columns are the predictors
formula = 'Like.n ~ ' + ' + '.join(columns)
print("Formula:")
print(formula)
mcdonalds['Like'] = pd.Categorical(mcdonalds['Like']).codes
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

mcdonalds['Like'] = pd.Categorical(mcdonalds['Like']).codes
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Reverse and transform the 'Like' column
mcdonalds['Like'] = pd.Categorical(mcdonalds['Like']).codes
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Prepare data for model fitting
# Convert categorical columns to numeric if any
categorical_columns = mcdonalds.select_dtypes(include=['object']).columns
for col in categorical_columns:
    mcdonalds[col] = pd.Categorical(mcdonalds[col]).codes

# Select the first 11 columns for the predictors
X = mcdonalds[mcdonalds.columns[1:12]]  # Adjust if your columns differ

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Gaussian Mixture Model with 2 clusters
gmm = GaussianMixture(n_components=2, n_init=10, random_state=1234)
gmm.fit(X_scaled)

# Print the results
print("Gaussian Mixture Model (GMM) Results:")
print(f"Converged: {gmm.converged_}")
print(f"Means:\n{gmm.means_}")
print(f"Covariances:\n{gmm.covariances_}")

# Predict cluster labels
labels = gmm.predict(X_scaled)
mcdonalds['GMM_Cluster'] = labels

# Evaluate the clustering
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Reverse and transform the 'Like' column
mcdonalds['Like'] = pd.Categorical(mcdonalds['Like']).codes
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Convert categorical columns to numeric
categorical_columns = mcdonalds.select_dtypes(include=['object']).columns
for col in categorical_columns:
    mcdonalds[col] = pd.Categorical(mcdonalds[col]).codes

# Select the first 11 columns for the predictors
X = mcdonalds[mcdonalds.columns[1:12]]  # Adjust if your columns differ

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Gaussian Mixture Model with 2 clusters
gmm = GaussianMixture(n_components=2, n_init=10, random_state=1234)
gmm.fit(X_scaled)

# Predict cluster labels
labels = gmm.predict(X_scaled)
mcdonalds['GMM_Cluster'] = labels

# Evaluate the clustering
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Optionally refit if needed with different parameters or data
# For demonstration, refitting with the same parameters
gmm_refit = GaussianMixture(n_components=2, n_init=10, random_state=1234)
gmm_refit.fit(X_scaled)

# Print refit results
print("Refit Gaussian Mixture Model (GMM) Results:")
print(f"Converged: {gmm_refit.converged_}")
print(f"Means:\n{gmm_refit.means_}")
print(f"Covariances:\n{gmm_refit.covariances_}")

# Optionally, calculate new silhouette score if needed
labels_refit = gmm_refit.predict(X_scaled)
silhouette_avg_refit = silhouette_score(X_scaled, labels_refit)
print(f"Silhouette Score (Refit): {silhouette_avg_refit:.3f}")
# Reverse and transform the 'Like' column
mcdonalds['Like'] = pd.Categorical(mcdonalds['Like']).codes
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Clean column names
mcdonalds.columns = mcdonalds.columns.str.replace(' ', '_').str.replace('.', '_')

# Convert categorical columns to dummy variables
mcdonalds_dummies = pd.get_dummies(mcdonalds, drop_first=True)

# Reverse the 'Like' column
mcdonalds_dummies['Like'] = pd.Categorical(mcdonalds_dummies['Like']).codes
mcdonalds_dummies['Like_n'] = 6 - mcdonalds_dummies['Like']

# Define the formula for regression
predictors = ' + '.join(mcdonalds_dummies.columns.difference(['Like', 'Like_n']))
formula = f'Like_n ~ {predictors}'

# Fit the OLS model
model = smf.ols(formula, data=mcdonalds_dummies).fit()
# Print the summary
print(model.summary())
# Extract the coefficients, standard errors, z-values, and p-values
params = model.params
std_errs = model.bse
z_values = model.tvalues
p_values = model.pvalues

# Create a DataFrame to display results
results_df = pd.DataFrame({
    'Estimate': params,
    'Std. Error': std_errs,
    'z value': z_values,
    'Pr(>|z|)': p_values
})

# Print the formatted results
print(results_df)
mcdonalds_dummies = pd.get_dummies(mcdonalds, drop_first=True)
mcdonalds_dummies['Like'] = pd.Categorical(mcdonalds_dummies['Like']).codes
mcdonalds_dummies['Like_n'] = 6 - mcdonalds_dummies['Like']
predictors = ' + '.join(mcdonalds_dummies.columns.difference(['Like', 'Like_n']))
formula = f'Like_n ~ {predictors}'

# Fit the model
model = smf.ols(formula, data=mcdonalds_dummies).fit()
# Extract the coefficients, standard errors, z-values, and p-values
params = model.params
std_errs = model.bse
z_values = model.tvalues
p_values = model.pvalues

# Create a DataFrame for plotting
results_df = pd.DataFrame({
    'Estimate': params,
    'Std. Error': std_errs,
    'z value': z_values,
    'p-value': p_values
})

# Determine significance
results_df['Significance'] = results_df['p-value'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
)

# Reset index to make the variables a column for plotting
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Variable'}, inplace=True)

# Plot
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(x='Variable', y='Estimate', data=results_df)

# Annotate significance
for index, row in results_df.iterrows():
    bar_plot.text(index, row['Estimate'], row['Significance'], color='black', ha='center')

plt.xticks(rotation=90)
plt.xlabel('Variables')
plt.ylabel('Estimate')
plt.title('Regression Coefficients with Significance')
plt.tight_layout()
plt.show()

# Assuming the relevant data is in columns 1 to 11 for clustering
data = mcdonalds.iloc[:, 1:12]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=1234)  # Example: 4 clusters
clusters = kmeans.fit_predict(data_scaled)

# Ensure the length of clusters matches the data
if len(clusters) != len(data):
    print(f"Length mismatch: clusters ({len(clusters)}) vs data ({len(data)})")
else:
    print("Length check passed.")

# Add cluster labels to DataFrame
data['Cluster'] = clusters

# Convert the Cluster column to categorical type (important for plotting)
data['Cluster'] = data['Cluster'].astype('category')

# Debugging: Check that the 'Cluster' column is added correctly
print("Data columns before melting:", data.columns)
print(data.head())

# Melt the DataFrame for seaborn compatibility, keeping the 'Cluster' column
data_melted = pd.melt(data, id_vars=['Cluster'], var_name='Variable', value_name='Value')

# Debugging: Verify 'Cluster' column exists and is correctly formatted
print("Melted Data columns:", data_melted.columns)
print(data_melted.head())

# Plot bar chart with shading based on clusters
plt.figure(figsize=(12, 8))
sns.barplot(x='Variable', y='Value', data=data_melted, hue='Cluster', palette='viridis', dodge=True)
plt.title('Bar Chart Ordered by Clustering and Shaded by Cluster')
plt.xlabel('Variables')
plt.ylabel('Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Perform PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(data_scaled)

# Perform clustering (using 4 clusters as an example)
kmeans = KMeans(n_clusters=4, random_state=1234)
clusters = kmeans.fit_predict(data_scaled)
# Create a DataFrame with PCA components and clusters
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters

# Plotting the PCA projection with clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df, legend='full')

# Add PCA component vectors (axes)
for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, comp1, comp2, color='r', alpha=0.5, head_width=0.05, head_length=0.05)
    plt.text(comp1 * 1.15, comp2 * 1.15, f"PC{i + 1}", color='g', ha='center', va='center')

plt.title('Cluster Projection on PCA Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True)
plt.show()

# Perform clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=1234)
clusters = kmeans.fit_predict(data_scaled)

# Add clusters to the DataFrame
mcdonalds['Cluster'] = clusters

# Create a contingency table of clusters vs. the 'Like' variable
contingency_table = pd.crosstab(mcdonalds['Cluster'], mcdonalds['Like'])

# Plotting the mosaic plot
plt.figure(figsize=(10, 6))
mosaic(contingency_table.stack(), gap=0.02, title="", labelizer=lambda key: "")
plt.xlabel('Segment Number')
plt.ylabel('Proportion of "Like" Responses')
plt.show()

# Create a contingency table of clusters vs. Gender
contingency_table = pd.crosstab(mcdonalds['Cluster'], mcdonalds['Gender'])

# Plotting the mosaic plot
plt.figure(figsize=(10, 6))

# Create the mosaic plot
mosaic(contingency_table.stack(), title="", labelizer=lambda key: f"{key[1]}")
plt.xlabel('Cluster')
plt.ylabel('Gender')
plt.title('Mosaic Plot of Cluster vs. Gender')
plt.show()

# Encoding categorical variables: Adjust column names as needed
label_encoders = {}

# Assuming 'Like.n' is already numeric; encode Gender and other categorical features
for col in ['Gender', 'VisitFrequency']:  # Add other categorical variables as needed
    le = LabelEncoder()
    mcdonalds[col] = le.fit_transform(mcdonalds[col])
    label_encoders[col] = le

# Target variable: binary classification if the cluster is 3
mcdonalds['Target'] = (mcdonalds['Cluster'] == 3).astype(int)

# Features for the decision tree
features = ['Like', 'Age', 'VisitFrequency', 'Gender']
X = mcdonalds[features]
y = mcdonalds['Target']

# Fit the decision tree classifier
tree_model = DecisionTreeClassifier(random_state=1234)
tree_model.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(14, 10))
plot_tree(tree_model, feature_names=features, class_names=['Not Cluster 3', 'Cluster 3'], filled=True, rounded=True)
plt.title('Decision Tree for Predicting Cluster 3')
plt.show()
#Calculating the mean
#Visit frequency
mcdonalds['VisitFrequency'] = LabelEncoder().fit_transform(mcdonalds['VisitFrequency'])
mcdonalds['cluster_num'] = kmeans.labels_
visit = mcdonalds.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
#Like
mcdonalds['Like'] = LabelEncoder().fit_transform(mcdonalds['Like'])
Like = mcdonalds.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
#Gender
mcdonalds['Gender'] = LabelEncoder().fit_transform(mcdonalds['Gender'])
Gender = mcdonalds.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
#Target segments

plt.figure(figsize = (10,5))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="green")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 16)
plt.xlabel("Visit", fontsize = 10)
plt.ylabel("Like", fontsize = 10)
plt.show()
