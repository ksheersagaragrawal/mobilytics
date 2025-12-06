# This file contains code for K Means clustering to identify the type of user based on certain
# features and identify clusters accordingly.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .data_preprocessing import (
    FEATURE_COLS,
    PROJECT_ROOT,
    RANDOM_STATE,
    preprocess_data,
    read_dataset,
    scale_features,
)
from .visualization import plot_clusters, plot_elbow_method

def find_optimal_k(X_scaled, random_state=RANDOM_STATE):
    """
    Use elbow method to find optimal k value.
    """

    inertias = []
    k_values = list(range(2, 10))
    
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    return k_values, inertias

def perform_clustering(X_scaled, n_clusters, random_state=RANDOM_STATE):
    """
    Perform KMeans clustering using the best k value
    """

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    y = km.fit_predict(X_scaled)
    
    return y, km

def assign_user_types(df_clust, feature_cols=FEATURE_COLS, sort_column="Screen On Time (hours/day)"):
    """
    Assign user type labels to clusters based on feature means.
    """

    user_type_labels = [
    "Light User",
    "Moderate User",
    "Heavy User",
    "Very Heavy User",
    "Extreme User"
    ]

    # Calculate mean features per category
    category_details = df_clust.groupby("category")[feature_cols].mean()
    
    # Sort categories by the specified column
    sorted_categories = category_details[sort_column].sort_values().index.tolist()
    
    # Map category IDs to user type labels
    category_to_label = {
        category_id: label
        for category_id, label in zip(sorted_categories, user_type_labels)
    }
    
    # Assign user type labels
    df_clust["Type of User"] = df_clust["category"].map(category_to_label)
    
    return df_clust, category_details, category_to_label

def apply_pca(X_scaled, n_components=2, random_state=RANDOM_STATE):
    """
    Apply PCA for dimensionality reduction.
    """

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca

def main():
    """
    Main function to execute the complete clustering pipeline. 
    """
    # Step 1: Load data
    df = read_dataset()
    
    # Step 2: Preprocess data
    X, df_clust = preprocess_data(df)
    X_scaled, scaler = scale_features(X)
    
    # Step 3: Find optimal k using elbow method
    k_values, inertias = find_optimal_k(X_scaled)
    
    # Save elbow plot
    elbow_plot_path = PROJECT_ROOT / 'data' / 'elbow_method.png'
    plot_elbow_method(k_values, inertias, save_path=elbow_plot_path)
    
    # Step 4: Perform clustering with best k
    BEST_K = 5
    y, km_model = perform_clustering(X_scaled, n_clusters=BEST_K)
    df_clust["category"] = y
    
    # Step 5: Assign user type labels
    df_clust, category_details, category_to_label = assign_user_types(df_clust)
    
    # Step 6: Apply PCA for visualization
    X_pca, pca_model = apply_pca(X_scaled)
    df_clust["pca1"] = X_pca[:, 0]
    df_clust["pca2"] = X_pca[:, 1]
    
    # Save cluster plot
    cluster_plot_path = PROJECT_ROOT / 'data' / 'cluster_visualization.png'
    plot_clusters(df_clust, save_path=cluster_plot_path)
    
    return df_clust, km_model, pca_model, category_details

if __name__ == "__main__":
    """
    Execute the main clustering pipeline when script is run directly.
    """

    df_clust, km_model, pca_model, category_details = main()
