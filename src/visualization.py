import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import read_dataset, preprocess_data, scale_features, FEATURE_COLS, PROJECT_ROOT
from pathlib import Path

def plot_elbow_method(k_values, inertias, save_path=None):
    """
    Plot elbow curve for optimal k selection.
    """

    plt.figure(figsize=(10, 10))
    plt.plot(k_values, inertias, marker='o')
    plt. xlabel("K (The number of clusters)")
    plt.ylabel("Inertia (Within cluster sum of squares)")
    plt.title("Elbow Method for Optimal K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_clusters(df_clust, save_path=None):
    """
    Visualize clusters using PCA components.
    """

    plt.figure(figsize=(10,10))
    sns.scatterplot(data=df_clust,
        x="pca1",
        y="pca2",
        hue="Type of User")
    
    plt.title("User Behavior Categories/Clusters")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2 ")
    plt.legend()
    plt. grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt. close()

def plot_feature_distributions(df_clust, feature_cols, save_path=None):
    """
    Plot distribution of features across different user types.
    
    Args:
        df_clust (pd.DataFrame): Clustered dataframe
        feature_cols (list): List of feature columns to plot
        save_path (str or Path): Path to save the figure
    """
    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(10, 3*n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feature in enumerate(feature_cols):
        sns.boxplot(
            data=df_clust,
            x="Type of User",
            y=feature,
            palette="Set2",
            ax=axes[idx]
        )
        axes[idx].set_title(f"Distribution of {feature}", fontweight='bold')
        axes[idx].set_xlabel("")
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  # Close figure to free memory


def create_all_visualizations(df_clust, k_values, inertias, feature_cols, output_dir):
    """
    Generate all visualizations for the analysis and save to files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    elbow_path = output_dir / 'elbow_method.png'
    plot_elbow_method(k_values, inertias, save_path=elbow_path)
    
    cluster_path = output_dir / 'cluster_visualization.png'
    plot_clusters(df_clust, save_path=cluster_path)
    
    dist_path = output_dir / 'feature_distributions.png'
    plot_feature_distributions(df_clust, feature_cols, save_path=dist_path)
    
if __name__ == "__main__":
    """
    Test visualization functions with sample data.
    """

    # Load and process data
    df = read_dataset()
    X, df_clust = preprocess_data(df)
    X_scaled, scaler = scale_features(X)
    
    # Perform clustering
    k_values, inertias = find_optimal_k(X_scaled)
    y, km_model = perform_clustering(X_scaled, n_clusters=5)
    df_clust["category"] = y
    df_clust, category_details, category_to_label = assign_user_types(df_clust)
    
    # Apply PCA
    X_pca, pca_model = apply_pca(X_scaled)
    df_clust["pca1"] = X_pca[:, 0]
    df_clust["pca2"] = X_pca[:, 1]
    
    # Create all visualizations
    output_dir = PROJECT_ROOT / 'data'
    create_all_visualizations(df_clust, k_values, inertias, FEATURE_COLS, output_dir)
