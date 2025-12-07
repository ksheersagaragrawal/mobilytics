#This file is used to generate visualizations for example clustering and regression visualizations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import read_dataset, preprocess_data, scale_features, FEATURE_COLS, PROJECT_ROOT
from pathlib import Path
import shap
from src.clustering import find_optimal_k, perform_clustering, assign_user_types, apply_pca

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
    
    plt.close()


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

    hist_path = output_dir / 'feature_histograms.png'
    plot_eda_histograms(df_clust, save_path=hist_path)

    box_out_path = output_dir / 'boxplot_usage_outliers.png'
    plot_eda_boxplot_usage_outliers(df_clust, save_path=box_out_path)

    useage_regressions_path = output_dir / 'regression_'
    plot_eda_usage_feature_regressions(df_clust,useage_regressions_path)

    user_categories_path = output_dir / 'eda_user_categories.png'
    plot_eda_user_categories(df_clust,user_categories_path)

    correlation_heatmap_path = output_dir / 'eda_correlation_heatmap.png'
    plot_eda_correlation_heatmap(df_clust,correlation_heatmap_path)



def plot_rf_feature_importance(feat_imp, top_n=15, save_path=None):
    """
    Plot Random Forest feature importance bar plot.
    """

    plt.figure(figsize=(10, 10))
    top_features = feat_imp.head(top_n)
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title("Top Feature Importances for Battery Drain")
    plt.xlabel("Importance")
    plt.ylabel("Category")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_shap_summary(shap_values, X_sample_transformed, feature_names, save_path=None):
    """
    Plot SHAP summary plot for Random Forest.
    """

    shap.summary_plot(
        shap_values, 
        X_sample_transformed, 
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_xgb_feature_importance(feature_importance_df, top_n=10, save_path=None):
    """
    Plot XGBoost feature importance bar plot.
    """

    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 10))
    sns.barplot(
        x='Feature', 
        y='Importance', 
        hue='Feature',
        data=top_features, 
        palette='viridis',
        legend=False
    )
    
    plt.title('Top 10 Feature Importances XGBoost Model for Data Usage', fontsize=16)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt. xticks(rotation=45, ha='right')
    plt. tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def create_all_regression_visualizations(results, output_dir):
    """
    Generate all regression visualizations and save to files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Random Forest Feature Importance
    rf_feat_imp_path = output_dir / 'rf_feature_importance.png'
    plot_rf_feature_importance(
        results['rf']['feature_importance'], 
        save_path=rf_feat_imp_path
    )
    
    # Random Forest SHAP Summary
    rf_shap_path = output_dir / 'rf_shap_summary.png'
    plot_shap_summary(
        results['rf']['shap_values'],
        results['rf']['X_sample_transformed'],
        results['rf']['feature_names'],
        save_path=rf_shap_path
    )
    
    # XGBoost Feature Importance
    xgb_feat_imp_path = output_dir / 'xgb_feature_importance.png'
    plot_xgb_feature_importance(
        results['xgb']['feature_importance'],
        save_path=xgb_feat_imp_path
    )

def plot_eda_histograms(device_usage, save_path):
    """
    Plot histograms of usage features.
    """
    usage_cols = [
    "Screen On Time (hours/day)",
    "App Usage Time (min/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)"
    ]

    # making a one plot for each usage column
    fig, axes = plt.subplots(5, 1, figsize=(5, 20))
    for ax, col in zip(axes, usage_cols):
        # histogram + kde
        sns.histplot(device_usage[col], kde=True, ax=ax, bins=40)
        # adding labels onto plots
        ax.set_title(f"distribution of {col}")
        ax.set_ylabel("count")
        ax.set_xlabel(col)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_eda_boxplot_usage_outliers(device_usage, save_path):
    """
    Plot boxplot of usage features to see outliers.
    """
    usage_cols = [
    "Screen On Time (hours/day)",
    "App Usage Time (min/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)"
    ]

    # boxplot to see outliers in the usage data
    plt.figure(figsize=(12,8))
    sns.boxplot(data=device_usage[usage_cols])

    plt.title("boxplot of usage columns (looking for outliers)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_eda_usage_feature_regressions(device_usage, save_path_base):
    """
    Simple regression plots to see how usage features relate.
    """
    # Screen time vs battery drain
    plt.figure(figsize=(8,6))
    sns.regplot(
        data=device_usage,
        x="Screen On Time (hours/day)",
        y="Battery Drain (mAh/day)",
        scatter_kws={"alpha": 0.3}
    )
    plt.title("Screen Time vs Battery Drain")
    if save_path_base:
        new_path = save_path_base.with_name(save_path_base.name + "screentime_vs_battery.png")
        plt.savefig(new_path, dpi=300, bbox_inches='tight')
    
    plt.close()


    # App usage vs data usage (expecting these to be related)
    plt.figure(figsize=(8,6))
    sns.regplot(
        data=device_usage,
        x="App Usage Time (min/day)",
        y="Data Usage (MB/day)",
        scatter_kws={"alpha": 0.3}
    )
    plt.title("App Usage vs Data Usage")
    if save_path_base:
        new_path = save_path_base.with_name(save_path_base.name + "appuse_vs_datause.png")
        plt.savefig(new_path, dpi=300, bbox_inches='tight')
    
    plt.close()


    # do people with more apps spend more time on their phone?
    plt.figure(figsize=(8,6))
    sns.regplot(
        data=device_usage,
        x="Number of Apps Installed",
        y="Screen On Time (hours/day)",
        scatter_kws={"alpha": 0.3}
    )
    plt.title("Apps Installed vs Screen Time")
    if save_path_base:
        new_path = save_path_base.with_name(save_path_base.name + "appcount_vs_screentime.png")
        plt.savefig(new_path, dpi=300, bbox_inches='tight')
    plt.close()


    # Checking if older users install fewer apps
    plt.figure(figsize=(8,6))
    sns.regplot(
        data=device_usage,
        x="Age",
        y="Number of Apps Installed",
        scatter_kws={"alpha": 0.3}
    )
    plt.title("Age vs Number of Apps Installed")
    if save_path_base:
        new_path = new_path.with_name(save_path_base.name + "age_vs_appcount.png")
        plt.savefig(save_path_base, dpi=300, bbox_inches='tight')
    plt.close()

def plot_eda_user_categories(device_usage, save_path):
    """
    Plot regression with categorization of 75th percentile as a cutoff for "high use".
    """
    # using the 75th percentile as a cutoff for "high use"
    battery_75 = device_usage["Battery Drain (mAh/day)"].quantile(0.75)
    data_75 = device_usage["Data Usage (MB/day)"].quantile(0.75)

    # label each user (like categories) of high use for one or both or neither
    def usage_label(row):
        bat = row["Battery Drain (mAh/day)"]
        dat = row["Data Usage (MB/day)"]

        # top 25% in BOTH battery drain + data usage
        if bat >= battery_75 and dat >= data_75:
            return "high use"
        # below both thresholds == likely lighter users
        elif bat < battery_75 and dat < data_75:
            return "lower use"
        # anything else is probably mixed behavior
        else:
            return "mixed"

    device_usage["usage"] = device_usage.apply(usage_label, axis=1)

    # plotting to visualize if clustering is a good approach
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        data=device_usage,
        x="Battery Drain (mAh/day)",
        y="Data Usage (MB/day)",
        hue="usage",
        alpha=0.6
    )

    plt.title("Categories by Battery Drain vs Data Usage")
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_eda_correlation_heatmap(device_usage, save_path):
    """
    Plot a heatmap of the correlation between numeric columns.

    Parameters:
    device_usage (pd.DataFrame): The dataframe containing the numeric columns.
    save_path (str): The path to save the plot.

    Returns:
    None
    """
    NUMERIC_COLUMNS = [
	"App Usage Time (min/day)",
	"Screen On Time (hours/day)",
	"Battery Drain (mAh/day)",
	"Number of Apps Installed",
	"Data Usage (MB/day)",
	"Age",
	"User Behavior Class",
    ]

    num_df = device_usage[NUMERIC_COLUMNS]
    corr = num_df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    if save_path:
        plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    """
    Test visualization functions with sample data.
    """

    # Load and process data
    df = read_dataset()
    X, df_clean = preprocess_data(df)
    X_scaled, scaler = scale_features(X)

    # Perform clustering
    k_values, inertias = find_optimal_k(X_scaled)
    y, km_model = perform_clustering(X_scaled, n_clusters=5)
    df_clean["category"] = y
    df_clean, category_details, category_to_label = assign_user_types(df_clean)
    
    # Apply PCA
    X_pca, pca_model = apply_pca(X_scaled)
    df_clean["pca1"] = X_pca[:, 0]
    df_clean["pca2"] = X_pca[:, 1]
    
    # Create all visualizations
    output_dir = PROJECT_ROOT / 'data' / 'visualization'
    create_all_visualizations(df_clean, k_values, inertias, FEATURE_COLS, output_dir)
