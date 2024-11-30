import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

def plot_pca(real_data, synthetic_data):
    # Add source labels
    real_data["source"] = "Real"
    synthetic_data["source"] = "Synthetic"

    # Combine datasets
    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)

    numeric_data = combined_data.select_dtypes(include="number").dropna()

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(numeric_data)

    # Create DataFrame for PCA results
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["source"] = combined_data["source"]

    # Plot PCA results
    plt.figure(figsize=(10, 6))
    for source, color in zip(["Real", "Synthetic"], ["blue", "orange"]):
        subset = pca_df[pca_df["source"] == source]
        plt.scatter(subset["PCA1"], subset["PCA2"], label=source, alpha=0.5, s=10, color=color)

    plt.title("PCA of Real vs Synthetic Data")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend()
    plt.show()


def ks_test(real_data, synthetic_data):
    real_data_numeric = real_data.select_dtypes(include="number").dropna()
    synthetic_data_numeric = synthetic_data.select_dtypes(include="number").dropna()

    ks_results = []
    for i in range(real_data_numeric.shape[1]):
        stat, p_value = ks_2samp(real_data_numeric.iloc[:, i], synthetic_data_numeric.iloc[:, i])
        ks_results.append({"Feature": real_data_numeric.columns[i], "KS Statistic": stat, "p-value": p_value})

    ks_df = pd.DataFrame(ks_results)
    print(ks_df)
    return ks_df


def kde_plot(real_data, synthetic_data, num_features_to_plot=5):
    real_data_numeric = real_data.select_dtypes(include="number").dropna()
    synthetic_data_numeric = synthetic_data.select_dtypes(include="number").dropna()

    num_features_to_plot = min(num_features_to_plot, real_data_numeric.shape[1])

    for idx in range(num_features_to_plot):
        plt.figure(figsize=(8, 5))
        sns.kdeplot(real_data_numeric.iloc[:, idx], label="Original Minority", fill=True)
        sns.kdeplot(synthetic_data_numeric.iloc[:, idx], label="Synthetic Minority", fill=True)
        plt.title(f"Distribution of Feature {real_data_numeric.columns[idx]}")
        plt.legend()
        plt.show()
