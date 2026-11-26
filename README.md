# Unsupervised Learning: Weekly Sales Product Segmentation

An end-to-end unsupervised machine learning pipeline built in Python to segment **811 products** based on **52 weeks of sales transaction data**. The project combines feature scaling, dimensionality reduction (**PCA**), and clustering (**K-Means**) inside a production-ready **Scikit-Learn Pipeline** persisted with **Joblib**.

---

## Project Overview

Analyzing raw weekly sales across 52 weeks introduces high dimensionality and collinearity, which degrades distance-based clustering algorithms (*curse of dimensionality*). This project solves that problem through a structured workflow:

1. **Data Cleaning & Feature Selection:** Indexed products by `Product_Code` and filtered out pre-normalized columns (`MIN`, `MAX`, `Normalized 0–51`) to retain raw weekly sales (`W0`–`W51`).
2. **Standardization:** Applied `StandardScaler` (Z-score normalization) so that every week contributes equally to variance calculations.
3. **Dimensionality Reduction:** Compressed the 52 weekly features into **5 orthogonal Principal Components** via `PCA`, retaining the core variance structure while eliminating noise.
4. **Clustering:** Segmented products into **3 distinct behavioral groups** using `KMeans` (`n_clusters=3`).
5. **Pipeline & Model Persistence:** Encapsulated preprocessing, PCA, and K-Means into a single `Pipeline` object serialized with `joblib` (`sales_clustering_pipeline.joblib`) to prevent data leakage and enable direct inference on new data.
6. **Validation & Business Profiling:** Evaluated cumulative explained variance, validated $k$ via the Elbow Method and Silhouette Score, and analyzed average weekly sales trajectories per cluster.

---

## Repository Structure

```text
Unsupervisedlearning_salesproblem/
├── Notebook - Unsupervised Project.ipynb   # Full exploratory analysis, pipeline, and visualizations
├── Sales_Transactions_Dataset_Weekly.csv   # Weekly sales dataset (811 products x 52 weeks)
├── sales_clustering_pipeline.joblib        # Serialized Scikit-Learn Pipeline for inference
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
