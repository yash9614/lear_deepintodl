# Krish Naik Bootcamp Notes — ML + Deep Learning

Extracted and organized from materials in [Nrish_Karik_SataDience/ML](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/ML) and the DL folder. Full PDFs and notebooks remain in the source repo.

---

## 1. Machine Learning Fundamentals

### Types of Learning
- **Supervised**: Regression (continuous target) and Classification (discrete target).
- **Unsupervised**: Clustering, Dimensionality Reduction, Anomaly Detection.
- **Instance-based vs Model-based**: KNN (instance-based, lazy) vs Linear models / Trees (model-based, eager).

### Linear Models
- **Simple Linear Regression**: `y = β0 + β1x`. Closed-form OLS solution. High correlation → strong linear relationship (height-weight example).
- **Multiple Linear Regression**: Multiple features. Use for economics-style datasets.
- **Polynomial Regression**: Capture non-linear relationships by adding powers of features.
- **Regularization**:
  - **Ridge (L2)**: Shrinks coefficients, keeps all features.
  - **Lasso (L1)**: Can zero out coefficients (feature selection).
  - **ElasticNet**: Mix of L1 + L2.
- **Logistic Regression**: Sigmoid + log-loss for binary classification. Outputs probabilities. Decision boundary is a hyperplane.

### Distance-based & Probabilistic
- **KNN Classifier / Regressor**: Predict based on k nearest neighbors. Critical to scale features. Use GridSearchCV for best k. Supports KD-Tree / Ball-Tree for efficiency.
- **Naive Bayes**: Assumes feature independence. Variants (Gaussian, Multinomial, Bernoulli). Strong baseline for text/categorical data.

### Support Vector Machines
- **SVC**: Maximum margin classifier. Soft margin for non-separable data.
- **SVR**: Epsilon-insensitive loss for regression.
- **Kernels**: Linear, Polynomial, RBF (most common), Sigmoid. Kernel trick maps to higher dimension without explicit computation.

### Tree-based Models
- **Decision Tree Classifier / Regressor**: Recursive splits (Gini / Entropy / MSE). Can overfit easily — control max_depth, min_samples_split.
- **Numerical feature splits**: Sort values and evaluate thresholds.
- **Random Forest**: Bagging of trees + random feature subsets. Out-of-bag (OOB) evaluation as free validation.
- **AdaBoost**: Sequentially reweights misclassified samples. Focuses on hard examples.
- **Gradient Boosting**: Sequentially fits residuals (gradient of loss). Strong but can overfit.
- **XGBoost**: Optimized gradient boosting with regularization, handling missing values, and parallelization. Excellent default for tabular data.

### Unsupervised Learning
- **K-Means**: Partition into k clusters by minimizing within-cluster variance. Sensitive to initialization and scale.
- **Hierarchical Clustering**: Agglomerative (bottom-up). Dendrogram shows merge hierarchy. No need to specify k upfront.
- **DBSCAN**: Density-based. Finds arbitrary shapes, detects noise. Parameters: eps, min_samples.
- **Anomaly Detection**:
  - **Isolation Forest**: Isolates anomalies with random splits (shorter path length = anomaly).
  - **Local Outlier Factor (LOF)**: Compares local density of a point to its neighbors.
- **PCA**: Linear dimensionality reduction. Projects onto principal components (directions of maximum variance). Useful for visualization and noise reduction.

### Evaluation & Model Diagnostics
- **Regression metrics**: MSE, RMSE, MAE. RMSE penalizes large errors more.
- **Classification metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC.
- **Overfitting vs Underfitting**: High train / low test = overfit. Both low = underfit. Solutions: more data, regularization, simpler model, early stopping.
- **Cross-Validation**: K-Fold, Stratified K-Fold, Time Series Split. Reduces variance of performance estimate.

---

## 2. Deep Learning (from DL folder)

### RNNs & Sequential Models
- Simple RNN for sequence data (IMDB sentiment classification).
- Vanishing / exploding gradients limit long-term memory.
- **LSTM**: Forget, input, and output gates control information flow.
- **GRU**: Simplified version of LSTM (fewer parameters).
- **Bidirectional RNN**: Processes sequence in both directions for richer context.
- **Encoder-Decoder + Attention**: Seq2Seq foundation. Attention allows the decoder to focus on relevant encoder states.

### Attention & Transformers
- Self-attention (Query, Key, Value).
- Multi-head attention + positional encoding.
- Basis of modern LLMs (GPT, BERT, etc.).

### Practical ANN Example
- Customer Churn prediction with full preprocessing (LabelEncoder, OneHotEncoder, StandardScaler).
- Hyperparameter tuning and Streamlit deployment.

---

## 3. Production & End-to-End Patterns (from ML materials)

- Full pipeline structure seen in **NETWORKsecurity** project:
  - Data Ingestion → Validation (drift reports) → Transformation → Model Training → Artifacts storage.
- **MLflow + Dagshub**: Experiment tracking, model registry, and reproducibility. Reference in `mlflow_dagshub.txt`.
- Streamlit apps for serving (already copied for RNN, ANN Churn, LSTM).

---

## 4. Recommended Learning Path

1. **Tabular ML first**: Linear models → Trees → Ensembles (RF, XGBoost) → Metrics & CV.
2. **Unsupervised**: Clustering + Anomaly Detection for feature engineering / outlier handling.
3. **Deep Learning**: ANNs → RNNs/LSTM → Attention/Transformers.
4. **Production**: MLflow tracking, Streamlit deployment, modular pipeline structure.
5. Combine with existing exercises in this repo (PyTorch, CNN, Transformers) for a complete portfolio.

---

**Source**: All PDFs, notebooks, and datasets are in `Nrish_Karik_SataDience/ML` and `/DL`. Download locally for hands-on practice. Last expanded: 2026-11-23.