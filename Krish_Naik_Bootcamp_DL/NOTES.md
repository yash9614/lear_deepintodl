# Krish Naik Bootcamp Notes — ML + Deep Learning

Extracted and organized from materials in [Nrish_Karik_SataDience/ML](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/ML) and the DL folder, plus Codebasics Deep Learning Takeaways (CH7–CH13) already present in this repo.

**See also**: [INTERVIEW_QUESTIONS.md](./INTERVIEW_QUESTIONS.md) for interview-focused questions with industrial relevance.

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

## 2. Deep Learning Core (Codebasics Takeaways CH7–CH13 + Krish materials)

### Neural Networks in PyTorch (CH7)
- Subclass `nn.Module` and implement `forward`.
- Use `DataLoader` for batching + shuffling (reduces overfitting).
- **Binary Cross Entropy + Sigmoid** preferred over MSE for binary classification (convex surface, penalizes confident errors).
- Categorical Cross Entropy for multi-class.

### Optimizers (CH8)
- **SGD + Momentum**: Accelerates in consistent directions, dampens oscillations.
- **RMSProp**: Adaptive learning rate using moving average of squared gradients.
- **Adam**: Combines momentum + RMSProp. Default choice for most deep learning tasks.

### Regularization (CH9)
- **Dropout**: Randomly drops neurons during training (typical p=0.2–0.5).
- **L1/L2**: Penalty on large weights (L1 can zero features).
- **Batch Normalization**: Normalizes layer inputs → higher learning rates, less sensitivity to initialization, mild regularization.
- **Early Stopping**: Stop when validation performance plateaus (patience parameter).

### Hyperparameter Tuning (CH10)
- GridSearchCV (exhaustive, expensive) vs RandomSearchCV (faster sampling).
- **Optuna**: Bayesian optimization + pruning of unpromising trials. Preferred for large search spaces in deep learning.

### CNNs (CH11)
- Kernels extract spatial features (edges, textures).
- Pooling reduces spatial size and computation.
- Hierarchical feature learning (low-level → high-level).
- CNNs lack built-in scale/rotation invariance → need data augmentation.

### Sequence Models (CH12)
- Types: One-to-Many, Many-to-One, Many-to-Many.
- Vanishing gradients → LSTM (3 gates: Forget, Input, Output) and GRU (2 gates: Update, Reset).
- LSTM/GRU still relevant for time-series even though Transformers dominate NLP.

### Transformers (CH13)
- Encoder produces contextual embeddings; Decoder generates output sequence.
- Self-attention (Q, K, V) + Multi-head attention.
- Masked self-attention in decoder prevents looking ahead.
- Self-supervised training: CLM (GPT-style) and MLM (BERT-style).
- Positional encodings inject order information.

### Practical ANN Example (Krish)
- Customer Churn with full preprocessing pipeline + Streamlit deployment.

---

## 3. Production & End-to-End Patterns

- Modular pipeline (NETWORKsecurity example): Ingestion → Validation → Transformation → Training → Artifacts.
- **MLflow + Dagshub** for experiment tracking and model registry.
- Streamlit apps for quick serving of trained models.

---

## 4. Recommended Learning Path

1. **Tabular ML first**: Linear models → Trees → Ensembles (RF, XGBoost) → Metrics & CV.
2. **Unsupervised**: Clustering + Anomaly Detection.
3. **Deep Learning fundamentals**: PyTorch basics → Optimizers → Regularization → Hyperparameter tuning.
4. **Architectures**: CNNs → Sequence models (RNN/LSTM/GRU) → Transformers.
5. **Production**: MLflow, Streamlit, modular pipelines.
6. Practice with the interview questions in `INTERVIEW_QUESTIONS.md`.
7. Combine with existing exercises in this repo (PyTorch, CNN, Transformers folders).

---

**Sources**:
- Krish Naik bootcamp: `Nrish_Karik_SataDience/ML` and `/DL`
- Codebasics Deep Learning Takeaways: `DL_Takeaways_CH7.pdf` through `CH13.pdf` in this repo

Last expanded: 2026-11-23