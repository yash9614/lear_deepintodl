# Krish Naik Bootcamp Materials (ML + Deep Learning)

This folder organizes and extracts key materials from Krish Naik's bootcamp + Codebasics Deep Learning Takeaways + Exercise Solutions, sourced from:
- **Deep Learning**: [Nrish_Karik_SataDience/DL](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/DL)
- **Machine Learning**: [Nrish_Karik_SataDience/ML](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/ML)
- **Codebasics Takeaways**: `DL_Takeaways_CH7.pdf` – `CH13.pdf` (repo root)
- **Exercise Solutions**: PyTorch, CNN, Regularization, Hyperparameter Tuning, Transformers notebooks

### Key Documents
| File | Purpose |
|------|---------|
| **MOST_IMPORTANT_CONCEPTS.md** | 🔴 Highlighted must-master concepts (start here) |
| **NOTES.md** | Structured concept summaries, algorithms, learning path |
| **INTERVIEW_QUESTIONS.md** | Interview questions + industrial relevance (by topic) |

---

## Deep Learning Topics
- PyTorch Neural Networks (CH7)
- Optimizers: Momentum, RMSProp, Adam, AdamW (CH8)
- Regularization: Dropout, L1/L2, BatchNorm, Early Stopping (CH9)
- Hyperparameter Tuning: Grid/Random Search, Optuna (CH10)
- CNNs: Kernels, Pooling, Hierarchical features, Transfer Learning (CH11)
- Sequence Models: RNN types, Vanishing Gradients, LSTM, GRU (CH12)
- Transformers: Attention, Multi-head, Encoder-Decoder, BERT fine-tuning (CH13)
- Plus Krish materials: Simple RNN, LSTM/GRU, Attention, Transformers PDFs

### Practical DL Projects (Streamlit apps included)
| Project | App Path | What it does |
|---------|----------|--------------|
| Simple RNN IMDB | `simple_rnn_imdb/app.py` | Sentiment analysis |
| ANN Customer Churn | `ann_churn/app.py` | Churn probability prediction |
| LSTM Next Word | `lstm_next_word/app.py` | Next-word prediction (Hamlet) |

Model files (`.h5`, `.pkl`) and full notebooks are in the source `DL/` folder.

---

## Machine Learning Topics (from Krish)
- Linear / Multiple / Polynomial Regression + Ridge/Lasso/ElasticNet
- Logistic Regression
- KNN (Classifier & Regressor) + KD/Ball Tree
- Naive Bayes (and variants)
- SVM / SVC / SVR + Kernels
- Decision Trees, Random Forest, AdaBoost, Gradient Boosting, XGBoost
- Clustering: K-Means, Hierarchical, DBSCAN
- Anomaly Detection: Isolation Forest, LOF
- PCA
- Performance Metrics, Overfitting/Underfitting, Cross-Validation
- End-to-end project structure (`NETWORKsecurity`) + MLflow/Dagshub

Full notebooks, datasets, and PDFs are in the source `ML/` folder.

---

## How to Use
1. **Start with MOST_IMPORTANT_CONCEPTS.md** — the highlighted essentials.
2. Read **NOTES.md** for deeper structured concepts and learning path.
3. Practice with **INTERVIEW_QUESTIONS.md** (questions + industrial relevance).
4. Study the takeaway PDFs in the repo root and the exercise notebooks for hands-on coding.
5. Run the included Streamlit apps after placing model files:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
6. Combine with existing exercises in this repo for a complete portfolio.
