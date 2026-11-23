# Krish Naik Bootcamp Materials (ML + Deep Learning)

This folder organizes and extracts key materials from Krish Naik's bootcamp, sourced from:
- **Deep Learning**: [Nrish_Karik_SataDience/DL](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/DL)
- **Machine Learning**: [Nrish_Karik_SataDience/ML](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/ML)

See **NOTES.md** for extracted concepts, algorithm summaries, evaluation tips, and a recommended learning path.

---

## Deep Learning Topics (PDFs + Notebooks)
- Simple RNN, Bidirectional RNN
- Backward Propagation
- Problems with RNN → LSTM & GRU
- Attention Mechanism
- Encoder-Decoder Seq2Seq
- Transformers

### Practical DL Projects (Streamlit apps included)
| Project | App Path | What it does |
|---------|----------|--------------|
| Simple RNN IMDB | `simple_rnn_imdb/app.py` | Sentiment analysis |
| ANN Customer Churn | `ann_churn/app.py` | Churn probability prediction |
| LSTM Next Word | `lstm_next_word/app.py` | Next-word prediction (Hamlet) |

Model files (`.h5`, `.pkl`) and full notebooks are in the source `DL/` folder.

---

## Machine Learning Topics (PDFs + Notebooks)
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
- End-to-end project structure (`NETWORKsecurity`) + MLflow/Dagshub reference

Full notebooks, datasets (height-weight, economic index, Travel, cardekho, etc.), and PDFs are in the source `ML/` folder.

---

## How to Use
1. Read **NOTES.md** for a structured summary of concepts.
2. Download specific PDFs/notebooks from the source repository for deep study and hands-on coding.
3. Run the included Streamlit apps after placing model files:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Use this alongside the existing exercises in `lear_deepintodl` (PyTorch, CNN, Transformers, etc.) for a complete portfolio.

Last updated: 2026-11-23