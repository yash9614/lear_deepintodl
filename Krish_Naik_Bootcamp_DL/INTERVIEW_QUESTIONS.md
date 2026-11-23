# Deep Learning & ML Interview Questions (with Industrial Relevance)

Based on Krish Naik bootcamp materials + Codebasics Deep Learning Takeaways (CH7–CH13) + Exercise Solutions.  
Organized by topic. Each section includes **Interview Questions** and **Industrial Relevance**.

**See also**: [MOST_IMPORTANT_CONCEPTS.md](./MOST_IMPORTANT_CONCEPTS.md) for the highlighted must-master list.

---

## 1. Neural Networks & PyTorch Basics (CH7)

**Interview Questions**
1. Why is Binary Cross Entropy preferred over MSE for binary classification?
2. What is the role of `nn.Module` in PyTorch and how does the `forward` method work?
3. Explain the difference between Dataset and DataLoader in PyTorch.
4. How does Categorical Cross Entropy differ from Binary Cross Entropy?
5. Why do we shuffle data every epoch in DataLoader?
6. How do you implement a simple network with `nn.Sequential` (e.g., 128 → 64 → 64 → 1)?

**Industrial Relevance**
- PyTorch is the dominant framework for research and production ML at companies like Meta, Tesla, and OpenAI.
- Proper loss function choice (BCE vs MSE) is critical in production binary classifiers (fraud detection, medical diagnosis, churn prediction).
- Efficient DataLoader usage directly impacts training throughput on GPU clusters.

---

## 2. Optimizers (CH8)

**Interview Questions**
1. Explain the difference between SGD with Momentum, RMSProp, and Adam.
2. Why does Adam often converge faster than plain SGD?
3. What is the role of the momentum coefficient (β)?
4. When would you prefer RMSProp over Adam?
5. How do adaptive optimizers help with sparse gradients?
6. What is AdamW and why is it preferred for fine-tuning Transformers?
7. Difference between Batch GD, Mini-batch GD, and Stochastic GD?

**Industrial Relevance**
- Adam is the default optimizer in most modern deep learning pipelines (NLP, CV, recommendation systems).
- Choice of optimizer affects training stability and wall-clock time — critical for large-scale model training cost.
- Companies tune optimizers carefully for production models to balance convergence speed and final accuracy.

---

## 3. Regularization Techniques (CH9)

**Interview Questions**
1. How does Dropout prevent overfitting? What is a typical dropout rate?
2. Compare L1 and L2 regularization. When is L1 preferred?
3. What is Batch Normalization and how does it help training?
4. Explain Early Stopping and the concept of patience.
5. How do BatchNorm and Dropout complement each other?
6. How would you manually compute the L1 regularization term and its gradient in PyTorch?
7. Why must you call `model.train()` vs `model.eval()` when using Dropout/BatchNorm?

**Industrial Relevance**
- Dropout and BatchNorm are standard in almost every production CNN/Transformer.
- L1 regularization is used for feature selection in high-dimensional tabular models (credit scoring, marketing).
- Early stopping saves significant compute cost in cloud training jobs.

---

## 4. Hyperparameter Tuning (CH10)

**Interview Questions**
1. Difference between GridSearchCV and RandomSearchCV?
2. What is Optuna and how does it improve over random/grid search?
3. Explain Bayesian optimization in the context of hyperparameter tuning.
4. What is pruning in Optuna?
5. Which hyperparameters are most sensitive in deep learning (learning rate, batch size, etc.)?
6. How would you use Optuna to tune a FashionMNIST classifier (learning rate, dropout, number of layers)?

**Industrial Relevance**
- Hyperparameter tuning is a major cost center. Optuna + pruning is widely used to reduce experiment time.
- Learning rate and batch size choices directly impact GPU utilization and final model quality.
- Automated tuning pipelines are part of MLOps platforms at scale (Google, Netflix, Uber).

---

## 5. Convolutional Neural Networks (CH11)

**Interview Questions**
1. What is a convolutional kernel/filter and how does it extract features?
2. Role of pooling layers? Difference between MaxPool and AveragePool?
3. Why do CNNs struggle with scale and rotation by default?
4. Explain hierarchical feature learning in CNNs.
5. How does the receptive field grow as we go deeper?
6. What does a typical CNN block look like (Conv → BatchNorm → ReLU → Pool)?
7. Explain Transfer Learning: when and why do you freeze the backbone?

**Industrial Relevance**
- CNNs power computer vision systems in autonomous driving (Tesla, Waymo), medical imaging (radiology), and retail (Amazon Go).
- Understanding kernels and pooling is essential for debugging model performance on image tasks.
- Data augmentation is critical in industry because pure CNNs lack built-in scale/rotation invariance.
- Transfer learning is the standard approach for most production vision systems (faster, better with less data).

---

## 6. Sequence Models – RNNs, LSTM, GRU (CH12)

**Interview Questions**
1. What are the types of RNN architectures (One-to-Many, Many-to-One, Many-to-Many)?
2. Explain the vanishing gradient problem and how LSTM solves it.
3. Describe the three gates in LSTM and their roles.
4. How does GRU differ from LSTM? When would you choose GRU?
5. Why have Transformers largely replaced LSTMs in modern NLP?

**Industrial Relevance**
- LSTM/GRU still used in time-series forecasting (finance, demand prediction) and speech recognition.
- Understanding vanishing gradients helps diagnose training issues in any deep sequential model.
- Sequence models form the foundation of production systems for recommendation (session-based), fraud detection (transaction sequences), and predictive maintenance.

---

## 7. Transformers & Attention (CH13)

**Interview Questions**
1. Explain self-attention with Query, Key, and Value.
2. What is multi-head attention and why is it useful?
3. Difference between Encoder and Decoder in Transformers?
4. What is masked self-attention and why is it needed in the decoder?
5. Compare Causal Language Modeling (CLM) vs Masked Language Modeling (MLM).
6. How do positional encodings work?
7. Why can Transformers capture long-range dependencies better than RNNs?
8. How do you fine-tune BERT for a classification task in PyTorch (tokenizer, BertForSequenceClassification, AdamW, learning rate)?
9. What is the role of `attention_mask` and `input_ids`?

**Industrial Relevance**
- Transformers (BERT, GPT, T5, Llama) are the backbone of modern NLP and multimodal systems used by Google, OpenAI, Anthropic, and every major tech company.
- Self-attention powers search ranking, chatbots, code generation, and document understanding.
- Understanding encoder vs decoder is key for choosing the right architecture (BERT for classification, GPT for generation).
- Production LLM systems rely on these concepts for fine-tuning, RAG, and agentic applications.
- Fine-tuning pre-trained models (rather than training from scratch) is the industry standard for NLP tasks.

---

## 8. Broader ML Concepts (from Krish Naik ML materials)

**Interview Questions**
1. Bias-Variance tradeoff and how it relates to overfitting.
2. Difference between Bagging and Boosting.
3. How does XGBoost handle missing values and regularization?
4. When would you use Isolation Forest vs DBSCAN for anomaly detection?
5. Explain PCA and when it is useful vs when it can hurt.

**Industrial Relevance**
- XGBoost/LightGBM dominate tabular data competitions and production risk/credit models.
- Anomaly detection is critical in cybersecurity, fraud, and industrial IoT.
- PCA and dimensionality reduction are standard preprocessing steps in high-dimensional enterprise data.

---

## Quick Prep Tips
- Be ready to write PyTorch code for a simple NN, CNN, or BERT fine-tuning loop.
- Explain trade-offs (accuracy vs latency, overfitting vs underfitting).
- Connect concepts to real systems (recommendation, vision, NLP, time-series).
- Practice explaining the “why” behind each technique, not just the “what”.
- Review MOST_IMPORTANT_CONCEPTS.md before any interview.

Last updated: 2026-11-23