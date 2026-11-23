# Krish Naik Deep Learning Bootcamp Materials

This folder integrates materials from Krish Naik's Deep Learning bootcamp (sourced from the [Nrish_Karik_SataDience](https://github.com/yash9614/Nrish_Karik_SataDience/tree/main/DL) repository).

## Topics Covered (PDFs)
- Simple RNN
- Deep Learning Basics (1-4 Deep Learning)
- Backward Propagation
- Problems with RNN
- Bidirectional RNN
- Complete LSTM and GRU
- Attention Mechanism
- Encoder-Decoder Seq2Seq
- Transformers (best final transformers)

Full PDFs and large model files (.h5, .pkl) are available in the source repo under the `DL/` directory. Download them locally for detailed study.

## Practical Projects (copied here for easy access)
### 1. Simple RNN - IMDB Sentiment Analysis
- Notebooks: `simplernn.ipynb`, `embedding.ipynb`, `prediction.ipynb` (in source)
- Streamlit app: `simple_rnn_imdb/app.py` (see below)
- Model: simple_rnn_imdb.h5 (download from source)

### 2. ANN Classification - Customer Churn Prediction
- Notebooks: experiments.ipynb, hyperparametertuningann.ipynb, salaryregression.ipynb, prediction.ipynb
- Streamlit app: `ann_churn/app.py`
- Supporting files: model.h5, encoders, scaler (from source)

### 3. LSTM Next Word Prediction (Hamlet)
- Notebook: experiemnts.ipynb
- Streamlit app: `lstm_next_word/app.py`
- Model and tokenizer in source

## How to Use
1. Clone or download the full materials from the source repository.
2. Run the Streamlit apps locally after placing the model files in the respective folders:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Study the PDFs for theory and notebooks for hands-on TensorFlow/Keras implementations.

This structure helps organize your deep learning learning path alongside the existing exercises in this repo (CNN, PyTorch, Transformers, etc.).

Last updated: 2026-11-23
