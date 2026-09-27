# Colab solutions

Worked copies of the Codebasics Deep Learning exercises. Original blank notebooks stay in their course folders.

Open any `.ipynb` in **Google Colab** (File → Upload, or paste the raw GitHub URL). Then **Runtime → Run all**.

Use a **GPU** runtime for 04, 07, 08, 09 and 10.

## Notebooks

| # | File | Source exercise | Data |
|---|---|---|---|
| 01 | `01_pytorch_basics_solved.ipynb` | `PyTorch_Exercise` | none |
| 02 | `02_nn_pytorch_habitat_solved.ipynb` | `NeuralNetworks_PyTorch_Exercise` | `habitat_images_codebasics_DL.csv` |
| 03 | `03_nn_training_agritech_solved.ipynb` | `NeuralNetworks_Training_Exercise` | `agriculture_dataset_codebasics_DL.csv` |
| 04 | `04_optimizers_fashionmnist_solved.ipynb` | Training algorithms ex. 1 | FashionMNIST (auto) |
| 05 | `05_optimizers_math_bonus_solved.ipynb` | Training algorithms ex. 2 | none |
| 06 | `06_regularization_churn_solved.ipynb` | Regularization | `AtliQ_Churn_Prediction_Codebasics_DL.csv` |
| 07 | `07_hyperparameter_tuning_solved.ipynb` | Hyperparameter tuning | FashionMNIST (auto) + Optuna |
| 08 | `08_cnn_fashionmnist_solved.ipynb` | `CNN_Exercise/CNNs_Exercise.ipynb` | FashionMNIST (auto) |
| 09 | `09_transfer_learning_flowers_solved.ipynb` | Transfer learning | Oxford Flowers102 (auto) |
| 10 | `10_transformers_bert_covid_solved.ipynb` | `Transformers_Exercise` | `covid_twitter_dataset_codebasics_DL.csv` |

## CSV fallback

Notebooks try GitHub raw first, then a local path next to the original exercise folder, then Colab file upload.

## PyTorch notes

- `nn.CrossEntropyLoss` already includes log-softmax. Models here output **logits**. The course templates sometimes add an extra Softmax / LogSoftmax — that is skipped on purpose.
- Dropout / BatchNorm templates in the regularization exercise have mismatched `Linear` sizes. All churn models here are `6 → 32 → 16 → 1`.
- Hyperparameter-tuning epoch counts are shortened so Colab finishes. Bump `BASE_EPOCHS` / `SEARCH_EPOCHS` / `FINAL_EPOCHS` if you have time.
