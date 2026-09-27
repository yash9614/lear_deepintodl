# Colab solutions

Solved versions of the Codebasics DL exercises. Open any `.ipynb` in **Google Colab** (File → Upload notebook, or raw GitHub URL).

Runtime → GPU is optional. FashionMNIST / CNN notebooks are faster with GPU.

## How to run

1. Open the notebook in Colab.
2. Runtime → Run all.
3. CSV-based notebooks first try GitHub raw. If the repo is private, upload the csv from the matching exercise folder when Colab asks.

| Notebook | Source exercise | Data |
|---|---|---|
| `01_pytorch_basics_solved.ipynb` | `PyTorch_Exercise` | none |
| `02_nn_pytorch_habitat_solved.ipynb` | `NeuralNetworks_PyTorch_Exercise` | `habitat_images_codebasics_DL.csv` |
| `03_nn_training_agritech_solved.ipynb` | `NeuralNetworks_Training_Exercise` | `agriculture_dataset_codebasics_DL.csv` |
| `04_optimizers_fashionmnist_solved.ipynb` | `Model_Optimization_Training_Exercises` | FashionMNIST (auto-download) |
| `05_cnn_fashionmnist_solved.ipynb` | `CNN_Exercise/CNNs_Exercise.ipynb` | FashionMNIST (auto-download) |

Original blank exercises stay in their course folders. These files are the worked copies.

## Notes

- `nn.CrossEntropyLoss` already includes softmax. Models here output **logits**, not a second softmax.
- Small subsets are used on FashionMNIST so Colab finishes in a few minutes.
