# Most Important Deep Learning Concepts (Highlighted)

Extracted from Codebasics exercises/solutions + Krish Naik materials + Takeaways.

## 🔴 Critical Concepts (Must Master)

### 1. PyTorch Fundamentals
- **`nn.Module` + `forward()`**: Subclass to define custom architectures.
- **`nn.Sequential`**: Quick stacking of layers (Linear → ReLU → Linear).
- **DataLoader + Dataset**: Batching, shuffling, custom `__getitem__`.
- **Device handling**: `.to(device)` for GPU/CPU.
- **Loss functions**:
  - BCEWithLogitsLoss / CrossEntropyLoss (preferred over MSE for classification).
  - Binary Cross Entropy + Sigmoid gives convex surface.

### 2. Optimizers & Training
- **Adam** (default): Combines momentum + adaptive learning rates. Use `lr=1e-3` or `1e-5` for fine-tuning.
- **AdamW**: Preferred for Transformers (decoupled weight decay).
- **SGD + Momentum**: Still useful; momentum β ≈ 0.9.
- **Gradient Descent variants**:
  - Batch GD (full dataset)
  - Mini-batch (typical 16–128)
  - Stochastic (batch size = 1)
- Always call `optimizer.zero_grad()` before `loss.backward()`.

### 3. Regularization (Prevent Overfitting)
- **Dropout** (`nn.Dropout(p=0.5)`): Drop neurons during training only.
- **BatchNorm** (`nn.BatchNorm1d/2d`): Normalize activations → higher LR possible, less sensitive to init.
- **Weight Decay** (L2): `weight_decay` in optimizer.
- **Early Stopping**: Monitor validation loss; stop after `patience` epochs with no improvement.
- **Data Augmentation** (especially for CNNs): Critical because CNNs lack built-in invariance to scale/rotation.

### 4. CNNs
- **Conv2d**: `nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)`.
- **Pooling**: `nn.MaxPool2d(2)` reduces spatial size.
- **Typical block**: Conv → BatchNorm → ReLU → MaxPool.
- Hierarchical features: early layers detect edges, deeper layers detect objects.
- **Transfer Learning**: Freeze backbone, train classifier head (very common in industry).

### 5. Transformers & BERT Fine-tuning
- **Tokenizer**: `AutoTokenizer.from_pretrained("bert-base-cased")` → `input_ids` + `attention_mask`.
- **Model**: `BertForSequenceClassification.from_pretrained(...)`.
- **Fine-tuning**:
  - Low learning rate (1e-5 to 5e-5).
  - AdamW optimizer.
  - CrossEntropyLoss.
- **Key ideas**: Self-attention (Q, K, V), Multi-head, Masked attention in decoder, Positional encodings.
- CLM (GPT) vs MLM (BERT).

### 6. Hyperparameter Tuning
- **Optuna**: Bayesian optimization + pruning of bad trials.
- Most sensitive: learning rate, batch size, dropout rate, number of layers/units.
- Prefer Random Search or Optuna over pure Grid Search for large spaces.

### 7. Sequence Models
- **LSTM gates**: Forget, Input, Output.
- **GRU**: Update + Reset gates (lighter than LSTM).
- Vanishing gradients → gating mechanisms.
- Still relevant for time-series even though Transformers dominate NLP.

## 🟡 High-Value Practical Tips
- Always set `model.train()` vs `model.eval()`.
- Use `torch.no_grad()` during validation/inference.
- Normalize inputs (e.g., FashionMNIST to [-1, 1]).
- Monitor both train and validation loss curves.
- For production: prefer pre-trained models + fine-tuning over training from scratch.

## 🟢 Interview Quick Hits
- Why Adam over SGD? Adaptive rates + momentum.
- Why Dropout? Prevents co-adaptation of neurons.
- Why BatchNorm? Internal covariate shift + regularization.
- Why Transformers beat RNNs? Parallelization + long-range dependencies via attention.
- BCE vs MSE for classification? Convex surface + better gradients.

Last updated: 2026-11-23