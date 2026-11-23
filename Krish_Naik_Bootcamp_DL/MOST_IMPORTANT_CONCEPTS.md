# Most Important Deep Learning Concepts (Highlighted)

Extracted from Codebasics exercises/solutions + Krish Naik materials + Takeaways.  
**Start here** for the highest-value concepts.

---

## 🔴 Critical Concepts (Must Master)

### 1. PyTorch Fundamentals
- **`nn.Module` + `forward()`**: Subclass to define custom architectures.
- **`nn.Sequential`**: Quick stacking of layers (Linear → ReLU → Linear).
- **DataLoader + Dataset**: Batching, shuffling, custom `__getitem__`.
- **Device handling**: `.to(device)` for GPU/CPU.
- **Loss functions**:
  - `BCEWithLogitsLoss` / `CrossEntropyLoss` (preferred over MSE for classification).
  - Binary Cross Entropy + Sigmoid gives a convex surface → easier optimization.

**Practical pattern (from solutions):**
```python
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
```

### 2. Optimizers & Training Loop
- **Adam** (default): Combines momentum + adaptive learning rates. Typical `lr=1e-3`.
- **AdamW**: Preferred for Transformers (decoupled weight decay). Use `lr=1e-5` to `5e-5` for fine-tuning.
- **SGD + Momentum**: Still useful; momentum β ≈ 0.9.
- **Gradient Descent variants**:
  - Batch GD (full dataset)
  - Mini-batch (typical 16–128) ← most common
  - Stochastic (batch size = 1)
- Always call `optimizer.zero_grad()` **before** `loss.backward()`.

**Standard training loop pattern:**
```python
model.train()
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### 3. Regularization (Prevent Overfitting)
- **Dropout** (`nn.Dropout(p=0.3–0.5)`): Drop neurons during training only. Expect noisier training loss curves.
- **BatchNorm** (`nn.BatchNorm1d/2d`): Normalize activations → higher learning rates possible, less sensitive to initialization.
- **Weight Decay** (L2): `weight_decay` parameter in optimizer. Too high → loss can diverge.
- **Early Stopping**: Monitor validation loss; stop after `patience` epochs with no improvement.
- **Data Augmentation** (especially for CNNs): Random flip, crop, etc. Critical because CNNs lack built-in scale/rotation invariance.

**Observation from solutions**: Dropout makes training loss fluctuate but validation loss stays smoother and lower → effective regularization.

### 4. CNNs
- **Conv2d**: `nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)`.
- **Pooling**: `nn.MaxPool2d(2)` reduces spatial size.
- **Typical block**: Conv → ReLU → MaxPool (or Conv → BatchNorm → ReLU → MaxPool).
- Hierarchical features: early layers detect edges/textures, deeper layers detect objects.
- **Transfer Learning**: Freeze backbone (`param.requires_grad = False`), replace final FC layer, train only the new head. Industry standard.

**CNN block pattern (from FashionMNIST solution):**
```python
nn.Conv2d(1, 32, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2),
nn.Conv2d(32, 64, kernel_size=3, padding=1),
nn.ReLU(),
nn.MaxPool2d(2)
```

### 5. Transformers & BERT Fine-tuning
- **Tokenizer**: `AutoTokenizer.from_pretrained("bert-base-cased")` → produces `input_ids` + `attention_mask`.
- **Model**: `BertForSequenceClassification.from_pretrained(...)`.
- **Fine-tuning best practices**:
  - Low learning rate (1e-5 to 5e-5).
  - `AdamW` optimizer.
  - `CrossEntropyLoss`.
  - Padding + truncation during tokenization.
- **Key ideas**: Self-attention (Q, K, V), Multi-head attention, Masked attention in decoder, Positional encodings.
- CLM (GPT-style next token) vs MLM (BERT-style masked tokens).

### 6. Hyperparameter Tuning
- **Optuna**: Bayesian optimization + pruning of unpromising trials. Preferred for large search spaces.
- Most sensitive hyperparameters: learning rate, batch size, dropout rate, number of layers/units.
- Prefer Random Search or Optuna over pure Grid Search (exhaustive but expensive).
- Normalize inputs (e.g., FashionMNIST to [-1, 1]) for faster, more stable convergence.

### 7. Sequence Models
- **LSTM gates**: Forget, Input, Output.
- **GRU**: Update + Reset gates (lighter, often comparable performance).
- Vanishing gradients → gating mechanisms solve short-term memory problem.
- Still relevant for time-series forecasting even though Transformers dominate NLP.

---

## 🟡 High-Value Practical Tips (from exercise solutions)

- Always set `model.train()` before training and `model.eval()` before validation/inference.
- Use `torch.no_grad()` during validation to save memory and speed up.
- Normalize inputs (pixel values to [-1, 1] or ImageNet mean/std for transfer learning).
- Monitor **both** train and validation loss curves. Divergence between them signals overfitting.
- For production: prefer pre-trained models + fine-tuning over training from scratch.
- In Transfer Learning, only pass the new head's parameters to the optimizer (`model.fc.parameters()`).
- Data augmentation (random horizontal flip, crop) is one of the highest-ROI techniques for image models.
- Too aggressive weight decay can cause loss to increase (model diverges).

---

## 🟢 Interview Quick Hits

- **Why Adam over SGD?** Adaptive per-parameter learning rates + momentum → faster and more stable convergence.
- **Why Dropout?** Prevents co-adaptation of neurons; forces the network to learn redundant representations.
- **Why BatchNorm?** Reduces internal covariate shift; allows higher learning rates; mild regularization effect.
- **Why Transformers beat RNNs?** Parallelization across sequence + direct long-range dependencies via attention.
- **BCE vs MSE for classification?** BCE + Sigmoid produces a convex loss surface with better gradients.
- **Why freeze layers in Transfer Learning?** Preserves useful low-level features learned on large datasets (ImageNet); only adapts the task-specific head.
- **What does a healthy loss curve look like?** Both train and validation loss decrease smoothly; validation should not rise while train keeps falling.

---

Last updated: 2026-11-23 (further iteration with practical patterns)