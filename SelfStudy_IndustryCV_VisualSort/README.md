# Self-study: Visual Sort with Transfer Learning

Industry-style computer vision on a small labeled set.
Start from ImageNet weights. Freeze the backbone. Train a new head.
Report more than accuracy. Save the checkpoint a teammate would actually load.

**Repo folder:** `SelfStudy_IndustryCV_VisualSort/`  
**Main notebook:** `visual_sort.ipynb` (run top to bottom on Colab GPU)  
**Working session:** `visual_sort_colab_session.ipynb` (the Colab run from 29 Sep 2026)  
**Protocol notes:** `TECHNIQUES.md`  
**Concept answers:** `notes.md`  
**Original brief:** `ASSIGNMENT.md`

Story kept on purpose: a warehouse camera sorts packages into
**vehicle-like / animal-like / other-object**. The stand-in data is a
3-class slice of CIFAR-10 (`automobile`, `cat`, `ship`).

## What this folder is for

The Codebasics flowers notebook already showed “freeze ResNet, swap `fc`.”
This assignment adds the parts that notebook skipped:

- filter and remap labels so CrossEntropy sees `0..2`
- ImageNet resize + mean/std, flip on train only
- val carved from official train; official test held out
- checkpoint on **val loss**, not train accuracy
- per-class P/R/F1 and a confusion matrix
- one-image inference with probabilities
- optional last-block fine-tune at a smaller LR

## Result from the session run

Frozen ResNet18 head, 5 epochs, Adam `1e-3`, batch 32, 224px input.

| split | accuracy |
| --- | ---: |
| val (epoch 5) | 0.960 |
| test | **0.955** |

| class | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| vehicle | 0.973 | 0.933 | 0.953 |
| animal | 0.956 | 0.972 | 0.964 |
| other-object | 0.938 | 0.961 | 0.950 |

```
confusion matrix  rows = true, cols = pred
                 vehicle  animal  other
vehicle             933      26     41
animal                6     972     22
other-object         20      19    961
```

Weakest class: **vehicle** (recall 0.933). Biggest leak: 41 cars called ships.

Val loss fell the whole way (`0.185 → 0.120`). The last `best.pt` is the one to keep.

## How to rerun

1. Open `visual_sort.ipynb` in Colab.
2. Runtime → GPU (T4 is enough).
3. Run all. CIFAR-10 is ~170 MB. If `download=True` crawls on `cs.toronto.edu`, use the mirror cell in the notebook.
4. You should see a batch shape `[B, 3, 224, 224]` before any training starts.

CPU works. Frozen ResNet18 at 224px on 12k images is tens of minutes per epoch on CPU. Do not drop back to 32×32 to “make it faster.” That breaks the pretrained contract.

## Layout vs the rest of this repo

| Already in the repo | This folder adds |
| --- | --- |
| Flowers transfer notebook | Explicit freeze vs last-block fine-tune, val checkpoint, test report |
| FashionMNIST CNN | ImageNet norm + real transfer protocol |
| Churn regularization | Confusion matrix on images, product metric choice |

Do not submit `CNN_Exercise/Transfer_Learning_Solutions.ipynb` as this assignment.
Same backbone idea. Different job.

## Self-score from the session

- [x] 3-class subset correct; counts printed
- [x] DataLoader batch is `B×3×224×224`
- [x] Frozen vs trainable param counts printed (`1539` vs `11,178,051`)
- [x] Val used to pick checkpoint
- [x] Test report + confusion matrix
- [x] Notes A–H written as explanations
- [ ] Fine-tune last block (optional; frozen val loss was still falling)
- [x] Protocol for a single-image inference cell is in the notebook

Fine-tune was skipped on purpose. See `notes.md` Question G.
