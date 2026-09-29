# Assignment brief — Visual Sort with Transfer Learning (PyTorch)

Full original constraints, kept next to the work so this folder stands alone.

**Time box:** one focused session (2–3 hours).  
**Hardware:** Colab GPU. CPU works if you accept slower epochs.  
**Do not do:** `torch.tensor` tutorials, from-scratch autograd, MNIST hello world.  
**Do not submit:** `CNN_Exercise/Transfer_Learning_Solutions.ipynb` with the class names swapped.

## Data

3-class subset of CIFAR-10:

| CIFAR id | CIFAR label | Warehouse bucket |
| ---: | --- | --- |
| 1 | automobile | vehicle |
| 3 | cat | animal |
| 8 | ship | other-object |

Remap to `0, 1, 2` before the loss.

## Hard constraints

- PyTorch + torchvision
- Start from `resnet18` + ImageNet weights. Optional bake-off: `mobilenet_v2`
- ≤ 8 epochs per run
- Batch 64 on GPU, 32 on CPU
- `Dataset` / `DataLoader` only
- Save `best.pt` from **validation loss**, not train accuracy
- Resize to 224 and normalize with ImageNet mean/std. Do not stay at 32×32

## Tasks

0. Inspect. Print 3-class counts. Show 8 images. Say whether raw accuracy is trustworthy.
1. Train / val from official train (80/20). Test = official test, filtered. Check one batch is `B×3×224×224`.
2. `make_model(num_classes=3, freeze_backbone=True)`. Print trainable params both ways.
3. Train the frozen extractor 5–8 epochs. Save on best val loss.
4. Test report: accuracy, per-class P/R/F1, confusion matrix. Short write-up.
5. Optional: unfreeze `layer4` + `fc`, smaller LR, 3–5 epochs. Compare on the curves.
6. One test image: pred, probability, true label.
7. Optional MobileNetV2 bake-off table.

## Definition of done

You can say out loud, without the notebook:

1. What you reused from ImageNet and what you trained.
2. Why val loss picked the file you would send a teammate.
3. Which class the model confuses, and whether fine-tuning changed that.

“Accuracy went up” is not done.

Answers to A–H and the Task 4 paragraph are in `notes.md`.
The industry framing is in `TECHNIQUES.md`.
