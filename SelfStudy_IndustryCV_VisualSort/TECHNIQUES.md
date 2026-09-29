# Techniques used, and why they show up in industry CV

This is the “what we actually applied” note. Concepts and Q&A live in `notes.md`.

## The default industry move

Nobody trains ResNet from random weights on a few thousand labels if ImageNet weights exist.

The job is usually:

1. Small labeled set for *your* classes.
2. ImageNet backbone.
3. Freeze most of the net, train a new head.
4. Maybe unfreeze the last block and fine-tune gently.
5. Report more than accuracy.
6. Save weights. Run one inference the way a teammate would.

That is this notebook. The flowers exercise stopped at step 3 and an accuracy number.

## Data contract with a pretrained CNN

CIFAR is 32×32. ResNet18 was trained at ~224×224 on ImageNet color stats.

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

| split | transform |
| --- | --- |
| train | Resize(224), RandomHorizontalFlip, ToTensor, ImageNet Normalize |
| val / test | Resize(224), ToTensor, ImageNet Normalize |

Flip belongs on train only. Val is an exam. Do not let the exam see the cheat.

If you feed 32×32 tensors straight in, the first conv sees the wrong scale and the wrong color distribution. The backbone stops being a feature extractor and becomes a thing you are fighting.

## Label remap

CIFAR ids `{1, 3, 8}` are not a 3-class problem. `CrossEntropyLoss` wants `0..2`.

```text
automobile 1 → vehicle      0
cat        3 → animal       1
ship       8 → other-object 2
```

Filter by index. Do not copy pixels into a new folder. Wrap the existing `CIFAR10` object.

## Train / val / test split that does not cheat

- Official CIFAR **train**, filtered to 3 classes: 15,000
- Shuffle with a fixed seed, 80 / 20 → 12,000 train + 3,000 val
- Official CIFAR **test**, filtered: 3,000, untouched until the report

Val is for picking `best.pt`. Test is for the number you write down.

## Backbone vs head

```text
image
  → ResNet18 conv blocks     (ImageNet features)
  → global pool
  → Linear(512 → 3)          (your warehouse buckets)
```

Two modes that matter:

| mode | backbone | head | LR |
| --- | --- | --- | --- |
| feature extraction | `requires_grad = False` | train | `1e-3` |
| fine-tune last block | unfreeze `layer4` only | train | `1e-4` |

Frozen trainable count on this run: **1,539** (`512×3 + 3`).
Full network: **11,178,051**.

Optimizer must see only trainable params. Otherwise the fine-tune step will surprise you.

The head outputs **logits**. `CrossEntropyLoss` already does log-softmax. Do not stack `LogSoftmax` on `fc` the way the flowers scratch CNN did.

## Training loop discipline

```text
model.train()
  zero_grad → forward → loss → backward → step

model.eval()
  torch.no_grad()
  accumulate val loss / acc

if val_loss improved:
    save state_dict as best.pt
```

`model.eval()` is not a dropout ritual. ResNet18 is full of BatchNorm. In train mode BN uses the current batch. A batch of 32 upscaled CIFAR chips is a bad estimate of ImageNet running stats. Eval mode uses those running stats. That is the point of the backbone.

## Metrics that survive a product meeting

Accuracy on three balanced classes can hide a model that never predicts “animal.”

This run computed:

- overall accuracy
- per-class precision, recall, F1
- confusion matrix

Product call from the matrix: vehicle recall is the weak spot. 41 cars went to “other-object.” That mix-up is visual (rigid manufactured thing), not a broken head.

Would this deploy on a dock camera? No. These are centered 32×32 chips stretched to 224. Motion blur, tape, and half a pallet are a different dataset.

## Checkpoint rule

Save on **val loss**, not the last epoch, not train accuracy.

This run: val loss still falling at epoch 5, so the last file is honest.
If val acc is high and val loss is rising, ship the earlier checkpoint. The model is getting confident and wrong on a slice.

## Inference a teammate can read

Load `best.pt`. One test image. Print:

- predicted class name
- softmax probabilities for all three
- true class

Logits alone are rude.

## Optional next move (not run in the session)

Starting from frozen `best.pt`:

1. `requires_grad = True` on `layer4.*` and `fc.*` only.
2. New Adam at `1e-4`.
3. 3–5 epochs.
4. Compare test F1 and the 41 car→ship errors, not vibes.

Unfreezing the whole net at `1e-3` on 12k images overwrites early filters that already know edges.

## MobileNet bake-off (optional, not run)

Same protocol, swap the backbone. MobileNetV2’s head is `model.classifier[1]`, not `model.fc`. One table: trainable params, test acc, macro F1, minutes. That is the Suneel-style comparison at assignment scale.

## Practical Colab notes

- Toronto’s CIFAR host can crawl at ~50 KB/s. Prefetch from
  `https://ossci-datasets.s3.amazonaws.com/cifar/cifar-10-python.tar.gz`
  then `CIFAR10(..., download=False)`.
- `num_workers=0` until the pipeline works.
- Do not materialize 15,000 `224×224` float tensors in RAM.
- GPU does not speed up the dataset download. It speeds up the forward pass after the batch exists.

## What carries to the next modality

Same skeleton on the spam / BERT notebook later:

- freeze encoder, train head
- unfreeze last layers at a smaller LR
- report F1, not accuracy
- save the val-best file
- one example inference with probabilities

Different pixels. Same protocol.
