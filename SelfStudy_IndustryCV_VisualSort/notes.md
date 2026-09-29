# Concept notes (Questions A–H) and Task 4 write-up

Written after the frozen ResNet18 run on the 3-class CIFAR slice.
Numbers below are from that run, not a wish.

## A. If ImageNet has no “ship” class, why do those weights still help?

ImageNet never needed a ship logit. It needed edges, water-ish texture, hull-like shapes, superstructure-like parts. Those live in the conv blocks. Transfer learning reuses that feature extractor and only trains a new 3-way head that maps “those parts” onto vehicle / animal / other-object. A ship is still made of generic visual parts.

## B. What goes wrong if you unfreeze the whole ResNet and use `lr=1e-3` on ~3k–12k images?

You give 11 million weights a large step size on a small set. Early layers already know edges. A hot LR overwrites them before the new head has settled. Typical result: train accuracy shoots up, val gets worse, or the loss spikes. Fine-tune last block only, and drop the LR about 10×.

## C. What happens if you feed 32×32 tensors straight into `resnet18` with no resize / normalize?

Wrong spatial scale and wrong color distribution. The first conv was built for ~224px ImageNet stats. Activations come out off-distribution. You spend training fighting the backbone instead of using it. That is why the assignment forbids shrinking back to 32×32 “to make Colab faster.”

## D. Warehouse: which error is worse?

Calling a **vehicle an animal** is the worse chute mistake here. A heavy rigid parcel then hits a line that was not built for it. A cat called a ship is messy, but the matrix already says the live leak is car ↔ ship (41 + 20). If this left the notebook, that cell is the one I would watch. There is no universal right answer. This is the product call I am owning.

## E. Val accuracy 92%+ but val loss rising. Last epoch or earlier checkpoint?

Earlier checkpoint. Rising val loss with high acc usually means the model is more confident and more wrong on a slice, or the probabilities are drifting. Acc can bounce on three balanced classes. Val loss is the cleaner signal. This run did not have that problem: val loss fell through epoch 5 (`0.1854 → 0.1204`), so the last `best.pt` is the file to send.

## F. Why is the frozen model so much cheaper to train?

Backprop stops at the new `fc`. Only 1,539 weights get gradients (`512×3 + 3`). The other ~11.2 million are spectators. Optimizer state stays tiny. That is the whole point of feature extraction on a small labeled set.

## G. Did fine-tuning help, hurt, or do nothing?

Not run in this session. Frozen val loss was still falling at epoch 5 and test was already 0.955. Fine-tuning `layer4` at `1e-4` is the next experiment if the 41 car→ship errors need to move. I would only call it a win if val loss keeps falling *and* that off-diagonal shrinks. Accuracy vibes would not count.

## H. Why `model.eval()` on validation even if dropout is off?

BatchNorm. ResNet18 is full of it. Train mode uses batch mean/var. A batch of 32 upscaled CIFAR chips is a noisy estimate. Eval mode uses the ImageNet running stats the backbone was shipped with. Skip `eval()` and val numbers lie, even with dropout off.

---

## Task 4 write-up

Test accuracy **0.955** (val was 0.960). The checkpoint generalized.

Weakest class is **vehicle** (recall 0.933, F1 0.953). Animal is strongest (recall 0.972). Other-object precision is the lowest of the three because vehicles leak into it.

The errors look like pixels, not a broken head. Forty-one automobiles became ships. Six cats became vehicles. Fur vs metal is easy. Boxy rigid thing vs boxy rigid thing is not.

I would not deploy this on a warehouse camera. The test images are still centered CIFAR postage stamps stretched to 224. A real bay has blur, odd angles, tape, and half a pallet. 95% here means the transfer protocol works, not that the dock is solved.

`best.pt` was picked by val loss, which kept improving, so the file on disk matches the last epoch. That is the file a teammate should load.
