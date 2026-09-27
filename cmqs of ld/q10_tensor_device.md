# Q10. Using the torch.Tensor constructor with device argument

You place a tensor on GPU at creation time by passing `device`:

```python
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
# or
x = torch.zeros(2, 3, device="cuda")
```

Equivalent later move: `x = x.to("cuda")` or `x.cuda()`.

* `np.array` creates a NumPy array on CPU, not a GPU tensor.
* There is no `torch.gpu_tensor`.
* Shouting does nothing.

Note: prefer the factory functions (`torch.tensor`, `torch.zeros`, `torch.empty`, …) over the raw `torch.Tensor(...)` constructor; they all accept `device=`.
