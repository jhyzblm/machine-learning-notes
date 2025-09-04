# auto grad

这是我学习autograd的第一份笔记。

## 公式
$$
y = Xw + b
$$

## PyTorch 示例
```python
import torch
X = torch.rand(100, 3)
w = torch.tensor([1.0, 2.0, 3.0])
b = 0.5
y = X @ w + b



add first note on autograd
