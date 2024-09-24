import numpy as np
from typing import Optional


class Tensor:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.array(data)
        self.label = label
        self._prev = set(_children)
        self._op = _op

        # Optional gradient
        self.grad = Optional[Tensor] = None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data, (self, other), "+")
        return result

    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = other
        elif isinstance(other, (int, float)):
            other = Tensor([other] * len(self.data))
        else:
            Tensor(other)

        result = Tensor(self.data * other.data, (self, other), "*")
        return result

    def __rmul__(self, other):
        if isinstance(other, Tensor):
            other = other
        elif isinstance(other, (int, float)):
            other = Tensor([other] * len(self.data))
        else:
            Tensor(other)

        result = Tensor(self.data * other.data, (self, other), "*")
        return result

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def __pow__(self, other: int):
        other = isinstance(other, (int, float))
        result = Tensor(self.data ** other, (self, other), "**")
        return result

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return self * other ** -1

    def exp(self):
        x = self.data
        result = Tensor(np.exp(x), (self, ), "exp")
        return result

    def __repr__(self):
        return f"Tensor({self.data})"
