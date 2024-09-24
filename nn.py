from tensor import Tensor

x1 = Tensor([2, 2, 2])
w1 = Tensor([-3, -3, -3])

x2 = Tensor([0, 0, 0])
w2 = Tensor([1, 1, 1])

b = Tensor([6.881373587019, 6.881373587019, 6.881373587019])

y = x1 * w1 + x2 * w2 + b
print(y)

e = (2*y).exp()
print(e)

o = (e - 1) / (e + 1)
print(e-1)
print(e+1)
print(o)
