import torch
import torch.nn as nn 
import matplotlib.pyplot as plt

#linear regression
model = nn.Linear(10, 1)
x = torch.ones(10)
x1 = torch.zeros(10)
'''
print(model)
print(model.weight)
print(model.bias)
print(model(x))
print(model(x1))
'''
model = torch.nn.Linear(10, 1)
x2 = torch.randn(20, 10)
y2 = torch.randn(20, 1)
#print (f'{x2=}{y2=}')

pred_y2 = model(x2)
#print(pred_y2)

#y2, pred_y
# type of loss code
loss = torch.nn.functional.mse_loss(pred_y2, y2)
print(loss)
lossbina = torch.nn.functional.binary_cross_entropy_with_logits(pred_y2, y2)
print(lossbina)

def f(x: torch.Tensor)->torch.Tensor:
    return 3*x**2+torch.sin(5*x)
plt.plot(torch.arange(-1, 1, 0.01), f(torch.arange(-1, 1, 0.01)))
