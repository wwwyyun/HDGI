import torch
import numpy as np

seed=123
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

a1 = np.array([[2.1, 3.2], [4.9,9.1]])

a = torch.Tensor(a1)
linear = torch.nn.Linear(2,1)
opt = torch.optim.Adam(linear.parameters())
loss = torch.nn.BCEWithLogitsLoss()
with torch.no_grad():
#    linear.weight = torch.Tensor(np.linalg.norm(linear.weight.numpy()))
    linear.weight = linear.weight.div(torch.norm(linear.weight, dim=1, keepdim=True))
#with torch.no_grad():
#    linear.weight.div_(torch.norm(linear.weight, dim=1, keepdim=True))
b = linear(a)
ls = loss(b, torch.Tensor([1,1]).unsqueeze(1))

linear.train()
opt.zero_grad()
ls.backward()
opt.step()

print(ls.item())
print(linear.weight.grad)
