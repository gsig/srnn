import torch
import torch.optim as optim
import srnn
import numpy as np
import matplotlib.pyplot as plt

losses = []
torch.manual_seed(1)
inputs = [torch.randn(1, 10) for _ in range(20)]  # make a sequence

model = srnn.SRNN(10, 10, subset=5)  # Input dim is 3, output dim is 3
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
similarity = lambda x, y: (x * y).sum(2).sum(1)

for epoch in range(3000):
    model.zero_grad()
    model.hidden = model.init_hidden()
    model.output = model.init_output()
    loss, out, picks = model(inputs)
    loss.backward()
    optimizer.step()
    losses.append(loss.detach().numpy())

print(loss)
print(out)
print(picks)
print(np.mean(losses))

plt.plot(losses)
plt.show()
