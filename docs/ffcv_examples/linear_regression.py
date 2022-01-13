import numpy as np

# 1,000,000 inputs each of dimension 10,000 = 40GB of data
N, D = 1000000, 10000
X = np.random.rand(N, D)
# Ground-truth vector
W = np.random.rand(D)
# Response variables
Y = X @ W + np.random.randn(N) 

import torch as ch

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(ch.tensor(X), ch.tensor(Y))
train_loader = DataLoader(dataset, num_workers=8, shuffle=True)
# ... rest of code as above

# Calculate data mean and variance for normalization
def calculate_stats(loader, N):
    mean, stdev = 0., 0.
    for x_batch, _ in loader:
        mean += x_batch.sum(0) / N
        stdev += x_batch.pow(2).sum(0) / N
    return mean, ch.sqrt(stdev - mean.pow(2))

mean, stdev = calculate_stats(train_loader, N)
w_est = ch.zeros(D) # Initial guess for W
num_epochs = 100 # Number of full passes over the data to do

for _ in range(num_epochs):
    total_loss, num_examples = 0., 0.
    for (x_batch, y_batch) in train_loader:
        # Normalize the data for stability
        x_batch = (x_batch - mean) / stdev
        residual = x_batch @ W - y_batch 
        grad = x_batch.T @ residual
        W = W - lr * grad
        total_loss += residual.pow(2).sum()
        num_examples += x_batch.shape[0]

    print(f'Average loss: {total_loss / num_examples:.3f}')

