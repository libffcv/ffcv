Large-Scale Linear Regression
==============================

In this example, we'll see how to run large-scale regularized linear
regression with FFCV (by "large-scale" here we mean a dataset that *definitely*
doesn't fit in GPU memory, and may barely even fit in RAM).

Setup: Generating a fake dataset
--------------------------------

Let's start by generating a fake dataset on which we'll run linear regression.
We'll generate the independent variables (also known as the covariates or
inputs) as random uniform vectors, and the dependent variable (also known as the
responses or outputs) as the noised product of the dependent variable and a
fixed weight vector:

.. code-block:: python

    import numpy as np
    
    # 1,000,000 inputs each of dimension 10,000 = 40GB of data
    N, D = 1_000_000, 10_000
    X = np.random.rand(N, D)
    # Ground-truth vector
    W = np.random.rand(D)
    # Response variables
    Y = X @ W + np.random.randn(N) 

Our goal is to, given ``X`` and ``Y``, recover the true parameter ``W``. We will
accomplish this via *SGD* on the squared-loss:

.. code-block:: python

    import torch as ch

    train_loader, val_loader = None # TODO!
    mean, stdev = calculate_stats(train_loader) # TODO!
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

.. note::

    Note that in general, using vanilla gradient descent to solve regularized
    linear regression is typically a very bad idea, and users are better served
    implementing an algorithm based on *conjugate gradients* or
    *variance-reduced gradient methods*. That said, the exact same principles
    here apply to any algorithm, so we use gradient descent to keep the code as
    clean as possible.

Naive approach: PyTorch TensorDataset
--------------------------------------
