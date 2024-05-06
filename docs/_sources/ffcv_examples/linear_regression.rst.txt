Large-Scale Linear Regression
==============================

In this example, we'll see how to run large-scale regularized linear
regression with FFCV (by "large-scale" here we mean a dataset that *definitely*
doesn't fit in GPU memory, and may barely even fit in RAM).

See `here <https://github.com/libffcv/ffcv/blob/main/examples/docs_examples/linear_regression.py>`_ for the script corresponding to this tutorial.

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
    N, D = 1000000, 10000
    X = np.random.rand(N, D).astype('float32')
    # Ground-truth vector
    W, b = np.random.rand(D).astype('float32'), np.random.rand()
    # Response variables
    Y = X @ W + np.random.randn(N)
    # Save the dataset:
    pkl.dump((X, W, b, Y), open('/tmp/linreg_data.pkl', 'wb'))

Basic code template
-------------------

Our goal is to, given ``X`` and ``Y``, recover the true parameter ``W``. We will
accomplish this via *SGD* on the squared-loss:

.. code-block:: python

    import torch as ch
    from tqdm import tqdm
    import time

    train_loader = None # TODO!

    # Calculate data mean and variance for normalization
    def calculate_stats(loader, N):
        mean, stdev = 0., 0.
        for x_batch, _ in tqdm(loader):
            mean += x_batch.sum(0) / N
            stdev += x_batch.pow(2).sum(0) / N
        return mean, ch.sqrt(stdev - mean.pow(2))

    start_time = time.time()
    mean, stdev = calculate_stats(train_loader, N)
    mean, stdev = mean.cuda(), stdev.cuda()
    w_est, b_est = ch.zeros(D).cuda(), ch.zeros(1).cuda() # Initial guess for W
    num_epochs = 10 # Number of full passes over the data to do

    lr = 5e-2
    for _ in range(num_epochs):
        total_loss, num_examples = 0., 0.
        start_time = time.time()
        for (x_batch, y_batch) in tqdm(train_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            # Normalize the data for stability
            x_batch = (x_batch - mean) / stdev
            residual = x_batch @ w_est + b_est - y_batch
            # Gradients
            w_grad = x_batch.T @ residual / x_batch.shape[0]
            b_grad = ch.mean(residual, dim=0)
            # import ipdb; ipdb.set_trace()
            w_est = w_est - lr * w_grad
            b_est = b_est - lr * b_grad
            total_loss += residual.pow(2).sum()
            num_examples += x_batch.shape[0]
        print('Epoch time:', time.time() - start_time)
        print(f'Average loss: {total_loss / num_examples:.3f} | ',
            f'Norm diff', ch.norm(w_est / stdev - ch.tensor(W).cuda()).cpu().item())

    print(f'Total script running time: {time.time() - start_time:.2f}s')

.. note::

    Note that in general, using vanilla gradient descent to solve regularized
    linear regression is typically a very bad idea, and users are better served
    implementing an algorithm based on *conjugate gradients* or
    *variance-reduced gradient methods*. That said, the exact same principles
    here apply to any algorithm, so we use gradient descent to keep the code as
    clean as possible.

Naive approach: PyTorch TensorDataset
--------------------------------------

The only thing that remains unspecified in our implementation above is the
``train_loader``. The standard way of making a loader here would be to use
PyTorch's built-in ``TensorDataset`` class, as follows:

.. code-block:: python

    from torch.utils.data import TensorDataset, DataLoader

    X, W, b, Y = pkl.load(open('/tmp/linreg_data.pkl', 'rb'))
    dataset = TensorDataset(ch.tensor(X), ch.tensor(Y))
    train_loader = DataLoader(dataset, num_workers=8, shuffle=True)
    # ... rest of code as above

The resulting code is runnable and correct. It will use *40GB* of memory, since the
entire tensor ``X`` will be kept in RAM. Running our script in an environment
with a single A100 GPU and 8 CPU cores takes *16 seconds* per epoch.

Speeding things up with FFCV
-----------------------------

We'll now try to improve on these results by replacing the standard PyTorch
data loading pipeline with FFCV. The first step is to rewrite ``X`` and ``Y`` as
a FFCV dataset (as detailed in the :ref:`Writing a dataset to FFCV format`
guide):

.. code-block:: python

    from ffcv.fields import NDArrayField, FloatField

    class LinearRegressionDataset:
        def __getitem__(self, idx):
            return (X[idx], np.array(Y[idx]).astype('float32'))

        def __len__(self):
            return len(X)

    writer = DatasetWriter('/tmp/linreg_data.beton', {
        'covariate': NDArrayField(shape=(D,), dtype=np.dtype('float32')),
        'label': NDArrayField(shape=(1,), dtype=np.dtype('float32')),
    }, num_workers=16)

    writer.from_indexed_dataset(LinearRegressionDataset())

This allows us to replace the TensorDataset from the previous section with an
FFCV data loader:

.. code-block:: python

    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor, Squeeze, ToDevice

    train_loader = Loader('/tmp/linreg_data.beton', batch_size=2048,
                num_workers=8, order=OrderOption.RANDOM,
                pipelines={
                    'covariate': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'))],
                    'label': [NDArrayDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'))]
                })

**With just this simple substitution, our code goes from 16 seconds per epoch on
an A100 GPU to 6 seconds**.

As expected, GPU utilization also increases dramatically since data loading is
no longer a bottleneck---this allows us to make optimizations elsewhere and make
the code even faster!

More speed, less memory
-----------------------
We conclude this guide by suggesting a few ways to make our linear regression
program even faster, and to reduce its memory footprint:

- In our example above, FFCV *caches* the entire dataset in-memory: which means
  that, in the event of insufficient RAM, the program will not error our (unlike
  the TensorDataset example, which will raise a Segmentation Fault), but it will
  become significantly slower. An alternative discussed in the :ref:`Tuning Guide`
  that we didn't explore here is to initialize the loader with ``os_cache=False``
  and ``order=OrderOption.QUASI_RANDOM``---this will disable caching of the full
  dataset (and thus can operate with very little memory!), and will read examples
  in an order which is nearly random but still minimizes underlying disk reads.

- We can also optimize the main loop itself: for example, the gradient updates
  should be performed as in-place operations, as should the normalization. Since
  data loading is no longer the main bottleneck, such optimizations will result in
  improved performance.
