Making custom transforms
========================

In this document, we'll outline how to construct custom FFCV *transforms*.
Transforms are the building blocks that together form pipelines, which specify
how data should be read and preprocessed (this is outlined in the :ref:`Making
an FFCV dataloader` guide, which we strongly recommend reading first).

.. note::

    In general, any ``torch.nn.Module`` can be placed in a pipeline and used as
    a transform without using anything from this guide. This document only
    concerns making custom *FFCV-specific* transforms, which may be faster since
    they can be put on CPU and pre-compiled with `Numba <https://numba.org>`_

In this guide, we will implement a transform that computes a (made-up)
``MaybeBrighten`` data augmentation. This augmentation will operate on image
data, and will:

1. First check if the image's brightness (i.e., the average pixel value) exceeds 127/255 pixels

2. If it does, we leave the image unchanged---otherwise, we increase every pixel by 127/255 pixels


FFCV transforms are implemented by subclassing the
:class:`ffcv.pipeline.operation.Operation` class. 
Doing so requires providing implementation for two functions:

.. code-block:: python

    from ffcv.pipeline.operation import Operation

    class MaybeBrighten(Operation):
        
        # Return the code to run this operation
        @abstractmethod
        def generate_code(self) -> Callable:
            raise NotImplementedError
        
        @abstractmethod
        def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]: 
            raise NotImplementedError

Advancing state and pre-allocating memory
------------------------------------------
As mentioned earlier, transforms are chained together in FFCV to form data
*pipelines*. 
In order to get maximum data processing performance, FFCV:

- keeps track of the *state* of the data being read at each stage in the
  pipeline, and 

- pre-allocates a *single* block of memory for the output of each transform in
  the pipeline; transforms thus (over-)write to the same block of memory for
  each batch, saving allocation time.

To help FFCV accomplish both of these tasks, every transform should implement a 
:meth:`~ffcv.pipeline.operation.Operation.declare_state_and_memory` method which
specifies (a) how the given transform will change the state of the data, and (b)
what memory to allocate such that the transform itself does not need to allocate
any additional memory for the rest of the program.

For our ``MaybeBrighten`` transform, a good ``declare_state_and_memory``
implementation looks like this:

.. code-block:: python

    # At the top of the file:
    from ffcv.pipeline.allocation_query import AllocationQuery

    # Inside the MaybeBrighten class:
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]: 
        new_state = previous_state # We are not changing the data format
        # We need to allocate memory for storing the mean pixel value of each
        # image in the batch---so below, we ask for a *scalar* memory allocation
        # (shape=(,)) of the same type as the image data
        mem_allocation = AllocationQuery((,), previous_state.dtype)
        return (new_state, mem_allocation)


Implementing the transform function
-----------------------------------

Altering state
--------------
At each stage in the pipeline, the data is stored as either a Numpy
array or a PyTorch tensor: transforms that act on NumPy arrays run on CPU and
can be compiled with Numba, while transforms acting on PyTorch tensors can run
on CPU or GPU (but cannot be pre-compiled).

- ``shape``, ``dtype``: these two rather familiar attributes keep track of the
shape and datatype of the data at any given point in the pipeline. The ``shape``
attribute should always be a Python ``tuple``; meanwhile ``dtype`` can be either
a Numpy dtype or a PyTorch dtype depending on how the data is stored.
- ``device``: if the data is in NumPy format, this property is irrelevant;
otherwise, ``device`` should be a ``torch.device`` instance that specifies where
the data is being stored.
- ``jit_mode``: this is a boolean flag for whether the data is in a
*compileable* state (i.e., whether it is on-CPU and in NumPy format).