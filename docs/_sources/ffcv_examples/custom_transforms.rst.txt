Fast custom image transforms
=============================

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
``PickACorner`` data augmentation. This augmentation will operate on image
data, and will, for each image, return either the top-left or bottom-right
quadrant of the image (deciding randomly).

FFCV transforms are implemented by subclassing the
:class:`ffcv.pipeline.operation.Operation` class.
Doing so requires providing implementation for two functions:

.. code-block:: python

    from ffcv.pipeline.operation import Operation

    class PickACorner(Operation):

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
  pipeline (for now, think of state as storing the shape and data type,
  and some additional info useful to the compiler), and

- pre-allocates a *single* block of memory for the output of each transform in
  the pipeline; transforms thus (over-)write to the same block of memory for
  each batch, saving allocation time.

To help FFCV accomplish both of these tasks, every transform should implement a
:meth:`~ffcv.pipeline.operation.Operation.declare_state_and_memory` method which
specifies (a) how the given transform will change the state of the data, and (b)
what memory to allocate such that the transform itself does not need to allocate
any additional memory for the rest of the program.

For our ``PickACorner`` transform, a good ``declare_state_and_memory``
implementation looks like this:

.. code-block:: python

    from ffcv.pipeline.allocation_query import AllocationQuery
    from dataclasses import replace

    # Inside the MaybeBrighten class:
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        h, w, c = previous_state.shape
        new_shape = (h // 2, w // 2, c)

        # Everything in the state stays the same other than the shape
        # States are immutable, so we have to edit them using the
        # dataclasses.replace function
        new_state = replace(previous_state, shape=new_shape)

        # We need to allocate memory for the new images
        # so below, we ask for a memory allocation whose width and height is
        # half the original image, with the same type
        # (shape=(,)) of the same type as the image data
        mem_allocation = AllocationQuery(new_shape, previous_state.dtype)
        return (new_state, mem_allocation)

Since we can't implement the augmentation in-place, we needed to allocate new
memory in which to store the output. We also tell downstream augmentations that
the state images will change at this stage in the pipeline.

Implementing the transform function
-----------------------------------
Now it is time to implement the transform itself: we do this using the
:meth:`~ffcv.operation.Operation.generate_code` function, which is actually a
factory function. That is, :meth:`~ffcv.operation.Operation.generate_code`
should return a *function*: this function takes as arguments (a) the output of
the previous operation in the pipeline (as a batch), and (b) a pointer to the
space in memory corresponding to this transformation's allocation query.

.. note::

    See below for how to *augment* the transformation function with a third
    argument containing the index of datapoint within the dataset!

Let's take a first pass at writing the transformation function for
``PickACorner``, not really worrying about performance for now:

.. code-block:: python

    import numpy as np

    def generate_code(self) -> Callable:
        def pick_a_corner(images, dst):
            which_corner = np.random.randint(low=0, high=4, size=(images.shape[0]))
            for i in range(images.shape[0]):
                if which_corner[i] == 0:
                    dst[i] = images[i,:images.shape[1]//2, :images.shape[2]//2]
                else:
                    dst[i] = images[i,-images.shape[1]//2:,
                    -images.shape[2]//2:]

            return dst
        return pick_a_corner

Note that if we did not care about performance, we would be done! We can put
together a little test script to check that our augmentation runs:

.. code-block:: python

    ds = torchvision.datasets.CIFAR10('/tmp', train=True, download=True)
    writer = DatasetWriter('/tmp/cifar.beton', {'image': RGBImageField(),
                                                'label': IntField()})
    writer.from_indexed_dataset(ds)

    BATCH_SIZE = 512
    image_pipelines = {
        'with': [SimpleRGBImageDecoder(), PickACorner(), ToTensor()],
        'without': [SimpleRGBImageDecoder(), ToTensor()]
    }

    for name, pipeline in image_pipelines.items():
        loader = Loader(f'/tmp/cifar.beton', batch_size=BATCH_SIZE,
                        num_workers=8, order=OrderOption.RANDOM,
                        drop_last=True, pipelines={'image': pipeline})

        # First epoch includes compilation time
        for ims, labs in loader: pass
        start_time = time.time()
        for _ in range(100):
            for ims, labs in loader: pass
        print(f'Method: {name} | Shape: {ims.shape} | Time per epoch: {(time.time() - start_time) / 100:.4f}s')

The output of this script is:

.. code-block::

    Method: with | Shape: torch.Size([512, 16, 16, 3]) | Time per epoch: 0.06596s
    Method: without | Shape: torch.Size([512, 32, 32, 3]) | Time per epoch: 0.02828s

Ok! It looks like the augmentation worked, but it also added 0.04s to the
per-epoch time, making our pipeline around 2.5x
slower. Thankfully, our implementation above is suboptimal in a number of
obvious ways. We'll start with the most obvious: we have a ``for`` loop running
in serial inside our augmentation! However, we can use FFCV to compile this for
loop to *parallel* machine code, as follows:

.. code-block:: python

    import numpy as np
    from ffcv.pipeline.compiler import Compiler

    def generate_code(self) -> Callable:
        parallel_range = Compiler.get_iterable()

        def pick_a_corner(images, dst):
            which_corner = np.random.randint(low=0, high=4, size=(images.shape[0]))
            for i in parallel_range(images.shape[0]):
                if which_corner[i] == 0:
                    dst[i] = images[i,:images.shape[1]//2, :images.shape[2]//2]
                else:
                    dst[i] = images[i,-images.shape[1]//2:,
                    -images.shape[2]//2:]

            return dst

        pick_a_corner.is_parallel = True
        return pick_a_corner

Dissecting the changes above: we replaced ``range`` with a parallelized compiled
counterpart given by :meth:`ffcv.pipeline.compiler.Compiler.get_iterator`: then
we assigned the ``is_parallel`` property of the transformation function to flag
to FFCV that the for loop should be compiled to parallel machine code. With just
these two changes, our new output is:

.. code-block::

    Method: with | Shape: torch.Size([512, 16, 16, 3]) | Time per epoch: 0.03404s
    Method: without | Shape: torch.Size([512, 32, 32, 3]) | Time per epoch: 0.02703s

Great! We've cut the overhead from abound 0.04s to just 0.007s, a 6x
improvement!

Advanced usage: more information about state
--------------------------------------------
In the above example, we only needed to update the shape in the pipeline state.
We now briefly provide some more information about the state object that may be
useful for other custom transforms:

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


See `here <https://github.com/libffcv/ffcv/blob/main/examples/docs_examples/custom_transform.py>`_
for the code corresponding to this post.