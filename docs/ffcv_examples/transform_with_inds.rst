Custom transforms with indices
===============================

Another invaluable feature of FFCV transforms is that, by assigning the
``with_indices`` property of the transformation function (so below, by setting
``corrupt_fixed.with_indices=True``), we get access to a *third* transform
argument that contains the index of each image in the batch within the dataset.
This feature makes it possible to implement transforms in FFCV that are not
possible in standard PyTorch: for example, we can implement an augmentation that
corrupts the labels of a *fixed* set of images throughout training.

.. code-block:: python

    class CorruptFixedLabels(Operation):
        def generate_code(self) -> Callable:
            parallel_range = Compiler.get_iterator()
            # dst will be None since we don't ask for an allocation
            def corrupt_fixed(labs, _, inds):
                for i in parallel_range(labs.shape[0]):
                    # Because the random seed is tied to the image index, the
                    # same images will be corrupted every epoch:
                    np.random.seed(inds[i])
                    if np.random.rand() < 0.05:
                        # They will also be corrupted to a deterministic label:
                        labs[i] = np.random.randint(low=0, high=10)
                return labs

            corrupt_fixed.is_parallel = True
            corrupt_fixed.with_indices = True
            return corrupt_fixed

        def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
            # No updates to state or extra memory necessary!
            return previous_state, None

We provide the corresponding script to test the above augmentation `here <https://github.com/libffcv/ffcv/blob/main/examples/docs_examples/transform_with_inds.py>`_.