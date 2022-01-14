"""
Example of defining a custom (image) transform using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/transform_with_inds.html.

"""
from dataclasses import replace
import time
from typing import Callable, Optional, Tuple
import numpy as np
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.pipeline.state import State
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter


class CorruptFixedLabels(Operation):
    def generate_code(self) -> Callable:
        # dst will be None since we don't ask for an allocation
        parallel_range = Compiler.get_iterator()
        def corrupt_fixed(labs, _, inds):
            for i in parallel_range(labs.shape[0]):
                # Because the random seed is tied to the image index, the
                # same images will be corrupted every epoch:
                np.random.seed(inds[i])
                if np.random.rand() < 0.2:
                    # They will also be corrupted to a deterministic label:
                    labs[i] = np.random.randint(low=0, high=10)
            return labs

        corrupt_fixed.is_parallel = True
        corrupt_fixed.with_indices = True
        return corrupt_fixed

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # No updates to state or extra memory necessary!
        return previous_state, None

# Step 1: Create an FFCV-compatible CIFAR-10 dataset
ds = torchvision.datasets.CIFAR10('/tmp', train=True, download=True)
writer = DatasetWriter('/tmp/cifar.beton', {
    'image': RGBImageField(),
    'label': IntField()
})
writer.from_indexed_dataset(ds)

# Step 2: Create data loaders
BATCH_SIZE = 512
label_pipelines = {
    'with': [IntDecoder(), CorruptFixedLabels(), ToTensor()],
    'without': [IntDecoder(), ToTensor()]
}

for name, pipeline in label_pipelines.items():
    # Use SEQUENTIAL ordering to compare labels.
    loader = Loader(f'/tmp/cifar.beton', batch_size=BATCH_SIZE,
                    num_workers=8, order=OrderOption.SEQUENTIAL,
                    drop_last=True, pipelines={'label': pipeline})

    # First epoch includes compilation time
    for ims, labs in loader: pass
    start_time = time.time()
    for ep in range(20):
        for i, (ims, labs) in enumerate(loader):
            if i == 0: # Inspect first batch
                print(f'> Labels (epoch {ep:2}): {labs[:40,0].tolist()}')
    print(f'Method: {name} | Shape: {ims.shape} | Time per epoch: {(time.time() - start_time) / 100:.5f}s')