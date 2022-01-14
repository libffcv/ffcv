"""
Example of defining a custom (image) transform using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/custom_transforms.html.

"""
import time
import numpy as np
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter
from dataclasses import replace

class PickACorner(Operation):
    def generate_code(self):
        parallel_range = Compiler.get_iterator()
        def pick_a_corner(images, dst):
            which_corner = np.random.rand(images.shape[0])
            for i in parallel_range(images.shape[0]):
                if which_corner[i] == 0:
                    dst[i] = images[i,:images.shape[1]//2, :images.shape[2]//2]
                else:
                    dst[i] = images[i,-images.shape[1]//2:, -images.shape[2]//2:]

            return dst

        pick_a_corner.is_parallel = True
        return pick_a_corner

    def declare_state_and_memory(self, previous_state):
        h, w, c = previous_state.shape
        new_shape = (h // 2, w // 2, c)

        new_state = replace(previous_state, shape=new_shape)
        mem_allocation = AllocationQuery(new_shape, previous_state.dtype)
        return (new_state, mem_allocation)

# Step 1: Create an FFCV-compatible CIFAR-10 dataset
ds = torchvision.datasets.CIFAR10('/tmp', train=True, download=True)
writer = DatasetWriter('/tmp/cifar.beton', {
    'image': RGBImageField(),
    'label': IntField()
})
writer.from_indexed_dataset(ds)

# Step 2: Create data loaders
BATCH_SIZE = 512
# Create loaders
image_pipelines = {
    'with': [SimpleRGBImageDecoder(), PickACorner(), ToTensor()],
    'without': [SimpleRGBImageDecoder(), ToTensor()]
}

for name, pipeline in image_pipelines.items():
    loader = Loader(f'/tmp/cifar.beton', batch_size=BATCH_SIZE,
                    num_workers=16, order=OrderOption.RANDOM,
                    drop_last=True, pipelines={'image': pipeline})

    # First epoch includes compilation time
    for ims, labs in loader: pass
    start_time = time.time()
    for _ in range(100):
        for ims, labs in loader: pass
    print(f'Method: {name} | Shape: {ims.shape} | Time per epoch: {(time.time() - start_time) / 100:.5f}s')