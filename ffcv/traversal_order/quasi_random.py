import random
from typing import Sequence, TYPE_CHECKING
from numba import njit
import numpy as np

from torch.utils.data import DistributedSampler

from .base import TraversalOrder

if TYPE_CHECKING:
    from ..loader.loader import Loader


@njit(parallel=False)
def generate_order_inner(seed, page_to_samples_array, page_sizes,
                         result, buffer_size=6):
    num_pages = len(page_sizes)
    random.seed(seed)
    np.random.seed(seed)
    current_pages = [0]
    current_pages.remove(0)  # Force the type
    for page_ix in range(num_pages):
        page_size = page_sizes[page_ix]
        random.shuffle(page_to_samples_array[page_ix, :page_size])
    next_page = 0
    page_order = np.random.permutation(num_pages)
    samples_consumed = np.zeros_like(page_sizes)
    for s_ix in range(result.shape[0]):
        while next_page < num_pages and len(current_pages) < buffer_size:
            page_to_add = page_order[next_page]
            if page_sizes[page_to_add] > 0:
                current_pages.append(page_order[next_page])
            next_page += 1
        selected_page_ix = np.random.randint(0, len(current_pages))
        page = current_pages[selected_page_ix]
        result[s_ix] = page_to_samples_array[page, samples_consumed[page]]
        samples_consumed[page] += 1
        if samples_consumed[page] >= page_sizes[page]:
            current_pages.remove(page)


class QuasiRandom(TraversalOrder):

    def __init__(self, loader: 'Loader'):
        super().__init__(loader)

        # TODO filter only the samples we care about!!
        self.page_to_samples = loader.memory_manager.page_to_samples

        if not self.page_to_samples:
            raise ValueError(
                "Dataset won't benefit from QuasiRandom order, use regular Random")

        if self.distributed:
            raise NotImplementedError(
                "distributed Not implemented yet for QuasiRandom")

        self.prepare_data_structures()


    def prepare_data_structures(self):
        index_set = set(self.indices)
        max_size = max(len(y) for y in self.page_to_samples.values())
        num_pages = max(k for k in self.page_to_samples.keys()) + np.uint64(1)

        self.page_to_samples_array = np.empty((num_pages, max_size),
                                              dtype=np.int64)
        self.page_sizes = np.zeros(num_pages, dtype=np.int64)

        for page, content in self.page_to_samples.items():
            for c in content:
                if c in index_set:
                    self.page_to_samples_array[page][self.page_sizes[page]] = c
                    self.page_sizes[page] += 1



    def sample_order(self, epoch: int) -> Sequence[int]:
        seed = self.seed * 912300 + epoch
        result_order = np.zeros(len(self.indices), dtype=np.int64)
        generate_order_inner(seed, self.page_to_samples_array,
                             self.page_sizes,
                             result_order,
                             2*self.loader.batch_size)

        return result_order