"""
General operations:

- Collation
- Conversion to PyTorch Tensor
- Change device of Tensor
"""
import torch as ch
import numpy as np
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from dataclasses import replace


class ToTensor(Operation):
    """Convert from Numpy array to PyTorch Tensor."""
    def __init__(self):
        super().__init__()

    def generate_code(self) -> Callable:
        def to_tensor(inp, dst):
            return ch.from_numpy(inp)
        return to_tensor

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        new_dtype = ch.from_numpy(np.empty((), dtype=previous_state.dtype)).dtype
        return replace(previous_state, jit_mode=False, dtype=new_dtype), None


class ToDevice(Operation):
    """Move tensor to device.

    Parameters
    ----------
    device: torch.device
        Device to move to.
    non_blocking: bool
        Asynchronous if copying from CPU to GPU.
    """
    def __init__(self, device, non_blocking=True):
        super().__init__()
        self.device = device
        # assert isinstance(device, ch.device), \
        #    f'Make sure device is a ch.device (not a {type(device)})'
        self.non_blocking = non_blocking

    def generate_code(self) -> Callable:
        def to_device(inp, dst):
            if len(inp.shape) == 4:
                if inp.is_contiguous(memory_format=ch.channels_last):
                    dst = dst.reshape(inp.shape[0], inp.shape[2], inp.shape[3], inp.shape[1])
                    dst = dst.permute(0, 3, 1, 2)
            dst = dst[:inp.shape[0]]
            dst.copy_(inp, non_blocking=self.non_blocking)
            return dst

        return to_device

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, device=self.device), AllocationQuery(previous_state.shape, dtype=previous_state.dtype, device=self.device)


class ToTorchImage(Operation):
    """Change tensor to PyTorch format for images (B x C x H x W).

    Parameters
    ----------
    channels_last : bool
        Use torch.channels_last.
    convert_back_int16 : bool
        Convert to float16.
    """
    def __init__(self, channels_last=True, convert_back_int16=True):
        super().__init__()
        self.channels_last = channels_last
        self.convert_int16 = convert_back_int16
        self.enable_int16conv = False

    def generate_code(self) -> Callable:
        do_conv = self.enable_int16conv
        def to_torch_image(inp: ch.Tensor, dst):
            # Returns a permuted view of the same tensor
            if do_conv:
                inp = inp.view(dtype=ch.float16)
                pass
            inp = inp.permute([0, 3, 1, 2])
            # If channels last, it's already contiguous so we're good
            if self.channels_last:
                assert inp.is_contiguous(memory_format=ch.channels_last)
                return inp

            # Otherwise, need to fill the allocated memory with the contiguous tensor
            dst[:inp.shape[0]] = inp.contiguous()
            return dst[:inp.shape[0]]

        return to_torch_image

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        alloc = None
        H, W, C = previous_state.shape
        new_type = previous_state.dtype

        if new_type is ch.int16 and self.convert_int16:
            new_type = ch.float16
            self.enable_int16conv = True

        if not self.channels_last:
            alloc = AllocationQuery((C, H, W), dtype=new_type)
        return replace(previous_state, shape=(C, H, W), dtype=new_type), alloc


class Convert(Operation):
    """Convert to target data type.

    Parameters
    ----------
    target_dtype: numpy.dtype or torch.dtype
        Target data type.
    """
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def generate_code(self) -> Callable:
        def convert(inp, dst):
            return inp.type(self.target_dtype)

        convert.is_parallel = True

        return convert

    # TODO: something weird about device to allocate on
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, dtype=self.target_dtype), None


class View(Operation):
    """View array using np.view or torch.view.

    Parameters
    ----------
    target_dtype: numpy.dtype or torch.dtype
        Target data type.
    """
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def generate_code(self) -> Callable:
        def convert(inp, dst):
            return inp.view(self.target_dtype)

        convert.is_parallel = True

        return convert

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, dtype=self.target_dtype, jit_mode=False), None
