"""
FFCV loader
"""
import enum
from os import environ
import ast
from multiprocessing import cpu_count
from re import sub
from typing import Any, Callable, Mapping, Sequence, Type, Union, Literal
from collections import defaultdict
from enum import Enum, unique, auto
from ffcv.fields.base import Field

import torch as ch
import numpy as np

from .epoch_iterator import EpochIterator
from ..reader import Reader
from ..traversal_order.base import TraversalOrder
from ..traversal_order import Random, Sequential, QuasiRandom
from ..pipeline import Pipeline
from ..pipeline.compiler import Compiler
from ..pipeline.operation import Operation
from ..transforms.ops import ToTensor
from ..transforms.module import ModuleWrapper
from ..memory_managers import (
    ProcessCacheManager, OSCacheManager, MemoryManager
)

@unique
class OrderOption(Enum):
    SEQUENTIAL = auto()
    RANDOM = auto()
    QUASI_RANDOM = auto()

ORDER_TYPE = Union[
    TraversalOrder,
    Literal[OrderOption.SEQUENTIAL,
            OrderOption.RANDOM]

]

ORDER_MAP: Mapping[ORDER_TYPE, TraversalOrder] = {
    OrderOption.RANDOM: Random,
    OrderOption.SEQUENTIAL: Sequential,
    OrderOption.QUASI_RANDOM: QuasiRandom
}

DEFAULT_PROCESS_CACHE = int(environ.get('FFCV_DEFAULT_CACHE_PROCESS', "0"))
DEFAULT_OS_CACHE = not DEFAULT_PROCESS_CACHE

class Loader:
    """FFCV loader class that can be used as a drop-in replacement
    for standard (e.g. PyTorch) data loaders.

    Parameters
    ----------
    fname: str
        Full path to the location of the dataset (.beton file format).
    batch_size : int
        Batch size.
    num_workers : int
        Number of workers used for data loading. Consider using the actual number of cores instead of the number of threads if you only use JITed augmentations as they usually don't benefit from hyper-threading.
    os_cache : bool
        Leverages the operating for caching purposes. This is beneficial when there is enough memory to cache the dataset and/or when multiple processes on the same machine training using the same dataset. See https://docs.ffcv.io/performance_guide.html for more information.
    order : OrderOption
        Traversal order, one of: SEQEUNTIAL, RANDOM, QUASI_RANDOM

        QUASI_RANDOM is a random order that tries to be as uniform as possible while minimizing the amount of data read from the disk. Note that it is mostly useful when `os_cache=False`. Currently unavailable in distributed mode.
    distributed : bool
        For distributed training (multiple GPUs). Emulates the behavior of DistributedSampler from PyTorch.
    seed : int
        Random seed for batch ordering.
    indices : Sequence[int]
        Subset of dataset by filtering only some indices.
    pipelines : Mapping[str, Sequence[Union[Operation, torch.nn.Module]]
        Dictionary defining for each field the sequence of Decoders and transforms to apply.
        Fileds with missing entries will use the default pipeline, which consists of the default decoder and `ToTensor()`,
        but a field can also be disabled by explicitly by passing `None` as its pipeline.
    custom_fields : Mapping[str, Field]
        Dictonary informing the loader of the types associated to fields that are using a custom type.
    drop_last : bool
        Drop non-full batch in each iteration.
    batches_ahead : int
        Number of batches prepared in advance; balances latency and memory.
    recompile : bool
        Recompile every iteration. This is necessary if the implementation of some augmentations are expected to change during training.
    """
    def __init__(self,
                 fname: str,
                 batch_size: int,
                 num_workers: int = -1,
                 os_cache: bool = DEFAULT_OS_CACHE,
                 order: ORDER_TYPE = OrderOption.SEQUENTIAL,
                 distributed: bool = False,
                 seed: int = 0,  # For ordering of samples
                 indices: Sequence[int] = None,  # For subset selection
                 pipelines: Mapping[str,
                                    Sequence[Union[Operation, ch.nn.Module]]] = {},
                 custom_fields: Mapping[str, Type[Field]] = {},
                 drop_last: bool = True,
                 batches_ahead: int = 3,
                 recompile: bool = False,  # Recompile at every epoch
                 ):

        # We store the original user arguments to be able to pass it to the
        # filtered version of the datasets
        self._args = {
            'fname': fname,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'os_cache': os_cache,
            'order': order,
            'distributed': distributed,
            'seed': seed,
            'indices': indices,
            'pipelines': pipelines,
            'drop_last': drop_last,
            'batches_ahead': batches_ahead,
            'recompile': recompile
        }
        self.fname: str = fname
        self.batch_size: int = batch_size
        self.batches_ahead = batches_ahead
        self.seed: int = seed
        self.reader: Reader = Reader(self.fname, custom_fields)
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last
        self.distributed: bool = distributed
        self.code_per_stage = None
        self.recompile = recompile

        if self.num_workers < 1:
            self.num_workers = cpu_count()

        Compiler.set_num_threads(self.num_workers)

        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        if os_cache:
            self.memory_manager: MemoryManager = ProcessCacheManager(
                self.reader)
        else:
            self.memory_manager: MemoryManager = OSCacheManager(self.reader)

        self.traversal_order: TraversalOrder = ORDER_MAP[order](self)

        memory_read = self.memory_manager.compile_reader()
        self.next_epoch: int = 0

        self.pipelines = {}
        self.field_name_to_f_ix = {}

        for f_ix, (field_name, field) in enumerate(self.reader.handlers.items()):
            self.field_name_to_f_ix[field_name] = f_ix
            DecoderClass = field.get_decoder_class()
            try:
                operations = pipelines[field_name]
                # We check if the user disabled this field
                if operations is None:
                    continue
                if not isinstance(operations[0], DecoderClass):
                    msg = "The first operation of the pipeline for "
                    msg += f"'{field_name}' has to be a subclass of "
                    msg += f"{DecoderClass}"
                    raise ValueError(msg)
            except KeyError:
                try:
                    operations = [
                        DecoderClass(),
                        ToTensor()
                    ]
                except Exception:
                    msg = f"Impossible to create a default pipeline"
                    msg += f"{field_name}, please define one manually"
                    raise ValueError(msg)

            for i, op in enumerate(operations):
                assert isinstance(op, (ch.nn.Module, Operation)), op
                if isinstance(op, ch.nn.Module):
                    operations[i] = ModuleWrapper(op)

            for op in operations:
                op.accept_field(field)
                op.accept_globals(self.reader.metadata[f'f{f_ix}'],
                                  memory_read)

            self.pipelines[field_name] = Pipeline(operations)

    def next_traversal_order(self):
        return self.traversal_order.sample_order(self.next_epoch)

    def __iter__(self):
        Compiler.set_num_threads(self.num_workers)
        order = self.next_traversal_order()
        selected_order = order[:len(self) * self.batch_size]
        self.next_epoch += 1

        # Compile at the first epoch
        if self.code_per_stage is None or self.recompile:
            self.generate_code()

        return EpochIterator(self, selected_order)

    def filter(self, field_name:str, condition: Callable[[Any], bool]) -> 'Loader':
        new_args = {**self._args}
        pipelines = {}

        # Disabling all the other fields
        for other_field_name in self.reader.handlers.keys():
            pipelines[other_field_name] = None

        # We reuse the original pipeline for the field we care about
        try:
            pipelines[field_name] = new_args['pipelines'][field_name]
        except KeyError:
            # We keep the default one if the user didn't setup a custom one
            del pipelines[field_name]
            pass

        new_args['pipelines'] = pipelines

        # We use sequential order for speed and to know which index we are
        # filtering
        new_args['order'] = OrderOption.SEQUENTIAL
        new_args['drop_last'] = False
        sub_loader = Loader(**new_args)
        selected_indices = []

        # Iterate through the loader and test the user defined condition
        for i, (batch,) in enumerate(sub_loader):
            for j, sample in enumerate(batch):
                sample_id = i * self.batch_size + j
                if condition(sample):
                    selected_indices.append(sample_id)

        final_args = {**self._args}
        final_args['indices'] = np.array(selected_indices)
        return Loader(**final_args)


    def __len__(self):
        next_order = self.next_traversal_order()
        if self.drop_last:
            return len(next_order) // self.batch_size
        else:
            return int(np.ceil(len(next_order) / self.batch_size))

    def generate_function_call(self, pipeline_name, op_id, needs_indices):
        p_ix = self.field_name_to_f_ix[pipeline_name]
        pipeline_identifier = f'code_{pipeline_name}_{op_id}'
        memory_identifier = f'memory_{pipeline_name}_{op_id}'
        result_identifier = f'result_{pipeline_name}'

        arg_id = result_identifier
        # This is the decoder so we pass the indices instead of the previous
        # result
        if op_id == 0:
            arg_id = 'batch_indices'

        tree = ast.parse(f"""
{result_identifier} = {pipeline_identifier}({arg_id}, {memory_identifier})
        """).body[0]

        # This is the first call of the pipeline, we pass the metadata and
        # storage state
        if op_id == 0:
            tree.value.args.extend([
                ast.Subscript(value=ast.Name(id='metadata', ctx=ast.Load()),
                              slice=ast.Index(value=ast.Constant(value=f'f{p_ix}', kind=None)), ctx=ast.Load()),
                ast.Name(id='storage_state', ctx=ast.Load()),
            ])
        if needs_indices:
            tree.value.args.extend([
                ast.Name(id='batch_indices', ctx=ast.Load()),
            ])
        return tree

    def generate_stage_code(self, stage, stage_ix, functions):
        fun_name = f'stage_{stage_ix}'
        base_code = ast.parse(f"""
def {fun_name}():
    pass
        """).body[0]

        function_calls = []
        memory_banks = []
        memory_banks_id = []
        for p_ix, pipeline_name, op_id, needs_indices in stage:
            function_calls.append(self.generate_function_call(pipeline_name,
                                                              op_id, needs_indices))
            arg = ast.arg(arg=f'memory_{pipeline_name}_{op_id}')
            memory_banks.append(arg)
            memory_banks_id.append((pipeline_name, op_id))

        base_code.body.pop()
        base_code.body.extend(function_calls)

        return_tuple = ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load()))

        base_code.args.args.append(ast.arg(arg='batch_indices'))

        for p_id in self.pipelines.keys():
            r = f'result_{p_id}'
            if stage_ix != 0:
                base_code.args.args.append(ast.arg(arg=r))
            return_tuple.value.elts.append(ast.Name(id=r, ctx=ast.Load()))


        base_code.body.append(return_tuple)
        base_code.args.args.extend(memory_banks)
        base_code.args.args.append(ast.arg(arg='metadata'))
        base_code.args.args.append(ast.arg(arg='storage_state'))

        module = ast.fix_missing_locations(
            ast.Module(body=[base_code],
                       type_ignores=[])
        )
        namespace = {
            **functions,
        }

        exec(compile(module, '', 'exec'), namespace)
        final_code = namespace[fun_name]
        if stage_ix % 2 == 0:
            final_code = Compiler.compile(final_code)

        return final_code, memory_banks_id

    def generate_code(self):
        schedule = defaultdict(lambda: [])
        compiled_functions = {}
        for p_ix, (p_id, p) in enumerate(self.pipelines.items()):
            stage = 0
            for jitted_block, block_content in p.operation_blocks:
                # Even stages are jitted Odds are not
                # If this doesn't match for this pipeline we
                # shift the operations
                if 1 - jitted_block % 2 != stage % 2:
                    stage += 1
                for op in block_content:
                    ops_code = p.compiled_ops[op]

                    needs_indices = False
                    if hasattr(ops_code, 'with_indices'):
                        needs_indices = ops_code.with_indices

                    if stage % 2 == 0:
                        ops_code = Compiler.compile(ops_code)
                    compiled_functions[f'code_{p_id}_{op}'] = ops_code
                    schedule[stage].append((p_ix, p_id, op, needs_indices))
                stage += 1

        memory_bank_keys_per_stage = {}
        self.code_per_stage = {}
        for stage_ix, stage in schedule.items():
            code_for_stage, mem_banks_ids = self.generate_stage_code(stage, stage_ix,
                                                                     compiled_functions)
            self.code_per_stage[stage_ix] = code_for_stage
            memory_bank_keys_per_stage[stage_ix] = mem_banks_ids

        self.memory_bank_keys_per_stage = memory_bank_keys_per_stage
