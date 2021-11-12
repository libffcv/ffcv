import enum
from os import environ
import ast
from multiprocessing import cpu_count
from typing import Mapping, Optional, Sequence, TYPE_CHECKING, Union, Literal
from collections import defaultdict
from enum import Enum, unique, auto

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
                 drop_last: bool = True,
                 batches_ahead: int = 3,
                 recompile: bool = False,  # Recompile at every epoch
                 ):

        self.fname: str = fname
        self.batch_size: int = batch_size
        self.batches_ahead = batches_ahead
        self.seed: int = seed
        self.reader: Reader = Reader(self.fname)
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last
        self.distributed: bool = distributed
        self.code_per_stage = None
        self.recompile = recompile
        Compiler.set_num_threads(self.num_workers)

        if self.num_workers < 1:
            self.num_workers = cpu_count()

        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        if os_cache:
            self.memory_manager: MemoryManager = OSCacheManager(self.reader)
        else:
            self.memory_manager: MemoryManager = ProcessCacheManager(
                self.reader)

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
                    raise
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

    def __iter__(self):
        cur_epoch = self.next_epoch
        self.next_epoch += 1
        Compiler.set_num_threads(self.num_workers)
        order = self.traversal_order.sample_order(cur_epoch)
        selected_order = order[:len(self) * self.batch_size]

        # Compile at the first epoch
        if self.code_per_stage is None or self.recompile:
            self.generate_code()

        return EpochIterator(self, selected_order)

    def __len__(self):
        # TODO handle drop_last
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return int(np.ceil(len(self.indices) / self.batch_size))

    def generate_function_call(self, pipeline_name, op_id):
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
        for p_ix, pipeline_name, op_id in stage:
            function_calls.append(self.generate_function_call(pipeline_name,
                                                              op_id))
            arg = ast.arg(arg=f'memory_{pipeline_name}_{op_id}')
            memory_banks.append(arg)
            memory_banks_id.append((pipeline_name, op_id))

        base_code.body.pop()
        base_code.body.extend(function_calls)

        return_tuple = ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load()))

        for p_id in self.pipelines.keys():
            r = f'result_{p_id}'
            if stage_ix != 0:
                base_code.args.args.append(ast.arg(arg=r))
            return_tuple.value.elts.append(ast.Name(id=r, ctx=ast.Load()))

        if stage_ix == 0:
            base_code.args.args.append(ast.arg(arg='batch_indices'))

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
                    if stage % 2 == 0:
                        ops_code = Compiler.compile(ops_code)
                    compiled_functions[f'code_{p_id}_{op}'] = ops_code
                    schedule[stage].append((p_ix, p_id, op))
                stage += 1

        memory_bank_keys_per_stage = {}
        self.code_per_stage = {}
        for stage_ix, stage in schedule.items():
            code_for_stage, mem_banks_ids = self.generate_stage_code(stage, stage_ix,
                                                                     compiled_functions)
            self.code_per_stage[stage_ix] = code_for_stage
            memory_bank_keys_per_stage[stage_ix] = mem_banks_ids

        self.memory_bank_keys_per_stage = memory_bank_keys_per_stage
