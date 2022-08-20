from distutils.log import warn
import warnings
import ast

try:
    # Useful for debugging
    import astor
except ImportError:
    pass

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set
from abc import ABC, abstractmethod
from ffcv.pipeline.allocation_query import AllocationQuery

from ffcv.pipeline.pipeline_spec import PipelineSpec
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import allocate_query
from .operation import Operation
from ..transforms import ModuleWrapper
from .state import State

import torch as ch
import numpy as np

# This is the starting state of the pipeline
INITIAL_STATE = State(jit_mode=True,
                       device=ch.device('cpu'),
                       dtype=np.dtype('u1'),
                       shape=None)


class Node(ABC):
    last_node_id: int = 0
    def __init__(self):
        self.id = Node.last_node_id
        self._code = None
        Node.last_node_id += 1

    @property
    @abstractmethod
    def is_jitted(self):
        raise NotImplemented()

    @property
    @abstractmethod
    def parent(self):
        raise NotImplemented()
    
    @property
    @abstractmethod
    def arg_id(self):
        raise NotImplemented()
    
    @property
    @abstractmethod
    def result_id(self):
        raise NotImplemented()
    
    @property
    @abstractmethod
    def result_id(self):
        raise NotImplemented()
    
    def get_shared_code_ast(self, done_ops):
        return ast.Pass()
    
    @abstractmethod
    def generate_code(self):
        raise NotImplemented()

    def recompile(self):
        self._code = self.generate_code()

    @property
    def with_indices(self):
        try:
            return self.code.with_indices
        except:
            return False

    @property
    def code(self):
        if self._code is None:
            self.recompile()

        return self._code

    @property
    def func_call_ast(self):
        pipeline_identifier = f'code_{self.id}'
        memory_identifier = f'memory_{self.id}'

        tree = ast.parse(f"""
{self.result_id} = {pipeline_identifier}({self.arg_id}, {memory_identifier})
        """).body[0]

        if self.with_indices:
            tree.value.args.extend([
                ast.Name(id='batch_indices', ctx=ast.Load()),
            ])
        return tree


class DecoderNode(Node):
    def __init__(self, field_name:str, decoder: Operation, f_ix:int):
        super().__init__()
        self.field_name = field_name
        self.decoder = decoder
        self.f_ix = f_ix

    @property
    def is_jitted(self):
        # Decoder have to jitted
        return True

    @property
    def parent(self):
        return None

    @property
    def arg_id(self):
        return 'batch_indices'

    @property
    def result_id(self):
        return f"result_{self.id}"

    def generate_code(self):
        return self.decoder.generate_code()

    @property
    def func_call_ast(self):
        tree = super().func_call_ast
        tree.value.args.extend([
            ast.Subscript(value=ast.Name(id='metadata', ctx=ast.Load()),
                          slice=ast.Index(value=ast.Constant(value=f'f{self.f_ix}', kind=None)), ctx=ast.Load()),
                 ast.Name(id='storage_state', ctx=ast.Load()),
        ])

        return tree


class TransformNode(Node):
    def __init__(self, parent:Node, operation: Operation):
        super().__init__()
        self._parent = parent
        self.operation = operation
        self.jitted = True

    def __repr__(self):
        return f'TransformerNode({self.operation})'

    def generate_code(self):
        return self.operation.generate_code()

    @property
    def parent(self):
        return self._parent

    @property
    def is_jitted(self):
        # Decoder have to jitted
        return self.jitted

    @property
    def arg_id(self):
        return self.parent.result_id

    @property
    def result_id(self):
        return f"result_{self.id}"

    def get_shared_code_ast(self, done_ops):
        if self.operation in done_ops:
            return ast.Pass()

        done_ops[self.operation] = self.id

        pipeline_identifier = f'init_shared_state_code_{self.id}'
        memory_identifier = f'shared_memory_{self.id}'

        tree = ast.parse(f"""{pipeline_identifier}({memory_identifier})""").body[0]


        return tree


class RefNode(Node):
    def __init__(self, ref_operation: Operation):
        super().__init__()
        self.ref_operation = ref_operation
        self._parent = None

    def resolve_operation(self, operation_to_node: Dict[Operation, List[Node]]):
        entries = operation_to_node[self.ref_operation]
        if not entries:
            raise ValueError(f"{self.ref_operation} not found in other pipelines")
        if len(entries) > 1:
            raise ValueError(f"Reference to {self.ref_operation} ambiguous")

        self._parent = entries[0]

    @property
    def parent(self):
        assert self._parent is not None
        return self._parent

    @property
    def is_jitted(self):
        # RefNodes can be either jitted or not,
        # whatever produces smaller pipelines
        return None

    @property
    def arg_id(self):
        return None  # Ref's don't have arguments

    def generate_code(self):
        def nop(*args, **kwargs):
            return None

    @property
    def func_call_ast(self):
        return ast.Pass()

    @property
    def result_id(self):
        return self.parent.result_id


class Graph:

    def __init__(self, pipeline_specs: Dict[str, PipelineSpec], handlers,
                 fieldname_to_fix, metadata, memory_read):

        self.memory_read = memory_read
        self.handlers = handlers
        self.fieldname_to_fix = fieldname_to_fix
        self.metadata = metadata
        self.pipeline_specs = pipeline_specs
        self.nodes: List[Node] = []
        self.root_nodes: Dict[Node, str] = {}
        self.leaf_nodes: Dict[str, Node] = {}
        self.operation_to_node = defaultdict(list)
        self.id_to_node = {}
        self.node_to_id = {}

        # Filling the default decoders
        for output_name, spec in pipeline_specs.items():
            if spec.source in self.handlers:
                field = self.handlers[spec.source]
                Decoder = field.get_decoder_class()
                spec.accept_decoder(Decoder, output_name)

        # registering nodes
        for output_name, spec in pipeline_specs.items():
            if spec.source is None:
                raise ValueError(f"Field {output_name} has no source")

            source = spec.source
            # This pipeline starts with a decoder
            if isinstance(source, str):
                assert spec.decoder is not None
                node = DecoderNode(source, spec.decoder, fieldname_to_fix[source])
                self.operation_to_node[spec.decoder].append(node)
                self.root_nodes[node] = source
            else:
                node = RefNode(source)
                assert spec.decoder is None

            self.nodes.append(node)

            for operation in spec.transforms:
                node = TransformNode(node, operation)
                self.operation_to_node[operation].append(node)
                self.nodes.append(node)

            self.leaf_nodes[output_name] = node
            
        # resolve references
        for node in self.nodes:
            if isinstance(node, RefNode):
                node.resolve_operation(self.operation_to_node)

        # Filling the adjacency list
        self.adjacency_list = defaultdict(list)
        for node in self.nodes:
            self.id_to_node[node.id] = node
            self.node_to_id[node] = node.id
            if node.parent is not None:
                self.adjacency_list[node.parent].append(node)
                

    def collect_requirements(self, state=INITIAL_STATE,
                             current_node: Node = None,
                             allocations: Dict[int, Optional[AllocationQuery]] = None,
                             code: Dict[int, Optional[Callable]] = None,
                             source_field:str = None):

        if allocations is None:
            allocations: Dict[int, Optional[AllocationQuery]] = {
                'shared': {},
                'operation': {}
            }
        if code is None:
            code: Dict[int, Optional[Callable]] = {
                'shared': {},
                'operation': {}
            }
        next_state = state
        if current_node is None:
            next_nodes = self.root_nodes.keys()
        else:
            if not isinstance(current_node, RefNode):
                if isinstance(current_node, TransformNode):
                    operation = current_node.operation
                else:
                    operation = current_node.decoder

                if isinstance(current_node, DecoderNode):
                    source_field = current_node.field_name

                fix = self.fieldname_to_fix[source_field]
                metadata = self.metadata[f'f{fix}']

                operation.accept_field(self.handlers[source_field])
                operation.accept_globals(metadata, self.memory_read)

                next_state, allocation = operation.declare_state_and_memory(state)
                state_allocation = operation.declare_shared_memory(state)

                if next_state.device.type != 'cuda' and isinstance(operation,
                    ModuleWrapper):
                    msg = ("Using a pytorch transform on the CPU is extremely"
                        "detrimental to the performance, consider moving the augmentation"
                        "on the GPU or using an FFCV native transform")
                    warnings.warn(msg, ResourceWarning)


                if isinstance(current_node, TransformNode):
                    current_node.jitted = next_state.jit_mode

                allocations['operation'][current_node.id] = allocation
                allocations['shared'][current_node.id] = state_allocation
                code['operation'][current_node.id] = operation.generate_code()
                code['shared'][current_node.id] = operation.generate_code_for_shared_state()

            next_nodes = self.adjacency_list[current_node]

        for node in next_nodes:
            self.collect_requirements(next_state, node, allocations, code, source_field=source_field)

        return allocations, code

    def allocate_memory(self, batch_size, batches_ahead):

        memory_buffers = defaultdict(dict)
        full_memory_requirements, _ = self.collect_requirements()

        for kind, requirements in full_memory_requirements.items():
            for node_id, memory_allocation in requirements.items():
                # If the operation didn't make a query we stop here
                allocated_buffer = None
                if isinstance(memory_allocation, AllocationQuery):
                    allocated_buffer = allocate_query(memory_allocation,
                                                                batch_size,
                                                                batches_ahead)
                elif isinstance(memory_allocation, Sequence):
                    allocated_buffer = tuple(
                        allocate_query(q, batch_size, batches_ahead) for q in memory_allocation
                    )

                memory_buffers[kind][node_id] = allocated_buffer

        return memory_buffers

    def group_operations(self):
        current_front: Set[Node] = set()
        next_front: Set[Node] = set()
        stages = []

        for node in self.root_nodes.keys():
            current_front.add(node)


        while current_front:
            current_stage = list()
            jitted_stage = len(stages) % 2 == 0

            while current_front:
                node = current_front.pop()
                if node.is_jitted == jitted_stage or node.is_jitted is None:
                    current_stage.append(self.node_to_id[node])
                    current_front.update(set(self.adjacency_list[node]))

                else:
                    next_front.add(node)

            stages.append(current_stage)
            current_front = next_front

        return stages

    def codegen_stage(self, stage:List[Node], s_ix:int, op_to_node, code, already_defined):
        fun_name = f"stage_code_{s_ix}"
        base_code = ast.parse(f"""
def {fun_name}(batch_indices, metadata, storage_state):
    pass
        """).body[0]


        base_code.args.args.extend([
            ast.arg(arg=f'memory_{x}') for x in code['operation']
        ])

        base_code.args.args.extend([
            ast.arg(arg=f'shared_memory_{x}') for x in code['shared']
        ])

        base_code.args.args.extend([
            ast.arg(f'result_{x}') for x in already_defined
        ])

        return_tuple = ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load()))

        defined_here = []

        base_code.body.pop()
        compiled_functions = {}
        for node_id in stage:
            node: Node = self.id_to_node[node_id]
            has_shared_state = node_id in code['shared'] and code['shared'][node_id] is not None

            try:
                compiled_functions[f'code_{node_id}'] = code['operation'][node_id]
            except KeyError:
                pass # No code for this node

            func_call_ast = node.func_call_ast
            if has_shared_state:
                fname = f'init_shared_state_code_{node_id}'
                compiled_functions[fname] = code['shared'][node_id]
                base_code.body.append(node.get_shared_code_ast(op_to_node))
                func_call_ast.value.args.extend([
                    ast.Name(id=f'shared_memory_{op_to_node[node.operation]}', ctx=ast.Load()),
                ])

            base_code.body.append(func_call_ast)
            return_tuple.value.elts.append(ast.Name(id=node.result_id, ctx=ast.Load()))
            already_defined.append(node.id)
            defined_here.append(node.id)

        # If the stage is even we are compiling it
        if s_ix % 2 == 0:
            compiled_functions = {k: Compiler.compile(v) for (k, v) in compiled_functions.items()}

        base_code.body.append(return_tuple)

        module = ast.fix_missing_locations(
            ast.Module(body=[base_code],
                       type_ignores=[])
        )

        # print(astor.to_source(base_code))
        namespace = {
            **compiled_functions
        }

        exec(compile(module, '', 'exec'), namespace)
        final_code = namespace[fun_name]
        return final_code, defined_here


    def codegen_all(self, code):
        stages = self.group_operations()
        code_stages = []
        already_defined = []

        # Set of operations that already had their state initialized
        # (We do not want to have their random state reset)
        op_to_node = {}

        for s_ix, stage in enumerate(stages):
            code_stages.append(self.codegen_stage(stage, s_ix, op_to_node, code, already_defined))

        final_output = [x.id for x in self.leaf_nodes.values()]
        return code_stages, final_output