from numba import njit, set_num_threads, prange
from multiprocessing import cpu_count
import torch as ch
import warnings

warnings.filterwarnings('ignore', '.*no transformation for parallel execution was possible.*',)


class Compiler:

    @classmethod
    def set_enabled(cls, b):
        cls.is_enabled = b
        
    @classmethod
    def set_num_threads(cls, n):
        if n < 1 :
            n = cpu_count()
        cls.num_threads = n
        set_num_threads(n)
        ch.set_num_threads(n)

    @classmethod
    def compile(cls, code):
        if cls.is_enabled:
            is_parallel = hasattr(code, 'parallel') and code.parallel
            return njit(fastmath=True, parallel=is_parallel)(code)
        return code

    @classmethod
    def get_iterator(cls):
        if cls.num_threads > 1:
            return prange
        else:
            return range

Compiler.set_enabled(True)
Compiler.set_num_threads(1)