from itertools import product
from time import time
from collections import defaultdict
from contextlib import redirect_stderr
import pathlib

import numpy as np
from tqdm import tqdm

from .benchmark import Benchmark

ALL_SUITES = {}

class FakeSink(object):
    def write(self, *args):
        pass
    def writelines(self, *args):
        pass
    def close(self, *args):
        pass
    def flush(self, *args):
        pass


def benchmark(arg_values={}):
    args_list = product(*arg_values.values())
    runs = [dict(zip(arg_values.keys(), x)) for x in args_list]
    def wrapper(cls):
        ALL_SUITES[cls.__name__] = (cls, runs)
    
    return wrapper

def run_all(runs=3, warm_up=1, pattern='*'):
    results = defaultdict(list)

    selected_suites = {}
    for sname in  ALL_SUITES.keys():
        if pathlib.PurePath(sname).match(pattern):
            selected_suites[sname] = ALL_SUITES[sname]

    it_suite = tqdm(selected_suites.items(), desc='Suite', leave=False)

    for suite_name, (cls, args_list) in it_suite:
        it_suite.set_postfix({'name': suite_name})
        it_args = tqdm(args_list, desc='configuration', leave=False)

        for args in it_args:
            # with redirect_stderr(FakeSink()):
            if True:
                benchmark: Benchmark = cls(**args)
                with benchmark:
                    for _ in range(warm_up):
                        benchmark.run()
                        
                    timings = []
                    for _ in range(runs):
                        start = time()
                        benchmark.run()
                        timings.append(time() - start)
                
            median_time = np.median(timings)
            
            throughput = None
            
            if 'n' in args:
                throughput = args['n'] / median_time
                
            unit = 'it/sec'
            if throughput < 1:
                unit = 'sec/it'
                throughput = 1 /throughput
                
            throughput = np.round(throughput * 10) / 10
             
            results[suite_name].append({
                **args,
                'time': median_time,
                'throughput': str(throughput) + ' ' + unit
            })
        it_args.close()
    it_suite.close()
    return results