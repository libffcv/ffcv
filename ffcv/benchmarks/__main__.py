import argparse

import pandas as pd
from terminaltables import SingleTable

from .suites import *
from .decorator import run_all

parser = argparse.ArgumentParser(description='Run ffcv micro benchmarks')
parser.add_argument('--runs', '-n', type=int,
                    help='Use the median of --runs runs of each test',
                    default=3)
parser.add_argument('--warm-up', '-w', type=int,
                    help='Runs each test --warm-up times before measuring',
                    default=1)
parser.add_argument('--pattern', '-p', type=str,
                    help='Run only tests matching this (glob style) pattern',
                    default='*')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='If defined will write to file instead of stdout.')

args = parser.parse_args()

all_results = run_all(args.runs,
                      args.warm_up,
                      pattern=args.pattern)

result_data = []
for suite_name, results in all_results.items():
    column_names = results[0].keys()
    table_data = [list(column_names)]
    
    for result in results:
        result_data.append({
            'suite_name': suite_name,
            **result
        })
        table_data.append(result.values())
        
    table = SingleTable(table_data, title=suite_name)

    if args.output is None:
        print(table.table)

if args.output is not None:
    frame = pd.DataFrame(result_data)
    frame.to_csv(args.output)
