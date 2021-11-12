from glob import glob
from pathlib import Path
import os

all_api_docs = glob('api/*.rst')
for api_doc in all_api_docs:
    lines = open(api_doc).readlines()
    delete_next = 0
    new_lines = []
    for line in lines:
        if delete_next > 0:
            delete_next -= 1
        elif ('Subpackages' in line) or ('Submodules' in line):
            print('Stripping title from', api_doc)
            delete_next += 1
        else:
            new_lines.append(line)
    open(api_doc, 'w').writelines(new_lines)