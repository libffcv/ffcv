import pandas as pd
import matplotlib
matplotlib.use('module://imgcat')
import matplotlib.pyplot as plt
import seaborn as sns

import torch as ch
import json
from pathlib import Path

grid_dir = Path('/mnt/nfs/home/engstrom/store/ffcv_cifar_grid')
jsons = sorted([str(x) for x in grid_dir.glob('*.json')])
torches = sorted([str(x) for x in grid_dir.glob('*.ch')])

rows = []
for json_path, torch_path in zip(jsons, torches):
    metadata = pd.read_json(json_path, typ='series', orient='records')
    results = pd.Series(ch.load(torch_path))
    row = pd.concat([metadata, results])
    rows.append(row)

df = pd.DataFrame(rows)
df = df[['training.epochs', 'top_1', 'training.lr', 'training.weight_decay']]
for k in df['training.epochs'].unique():
    print(k)
    df[df['training.epochs'] == k].plot.scatter(x='training.lr', y='top_1')
    plt.show()
    df[df['training.epochs'] == k].plot.scatter(x='training.weight_decay', y='top_1')
    plt.show()

df[df['training.epochs'].isin([48, 24, 14, 96])].groupby('training.epochs').max().reset_index().plot(x='training.epochs', y='top_1')
plt.show()
import pdb; pdb.set_trace()
print(df)