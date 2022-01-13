"""
Generic class for model training.
"""
from pathlib import Path
import matplotlib as mpl
import torch.optim as optim
import json
from abc import abstractmethod
from time import time
from uuid import uuid4
from pathlib import Path

import numpy as np
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from torch.cuda.amp import autocast
from tqdm import tqdm

