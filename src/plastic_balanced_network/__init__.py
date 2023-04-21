from .helpers import *
from .network import *

import os

with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r") as f:
    __version__ = f.readline().strip()
