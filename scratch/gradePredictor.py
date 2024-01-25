import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../code/.")
import graphviz
import IPython
#import mglearn
from IPython.display import HTML, display
from lectures.code.plotting_functions import *
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from lectures.code.utils import *

plt.rcParams["font.size"] = 16
pd.set_option("display.max_colwidth", 200)