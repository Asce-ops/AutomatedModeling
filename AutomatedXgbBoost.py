from typing import List, Dict
from datetime import date, timedelta
from random import seed
from pickle import dump
import warnings
warnings.filterwarnings(action="ignore") # 忽略所有警告

from toad import Combiner, WOETransformer, ScoreCard, quality
from toad.selection import select, stepwise
from toad.plot import bin_plot, badrate_plot
from toad.metrics import KS, KS_bucket, PSI, AUC
import xgboost as xgb
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from optbinning import OptimalBinning
from numpy import ndarray
from matplotlib import pyplot as plt
from seaborn import histplot 
plt.rcParams["font.family"] = "SimHei" # 替换为你选择的字体（否则绘图中可能无法正常显示中文）
# plt.rcParams["font.family"] = "QuanYi Zen Hei Mono"  # 替换为你选择的字体