# -*- coding: utf-8 -*-
import numpy as np
import pdb
import os
import glob
import pandas as pd
import pickle
from datetime import datetime, timedelta
from natsort import natsorted

# 台風の画像が保存されているディレクトリ

data_subpath = '/home/kurora/Dataout/*'
# 台風の情報(年・月・時・分・緯度・経度・最大風速・中心気圧)が保存されているディレクトリ
tylist_subpath = '/home/kurora/Tylist/*'


