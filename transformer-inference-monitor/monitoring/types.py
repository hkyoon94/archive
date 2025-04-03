from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

# Feature들의 값을 저장할 수 있는 array들의 타입
FeatureValues = Union[np.ndarray, torch.Tensor]

# Tensor 및 ndarray에 인덱싱을 수행할 수 있는 모든 종류의 타입
Indices = Union[int, list[int], np.ndarray, range, slice, torch.Tensor]

Figure = plt.Figure  # matplotlib의 캔버스(figure) 객체 타입
PlotAxis = plt.Axes  # matplotlib의 도표(axis) 객체 타입
