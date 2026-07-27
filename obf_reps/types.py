from typing import Union

import numpy as np
from jaxtyping import Float
from matplotlib.figure import Figure
from torch import Tensor

LoggingData = Union[np.ndarray, Tensor, int, Float, float, str, Figure]
