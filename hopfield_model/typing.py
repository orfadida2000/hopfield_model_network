from typing import Callable, Optional
from numpy.typing import NDArray
from enum import Enum
import numpy as np

# Type alias for 1D integer numpy arrays, e.g., neuron states, patterns or xlsx columns.
Vector = NDArray[np.integer]

# Type alias for numpy matrices with float values, e.g., weight matrices.
Matrix = NDArray[np.float64]

# Type alias for functions that plot network states.
StatePlottingFunction = Callable[[Vector, Optional[str]], None]


class Ordering(Enum):
	"""
	Enum for specifying the order of neuron updates in Hopfield networks.
	"""
	RANDOM = "random"
	SEQUENTIAL = "sequential"
