from enum import Enum
from typing import Union, List, Tuple
from numpy.typing import NDArray
import numpy as np

from matplotlib import figure, axes, image

# Type alias for matplotlib Figure object, used for plotting.
MPLFig = figure.Figure

# Type alias for matplotlib Axes object, used for plotting.
Axes = axes.Axes

# Type alias for matplotlib AxesImage object, used for displaying images.
AxesImage = image.AxesImage

# Constants for grayscale colors.
BlackColor = 0  # Black in grayscale
WhiteColor = 1  # White in grayscale

# Type alias for a vector, which can be used as an axis values in matplotlib plots.
PlottingVector = Union[
	List[float], List[int], Tuple[float], Tuple[int], NDArray[np.floating], NDArray[np.integer]]


class NetworkPlotType(Enum):
	"""
	Enum for specifying the type of network plot.
	"""
	INITIAL = "initial"
	INTERMEDIATE = "intermediate"
	FINAL = "final"
